"""Cameras: enumerate what is attached (sysfs + one V4L2 ioctl, no extra packages), propose rig
entries for them, and turn rig entries into LeRobot ``CameraConfig`` objects.

A RealSense exposes several video nodes (depth, infrared, colour, metadata); only the colour
node is useful as a plain OpenCV camera. Devices are identified for the rig by their
``/dev/v4l/by-path`` link (follows the USB port, so it survives reboots) plus the USB serial and
model when the device reports them (used to re-find a camera after it moved to another port).
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SYS_V4L = Path("/sys/class/video4linux")
DEV_BY_PATH = Path("/dev/v4l/by-path")
DEV_DIR = Path("/dev")

# V4L2: VIDIOC_ENUM_FMT = _IOWR('V', 2, struct v4l2_fmtdesc)  (64-byte struct)
_VIDIOC_ENUM_FMT = 0xC0405602
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_FMTDESC = struct.Struct("<III32sIIIII")

COLOR_FOURCCS = frozenset({"YUYV", "MJPG", "RGB3", "BGR3", "NV12", "NV21", "YU12", "YV12", "H264", "RGBP", "RGBR"})
NON_COLOR_FOURCCS = frozenset({"GREY", "Z16 ", "Z16", "Y8I ", "Y8I", "Y12I", "Y16 ", "Y16", "Y10 ", "Y12 "})

# Camera entries in the rig may carry these informational keys; LeRobot's config classes do not
# know them, so they are stripped before decoding.
INFO_KEYS = ("serial", "model", "notes")
DEFAULT_CAPTURE = {"width": 640, "height": 480, "fps": 30}
WRIST_MODELS = ("405",)  # RealSense models that are wrist cameras on a YAM rig


@dataclass
class VideoDevice:
    node: str  # /dev/video4
    name: str  # driver-reported name (sysfs `name`)
    index: int  # video-index<N> within its USB device
    model: str | None = None  # USB product string, e.g. "Intel(R) RealSense(TM) Depth Camera 405"
    serial: str | None = None  # USB serial (RealSense over USB 3 reports one; over USB 2 often not)
    usb_port: str | None = None  # port chain without the bus number, e.g. "1.2" (same as by-path)
    usb_device: str | None = None  # sysfs USB device name, e.g. "3-1.2" (groups nodes of one camera)
    usb_speed_mbps: int | None = None  # 480 = USB 2, 5000 = USB 3
    by_path: str | None = None  # /dev/v4l/by-path/... link, if any
    formats: list[str] = field(default_factory=list)  # FourCCs of capture formats
    error: str | None = None

    @property
    def short_model(self) -> str:
        """'RealSense D435', 'C920', ... — the product string with vendor boilerplate removed."""
        m = self.model or self.name or "camera"
        rs = re.search(r"RealSense.*?(\d{3})", m)
        if rs:
            return f"RealSense D{rs.group(1)}"
        return re.sub(r"\s+", " ", m).strip()

    @property
    def is_color(self) -> bool:
        f = set(self.formats)
        return bool(f & COLOR_FOURCCS) and not (f & NON_COLOR_FOURCCS)

    @property
    def is_wrist(self) -> bool:
        return any(w in self.short_model for w in WRIST_MODELS)

    @property
    def device_path(self) -> str:
        return self.by_path or self.node

    @property
    def label(self) -> str:
        bits = [self.short_model]
        if self.serial:
            bits.append(f"serial {self.serial}")
        if self.usb_port:
            bits.append(f"USB port {self.usb_port}")
        return ", ".join(bits)


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip() or None
    except OSError:
        return None


def capture_formats(node: str) -> list[str]:
    """FourCC codes of the capture formats of a V4L2 node (empty for metadata nodes)."""
    out: list[str] = []
    try:
        fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
    except OSError as e:
        raise OSError(e.errno, f"cannot open {node}: {e.strerror}") from None
    try:
        for i in range(64):
            req = _FMTDESC.pack(i, _V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, b"", 0, 0, 0, 0, 0)
            try:
                res = fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, req)
            except OSError as e:
                if e.errno in (errno.EINVAL, errno.ENOTTY):
                    break
                raise
            pixfmt = _FMTDESC.unpack(res)[4]
            out.append(pixfmt.to_bytes(4, "little").decode("ascii", "replace").strip())
    finally:
        os.close(fd)
    return out


def _link_map(link_dir: Path) -> dict[str, str]:
    """{'/dev/video4': '/dev/v4l/by-path/...'} for every link in the directory."""
    m: dict[str, str] = {}
    try:
        for link in sorted(link_dir.iterdir()):
            try:
                m.setdefault(str(link.resolve()), str(link))
            except OSError:
                continue
    except OSError:
        pass
    return m


def list_video_devices(sys_v4l: Path = SYS_V4L, by_path_dir: Path = DEV_BY_PATH, dev_dir: Path = DEV_DIR, formats_fn=capture_formats) -> list[VideoDevice]:
    """Every /dev/video* node with its USB identity and capture formats (no device is streamed)."""
    by_path = _link_map(by_path_dir)
    devices: list[VideoDevice] = []
    try:
        entries = sorted(sys_v4l.iterdir(), key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    except OSError:
        return devices
    for p in entries:
        if not p.name.startswith("video"):
            continue
        node = str(dev_dir / p.name)
        dev = VideoDevice(node=node, name=_read(p / "name") or p.name, index=int(_read(p / "index") or 0))
        iface = p / "device"
        if iface.exists():
            usb = iface.resolve().parent  # .../3-1.2:1.0 -> .../3-1.2
            if (usb / "idVendor").exists():
                dev.model = _read(usb / "product")
                dev.serial = _read(usb / "serial")
                dev.usb_device = usb.name
                dev.usb_port = usb.name.split("-", 1)[1] if "-" in usb.name else usb.name
                speed = _read(usb / "speed")
                dev.usb_speed_mbps = int(float(speed)) if speed and speed.replace(".", "").isdigit() else None
        dev.by_path = by_path.get(str(Path(node).resolve()))
        try:
            dev.formats = list(formats_fn(node))
        except OSError as e:
            dev.error = str(e)
        devices.append(dev)
    return devices


def color_cameras(devices: list[VideoDevice]) -> list[VideoDevice]:
    """One colour node per physical camera (the lowest-index colour node of each USB device)."""
    seen: set[str] = set()
    out: list[VideoDevice] = []
    for d in sorted(devices, key=lambda d: (_port_key(d)[0], d.usb_device or d.node, d.index)):
        key = d.usb_device or d.node
        if d.is_color and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _port_key(d: VideoDevice) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in (d.usb_port or "").split(".")), d.node


def _entry_for(d: VideoDevice, base: dict[str, Any] | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"type": "opencv", "index_or_path": d.device_path, **DEFAULT_CAPTURE}
    if base:
        for k in ("type", "width", "height", "fps"):
            if k in base:
                e[k] = base[k]
    if d.serial:
        e["serial"] = d.serial
    e["model"] = d.short_model
    e["notes"] = d.label + (f", USB {d.usb_speed_mbps} Mb/s" if d.usb_speed_mbps else "")
    return e


def suggest_cameras(devices: list[VideoDevice], existing: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Propose the rig `cameras:` section from attached devices.

    Existing entries keep their name and capture settings when their camera is still attached
    (matched by serial, then by device path, then by model). New cameras get conventional names:
    RealSense D405 → left_wrist / right_wrist (in USB-port order), any other camera → top, cam2, …
    Entries of non-opencv type are left untouched. Returns (cameras, warnings)."""
    existing = dict(existing or {})
    cams: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    free = color_cameras(devices)  # already in USB-port order

    def take(serial: str | None = None, path: str | None = None, model: str | None = None) -> VideoDevice | None:
        """Pop the first free device matching the given identity (exactly one criterion is used)."""
        for d in free:
            if (serial and d.serial == serial) or (path and path in (d.by_path, d.node)) or (model and d.short_model == model):
                free.remove(d)
                return d
        return None

    # 1) keep existing entries whose camera is still here
    unmatched: dict[str, dict[str, Any]] = {}
    for name, entry in existing.items():
        entry = dict(entry or {})
        if entry.get("type", "opencv") != "opencv":
            cams[name] = entry  # e.g. intelrealsense via pyrealsense2 — not ours to manage
            continue
        d = take(serial=entry.get("serial")) or take(path=str(entry.get("index_or_path") or ""))
        if d:
            cams[name] = _entry_for(d, entry)
        else:
            unmatched[name] = entry
    # 2) an entry whose camera moved ports: same model, in port order
    for name, entry in list(unmatched.items()):
        d = take(model=entry.get("model"))
        if d:
            cams[name] = _entry_for(d, entry)
            warnings.append(f"camera {name!r} re-found by model at {d.device_path} (was {entry.get('index_or_path')})")
            del unmatched[name]
    for name, entry in unmatched.items():
        warnings.append(f"camera {name!r} ({entry.get('notes') or entry.get('index_or_path')}) is not attached — removed from the rig")
    # 3) new cameras get conventional names
    def free_name(candidates: list[str], fallback: str) -> str:
        for c in candidates:
            if c not in cams:
                return c
        i = 2
        while f"{fallback}{i}" in cams:
            i += 1
        return f"{fallback}{i}"

    for d in [d for d in free if d.is_wrist]:
        cams[free_name(["left_wrist", "right_wrist"], "wrist")] = _entry_for(d)
    for d in [d for d in free if not d.is_wrist]:
        cams[free_name(["top"], "cam")] = _entry_for(d)
    order = {"top": 0, "left_wrist": 1, "right_wrist": 2}
    cams = dict(sorted(cams.items(), key=lambda kv: (order.get(kv[0], 3), kv[0])))
    return cams, warnings


def rig_camera_status(cameras: dict[str, dict[str, Any]], devices: list[VideoDevice] | None = None) -> list[tuple[str, bool, str]]:
    """[(name, ok, detail)] — does each rig camera's device exist right now?"""
    devices = devices if devices is not None else list_video_devices()
    by_dev = {d.device_path: d for d in devices}
    by_dev.update({d.node: d for d in devices})
    out = []
    for name, entry in cameras.items():
        if entry.get("type", "opencv") != "opencv":
            out.append((name, True, f"{entry.get('type')} camera (not checked)"))
            continue
        path = str(entry.get("index_or_path", ""))
        d = by_dev.get(path) or by_dev.get(str(Path(path).resolve())) if path else None
        if d is None and path.isdigit():
            d = by_dev.get(f"/dev/video{path}")
        if d is None:
            out.append((name, False, f"{path or '(no device)'} not found"))
        elif not d.is_color and not d.error:
            out.append((name, False, f"{path} is not a colour stream ({' '.join(d.formats) or 'no formats'})"))
        else:
            out.append((name, True, d.label))
    return out


def camera_configs_from_dicts(cams: dict[str, dict[str, Any]]) -> dict:
    """Decode `{name: {type: opencv, index_or_path: ..., width, height, fps}}` into CameraConfig objects."""
    if not cams:
        return {}
    import draccus
    from lerobot.cameras import CameraConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 — registers "opencv"

    try:
        from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401 — registers "intelrealsense"
    except Exception as e:  # noqa: BLE001 — pyrealsense2 is optional
        logging.getLogger(__name__).debug("realsense camera config unavailable: %s", e)
    out = {}
    for name, cfg in cams.items():
        cfg = {k: v for k, v in dict(cfg).items() if k not in INFO_KEYS}
        cfg.setdefault("type", "opencv")
        out[name] = draccus.decode(CameraConfig, cfg)
    return out
