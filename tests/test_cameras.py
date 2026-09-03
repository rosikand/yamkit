"""Camera enumeration / naming / rig-merge logic on a fake sysfs (no V4L2 device is opened)."""

from pathlib import Path

from yamkit.cameras import (
    VideoDevice,
    camera_configs_from_dicts,
    color_cameras,
    list_video_devices,
    rig_camera_status,
    suggest_cameras,
)


def _fake_sysfs(tmp_path: Path):
    """Two RealSense D405 (USB2, no serial) at ports 1.2/1.3, one D435 (USB3, serial) at 1.1 — like the
    real rig: each camera has depth (Z16), IR (GREY+UYVY), colour (YUYV) and metadata nodes."""
    sys_v4l = tmp_path / "sys"
    by_path = tmp_path / "by-path"
    dev = tmp_path / "dev"
    for d in (sys_v4l, by_path, dev):
        d.mkdir()
    formats: dict[str, list[str]] = {}
    n = 0

    def camera(bus: str, port: str, product: str, serial: str | None, speed: str, nodes: list[tuple[int, list[str]]]):
        nonlocal n
        usb = tmp_path / "usb" / f"{bus}-{port}"
        iface = usb / f"{bus}-{port}:1.0"
        iface.mkdir(parents=True)
        (usb / "idVendor").write_text("8086\n")
        (usb / "product").write_text(product + "\n")
        (usb / "speed").write_text(speed + "\n")
        if serial:
            (usb / "serial").write_text(serial + "\n")
        for index, fmts in nodes:
            node = f"video{n}"
            (sys_v4l / node).mkdir()
            (sys_v4l / node / "name").write_text("Intel(R) RealSense(TM) Depth Ca\n")
            (sys_v4l / node / "index").write_text(f"{index}\n")
            (sys_v4l / node / "device").symlink_to(iface)
            (dev / node).write_text("")  # stands in for the device node
            (by_path / f"pci-0000:05:00.4-usb-0:{port}:1.0-video-index{index}").symlink_to(dev / node)
            formats[str(dev / node)] = fmts
            n += 1

    rs = [(0, ["Z16"]), (1, []), (2, ["GREY", "UYVY", "GREY"]), (3, []), (4, ["YUYV"]), (5, [])]
    camera("3", "1.2", "Intel(R) RealSense(TM) Depth Camera 405", None, "480", rs)
    camera("4", "1.1", "Intel(R) RealSense(TM) Depth Camera 435", "939323022361", "5000", rs)
    camera("3", "1.3", "Intel(R) RealSense(TM) Depth Camera 405", None, "480", rs)
    return sys_v4l, by_path, dev, formats


def _devices(tmp_path):
    sys_v4l, by_path, dev, formats = _fake_sysfs(tmp_path)
    return list_video_devices(sys_v4l, by_path, dev, formats_fn=lambda node: formats[node])


def test_enumeration_and_color_nodes(tmp_path):
    devs = _devices(tmp_path)
    assert len(devs) == 18
    cams = color_cameras(devs)
    assert [(c.short_model, c.usb_port, c.serial, c.usb_speed_mbps) for c in cams] == [  # USB-port order
        ("RealSense D435", "1.1", "939323022361", 5000),
        ("RealSense D405", "1.2", None, 480),
        ("RealSense D405", "1.3", None, 480),
    ]
    assert all(c.formats == ["YUYV"] and c.index == 4 for c in cams)
    assert cams[1].by_path.endswith("usb-0:1.2:1.0-video-index4")
    assert cams[0].is_wrist is False and cams[1].is_wrist is True
    assert cams[0].label == "RealSense D435, serial 939323022361, USB port 1.1"
    ir = next(d for d in devs if "UYVY" in d.formats)
    assert ir.is_color is False  # infrared node lists a colour FourCC but is not a colour camera


def test_suggest_names_from_scratch(tmp_path):
    cams, warnings = suggest_cameras(_devices(tmp_path), {})
    assert list(cams) == ["top", "left_wrist", "right_wrist"]
    assert warnings == []
    assert cams["top"]["serial"] == "939323022361" and cams["top"]["model"] == "RealSense D435"
    assert cams["left_wrist"]["index_or_path"].endswith("usb-0:1.2:1.0-video-index4")
    assert cams["right_wrist"]["index_or_path"].endswith("usb-0:1.3:1.0-video-index4")
    assert all(c["width"] == 640 and c["height"] == 480 and c["fps"] == 30 and c["type"] == "opencv" for c in cams.values())
    assert "USB port 1.2" in cams["left_wrist"]["notes"]


def test_suggest_keeps_existing_names_and_settings(tmp_path):
    devs = _devices(tmp_path)
    first, _ = suggest_cameras(devs, {})
    # user swapped the wrists, renamed top → overhead, lowered fps, and added a non-opencv camera
    edited = {
        "overhead": {**first["top"], "fps": 15},
        "left_wrist": first["right_wrist"],
        "right_wrist": first["left_wrist"],
        "depth": {"type": "intelrealsense", "serial_number_or_name": "939323022361", "width": 640, "height": 480, "fps": 30},
    }
    again, warnings = suggest_cameras(devs, edited)
    assert warnings == []
    assert set(again) == {"overhead", "left_wrist", "right_wrist", "depth"}
    assert again["overhead"]["fps"] == 15 and again["overhead"]["serial"] == "939323022361"
    assert again["left_wrist"]["index_or_path"] == first["right_wrist"]["index_or_path"]  # swap survives
    assert again["depth"] == edited["depth"]  # not ours to manage


def test_suggest_refinds_moved_camera_and_drops_missing(tmp_path):
    devs = _devices(tmp_path)
    first, _ = suggest_cameras(devs, {})
    # the D435 moved to another port (serial match) and one D405 was moved (model match);
    # a fourth camera from the old rig is gone
    old = {
        "top": {**first["top"], "index_or_path": "/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.4:1.3-video-index0"},
        "left_wrist": {**first["left_wrist"], "index_or_path": "/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.9:1.0-video-index4"},
        "right_wrist": first["right_wrist"],
        "side": {"type": "opencv", "index_or_path": "/dev/v4l/by-id/usb-Logitech_C920-video-index0", "width": 640, "height": 480, "fps": 30},
    }
    cams, warnings = suggest_cameras(devs, old)
    assert set(cams) == {"top", "left_wrist", "right_wrist"}
    assert cams["top"]["index_or_path"] == first["top"]["index_or_path"]  # by serial, silently
    assert cams["left_wrist"]["index_or_path"] == first["left_wrist"]["index_or_path"]  # by model, with a warning
    assert any("left_wrist" in w and "re-found" in w for w in warnings)
    assert any("side" in w and "removed" in w for w in warnings)


def test_rig_camera_status(tmp_path):
    devs = _devices(tmp_path)
    cams, _ = suggest_cameras(devs, {})
    cams["gone"] = {"type": "opencv", "index_or_path": "/dev/video99", "width": 640, "height": 480, "fps": 30}
    depth_node = next(d for d in devs if "Z16" in d.formats)
    cams["depth_by_mistake"] = {"type": "opencv", "index_or_path": depth_node.by_path, "width": 640, "height": 480, "fps": 30}
    status = {n: (ok, detail) for n, ok, detail in rig_camera_status(cams, devs)}
    assert status["top"][0] and status["left_wrist"][0] and status["right_wrist"][0]
    assert status["gone"] == (False, "/dev/video99 not found")
    assert status["depth_by_mistake"][0] is False and "not a colour stream" in status["depth_by_mistake"][1]


def test_info_keys_are_stripped_before_lerobot_decode():
    cfgs = camera_configs_from_dicts({"top": {"type": "opencv", "index_or_path": "/dev/video10", "width": 640, "height": 480, "fps": 30,
                                              "serial": "939323022361", "model": "RealSense D435", "notes": "overhead"}})
    assert type(cfgs["top"]).__name__ == "OpenCVCameraConfig"
    assert cfgs["top"].width == 640 and str(cfgs["top"].index_or_path) == "/dev/video10"


def test_video_device_labels():
    d = VideoDevice(node="/dev/video0", name="HD Pro Webcam C920", index=0, model="HD Pro Webcam C920", serial="ABC", usb_port="2")
    assert d.short_model == "HD Pro Webcam C920" and d.is_wrist is False
    assert d.label == "HD Pro Webcam C920, serial ABC, USB port 2"
    assert VideoDevice(node="/dev/video0", name="x", index=0, formats=["MJPG", "YUYV"]).is_color is True
    assert VideoDevice(node="/dev/video0", name="x", index=0, formats=["GREY", "YUYV"]).is_color is False


def test_cameras_cli_and_swap(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from yamkit import cli
    from yamkit.config import RigConfig

    sys_v4l, by_path, dev, formats = _fake_sysfs(tmp_path)
    devs = list_video_devices(sys_v4l, by_path, dev, formats_fn=lambda node: formats[node])
    monkeypatch.setattr("yamkit.cameras.list_video_devices", lambda *a, **k: devs)
    rig_path = tmp_path / "rig.yaml"
    cams, _ = suggest_cameras(devs, {})
    RigConfig(cameras=cams).save(rig_path)
    res = CliRunner().invoke(cli.app, ["cameras", "--rig", str(rig_path)])
    assert res.exit_code == 0, res.output
    assert "left_wrist" in res.output and "MISSING" not in res.output

    res = CliRunner().invoke(cli.app, ["swap", "left_wrist", "right_wrist", "--rig", str(rig_path)])
    assert res.exit_code == 0, res.output
    loaded = RigConfig.load(rig_path).cameras
    assert loaded["left_wrist"]["index_or_path"] == cams["right_wrist"]["index_or_path"]
    assert loaded["right_wrist"]["index_or_path"] == cams["left_wrist"]["index_or_path"]
    assert loaded["left_wrist"]["notes"] == cams["right_wrist"]["notes"]
    assert list(loaded) == ["top", "left_wrist", "right_wrist"]  # names and order untouched
    assert CliRunner().invoke(cli.app, ["swap", "top", "nope", "--rig", str(rig_path)]).exit_code == 1
