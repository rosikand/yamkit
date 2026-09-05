"""`yamkit` command line. Hardware commands talk to the arms directly; `record`/`teleoperate`/
`rollout`/`train` are thin wrappers that exec the corresponding `lerobot-*` script with the YAM
plugins pre-configured from the rig file (extra args are passed through unchanged)."""

from __future__ import annotations

import logging
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .paths import DATASETS_DIR, DEFAULT_RIG, OUTPUT_DIR, ROOT

app = typer.Typer(help="I2RT YAM arms: CAN setup, teleop, LeRobot recording, VLA rollout.", no_args_is_help=True, add_completion=False)
app.pretty_exceptions_show_locals = False
console = Console()
err = Console(stderr=True)

RigOpt = Annotated[Path, typer.Option("--rig", help="rig yaml", show_default=True)]
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


class _QuietVendor(logging.Filter):
    """Drop the vendor SDK's INFO chatter (it logs on the *root* logger: control-loop rates every
    10 s, 30 s reports, motor bring-up dumps) so prompts and yamkit's own lines stay readable."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or "/i2rt/" not in (record.pathname or "").replace("\\", "/")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not verbose:  # the vendor SDK is chatty at INFO (use -v to see it)
        for name in ("i2rt", "can", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
        for h in logging.getLogger().handlers:
            h.addFilter(_QuietVendor())


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    console.print(f"yamkit {__version__} (root: {ROOT})")


# ----------------------------------------------------------------------------- CAN / discovery --
@app.command()
def can(udev: Annotated[bool, typer.Option(help="print udev rules for persistent names")] = False) -> None:
    """List CAN adapters (state, bitrate, USB serial) and how to bring them up."""
    from .can import INSTALL_HINT, bringup_commands, list_can_interfaces, udev_rules_text

    ifaces = list_can_interfaces()
    t = Table(title="SocketCAN interfaces")
    for c in ("iface", "state", "bitrate", "serial", "product", "usb", "rx", "tx", "err"):
        t.add_column(c)
    for i in ifaces:
        t.add_row(i.name, f"[green]{i.state}[/]" if i.up else f"[red]{i.state}[/]", str(i.bitrate), i.serial or "-", i.product or "-", i.usb_path or "-", str(i.rx_packets), str(i.tx_packets), str(i.bus_errors))
    console.print(t)
    down = [i.name for i in ifaces if not i.up]
    if down:
        console.print("[yellow]Interfaces down — run (needs sudo):[/]")
        for c in bringup_commands(down):
            console.print("  " + c)
        console.print(f"or once and for all:  {INSTALL_HINT}")
    if udev:
        console.print(udev_rules_text({i.serial: i.name for i in ifaces if i.serial}))


def _camera_table(devices, cams: dict) -> Table:
    """Attached cameras (one row per colour stream) and which rig name uses each."""
    from .cameras import color_cameras

    used = {str(c.get("index_or_path")): n for n, c in cams.items()}
    t = Table(title="Cameras")
    for c in ("device", "model", "serial", "USB port", "link", "stream", "rig name"):
        t.add_column(c)
    for d in color_cameras(devices):
        speed = f"USB {d.usb_speed_mbps / 1000:g} Gb/s" if d.usb_speed_mbps and d.usb_speed_mbps >= 1000 else (f"USB {d.usb_speed_mbps} Mb/s" if d.usb_speed_mbps else "-")
        name = used.get(d.by_path or "") or used.get(d.node) or "[dim]-[/]"
        t.add_row(d.node, d.short_model, d.serial or "-", d.usb_port or "-", speed, " ".join(d.formats), name)
    return t


@app.command()
def discover(
    rig: RigOpt = DEFAULT_RIG,
    write: Annotated[bool, typer.Option(help="write/refresh the rig file")] = False,
    cameras: Annotated[bool, typer.Option(help="also (re)detect cameras")] = True,
) -> None:
    """Passively probe the CAN buses (no motor is enabled) and detect cameras; --write saves the rig file.

    Arms already in the rig keep their names, calibration and left/right (matched by adapter serial);
    so do cameras (matched by serial, device path, then model). Re-run after changing cables."""
    from .cameras import list_video_devices, suggest_cameras
    from .can import list_can_interfaces
    from .config import RigConfig
    from .discovery import absent_arms, probe_all, suggest_rig

    ifaces = list_can_interfaces()
    probes = probe_all(ifaces)
    t = Table(title="CAN probe")
    for c in ("iface", "serial", "motors (id:type)", "handle encoder", "classification"):
        t.add_column(c)
    for p, i in zip(probes, ifaces):
        motors = " ".join(f"{m.id}:{m.motor_type}" if m.motor_type else f"{m.id}:Gr{m.gear_ratio:g}" for m in p.motors)
        t.add_row(p.iface, i.serial or "-", motors or "-", ", ".join(p.encoder_versions) or "-", (f"[red]{p.error}[/]" if p.error else p.classification))
        for m in p.type_mismatches:
            console.print(f"  [yellow]{p.iface}: {m}[/]")
    console.print(t)
    existing = RigConfig.load(rig) if rig.exists() else None
    cams = None
    if cameras:
        devices = list_video_devices()
        cams, warnings = suggest_cameras(devices, existing.cameras if existing else {})
        console.print(_camera_table(devices, cams))
        for w in warnings:
            console.print(f"  [yellow]{w}[/]")
    draft = suggest_rig(probes, ifaces, existing, cameras=cams)
    if not draft.arms and not draft.cameras:
        err.print("[red]no arms and no cameras found[/]")
        raise typer.Exit(1)
    for a in absent_arms(draft, ifaces):
        console.print(f"  [yellow]{a.name}: its CAN adapter ({a.can_serial or a.can_iface}) is not plugged in — kept; delete it from the rig if the arm is gone[/]")
    new = [a for a in draft.arms.values() if "verify" in (a.notes or "")]
    console.print(f"Arms: {', '.join(f'{a.name} ({a.gripper}, serial {a.can_serial})' for a in draft.arms.values()) or 'none'}")
    console.print(f"Pairs: {', '.join(f'{p.leader}->{p.follower}' for p in draft.pairs) or 'none'}")
    console.print("Cameras: " + (", ".join(f"{n} ({c.get('model') or c.get('type')})" for n, c in draft.cameras.items()) or "none"))
    if write:
        if rig.is_file():  # keep the previous rig: `yamkit swap`/`align`/calibration data is precious
            backup = rig.with_suffix(".yaml.bak")
            backup.write_text(rig.read_text())
            console.print(f"[dim]previous rig kept at {backup}[/]")
        path = draft.save(rig)
        console.print(f"[green]wrote {path}[/]")
        if new:
            console.print(f"[yellow]left/right of {', '.join(a.name for a in new)} is a guess — check with `yamkit read <arm>`, fix with `yamkit swap`[/]")
    else:
        console.print("Re-run with --write to save this as the rig file.")


@app.command()
def cameras(rig: RigOpt = DEFAULT_RIG) -> None:
    """List attached cameras (model, serial, USB port) and which rig name uses each. Never streams."""
    from .cameras import list_video_devices, rig_camera_status
    from .config import RigConfig

    devices = list_video_devices()
    cams = RigConfig.load(rig).cameras if rig.exists() else {}
    console.print(_camera_table(devices, cams))
    status = rig_camera_status(cams, devices)
    for name, found, detail in status:
        console.print(f"  {'[green]ok[/]  ' if found else '[red]MISSING[/]'} {name}: {detail}")
    if not all(found for _, found, _ in status):
        console.print("fix with:  yamkit discover --write")


# ------------------------------------------------------------------------------------- arms --
def _active_rig(rig_path: Path):
    from .config import RigConfig

    rig = RigConfig.load(rig_path)
    if problems := rig.validate():
        raise ValueError("invalid rig: " + "; ".join(problems))
    return rig


def _duration(duration: float | None) -> None:
    from .validation import finite_scalar

    if duration is not None:
        finite_scalar(duration, "duration", minimum=0)


def _connect(rig_path: Path, name: str):
    from .arm import YamArm, resolve_channel

    rig = _active_rig(rig_path)
    spec = rig.arm(name)
    return rig, YamArm.connect(spec, resolve_channel(spec), max_joint_speed=rig.control.max_joint_speed, max_gripper_speed=rig.control.max_gripper_speed)


@app.command()
def read(
    arms: Annotated[list[str] | None, typer.Argument(help="arm names from the rig (default: all)")] = None,
    rig: RigOpt = DEFAULT_RIG,
    hz: float = 10.0,
    duration: Annotated[float | None, typer.Option(help="seconds; default runs until Ctrl-C")] = None,
) -> None:
    """Connect (gravity-compensation mode, arm stays free to move) and stream joint state."""
    from .arm import close_all
    from .validation import finite_scalar

    finite_scalar(hz, "hz", positive=True)
    _duration(duration)
    cfg = _active_rig(rig)
    names = arms or list(cfg.arms)
    for name in names:
        cfg.arm(name)  # reject a later unknown name before opening the first arm
    connected = []
    try:
        for n in names:
            connected.append(_connect(rig, n)[1])
        t_end = None if duration is None else time.monotonic() + duration
        while t_end is None or time.monotonic() < t_end:
            lines = []
            for a in connected:
                st = a.read()
                q = " ".join(f"{v:+.3f}" for v in st.q)
                g = "-" if st.gripper is None else f"{st.gripper:.2f}"
                b = "" if st.buttons is None else " btn=" + "".join("1" if x else "0" for x in st.buttons)
                lines.append(f"{a.name:>16} q=[{q}] grip={g}{b}")
            console.print("\n".join(lines))
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        close_all(connected)


@app.command()
def teleop(
    rig: RigOpt = DEFAULT_RIG,
    pairs: Annotated[list[str] | None, typer.Option("--pair", help="restrict to pairs containing this arm name")] = None,
    auto_engage: Annotated[bool, typer.Option(help="engage immediately instead of waiting for the handle button")] = False,
    bilateral_kp: Annotated[float | None, typer.Option(help="leader force-feedback gain scale (0.1–0.2); default from rig")] = None,
    hz: Annotated[float | None, typer.Option(help="loop rate; default from rig")] = None,
    duration: Annotated[float | None, typer.Option(help="seconds; default until Ctrl-C")] = None,
    print_state: Annotated[bool, typer.Option("--print-state", help="also print per-arm joint state lines (same format as `yamkit read`; used by the web UI)")] = False,
    no_home: Annotated[bool, typer.Option("--no-home", help="skip the automatic move to home at start and stop")] = False,
) -> None:
    """Leader→follower teleoperation (press the teaching-handle button to engage/disengage).

    On start every arm moves slowly to its home pose; on Ctrl-C every arm returns there before being
    released (a second Ctrl-C releases them immediately). `control.home_speed` sets the pace."""
    from .teleop import TeleopSession

    _duration(duration)
    cfg = _active_rig(rig)
    kw = {"auto_engage": auto_engage}
    if no_home:
        kw["home_speed"] = 0.0
    if bilateral_kp is not None:
        kw["bilateral_kp"] = bilateral_kp
    if hz is not None:
        kw["hz"] = hz
    last_print = [0.0]

    def status(s: TeleopSession) -> None:
        now = time.monotonic()
        if now - last_print[0] < 0.5:
            return
        last_print[0] = now
        parts = []
        for p in s.pairs:
            g = "-" if p.last_leader is None or p.last_leader.gripper is None else f"{p.last_leader.gripper:.2f}"
            parts.append(f"{p.name}: {'ENGAGED' if p.engaged else 'idle   '} err={p.tracking_error:.3f}rad grip={g}")
        console.print(f"[{s.stats.rate_hz:5.1f}Hz] " + " | ".join(parts))
        if print_state:
            for p in s.pairs:
                for arm, st in ((p.leader, p.last_leader), (p.follower, p.last_follower)):
                    if st is None:
                        continue
                    q = " ".join(f"{v:+.3f}" for v in st.q)
                    sg = "-" if st.gripper is None else f"{st.gripper:.2f}"
                    b = "" if st.buttons is None else " btn=" + "".join("1" if x else "0" for x in st.buttons)
                    console.print(f"{arm.name:>16} q=[{q}] grip={sg}{b}")

    session = TeleopSession.from_rig(cfg, pairs, on_tick=status, **kw)
    try:
        if not auto_engage:
            console.print("[cyan]Press the top button on a teaching handle to engage its follower; press again to release. Ctrl-C to quit.[/]")
        stats = session.run(duration=duration)
    finally:
        session.shutdown(home=False)  # also covers cancellation before run starts
    console.print(f"done: {stats.ticks} ticks at {stats.rate_hz:.1f} Hz ({stats.overruns} overruns)")


@app.command("calibrate-gripper")
def calibrate_gripper(arm: str, rig: RigOpt = DEFAULT_RIG) -> None:
    """Run the SDK gripper limit auto-calibration once and store the limits in the rig (skipped afterwards)."""
    cfg = _active_rig(rig)
    spec = cfg.arm(arm)
    if not spec.has_motor_gripper:
        err.print(f"[red]{arm} has no motorised gripper[/]")
        raise typer.Exit(1)
    spec.gripper_limits = None  # force calibration
    from .arm import YamArm, resolve_channel

    a = YamArm.connect(spec, resolve_channel(spec))
    try:
        limits = a.gripper_limits
    finally:
        a.close()
    if limits is None:
        err.print("[red]SDK did not report gripper limits[/]")
        raise typer.Exit(1)
    spec.gripper_limits = [float(x) for x in limits]
    cfg.save()
    console.print(f"[green]{arm}: gripper_limits = {spec.gripper_limits} saved to {cfg.path}[/]")


@app.command()
def swap(a: str, b: str, rig: RigOpt = DEFAULT_RIG) -> None:
    """Swap the physical devices behind two rig names — two arms, or two cameras.

    Arms: exchanges the CAN adapter (serial/iface) and per-arm calibration; names, sides and pairs
    stay. Cameras: exchanges the video device; names and capture settings stay. Use it after finding
    that "left_leader" is really the right one, or that the wrist cameras are crossed."""
    from .config import RigConfig

    cfg = RigConfig.load(rig)
    if a in cfg.cameras or b in cfg.cameras:
        if a not in cfg.cameras or b not in cfg.cameras:
            err.print(f"[red]{a!r} and {b!r} must both be cameras (have: {sorted(cfg.cameras)})[/]")
            raise typer.Exit(1)
        x, y = dict(cfg.cameras[a]), dict(cfg.cameras[b])
        for f in ("index_or_path", "serial_number_or_name", "serial", "model", "notes"):
            xv, yv = x.pop(f, None), y.pop(f, None)
            if yv is not None:
                x[f] = yv
            if xv is not None:
                y[f] = xv
        cfg.cameras[a], cfg.cameras[b] = x, y
        cfg.save()
        console.print(f"[green]swapped cameras: {a} is now {x.get('notes') or x.get('index_or_path')}; {b} is now {y.get('notes') or y.get('index_or_path')}[/]")
        console.print("check in the UI (yamkit ui) — wave a hand in front of each camera")
        return
    x, y = cfg.arm(a), cfg.arm(b)
    if x.role != y.role:
        err.print(f"[red]{a} is a {x.role} and {b} is a {y.role}; only arms with the same role can be swapped[/]")
        raise typer.Exit(1)
    for f in ("can_serial", "can_iface", "gripper_limits", "rest_pose", "notes"):
        xv, yv = getattr(x, f), getattr(y, f)
        setattr(x, f, yv)
        setattr(y, f, xv)
    cfg.save()
    console.print(f"[green]swapped adapters: {a} is now {x.can_serial or x.can_iface}, {b} is now {y.can_serial or y.can_iface}[/]")
    console.print(f"verify with:  yamkit read {a}")


@app.command("zero-handle")
def zero_handle(arm: str, rig: RigOpt = DEFAULT_RIG, yes: Annotated[bool, typer.Option("--yes", help="skip confirmation")] = False) -> None:
    """Re-zero a leader's teaching-handle trigger encoder at its current (released) position.

    Writes the encoder's zero offset (device EEPROM). Only needed if `yamkit read <leader>` shows the
    trigger far from 1.0 while released."""
    import can as pycan
    from i2rt.utils.encoder_manager import PassiveJointEncoder

    from .arm import resolve_channel
    from .config import RigConfig

    spec = RigConfig.load(rig).arm(arm)
    if not spec.has_handle:
        err.print(f"[red]{arm} has no teaching handle[/]")
        raise typer.Exit(1)
    ch = resolve_channel(spec)
    bus = pycan.Bus(channel=ch, interface="socketcan")
    try:
        enc = PassiveJointEncoder(bus)
        before = enc.get_encoder_report(timeout=0.5)
        console.print(f"{arm} ({ch}) encoder before: {[r.position for r in before]} counts")
        if not yes and not typer.confirm("Trigger released? Write zero position now?"):
            raise typer.Exit(0)
        enc.reset_zero_position()
        time.sleep(0.2)
        after = enc.get_encoder_report(timeout=0.5)
        console.print(f"[green]{arm} encoder after: {[r.position for r in after]} counts[/]")
    finally:
        bus.shutdown()


@app.command("set-rest")
def set_rest(arm: str, rig: RigOpt = DEFAULT_RIG) -> None:
    """Store the arm's current pose as its home pose (where it parks at Start/Stop and with `yamkit rest`; default: all joints 0)."""
    cfg, a = _connect(rig, arm)
    try:
        q = a.read().q
        a.validate_command(q, None, limit_speed=False)
    finally:
        a.close()
    # Keep the sample's precision: rounding a value at a joint bound can move it outside.
    cfg.arm(arm).rest_pose = [float(x) for x in q]
    cfg.save()
    console.print(f"[green]{arm}: rest_pose = {cfg.arm(arm).rest_pose} saved[/]")


@app.command()
def rest(
    arms: Annotated[list[str] | None, typer.Argument(help="arm names (default: every arm in the rig)")] = None,
    rig: RigOpt = DEFAULT_RIG,
    speed: Annotated[float | None, typer.Option(help="rad/s; default control.home_speed from the rig")] = None,
) -> None:
    """Park: move arm(s) slowly to their home pose, then release them there.

    Home is `rest_pose` if stored (`yamkit set-rest`), otherwise all joints at 0 — the folded pose.
    Leaders move compliantly (a hand on the handle wins). Ctrl-C releases the arms where they are."""
    from .arm import close_all, go_home_all
    from .validation import finite_scalar

    cfg = _active_rig(rig)
    if speed is not None:
        finite_scalar(speed, "speed", positive=True)
    if speed is None and cfg.control.home_speed <= 0:
        err.print("[red]home_speed is 0 in the rig — pass --speed (rad/s)[/]")
        raise typer.Exit(1)
    names = arms or list(cfg.arms)
    for name in names:
        spec = cfg.arm(name)
        if speed is None and spec.role == "leader" and cfg.control.leader_home_speed <= 0:
            raise ValueError("leader_home_speed is 0 in the rig — pass --speed (rad/s)")
    connected = []
    try:
        for n in names:
            _, a = _connect(rig, n)
            connected.append(a)
        jobs = []
        for a in connected:
            leader = a.spec.role == "leader"
            spd = speed if speed is not None else (cfg.control.leader_home_speed if leader else cfg.control.home_speed)
            console.print(f"{a.name}: moving home at {spd:g} rad/s" + (" (compliant)" if leader else ""))
            jobs.append((a, {"speed": spd, "compliant": leader, "release": True}))
        go_home_all(jobs)  # all arms at once
    except KeyboardInterrupt:
        console.print("[yellow]aborted — arms released where they are[/]")
        raise typer.Exit(130) from None
    finally:
        close_all(connected)


def _joint_stops(spec):
    """(6, 2) lower/upper joint stops (rad) of this arm type, from the vendor's robot model."""
    from i2rt.robots.get_robot import _load_joint_limits_from_xml
    from i2rt.robots.utils import ArmType, GripperType

    return _load_joint_limits_from_xml(ArmType.from_string_name(spec.arm_type).get_xml_path(), GripperType.from_string_name(spec.gripper).get_xml_path())[:6]


STOP_TOL_DEG = 15.0  # a joint counts as "against its stop" when both arms read within this of the same limit
JOINT_LABELS = ("base yaw", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist roll")


@app.command()
def align(
    arm: str,
    rig: RigOpt = DEFAULT_RIG,
    yes: Annotated[bool, typer.Option("--yes", help="skip the confirmation prompt")] = False,
    reset: Annotated[bool, typer.Option("--reset", help="forget the leader's previous offsets first")] = False,
) -> None:
    """Line up a leader with its follower (once per pair; give either arm's name).

    Both arms are connected free to move. Push EVERY joint of both arms against a stop, the same
    way on both: shoulder and elbow folded all the way in, base turned all the way to one side until
    it stops, each wrist joint turned all the way the same way until it stops.
    Hold and confirm. Only joints that are at a stop on both arms are measured — metal decides the
    pose, not the eye — and their per-joint difference is stored on the leader (`joint_offsets`),
    so from then on "same angle" means "same direction" in teleop, recording and rollout. Both arms
    are then moved back home before being released."""
    import numpy as np

    from .arm import YamArm, close_all, resolve_channel
    from .validation import finite_scalar

    cfg = _active_rig(rig)
    tol = np.radians(finite_scalar(STOP_TOL_DEG, "alignment stop tolerance", positive=True))
    pair = cfg.pair_for(arm)
    if pair is None:
        err.print(f"[red]{arm!r} is not part of a leader/follower pair in the rig[/]")
        raise typer.Exit(1)
    lspec, fspec = cfg.arm(pair.leader), cfg.arm(pair.follower)
    previous = np.zeros(6) if reset or not lspec.joint_offsets else np.asarray(lspec.joint_offsets, dtype=float)
    lspec.joint_offsets = None  # measure the raw motor frame
    stops = _joint_stops(fspec)
    lchannel, fchannel = resolve_channel(lspec), resolve_channel(fspec)
    connected = []
    try:
        leader = YamArm.connect(lspec, lchannel, max_joint_speed=cfg.control.max_joint_speed, max_gripper_speed=cfg.control.max_gripper_speed)
        connected.append(leader)
        follower = YamArm.connect(fspec, fchannel, max_joint_speed=cfg.control.max_joint_speed, max_gripper_speed=cfg.control.max_gripper_speed)
        connected.append(follower)
        console.print(f"[cyan]{pair.leader} and {pair.follower} are free to move. Push EVERY joint of BOTH arms against a stop, the same way on both:[/]")
        console.print("  shoulder + elbow folded all the way in · base turned all the way to one side · each wrist joint turned all the way the same way")
        if not yes and not typer.confirm("Every joint against its stop on both arms, and holding still?", default=True):
            raise typer.Exit(0)
        ls, fs = [], []
        for _ in range(10):
            ls.append(leader.read().q)
            fs.append(follower.read().q)
            time.sleep(0.05)
        lq, fq = np.mean(ls, axis=0), np.mean(fs, axis=0)
        console.print("measured — let go; both arms now move back home")
        try:
            follower.go_home(cfg.control.home_speed or 0.5, release=True)
            leader.go_home(cfg.control.leader_home_speed or 0.25, compliant=True, release=True)
        except KeyboardInterrupt:
            console.print("[yellow]home move aborted — turn the bases back toward the front by hand before powering up again[/]")
    finally:
        close_all(connected)

    def near(q: float, lim: float) -> bool:  # distance on the circle: a base at +183° may read -177°
        return abs((q - lim + np.pi) % (2 * np.pi) - np.pi) < tol

    offsets = previous.copy()
    t = Table(title=f"align {pair.leader} → {pair.follower}")
    for c in ("joint", "leader (deg)", "follower (deg)", "stops (deg)", "result"):
        t.add_column(c)
    aligned, skipped = [], []
    for j in range(6):
        lo, hi = stops[j]
        at = [k for k, lim in enumerate((lo, hi)) if near(lq[j], lim) and near(fq[j], lim)]
        if at:
            offsets[j] = float((fq[j] - lq[j] + np.pi) % (2 * np.pi) - np.pi)  # wrap-safe difference
            aligned.append(j)
            result = f"[green]at {'lower' if at[0] == 0 else 'upper'} stop → offset {np.degrees(offsets[j]):+.2f}°[/]"
        else:
            skipped.append(j)
            result = f"[yellow]not at a stop on both arms → unchanged ({np.degrees(previous[j]):+.2f}°)[/]"
        t.add_row(JOINT_LABELS[j], f"{np.degrees(lq[j]):+.1f}", f"{np.degrees(fq[j]):+.1f}", f"{np.degrees(lo):+.0f} / {np.degrees(hi):+.0f}", result)
    console.print(t)
    if not aligned:
        err.print("[red]no joint was against a stop on both arms — nothing saved[/]")
        raise typer.Exit(1)
    lspec.joint_offsets = [round(float(x), 4) for x in offsets]
    lspec.validate()  # the new alignment must still admit the configured home pose
    cfg.save()
    console.print(f"[green]{pair.leader}: joint_offsets saved to {cfg.path}[/] ({len(aligned)} of 6 joints measured)")
    if skipped:
        console.print(f"[yellow]{', '.join(JOINT_LABELS[j] for j in skipped)}: push these against a stop on both arms and run again to align them too[/]")
    console.print("check during teleop (not on parked arms):  yamkit teleop")


# ------------------------------------------------------------------------------- LeRobot wrappers --
def _exec_lerobot(script: str, args: list[str], dry_run: bool) -> None:
    module = "yamkit.local_rollout" if script == "lerobot_rollout" else f"lerobot.scripts.{script}"
    if script in ("lerobot_record", "lerobot_teleoperate"):
        module, args = "yamkit.lerobot_teleop", [script.removeprefix("lerobot_"), *args]
    cmd = [sys.executable, "-m", module, *args]
    console.print("[dim]$ " + " ".join(shlex.quote(c) for c in cmd) + "[/]")
    if dry_run:
        return
    os.execv(sys.executable, cmd)


def _rig_arms(rig: Path, arms: list[str] | None):
    """Resolve which pairs to use → (mode, follower names, leader names)."""
    from .config import RigConfig

    cfg = RigConfig.load(rig)
    pairs = cfg.pairs if not arms else [p for p in cfg.pairs if p.follower in arms or p.leader in arms or cfg.arm(p.follower).side in arms]
    if not pairs:
        raise typer.BadParameter(f"no matching leader/follower pairs in {rig} for {arms}")
    if len(pairs) > 2:
        raise typer.BadParameter("LeRobot plugin supports one or two pairs; use --arms to select")
    return cfg, pairs


def _robot_args(rig: Path, pairs, robot_id: str) -> list[str]:
    if len(pairs) == 1:
        return ["--robot.type=yam_follower", f"--robot.rig={rig}", f"--robot.arm={pairs[0].follower}", f"--robot.id={robot_id}"]
    return ["--robot.type=bi_yam_follower", f"--robot.rig={rig}", f"--robot.left={pairs[0].follower}", f"--robot.right={pairs[1].follower}", f"--robot.id={robot_id}"]


def _teleop_args(rig: Path, pairs, teleop_id: str) -> list[str]:
    if len(pairs) == 1:
        return ["--teleop.type=yam_leader", f"--teleop.rig={rig}", f"--teleop.arm={pairs[0].leader}", f"--teleop.id={teleop_id}"]
    return ["--teleop.type=bi_yam_leader", f"--teleop.rig={rig}", f"--teleop.left={pairs[0].leader}", f"--teleop.right={pairs[1].leader}", f"--teleop.id={teleop_id}"]


@app.command(context_settings=PASSTHROUGH)
def teleoperate(ctx: typer.Context, rig: RigOpt = DEFAULT_RIG, arms: Annotated[list[str] | None, typer.Option("--arms")] = None, fps: int = 60, display: bool = False, dry_run: bool = False) -> None:
    """Teleop through LeRobot's `lerobot-teleoperate` (same plugins used for recording)."""
    _, pairs = _rig_arms(rig, arms)
    args = [*_robot_args(rig, pairs, "yam"), *_teleop_args(rig, pairs, "yam_leader"), f"--fps={fps}", f"--display_data={str(display).lower()}", *ctx.args]
    _exec_lerobot("lerobot_teleoperate", args, dry_run)


@app.command(context_settings=PASSTHROUGH)
def record(
    ctx: typer.Context,
    name: Annotated[str, typer.Option(help="dataset name → data/datasets/<name>")],
    task: Annotated[str, typer.Option(help="natural-language task description stored with every frame")],
    rig: RigOpt = DEFAULT_RIG,
    arms: Annotated[list[str] | None, typer.Option("--arms", help="restrict to pair(s) by arm name/side")] = None,
    episodes: int = 10,
    episode_s: float = 30,
    reset_s: float = 10,
    fps: int = 30,
    to: Annotated[str | None, typer.Option(help="where the dataset goes: local | hub | both (default: hub.datasets in the rig)")] = None,
    repo_id: Annotated[str | None, typer.Option(help="Hub repo id (default <hub.username>/<name>)")] = None,
    push: Annotated[bool, typer.Option(help="same as --to both")] = False,
    resume: bool = False,
    display: bool = False,
    dry_run: bool = False,
) -> None:
    """Record teleop episodes into a LeRobot dataset (`lerobot-record`), then upload it if asked.

    The recording is always written locally first (video encoding); with `--to hub` the local copy is
    removed after a successful upload. Uploading is done by yamkit after the recorder exits, so a
    session stopped early still uploads what it recorded."""
    from . import hub

    cfg, pairs = _rig_arms(rig, arms)
    dest = to or ("both" if push else cfg.hub.datasets)
    if dest not in hub.DESTINATIONS:
        raise typer.BadParameter(f"--to must be one of {hub.DESTINATIONS}")
    root = DATASETS_DIR / name
    # The recorder is started exactly as before (no Hub lookup, no network); the Hub account is only
    # resolved after the session, at upload time.
    rid = repo_id or f"yamkit/{name}"
    args = [
        *_robot_args(rig, pairs, "yam"),
        *_teleop_args(rig, pairs, "yam_leader"),
        f"--dataset.repo_id={rid}",
        f"--dataset.root={root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.episode_time_s={episode_s}",
        f"--dataset.reset_time_s={reset_s}",
        f"--dataset.fps={fps}",
        "--dataset.push_to_hub=false",  # yamkit uploads itself (also after an early stop)
        "--dataset.no_stamp=true",
        f"--resume={str(resume).lower()}",
        f"--display_data={str(display).lower()}",
        "--play_sounds=false",
        *ctx.args,
    ]
    if dest == "local":
        _exec_lerobot("lerobot_record", args, dry_run)
        return
    if dry_run:
        _exec_lerobot("lerobot_record", args, True)
        console.print("[dim]then: upload to the Hub" + (" and remove the local copy" if dest == "hub" else "") + "[/]")
        return
    _run_lerobot("lerobot_record", args)
    if not (root / "meta" / "info.json").is_file():
        err.print(f"[red]no dataset was written at {root} — nothing to upload[/]")
        raise typer.Exit(1)
    console.print(f"[yamkit] recording finished — uploading {name} to the Hub")
    try:
        url = hub.push_dataset(name, private=cfg.hub.private, rig_username=cfg.hub.username)
    except KeyboardInterrupt:
        err.print(f"[yellow][yamkit] upload cancelled — the recording is kept at {root}[/]")
        raise typer.Exit(130) from None
    except Exception as e:  # noqa: BLE001 — offline, not signed in, Hub error: the recording is safe locally
        err.print(f"[yellow][yamkit] upload failed ({e}) — the recording is kept at {root}; retry with: yamkit push-dataset {name}[/]")
        raise typer.Exit(2) from None
    console.print(f"[yamkit] uploaded: {url}")
    if dest == "hub":
        hub.remove_local_dataset(name)
        console.print(f"[yamkit] local copy removed ({root})")


def _run_lerobot(script: str, args: list[str]) -> int:
    """Run a lerobot script as a child and wait for it. Ctrl-C reaches the child too (same process
    group), which parks the arms and finalises the dataset; we keep waiting so the caller can upload."""
    import subprocess

    module = f"lerobot.scripts.{script}"
    if script in ("lerobot_record", "lerobot_teleoperate"):
        module, args = "yamkit.lerobot_teleop", [script.removeprefix("lerobot_"), *args]
    cmd = [sys.executable, "-m", module, *args]
    console.print("[dim]$ " + " ".join(shlex.quote(c) for c in cmd) + "[/]")
    proc = subprocess.Popen(cmd)
    interrupts = 0
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts >= 3:
                proc.terminate()
            console.print("[yellow]stopping — waiting for the arms to park and the dataset to be finalised[/]")


@app.command(context_settings=PASSTHROUGH)
def rollout(
    ctx: typer.Context,
    policy: Annotated[str, typer.Option(help="checkpoint dir or HF id, e.g. lerobot/smolvla_base or outputs/train/.../pretrained_model")],
    task: Annotated[str, typer.Option(help="language instruction for the policy")],
    rig: RigOpt = DEFAULT_RIG,
    arms: Annotated[list[str] | None, typer.Option("--arms")] = None,
    duration: float = 60,
    fps: float = 30,
    rtc: Annotated[bool, typer.Option(help="real-time chunking inference (recommended for slow VLAs on CPU)")] = False,
    strategy: str = "base",
    device: str | None = None,
    display: bool = False,
    dry_run: bool = False,
    backend: str = "local",
    gpu: str = "L40S",
    modal_app: str | None = None,
    center_crop: bool = False,
    async_chunks: Annotated[bool, typer.Option("--async/--no-async", help="unguided background chunks (Modal)")] = True,
) -> None:
    """Run a policy/VLA on the follower arm(s) (`lerobot-rollout`)."""
    from .deployment import InferenceOptions

    options = InferenceOptions(policy=policy, task=task, backend=backend, device=device or "cpu", gpu=gpu,
                               modal_app=modal_app, center_crop=center_crop, rtc=rtc,
                               async_chunks=async_chunks, duration=duration, fps=fps, arms=tuple(arms or ()))
    try:
        options.validate(motion=True)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    rig_config, pairs = _rig_arms(rig, arms)
    if backend == "modal":
        if ctx.args or strategy != "base" or display:
            raise typer.BadParameter("Modal supports base strategy without display or extra LeRobot flags")
        from .modal_ops import owned_service

        receipt = owned_service()
        app_name = modal_app or (receipt or {}).get("app_name")
        if not app_name:
            raise typer.BadParameter("run yamkit modal-prepare first, or specify --modal-app")
        if dry_run:
            console.print(f"Modal unguided async via LeRobot context: {policy}, {app_name}, {fps:g} Hz")
            return
        from lerobot.rollout.configs import RolloutConfig
        from lerobot_robot_yamkit import BiYamFollowerConfig

        from .inference.profiles import get_profile
        from .remote_policy import YamkitRemoteConfig
        from .remote_rollout import run_remote_rollout

        by_side = {rig_config.arm(p.follower).side: p.follower for p in pairs}
        if set(by_side) != {"left", "right"}:
            raise typer.BadParameter("Modal YAM mapping requires distinct left and right follower arms")
        config = RolloutConfig(
            robot=BiYamFollowerConfig(rig=str(rig), left=by_side["left"], right=by_side["right"], id="yam"),
            policy=YamkitRemoteConfig(profile=get_profile(policy).id, modal_app=app_name,
                                     center_crop=center_crop),
            task=task, duration=duration, fps=fps, device="cpu", play_sounds=False,
            use_torch_compile=False, return_to_initial_position=False,
        )
        from .inference.client import RemoteFault

        try:
            result = run_remote_rollout(config)
        except RemoteFault as exc:
            if getattr(exc, "metrics", None):
                _print_inference_result(exc.metrics)
            raise
        _print_inference_result(result)
        return
    from .inference.profiles import get_profile

    try:
        local_profile = get_profile(policy)
    except ValueError:
        local_profile = None
    if local_profile:
        policy = local_profile.repo_id
    args = [
        f"--strategy.type={strategy}",
        f"--policy.path={policy}",
        *_robot_args(rig, pairs, "yam"),
        f"--task={task}",
        f"--duration={duration}",
        f"--fps={fps}",
        f"--display_data={str(display).lower()}",
        "--play_sounds=false",
    ]
    if rtc:
        args.append("--inference.type=rtc")
    if device:
        args.append(f"--device={device}")
    if local_profile:
        args.append(f"--policy.pretrained_revision={local_profile.revision}")
        if local_profile.id == "molmoact2":
            import json

            from .inference.mapping import CAMERA_RENAME_MAP

            args.append("--rename_map=" + json.dumps(CAMERA_RENAME_MAP))
    _exec_lerobot("lerobot_rollout", [*args, *ctx.args], dry_run)


@app.command()
def agent(
    model: Annotated[str, typer.Option(help="OpenAI model ID with image and function-call support")],
    task: Annotated[str, typer.Option(help="instruction for this episode")],
    arm: Annotated[str, typer.Option(help="one follower arm name from the rig")],
    rig: RigOpt = DEFAULT_RIG,
    max_steps: Annotated[int, typer.Option(help="maximum model decisions, including invalid/observe-only replies")] = 50,
    settle_s: Annotated[float, typer.Option(help="settle time after an action, in seconds")] = 0.5,
    max_joint_delta: Annotated[float, typer.Option(help="maximum joint change per action, in radians")] = 0.10,
    motion_timeout_s: Annotated[float, typer.Option(help="deadline for each fixed-target motion, in seconds")] = 5.0,
    api_timeout_s: Annotated[float, typer.Option(help="deadline for each model request, in seconds")] = 30.0,
    episode_timeout_s: Annotated[float, typer.Option(help="deadline for the whole episode, in seconds")] = 300.0,
    log_path: Annotated[Path | None, typer.Option(help="new JSONL file inside the repo (default: outputs/agent/episode-<time>.jsonl)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="use labeled fixtures; OpenAI calls are still paid unless --offline")] = False,
    execute: Annotated[bool, typer.Option("--execute", help="request live execution (currently blocked by sensor acquisition freshness gaps)")] = False,
    offline: Annotated[bool, typer.Option("--offline", help="use the deterministic mocked provider; requires --dry-run")] = False,
) -> None:
    """Run a bounded multimodal LLM episode. Live execution is currently disabled; see docs/AGENT.md."""
    from .agent import AgentConfig, run_episode
    from .agent_openai import MockProvider, OpenAIProvider, ProviderError, credential_status
    from .agent_robot import FixtureRobot, LiveIntegrationError, RobotAdapter, make_live_robot, validate_rig

    if dry_run == execute:
        raise typer.BadParameter("choose exactly one of --dry-run or --execute")
    if offline and not dry_run:
        raise typer.BadParameter("--offline requires --dry-run")

    config = AgentConfig(
        model=model,
        task=task,
        max_steps=max_steps,
        settle_s=settle_s,
        max_joint_delta=max_joint_delta,
        motion_timeout_s=motion_timeout_s,
        api_timeout_s=api_timeout_s,
        episode_timeout_s=episode_timeout_s,
    )
    try:
        config.validate()
        validate_rig(rig, arm)
        destination = log_path or OUTPUT_DIR / "agent" / f"episode-{time.time_ns()}.jsonl"
        destination = (destination if destination.is_absolute() else ROOT / destination).resolve()
        if not destination.is_relative_to(ROOT.resolve()):
            raise ValueError("--log-path must stay inside this repository")
        if destination.exists():
            raise ValueError("--log-path must name a new file")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from None

    try:
        if execute:
            adapter = make_live_robot(rig, arm)
        else:
            console.print("Hardware dry-run: labeled synthetic fixtures; no physical arms or cameras are opened.")
        if offline:
            console.print("Offline mocked provider: no API requests or charges.")
            provider = MockProvider()
        else:
            console.print(f"Paid OpenAI API mode; images are sent to OpenAI. Credential: {credential_status()}.")
            provider = OpenAIProvider(model, task)
        if dry_run:
            adapter = RobotAdapter(FixtureRobot())
    except (LiveIntegrationError, ProviderError) as exc:
        err.print(str(exc), markup=False)
        raise typer.Exit(1) from None

    try:
        result = run_episode(adapter, provider, config, destination)
    except (OSError, ValueError) as exc:
        err.print(f"Episode could not run ({type(exc).__name__}); check the log path and permissions.", markup=False)
        raise typer.Exit(1) from None
    console.print(f"Episode: {result['status']}; {result['reason']}", markup=False)
    if result["success_basis"] == "model_declared":
        console.print(f"Success: {result['success']} (model-declared; not independently verified).", markup=False)
    console.print(f"Log: {destination}", markup=False)
    if result["status"] != "finished":
        raise typer.Exit(130 if result["status"] == "cancelled" else 1)


@app.command(context_settings=PASSTHROUGH)
def train(
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="dataset name under data/datasets, or a Hub id like <user>/<name>")],
    policy_type: Annotated[str, typer.Option(help="smolvla | act | pi05 | pi0 | diffusion")] = "smolvla",
    pretrained: Annotated[str | None, typer.Option(help="init from this checkpoint (e.g. lerobot/smolvla_base)")] = "lerobot/smolvla_base",
    steps: int = 20000,
    batch_size: int = 8,
    job_name: str | None = None,
    wandb: bool = False,
    push: Annotated[bool, typer.Option(help="upload the finished checkpoint to the Hub as <hub.username>/<job>")] = False,
    rig: RigOpt = DEFAULT_RIG,
    dry_run: bool = False,
) -> None:
    """Fine-tune a policy with `lerobot-train` (a GPU box for VLAs; ACT also trains on this CPU — slowly).

    A Hub dataset id (`<user>/<name>`) is downloaded automatically. Without CUDA the policy runs on
    the CPU and the data loader stays in-process (`--num_workers=0`): forked loader workers die
    silently at the first step on this box. Pass either flag to override."""
    from .config import RigConfig

    name = dataset.split("/", 1)[1] if "/" in dataset else dataset
    root = DATASETS_DIR / name
    job = job_name or f"{policy_type}_{name}"
    args = _cpu_train_defaults(ctx.args) + [
        f"--dataset.repo_id={dataset if '/' in dataset else 'yamkit/' + dataset}",
        f"--policy.type={policy_type}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--output_dir={OUTPUT_DIR / 'train' / job}",
        f"--job_name={job}",
        f"--wandb.enable={str(wandb).lower()}",
    ]
    if "/" not in dataset or root.is_dir():
        args.insert(1, f"--dataset.root={root}")  # local copy; otherwise LeRobot pulls from the Hub
    if push:
        from . import hub

        hub_cfg = RigConfig.load(rig).hub if rig.exists() else None
        rid = hub.repo_id(job, hub_cfg.username if hub_cfg else None, kind="model")
        args += ["--policy.push_to_hub=true", f"--policy.repo_id={rid}", f"--policy.private={str(hub_cfg.private if hub_cfg else True).lower()}"]
    else:
        args.append("--policy.push_to_hub=false")
    if pretrained:
        args.append(f"--policy.pretrained_path={pretrained}")
    _exec_lerobot("lerobot_train", [*args, *ctx.args], dry_run)


def _cpu_train_defaults(extra: list[str]) -> list[str]:
    """`--policy.device=cpu --num_workers=0` on a machine without CUDA, unless given explicitly."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_cuda = False
    if has_cuda:
        return []
    given = {a.split("=", 1)[0] for a in extra}
    out = []
    if "--policy.device" not in given:
        out.append("--policy.device=cpu")
    if "--num_workers" not in given:
        out.append("--num_workers=0")
    return out


@app.command("policy-check")
def policy_check(
    policy: Annotated[str, typer.Option(help="checkpoint dir or HF id (e.g. lerobot/smolvla_base)")],
    rig: RigOpt = DEFAULT_RIG,
    arms: Annotated[list[str] | None, typer.Option("--arms")] = None,
    task: str = "pick up the object",
    device: str = "cpu",
    steps: int = 3,
    keep_policy_features: Annotated[bool, typer.Option(help="use the checkpoint's own input features instead of this rig's")] = False,
    backend: str = "local",
    gpu: str = "L40S",
    modal_app: str | None = None,
    center_crop: bool = False,
) -> None:
    """Load a policy/VLA for this rig and run it on a synthetic frame (no arm is energised)."""
    from .deployment import InferenceOptions
    from .inference.profiles import get_profile

    try:
        InferenceOptions(policy=policy, task=task, backend=backend, device=device, gpu=gpu,
                         modal_app=modal_app, center_crop=center_crop).validate()
        try:
            profile = get_profile(policy)
        except ValueError:
            profile = None
        if profile is not None:
            from .inference_check import run_check

            result = run_check(profile.id, backend=backend, device=device, task=task, steps=steps,
                               modal_app=modal_app, center_crop=center_crop)
            _print_inference_result(result)
            return
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    from .policy_check import run_policy_check

    r = run_policy_check(policy, rig_path=str(rig), arms=arms, task=task, device=device, n_steps=steps, use_robot_features=not keep_policy_features)
    t = Table(title=f"policy-check: {policy}")
    t.add_column("field")
    t.add_column("value")
    t.add_row("type / device", f"{r.policy_type} / {r.device}")
    t.add_row("state dim / action dim", f"{r.state_dim} / {r.action_dim}")
    t.add_row("image inputs", ", ".join(r.image_keys) or "none")
    t.add_row("action chunk (n_action_steps)", str(r.chunk_size))
    t.add_row("first call (new chunk)", f"{r.first_call_s * 1e3:.0f} ms")
    t.add_row("next calls", ", ".join(f"{x * 1e3:.0f} ms" for x in r.step_call_s))
    t.add_row("sample action", ", ".join(f"{k}={v:+.3f}" for k, v in list(r.action.items())[:7]) + (" …" if len(r.action) > 7 else ""))
    console.print(t)


@app.command("modal-prepare")
def modal_prepare(policy: str = "molmoact2", gpu: str = "L40S", development: bool = False,
                  cache_volume: str = "yamkit-policy-weights") -> None:
    """Explicitly deploy and warm this workspace's dedicated cloud pool; never activate hardware."""
    from .modal_ops import prepare

    _print_inference_result(prepare(policy, gpu=gpu, development=development, cache_volume_name=cache_volume))


@app.command("modal-shutdown")
def modal_shutdown() -> None:
    """Shut down only the owned cloud app. Stop local robot execution separately first."""
    from .modal_ops import shutdown

    _print_inference_result(shutdown())


@app.command("policy-probe")
def policy_probe(
    policy: str = "molmoact2", task: str = "pick up the object", backend: str = "local",
    device: str = "cpu", gpu: str = "L40S", modal_app: str | None = None,
    center_crop: bool = False, saved: Path | None = None, live: bool = False,
    approve_active_read: bool = False, rig: RigOpt = DEFAULT_RIG,
    arms: Annotated[list[str] | None, typer.Option("--arms")] = None,
) -> None:
    """Inspect fresh targets without executing them; live mode needs explicit active-read approval."""
    from .deployment import InferenceOptions
    from .probe_runner import run_profile_probe
    from .probes import format_probe_report

    try:
        InferenceOptions(policy=policy, task=task, backend=backend, device=device, gpu=gpu,
                         modal_app=modal_app, center_crop=center_crop).validate()
        result = run_profile_probe(policy, rig_path=rig, saved=saved, live=live, approved=approve_active_read,
                                   backend=backend, device=device, modal_app=modal_app, task=task,
                                   arms=arms, center_crop=center_crop)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    console.print(format_probe_report(result), markup=False)
    _print_inference_result(result)


def _print_inference_result(result: dict) -> None:
    import json

    print("[yamkit-result] " + json.dumps(result, allow_nan=False), flush=True)


# ------------------------------------------------------------------------------------ doctor --
# ------------------------------------------------------------------------------- Hugging Face Hub --
hub_app = typer.Typer(help="Hugging Face Hub sign-in (datasets and models can then be uploaded / pulled).", no_args_is_help=True)
app.add_typer(hub_app, name="hub")


@hub_app.command("login")
def hub_login(token: Annotated[str | None, typer.Option(help="access token (https://huggingface.co/settings/tokens, 'write'); prompted if omitted")] = None) -> None:
    """Sign in once. The token is stored in ./data/hf (git-ignored), never in the rig file."""
    from . import hub

    tok = token or typer.prompt("Hugging Face token", hide_input=True)
    try:
        name = hub.login(tok)
    except Exception as e:  # noqa: BLE001
        err.print(f"[red]sign-in failed: {e}[/]")
        raise typer.Exit(1) from None
    console.print(f"[green]signed in as {name}[/] (token in {hub.token_path()})")


@hub_app.command("logout")
def hub_logout() -> None:
    """Forget the stored token."""
    from . import hub

    hub.logout()
    console.print("signed out")


@hub_app.command("status")
def hub_status(rig: RigOpt = DEFAULT_RIG) -> None:
    """Who is signed in, and what the rig's hub settings are."""
    from . import hub
    from .config import RigConfig

    st = hub.status()
    if not st["logged_in"]:
        console.print("not signed in — run:  yamkit hub login")
    elif st["online"]:
        console.print(f"[green]signed in as {st['username']}[/]")
    else:
        console.print(f"[yellow]token stored, but the Hub could not be reached: {st['error']}[/]")
    if rig.exists():
        h = RigConfig.load(rig).hub
        console.print(f"rig hub settings: username={h.username or '(signed-in account)'} private={h.private} datasets={h.datasets}")


@app.command("push-dataset")
def push_dataset(name: str, rig: RigOpt = DEFAULT_RIG, remove_local: Annotated[bool, typer.Option("--remove-local", help="delete data/datasets/<name> after a successful upload")] = False) -> None:
    """Upload a local dataset to the Hub as <hub.username>/<name>."""
    from . import hub
    from .config import RigConfig

    h = RigConfig.load(rig).hub if rig.exists() else None
    url = hub.push_dataset(name, private=h.private if h else True, rig_username=h.username if h else None)
    console.print(f"[green]uploaded: {url}[/]")
    if remove_local:
        hub.remove_local_dataset(name)
        console.print("local copy removed")


@app.command("pull-dataset")
def pull_dataset(repo: Annotated[str, typer.Argument(help="<user>/<name>, or just <name> for your own account")], rig: RigOpt = DEFAULT_RIG) -> None:
    """Download a Hub dataset into data/datasets/<name>."""
    from . import hub
    from .config import RigConfig

    h = RigConfig.load(rig).hub if rig.exists() else None
    dest = hub.pull_dataset(repo, rig_username=h.username if h else None)
    console.print(f"[green]downloaded to {dest}[/]")


@app.command("push-model")
def push_model(
    path: Annotated[Path, typer.Argument(help="checkpoint dir, e.g. outputs/train/<job>/checkpoints/last/pretrained_model")],
    name: Annotated[str | None, typer.Option(help="model name on the Hub (default: the training job name)")] = None,
    rig: RigOpt = DEFAULT_RIG,
) -> None:
    """Upload a checkpoint to the Hub as <hub.username>/<name>; `yamkit rollout --policy <user>/<name>` then uses it."""
    from . import hub
    from .config import RigConfig

    h = RigConfig.load(rig).hub if rig.exists() else None
    url = hub.push_model(path, name, private=h.private if h else True, rig_username=h.username if h else None)
    console.print(f"[green]uploaded: {url}[/]")


@app.command()
def doctor(rig: RigOpt = DEFAULT_RIG) -> None:
    """Check the environment: venv, torch, CAN, plugins, cameras, rig file, data dirs."""
    ok = "[green]ok[/]"
    bad = "[red]FAIL[/]"
    warn = "[yellow]warn[/]"
    rows: list[tuple[str, str, str]] = []
    py = Path(sys.executable).resolve()
    rows.append(("python", ok if str(py).startswith(str(ROOT)) else warn, f"{sys.version.split()[0]} @ {py}"))
    try:
        import torch

        rows.append(("torch", ok, f"{torch.__version__} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}"))
    except Exception as e:  # noqa: BLE001
        rows.append(("torch", bad, str(e)))
    try:
        import lerobot

        rows.append(("lerobot", ok, lerobot.__version__))
    except Exception as e:  # noqa: BLE001
        rows.append(("lerobot", bad, str(e)))
    try:
        import i2rt  # noqa: F401

        rows.append(("i2rt", ok, (ROOT / "third_party" / "i2rt.VERSION").read_text().split()[0][:12] if (ROOT / "third_party" / "i2rt.VERSION").exists() else "?"))
    except Exception as e:  # noqa: BLE001
        rows.append(("i2rt", bad, str(e)))
    for var in ("HF_HOME", "HF_LEROBOT_HOME", "TORCH_HOME", "WANDB_DIR"):
        v = os.environ.get(var, "")
        rows.append((var, ok if v.startswith(str(ROOT)) else warn, v or "(unset)"))
    try:
        from lerobot.robots.config import RobotConfig
        from lerobot.teleoperators.config import TeleoperatorConfig
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        rc = set(RobotConfig.get_known_choices())
        tc = set(TeleoperatorConfig.get_known_choices())
        want_r, want_t = {"yam_follower", "bi_yam_follower"}, {"yam_leader", "bi_yam_leader"}
        rows.append(("lerobot plugins", ok if want_r <= rc and want_t <= tc else bad, f"robots={sorted(want_r & rc)} teleops={sorted(want_t & tc)}"))
    except Exception as e:  # noqa: BLE001
        rows.append(("lerobot plugins", bad, str(e)))
    from .cameras import color_cameras, list_video_devices, rig_camera_status
    from .can import INSTALL_HINT, boot_bringup_installed, list_can_interfaces

    ifaces = list_can_interfaces()
    can_detail = ", ".join(f"{i.name}:{i.state}@{i.bitrate}" for i in ifaces) or "none"
    if ifaces and not all(i.up for i in ifaces):
        can_detail += " — scripts/can_up.sh now, or once: " + INSTALL_HINT.split("   ")[0]
    rows.append(("CAN", ok if ifaces and all(i.up for i in ifaces) else (warn if ifaces else bad), can_detail))
    installed, detail = boot_bringup_installed()
    rows.append(("CAN at boot", ok if installed else warn, detail))
    devices = list_video_devices()
    cams = color_cameras(devices)
    rows.append(("cameras", ok if cams else warn, "; ".join(f"{d.node} {d.label}" for d in cams) or "no colour camera found (VLA inference needs cameras)"))
    if rig.exists():
        from .config import RigConfig
        from .discovery import absent_arms

        try:
            cfg = RigConfig.load(rig)
            probs = cfg.validate()
            rows.append(("rig", ok if not probs else bad, f"{rig}: {len(cfg.arms)} arms, {len(cfg.pairs)} pairs, {len(cfg.cameras)} cameras" + ("; " + "; ".join(probs) if probs else "")))
            gone = absent_arms(cfg, ifaces)
            rows.append(("rig arms", ok if not gone else warn, "all adapters plugged in" if not gone else "adapter missing for " + ", ".join(f"{a.name} ({a.can_serial or a.can_iface})" for a in gone) + " — re-plug it, or `yamkit discover --write`"))
            status = rig_camera_status(cfg.cameras, devices)
            missing = [f"{n}: {d}" for n, okc, d in status if not okc]
            unassigned = [d for d in cams if d.device_path not in {str(c.get("index_or_path")) for c in cfg.cameras.values()}]
            cam_detail = "; ".join(missing) + (" — `yamkit discover --write`" if missing else "")
            if unassigned:
                cam_detail += ("; " if cam_detail else "") + "not in rig: " + ", ".join(f"{d.node} ({d.label})" for d in unassigned)
            rows.append(("rig cameras", ok if not missing else warn, cam_detail or ("all " + str(len(status)) + " found" if status else "none configured")))
        except Exception as e:  # noqa: BLE001
            rows.append(("rig", bad, str(e)))
    else:
        rows.append(("rig", warn, f"{rig} missing — run `yamkit discover --write` (or ./setup.sh)"))
    t = Table(title="yamkit doctor")
    t.add_column("check")
    t.add_column("status")
    t.add_column("detail")
    for r in rows:
        t.add_row(*r)
    console.print(t)


@app.command()
def ui(
    rig: RigOpt = DEFAULT_RIG,
    host: Annotated[str, typer.Option(help="bind address (keep it local)")] = "127.0.0.1",
    port: int = 8400,
) -> None:
    """Serve the local web UI (Live / Record / Datasets / Deployments / Models).

    Serving pages never energises a motor; hardware runs only when a Start button spawns the
    corresponding `yamkit` command as a child process."""
    from .ui.server import run

    console.print(f"yamkit ui → http://{host}:{port}  (rig: {rig})")
    run(rig, host=host, port=port)


@app.command()
def env() -> None:
    """Print the environment variables that keep everything inside this repo (for `eval`)."""
    for var in ("YAMKIT_ROOT", "HF_HOME", "HF_LEROBOT_HOME", "TORCH_HOME", "WANDB_DIR"):
        console.print(f"export {var}={shlex.quote(os.environ.get(var, ''))}")


if __name__ == "__main__":
    app()
