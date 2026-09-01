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
console = Console()
err = Console(stderr=True)

RigOpt = Annotated[Path, typer.Option("--rig", help="rig yaml", show_default=True)]
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not verbose:  # the vendor SDK is chatty at INFO
        for name in ("i2rt", "can", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


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
    from .can import bringup_commands, list_can_interfaces, udev_rules_text

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
    if udev:
        console.print(udev_rules_text({i.serial: i.name for i in ifaces if i.serial}))


@app.command()
def discover(rig: RigOpt = DEFAULT_RIG, write: Annotated[bool, typer.Option(help="write/refresh the rig file")] = False) -> None:
    """Passively probe each CAN interface (no motor is enabled) and classify leader/follower arms."""
    from .can import list_can_interfaces
    from .config import RigConfig
    from .discovery import probe_all, suggest_rig

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
    draft = suggest_rig(probes, ifaces, existing)
    if not draft.arms:
        err.print("[red]no arms found[/]")
        raise typer.Exit(1)
    console.print(f"Proposed arms: {', '.join(f'{a.name} ({a.gripper}, serial {a.can_serial})' for a in draft.arms.values())}")
    console.print(f"Proposed pairs: {', '.join(f'{p.leader}->{p.follower}' for p in draft.pairs) or 'none'}")
    if write:
        path = draft.save(rig)
        console.print(f"[green]wrote {path}[/] — verify which physical arm is left/right and edit names/sides if needed.")
    else:
        console.print("Re-run with --write to save this as the rig file.")


# ------------------------------------------------------------------------------------- arms --
def _connect(rig_path: Path, name: str):
    from .arm import YamArm, resolve_channel
    from .config import RigConfig

    rig = RigConfig.load(rig_path)
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
    from .config import RigConfig

    cfg = RigConfig.load(rig)
    names = arms or list(cfg.arms)
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
        for a in connected:
            a.close()


@app.command()
def teleop(
    rig: RigOpt = DEFAULT_RIG,
    pairs: Annotated[list[str] | None, typer.Option("--pair", help="restrict to pairs containing this arm name")] = None,
    auto_engage: Annotated[bool, typer.Option(help="engage immediately instead of waiting for the handle button")] = False,
    bilateral_kp: Annotated[float | None, typer.Option(help="leader force-feedback gain scale (0.1–0.2); default from rig")] = None,
    hz: Annotated[float | None, typer.Option(help="loop rate; default from rig")] = None,
    duration: Annotated[float | None, typer.Option(help="seconds; default until Ctrl-C")] = None,
    print_state: Annotated[bool, typer.Option("--print-state", help="also print per-arm joint state lines (same format as `yamkit read`; used by the web UI)")] = False,
) -> None:
    """Leader→follower teleoperation (press the teaching-handle button to engage/disengage)."""
    from .config import RigConfig
    from .teleop import TeleopSession

    cfg = RigConfig.load(rig)
    kw = {"auto_engage": auto_engage}
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
    if not auto_engage:
        console.print("[cyan]Press the top button on a teaching handle to engage its follower; press again to release. Ctrl-C to quit.[/]")
    stats = session.run(duration=duration)
    console.print(f"done: {stats.ticks} ticks at {stats.rate_hz:.1f} Hz ({stats.overruns} overruns)")


@app.command("calibrate-gripper")
def calibrate_gripper(arm: str, rig: RigOpt = DEFAULT_RIG) -> None:
    """Run the SDK gripper limit auto-calibration once and store the limits in the rig (skipped afterwards)."""
    from .config import RigConfig

    cfg = RigConfig.load(rig)
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
    """Swap the physical arms behind two rig names (e.g. after finding "left_leader" is really the right one).

    Exchanges the CAN adapter (serial/iface) and per-arm calibration between the two entries; names,
    sides and pairs stay as they are."""
    from .config import RigConfig

    cfg = RigConfig.load(rig)
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
    """Store the arm's current pose as its rest pose (used by `yamkit rest`)."""
    cfg, a = _connect(rig, arm)
    try:
        q = a.read().q
    finally:
        a.close()
    cfg.arm(arm).rest_pose = [round(float(x), 4) for x in q]
    cfg.save()
    console.print(f"[green]{arm}: rest_pose = {cfg.arm(arm).rest_pose} saved[/]")


@app.command()
def rest(arms: Annotated[list[str] | None, typer.Argument(help="arm names (default: all with a rest_pose)")] = None, rig: RigOpt = DEFAULT_RIG, duration: float = 4.0) -> None:
    """Move arm(s) slowly to their stored rest pose, then release."""
    from .config import RigConfig

    cfg = RigConfig.load(rig)
    names = arms or [a.name for a in cfg.arms.values() if a.rest_pose]
    for n in names:
        spec = cfg.arm(n)
        if not spec.rest_pose:
            err.print(f"[yellow]{n}: no rest_pose stored (use `yamkit set-rest {n}`)[/]")
            continue
        _, a = _connect(rig, n)
        try:
            console.print(f"{n}: moving to rest over {duration}s")
            a.move_to(spec.rest_pose, 1.0 if spec.has_motor_gripper else None, duration=duration)
            time.sleep(0.5)
        finally:
            a.close()


# ------------------------------------------------------------------------------- LeRobot wrappers --
def _exec_lerobot(script: str, args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", f"lerobot.scripts.{script}", *args]
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
    repo_id: Annotated[str | None, typer.Option(help="HF repo id (default yamkit/<name>)")] = None,
    push: Annotated[bool, typer.Option(help="push to the HF hub when done")] = False,
    resume: bool = False,
    display: bool = False,
    dry_run: bool = False,
) -> None:
    """Record teleop episodes into a LeRobot dataset (`lerobot-record`)."""
    _, pairs = _rig_arms(rig, arms)
    root = DATASETS_DIR / name
    args = [
        *_robot_args(rig, pairs, "yam"),
        *_teleop_args(rig, pairs, "yam_leader"),
        f"--dataset.repo_id={repo_id or 'yamkit/' + name}",
        f"--dataset.root={root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.episode_time_s={episode_s}",
        f"--dataset.reset_time_s={reset_s}",
        f"--dataset.fps={fps}",
        f"--dataset.push_to_hub={str(push).lower()}",
        "--dataset.no_stamp=true",
        f"--resume={str(resume).lower()}",
        f"--display_data={str(display).lower()}",
        "--play_sounds=false",
        *ctx.args,
    ]
    _exec_lerobot("lerobot_record", args, dry_run)


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
) -> None:
    """Run a policy/VLA on the follower arm(s) (`lerobot-rollout`)."""
    _, pairs = _rig_arms(rig, arms)
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
    _exec_lerobot("lerobot_rollout", [*args, *ctx.args], dry_run)


@app.command(context_settings=PASSTHROUGH)
def train(
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="dataset name under data/datasets (or repo id with --dataset-root)")],
    policy_type: Annotated[str, typer.Option(help="smolvla | act | pi05 | pi0 | diffusion")] = "smolvla",
    pretrained: Annotated[str | None, typer.Option(help="init from this checkpoint (e.g. lerobot/smolvla_base)")] = "lerobot/smolvla_base",
    steps: int = 20000,
    batch_size: int = 8,
    job_name: str | None = None,
    wandb: bool = False,
    dry_run: bool = False,
) -> None:
    """Fine-tune a policy with `lerobot-train` (needs a GPU box; see README for the remote workflow)."""
    root = DATASETS_DIR / dataset
    job = job_name or f"{policy_type}_{dataset}"
    args = [
        f"--dataset.repo_id=yamkit/{dataset}",
        f"--dataset.root={root}",
        f"--policy.type={policy_type}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--output_dir={OUTPUT_DIR / 'train' / job}",
        f"--job_name={job}",
        f"--wandb.enable={str(wandb).lower()}",
        "--policy.push_to_hub=false",
    ]
    if pretrained:
        args.append(f"--policy.pretrained_path={pretrained}")
    _exec_lerobot("lerobot_train", [*args, *ctx.args], dry_run)


@app.command("policy-check")
def policy_check(
    policy: Annotated[str, typer.Option(help="checkpoint dir or HF id (e.g. lerobot/smolvla_base)")],
    rig: RigOpt = DEFAULT_RIG,
    arms: Annotated[list[str] | None, typer.Option("--arms")] = None,
    task: str = "pick up the object",
    device: str = "cpu",
    steps: int = 3,
    keep_policy_features: Annotated[bool, typer.Option(help="use the checkpoint's own input features instead of this rig's")] = False,
) -> None:
    """Load a policy/VLA for this rig and run it on a synthetic frame (no arm is energised)."""
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


# ------------------------------------------------------------------------------------ doctor --
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
    from .can import list_can_interfaces

    ifaces = list_can_interfaces()
    rows.append(("CAN", ok if ifaces and all(i.up for i in ifaces) else (warn if ifaces else bad), ", ".join(f"{i.name}:{i.state}@{i.bitrate}" for i in ifaces) or "none"))
    vids = sorted(Path("/dev").glob("video*"))
    rows.append(("cameras", ok if vids else warn, ", ".join(v.name for v in vids) or "no /dev/video* (VLA inference needs cameras)"))
    if rig.exists():
        from .config import RigConfig

        try:
            cfg = RigConfig.load(rig)
            probs = cfg.validate()
            rows.append(("rig", ok if not probs else bad, f"{rig}: {len(cfg.arms)} arms, {len(cfg.pairs)} pairs, {len(cfg.cameras)} cameras" + ("; " + "; ".join(probs) if probs else "")))
        except Exception as e:  # noqa: BLE001
            rows.append(("rig", bad, str(e)))
    else:
        rows.append(("rig", warn, f"{rig} missing — run `yamkit discover --write`"))
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
