"""Explicit fake lifecycle coverage; no real device constructors or model calls."""

from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit.inference.mapping import YAM_NAMES
from yamkit.openpi import artifacts, rollout


class FakeThread:
    def __init__(self, **_kwargs):
        self.ident = None
        self.joined = False

    def start(self):
        self.ident = 1

    def join(self, **_kwargs):
        if self.ident is None:
            raise RuntimeError("cannot join unstarted fake thread")
        self.joined = True


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    from yamkit import fake_inference
    from yamkit.pi05 import rollout as pi_rollout

    calls, frames = [], [{"state": np.zeros(14)}]
    sdk = SimpleNamespace(closed=False)
    options = {"engine_failure": None, "disconnect_failure": None, "context_failure": None,
               "reserve_failure": None, "connect_failure": None, "home_failure": None,
               "skip_sdk_close": False, "guard_active": False, "stop_in_run": False}
    transport = SimpleNamespace(ready=lambda: {"session_expires_at": 1e20, "instance_id": "test-instance"},
                                ensure_session_active=lambda: None, cancel=lambda: calls.append("cancel"))

    class Capture:
        def __init__(self, **kwargs):
            self.config = kwargs
            self.counts = {"observation_frames_seen": 0}

        def reserve(self):
            calls.append("reserve")
            if options["reserve_failure"]:
                raise options["reserve_failure"]

        def start(self): calls.append("capture_start")
        def end(self): calls.append("capture_end")
        def observation(self, value, **kwargs): pass
        def response(self, value, **kwargs): pass
        def event(self, kind, **kwargs): pass
        def safely(self, callback, *args, **kwargs): callback(*args, **kwargs)

        def finalize(self, destination, report, **kwargs):
            calls.append("finalize")
            assert not options["guard_active"]
            self.report, self.finalize_options = report, kwargs
            return {"status": "TRACE_SAVED" if report["released"] else "EXPORT_SKIPPED_RESOURCES_OPEN",
                    "resources_released": report["released"],
                    "exit_status": 0 if report["status"] == "completed" and report["released"] else 1}

    class Robot:
        def connect(self):
            calls.append("connect")
            if options["connect_failure"]:
                raise options["connect_failure"]

        def disconnect(self, *, home):
            assert home is False
            calls.append("disconnect_no_home")
            if options["disconnect_failure"]:
                raise options["disconnect_failure"]
            if not options["skip_sdk_close"]:
                sdk.closed = True

        def validate_action_target(self, target): pass

    robot = Robot()

    def make_robot(*_args):
        assert options["guard_active"], "Fake mode must enter the actual device guard before construction"
        calls.append("make_robot")
        return robot

    class Engine:
        def __init__(self, **kwargs):
            self.options = kwargs

        def run(self, **kwargs):
            calls.append("run")
            if options["stop_in_run"]:
                self.options["stop"].set()
            if options["engine_failure"]:
                raise options["engine_failure"]

        def metrics(self):
            return {"fake_execution": True}

    @contextmanager
    def fake_devices(observations):
        assert observations is frames
        calls.append("fake_guard_enter")
        options["guard_active"] = True
        try:
            yield [sdk]
        finally:
            options["guard_active"] = False
            calls.append("fake_guard_exit")
            if options["context_failure"]:
                raise options["context_failure"]

    def home(*_args):
        calls.append("home")
        if options["home_failure"]:
            raise options["home_failure"]

    monkeypatch.setattr(rollout, "validate_qualification", lambda *_a, **_kw: calls.append("qualification"))
    monkeypatch.setattr(rollout, "adapter_build_id", lambda: "test-build")
    monkeypatch.setattr(rollout, "rig_contract", lambda _: {"max_joint_speed": 3, "max_gripper_speed": 3})
    monkeypatch.setattr(rollout, "make_robot", make_robot)
    monkeypatch.setattr(rollout, "OpenPiYamExecutor", Engine)
    monkeypatch.setattr(rollout.threading, "Thread", FakeThread)
    monkeypatch.setattr(fake_inference, "molmo_fake_devices", fake_devices)
    monkeypatch.setattr(pi_rollout, "_home", home)
    monkeypatch.setattr(artifacts, "OpenPiCapture", Capture)
    monkeypatch.setattr(artifacts, "rollout_phase", lambda phase: calls.append("phase_" + phase))
    args = {"task": "put the red cube into the black container", "duration_s": 5,
            "rig_path": tmp_path / "rig.yaml", "statistics": object(),
            "qualification": {"service_identity": {"instance_id": "test-instance"}},
            "artifact_dir": tmp_path / "run", "fake_hardware": True, "fake_observations": frames}
    return SimpleNamespace(calls=calls, args=args, options=options, transport=transport, robot=robot,
                           sdk=sdk, frames=frames)


@pytest.mark.parametrize("confirm,mapping", [(False, False), (True, False), (False, True), (1, True), (True, 1)])
def test_physical_approval_requires_both_exact_booleans_before_any_access(lifecycle, confirm, mapping):
    args = {**lifecycle.args, "fake_hardware": False, "fake_observations": None,
            "confirm_supervised": confirm, "accept_mapping": mapping}
    with pytest.raises(ValueError, match="fresh supervised"):
        rollout.run_rollout(lifecycle.transport, **args)
    assert not lifecycle.calls


@pytest.mark.parametrize("changes", [{"confirm_supervised": True}, {"accept_mapping": True},
                                      {"fake_observations": None}, {"fake_observations": []},
                                      {"fake_hardware": "true"}])
def test_fake_flag_never_bypasses_guard_requirements(lifecycle, changes):
    with pytest.raises(ValueError, match="fake"):
        rollout.run_rollout(lifecycle.transport, **{**lifecycle.args, **changes})
    assert not lifecycle.calls


def test_healthy_fake_lifecycle_homes_then_releases_before_export(lifecycle):
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    order = lifecycle.calls
    assert order.index("qualification") < order.index("reserve") < order.index("fake_guard_enter")
    assert order.index("fake_guard_enter") < order.index("make_robot") < order.index("connect") < order.index("run")
    assert order.index("run") < order.index("home") < order.index("disconnect_no_home") < order.index("finalize")
    assert result["hardware_tested"] is False
    assert result["fake_hardware"] is True
    assert result["released"] is True
    assert result["home_completed"] is True
    assert result["exit_status"] == 0


def test_fake_policy_replays_intact_state_images_but_transition_reads_sdk(lifecycle, monkeypatch):
    actual = np.full(14, .4)
    saved = np.full(14, .8)
    lifecycle.frames[0].update(state=saved.copy(), **{
        name: np.full((2, 2, 3), 123, dtype=np.uint8) for name in ("top", "left_wrist", "right_wrist")})
    lifecycle.robot.get_observation = lambda: {**dict(zip(YAM_NAMES, actual.tolist(), strict=True)), **{
        name: np.zeros((2, 2, 3), dtype=np.uint8) for name in ("top", "left_wrist", "right_wrist")}}
    seen = []

    class Engine:
        def __init__(self, **kwargs): self.options = kwargs
        def run(self, **kwargs):
            seen.append(self.options["observe_policy_input"]())
            seen.append(self.options["observe"]())
        def metrics(self): return {}

    monkeypatch.setattr(rollout, "OpenPiYamExecutor", Engine)
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["status"] == "completed" and "Not a closed-loop" in result["fake_scope"]
    np.testing.assert_array_equal(seen[0]["state"], saved)
    np.testing.assert_array_equal(seen[1]["state"], actual)
    assert np.all(seen[0]["top"] == 123) and np.all(seen[1]["top"] == 0)


def test_stop_never_homes_and_retains_release_first(lifecycle):
    lifecycle.options["stop_in_run"] = True
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["status"] == "stopped"
    assert "home" not in lifecycle.calls
    assert result["released"] is True
    assert lifecycle.calls.index("disconnect_no_home") < lifecycle.calls.index("finalize")


def test_fault_never_homes_or_retries(lifecycle):
    lifecycle.options["engine_failure"] = RuntimeError("fake control fault")
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["status"] == "failed"
    assert result["failure_type"] == "RuntimeError"
    assert result["released"] is True
    assert "home" not in lifecycle.calls
    assert lifecycle.calls.count("run") == 1
    assert lifecycle.calls.count("disconnect_no_home") == 1


def test_failed_partial_connect_is_released_without_home(lifecycle):
    lifecycle.options["connect_failure"] = OSError("fake second arm failed")
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["released"] is True
    assert "run" not in lifecycle.calls and "home" not in lifecycle.calls
    assert result["failure_type"] == "OSError"


def test_ram_reservation_failure_precedes_all_device_construction(lifecycle):
    lifecycle.options["reserve_failure"] = MemoryError("fake RAM admission")
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["failure_type"] == "MemoryError"
    assert "make_robot" not in lifecycle.calls
    assert "fake_guard_enter" not in lifecycle.calls
    assert result["released"] is True  # no resources were acquired


def test_stop_before_start_prevents_device_construction(lifecycle):
    stop = Event()
    stop.set()
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args, shutdown_event=stop)
    assert "make_robot" not in lifecycle.calls
    assert result["status"] == "stopped"


def test_insufficient_service_lifetime_blocks_before_capture_and_hardware(lifecycle):
    lifecycle.transport.ready = lambda: {"session_expires_at": 0, "instance_id": "expired"}
    with pytest.raises(ValueError, match="time for"):
        rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert lifecycle.calls == ["qualification"]


def test_disconnect_failure_keeps_export_gate_closed_and_primary_error(lifecycle):
    lifecycle.options["engine_failure"] = ValueError("fake primary failure")
    lifecycle.options["disconnect_failure"] = OSError("fake release failure")
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["failure_type"] == "ValueError"
    assert result["release_failure_type"] == "OSError"
    assert result["released"] is False
    assert result["capture"]["status"] == "EXPORT_SKIPPED_RESOURCES_OPEN"
    assert result["exit_status"] != 0


def test_unconfirmed_fake_sdk_release_is_failure(lifecycle):
    lifecycle.options["skip_sdk_close"] = True
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["released"] is False
    assert result["release_failure_type"] == "FakeSdkNotReleased"
    assert result["exit_status"] != 0


def test_context_cleanup_failure_restores_signals_and_retains_report(lifecycle):
    before = {signum: rollout.signal.getsignal(signum) for signum in (rollout.signal.SIGINT, rollout.signal.SIGTERM)}
    lifecycle.options["context_failure"] = RuntimeError("fake guard cleanup failed")
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    after = {signum: rollout.signal.getsignal(signum) for signum in before}
    assert before == after
    assert result["released"] is False
    assert result["release_failure_type"] == "RuntimeError"
    assert result["exit_status"] != 0


def test_watcher_start_failure_does_not_mask_original_error_or_leak_handlers(lifecycle, monkeypatch):
    class FailingThread(FakeThread):
        def start(self):
            raise OSError("fake thread startup failed")
    before = rollout.signal.getsignal(rollout.signal.SIGINT)
    monkeypatch.setattr(rollout.threading, "Thread", FailingThread)
    result = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
    assert result["failure_type"] == "OSError"
    assert rollout.signal.getsignal(rollout.signal.SIGINT) == before
    assert "make_robot" not in lifecycle.calls


def test_actual_capture_never_encodes_packages_or_uploads_without_release(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("No export helper may run before confirmed release")
    monkeypatch.setattr(artifacts, "_trace_tools", forbidden)
    monkeypatch.setattr(artifacts, "package_openpi_rollout", forbidden)
    monkeypatch.setattr(artifacts, "upload_rollout", forbidden)
    capture = artifacts.OpenPiCapture(task="test", duration_s=1)
    capture.start()
    capture.end()
    summary = capture.finalize(tmp_path, {"released": False, "status": "failed", "hardware_tested": False,
                                        "started_at": 100., "completed_at": 101.},
                               upload_repo_id="test/never-upload")
    assert summary["status"] == "EXPORT_SKIPPED_RESOURCES_OPEN"
    assert summary["exit_status"] == 1
    assert (tmp_path / "report.json").is_file()


def arm_robot(*, invalid_right=False, stop_after_left=False):
    calls, stopped = [], Event()
    names = tuple([f"joint_{i}" for i in range(1, 7)] + ["gripper"])

    class Arm:
        def __init__(self, side): self.side = side

        def validate_command(self, q, gripper, *, limit_speed):
            assert limit_speed is True
            calls.append("validate_" + self.side)
            if invalid_right and self.side == "right":
                raise ValueError("fake right arm invalid")

        def command(self, q, gripper, *, limit_speed):
            assert limit_speed is True
            calls.append("command_" + self.side)
            if stop_after_left and self.side == "left":
                stopped.set()
            return np.concatenate([q, [gripper]])

    def target(value):
        return np.array([value[name + ".pos"] for name in names[:6]]), value["gripper.pos"]
    robot = SimpleNamespace(_sides={side: SimpleNamespace(arm=Arm(side), names=names,
                                                        features={name + ".pos": float for name in names},
                                                        target=target) for side in ("left", "right")})

    def check():
        if stopped.is_set():
            raise RuntimeError("fake stopped")
    return robot, calls, check


def test_ordinary_send_prevalidates_both_sides_and_retains_speed_clamps():
    robot, calls, check = arm_robot()
    target = dict(zip(YAM_NAMES, [0.] * 14, strict=True))
    assert rollout.ordinary_send(robot, target, check) == target
    assert calls == ["validate_left", "validate_right", "command_left", "command_right"]


def test_invalid_right_arm_prevents_left_send():
    robot, calls, check = arm_robot(invalid_right=True)
    with pytest.raises(ValueError, match="right arm"):
        rollout.ordinary_send(robot, dict(zip(YAM_NAMES, [0.] * 14, strict=True)), check)
    assert calls == ["validate_left", "validate_right"]


def test_stop_after_left_prevents_right_without_claiming_atomic_transaction():
    robot, calls, check = arm_robot(stop_after_left=True)
    with pytest.raises(RuntimeError, match="stopped"):
        rollout.ordinary_send(robot, dict(zip(YAM_NAMES, [0.] * 14, strict=True)), check)
    assert calls == ["validate_left", "validate_right", "command_left"]


def test_explicit_fake_context_blocks_real_camera_and_can_constructors():
    import socket

    import cv2

    from yamkit.fake_inference import molmo_fake_devices

    saved = [{"state": np.zeros(14),
              **{name: np.zeros((480, 640, 3), dtype=np.uint8)
                 for name in ("top", "left_wrist", "right_wrist")}}]
    with molmo_fake_devices(saved) as devices:
        with pytest.raises(RuntimeError, match="CAN is forbidden"):
            socket.socket(socket.AF_CAN)
        with pytest.raises(RuntimeError, match="camera construction is forbidden"):
            cv2.VideoCapture("never-open-this-device")
        assert devices == []


def test_ordinary_send_uses_actual_yamarm_clamping_with_fake_sdk(rig, monkeypatch):
    from tests.conftest import FakeRobot
    from yamkit import arm as arm_mod

    robot, _, _ = arm_robot()
    now = [100.0]
    monkeypatch.setattr(arm_mod.time, "monotonic", lambda: now[0])
    sdks = []
    for side, handle in robot._sides.items():
        sdk = FakeRobot()
        sdks.append(sdk)
        handle.arm = arm_mod.YamArm(rig.arm(side + "_follower"), "FAKE-" + side, sdk,
                                    max_joint_speed=3., max_gripper_speed=3.)
    first = dict(zip(YAM_NAMES, [0.01] * 14, strict=True))
    assert rollout.ordinary_send(robot, first, lambda: None) == first
    now[0] += 0.02
    jump = dict(zip(YAM_NAMES, [1.] * 14, strict=True))
    receipt = rollout.ordinary_send(robot, jump, lambda: None)
    # The application waited .02 seconds, but ordinary command earns at most
    # .01 seconds of movement, including grippers. No reference bypass exists.
    np.testing.assert_array_equal(list(receipt.values()), [0.04] * 14)
    for sdk in sdks:
        np.testing.assert_array_equal(sdk.commands[-1], [0.04] * 7)


def test_real_bimanual_plugin_and_executor_preserve_inserted_points_with_fake_sdk(rig, tmp_path, monkeypatch):
    from yamkit import arm as arm_mod
    from yamkit.fake_inference import molmo_fake_devices
    from yamkit.openpi.executor import OpenPiYamExecutor

    # This particular unit fixture isolates execution from startup-home timing;
    # the full CLI fake qualification separately covers the real home lifecycle.
    # All SDK/CAN/camera entrypoints are guarded before plugin construction.
    rig.control.home_speed = 0
    rig.cameras = {name: {"type": "opencv", "index_or_path": "/dev/never-open",
                          "width": 640, "height": 480, "fps": 30}
                   for name in ("top", "left_wrist", "right_wrist")}
    for side in ("left", "right"):
        rig.arm(side + "_follower").gripper_limits = [0., 6.5]
    path = tmp_path / "executor-rig.yaml"
    rig.save(path)
    frames = {name: np.zeros((480, 640, 3), dtype=np.uint8)
              for name in ("top", "left_wrist", "right_wrist")}
    saved = [{"state": np.zeros(14), **frames}]
    now, events, pending = [100.0], [], []
    first, second = np.full((50, 14), 0.367), np.full((50, 14), 0.1)
    first[:, [6, 13]] = 1.0333
    second[:, [6, 13]] = 0.25
    pending.extend([first, second])
    monkeypatch.setattr(arm_mod.time, "monotonic", lambda: now[0])

    def wait(delay):
        now[0] += max(delay, 1e-12)

    with molmo_fake_devices(saved) as devices:
        robot = rollout.make_robot(path, Event())
        robot.config._reference_startup = False
        robot.connect()
        try:
            def observe():
                value = robot.get_observation()
                return {"state": np.array([value[name] for name in YAM_NAMES]),
                        **{name: value[name] for name in frames}}

            def predict(observation, timeout):
                now[0] += 0.1
                return pending.pop(0)

            engine = OpenPiYamExecutor(
                predict=predict, observe=observe,
                send=lambda target, check: rollout.ordinary_send(robot, target, check),
                validate_target=robot.validate_action_target, stop=Event(),
                clock=lambda: now[0], wait=wait,
                event=lambda kind, **data: events.append((kind, data)))
            report = engine.run(duration_s=10, max_chunks=2)
            assert report["completed_rows"] == 50
            assert report["completed_chunks"] == 2
            assert report["inserted_transition_points"] > 50
            assert report["modified_commands"] == report["coherence_violations"] == 0
            assert report["projected_gripper_values"] == 100
            assert report["executed_projected_gripper_values"] == 50
            endpoints = [data for kind, data in events if kind == "dispatch" and data["endpoint"]]
            expected = np.concatenate([first[:25], second[:25]])
            expected[:, [6, 13]] = np.clip(expected[:, [6, 13]], 0, 1)
            np.testing.assert_array_equal([[row["requested"][name] for name in YAM_NAMES]
                                           for row in endpoints], expected)
            assert len(devices) == 2
            assert all(len(device.commands) == report["completed_points"] for device in devices)
            for device in devices:
                points = np.vstack([np.zeros(7), device.commands])
                assert np.max(np.abs(np.diff(points, axis=0))) <= 0.03 + 1e-15
        finally:
            robot.disconnect(home=False)
        assert all(device.closed for device in devices)
        assert not robot.is_connected
