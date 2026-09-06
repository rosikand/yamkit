import importlib.util
import json
import logging
import threading
from pathlib import Path

import pytest

from yamkit.teleop import TeleopSession


@pytest.fixture
def profiler():
    path = Path(__file__).resolve().parents[1] / "scripts" / "profile_teleop.py"
    spec = importlib.util.spec_from_file_location("profile_teleop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_uses_native_cli_flags_and_writes_only_after_closure(profiler, rig, fake_connect, tmp_path, monkeypatch):
    rig_path = tmp_path / "rig.yaml"
    rig.save(rig_path)
    output = tmp_path / "profile.json"
    args = ["--rig", str(rig_path), "--pair", "right_follower", "--duration", "0.03",
            "--hz", "100", "--no-home", "--bilateral-kp", "0", "--print-state"]
    original_dump = profiler.json.dump
    original_run = TeleopSession.run
    handlers = list(logging.getLogger().handlers)
    closed_at_write = []

    def dump(report, stream, **kwargs):
        closed_at_write.append(all(robot.closed for robot in fake_connect.values()))
        assert set(fake_connect) == {"right_leader", "right_follower"}
        return original_dump(report, stream, **kwargs)

    monkeypatch.setattr(profiler.json, "dump", dump)
    with pytest.raises(SystemExit) as exit_info:
        profiler.main(["--output", str(output), "--", *args])
    assert exit_info.value.code == 0
    assert closed_at_write == [True]
    assert TeleopSession.run is original_run
    assert logging.getLogger().handlers == handlers
    report = json.loads(output.read_text())
    assert report["command"] == ["yamkit", "teleop", *args]
    assert report["arms_closed"] and report["cli_error"] is None
    assert not report["timing_failed"]
    ticks = report["stats"]["ticks"]
    assert ticks >= 2
    assert report["timings"]["session.step"]["count"] == ticks
    assert report["timings"]["session.on_tick"]["count"] == ticks
    assert report["timings"]["right_leader.get_observations"]["count"] == ticks
    assert report["timings"]["right_leader.get_same_bus_device_states"]["count"] == ticks
    assert report["timings"]["right_follower.get_observations"]["count"] == ticks * 3 + 1


@pytest.mark.parametrize("args", [["--help"], ["--output", "unused", "--", "--help"]])
def test_both_help_paths_never_open_arms_or_write(profiler, monkeypatch, tmp_path, args):
    output = tmp_path / "unused.json"
    args = [str(output) if value == "unused" else value for value in args]
    monkeypatch.setattr(TeleopSession, "from_rig", lambda *a, **k: pytest.fail("help opened arms"))
    with pytest.raises(SystemExit) as exit_info:
        profiler.main(args)
    assert exit_info.value.code == 0
    assert not output.exists()


@pytest.mark.parametrize("invalid", ["outside", "existing", "unknown_flag"])
def test_invalid_output_or_cli_options_fail_before_connection(profiler, monkeypatch, tmp_path, invalid):
    output = tmp_path / "profile.json"
    if invalid == "outside":
        output = profiler.ROOT.parent / "not-written.json"
    elif invalid == "existing":
        output.write_text("retained")
    monkeypatch.setattr(TeleopSession, "from_rig", lambda *a, **k: pytest.fail("invalid command opened arms"))
    args = ["--output", str(output), "--"]
    if invalid == "unknown_flag":
        args.append("--not-a-teleop-flag")
    with pytest.raises(SystemExit) as exit_info:
        profiler.main(args)
    assert exit_info.value.code == 2
    if invalid == "existing":
        assert output.read_text() == "retained"
    else:
        assert not output.exists()


def test_sdk_background_calls_are_excluded_and_wrappers_restored(profiler, rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["left_follower"], home_speed=0)
    robot = fake_connect["left_leader"]
    original_get = robot.get_observations
    original_step = session.step
    profile = profiler.Profile([])

    def step():
        worker = threading.Thread(target=robot.get_observations)
        worker.start()
        worker.join()
        original_step()

    session.step = step
    profile.run(TeleopSession.run, session, duration=0)
    assert profile.samples["left_leader.get_observations"].count == 1
    assert robot.get_observations == original_get
    assert session.step is step
    assert all(item.closed for item in fake_connect.values())


@pytest.mark.parametrize("fault", ["step", "observation"])
def test_movement_error_preserved_when_reporting_fails(profiler, rig, fake_connect, monkeypatch, tmp_path, capsys, fault):
    rig_path = tmp_path / "rig.yaml"
    rig.save(rig_path)
    failure = RuntimeError("fixture motor failure")

    def fail_step(*args, **kwargs):
        raise failure

    def fail_report(*args, **kwargs):
        assert all(robot.closed for robot in fake_connect.values())
        raise OSError("fixture disk full")

    if fault == "step":
        monkeypatch.setattr(TeleopSession, "step", fail_step)
    else:
        original_install = profiler.Profile._install

        def install(profile, stack, session):
            # The connection and startup reads have already succeeded.
            session.pairs[0].follower.robot.get_observations = fail_step
            original_install(profile, stack, session)

        monkeypatch.setattr(profiler.Profile, "_install", install)
    monkeypatch.setattr(profiler.json, "dump", fail_report)
    handlers = list(logging.getLogger().handlers)
    with pytest.raises(RuntimeError) as error:
        profiler.main(["--output", str(tmp_path / "failure.json"), "--", "--rig", str(rig_path),
                       "--pair", "left_follower", "--duration", "0", "--no-home"])
    assert error.value is failure
    assert all(robot.closed for robot in fake_connect.values())
    assert logging.getLogger().handlers == handlers
    assert "fixture disk full" in capsys.readouterr().err


@pytest.mark.parametrize("fault", ["clock", "samples", "install"])
def test_diagnostic_failure_does_not_change_teleop_cleanup(profiler, rig, fake_connect, monkeypatch, fault):
    session = TeleopSession.from_rig(rig, ["left_follower"], home_speed=0)
    profile = profiler.Profile([])

    def fail(*args, **kwargs):
        raise RuntimeError("fixture profiling error")

    if fault == "clock":
        monkeypatch.setattr(profiler.time, "perf_counter_ns", fail)
    elif fault == "samples":
        monkeypatch.setattr(profiler.Samples, "add", fail)
    else:
        monkeypatch.setattr(profile, "_install", fail)
    stats = profile.run(TeleopSession.run, session, duration=0)
    assert stats.ticks == 1
    assert profile.timing_failed
    assert all(robot.closed for robot in fake_connect.values())


def test_incomplete_closure_withholds_report(profiler, rig, fake_connect, tmp_path, capsys):
    session = TeleopSession.from_rig(rig, ["left_follower"], home_speed=0)
    profile = profiler.Profile([])
    profile.session = session
    output = tmp_path / "not-written.json"
    try:
        profile.write_after_cleanup(output)
        assert not output.exists()
        assert "closure was not confirmed" in capsys.readouterr().err
    finally:
        session.shutdown(home=False)


def test_bounded_sample_totals_and_sdk_info_capture(profiler, monkeypatch):
    monkeypatch.setattr(profiler, "SAMPLE_LIMIT", 3)
    samples = profiler.Samples()
    for value in (100, 2, 3, 4, 5):
        samples.add(value * 1_000_000)
    summary = samples.summary()
    assert summary["count"] == 5 and summary["mean_ms"] == 22.8 and summary["max_ms"] == 100
    assert summary["retained_samples"] == 3
    assert summary["recent_percentiles_ms"] == {"50": 4, "95": 5, "99": 5}
    monkeypatch.setattr(profiler, "LOG_LIMIT", 2)
    handler = profiler.SdkTimingHandler()
    for text in ("unrelated SDK info", "Total rate: 200", "Grav Comp Control Frequency: 150", "step_time > 0.007: 2"):
        record = logging.LogRecord("root", logging.INFO, "/repo/i2rt/driver.py", 1, text, (), None)
        handler.handle(record)
    assert handler.count == 3 and len(handler.records) == 2
    assert not handler.failed
