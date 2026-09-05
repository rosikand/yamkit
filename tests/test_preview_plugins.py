"""Real follower observation paths and LeRobot resets, with fake arms/cameras only."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=[False, True], ids=["single", "bimanual"])
def preview_robot(request, rig, fake_connect, monkeypatch):
    from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig
    from lerobot_robot_yamkit import yam_follower as module

    events = []

    class Camera:
        is_connected = False
        frame_lock = threading.Lock()

        def connect(self):
            events.append("connect")
            self.is_connected = True

        def read_latest(self):
            events.append("read")
            self.latest_frame = np.full((12, 16, 3), len(events), dtype=np.uint8)
            self.latest_timestamp = time.perf_counter()
            return self.latest_frame

        def disconnect(self):
            events.append("disconnect")
            self.is_connected = False

    class Preview:
        enabled = True

        def __init__(self):
            self.offers = []

        def offer(self, key, frame, *, source_time=None):
            self.offers.append((key, frame, source_time))

        def close(self):
            events.append("close preview")

    camera, preview = Camera(), Preview()

    def claim(names):
        assert names == ["top"]
        events.append("claim")
        return SimpleNamespace(owner="lease", release=lambda: events.append("release"))

    def start(modes, *, owner):
        assert modes == {"top": "rgb"} and owner == "lease"
        events.append("start preview")
        return preview

    rig.control.home_speed = 0
    rig.cameras = {"top": {"type": "opencv", "index_or_path": 99, "width": 16, "height": 12, "fps": 30}}
    rig.save()
    monkeypatch.setattr(module, "make_cameras_from_configs", lambda _: {"top": camera})
    monkeypatch.setattr(module, "claim_from_env", claim)
    monkeypatch.setattr(module, "start_from_env", start)
    config = BiYamFollowerConfig(rig=str(rig.path)) if request.param else YamFollowerConfig(rig=str(rig.path), arm="left_follower")
    robot = module.BiYamFollower(config) if request.param else module.YamFollower(config)
    return robot, camera, preview, events


def test_observation_integrity_and_lifecycle(preview_robot):
    robot, camera, preview, events = preview_robot
    robot.connect()
    assert events == ["claim", "connect", "start preview"]
    obs = robot.get_observation()
    assert events.count("read") == 1
    assert obs["top"] is camera.latest_frame is preview.offers[0][1]
    assert preview.offers[0][2] == camera.latest_timestamp
    assert obs["top"].shape == robot.observation_features["top"]
    robot.disconnect()
    assert events[-3:] == ["disconnect", "close preview", "release"]


def test_preview_failure_does_not_break_observation_or_cleanup(preview_robot, monkeypatch):
    robot, camera, preview, events = preview_robot
    robot.connect()
    monkeypatch.setattr(preview, "offer", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("preview failed")))
    assert robot.get_observation()["top"] is camera.latest_frame
    monkeypatch.setattr(preview, "close", lambda: (_ for _ in ()).throw(RuntimeError("close failed")))
    robot.disconnect()
    assert events[-1] == "release"


def test_preview_start_failure_does_not_release_cameras(preview_robot, monkeypatch):
    from lerobot_robot_yamkit import yam_follower as module

    robot, camera, _, events = preview_robot
    monkeypatch.setattr(module, "start_from_env", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no preview")))
    robot.connect()
    assert camera.is_connected and "release" not in events
    assert robot.get_observation()["top"] is camera.latest_frame
    robot.disconnect()
    assert events[-1] == "release"


def test_failed_camera_release_keeps_lease(preview_robot, monkeypatch):
    robot, camera, _, events = preview_robot
    robot.connect()
    lease = robot._camera_lease
    disconnect = camera.disconnect
    monkeypatch.setattr(camera, "disconnect", lambda: (_ for _ in ()).throw(RuntimeError("busy")))
    with pytest.raises(RuntimeError, match="busy"):
        robot.disconnect()
    assert "release" not in events
    assert robot._camera_lease is lease and robot._opened_cameras == [camera]
    with pytest.raises(RuntimeError, match="previous resources"):
        robot.connect()
    monkeypatch.setattr(camera, "disconnect", disconnect)
    robot.disconnect_no_home()
    assert events[-1] == "release" and robot._camera_lease is None


def test_partial_camera_connect_releases_before_propagating(preview_robot, monkeypatch):
    robot, camera, _, events = preview_robot

    def fail():
        camera.is_connected = True
        raise RuntimeError("partial acquisition")

    monkeypatch.setattr(camera, "connect", fail)
    with pytest.raises(RuntimeError, match="partial acquisition"):
        robot.connect()
    assert not camera.is_connected and events[-1] == "release"


def test_preview_start_cancellation_releases_camera_lease_and_arms(preview_robot, fake_connect, monkeypatch):
    from lerobot_robot_yamkit import yam_follower as module

    robot, camera, _, events = preview_robot

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt("cancel preview startup")

    monkeypatch.setattr(module, "start_from_env", cancel)
    with pytest.raises(KeyboardInterrupt, match="cancel preview startup"):
        robot.connect()
    assert not camera.is_connected and events[-1] == "release"
    assert robot._camera_lease is None and robot._opened_cameras == []
    assert all(arm.closed and not arm.commands for arm in fake_connect.values())


def test_no_home_alias_releases_lease_and_arms_when_preview_close_is_cancelled(
    preview_robot, fake_connect, monkeypatch,
):
    robot, camera, preview, events = preview_robot
    robot.connect()
    handles = robot._sides.values() if hasattr(robot, "_sides") else [robot._h]
    for handle in handles:
        handle.home_speed = 1

    def cancel():
        raise KeyboardInterrupt("cancel preview cleanup")

    monkeypatch.setattr(preview, "close", cancel)
    with pytest.raises(KeyboardInterrupt, match="cancel preview cleanup"):
        robot.disconnect_no_home()
    assert not camera.is_connected and events[-1] == "release"
    assert robot._camera_lease is None and robot._opened_cameras == []
    assert all(arm.closed and not arm.commands for arm in fake_connect.values())


def test_retained_lease_blocks_reconnect_until_release_retried(preview_robot, fake_connect, monkeypatch):
    robot, camera, _, events = preview_robot
    robot.connect()
    lease = robot._camera_lease
    release = lease.release

    def cancel():
        raise KeyboardInterrupt("cancel lease release")

    monkeypatch.setattr(lease, "release", cancel)
    with pytest.raises(KeyboardInterrupt, match="cancel lease release"):
        robot.disconnect_no_home()
    assert not camera.is_connected and robot._opened_cameras == []
    assert robot._camera_lease is lease
    assert all(arm.closed and not arm.commands for arm in fake_connect.values())
    before = events.copy()
    with pytest.raises(RuntimeError, match="previous resources"):
        robot.connect()
    assert events == before
    monkeypatch.setattr(lease, "release", release)
    robot.disconnect_no_home()
    assert robot._camera_lease is None and events[-1] == "release"


def test_installed_lerobot_record_and_reset_both_publish(preview_robot, monkeypatch):
    from lerobot.scripts import lerobot_record as recorder
    from lerobot_teleoperator_yamkit import BiYamLeaderConfig, YamLeaderConfig
    from lerobot_teleoperator_yamkit.yam_leader import BiYamLeader, YamLeader

    robot, _, preview, events = preview_robot
    if robot.name == "bi_yam_follower":
        teleop = BiYamLeader(BiYamLeaderConfig(rig=robot.config.rig))
    else:
        teleop = YamLeader(YamLeaderConfig(rig=robot.config.rig, arm="left_leader"))
    frames = []
    dataset = SimpleNamespace(fps=30, features={}, add_frame=frames.append)
    monkeypatch.setattr(recorder, "build_dataset_frame", lambda features, obs, prefix: {f"{prefix}.{k}": v for k, v in obs.items()})
    robot.connect()
    teleop.connect()
    try:
        args = {"robot": robot, "teleop": teleop, "fps": 30, "events": {"exit_early": False},
                "teleop_action_processor": lambda pair: pair[0], "robot_action_processor": lambda pair: pair[0],
                "robot_observation_processor": lambda obs: obs, "control_time_s": 0.1}
        recorder.record_loop(**args, dataset=dataset)
        count = len(preview.offers)
        assert count == len(frames) > 0
        assert all(frame["observation.top"] is offer[1] for frame, offer in zip(frames, preview.offers))
        recorder.record_loop(**args)  # LeRobot's reset calls exactly this path, dataset=None
        assert len(preview.offers) > count and len(frames) == count
        assert events.count("read") == len(preview.offers)
    finally:
        robot.disconnect()
        teleop.disconnect()
