"""Camera previews published by the recording process for the web UI (yamkit.frames + camstream.file_frames)."""

import threading
import time

import numpy as np

from yamkit.frames import FramePublisher
from yamkit.ui.camstream import file_frames


def test_publisher_writes_jpegs_atomically_and_throttles(tmp_path):
    pub = FramePublisher(tmp_path / "frames", hz=1000)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[..., 0] = 255  # red in RGB
    assert pub.publish("top", frame) is True
    f = tmp_path / "frames" / "top.jpg"
    assert f.is_file() and f.read_bytes().startswith(b"\xff\xd8") and not list((tmp_path / "frames").glob("*.tmp"))
    slow = FramePublisher(tmp_path / "slow", hz=1)
    assert slow.publish("top", frame) is True and slow.publish("top", frame) is False  # second one throttled
    assert slow.publish("left_wrist", frame) is True  # per camera
    assert FramePublisher(None).publish("top", frame) is False  # disabled without $YAMKIT_FRAMES_DIR
    pub.clear()
    assert not list((tmp_path / "frames").glob("*.jpg"))


def test_publisher_is_enabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMKIT_FRAMES_DIR", str(tmp_path / "env"))
    assert FramePublisher().enabled
    monkeypatch.delenv("YAMKIT_FRAMES_DIR")
    assert not FramePublisher().enabled


def test_file_frames_streams_only_new_complete_jpegs(tmp_path):
    path = tmp_path / "top.jpg"
    stop = threading.Event()
    gen = file_frames(path, stop, hz=200, stale_s=0.3)
    assert list(gen) == []  # nothing ever published: ends after stale_s so the tile can fall back
    path.write_bytes(b"\xff\xd8JPEG-1")
    gen = file_frames(path, stop, hz=200)
    part = next(gen)
    assert b"Content-Type: image/jpeg" in part and part.endswith(b"JPEG-1\r\n")
    time.sleep(0.02)
    path.write_bytes(b"\xff\xd8JPEG-2")
    import os

    os.utime(path, None)
    assert next(gen).endswith(b"JPEG-2\r\n")
    stop.set()
    assert list(gen) == []


def test_follower_publishes_previews_of_its_cameras(rig, fake_connect, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    from lerobot.robots.utils import make_robot_from_config
    from lerobot_robot_yamkit import YamFollowerConfig

    robot = make_robot_from_config(YamFollowerConfig(rig=str(rig.path), arm="left_follower"))

    class FakeCam:
        is_connected = True

        def connect(self):
            pass

        def disconnect(self):
            pass

        def read_latest(self):
            return np.full((8, 8, 3), 90, dtype=np.uint8)

    robot.cameras = {"observation.images.top": FakeCam()}
    robot._frames = FramePublisher(tmp_path / "frames")
    robot.connect()
    obs = robot.get_observation()
    assert obs["observation.images.top"].shape == (8, 8, 3)
    assert (tmp_path / "frames" / "top.jpg").is_file()
    robot.disconnect()
