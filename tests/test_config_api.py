"""Settings must reject invalid control values before touching the rig or cameras."""

import pytest
from fastapi.testclient import TestClient

from yamkit.config import RigConfig
from yamkit.ui.server import create_app


@pytest.mark.parametrize("control", [
    {"max_joint_speed": -1},
    {"max_gripper_speed": 0},
    {"teleop_hz": 0},
    {"sync_seconds": -1},
    {"home_speed": -1},
    {"leader_home_speed": -1},
    {"bilateral_kp": -1},
    {"engage_button": -1},
    {"engage_button": 0.5},
    {"engage_button": True},
    {"max_joint_speed": True},
    {"max_joint_speed": "3"},
    {"max_joint_speed": None},
    {"max_joint_speed": [3]},
])
def test_invalid_settings_preserve_rig_and_camera_configuration(rig, tmp_path, control):
    rig.cameras = {"top": {"type": "opencv", "index_or_path": "/dev/video99", "fps": 30}}
    rig.save()
    before = rig.path.read_bytes()
    app = create_app(rig.path, datasets_dir=tmp_path / "datasets", outputs_dir=tmp_path / "outputs")
    with TestClient(app) as client:
        cameras = client.get("/api/cameras").json()
        response = client.post("/api/config", json={"control": control})
        assert response.status_code == 422
        assert rig.path.read_bytes() == before
        assert client.get("/api/config").json()["found"] is True
        assert client.get("/api/cameras").json() == cameras
    assert RigConfig.load(rig.path).validate() == []


def test_nonfinite_control_rejected_without_overwriting_rig(rig, tmp_path):
    before = rig.path.read_bytes()
    app = create_app(rig.path, datasets_dir=tmp_path / "datasets", outputs_dir=tmp_path / "outputs")
    with TestClient(app) as client:
        response = client.post("/api/config", content='{"control":{"max_joint_speed":1e999}}',
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert rig.path.read_bytes() == before


def test_valid_settings_keep_zero_home_option_and_integer_button(rig, tmp_path):
    app = create_app(rig.path, datasets_dir=tmp_path / "datasets", outputs_dir=tmp_path / "outputs")
    with TestClient(app) as client:
        response = client.post("/api/config", json={"control": {
            "engage_button": 1, "home_speed": 0, "leader_home_speed": 0, "max_joint_speed": 2.5,
        }})
        assert response.status_code == 200
    saved = RigConfig.load(rig.path)
    assert saved.control.engage_button == 1
    assert saved.control.home_speed == saved.control.leader_home_speed == 0
    assert saved.control.max_joint_speed == 2.5
