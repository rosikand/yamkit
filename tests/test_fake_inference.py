"""The explicit fake CLI can never turn a historical approval into device I/O."""
# ruff: noqa: F811

import socket

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from tests.test_pi05_workflow import native  # noqa: F401
from yamkit import backend_workflow, cli, fake_inference, pi05_workflow
from yamkit.fake_inference import SavedRobot, forbid_device_io
from yamkit.inference.mapping import YAM_NAMES


def test_fake_device_guard_refuses_can_and_camera_construction():
    with forbid_device_io():
        with pytest.raises(RuntimeError, match="CAN"):
            socket.socket(socket.AF_CAN, socket.SOCK_RAW)
        with pytest.raises(RuntimeError, match="Real camera"):
            cv2.VideoCapture(0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            pass  # TCP class remains usable by the authenticated inference transport.


def test_saved_robot_uses_recorded_rgb_and_fake_target_receipts_only():
    frames = [{"state": np.zeros(14), "top": np.full((8, 12, 3), i, dtype=np.uint8)} for i in (1, 2)]
    checked = []
    robot = SavedRobot(frames, checked.append)
    robot.connect()
    assert np.all(robot.get_observation()["top"] == 1)
    action = dict.fromkeys(YAM_NAMES, .5)
    assert robot.send_reference_action(action, dispatch_check=lambda: None) == action
    observation = robot.get_observation()
    assert np.all(observation["top"] == 2) and observation["left_gripper.pos"] == .5
    robot.disconnect(home=False)
    with pytest.raises(RuntimeError, match="released"):
        robot.send_reference_action(action, dispatch_check=lambda: None)
    assert checked == [action]


@pytest.mark.parametrize("flag", ["--confirm-supervised", "--accept-mapping", "--dry-run"])
def test_fake_cli_rejects_approval_and_dry_run_before_preparation(flag, monkeypatch):
    from yamkit import inference_workflow

    monkeypatch.setattr(inference_workflow, "prepare_inference", lambda **_: pytest.fail("invalid fake selection"))
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "pi05_yam",
                                         "--task", "cube", "--fake-hardware", flag])
    assert result.exit_code == 2 and "cannot carry motion approval" in result.output


def test_fake_cli_is_explicit_and_has_no_confirmation_or_physical_delegate(native, monkeypatch):
    monkeypatch.setattr(backend_workflow, "assert_ui_idle", lambda **_: None)
    monkeypatch.setattr(pi05_workflow, "run_prepared_pi05", lambda *_a, **_k: pytest.fail("real delegate forbidden"))
    calls = []
    monkeypatch.setattr(fake_inference, "run_fake_pi05", lambda selection, **kw: calls.append(kw) or {"status": "completed"})
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "pi05_yam",
                                         "--task", "cube", "--rig", str(native.rig), "--fake-hardware"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and "I am on site" not in result.output
    assert '"hardware_tested": false' in result.output and '"motion_approval_received": false' in result.output


def test_arbitrary_fake_factory_cannot_bypass_physical_approval(native):
    with pytest.raises(ValueError, match="only SavedRobot"):
        pi05_workflow.run_prepared_pi05(None, confirm_supervised=False, accept_mapping=False, fake_robot=object())


def test_fake_cli_reports_nonzero_artifact_or_execution_failure(native, monkeypatch):
    monkeypatch.setattr(backend_workflow, "assert_ui_idle", lambda **_: None)
    monkeypatch.setattr(fake_inference, "run_fake_pi05", lambda *_a, **_kw: {
        "status": "fault", "exit_status": 1, "released": True,
    })
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "pi05_yam",
                                         "--task", "cube", "--rig", str(native.rig), "--fake-hardware"])
    assert result.exit_code == 1
    assert '"status": "fault"' in result.output and '"hardware_tested": false' in result.output
