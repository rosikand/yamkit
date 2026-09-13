"""Faulty child diagnostics must not strand UI ownership or break status/Stop JSON."""

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from yamkit.ui.server import create_app
from yamkit.ui.sessions import SessionManager, parse_line


@pytest.mark.parametrize("line", [
    "[ 50.0Hz] a->b: idle err=nanrad grip=-",
    "[ 50.0Hz] a->b: idle err=infrad grip=nan",
    "    left_follower q=[nan +inf -inf 1e999 0 1] grip=nan",
    '[yamkit-result] {"latency": NaN, "nested": [Infinity, -Infinity, 1e999]}',
])
def test_nonfinite_child_diagnostics_remain_json_serializable(line):
    manager = SessionManager()
    parse_line(line, manager.parsed)
    response = JSONResponse(manager.status())
    value = json.loads(response.body)
    assert value["parsed"]
    assert "NaN" not in response.body.decode() and "Infinity" not in response.body.decode()


def test_nonfinite_arm_display_is_unknown_not_zero():
    parsed = {}
    parse_line("    left_follower q=[nan +inf -inf 1e999 0 1] grip=nan", parsed)
    assert parsed["arms"]["left_follower"]["q"] == [None, None, None, None, 0.0, 1.0]
    assert parsed["arms"]["left_follower"]["gripper"] is None


def test_unknown_joint_value_renders_without_claiming_a_measured_zero():
    quickjs = pytest.importorskip("quickjs")
    source = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()
    ctx = quickjs.Context()
    ctx.eval("function esc(value){return value;}")
    start = source.index("function armPanelHTML(")
    end = source.index("\n}\n", start) + 3
    ctx.eval(source[start:end])
    rendered = ctx.eval('armPanelHTML("left_follower",{q:[null,1.25],gripper:null},"follower")')
    assert '<span class="val">–</span>' in rendered
    assert '<span class="val">1.250</span>' in rendered
    assert "0.000" not in rendered and "NaN" not in rendered


def test_invalid_native_stdout_bytes_do_not_strand_completion_or_next_start(tmp_path):
    finalized = []
    ownership = []
    manager = SessionManager(on_exit=finalized.append,
                             on_camera_acquire=lambda owner: ownership.append(("acquire", owner)),
                             on_camera_release=lambda owner: ownership.append(("release", owner)))
    log_path = tmp_path / "session.log"
    manager.start("software-fixture", [sys.executable, "-c",
                  ("from yamkit.camera_ownership import claim_from_env; "
                  "lease=claim_from_env(['fixture-camera']); "
                  "import os; os.write(1,b'native \\xff diagnostic\\n[yamkit-prepare] ready\\n')")],
                  meta={"session_log_path": str(log_path)})
    assert manager.wait(timeout=5) == 0
    assert not manager.active and not manager.cameras_owned
    assert manager.parsed["preparation_phase"] == "ready"
    assert len(finalized) == 1 and finalized[0]["returncode"] == 0
    assert [event for event, _ in ownership] == ["acquire", "release"]
    assert ownership[0][1] == ownership[1][1]
    assert "native \ufffd diagnostic" in log_path.read_text()
    manager.start("software-fixture", [sys.executable, "-c", "print('next session')"])
    assert manager.wait(timeout=5) == 0
    assert len(finalized) == 2


def test_nonfinite_fault_keeps_real_status_and_stop_routes_available(rig, tmp_path):
    manager = SessionManager()
    app = create_app(rig.path, outputs_dir=tmp_path, session_manager=manager)
    with TestClient(app) as client:
        manager.start("software-fixture", [sys.executable, "-u", "-c",
                      ("import time; print('[ 50.0Hz] a->b: idle err=nanrad grip=nan'); "
                      "print('FAULT_FIXTURE_READY'); time.sleep(30)")])
        deadline = time.monotonic() + 5
        while "FAULT_FIXTURE_READY" not in manager.log and time.monotonic() < deadline:
            time.sleep(.01)
        assert "FAULT_FIXTURE_READY" in manager.log
        status = client.get("/api/session")
        assert status.status_code == 200 and status.json()["active"]
        assert status.json()["parsed"]["pairs"]["a->b"]["error_rad"] is None
        stopped = client.post("/api/session/stop")
        assert stopped.status_code == 200 and stopped.json()["stop_requested"]
        assert manager.wait(timeout=5) is not None
        assert not client.get("/api/session").json()["active"]
