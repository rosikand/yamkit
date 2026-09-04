"""The web UI's page code must at least parse: it is edited without a browser or node in the loop."""

import json
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "ui"


def test_app_js_parses():
    quickjs = pytest.importorskip("quickjs")
    src = (UI / "app.js").read_text()
    ctx = quickjs.Context()
    try:
        ctx.eval("new Function(" + json.dumps(src) + ")")  # compile only — never runs the page code
    except quickjs.JSException as e:  # pragma: no cover - the message is the point
        pytest.fail(f"ui/app.js does not parse: {e}")


def test_app_js_uses_direct_mjpeg_streams_only():
    src = (UI / "app.js").read_text()
    assert "/stream" in src and "/api/cameras/${encodeURIComponent(c.name)}/stream" in src
    for gone in ("/frame", "camPoll", "createObjectURL"):
        assert gone not in src, f"snapshot polling must not come back ({gone})"
    assert (UI / "index.html").read_text().count("app.js") == 1
