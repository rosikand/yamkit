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


def _camera_context():
    quickjs = pytest.importorskip("quickjs")
    src = (UI / "app.js").read_text()
    ctx = quickjs.Context()
    ctx.eval("let overview = null;")
    ctx.eval(src[src.index("let camsRendered = null;"):src.index("function armPanelHTML")])
    return ctx


def test_camera_age_labels_do_not_claim_stale_images_are_live():
    ctx = _camera_context()
    assert ctx.eval('cameraStateText({preview_state:"waiting"})') == "waiting for frames"
    assert ctx.eval('cameraStateText({preview_state:"stale",frame_age_s:4.25})') == "stale · last frame 4.3 s ago"
    assert "unavailable" in ctx.eval('cameraStateText({preview_state:"unavailable"})')
    assert ctx.eval('cameraStateText({preview_state:"live",preview_source:"session"})') == "live · session camera"


def test_camera_reconnect_is_bounded_and_does_not_replace_tile_for_status_changes():
    ctx = _camera_context()
    ctx.eval('''
      let now = 10000;
      Date.now = () => now;
      overview = {cameras:[{name:"top",preview_source:"session",preview_generation:2,preview_state:"stale",frame_age_s:4}]};
      const image = {dataset:{connectedAt:"8000"},complete:false,naturalWidth:640,src:"original"};
      const label = {textContent:""};
      const tile = {dataset:{cam:"top"}};
      image.parentElement = tile;
      const slot = {querySelectorAll: () => [tile], innerHTML:"unchanged"};
      const $ = (selector, parent) => selector === "#cams-slot" ? slot : selector === "img" ? image : label;
      camsRendered = cameraKey();
      cameraStreamError(image);
      syncCams();
    ''')
    assert ctx.eval("image.src") == "original"
    assert ctx.eval("label.textContent").startswith("stale")
    ctx.eval("now = 12001; syncCams();")
    retried = ctx.eval("image.src")
    assert retried.startswith("/api/cameras/top/stream?generation=2&retry=")
    ctx.eval("overview.cameras[0].preview_state = 'live'; image.complete = true; now = 18000; syncCams();")
    assert ctx.eval("image.src") == retried
    assert ctx.eval("slot.innerHTML") == "unchanged"
    ctx.eval("now = 43000; syncCams();")
    assert ctx.eval("image.src") != retried  # bounded renewal handles native MJPEG's silent EOF
