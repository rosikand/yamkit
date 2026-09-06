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


def _drain_js(ctx):
    for _ in range(100):
        if not ctx.execute_pending_job():
            return
    pytest.fail("episode viewer has an unbounded microtask loop")


@pytest.fixture
def episode_js():
    quickjs = pytest.importorskip("quickjs")
    src = (UI / "app.js").read_text()
    ctx = quickjs.Context()
    ctx.eval("""
      var nodes={}, timers={}, nextTimer=0, disposedCharts=0, loadedVideos=0, pausedVideos=0;
      var deferEpisode=false, resolveEpisode, failure=null, renders=0;
      function $(selector) { return nodes[selector] ||= {innerHTML:'',textContent:'',isConnected:true,
        value:0,appendChild:()=>{}}; }
      var video={currentTime:0,paused:true,src:'episode.mp4',
        play(){this.paused=false;return Promise.resolve();},
        pause(){this.paused=true;pausedVideos++;},
        removeAttribute(name){delete this[name];},load(){loadedVideos++;}};
      var document={createElement:()=>({}),getElementById:()=>video,querySelectorAll:()=>[]};
      var location={hash:'#/live'};
      var pages={live:{render(){renders++;$('#ep-body').isConnected=false;}}};
      function setInterval(callback){timers[++nextTimer]=callback;return nextTimer;}
      function clearInterval(id){delete timers[id];}
      var performance={now:()=>0};
      function esc(value){return value;} function errBanner(value){return value;}
      function seriesColors(){return {};}
      function makeChart(){return {setCursor:()=>{},dispose(){disposedCharts++;}};}
      var series={timestamp:[0,1],names:['joint'],action:[[0],[1]],'observation.state':[[0],[1]]};
      function api(){return deferEpisode?new Promise(resolve=>{resolveEpisode=resolve;}):Promise.resolve(series);}
    """)
    # Routing closes MJPEG requests before removing a page. The episode document
    # has no camera tiles, but exercise the actual shared cleanup helper.
    ctx.eval(src[src.index("function releaseCameraStreams"):src.index("function syncCams")])
    ctx.eval(src[src.index("async function renderEpisodeViewer"):src.index("// tiny canvas line chart")])
    ctx.eval(src[src.index("let current = null;"):src.index('window.addEventListener("hashchange"')])
    return ctx


@pytest.mark.parametrize("with_video", [True, False])
def test_leaving_playing_episode_stops_media_timers_and_disposes_charts(episode_js, with_video):
    ctx = episode_js
    videos = {"top": {"from_timestamp": 0, "to_timestamp": 1}} if with_video else {}
    detail = {"episode_list": [{"episode_index": 0, "videos": videos}]}
    ctx.eval("renderEpisodeViewer($('#viewer'),'test'," + json.dumps(detail) + ",0)")
    _drain_js(ctx)
    ctx.eval("$('#ep-play').onclick()")
    assert ctx.eval("Object.keys(timers).length") == 1
    if with_video:
        assert not ctx.eval("video.paused")

    ctx.eval("route()")
    assert ctx.eval("Object.keys(timers).length") == 0
    assert ctx.eval("disposedCharts") == 1
    assert ctx.eval("renders") == 1
    if with_video:
        assert ctx.eval("video.paused")
        assert ctx.eval("loadedVideos") == 1
        assert ctx.eval("video.src === undefined")
    ctx.eval("route()")
    assert ctx.eval("disposedCharts") == 1  # cleanup is consumed once, including theme/navigation redraws


def test_episode_response_after_navigation_cannot_recreate_viewer(episode_js):
    ctx = episode_js
    ctx.eval("""
      deferEpisode=true;
      renderEpisodeViewer($('#viewer'),'test',{episode_list:[]},0).catch(error=>{failure=String(error);});
      route(); resolveEpisode(series);
    """)
    _drain_js(ctx)
    assert ctx.eval("failure") is None
    assert ctx.eval("pageCleanup") is None
    assert ctx.eval("$('#ep-body').innerHTML") == ""
