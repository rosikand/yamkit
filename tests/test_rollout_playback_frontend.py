"""Hardware-free JavaScript regression tests for the saved rollout video viewer."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def playback_js():
    quickjs = pytest.importorskip("quickjs")
    source = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()
    ctx = quickjs.Context()
    ctx.eval("""
      const nodes = {}, timers = {};
      let timerSequence = 0, playError = null, paused = 0, unloaded = 0;
      function $(selector) { return nodes[selector] ||= {textContent:'',value:'0',max:'0'}; }
      function setInterval(callback) { timers[++timerSequence] = callback; return timerSequence; }
      function clearInterval(id) { delete timers[id]; }
      const videos = Array.from({length:3}, () => ({
        currentTime:0, duration:5, paused:true, ended:false, src:'fixture.mp4',
        play() { this.paused=false; return playError ? Promise.reject(playError) : Promise.resolve(); },
        pause() { this.paused=true; paused++; },
        removeAttribute(name) { delete this[name]; }, load() { unloaded++; }
      }));
      const slot = {querySelectorAll:()=>videos};
    """)
    ctx.eval(source[source.index("function setupRunPlayback"):source.index("// ---- models ----")])
    return ctx


def drain_jobs(ctx):
    for _ in range(100):
        if not ctx.execute_pending_job():
            return
    pytest.fail("Playback promise handling did not settle")


def test_shared_play_pause_scrub_and_sync_controls_all_three_videos(playback_js):
    ctx = playback_js
    ctx.eval("const cleanup=setupRunPlayback(slot,5); $('#run-play').onclick();")
    drain_jobs(ctx)
    assert ctx.eval("videos.every(video=>!video.paused)")
    assert ctx.eval("Object.keys(timers).length") == 1
    assert ctx.eval("$('#run-play').textContent") == "Pause"
    ctx.eval("videos[0].currentTime=1; videos[1].currentTime=.5; Object.values(timers)[0]();")
    assert ctx.eval("videos[1].currentTime") == 1
    assert ctx.eval("$('#run-scrub').value") == 1
    ctx.eval("$('#run-scrub').value='3.25'; $('#run-scrub').oninput();")
    assert ctx.eval("videos.every(video=>video.paused && video.currentTime===3.25)")
    assert ctx.eval("Object.keys(timers).length") == 0
    assert ctx.eval("$('#run-play-time').textContent") == "3.3 / 5.0 s"
    ctx.eval("$('#run-play').onclick(); $('#run-play').onclick();")
    assert ctx.eval("videos.every(video=>video.paused)")
    assert ctx.eval("Object.keys(timers).length") == 0


def test_natural_end_stops_all_views_and_play_restarts_at_zero(playback_js):
    ctx = playback_js
    ctx.eval("setupRunPlayback(slot,5); $('#run-play').onclick();")
    ctx.eval("videos[0].currentTime=5; videos[0].ended=true; videos[0].onended();")
    assert ctx.eval("videos.every(video=>video.paused)")
    assert ctx.eval("Object.keys(timers).length") == 0
    ctx.eval("$('#run-play').onclick();")
    assert ctx.eval("videos.every(video=>video.currentTime===0)")


def test_unknown_duration_uses_real_video_metadata(playback_js):
    ctx = playback_js
    ctx.eval("setupRunPlayback(slot,null); videos[0].duration=2.75; videos[0].onloadedmetadata();")
    assert ctx.eval("$('#run-scrub').max") == 2.75
    assert ctx.eval("$('#run-play-time').textContent") == "0.0 / 2.8 s"


def test_dispose_pauses_unloads_all_videos_and_removes_polling(playback_js):
    ctx = playback_js
    ctx.eval("const cleanup=setupRunPlayback(slot,5); $('#run-play').onclick(); cleanup();")
    drain_jobs(ctx)
    assert ctx.eval("Object.keys(timers).length") == 0
    assert ctx.eval("videos.every(video=>video.paused && video.src===undefined)")
    assert ctx.eval("videos.every(video=>video.onerror===null && video.onloadedmetadata===null)")
    assert ctx.eval("unloaded") == 3


def test_failed_video_play_is_visible_and_cancels_all_views(playback_js):
    ctx = playback_js
    ctx.eval("playError={name:'NotSupportedError'}; setupRunPlayback(slot,5); $('#run-play').onclick();")
    drain_jobs(ctx)
    assert ctx.eval("videos.every(video=>video.paused)")
    assert ctx.eval("Object.keys(timers).length") == 0
    assert "Video playback failed" in ctx.eval("$('#run-play-error').textContent")


def test_late_play_rejection_after_navigation_cannot_touch_disposed_dom(playback_js):
    ctx = playback_js
    ctx.eval("""
      let rejectPlay;
      videos[0].play=()=>new Promise((resolve,reject)=>{rejectPlay=reject;});
      const cleanup=setupRunPlayback(slot,5); $('#run-play').onclick(); cleanup();
      rejectPlay({name:'NotSupportedError'});
    """)
    drain_jobs(ctx)
    assert ctx.eval("$('#run-play-error').textContent") == ""
    assert ctx.eval("Object.keys(timers).length") == 0


@pytest.mark.parametrize("state,label", [
    ("pending", "Recording / saving…"),
    ("available", "Watch recording"),
    ("partial", "Partial recording"),
    ("not_recorded", "Not recorded"),
    ("unavailable", "Recording unavailable"),
])
def test_history_recording_labels_are_separate_from_rollout_success(state, label):
    quickjs = pytest.importorskip("quickjs")
    source = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()
    ctx = quickjs.Context()
    ctx.eval(source[source.index("function recordingLabel"):source.index("function updateRunDetail")])
    record = {"status": "success", "recording": {"state": state}}
    assert ctx.eval("recordingLabel(" + json.dumps(record) + ")") == label


@pytest.fixture
def detail_js():
    quickjs = pytest.importorskip("quickjs")
    source = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()
    ctx = quickjs.Context()
    ctx.eval("""
      const nodes = {};
      let playbackStarts=0, playbackDisposals=0;
      function $(selector) { return nodes[selector] ||= {innerHTML:'',textContent:''}; }
      function esc(value) { return String(value??''); }
      function st(ok,value) { return value; }
      function fmtDate(value) { return value; }
      function fmtDur(value) { return value; }
      function setupRunPlayback() { playbackStarts++; return ()=>{playbackDisposals++;}; }
      const view={id:'fixture',alive:true,body:{isConnected:true},replayKey:null};
      const detail={status:'success',task:'green bowl',videos:['top.mp4','left_wrist.mp4','right_wrist.mp4'],
        recording:{state:'available',duration_s:2},upload:{status:'uploading'}};
    """)
    ctx.eval(source[source.index("function rolloutUploadHTML"):source.index("function syncRunStop")])
    ctx.eval(source[source.index("function updateRunDetail"):source.index("function setupRunPlayback")])
    return ctx


def test_upload_status_poll_does_not_recreate_or_dispose_playback(detail_js):
    ctx = detail_js
    ctx.eval("updateRunDetail(view,detail); detail.upload.status='uploaded'; updateRunDetail(view,detail);")
    assert ctx.eval("playbackStarts") == 1
    assert ctx.eval("playbackDisposals") == 0
    assert not ctx.eval("view.pending")
    assert "uploaded" in ctx.eval("$('#run-summary').innerHTML")


def test_pending_export_hides_partial_video_files_then_renders_finished_recording(detail_js):
    ctx = detail_js
    ctx.eval("detail.recording.state='pending'; updateRunDetail(view,detail);")
    assert ctx.eval("view.pending")
    assert ctx.eval("playbackStarts") == 0
    assert "<video" not in ctx.eval("$('#run-recording').innerHTML")
    ctx.eval("detail.recording.state='available'; updateRunDetail(view,detail);")
    assert ctx.eval("playbackStarts") == 1
    assert ctx.eval("$('#run-recording').innerHTML").count("<video") == 3


@pytest.mark.parametrize("detached", [False, True])
def test_refresh_result_after_navigation_cannot_render_or_restart_video(detail_js, detached):
    ctx = detail_js
    ctx.eval("view.body.isConnected=false;" if detached else "view.alive=false;")
    ctx.eval("updateRunDetail(view,detail);")
    assert ctx.eval("playbackStarts") == 0
    assert ctx.eval("Object.keys(nodes).length") == 0
