"""Display-only rollout clocks: no hardware, network, wall-clock guesses or fake progress."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def progress_js():
    quickjs = pytest.importorskip("quickjs")
    source = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()
    ctx = quickjs.Context()
    ctx.eval("""
      let now=1000, sessionReceivedAt=1000, sessionDisconnected=false, sessionFrozenDelta=null,sessionReadFailed=false;
      const performance={now:()=>now};
      let session={active:true,mode:'rollout',meta:{task:'green bowl'},rollout_progress:{
        phase:'running',policy_elapsed_s:2,target_duration_s:20,policy_timer_running:true,
        phase_elapsed_s:2,resources_released:false,operation_id:'run1',capture_requested:true
      }};
    """)
    ctx.eval(source[source.index("const fmtClock ="):source.index("// series colors")])
    ctx.eval(source[source.index("const ROLLOUT_SAVE_PHASES"):source.index("function armPanelHTML")])
    return ctx


def model(ctx, expression="currentRolloutProgress()"):
    return json.loads(ctx.eval(f"JSON.stringify({expression})"))


@pytest.fixture
def polling_js(progress_js):
    source = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()
    progress_js.eval("""
      let sessionRequest=0, sessionApplied=0, lastSessionKey='',sessionFlight=null,stopRequestsPending=0;
      const pending=[];
      function api() { return new Promise((resolve,reject)=>pending.push({resolve,reject})); }
      function pollRead() { return api(); }
      function updateSidebar() {} function refreshOverview() {}
      const document={visibilityState:'visible',dispatchEvent(){}};
      class CustomEvent { constructor(name) {} }
    """)
    progress_js.eval(source[source.index("function sessionUpdated()"):source.index("function updateSidebar()")])
    return progress_js


def drain_jobs(ctx):
    for _ in range(100):
        if not ctx.execute_pending_job():
            return
    pytest.fail("Session polling promise handling did not settle")


@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00.0"), (5.25, "00:05.3"), (59.99, "01:00.0"), (90, "01:30.0"),
    (3600, "60:00.0"), (-2, "00:00.0"), (None, "–"),
])
def test_clock_has_readable_minutes_seconds_and_tenths(progress_js, seconds, expected):
    assert progress_js.eval(f"fmtClock({json.dumps(seconds)})") == expected


@pytest.mark.parametrize("phase", ["preparing", "homing"])
def test_preparation_never_consumes_requested_policy_time(progress_js, phase):
    progress_js.eval(f"session.rollout_progress.phase={json.dumps(phase)}; "
                     "session.rollout_progress.policy_elapsed_s=null; "
                     "session.rollout_progress.policy_timer_running=false; now=3000;")
    result = model(progress_js)
    assert result["clock"] == "Policy timer not started"
    assert result["elapsed"] is None and result["remaining"] is None
    assert result["barValue"] is None and result["stage"] == 0


def test_running_clock_uses_monotonic_snapshot_delta_and_survives_reload(progress_js):
    progress_js.eval("now=1250;")
    result = model(progress_js)
    assert result["elapsed"] == 2.25 and result["remaining"] == 17.75
    assert result["barValue"] == 2.25 and result["barMax"] == 20
    progress_js.eval("session.rollout_progress.policy_elapsed_s=12; now=2000; sessionReceivedAt=2000;")
    assert model(progress_js)["elapsed"] == 12  # Fresh server snapshot, not time since page load.


def test_hanging_poll_freezes_extrapolation_after_three_seconds(progress_js):
    progress_js.eval("now=9000;")
    first = model(progress_js)
    progress_js.eval("now=90000;")
    second = model(progress_js)
    assert first["elapsed"] == second["elapsed"] == 5
    assert first["stale"] and second["stale"]
    assert second["phase"] == "running" and not second["finished"]


def test_failed_poll_freezes_at_disconnect_without_changing_robot_state(progress_js):
    progress_js.eval("now=1600; sessionDisconnected=true; sessionFrozenDelta=.6;")
    before = model(progress_js)
    progress_js.eval("now=5000;")
    after = model(progress_js)
    assert before["elapsed"] == after["elapsed"] == 2.6
    assert after["stale"] and after["phase"] == "running" and not after["released"]


def test_slow_poll_response_stays_stale_until_fresh_response(polling_js):
    ctx = polling_js
    ctx.eval("refreshSession(); now=4500; pending[0].resolve(session);")
    drain_jobs(ctx)
    assert model(ctx)["stale"] and model(ctx)["elapsed"] == 2
    ctx.eval("now=6000;")
    assert model(ctx)["elapsed"] == 2
    ctx.eval("refreshSession(); now=6100; pending[1].resolve(session);")
    drain_jobs(ctx)
    assert not model(ctx)["stale"]
    ctx.eval("now=6400;")
    assert model(ctx)["elapsed"] == 2.3


def test_actual_failed_poll_freezes_at_failure_and_keeps_stop_state(polling_js):
    ctx = polling_js
    ctx.eval("refreshSession(); now=1400; pending[0].reject(new Error('offline'));")
    drain_jobs(ctx)
    assert model(ctx)["stale"] and model(ctx)["elapsed"] == 2.4
    ctx.eval("now=3000;")
    assert model(ctx)["elapsed"] == 2.4
    assert ctx.eval("session.active")


def test_late_old_poll_cannot_regress_a_confirmed_released_phase(polling_js):
    ctx = polling_js
    ctx.eval("""
      const old=JSON.parse(JSON.stringify(session));
      refreshSession();
      applySessionReceipt({active:true,mode:'rollout',meta:{operation_id:'new-run'},
        rollout_progress:{phase:'saving_frames',resources_released:true}},now);
    """)
    drain_jobs(ctx)
    ctx.eval("pending[0].resolve(old);")
    drain_jobs(ctx)
    assert model(ctx)["phase"] == "saving_frames" and model(ctx)["released"]


def test_unresolved_periodic_polls_coalesce_instead_of_building_a_queue(polling_js):
    ctx = polling_js
    ctx.eval("for(let i=0;i<30;i++){now+=1000;refreshSession();}")
    assert ctx.eval("pending.length") == 1
    assert ctx.eval("sessionRequest") == 1
    ctx.eval("pending[0].resolve(session);")
    drain_jobs(ctx)
    ctx.eval("refreshSession();")
    assert ctx.eval("pending.length") == 2


def test_hidden_or_stop_pending_poll_does_not_queue_gets(polling_js):
    ctx = polling_js
    ctx.eval("document.visibilityState='hidden';for(let i=0;i<30;i++)refreshSession();")
    assert ctx.eval("pending.length") == 0
    ctx.eval("document.visibilityState='visible';stopRequestsPending=1;refreshSession();")
    assert ctx.eval("pending.length") == 0
    ctx.eval("stopRequestsPending=0;refreshSession();")
    assert ctx.eval("pending.length") == 1


def test_timeout_is_delayed_not_a_claim_that_the_connection_was_lost(polling_js):
    ctx = polling_js
    ctx.eval("refreshSession();now=6000;pending[0].reject({name:'AbortError'});")
    drain_jobs(ctx)
    assert model(ctx)["stale"] and not model(ctx)["readFailed"]
    ctx.eval("refreshSession();pending[1].reject({name:'TypeError'});")
    drain_jobs(ctx)
    assert model(ctx)["readFailed"]
    ctx.eval("refreshSession();pending[2].resolve(session);")
    drain_jobs(ctx)
    assert not model(ctx)["stale"] and not model(ctx)["readFailed"]


def test_launch_receipt_immediately_replaces_previous_released_state_and_fences_old_get(polling_js):
    ctx = polling_js
    ctx.eval("""
      session={active:false,mode:'rollout',started_at:100,meta:{operation_id:'old-run'},
        rollout_progress:{phase:'done',resources_released:true,operation_id:'old-run'}};
      const old=JSON.parse(JSON.stringify(session));refreshSession();
      applySessionReceipt({active:true,mode:'rollout',started_at:200,meta:{operation_id:'new-run'},
        rollout_progress:{phase:'preparing',resources_released:false,operation_id:'new-run'}},now);
    """)
    assert ctx.eval("session.active")
    assert ctx.eval("session.meta.operation_id") == "new-run"
    assert not model(ctx)["released"]
    ctx.eval("pending[0].resolve(old);")
    drain_jobs(ctx)
    assert ctx.eval("session.meta.operation_id") == "new-run" and not model(ctx)["released"]


def test_late_launch_receipt_cannot_regress_same_operation_or_replace_newer_run(polling_js):
    ctx = polling_js
    ctx.eval("""
      session.started_at=200;session.meta.operation_id='current-run';
      applySessionReceipt({active:true,mode:'rollout',started_at:200,meta:{operation_id:'current-run'},
        rollout_progress:{phase:'preparing',resources_released:false}},now);
    """)
    assert model(ctx)["phase"] == "running"
    ctx.eval("applySessionReceipt({active:false,started_at:100,meta:{operation_id:'old-run'},rollout_progress:{phase:'done',resources_released:true}},now);")
    assert ctx.eval("session.meta.operation_id") == "current-run"
    assert not model(ctx)["released"]


def test_stop_receipt_applies_current_operation_but_cannot_regress_confirmed_end(polling_js):
    ctx = polling_js
    ctx.eval("""
      session.meta.operation_id='current-run';session.rollout_progress.server_time=100;
      applySessionReceipt({active:true,stopping:true,mode:'rollout',meta:{operation_id:'current-run'},
        rollout_progress:{phase:'releasing',resources_released:false,server_time:101}},now,{allowSameOperation:true});
    """)
    assert ctx.eval("session.stopping")
    assert model(ctx)["phase"] == "releasing"
    ctx.eval("""
      session.active=false;session.rollout_progress.phase='done';session.rollout_progress.resources_released=true;
      applySessionReceipt({active:true,mode:'rollout',meta:{operation_id:'current-run'},
        rollout_progress:{phase:'running',resources_released:false,server_time:100}},now,{allowSameOperation:true});
    """)
    assert model(ctx)["phase"] == "done" and model(ctx)["released"]


def test_reaching_duration_never_promotes_clock_to_done_or_release(progress_js):
    progress_js.eval("session.rollout_progress.policy_elapsed_s=19.5; now=3000;")
    result = model(progress_js)
    assert result["elapsed"] == 20 and result["remaining"] == 0
    assert result["phase"] == "running" and not result["finished"] and not result["released"]


@pytest.mark.parametrize("phase", ["returning_home", "releasing", "saving_frames", "encoding_videos", "uploading", "done", "failed", "stopped"])
def test_cleanup_saving_and_terminal_states_freeze_policy_clock(progress_js, phase):
    progress_js.eval(f"session.rollout_progress.phase={json.dumps(phase)}; now=3000;")
    assert model(progress_js)["elapsed"] == 2  # Even contradictory timer_running cannot extend non-policy time.


def test_stop_request_freezes_clock_before_cleanup_marker(progress_js):
    progress_js.eval("session.stopping=true; now=3000;")
    result = model(progress_js)
    assert result["elapsed"] == 2
    assert result["label"].startswith("Stopping")


def test_saving_progress_counts_real_stage_units_not_an_invented_overall_percentage(progress_js):
    progress_js.eval("Object.assign(session.rollout_progress,{phase:'saving_frames',completed:150,total:450,unit:'frames',resources_released:true});")
    result = model(progress_js)
    assert result["barValue"] == 150 and result["barMax"] == 450
    assert result["stageCount"] == "150 / 450 frames"
    assert result["stage"] == 3 and result["released"]


@pytest.mark.parametrize("phase", ["encoding_videos", "rendering", "packaging", "uploading"])
def test_unknown_saving_totals_are_indeterminate(progress_js, phase):
    progress_js.eval(f"session.rollout_progress.phase={json.dumps(phase)};")
    result = model(progress_js)
    assert result["barValue"] is None and result["stageCount"] == ""


def test_unconfirmed_release_cannot_be_claimed_from_saving_phase(progress_js):
    progress_js.eval("session.rollout_progress.phase='uploading'; session.rollout_progress.resources_released=false;")
    result = model(progress_js)
    assert "Arms released" not in result["label"] and not result["released"]


def test_legacy_phase_has_label_but_no_unreliable_session_elapsed_clock(progress_js):
    progress_js.eval("session={active:true,mode:'rollout',elapsed_s:100,phase_elapsed_s:100,meta:{duration:20},parsed:{rollout_phase:'running'}};")
    result = model(progress_js)
    assert result["phase"] == "running" and result["elapsed"] is None
    assert result["barValue"] is None
    assert result["clock"] == "Policy time unavailable"


@pytest.mark.parametrize("error, label", [
    ("upload_failed", "Rollout finished — HF upload failed"),
    ("recording_export_incomplete", "Rollout finished — recording saving incomplete"),
])
def test_postprocessing_failure_is_not_a_failed_physical_rollout(progress_js, error, label):
    progress_js.eval("Object.assign(session.rollout_progress," + json.dumps({
        "phase": "failed", "outcome": "completed", "postprocess_error": error,
        "resources_released": True,
    }) + ");")
    result = model(progress_js)
    assert result["label"] == label
    assert result["released"] and result["finished"]


def test_completed_phase_age_does_not_extrapolate(progress_js):
    progress_js.eval("session.rollout_progress.phase='done'; now=3000;")
    assert model(progress_js)["phaseAge"] == 2


def test_non_rollout_has_no_rollout_progress(progress_js):
    progress_js.eval("session={active:true,mode:'record',parsed:{phase:'recording'}};")
    assert model(progress_js) is None


def test_display_tick_does_not_open_cameras_fetch_or_change_execution():
    source = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()
    tick = source[source.index("function updateRolloutClocks"):source.index("function armPanelHTML")]
    for forbidden in ("api(", "post(", "fetch(", ".src", "refreshCameras", "syncCams"):
        assert forbidden not in tick
    assert 'setInterval(updateRolloutClocks, 100)' in source
