"""Read-only browser transport and preview-lifetime checks, with no real endpoints."""

from pathlib import Path

import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "ui" / "app.js").read_text()


def drain(ctx):
    for _ in range(100):
        if not ctx.execute_pending_job():
            return
    pytest.fail("Polling promise queue did not settle")


@pytest.fixture
def transport_js():
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval("""
      const pollControllers=new Set(),timers={},requests=[];
      let sequence=0,result=null,failure=null;
      function setTimeout(callback,delay){timers[++sequence]={callback,delay};return sequence;}
      function clearTimeout(id){delete timers[id];}
      class AbortController{
        constructor(){this.signal={aborted:false,onAbort:null};}
        abort(){this.signal.aborted=true;this.signal.onAbort?.();}
      }
      function api(path,options){return new Promise((resolve,reject)=>{
        options.signal.onAbort=()=>reject({name:'AbortError'});
        requests.push({path,options,resolve,reject});
      });}
    """)
    ctx.eval(SOURCE[SOURCE.index("async function pollRead("):SOURCE.index("async function refreshOverview()")])
    return ctx


def test_read_only_transport_has_five_second_abort_without_retry(transport_js):
    ctx = transport_js
    ctx.eval("pollRead('/session').then(value=>result=value).catch(error=>failure=error.name);")
    assert ctx.eval("requests.length") == 1
    assert ctx.eval("requests[0].options.cache") == "no-store"
    assert ctx.eval("Object.values(timers)[0].delay") == 5000
    ctx.eval("Object.values(timers)[0].callback();")
    drain(ctx)
    assert ctx.eval("failure") == "AbortError"
    assert ctx.eval("pollControllers.size") == 0
    assert ctx.eval("Object.keys(timers).length") == 0
    assert ctx.eval("requests.length") == 1  # No retry inside this request.


def test_successful_read_releases_its_deadline_and_controller(transport_js):
    ctx = transport_js
    ctx.eval("pollRead('/session').then(value=>result=value);requests[0].resolve({active:false});")
    drain(ctx)
    assert not ctx.eval("result.active")
    assert ctx.eval("Object.keys(timers).length") == 0
    assert ctx.eval("pollControllers.size") == 0


@pytest.fixture
def camera_js():
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval("""
      let now=1000,stopRequestsPending=0;
      Date.now=()=>now;
      const image={src:'/api/cameras/top/stream',dataset:{connectedAt:'1000'},onerror:()=>{},
        getAttribute(name){return this[name]||null;},removeAttribute(name){delete this[name];}};
      const label={textContent:''};const tile={dataset:{cam:'top'},querySelectorAll:()=>[image]};
      const slot={querySelectorAll:selector=>selector==='.cam'?[tile]:[image],innerHTML:'original'};
      const document={visibilityState:'visible',querySelectorAll:()=>[image]};
      const overview={cameras:[{name:'top',preview_source:'direct',preview_generation:0,preview_state:'live'}]};
      function $(selector){return selector==='#cams-slot'?slot:selector==='img'?image:label;}
    """)
    ctx.eval(SOURCE[SOURCE.index("let camsRendered = null;"):SOURCE.index("function camsHTML()")])
    ctx.eval("camsRendered=cameraKey();")
    return ctx


def test_hidden_preview_stops_mjpeg_and_only_resumes_existing_intent(camera_js):
    ctx = camera_js
    ctx.eval("document.visibilityState='hidden';syncCams();")
    assert ctx.eval("image.src===undefined")
    assert ctx.eval("image.dataset.pausedSrc") == "/api/cameras/top/stream"
    ctx.eval("now=10000;syncCams();")
    assert ctx.eval("image.src===undefined")
    ctx.eval("document.visibilityState='visible';syncCams();")
    assert ctx.eval("image.src") == "/api/cameras/top/stream"
    assert ctx.eval("image.dataset.pausedSrc===undefined")
    assert ctx.eval("image.dataset.connectedAt") == "10000"
    assert ctx.eval("slot.innerHTML") == "original"


def test_page_cleanup_clears_paused_stream_intent(camera_js):
    ctx = camera_js
    ctx.eval("pauseCameraStreams();releaseCameraStreams();")
    assert ctx.eval("image.src===undefined")
    assert ctx.eval("image.dataset.pausedSrc===undefined")


def test_stop_pending_does_not_reopen_paused_previews(camera_js):
    ctx = camera_js
    ctx.eval("stopRequestsPending=1;syncCams();now=2000;syncCams();")
    assert ctx.eval("image.src===undefined")
    ctx.eval("stopRequestsPending=0;syncCams();")
    assert ctx.eval("image.src") == "/api/cameras/top/stream"


def test_hidden_page_markup_defers_stream_src_until_visibility_returns():
    markup = SOURCE[SOURCE.index("function camsHTML()"):SOURCE.index("const ROLLOUT_SAVE_PHASES")]
    assert 'document.visibilityState === "hidden" ? "data-paused-src" : "src"' in markup
    assert "releaseCameraStreams();" in SOURCE[SOURCE.index("function cleanupPage()"):]


def test_status_wording_distinguishes_delay_error_and_completed_work():
    renderer = SOURCE[SOURCE.index("function renderRolloutProgress("):SOURCE.index("function currentRolloutProgress()")]
    assert "Status updates delayed" in renderer and "Status request failed" in renderer
    assert "Status connection lost" not in renderer
    assert "showing the last confirmed" in renderer and "model.finished" in renderer
    assert "physical execution may continue" in renderer


@pytest.mark.parametrize("active_after_stop", [True, False])
def test_stop_cancels_only_background_gets_and_restores_receipt_controls_before_hanging_get(active_after_stop):
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval("""
      let stopRequestsPending=0,paused=0,aborted=0,refreshed=0,posts=0,receipts=0,resolvePost;
      const performance={now:()=>0};
      const pollControllers=new Set([{abort(){aborted++;}}]);
      function pauseCameraStreams(){paused++;} function syncCams(){}
      let session={active:true};
      function refreshSession(){refreshed++;return new Promise(()=>{});}
      function post(){posts++;return new Promise(resolve=>resolvePost=resolve);}
      function applySessionReceipt(result){receipts++;session=result;sessionUpdated();} function alert(){}
      const button={disabled:false};
      function sessionUpdated(){button.disabled=!session.active;}
    """)
    ctx.eval(SOURCE[SOURCE.index("async function doPost("):SOURCE.index("// --------------------------------------------------------------------------------- pages")])
    ctx.eval("doPost('/session/stop',{},button);")
    assert ctx.eval("paused") == ctx.eval("aborted") == ctx.eval("posts") == 1
    assert ctx.eval("stopRequestsPending") == 1
    assert ctx.eval("refreshed") == 0
    ctx.eval(f"resolvePost({{active:{str(active_after_stop).lower()}}});")
    drain(ctx)
    assert ctx.eval("stopRequestsPending") == 0
    assert ctx.eval("refreshed") == ctx.eval("receipts") == ctx.eval("posts") == 1
    assert ctx.eval("button.disabled") is not active_after_stop
