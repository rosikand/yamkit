/* yamkit ui — vanilla JS single-page app. Talks only to /api/*; every hardware action is a
   POST that spawns the corresponding `yamkit` CLI on the host. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtBytes = (b) => b == null ? "–" : b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " kB";
const fmtDate = (t) => t == null ? "–" : new Date(t * 1000).toLocaleString();
const fmtDur = (s) => s == null ? "–" : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`;
const fmtClock = (s) => {
  if (s == null || !Number.isFinite(Number(s))) return "–";
  const tenths = Math.max(0, Math.round(Number(s) * 10));
  return `${String(Math.floor(tenths / 600)).padStart(2, "0")}:${String(Math.floor(tenths / 10) % 60).padStart(2, "0")}.${tenths % 10}`;
};

// series colors follow the active theme (validated pair per mode — see style.css)
function seriesColors() {
  const cs = getComputedStyle(document.documentElement);
  return {
    state: cs.getPropertyValue("--series-state").trim() || "#3987e5",
    action: cs.getPropertyValue("--series-action").trim() || "#d95926",
    grid: cs.getPropertyValue("--grid-line").trim() || "rgba(255,255,255,.07)",
    cursor: cs.getPropertyValue("--cursor-line").trim() || "rgba(255,255,255,.35)",
  };
}

async function api(path, opts) {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

// ------------------------------------------------------------------------------- theming ----
function applyThemePref(pref, { persist = true } = {}) {
  if (persist) localStorage.setItem("yamkit-theme", pref);
  document.documentElement.dataset.themePref = pref;
  document.documentElement.dataset.theme = pref === "system"
    ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
    : pref;
  syncThemeButtons();
  route(); // re-render so canvases and legends pick up the new palette
}
function syncThemeButtons() {
  const pref = document.documentElement.dataset.themePref || "system";
  document.querySelectorAll("#theme-switch button").forEach((b) =>
    b.classList.toggle("active", b.dataset.themePref === pref));
}
matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
  if ((document.documentElement.dataset.themePref || "system") === "system") applyThemePref("system", { persist: false });
});

// ---------------------------------------------------------------- global state + pollers ----
let overview = null;
let session = { active: false, mode: null, parsed: {}, log: [] };

async function refreshOverview() {
  try { overview = await api("/overview"); } catch { overview = null; }
  updateSidebar();
  document.dispatchEvent(new CustomEvent("overview"));
}
let lastSessionKey = "";
let sessionRequest = 0;
let sessionApplied = 0;
let sessionReceivedAt = null;
let sessionDisconnected = false;
let sessionFrozenDelta = null;
async function refreshSession() {
  const request = ++sessionRequest;
  const requestedAt = performance.now();
  try {
    const next = await api("/session");
    if (request < sessionApplied) return; // an older poll cannot restore a previous run/phase
    sessionApplied = request;
    session = next;
    sessionReceivedAt = performance.now();
    // A delayed reply may contain an old snapshot. Await a prompt poll before
    // advancing its display clock; do not compare browser and host wall clocks.
    sessionDisconnected = sessionReceivedAt - requestedAt >= 3000;
    sessionFrozenDelta = sessionDisconnected ? 0 : null;
  } catch {
    if (request < sessionApplied) return;
    // Stop the display clock on a failed poll, without inventing a robot phase change.
    if (!sessionDisconnected) sessionFrozenDelta = sessionReceivedAt == null ? 0 : Math.min(3, Math.max(0, (performance.now() - sessionReceivedAt) / 1000));
    sessionDisconnected = true;
  }
  updateSidebar();
  document.dispatchEvent(new CustomEvent("session"));
  // a session starting, ending or handing the cameras back changes what the tiles should show: refresh now
  const key = `${session.active}|${session.mode}|${session.parsed?.phase || ""}|${session.preview_generation || 0}`;
  if (key !== lastSessionKey) { lastSessionKey = key; refreshOverview(); }
}
function updateSidebar() {
  const dot = $("#side-dot");
  dot.className = "dot " + (session.active ? "run" : "");
  $("#side-mode").textContent = (session.active ? session.mode : "idle") + (session.stopping && session.active ? " (stopping…)" : "");
  const hz = session.parsed && session.parsed.rate_hz;
  $("#side-hz").textContent = session.active && !session.stopping && (!session.parsed?.operator_phase || session.parsed.operator_phase === "ready") && hz ? hz.toFixed(0) + " Hz" : "";
}

// -------------------------------------------------------------------------- shared views ----
const errBanner = (msg) => `<div class="error-banner">${esc(msg)}</div>`;
const st = (ok, label, warn = false) =>
  `<span class="badge ${ok ? "ok" : warn ? "warn" : "err"}"><span class="dot"></span>${esc(label)}</span>`;
const stN = (label, run = false) =>
  `<span class="badge${run ? " run" : ""}">${run ? '<span class="dot"></span>' : ""}${esc(label)}</span>`;

function pageHead(title, sub = "", toolbar = "") {
  return `<div class="page-head"><h1>${esc(title)}</h1>${sub ? `<span class="sub">${sub}</span>` : ""}
    ${toolbar ? `<div class="toolbar">${toolbar}</div>` : ""}</div>`;
}

const PREFERRED_CAMS = ["top", "left_wrist", "right_wrist"];
function cameraNames() {
  const configured = (overview?.cameras || []).map((c) => c.name);
  if (!configured.length) return PREFERRED_CAMS.map((n) => ({ name: n, configured: false }));
  const ordered = [...configured].sort((a, b) => {
    const ia = PREFERRED_CAMS.findIndex((p) => a.includes(p.split("_")[0]) || a === p);
    const ib = PREFERRED_CAMS.findIndex((p) => b.includes(p.split("_")[0]) || b === p);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return ordered.slice(0, 3).map((n) => ({ name: n, configured: true }));
}

let camsRendered = null;
const cameraKey = () => (overview?.cameras || []).map((c) =>
  `${c.name}:${c.preview_source || "direct"}:${c.preview_generation || 0}`).join("|");

// Poll status only; pixels remain one MJPEG connection per visible tile.
async function refreshCameras() {
  if (!$("#cams-slot")) return;
  try {
    const cameras = await api("/cameras");
    if (overview) overview.cameras = cameras;
    syncCams();
  } catch {
    document.querySelectorAll(".cam .preview-state").forEach((el) => {
      el.textContent = "preview unavailable";
    });
  }
}
function cameraStreamError(img) {
  img.dataset.retryAt = String(Date.now() + 2000);
  const state = $(".preview-state", img.parentElement);
  if (state) state.textContent = "preview unavailable · reconnecting";
}
function cameraStateText(c) {
  const state = c.preview_state || (c.error ? "unavailable" : c.streaming ? "live" : "waiting");
  const age = c.frame_age_s;
  if (state === "stale") return `stale · last frame ${age == null ? "age unknown" : age.toFixed(1) + " s ago"}`;
  if (state === "unavailable") return "preview unavailable · reconnecting";
  if (state === "waiting" || state === "idle") return "waiting for frames";
  return c.preview_source === "session" ? "live · session camera" : "live";
}
function releaseCameraStreams(root = document) {
  // Removing an <img> from the DOM does not cancel its MJPEG request in Chrome.
  // Close it explicitly so abandoned previews cannot occupy every HTTP connection
  // and prevent a queued Stop request from reaching the server.
  root.querySelectorAll(".cam img").forEach((img) => {
    img.onerror = null;
    img.removeAttribute("src");
  });
}
function syncCams() {
  const slot = $("#cams-slot");
  if (!slot) return;
  if (cameraKey() !== camsRendered) {
    releaseCameraStreams(slot);
    slot.innerHTML = camsHTML();
  }
  slot.querySelectorAll(".cam").forEach((tile) => {
    const c = (overview?.cameras || []).find((item) => item.name === tile.dataset.cam);
    if (!c) return;
    const img = $("img", tile);
    const label = $(".preview-state", tile);
    if (label) label.textContent = cameraStateText(c);
    if (!img) return;
    // Native MJPEG images can report complete=true while still streaming. Use source status
    // and errors for quick retries; a slow renewal also recovers a silently ended response.
    const now = Date.now();
    const retry = +img.dataset.retryAt || 0;
    const elapsed = now - (+img.dataset.connectedAt || 0);
    const needsFrames = c.preview_state && c.preview_state !== "live";
    if ((retry && now >= retry) || (!retry && needsFrames && elapsed > 2000) || elapsed > 30000) {
      delete img.dataset.retryAt;
      img.dataset.connectedAt = String(now);
      img.src = `/api/cameras/${encodeURIComponent(c.name)}/stream?generation=${c.preview_generation || 0}&retry=${now}`;
    }
  });
}
function camsHTML() {
  camsRendered = cameraKey();
  return `<div class="cams">` + cameraNames().map((c) => `
    <div class="cam" data-cam="${esc(c.name)}">
      <span class="label">${esc(c.name)}</span>
      <span class="rollout-timer" hidden aria-label="Rollout policy time"></span>
      ${c.configured
        ? `<img src="/api/cameras/${encodeURIComponent(c.name)}/stream" alt="${esc(c.name)}"
             data-connected-at="${Date.now()}" onerror="cameraStreamError(this)" />
           <span class="preview-state">waiting for frames</span>`
        : `<div class="placeholder">no camera configured in rig.yaml</div>`}
    </div>`).join("") + `</div>`;
}

// The server owns phase transitions and the policy clock. These display-only clocks
// never start hardware or infer completion from reaching the requested duration.
const ROLLOUT_SAVE_PHASES = ["saving_frames", "encoding_videos", "rendering", "finalizing", "packaging", "uploading"];
const rolloutProgressPending = (progress) => !!progress?.phase && !["done", "failed", "stopped"].includes(progress.phase);
function rolloutProgressView(state, delta = 0, stale = false) {
  const legacyPhase = state.parsed?.rollout_phase;
  const p = state.rollout_progress || (state.mode === "rollout" && (state.active || state.meta?.operation_id) ? {
    phase: !state.active ? state.stop_requested ? "stopped" : state.returncode === 0 ? "done" : "failed"
      : legacyPhase === "released" ? "finalizing" : legacyPhase || "preparing",
    resources_released: legacyPhase === "released", target_duration_s: state.meta?.duration,
    capture_requested: state.meta?.capture_trace,
    // Legacy snapshots do not have a trustworthy policy-only clock.
    policy_elapsed_s: null, policy_timer_running: false,
  } : null);
  if (!p) return null;
  const phase = p.phase || "preparing", moving = ["homing", "running", "returning_home", "releasing"].includes(phase);
  const labels = {preparing: "Preparing cameras and arms…", homing: "Preparing — moving to the start pose",
    running: "Policy running — 30 Hz", returning_home: "Returning home — keep clear; Stop releases the arms",
    releasing: "Releasing arms…", saving_frames: "Arms released — saving recording frames",
    encoding_videos: "Arms released — encoding camera videos", rendering: "Arms released — rendering recording details",
    finalizing: "Arms released — finalizing local files", packaging: "Arms released — preparing HF upload",
    uploading: "Arms released — uploading to Hugging Face", done: "Rollout finished",
    stopped: "Rollout stopped", failed: "Rollout failed"};
  const target = Number.isFinite(p.target_duration_s) && p.target_duration_s > 0 ? p.target_duration_s : null;
  let elapsed = Number.isFinite(p.policy_elapsed_s) ? Math.max(0, p.policy_elapsed_s) : null;
  const running = phase === "running" && p.policy_timer_running === true && !state.stopping;
  if (elapsed != null && running) elapsed += Math.max(0, Math.min(3, delta));
  if (elapsed != null && target != null) elapsed = Math.min(target, elapsed);
  const remaining = elapsed != null && target != null ? Math.max(0, target - elapsed) : null;
  const counted = Number.isFinite(p.completed) && Number.isFinite(p.total) && p.total > 0;
  const stageCount = counted ? `${Math.min(p.total, Math.max(0, p.completed)).toLocaleString()} / ${p.total.toLocaleString()} ${p.unit || "items"}${p.camera ? ` · ${p.camera}` : ""}` : "";
  const isSaving = ROLLOUT_SAVE_PHASES.includes(phase);
  const finished = ["done", "failed", "stopped"].includes(phase);
  let label = labels[phase] || "Waiting for operation status…";
  if (p.postprocess_error) {
    const outcome = p.outcome === "completed" ? "Rollout finished" : p.outcome === "stopped" ? "Rollout stopped" : "Rollout failed";
    label = `${outcome} — ${p.postprocess_error === "upload_failed" ? "HF upload failed" : "recording saving incomplete"}`;
  }
  // Never claim released motors solely because a saving-like phase was reported.
  if (!p.resources_released && (isSaving || phase === "done")) label = label.replace("Arms released — ", "");
  if (state.stopping && !p.resources_released) label = "Stopping — releasing arms and finishing cleanup…";
  const clock = elapsed == null ? ["preparing", "homing"].includes(phase) ? "Policy timer not started" : "Policy time unavailable"
    : `${fmtClock(elapsed)}${target ? ` / ${fmtClock(target)}` : ""}`;
  const phaseAge = Number.isFinite(p.phase_elapsed_s) ? p.phase_elapsed_s + (!finished ? Math.max(0, Math.min(3, delta)) : 0) : null;
  const stage = ["preparing", "homing"].includes(phase) ? 0 : phase === "running" ? 1 : moving ? 2 : isSaving ? 3 : phase === "done" ? 4 : -1;
  return {phase, label, clock, elapsed, target, remaining, stage, stageCount, phaseAge, stale, finished,
    released: p.resources_released === true, capture: p.capture_requested, isSaving, postprocessError: p.postprocess_error,
    barValue: counted ? Math.max(0, Math.min(p.total, p.completed)) : phase === "running" && elapsed != null && target ? elapsed : phase === "done" ? 1 : null,
    barMax: counted ? p.total : phase === "running" && target ? target : 1};
}

function rolloutProgressHTML() {
  return `<div class="rollout-progress" hidden>
    <div class="rollout-progress-phase" data-rollout-phase role="status" aria-live="polite"></div>
    <ol class="rollout-stages" aria-label="Rollout stages">${["Prepare", "Run", "Release", "Save", "Done"].map((stage, i) => `<li data-rollout-stage="${i}">${stage}</li>`).join("")}</ol>
    <div class="rollout-clock-row"><span class="rollout-clock mono" data-rollout-clock></span><span class="hint" data-rollout-remaining></span></div>
    <progress data-rollout-bar aria-label="Current rollout stage progress"></progress>
    <div class="rollout-progress-detail"><span data-rollout-stage-count></span><span data-rollout-stage-time></span></div>
    <div class="hint" data-rollout-note></div><div class="hint warn" role="status" data-rollout-stale hidden></div>
  </div>`;
}

function renderRolloutProgress(root, model) {
  if (!root) return;
  root.hidden = !model;
  if (!model) return;
  const set = (selector, text) => { const node = $(selector, root); if (node && node.textContent !== text) node.textContent = text; };
  set("[data-rollout-phase]", model.label);
  set("[data-rollout-clock]", model.clock);
  set("[data-rollout-remaining]", model.phase === "running" && model.remaining != null
    ? model.remaining > 0 ? `${fmtClock(model.remaining)} remaining` : "Duration reached · waiting for cleanup" : "Policy time · excludes preparation and cleanup");
  set("[data-rollout-stage-count]", model.stageCount || (model.finished ? model.label : model.isSaving ? "Working… progress reported when available" : model.label));
  set("[data-rollout-stage-time]", model.phaseAge == null ? "" : `${fmtClock(model.phaseAge)} in this stage`);
  set("[data-rollout-note]", model.finished ? `${model.released ? "Arms released. " : ""}${model.postprocessError ? "Local originals are retained; check the run log before retrying saving or upload. " : ""}Run completion does not prove task success.`
    : model.released ? "Motors released. Saving and upload keep local originals; leave the UI service running."
    : "Stay at the arms. Stop and physical power cutoff must remain accessible.");
  const stale = $("[data-rollout-stale]", root);
  stale.hidden = !model.stale;
  set("[data-rollout-stale]", model.released ? "Status connection lost — showing the last confirmed status; arms were released. Check Lenovo for saving or upload progress."
    : "Status connection lost — timer paused; physical execution may continue. Check Lenovo. Closing the browser is not Stop.");
  const bar = $("[data-rollout-bar]", root);
  bar.max = model.barMax;
  if (model.barValue == null) bar.removeAttribute("value"); else bar.value = model.barValue;
  bar.classList.toggle("paused", model.stale || ["failed", "stopped"].includes(model.phase));
  bar.setAttribute("aria-label", `${model.label}${model.stageCount ? `: ${model.stageCount}` : ""}`);
  root.querySelectorAll("[data-rollout-stage]").forEach((node) => {
    const i = Number(node.dataset.rolloutStage);
    node.classList.toggle("current", i === model.stage);
    node.classList.toggle("complete", model.stage >= 0 && i < model.stage);
    if (i === model.stage) node.setAttribute("aria-current", "step"); else node.removeAttribute("aria-current");
    node.hidden = i === 3 && !model.capture && !model.isSaving;
  });
}

function currentRolloutProgress() {
  const age = sessionReceivedAt == null ? 0 : Math.max(0, (performance.now() - sessionReceivedAt) / 1000);
  return rolloutProgressView(session, sessionFrozenDelta ?? Math.min(3, age), sessionDisconnected || age >= 3);
}

function updateRolloutClocks() {
  const model = currentRolloutProgress();
  renderRolloutProgress($("#inf-progress .rollout-progress"), model);
  const view = pages.inference?._detailView;
  if (view?.alive && view.body?.isConnected && view.progressOperationId && view.progressOperationId === session.rollout_progress?.operation_id)
    renderRolloutProgress($("#run-live-progress .rollout-progress", view.body), model);
  document.querySelectorAll(".cam .rollout-timer").forEach((node) => {
    node.hidden = !model;
    if (model) node.textContent = `${model.stale ? "Status paused" : "Rollout"} · ${model.phase === "running" ? "policy" : model.label.replace("Arms released — ", "")}${model.elapsed == null ? "" : ` · ${model.clock}`}`;
  });
}

function armPanelHTML(armName, stt, role) {
  const isLeader = role === "leader";
  const range = Math.PI; // display range ±π rad
  const rows = (stt?.q || []).map((v, i) => {
    const frac = Math.max(-1, Math.min(1, v / range));
    const left = frac < 0 ? 50 + frac * 50 : 50, width = Math.abs(frac) * 50;
    return `<div class="joint"><span class="name">joint_${i + 1}</span>
      <span class="track"><span class="mid"></span><span class="fill" style="left:${left}%;width:${Math.max(width, 0.7)}%"></span></span>
      <span class="val">${v.toFixed(3)}</span></div>`;
  }).join("");
  const grip = stt?.gripper;
  const gripRow = `<div class="joint"><span class="name">${isLeader ? "trigger" : "gripper"}</span>
      <span class="track"><span class="fill" style="left:0;width:${grip != null ? grip * 100 : 0}%"></span></span>
      <span class="val">${grip != null ? grip.toFixed(2) : "–"}</span></div>`;
  // teaching-handle buttons (leaders only; parsed as a string of 0/1)
  const btnRow = stt?.buttons
    ? `<div class="joint"><span class="name">buttons</span>
        <span class="btns">${[...stt.buttons].map((b, i) =>
          `<span class="btn-ind${b === "1" ? " on" : ""}">${i === 0 ? "engage" : "btn " + (i + 1)}</span>`).join("")}</span>
        <span></span></div>`
    : "";
  return `<div class="panel arm-panel"><div class="arm-name">${esc(armName)}
      ${role ? `<span class="crumb role-tag">${esc(role)}</span>` : ""}</div>
    ${stt ? rows + gripRow + btnRow : `<div class="empty">no state — start the state stream or teleop</div>`}</div>`;
}

// "· 12 s / 30 s" for the current recording phase (server-timed, so it survives page reloads).
// Phase changes come from the recorder's actual logs; elapsed time alone does not imply saving.
function phaseClock(total) {
  const t = session.phase_elapsed_s;
  if (t == null) return "";
  return ` · ${Math.min(Math.round(t), total ? Math.round(total) : Infinity)} s${total ? ` / ${Math.round(total)} s` : ""}`;
}

// show Start buttons only when idle, Stop only while a session runs
function syncRunButtons(startIds, stopId) {
  const stop = $(stopId);
  if (!stop) return;
  for (const id of startIds) {
    const b = $(id);
    if (b) b.hidden = session.active;
  }
  stop.hidden = !session.active;
  stop.disabled = false;  // a second click during the return-home move releases the arms immediately
  if (session.stopping && session.active) {
    stop.textContent = session.mode === "record" ? "Stopping recording… click again to interrupt" : "Stopping… arms returning home — click again to release now";
  } else {
    stop.textContent = session.active && session.mode === "record" && session.parsed?.phase === "saving" ? "Stop (interrupt save)" : "Stop";
  }
}

const logPaneHTML = (id = "log", tall = false) => `<pre class="log${tall ? " tall" : ""}" id="${id}"></pre>`;
function fillLog(id = "log") {
  const el = document.getElementById(id);
  if (!el) return;
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
  el.textContent = (session.log || []).slice(-200).join("\n") || "(no output yet)";
  if (atBottom) el.scrollTop = el.scrollHeight;
}

async function doPost(path, body, btn) {
  if (btn) btn.disabled = true;
  try { await post(path, body); }
  catch (e) { alert(e.message); }
  finally { if (btn) btn.disabled = false; await refreshSession(); }
}

// --------------------------------------------------------------------------------- pages ----
const pages = {};

// ---- live ----
pages.live = {
  render(el) {
    el.innerHTML = `
      ${pageHead("Live", "read-only — opening this page never energises a motor", `
        <button id="btn-read" class="primary">Start state stream</button>
        <button id="btn-stop" class="danger">Stop</button>
        <button id="btn-refresh">Refresh</button>`)}
      <div class="hint">The state stream runs <code>yamkit read</code>: arms connect in gravity-compensation
        mode (motors energised but compliant — nothing moves).</div>
      <div class="sect"><div class="sect-head">Cameras</div><div id="cams-slot">${camsHTML()}</div></div>
      <div class="sect"><div class="sect-head">Arm state</div><div class="cols cols-2" id="arm-panels"></div></div>
      <div class="sect"><div class="sect-head">Status</div><div class="st-list" id="status-list"></div>
        <div class="hint" id="bringup"></div></div>
      <div class="sect"><div class="sect-head">Session output</div>${logPaneHTML()}</div>`;
    $("#btn-read").onclick = (e) => doPost("/session/read", { hz: 5 }, e.target);
    $("#btn-stop").onclick = (e) => doPost("/session/stop", {}, e.target);
    $("#btn-refresh").onclick = () => { refreshOverview(); refreshSession(); };
    this.update();
  },
  update() {
    const panels = $("#arm-panels");
    if (!panels) return;
    syncRunButtons(["#btn-read"], "#btn-stop");
    syncCams();
    const rigArms = Object.entries(overview?.rig?.arms || {});
    const byRole = (role) => rigArms.filter(([, a]) => a.role === role).map(([n, a]) => [n, a.role]);
    const arms = rigArms.length
      ? [...byRole("follower"), ...byRole("leader")]
      : Object.keys(session.parsed?.arms || {}).map((n) => [n, null]);
    panels.innerHTML = arms.map(([n, role]) => armPanelHTML(n, session.parsed?.arms?.[n], role)).join("") ||
      `<div class="empty">no rig file — run <code>yamkit discover --write</code></div>`;
    const list = $("#status-list");
    if (list) {
      const rows = [];
      rows.push(overview?.rig?.found
        ? st(!(overview.rig.problems || []).length, `rig: ${Object.keys(overview.rig.arms).length} arms, ${overview.rig.pairs.length} pairs`)
        : st(false, "rig file missing"));
      rows.push(stN(session.active ? `mode: ${session.mode}` : "mode: idle", session.active));
      const can = (overview?.can || []).map((i) => st(i.up, `${i.name} ${i.up ? "UP" : "DOWN"}${i.in_rig ? "" : " (not in rig)"}`));
      rows.push(...(can.length ? can : [st(false, "no CAN adapters", true)]));
      const cams = (overview?.cameras || []).map((c) => st(c.streaming && !c.error, `cam ${c.name}${c.error ? ": " + c.error : ""}`, !c.error));
      rows.push(...(cams.length ? cams : [st(false, "no cameras in rig", true)]));
      list.innerHTML = rows.join("");
      $("#bringup").innerHTML = (overview?.can_bringup || []).length
        ? "bring interfaces up: <code>" + overview.can_bringup.map(esc).join("</code> · <code>") + "</code>" : "";
    }
    fillLog();
  },
};

// ---- record ----
const DEFAULT_FPS = 30;
pages.record = {
  render(el) {
    const camFps = (overview?.cameras || []).map((c) => +c.fps).filter((f) => f > 0);
    const maxFps = camFps.length ? Math.min(...camFps) : DEFAULT_FPS;
    const hubCfg = overview?.hub || {};
    const hubReady = !!hubCfg.logged_in;
    const hubUser = hubCfg.username || "you";
    const hubPrivate = hubCfg.private !== false;
    const hubDefault = hubReady ? (hubCfg.datasets || "local") : "local";  // uploading is opt-in
    el.innerHTML = `
      ${pageHead("Record", "teleoperation and dataset recording", `<button id="btn-stop-top" class="danger">Stop</button>`)}
      <div class="sect"><div class="sect-head">Cameras</div><div id="cams-slot">${camsHTML()}</div></div>
      <div class="cols cols-2">
        <div class="sect"><div class="sect-head">Teleop</div><div class="panel pad">
          <div id="teleop-ready"></div>
          <div id="teleop-status"></div>
          <div class="toolbar" style="margin-top:12px">
            <button id="btn-teleop" class="primary">Start Teleop</button>
          </div>
          <div class="hint">Start connects the arms, moves them home and synchronizes the followers.
            Wait for “Teleop ready”, then move the leaders. No handle-button press is needed.
            Stop returns the arms home and releases them. Let go of the handles during homing.</div>
        </div></div>
        <div class="sect"><div class="sect-head">Recording</div><div class="panel pad">
          <label class="field">dataset name<input type="text" id="rec-name" placeholder="pick_cube" /></label>
          <label class="field">task instruction<input type="text" id="rec-task" placeholder="put the red cube into the black container" /></label>
          <div class="form-grid">
            <label class="field">episodes<input type="number" id="rec-episodes" value="10" min="1" /></label>
            <label class="field">episode duration (s)<input type="number" id="rec-episode-s" value="30" /></label>
            <label class="field">reset duration (s)<input type="number" id="rec-reset-s" value="10" min="0" /></label>
          </div>
          <div class="field" style="margin-top:12px">save to
            <div class="checks">
              <label class="check"><input type="checkbox" id="rec-local" ${hubDefault !== "hub" ? "checked" : ""} /> this computer (data/datasets)</label>
              <label class="check"><input type="checkbox" id="rec-hub" data-unavailable="${!hubReady}" ${hubDefault !== "local" ? "checked" : ""} ${hubReady ? "" : "disabled"} />
                also upload to Hugging Face Hub${hubReady ? ` (as ${esc(hubUser)}/…, ${hubPrivate ? "private" : "public"})` : " — sign in on the Settings page first"}</label>
            </div>
            <div class="hint">Recording is identical either way; the upload only starts after the session has ended and the arms are parked.</div>
          </div>
          <details class="advanced">
            <summary>Advanced</summary>
            <label class="field">recording rate (frames per second)<input type="number" id="rec-fps" value="${DEFAULT_FPS}" min="10" max="${maxFps}" step="5" /></label>
            <div class="hint">${DEFAULT_FPS} is the standard for YAM datasets and what the pretrained policies expect. Valid: 10 to ${maxFps}
              (your slowest camera). Lower values make smaller datasets but choppier policies. Leave it unless you know why.</div>
          </details>
          <div class="toolbar" style="margin-top:12px">
            <button id="btn-record" class="primary">Start Recording</button>
          </div>
          <div class="hint">Start Recording prepares the arms and cameras, then followers track the leaders automatically.
            The episode clock and recorded frames start when the arms are ready.
            Stop ends acquisition, finishes saving and returns the arms home.</div>
        </div></div>
      </div>
      <div class="sect"><div class="sect-head">Progress</div><div class="st-list" id="rec-progress"></div></div>
      <div class="sect"><div class="sect-head">Output</div>${logPaneHTML()}</div>`;
    $("#btn-teleop").onclick = () => this.start("teleop", {});
    $("#btn-record").onclick = () => {
      const name = $("#rec-name").value.trim(), task = $("#rec-task").value.trim();
      if (!name || !task) return alert("dataset name and task instruction are required");
      const toLocal = $("#rec-local").checked, toHub = $("#rec-hub").checked && !$("#rec-hub").disabled;
      if (!toLocal && !toHub) return alert("pick at least one place to save the recording");
      this.start("record", {
        name, task, to: toLocal && toHub ? "both" : toHub ? "hub" : "local",
        episodes: +$("#rec-episodes").value || 10,
        episode_s: +$("#rec-episode-s").value || 30,
        reset_s: $("#rec-reset-s").value === "" ? 10 : +$("#rec-reset-s").value,
        fps: Math.min(Math.max(+$("#rec-fps").value || DEFAULT_FPS, 1), maxFps),
      });
    };
    $("#btn-stop-top").onclick = () => this.stop();
    this.update();
  },
  async start(mode, body) {
    if (this._starting || session.active) return;
    this._starting = mode;
    this.update();
    try { await post(`/session/${mode}`, body); }
    catch (error) { alert(error.message); }
    finally {
      await refreshSession();
      this._starting = null;
      this.update();
    }
  },
  async stop() {
    if (this._stopPending || !session.active) return;
    this._stopPending = true;
    this.update();
    try { await post("/session/stop", {}); }
    catch (error) { alert(error.message); }
    finally {
      await refreshSession();
      this._stopPending = false;
      this.update();
    }
  },
  update() {
    const ts = $("#teleop-status");
    if (!ts) return;
    const p = session.active ? (session.parsed || {}) : {};
    const meta = session.active ? (session.meta || {}) : {};
    const busy = session.active || !!this._starting;
    const mode = this._starting || (session.active ? session.mode : null);
    const stopping = session.active && (session.stopping || p.operator_stopping || p.operator_phase === "stopping" || this._stopPending);
    const operator = p.operator_phase || "starting";
    const preparingRecord = mode === "record" && (!p.phase || p.phase === "preparing");
    const settingUp = ["starting", "homing", "synchronizing"].includes(operator) || preparingRecord;
    const gracefulRecordStop = !!session.record_stop_ready && !stopping;
    const returningHome = operator === "homing" && (stopping || p.phase === "finishing");
    for (const name of ["teleop", "record"]) {
      const button = $(`#btn-${name}`);
      button.hidden = false;
      button.disabled = busy;
      const label = name === "teleop" ? "Teleop" : "Recording";
      button.textContent = mode !== name ? `Start ${label}`
        : returningHome ? "Returning home…"
        : stopping ? "Stopping…"
        : p.phase === "saving" ? "Saving recording…"
        : p.phase === "upload" ? "Uploading…"
        : p.phase === "finishing" || operator === "closing" ? "Finishing…"
        : settingUp ? `Starting ${label}…`
        : name === "teleop" ? "Teleop running" : "Recording running";
    }
    document.querySelectorAll('[id^="rec-"] input, input[id^="rec-"]').forEach((input) => {
      input.disabled = busy || input.dataset.unavailable === "true";
    });
    const stop = $("#btn-stop-top");
    stop.hidden = !session.active;
    stop.disabled = !!this._stopPending;
    stop.textContent = this._stopPending ? "Stopping…"
      : stopping && operator === "homing" ? "Release now"
      : mode === "record" && p.phase === "saving" && !gracefulRecordStop ? "Stop (interrupt save)"
      : stopping ? "Stop again (interrupt)" : "Stop";
    stop.title = stopping ? "Interrupt cleanup and release the arms; an unfinished episode may be lost." : "End this session and return the arms home.";
    syncCams();
    const pairs = session.active && !stopping && ["ready", "holding", "synchronizing"].includes(operator) && ["teleop", "record"].includes(mode) ? (p.pairs || {}) : {};
    const ready = $("#teleop-ready");
    if (ready) {
      let message = "Ready to start", style = "ready";
      if (returningHome) {
        style = "setup";
        message = "Returning home — let go of the handles. Wait until the arms are released.";
      } else if (stopping) {
        style = "setup";
        message = mode === "record" && p.phase === "saving" ? "Stopping recording — saving the episode. Stop again interrupts saving and may discard it."
          : "Stopping — finishing the session and releasing the arms. Please wait.";
      } else if (mode === "record" && p.phase === "upload") {
        message = `Recording finished — uploading to the Hub. The arms are released.${phaseClock()}`;
      } else if (operator === "closing" || mode === "record" && p.phase === "finishing") {
        style = "setup";
        message = "Finishing — finalizing data and disconnecting arms; configured homing may run.";
      } else if (mode === "record" && p.phase === "saving") {
        style = "setup";
        message = `Saving episode ${(p.episode ?? 0) + 1} — encoding videos; followers hold their last command. ${gracefulRecordStop ? "Stop finishes saving, then returns home." : "Stop interrupts saving and may discard this episode."}${phaseClock()}`;
      } else if (busy && ["teleop", "record"].includes(mode)) {
        if (settingUp) {
          style = "setup";
          message = operator === "homing" ? "Starting — arms moving home. Let go of the handles and wait."
            : operator === "synchronizing" ? "Starting — followers synchronizing with the leaders. Please wait."
            : p.phase === "preparing" && operator === "holding" ? `Preparing episode ${(p.episode ?? 0) + 1} — following is paused by a handle button. Press that button again to resume; the episode clock has not started.`
            : p.phase === "preparing" ? `Preparing episode ${(p.episode ?? 0) + 1} — waiting for the followers to be ready. The episode clock has not started.`
            : `Starting ${mode === "teleop" ? "teleop" : "recording"} — connecting arms${mode === "record" ? " and cameras" : ""}. Please wait.`;
        } else if (operator === "holding") {
          message = "Following paused by a handle button — press that button again to resume, or Stop to return home.";
        } else if (mode === "record" && p.phase === "reset") {
          message = `Reset — reset the scene using the leaders; followers continue tracking.${phaseClock(meta.reset_s)}`;
        } else if (mode === "record") {
          style = "engaged";
          message = `Recording — move the leaders. Episode ${(p.episode ?? 0) + 1}${meta.episodes ? " of " + meta.episodes : ""}${phaseClock(meta.episode_s)}`;
        } else {
          style = "engaged";
          message = "Teleop ready — move the leaders; followers are tracking.";
        }
      } else if (session.active) {
        message = `Another session is active (${session.mode}). Stop it before starting teleop or recording.`;
      } else if (session.returncode != null && session.returncode !== 0 && !session.stop_requested) {
        ready.innerHTML = errBanner(`The last session failed (exit ${session.returncode}). Check Output for details.`);
        message = null;
      }
      if (message !== null) ready.innerHTML = `<div class="ready-banner ${style}" role="status"><span class="dot"></span>${esc(message)}</div>`;
    }
    ts.innerHTML = Object.entries(pairs).map(([name, pair]) => `<div class="st-list" style="margin:4px 0">
        ${st(pair.engaged, `${name}: ${pair.engaged ? "following" : "holding"}`, !pair.engaged)}
        <span class="mono" style="color:var(--muted)">err ${pair.error_rad != null ? pair.error_rad.toFixed(3) + " rad" : "–"} · grip ${pair.gripper != null ? pair.gripper.toFixed(2) : "–"}</span>
      </div>`).join("");
    const bits = [busy ? stN(stopping ? "stopping" : settingUp ? "starting" : `${mode} running`, true) : stN("idle")];
    if (session.active) {
      bits.push(stN(`session elapsed ${fmtDur(session.elapsed_s)}`));
      if (p.episode != null) bits.push(stN(`episode ${p.episode + 1}${meta.episodes ? " of " + meta.episodes : ""}`));
      if (p.phase) bits.push(stN(p.phase + (session.phase_elapsed_s != null ? ` ${fmtDur(session.phase_elapsed_s)}` : "")));
      if (!stopping && operator === "ready" && p.rate_hz) bits.push(stN(`${p.rate_hz.toFixed(0)} Hz`));
    }
    $("#rec-progress").innerHTML = bits.join("");
    fillLog();
  },
};

// ---- datasets ----
const transferOf = (session) => (session.active && (session.mode === "push" || session.mode === "pull") ? session : null);
function transferBadge(name, kind) {
  const t = transferOf(session);
  if (!t || !(t.meta?.name || "").endsWith(name)) return null;
  return `<span class="badge run"><span class="spin"></span>${t.mode === "push" ? "uploading" : "downloading"}… ${fmtDur(t.elapsed_s)}</span>`;
}
pages.datasets = {
  update() {
    // a finished upload / download changes where things are: redraw the list once it ends
    const key = `${!!transferOf(session)}|${session.mode}`;
    if (this._key !== undefined && this._key !== key && !transferOf(session)) this.render($("#main"), []);
    this._key = key;
    document.querySelectorAll("[data-live-elapsed]").forEach((b) => { b.textContent = fmtDur(session.elapsed_s); });
  },
  async render(el, args) {
    cleanupPage();
    this._key = `${!!transferOf(session)}|${session.mode}`;
    if (args.length) return renderDatasetDetail(el, decodeURIComponent(args[0]), args[1]);
    el.innerHTML = `${pageHead("Datasets", "<span class='mono'>data/datasets/</span>")}<div id="ds-list" class="sect">loading…</div>`;
    try {
      const list = await api("/datasets");
      const hubNote = overview?.hub?.logged_in ? "" : `<div class="hint">Sign in to Hugging Face on the Settings page to see and upload datasets in your account.</div>`;
      list.sort((a, b) => (b.modified || 0) - (a.modified || 0));  // newest first
      $("#ds-list").innerHTML = `<div class="panel">` + (list.length ? `<table><tr><th>name</th><th>recorded</th><th>where</th><th class="num">episodes</th><th class="num">frames</th><th class="num">fps</th><th>robot</th><th>tasks</th><th>cameras</th><th class="num">size</th><th></th></tr>` +
        list.map((d) => `<tr class="${d.where === "cloud" ? "" : "click"}" ${d.where === "cloud" ? "" : `onclick="location.hash='#/datasets/${encodeURIComponent(d.name)}'"`}>
          <td class="mono">${esc(d.name)}</td><td>${fmtDate(d.modified)}</td><td>${whereTag(d)}</td><td class="num">${d.episodes ?? "–"}</td><td class="num">${d.frames ?? "–"}</td><td class="num">${d.fps ?? "–"}</td>
          <td>${esc(d.robot_type ?? "–")}</td><td>${esc((d.tasks || []).join("; ") || "–")}</td>
          <td class="mono">${esc((d.cameras || []).join(", ") || "none")}</td><td class="num">${fmtBytes(d.size_bytes)}</td>
          <td class="actions" onclick="event.stopPropagation()">${transferBadge(d.name) ?? `${d.where === "local" && overview?.hub?.logged_in ? `<button data-push="${esc(d.name)}" ${session.active ? "disabled title='wait for the running session to finish'" : ""}>Upload</button>` : ""}
            ${d.where === "cloud" ? `<button data-pull="${esc(d.repo_id)}" ${session.active ? "disabled title='wait for the running session to finish'" : ""}>Download</button>` : ""}`}
            ${d.url ? `<a href="${esc(d.url)}" target="_blank" rel="noopener">Hub ↗</a>` : ""}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no datasets yet — record one from the Record page</div>`) + hubNote + `</div>`;
      const startTransfer = (path, body) => async (e) => {
        e.target.disabled = true;
        e.target.innerHTML = `<span class="spin"></span> starting…`;
        try { await post(path, body); await refreshSession(); this.render(el, []); }
        catch (err) { alert(err.message); this.render(el, []); }
      };
      document.querySelectorAll("[data-push]").forEach((b) => b.onclick = startTransfer("/hub/push-dataset", { name: b.dataset.push }));
      document.querySelectorAll("[data-pull]").forEach((b) => b.onclick = startTransfer("/hub/pull-dataset", { name: b.dataset.pull }));
    } catch (e) { $("#ds-list").innerHTML = errBanner(e.message); }
  },
};

async function renderDatasetDetail(el, name, epArg) {
  el.innerHTML = `<a class="back" href="#/datasets">← datasets</a>${pageHead(name)}<div id="ds-detail">loading…</div>`;
  const detailEl = $("#ds-detail", el);
  let d;
  try { d = await api(`/datasets/${encodeURIComponent(name)}`); }
  catch (e) { if (detailEl.isConnected) detailEl.innerHTML = errBanner(e.message); return; }
  if (!detailEl.isConnected) return;
  const eps = d.episode_list || [];
  detailEl.innerHTML = `
    <div class="sect"><div class="kv panel">
      <div>episodes / frames</div><div>${d.episodes} / ${d.frames}</div>
      <div>fps</div><div>${d.fps}</div>
      <div>robot</div><div>${esc(d.robot_type || "?")}</div>
      <div>tasks</div><div>${esc((d.tasks || []).join("; ") || "–")}</div>
      <div>cameras</div><div class="mono">${esc((d.cameras || []).join(", ") || "none")}</div>
      <div>size on disk</div><div>${fmtBytes(d.size_bytes)}</div>
      <div>path</div><div class="mono">${esc(d.path)}</div>
    </div></div>
    <div class="sect"><div class="sect-head">Episodes</div><div class="panel">
      <table><tr><th class="num">#</th><th class="num">frames</th><th class="num">duration</th><th>tasks</th><th></th></tr>
      ${eps.map((e) => `<tr class="click" onclick="location.hash='#/datasets/${encodeURIComponent(name)}/${e.episode_index}'">
        <td class="num mono">${e.episode_index}</td><td class="num">${e.length ?? "–"}</td><td class="num">${d.fps && e.length ? fmtDur(e.length / d.fps) : "–"}</td>
        <td>${esc(Array.isArray(e.tasks) ? e.tasks.join("; ") : e.tasks ?? "–")}</td><td>view →</td></tr>`).join("")}
      </table></div></div>
    <div id="ep-viewer"></div>`;
  const ep = epArg != null ? +epArg : (eps.length ? eps[0].episode_index : null);
  if (ep != null) renderEpisodeViewer($("#ep-viewer", detailEl), name, d, ep);
}

async function renderEpisodeViewer(el, name, detail, ep) {
  el.innerHTML = `<div class="sect"><div class="sect-head">Episode ${ep}</div><div id="ep-body">loading…</div></div>`;
  const body = $("#ep-body", el);
  let s;
  try { s = await api(`/datasets/${encodeURIComponent(name)}/episodes/${ep}`); }
  catch (e) { if (body.isConnected) body.innerHTML = errBanner(e.message); return; }
  if (!body.isConnected) return;
  const epMeta = (detail.episode_list || []).find((e) => e.episode_index === ep) || {};
  const cams = Object.keys(epMeta.videos || {});
  const t = s.timestamp || [];
  const t0 = t.length ? t[0] : 0, t1 = t.length ? t[t.length - 1] : 1;
  const colors = seriesColors();
  body.innerHTML = `
    ${cams.length ? `<div class="cams" style="margin-bottom:12px">` + cams.map((c) => `
      <div class="cam"><span class="label">${esc(c)}</span>
        <video id="vid-${esc(c)}" src="/api/datasets/${encodeURIComponent(name)}/video/${encodeURIComponent(c)}/${ep}" muted playsinline></video></div>`).join("") + `</div>
      <div class="toolbar"><button id="ep-play" class="primary">Play</button><span class="hint" style="margin:0">videos + charts play in sync</span></div>`
      : `<div class="toolbar"><button id="ep-play" class="primary">Play</button>
         <span class="hint" style="margin:0">no videos in this dataset — playing sweeps the cursor over the state/action charts</span></div>`}
    <div id="ep-play-error" role="status"></div>
    <input type="range" id="ep-scrub" min="${t0}" max="${t1}" step="0.01" value="${t0}" />
    <div class="legend"><span><span class="k" style="background:${colors.state}"></span>observation.state</span>
      <span><span class="k" style="background:${colors.action}"></span>action</span></div>
    <div class="charts" id="ep-charts"></div>`;

  // small multiples: one panel per state dimension, state + action series
  const names = s.names || (s["observation.state"]?.[0] || []).map((_, i) => "dim_" + i);
  const chartsEl = $("#ep-charts", body);
  const charts = names.map((dim, i) => {
    const cell = document.createElement("div");
    cell.className = "chart-cell";
    cell.innerHTML = `<div class="t">${esc(dim)}</div><canvas></canvas>`;
    chartsEl.appendChild(cell);
    return makeChart($("canvas", cell), dim, t,
      (s["observation.state"] || []).map((r) => r[i]),
      (s.action || []).map((r) => r[i]));
  });
  const setCursor = (tc) => charts.forEach((c) => c.setCursor(tc));

  const scrub = $("#ep-scrub", body);
  const playButton = $("#ep-play", body);
  const videos = cams.map((c) => ({ el: document.getElementById("vid-" + c), meta: epMeta.videos[c] }));
  const lead = videos[0];
  let playTimer = null;
  let playGeneration = 0;
  const stopPlay = () => { playGeneration++; if (playTimer) { clearInterval(playTimer); playTimer = null; } videos.forEach((v) => v.el.pause()); playButton.textContent = "Play"; };
  pageCleanup = () => {
    stopPlay();
    videos.forEach((v) => { v.el.removeAttribute("src"); v.el.load(); });
    charts.forEach((c) => c.dispose());
  };
  scrub.oninput = () => {
    stopPlay();
    const tc = +scrub.value;
    setCursor(tc);
    videos.forEach((v) => { v.el.currentTime = (v.meta.from_timestamp || 0) + (tc - t0); });
  };
  playButton.onclick = () => {
    if (playTimer || (lead && !lead.el.paused)) return stopPlay();
    const generation = ++playGeneration;
    $("#ep-play-error", body).innerHTML = "";
    playButton.textContent = "Pause";
    if (lead) {
      videos.forEach((v) => {
        v.el.currentTime = (v.meta.from_timestamp || 0) + (+scrub.value - t0);
        v.el.play().catch((error) => {
          // Seeking, pausing or leaving can cancel a pending browser play request.
          if (!body.isConnected || generation !== playGeneration) return;
          stopPlay();
          if (error.name !== "AbortError") $("#ep-play-error", body).innerHTML = errBanner(error.message);
        });
      });
      playTimer = setInterval(() => {
        const tc = t0 + (lead.el.currentTime - (lead.meta.from_timestamp || 0));
        if (lead.meta.to_timestamp && lead.el.currentTime >= lead.meta.to_timestamp) return stopPlay();
        scrub.value = tc; setCursor(tc);
      }, 66);
    } else {
      const startWall = performance.now(), startT = +scrub.value >= t1 - 0.05 ? t0 : +scrub.value;
      playTimer = setInterval(() => {
        const tc = startT + (performance.now() - startWall) / 1000;
        if (tc >= t1) return stopPlay();
        scrub.value = tc; setCursor(tc);
      }, 50);
    }
  };
  setCursor(t0);
}

// tiny canvas line chart with hover tooltip + external time cursor
function makeChart(canvas, label, t, s1, s2) {
  const tip = $("#viz-tip");
  const dpr = window.devicePixelRatio || 1;
  let cursorT = null;
  const draw = () => {
    const colors = seriesColors();
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 88;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!t.length) return;
    const all = [...s1, ...s2].filter((v) => v != null && isFinite(v));
    let lo = Math.min(...all), hi = Math.max(...all);
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
    const tA = t[0], tB = t[t.length - 1] || 1;
    const X = (tv) => ((tv - tA) / (tB - tA || 1)) * (w - 8) + 4;
    const Y = (v) => h - 6 - ((v - lo) / (hi - lo)) * (h - 12);
    // recessive grid: midline only
    ctx.strokeStyle = colors.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(4, Y((lo + hi) / 2)); ctx.lineTo(w - 4, Y((lo + hi) / 2)); ctx.stroke();
    for (const [series, color] of [[s1, colors.state], [s2, colors.action]]) {
      if (!series.length) continue;
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
      series.forEach((v, i) => { const x = X(t[i]), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    if (cursorT != null) {
      ctx.strokeStyle = colors.cursor; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(cursorT), 2); ctx.lineTo(X(cursorT), h - 2); ctx.stroke();
    }
  };
  canvas.addEventListener("mousemove", (ev) => {
    if (!t.length) return;
    const r = canvas.getBoundingClientRect();
    const frac = (ev.clientX - r.left - 4) / (r.width - 8);
    const idx = Math.max(0, Math.min(t.length - 1, Math.round(frac * (t.length - 1))));
    tip.style.display = "block";
    tip.style.left = ev.clientX + 12 + "px";
    tip.style.top = ev.clientY + 12 + "px";
    tip.innerHTML = `<b>${esc(label)}</b> @ ${t[idx].toFixed(2)}s<br/>
      state ${s1[idx] != null ? s1[idx].toFixed(4) : "–"}<br/>action ${s2[idx] != null ? s2[idx].toFixed(4) : "–"}`;
  });
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  draw();
  return { setCursor(tc) { cursorT = tc; draw(); }, dispose() { observer.disconnect(); tip.style.display = "none"; } };
}

// ---- inference (policy runs; backend routes remain /api/deployments) ----
pages.inference = {
  async render(el, args) {
    if (args.length) return renderRunDetail(el, decodeURIComponent(args[0]));
    this._runDetailId = null;
    this._detailView = null;
    this._submitted = null;
    this._activeSelection = null;
    this._previews = false;
    this._qualification = null;
    this._checking = false;
    this._checkSequence = (this._checkSequence || 0) + 1;
    this._defaultsLoaded = false;
    this._defaultsError = null;
    const form = this._form = {};
    clearTimeout(this._checkTimer);
    el.innerHTML = `
      ${pageHead("Inference", "run a supervised task with your attached policy")}
      <div class="sect panel pad inference-card">
        <div id="inf-selection-summary" class="inference-summary">Loading attached policy…</div>
        <div class="inference-main-fields">
          <label class="field">Task<input type="text" id="inf-task" value="" placeholder="Describe what the arms should do" /></label>
          <label class="field">Duration (seconds)<input type="number" id="inf-duration" value="60" min="1" max="3600" /></label>
        </div>
        <div id="inf-capture-controls" class="inference-recording">
          <div class="toolbar">
            <label class="check"><input type="checkbox" id="inf-trace" /> Save recording locally</label>
            <label class="check"><input type="checkbox" id="inf-upload" /> Also upload to Hugging Face</label>
          </div>
          <label class="field" id="inf-upload-destination" hidden>Private HF dataset<input type="text" id="inf-upload-repo" placeholder="your-namespace/yamkit-rollouts" /></label>
          <div id="inf-capture-note" class="hint">Save all three camera videos and joint traces, then click the run below to watch. Upload keeps the local originals.</div>
        </div>
        <div id="inf-attach-controls">
          <label class="check"><input type="checkbox" id="inf-mapping" /> I verified the arm / camera mapping and gripper calibration.</label>
        </div>
        <div class="inference-readiness" role="status" aria-live="polite">
          <div id="inf-qualification-status" class="hint"></div>
          <button id="btn-inf-preflight">Recheck readiness</button>
        </div>
        <div class="toolbar inference-actions">
          <button id="btn-ro" class="primary">Start rollout</button><button id="btn-inf-stop" class="danger">Stop local execution</button>
        </div>
        <div class="hint warn">Start enables motors and moves the selected followers. Stay at the arms with mounts secure, the area clear, Stop and power cutoff ready.</div>
        <div class="hint">Duration is policy time; startup and return home take additional time. Normal completion returns home, then releases. Stop/fault releases without home. Closing the browser is not Stop.</div>
        <details class="advanced inference-advanced" id="inf-advanced"><summary>Advanced settings</summary>
        <div class="form-grid">
          <label class="field">preset<select id="inf-preset"><option value="smolvla">SmolVLA base · forward check</option><option value="molmoact2">MolmoAct2 · bimanual YAM</option><option value="pi05">pi05 base · forward check</option><option value="custom">Custom compatible local checkpoint</option></select></label>
          <label class="field">backend<select id="inf-backend"><option value="local">Local (default)</option><option value="modal">Modal · retained MolmoAct2 session</option><option value="external">Lambda / own GPU host · SSH</option></select></label>
          <label class="field">remote controller<select id="inf-controller"><option value="reference">Reference full chunks</option><option value="async">Experimental async chunks</option></select></label>
          <label class="field">checkpoint<input type="text" id="inf-policy" value="smolvla" list="policy-list" /></label>
          <label class="field">followers<select id="inf-arms"><option value="">Both arms</option><option value="left">Left only (compatible local model)</option><option value="right">Right only (compatible local model)</option></select></label>
          <label class="field">local device<select id="inf-device"><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">MPS</option></select></label>
          <label class="field">Modal GPU<select id="inf-gpu"><option value="L40S">L40S · one container</option><option value="H100!">H100 · one retained container</option></select></label>
          <label class="field">Retained session<input type="text" id="inf-modal-app" placeholder="yamkit-vla-session-…" /></label>
          <label class="field">External service<input type="text" id="inf-external-service" placeholder="lambda-georgia" /></label>
        </div>
        <div class="toolbar"><button id="btn-attach-owned">Use owned MolmoAct2 session</button></div>
        <datalist id="policy-list"></datalist>
        <div class="toolbar"><label class="check"><input type="checkbox" id="inf-rtc" /> Local RTC (policy must support guidance)</label>
          <label class="check"><input type="checkbox" id="inf-crop" /> Optional center crop to 16:9 at remote policy boundary</label></div>
        <div id="inf-profile-note" class="hint"></div>
        <div class="hint">Reference full chunks executes one complete prediction before requesting the next. Experimental async chunks requests predictions in the background. Both use unguided inference. Changing controller requires a matching qualification.</div>
        <div class="toolbar" style="margin-top:12px">
          <button id="btn-pc">Check (no hardware)</button><button id="btn-prepare">Prepare Modal</button>
          <button id="btn-cloud-stop">Shut down owned cloud service</button>
        </div>
        <div class="hint">Remote defaults: 30 Hz, HTTP, ten-step inference, raw RGB. Readiness is checked without opening hardware, and checked again by the server before Start.</div>
      <div class="sect"><div class="sect-head">Diagnostics · not needed for a normal run</div>
        <label class="field">saved observation (.npz path inside this repository)<input type="text" id="inf-saved" placeholder="data/probes/observation.npz" /></label>
        <div class="toolbar"><button id="btn-probe-saved">Probe saved observation</button><button id="btn-probe-live">Probe live active read</button></div>
        <div class="hint warn">Live probe is GRAVITY-COMPENSATION ACTIVE READ: motors are active and this is not guaranteed motion-free. All gripper calibrations must be valid first. A successful probe never approves motion or replays its chunk.</div>
      </div></details></div>
      <div class="sect"><div class="sect-head">Operation</div><div class="panel pad"><div id="inf-status" role="status" aria-live="polite">No operation for this selection.</div><div id="inf-progress">${rolloutProgressHTML()}</div></div><details class="advanced"><summary>Operation log</summary><pre id="inf-result" class="log tall"></pre></details></div>
      <div class="sect"><div class="sect-head">Cameras</div>
        <button id="btn-inf-cameras">Show camera previews</button>
        <div class="hint">Showing previews opens the cameras and sends no motor commands.</div>
        <div id="inf-cams-content"></div></div>
      <div class="sect"><div class="sect-head">Runs</div><div id="run-list">loading…</div></div>`;
    this._profiles = [];
    api("/inference/profiles").then((data) => {
      if (this._form !== form || !$("#inf-policy")) return;
      this._profiles = data.profiles || [];
      this._ownedService = data.owned_service;
      if (data.defaults) this.applyDefaults(data.defaults);
      if (data.rollout_repo) $("#inf-upload-repo").value = data.rollout_repo;
      this._defaultsLoaded = true;
      this.syncForm();
      this.scheduleCheck();
    }).catch((e) => {
      if (this._form !== form || !$("#inf-policy")) return;
      this._defaultsError = `Could not load policy settings: ${e.message}. Reload this page to retry.`;
      this.syncForm();
    });
    api("/models").then((list) => {
      const dl = $("#policy-list");
      if (dl) dl.innerHTML = list.map((m) => `<option value="${esc(m.where === "cloud" ? m.repo_id : "outputs/" + m.path)}">${esc(m.policy_type ?? "")}</option>`).join("");
    }).catch(() => {});
    $("#inf-preset").onchange = () => { $("#inf-policy").value = $("#inf-preset").value === "custom" ? "" : $("#inf-preset").value; this.formChanged("inf-policy"); };
    ["inf-backend", "inf-controller", "inf-policy", "inf-task", "inf-arms", "inf-duration", "inf-device", "inf-gpu", "inf-rtc", "inf-crop", "inf-saved", "inf-modal-app", "inf-external-service", "inf-mapping", "inf-trace", "inf-upload", "inf-upload-repo"].forEach((id) => {
      document.getElementById(id).addEventListener("input", () => this.formChanged(id));
    });
    $("#btn-pc").onclick = (e) => this.launch("/session/policy-check", {}, e.target);
    $("#btn-prepare").onclick = (e) => this.launch("/session/modal-prepare", {}, e.target);
    $("#btn-ro").onclick = (e) => {
      this.syncForm();
      if ($("#btn-ro").disabled) return;
      const selected = this.selection();
      if (!confirm(`Start ${selected.duration}-second rollout?\n\nTask: ${selected.task}\n${selected.policy} · ${selected.backend} · ${selected.controller_mode} controller\n\nMotors WILL move. Confirm you are at the arms, mounts secure, grippers empty, workspace clear, Stop and physical power cutoff ready. Normal completion returns home then releases; Stop/fault releases without home.`)) return;
      this.launch("/session/rollout", { confirm_motion: true, supervised_confirmed: true }, e.target);
    };
    $("#btn-inf-preflight").onclick = () => this.checkAttachment();
    $("#btn-attach-owned").onclick = async () => {
      if (session.active) return;
      try { this._ownedService = (await api("/inference/profiles")).owned_service; }
      catch (e) { alert(e.message); return; }
      const owned = this._ownedService;
      if (!owned || owned.status !== "ready" || owned.profile_id !== "molmoact2"
          || owned.transport !== "http" || owned.execution_mode !== "cuda_graph10") {
        alert("No retained MolmoAct2 HTTP session is ready. Prepare and qualify it in Conductor first.");
        return;
      }
      $("#inf-backend").value = "modal";
      $("#inf-preset").value = $("#inf-policy").value = "molmoact2";
      $("#inf-modal-app").value = owned.app_name;
      $("#inf-gpu").value = "H100!";
      $("#inf-arms").value = "";
      $("#inf-mapping").checked = false;
      this._qualification = null;
      this.syncForm();
      this.scheduleCheck();
    };
    $("#btn-probe-saved").onclick = (e) => this.launch("/session/policy-probe", { saved: $("#inf-saved").value.trim() }, e.target);
    $("#btn-probe-live").onclick = (e) => {
      if (!confirm("Approve GRAVITY-COMPENSATION ACTIVE READ? Motors will be active. This is not guaranteed motion-free. No predicted position will be executed.")) return;
      this.launch("/session/policy-probe", { live: true, confirm_active_read: true }, e.target);
    };
    $("#btn-inf-cameras").onclick = () => {
      this._previews = !this._previews;
      const slot = $("#inf-cams-content");
      releaseCameraStreams(slot);
      slot.innerHTML = this._previews ? `<div id="cams-slot">${camsHTML()}</div>` : "";
      $("#btn-inf-cameras").textContent = this._previews ? "Hide camera previews" : "Show camera previews";
      updateRolloutClocks();
    };
    $("#btn-inf-stop").onclick = (e) => doPost("/session/stop", {}, e.target);
    $("#btn-cloud-stop").onclick = (e) => doPost("/session/modal-shutdown", {}, e.target);
    this.syncForm();
    this.refreshList();
  },
  applyDefaults(defaults) {
    this._followerArms = defaults.arms?.length === 2 ? [...defaults.arms] : ["left_follower", "right_follower"];
    const fields = { policy: "policy", task: "task", backend: "backend", controller_mode: "controller",
      duration: "duration", device: "device", gpu: "gpu", modal_app: "modal-app", external_service: "external-service" };
    for (const [key, id] of Object.entries(fields)) {
      if (key in defaults) $("#inf-" + id).value = defaults[key] ?? "";
    }
    $("#inf-preset").value = ["smolvla", "molmoact2", "pi05"].includes(defaults.policy) ? defaults.policy : "custom";
    $("#inf-arms").value = defaults.arms?.length === 1 ? defaults.arms[0].replace("_follower", "") : "";
    // Defaults are never approval and never opt in to recording or upload.
    for (const id of ["mapping", "trace", "upload", "rtc", "crop"]) $("#inf-" + id).checked = false;
  },
  formChanged(id) {
    if (["inf-policy", "inf-task", "inf-backend", "inf-controller", "inf-arms", "inf-crop", "inf-modal-app", "inf-external-service"].includes(id))
      $("#inf-mapping").checked = false;
    this._qualification = null;
    this.syncForm();
    this.scheduleCheck();
  },
  scheduleCheck() {
    clearTimeout(this._checkTimer);
    if (!this._defaultsLoaded || session.active || this._launching || !$("#inf-policy") || $("#inf-backend").value === "local") return;
    const form = this._form;
    this._checkTimer = setTimeout(() => {
      if (this._form === form && $("#inf-policy")) this.checkAttachment();
    }, 400);
  },
  selection() {
    const modal = $("#inf-backend").value === "modal";
    const external = $("#inf-backend").value === "external", remote = modal || external;
    return { policy: $("#inf-policy").value.trim(), task: $("#inf-task").value.trim(),
      backend: $("#inf-backend").value, device: $("#inf-device").value, gpu: $("#inf-gpu").value,
      duration: Number($("#inf-duration").value), fps: 30, rtc: $("#inf-rtc").checked,
      center_crop: $("#inf-crop").checked, async_chunks: !remote || $("#inf-controller").value !== "reference",
      controller_mode: remote ? $("#inf-controller").value : "async",
      modal_app: modal ? $("#inf-modal-app").value.trim() || null : null,
      external_service: external ? $("#inf-external-service").value.trim() || null : null,
      call_mode: remote ? "http" : "remote", execution_mode: remote ? "cuda_graph10" : "eager",
      image_encoding: "rgb8", jpeg_quality: 85, prediction_queue_threshold: null,
      mapping_accepted: remote && $("#inf-mapping").checked,
      capture_trace: $("#inf-trace").checked || $("#inf-upload").checked,
      upload_repo_id: $("#inf-upload").checked ? $("#inf-upload-repo").value.trim() : null,
      arms: $("#inf-arms").value ? [$("#inf-arms").value] : remote ? this._followerArms || ["left_follower", "right_follower"] : null };
  },
  async checkAttachment() {
    clearTimeout(this._checkTimer);
    if (this._checking || session.active || this._launching || !this._defaultsLoaded) return;
    const selected = this.selection(), key = JSON.stringify(selected), sequence = ++this._checkSequence;
    const form = this._form;
    const requestedAt = Date.now();
    this._checking = true;
    this._qualification = null;
    this.syncForm();
    try {
      const result = await post("/inference/preflight", selected);
      if (this._form !== form || !$("#inf-policy") || sequence !== this._checkSequence || JSON.stringify(this.selection()) !== key) return;
      this._qualification = { ...result, key,
        deadline: requestedAt + Math.max(0, (result.expires_at - result.checked_at - selected.duration - 60) * 1000) };
    } catch (e) {
      if (this._form === form && $("#inf-policy") && sequence === this._checkSequence && JSON.stringify(this.selection()) === key)
        this._qualification = { key, ready: false, reason: e.message };
    } finally {
      if (this._form !== form || !$("#inf-policy")) return;
      this._checking = false;
      this.syncForm();
      if (JSON.stringify(this.selection()) !== key) this.scheduleCheck();
    }
  },
  syncForm() {
    if (!$("#inf-policy")) return;
    // A newly opened tab must show the running job, not the next-run defaults.
    // Import display settings only: a prior job's approvals are never inherited.
    if (this._defaultsLoaded && session.active && session.mode === "rollout" && session.meta?.policy
        && this._submitted?.id !== session.meta.operation_id) {
      const activeSelection = JSON.stringify(session.meta);
      if (this._activeSelection !== activeSelection) {
        this.applyDefaults(session.meta);
        $("#inf-trace").checked = session.meta.capture_trace === true;
        $("#inf-upload").checked = !!session.meta.upload_repo_id;
        if (session.meta.upload_repo_id) $("#inf-upload-repo").value = session.meta.upload_repo_id;
        this._activeSelection = activeSelection;
        this._qualification = null;
      }
    }
    const external = $("#inf-backend").value === "external";
    const modal = ["modal", "external"].includes($("#inf-backend").value);
    const busy = session.active || this._launching || !this._defaultsLoaded;
    for (const id of ["preset", "policy", "task", "backend", "arms", "duration", "mapping", "saved"])
      $("#inf-" + id).disabled = busy;
    $("#inf-device").disabled = modal || busy;
    $("#inf-controller").disabled = !modal || busy;
    $("#inf-gpu").disabled = !modal || external || busy;
    $("#inf-modal-app").disabled = !modal || external || busy;
    $("#inf-external-service").disabled = !external || busy;
    $("#inf-attach-controls").hidden = !modal;
    $("#btn-attach-owned").disabled = busy;
    $("#btn-attach-owned").hidden = !this._ownedService;
    $("#inf-rtc").disabled = modal || busy;
    if (modal) $("#inf-rtc").checked = false;
    $("#inf-crop").disabled = !modal || busy;
    if (!modal) $("#inf-crop").checked = false;
    const captureSupported = modal && ["molmoact2", "lerobot/MolmoAct2-BimanualYAM-LeRobot"].includes($("#inf-policy").value.trim())
      && !!$("#inf-task").value.trim() && !$("#inf-arms").value && !$("#inf-crop").checked
      && [5, 10, 20, 30, 45, 60, 90].includes(Number($("#inf-duration").value));
    if ($("#inf-upload").checked) $("#inf-trace").checked = true;
    $("#inf-trace").disabled = busy || (!captureSupported && !$("#inf-trace").checked) || $("#inf-upload").checked;
    $("#inf-upload").disabled = busy || (!captureSupported && !$("#inf-upload").checked);
    $("#inf-upload-repo").disabled = busy || !modal || !$("#inf-upload").checked;
    $("#inf-upload-destination").hidden = !$("#inf-upload").checked;
    $("#inf-capture-note").textContent = captureSupported
      ? $("#inf-trace").checked
        ? "Saves all three camera videos and joint traces after motor release. Click the completed run below to watch; local originals are kept."
        : "Recording is off. Enable local saving before Start to watch this rollout later."
      : "Recording requires a qualified MolmoAct2 remote session, both followers, full images, and 5, 10, 20, 30, 45, 60 or 90 seconds. Change these settings or turn recording off.";
    const selected = this.selection();
    const profile = this._profiles.find((p) => p.id === selected.policy || p.repo_id === selected.policy);
    const qualification = this._qualification?.key === JSON.stringify(selected) ? this._qualification : null;
    if (selected.capture_trace && qualification?.capture_memory) {
      const memory = qualification.capture_memory;
      $("#inf-capture-note").textContent += ` Capture memory: ${(memory.required_bytes / 1e9).toFixed(2)} GB required, ${(memory.available_bytes / 1e9).toFixed(2)} GB available.`;
    }
    const qualified = qualification?.ready === true && Number.isFinite(qualification.deadline) && Date.now() < qualification.deadline;
    const modalBlocked = modal && (!qualified || !selected.mapping_accepted);
    $("#inf-selection-summary").textContent = this._defaultsLoaded
      ? `${profile?.id === "molmoact2" ? "MolmoAct2" : selected.policy || "Choose a policy"} · ${external ? selected.external_service || "Choose an external service" : selected.backend} · ${modal ? selected.controller_mode + " controller · " : ""}${selected.arms?.length === 1 ? "One arm" : "Both arms"} · 30 Hz`
      : "Loading attached policy…";
    const invalidSelection = !selected.task || !selected.policy || !Number.isFinite(selected.duration) || selected.duration < 1 || selected.duration > 3600;
    $("#inf-qualification-status").textContent = this._defaultsError || (!this._defaultsLoaded ? "Loading current settings…"
      : invalidSelection ? "Enter a task, policy and duration between 1 and 3600 seconds."
      : !modal ? "Local policy selected. Verify checkpoint and rig compatibility before Start."
      : this._checking ? "Checking local qualification…"
      : qualified ? `Qualified for this selection. Start within ${Math.ceil((qualification.deadline - Date.now()) / 1000)} seconds; the server checks again before launch.`
      : qualification?.ready ? "Session expires too soon. Refresh it in Conductor and check again."
      : qualification?.reason || "Checking these settings requires no hardware or cloud call. Use Recheck readiness if needed.");
    const profileNote = profile ? `Revision ${profile.revision}. ${profile.mapping_note}` : "Custom checkpoints use the existing local LeRobot path; verify their rig compatibility before motion.";
    const blockedReason = qualified ? "Accept the verified YAM mapping before supervised Start."
      : qualification?.reason || "Physical remote rollout BLOCKED until this exact retained session passes the local check.";
    $("#inf-profile-note").textContent = profileNote + (modalBlocked ? ` ${blockedReason}` : "");
    ["btn-pc", "btn-prepare", "btn-ro", "btn-probe-saved", "btn-probe-live", "btn-cloud-stop"].forEach((id) => { document.getElementById(id).disabled = busy; });
    $("#btn-pc").hidden = modal;
    $("#btn-prepare").hidden = modal;
    $("#btn-cloud-stop").hidden = modal;
    $("#btn-prepare").disabled = true;
    $("#btn-cloud-stop").disabled = true;
    $("#btn-probe-saved").disabled ||= external;
    $("#btn-probe-live").disabled ||= external;
    $("#btn-inf-preflight").disabled = busy || this._checking || !modal || invalidSelection;
    $("#btn-inf-preflight").hidden = !modal;
    $("#btn-ro").textContent = this._launching ? "Starting…" : `Start ${selected.duration || ""}s rollout`;
    $("#btn-ro").disabled ||= busy || invalidSelection || modalBlocked || (selected.capture_trace && !captureSupported)
      || !!profile && (!profile.mapping_verified || (profile.id === "molmoact2" && selected.rtc));
    $("#btn-inf-stop").disabled = !session.active;
    const rolloutPhase = session.parsed?.rollout_phase;
    $("#btn-inf-stop").textContent = session.active && session.mode === "rollout"
      ? session.rollout_progress?.resources_released || rolloutPhase === "released" ? "Interrupt saving" : "Stop and release arms"
      : "Stop local execution";
    const submitted = this._submitted;
    const matches = submitted && submitted.selection === JSON.stringify(selected) && submitted.saved === $("#inf-saved").value && submitted.id === session.meta?.operation_id;
    $("#inf-status").textContent = matches ? `${session.mode}: ${session.active ? (session.stopping ? "stopping local process…" : "running…") : session.stop_requested ? "stopped by user" : session.returncode === 0 ? "completed" : "failed or stopped"} · operation ${submitted.id}`
      : session.active ? `Another ${session.mode || "UI"} operation is active. Start is locked; do not run a second job.`
      : session.mode === "rollout" && session.meta?.operation_id ? `Last UI rollout: ${session.stop_requested ? "stopped" : session.returncode === 0 ? "completed" : "failed or stopped"} · ${session.meta.task || ""}. See Runs below; completion alone does not prove task success.`
      : "Idle. No rollout is running in this UI. Runs started separately in a terminal must be stopped in that terminal.";
    const managedRollout = session.active && session.mode === "rollout";
    if (managedRollout) {
      const phases = {
        running: "Policy running — 30 Hz",
        returning_home: "Returning home — keep clear; Stop releases the arms",
        releasing: "Releasing arms…",
        released: session.meta?.capture_trace ? "Arms released — saving video and joint traces…" : "Arms released — finishing…",
      };
      const status = session.stopping ? "Stopping — releasing arms and finishing cleanup…"
        : phases[rolloutPhase] || "Preparing cameras and arms…";
      $("#inf-status").textContent = `${status} · ${session.meta?.task || ""}`;
    }
    const progress = rolloutProgressView(session);
    if (progress) $("#inf-status").textContent = `${progress.label} · ${session.meta?.task || ""}`;
    $("#inf-result").textContent = matches || managedRollout ? (session.log || []).join("\n") +
      (session.parsed?.result ? "\n" + JSON.stringify(session.parsed.result, null, 2) : "") : "";
    syncCams();
    updateRolloutClocks();
  },
  async launch(path, extra, button) {
    if (this._launching || session.active) return;
    const selected = this.selection();
    const saved = $("#inf-saved").value, form = this._form;
    this._launching = true;
    this.syncForm();
    try {
      const result = await post(path, { ...selected, ...extra });
      if (this._form === form)
        this._submitted = { id: result.meta?.operation_id, selection: JSON.stringify(selected), saved };
      await refreshSession();
    } catch (e) { alert(e.message); }
    finally { this._launching = false; if (this._form === form) this.syncForm(); }
  },
  async refreshList() {
    const el = $("#run-list");
    if (!el) return;
    try {
      const list = await api("/deployments");
      this._uploadsPending = list.some((d) => d.status === "running" || d.recording?.state === "pending" || rolloutProgressPending(d.rollout_progress) || ["queued", "packaging", "uploading"].includes(d.upload?.status));
      this._lastRunRefresh = Date.now();
      el.innerHTML = `<div class="panel">` + (list.length ? `<table><tr><th>run</th><th>task</th><th class="num">duration</th><th>status</th><th>recording</th><th>HF upload</th></tr>` +
        list.map((d) => `<tr class="click" onclick="location.hash='#/inference/${encodeURIComponent(d.id)}'">
          <td class="mono"><a href="#/inference/${encodeURIComponent(d.id)}">${esc(d.id)}</a></td><td>${esc(d.task ?? "–")}</td>
          <td class="num">${fmtDur(d.duration_s)}</td>
          <td>${st(d.status === "success", d.status ?? "?", d.status === "running" || d.status === "stopped")}</td>
          <td>${esc(recordingLabel(d))}</td><td>${esc(d.upload?.status || "off")}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no policy runs yet — run a policy check or rollout above</div>`) + `</div>`;
    } catch (e) { el.innerHTML = errBanner(e.message); }
  },
  update() {
    this.syncForm();
    const view = this._detailView;
    if (view?.alive) syncRunStop(view);
    if (view?.alive && view.pending && !view.requestBusy && Date.now() - view.lastRefresh > 2000) {
      view.requestBusy = true;
      view.lastRefresh = Date.now();
      api(`/deployments/${encodeURIComponent(view.id)}`).then((d) => {
        if (view.alive) updateRunDetail(view, d);
      }).catch(() => {}).finally(() => { view.requestBusy = false; });
    }
    if (this._wasActive !== session.active || this._uploadsPending && Date.now() - (this._lastRunRefresh || 0) > 2000) {
      this._wasActive = session.active; this.refreshList();
    }
  },
};

function rolloutUploadHTML(upload) {
  if (!upload) return "Not requested";
  const link = /^https:\/\/huggingface\.co\/datasets\//.test(upload.url || "")
    ? ` · <a href="${esc(upload.url)}" target="_blank" rel="noopener">Open private HF run</a>` : "";
  return `${esc(upload.status)}${upload.repo_id ? ` · ${esc(upload.repo_id)}` : ""}${link}` +
    (upload.error ? `<div class="hint warn">${esc(upload.error)}</div>` : "") +
    (upload.retry_command ? `<div class="hint mono">${esc(upload.retry_command)}</div>` : "");
}

function syncRunStop(view) {
  if (!view.body?.isConnected) return;
  const button = $("#run-stop", view.body);
  if (!button) return;
  button.hidden = !session.active;
  button.disabled = !session.active;
  button.textContent = session.rollout_progress?.resources_released || session.parsed?.rollout_phase === "released" ? "Interrupt current saving" : "Stop current local execution";
}

async function renderRunDetail(el, id) {
  pages.inference._runDetailId = id;
  const view = pages.inference._detailView = {id, alive: true, pending: false, lastRefresh: Date.now(), replayKey: null};
  pageCleanup = () => { view.alive = false; view.cleanupPlayback?.(); };
  el.innerHTML = `<a class="back" href="#/inference">← inference</a>${pageHead(id)}<div id="run-detail">loading…</div>`;
  view.body = $("#run-detail");
  let d;
  try { d = await api(`/deployments/${encodeURIComponent(id)}`); }
  catch (e) { if (view.alive) view.body.innerHTML = errBanner(e.message); return; }
  if (!view.alive) return;
  view.body.innerHTML = `<div class="toolbar"><button id="run-stop" class="danger" hidden disabled>Stop current local execution</button></div><div id="run-summary" class="sect"></div>
    <div id="run-live-progress">${rolloutProgressHTML()}</div>
    <div class="sect"><div class="sect-head">Recording</div><div id="run-recording"></div></div>
    <details class="advanced sect"><summary>Run details, artifacts and log</summary>
      <div id="run-diagnostics"></div><div id="run-artifacts" class="panel pad"></div>
      <pre id="run-log" class="log tall"></pre></details>`;
  $("#run-stop", view.body).onclick = (event) => doPost("/session/stop", {}, event.target);
  syncRunStop(view);
  updateRunDetail(view, d);
}

function recordingLabel(d) {
  const state = d.recording?.state;
  return ({available: "Watch recording", pending: "Recording / saving…", partial: "Partial recording",
    unavailable: "Recording unavailable", not_recorded: "Not recorded"})[state]
    || (d.videos?.length ? "Watch recording" : "Not recorded");
}

function updateRunDetail(view, d) {
  if (!view.alive || !view.body.isConnected) return;
  view.pending = d.status === "running" || d.recording?.state === "pending" || rolloutProgressPending(d.rollout_progress) || ["queued", "packaging", "uploading"].includes(d.upload?.status);
  view.progressOperationId = d.rollout_progress?.operation_id;
  renderRolloutProgress($("#run-live-progress .rollout-progress", view.body), rolloutProgressView({rollout_progress: d.rollout_progress}));
  $("#run-summary", view.body).innerHTML = `<div class="kv panel">
      <div>status</div><div>${st(d.status === "success", d.status, d.status !== "failed")}${d.termination ? ` <span class="crumb">— ${esc(d.termination)}</span>` : ""}</div>
      <div>task</div><div>${esc(d.task ?? "–")}</div>
      <div>started</div><div>${fmtDate(d.started_at)}</div>
      <div>session duration</div><div>${fmtDur(d.duration_s)}</div>
      <div>HF upload</div><div id="run-upload">${rolloutUploadHTML(d.upload)}</div>
    </div>`;
  $("#run-diagnostics", view.body).innerHTML = `<div class="kv panel sect">
      <div>kind</div><div>${esc(d.kind)}</div>
      <div>model</div><div class="mono">${esc(d.policy ?? "–")}</div>
      <div>latency (first call)</div><div>${d.first_call_ms != null ? d.first_call_ms.toFixed(0) + " ms" : "–"}</div>
      <div>latency (next calls)</div><div>${d.step_call_ms ? d.step_call_ms.map((x) => x.toFixed(0)).join(" / ") + " ms" : "–"}</div>
      <div>exit code</div><div class="mono">${d.returncode ?? "–"}</div>
    </div>`;
  $("#run-artifacts", view.body).innerHTML = (d.artifacts || []).map((name) =>
    `<a href="/api/deployments/${encodeURIComponent(view.id)}/artifact/${encodeURIComponent(name)}" target="_blank" rel="noopener">${esc(name)}</a>`).join(" · ") || "No exported debug artifacts.";
  $("#run-log", view.body).textContent = (d.log || []).join("\n") || "(empty)";
  const replayKey = JSON.stringify([d.videos, d.recording]);
  if (view.replayKey === replayKey) return; // Upload/status polling must not interrupt playback.
  view.replayKey = replayKey;
  view.cleanupPlayback?.();
  const slot = $("#run-recording", view.body), recording = d.recording || {};
  const videos = recording.state === "pending" ? [] : d.videos || [];
  const message = videos.length ? recording.state === "partial" ? "Only part of this recording is available. Local originals are retained." : "Saved locally. Play or scrub all camera views together."
    : recording.state === "pending" ? "Recording / export is in progress. Videos will appear here automatically after the arms are released and export finishes."
    : recording.state === "unavailable" ? "Recording was requested, but no playable video was exported. Check the details and log below; original files are retained."
    : "This rollout was not recorded. Enable Save recording locally before your next run; video cannot be recovered retroactively.";
  slot.innerHTML = `<div class="panel pad"><div>${esc(message)}</div>
    ${(recording.errors || []).map((error) => `<div class="hint warn">${esc(error)}</div>`).join("")}
    ${videos.length ? `<div class="cams rollout-replay">${videos.map((name) => {
      const label = ({"top.mp4": "Top camera", "left_wrist.mp4": "Left wrist", "right_wrist.mp4": "Right wrist"})[name] || name;
      const url = `/api/deployments/${encodeURIComponent(view.id)}/video/${encodeURIComponent(name)}`;
      return `<div><div class="cam"><span class="label">${esc(label)}</span><span class="replay-timer" aria-label="Recorded video time"></span><video src="${url}" aria-label="${esc(label)} recording" preload="metadata" muted playsinline></video></div><a class="hint" href="${url}" download="${esc(name)}">Download video</a></div>`;
    }).join("")}</div><div class="toolbar"><button id="run-play" class="primary">Play recording</button><span id="run-play-time" class="mono">00:00.0</span><span id="run-play-remaining" class="hint"></span></div>
      <input type="range" id="run-scrub" min="0" max="${recording.duration_s || 0}" step="0.01" value="0" aria-label="Recording playback position" />
      <div id="run-play-error" class="hint warn" role="status"></div>
      <div class="hint">Policy-phase video only; startup and return home are not recorded. Timing follows observation receipt, not camera exposure.</div>` : ""}</div>`;
  view.cleanupPlayback = videos.length ? setupRunPlayback(slot, recording.duration_s) : null;
}

function setupRunPlayback(slot, knownDuration) {
  const videos = [...slot.querySelectorAll("video")], lead = videos[0];
  const button = $("#run-play", slot), scrub = $("#run-scrub", slot), clock = $("#run-play-time", slot);
  let alive = true, generation = 0, timer = null;
  const stop = () => {
    generation++;
    clearInterval(timer); timer = null;
    videos.forEach((video) => video.pause());
    button.textContent = "Play recording";
  };
  const update = () => {
    const duration = Number.isFinite(lead.duration) && lead.duration > 0 ? lead.duration : knownDuration;
    if (Number.isFinite(duration) && duration > 0) scrub.max = duration;
    scrub.value = lead.currentTime;
    clock.textContent = `${fmtClock(lead.currentTime)} / ${Number.isFinite(duration) ? fmtClock(duration) : "…"}`;
    $("#run-play-remaining", slot).textContent = Number.isFinite(duration) ? `${fmtClock(Math.max(0, duration - lead.currentTime))} remaining` : "";
    videos.forEach((video) => {
      const overlay = $(".replay-timer", video.parentElement);
      const total = Number.isFinite(video.duration) ? video.duration : knownDuration;
      if (overlay) overlay.textContent = `${fmtClock(video.currentTime)} / ${total ? fmtClock(total) : "…"}`;
    });
  };
  const fail = () => { stop(); $("#run-play-error", slot).textContent = "Video playback failed. Try downloading the saved video; check the run log if export was incomplete."; };
  videos.forEach((video) => { video.onerror = fail; video.onloadedmetadata = video.ontimeupdate = update; });
  lead.onended = () => { stop(); update(); };
  scrub.oninput = () => {
    stop();
    videos.forEach((video) => { video.currentTime = Number(scrub.value); });
    update();
  };
  button.onclick = () => {
    if (timer || !lead.paused) return stop();
    const startAt = lead.ended ? 0 : Number(scrub.value), request = ++generation;
    $("#run-play-error", slot).textContent = "";
    button.textContent = "Pause";
    videos.forEach((video) => {
      video.currentTime = startAt;
      video.play().catch((error) => { if (alive && generation === request && error.name !== "AbortError") fail(); });
    });
    timer = setInterval(() => {
      update();
      for (const video of videos.slice(1)) {
        if (Math.abs(video.currentTime - lead.currentTime) > 0.15) video.currentTime = lead.currentTime;
      }
    }, 100);
  };
  update();
  return () => {
    alive = false; stop();
    videos.forEach((video) => { video.onerror = video.onloadedmetadata = video.onended = video.ontimeupdate = null; video.removeAttribute("src"); video.load(); });
  };
}

// ---- models ----
function whereTag(x) {
  const w = x.where || "local";
  const label = w === "both" ? "local + cloud" : w === "cloud" ? "cloud" : "local";
  return `<span class="where where-${w}" title="${w === "cloud" ? "only on the Hugging Face Hub" : w === "both" ? "on this computer and on the Hub" : "only on this computer"}">${label}${x.private ? " · private" : ""}</span>`;
}

async function renderHubModelDetail(el, repo) {
  el.innerHTML = `<a class="back" href="#/models">← models</a>${pageHead(repo.split("/").pop(), `<span class="mono">${esc(repo)}</span> on the Hugging Face Hub`)}<div id="model-detail">loading…</div>`;
  let d;
  try { d = await api(`/hub/models/${repo}`); }
  catch (e) { $("#model-detail").innerHTML = errBanner(e.message); return; }
  const tc = d.train_config || {};
  $("#model-detail").innerHTML = `
    <div class="sect"><div class="kv panel">
      <div>policy type</div><div>${esc(d.policy_type ?? "?")}</div>
      <div>size</div><div>${fmtBytes(d.size_bytes)}</div>
      <div>modified</div><div>${fmtDate(d.modified)}</div>
      <div>train steps</div><div>${tc.steps ?? "–"}</div>
      <div>train dataset</div><div class="mono">${esc((tc.dataset || {}).repo_id ?? "–")}</div>
      <div>use it</div><div class="mono">yamkit rollout --policy ${esc(repo)} --task "…"</div>
      <div>link</div><div><a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.url)}</a></div>
    </div></div>
    <div class="sect"><div class="sect-head">Files</div><div class="panel">
      <table><tr><th>file</th><th class="num">size</th></tr>
      ${(d.files || []).map((f) => `<tr><td class="mono">${esc(f.name)}</td><td class="num">${fmtBytes(f.size_bytes)}</td></tr>`).join("")}
      </table></div></div>
    <div class="sect"><div class="sect-head">config.json</div>
      <pre class="log tall">${esc(JSON.stringify(d.config || {}, null, 2))}</pre></div>`;
}

pages.models = {
  update() {
    const key = `${!!transferOf(session)}|${session.mode}`;
    if (this._key !== undefined && this._key !== key && !transferOf(session)) this.render($("#main"), []);
    this._key = key;
  },
  async render(el, args) {
    this._key = `${!!transferOf(session)}|${session.mode}`;
    if (args[0] === "hub" && args.length >= 3) return renderHubModelDetail(el, args.slice(1).map(decodeURIComponent).join("/"));
    if (args.length) return renderModelDetail(el, decodeURIComponent(args[0]));
    el.innerHTML = `${pageHead("Models", "checkpoints under <span class='mono'>outputs/</span>")}<div id="model-list" class="sect">loading…</div>`;
    try {
      const list = await api("/models");
      $("#model-list").innerHTML = `<div class="panel">` + (list.length ? `<table><tr><th>path</th><th>where</th><th>type</th><th class="num">steps</th><th>dataset</th><th class="num">size</th><th>modified</th><th></th></tr>` +
        list.map((m) => `<tr class="click" onclick="location.hash='${m.where === "cloud" ? `#/models/hub/${m.repo_id}` : `#/models/${encodeURIComponent(m.path)}`}'">
          <td class="mono">${m.where === "cloud" ? esc(m.repo_id) : "outputs/" + esc(m.path)}</td><td>${whereTag(m)}</td><td>${esc(m.policy_type ?? "?")}</td>
          <td class="num">${m.steps ?? "–"}</td><td class="mono">${esc(m.dataset ?? "–")}</td>
          <td class="num">${fmtBytes(m.size_bytes)}</td><td>${fmtDate(m.modified)}</td>
          <td class="actions" onclick="event.stopPropagation()">${transferBadge(m.path) ?? (m.where === "local" && overview?.hub?.logged_in ? `<button data-push-model="${esc(m.path)}" ${session.active ? "disabled title='wait for the running session to finish'" : ""}>Upload</button>` : "")}
            ${m.url ? `<a href="${esc(m.url)}" target="_blank" rel="noopener">Hub ↗</a>` : ""}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no checkpoints under outputs/ — see README §6 for training</div>`) + `</div>`;
      document.querySelectorAll("[data-push-model]").forEach((b) => b.onclick = async (e) => {
        e.target.disabled = true;
        e.target.innerHTML = `<span class="spin"></span> starting…`;
        try { await post("/hub/push-model", { name: b.dataset.pushModel }); await refreshSession(); this.render(el, []); }
        catch (err) { alert(err.message); this.render(el, []); }
      });
    } catch (e) { $("#model-list").innerHTML = errBanner(e.message); }
  },
};

async function renderModelDetail(el, path) {
  el.innerHTML = `<a class="back" href="#/models">← models</a>${pageHead(path.split("/").pop() || path, `<span class="mono">outputs/${esc(path)}</span>`)}<div id="model-detail">loading…</div>`;
  let d;
  try { d = await api(`/models/${path.split("/").map(encodeURIComponent).join("/")}`); }
  catch (e) { $("#model-detail").innerHTML = errBanner(e.message); return; }
  const tc = d.train_config || {};
  $("#model-detail").innerHTML = `
    <div class="sect"><div class="kv panel">
      <div>policy type</div><div>${esc(d.policy_type ?? "?")}</div>
      <div>size on disk</div><div>${fmtBytes(d.size_bytes)}</div>
      <div>modified</div><div>${fmtDate(d.modified)}</div>
      <div>train steps</div><div>${tc.steps ?? "–"}</div>
      <div>train dataset</div><div class="mono">${esc((tc.dataset || {}).repo_id ?? "–")}</div>
    </div></div>
    <div class="sect"><div class="sect-head">Files</div><div class="panel">
      <table><tr><th>file</th><th class="num">size</th></tr>
      ${(d.files || []).map((f) => `<tr><td class="mono">${esc(f.name)}</td><td class="num">${fmtBytes(f.size_bytes)}</td></tr>`).join("")}
      </table></div></div>
    <div class="sect"><div class="sect-head">config.json</div>
      <pre class="log tall">${esc(JSON.stringify(d.config || {}, null, 2))}</pre></div>
    ${Object.keys(tc).length ? `<div class="sect"><div class="sect-head">train_config.json</div>
      <pre class="log">${esc(JSON.stringify(tc, null, 2))}</pre></div>` : ""}`;
}

// ---- settings ----
pages.settings = {
  async render(el) {
    el.innerHTML = `${pageHead("Settings", "", `<button id="cfg-reload">Reload</button>`)}<div id="cfg-body">loading…</div>`;
    const body = $("#cfg-body", el);
    $("#cfg-reload").onclick = () => this.render(el);
    let c;
    try { c = await api("/config"); }
    catch (e) { if (body.isConnected) body.innerHTML = errBanner(e.message); return; }
    if (!body.isConnected) return;
    this.cfg = c;
    const ctl = c.control || {};
    const hubCfg = c.hub || {};
    const ctlFields = [
      ["teleop_hz", "teleop loop rate (Hz)"], ["sync_seconds", "engage sync move (s)"],
      ["bilateral_kp", "bilateral force-feedback gain"], ["engage_button", "engage button index"],
      ["max_joint_speed", "max joint speed (rad/s)"], ["max_gripper_speed", "max gripper speed (1/s)"],
      ["home_speed", "return-to-home speed, followers (rad/s, 0 = off)"], ["leader_home_speed", "return-to-home speed, leaders (rad/s)"],
    ];
    body.innerHTML = `
      <div class="kv panel" style="margin-top:16px">
        <div>rig file</div><div class="mono">${esc(c.path)}${c.found ? "" : " (missing)"}</div>
        <div>validation</div><div>${(c.problems || []).length ? st(false, c.problems.join("; ")) : st(true, "ok")}</div>
      </div>
      <div class="sect"><div class="sect-head">Control</div><div class="panel pad">
        <div class="form-grid" style="max-width:640px">
          ${ctlFields.map(([k, label]) => `<label class="field">${esc(label)}
            <input type="number" step="any" data-ctl="${k}" value="${ctl[k] ?? ""}" ${c.found ? "" : "disabled"} /></label>`).join("")}
        </div>
        <div class="toolbar" style="margin-top:14px">
          <button id="ctl-save" class="primary" ${c.found ? "" : "disabled"}>Save control</button>
          <span class="save-note" id="ctl-note"></span>
        </div>
        <div class="hint">Speed clamps bound every commanded move (teleop and rollout). Saving is refused
          while a hardware session is running.</div>
      </div></div>
      <div class="sect"><div class="sect-head">Hugging Face Hub <span class="crumb">optional — upload recordings, pull models</span></div><div class="panel pad">
        <div id="hub-status">checking…</div>
        <div class="form-grid" style="max-width:640px; margin-top:8px">
          <label class="field">access token <span class="crumb">huggingface.co → Settings → Access Tokens, type "write"</span>
            <input type="password" id="hub-token" placeholder="hf_…" autocomplete="off" /></label>
        </div>
        <div class="toolbar" style="margin-top:10px">
          <button id="hub-login" class="primary">Sign in</button>
          <button id="hub-logout">Sign out</button>
          <span class="save-note" id="hub-note"></span>
        </div>
        <div class="hint">The token is stored in <span class="mono">data/hf/token</span> inside this folder (ignored by git) and never in the rig file.</div>
        <div class="form-grid" style="max-width:640px; margin-top:14px">
          <label class="field">account to push under <span class="crumb">empty = signed-in account</span><input type="text" id="hub-username" value="${esc(hubCfg.username ?? "")}" /></label>
          <label class="field">recordings go to
            <select id="hub-datasets">${["both", "local", "hub"].map((v) => `<option value="${v}" ${hubCfg.datasets === v ? "selected" : ""}>${{ both: "this computer + Hub", local: "this computer only", hub: "Hub only (local copy removed after upload)" }[v]}</option>`).join("")}</select></label>
        </div>
        <label class="check"><input type="checkbox" id="hub-private" ${hubCfg.private === false ? "" : "checked"} /> keep uploaded datasets and models private</label>
        <div class="toolbar" style="margin-top:10px">
          <button id="hub-save" class="primary" ${c.found ? "" : "disabled"}>Save Hub settings</button>
          <span class="save-note" id="hub-save-note"></span>
        </div>
      </div></div>
      <div class="sect"><div class="sect-head">Arms <span class="crumb">edit via YAML below</span></div><div class="panel">
        ${Object.keys(c.arms || {}).length ? `<table><tr><th>name</th><th>role</th><th>side</th><th>type</th><th>gripper</th><th>CAN serial</th><th>calibrated</th><th>rest pose</th></tr>
          ${Object.entries(c.arms).map(([n, a]) => `<tr><td class="mono">${esc(n)}</td><td>${esc(a.role)}</td><td>${esc(a.side ?? "–")}</td>
            <td>${esc(a.arm_type)}</td><td>${esc(a.gripper)}</td><td class="mono">${esc(a.can_serial ?? a.can_iface ?? "–")}</td>
            <td>${a.gripper_limits ? "gripper" : "–"}</td><td>${a.rest_pose ? "stored" : "–"}</td></tr>`).join("")}</table>`
        : `<div class="empty">no arms — run <code>yamkit discover --write</code></div>`}
        <div class="hint">Left/right is a physical check: <code>yamkit read left_follower</code> (the arm stays free to move) and
          <code>yamkit swap left_follower right_follower</code> if it was the other one.</div>
      </div></div>
      <div class="sect"><div class="sect-head">Cameras</div><div class="panel">
        ${Object.keys(c.cameras || {}).length ? `<table><tr><th>name</th><th>camera</th><th>device</th><th class="num">resolution</th><th class="num">fps</th></tr>
          ${Object.entries(c.cameras).map(([n, cam]) => `<tr><td class="mono">${esc(n)}</td><td>${esc(cam.notes ?? cam.model ?? cam.type ?? "opencv")}</td>
            <td class="mono">${esc(String(cam.index_or_path ?? cam.serial_number_or_name ?? "–"))}</td>
            <td class="num">${cam.width && cam.height ? cam.width + "×" + cam.height : "–"}</td><td class="num">${cam.fps ?? "–"}</td></tr>`).join("")}</table>
          <div class="hint">Cameras are found by <code>yamkit discover --write</code> (re-run it after moving a camera to another USB port).
            Left and right wrist crossed? Run <code>yamkit swap left_wrist right_wrist</code>, or exchange the two
            <code>index_or_path</code> lines in the YAML below. Saving reloads the camera feeds.</div>`
        : `<div class="empty">no cameras configured — run <code>yamkit discover --write</code>, or add them under <code>cameras:</code> in the YAML below</div>`}
      </div></div>
      <div class="sect"><div class="sect-head">Raw YAML</div>
        <textarea class="yaml" id="cfg-yaml" spellcheck="false">${esc(c.yaml)}</textarea>
        <div class="toolbar" style="margin-top:10px">
          <button id="yaml-validate">Validate</button>
          <button id="yaml-save" class="primary">Save YAML</button>
          <span class="save-note" id="yaml-note"></span>
        </div>
        <div class="hint">The full rig file (arms, pairs, cameras, control). Saved verbatim after validation —
          comments and ordering are kept. The rig holds hardware identifiers only (no credentials).</div>
      </div>`;
    const note = (id, ok, msg) => {
      if (!body.isConnected) return;
      const n = $(id, body); n.className = "save-note " + (ok ? "ok" : "err"); n.textContent = msg;
    };
    const hubStatus = $("#hub-status", body);
    let hubRequest = 0;
    const showHub = async () => {
      const request = ++hubRequest;
      try {
        const h = await api("/hub");
        if (!body.isConnected || request !== hubRequest) return;
        hubStatus.innerHTML = !h.logged_in ? st(false, "not signed in", true)
          : h.online ? st(true, `signed in as ${esc(h.username)}`) : st(false, `token stored, but the Hub is unreachable: ${esc(h.error || "")}`, true);
      } catch (e) { if (body.isConnected && request === hubRequest) hubStatus.innerHTML = errBanner(e.message); }
    };
    showHub();
    $("#hub-login").onclick = async (e) => {
      const tokenInput = $("#hub-token", body);
      const token = tokenInput.value.trim();
      if (!token) return note("#hub-note", false, "paste a token first");
      e.target.disabled = true;
      try { const r = await post("/hub/login", { token }); tokenInput.value = ""; note("#hub-note", true, `signed in as ${r.username}`); if (body.isConnected) await showHub(); await refreshOverview(); }
      catch (err) { note("#hub-note", false, err.message); }
      finally { e.target.disabled = false; }
    };
    $("#hub-logout").onclick = async () => { try { await post("/hub/logout", {}); note("#hub-note", true, "signed out"); await showHub(); await refreshOverview(); } catch (err) { note("#hub-note", false, err.message); } };
    $("#hub-save").onclick = async (e) => {
      e.target.disabled = true;
      const hub = { username: $("#hub-username").value.trim(), datasets: $("#hub-datasets").value, private: $("#hub-private").checked };
      try { await post("/config", { hub }); note("#hub-save-note", true, "saved"); await refreshOverview(); }
      catch (err) { note("#hub-save-note", false, err.message); }
      finally { e.target.disabled = false; }
    };
    $("#ctl-save").onclick = async (e) => {
      const control = {};
      document.querySelectorAll("[data-ctl]").forEach((i) => { if (i.value !== "") control[i.dataset.ctl] = +i.value; });
      e.target.disabled = true;
      try { await post("/config", { control }); note("#ctl-note", true, "saved"); }
      catch (err) { note("#ctl-note", false, err.message); }
      finally { e.target.disabled = false; refreshOverview(); }
    };
    $("#yaml-validate").onclick = async () => {
      try { await post("/config", { yaml_text: $("#cfg-yaml").value, validate_only: true }); note("#yaml-note", true, "valid"); }
      catch (err) { note("#yaml-note", false, err.message); }
    };
    $("#yaml-save").onclick = async (e) => {
      e.target.disabled = true;
      try { await post("/config", { yaml_text: $("#cfg-yaml").value }); note("#yaml-note", true, "saved"); refreshOverview(); }
      catch (err) { note("#yaml-note", false, err.message); }
      finally { e.target.disabled = false; }
    };
  },
};

// -------------------------------------------------------------------------------- router ----
let current = null;
let pageCleanup = null;
function cleanupPage() {
  releaseCameraStreams();
  if (pageCleanup) pageCleanup();
  pageCleanup = null;
}
function route() {
  cleanupPage();
  let [page, ...args] = (location.hash.replace(/^#\//, "") || "live").split("/");
  if (page === "deployments") page = "inference"; // old links keep working
  const p = pages[page] || pages.live;
  current = p;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#/" + page));
  p.render($("#main"), args);
}
window.addEventListener("hashchange", route);

document.addEventListener("session", () => current?.update && current.update());
document.addEventListener("overview", () => (current === pages.live || current === pages.record) && current.update && current.update());

document.querySelectorAll("#theme-switch button").forEach((b) => {
  b.onclick = () => applyThemePref(b.dataset.themePref);
});

(async function init() {
  syncThemeButtons();
  await refreshOverview();
  await refreshSession();
  route();
  setInterval(refreshSession, 1000);
  setInterval(refreshOverview, 5000);
  setInterval(refreshCameras, 1000);
  setInterval(updateRolloutClocks, 100); // Display only; no HTTP, camera reads or phase transitions.
})();
