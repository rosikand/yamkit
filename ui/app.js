/* yamkit ui — vanilla JS single-page app. Talks only to /api/*; every hardware action is a
   POST that spawns the corresponding `yamkit` CLI on the host. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtBytes = (b) => b == null ? "–" : b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " kB";
const fmtDate = (t) => t == null ? "–" : new Date(t * 1000).toLocaleString();
const fmtDur = (s) => s == null ? "–" : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`;

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
async function refreshSession() {
  try { session = await api("/session"); } catch { /* server gone */ }
  updateSidebar();
  document.dispatchEvent(new CustomEvent("session"));
  // a session starting, ending or handing the cameras back changes what the tiles should show: refresh now
  const key = `${session.active}|${session.mode}|${session.parsed?.phase || ""}`;
  if (key !== lastSessionKey) { lastSessionKey = key; refreshOverview(); }
}
function updateSidebar() {
  const dot = $("#side-dot");
  dot.className = "dot " + (session.active ? "run" : "");
  $("#side-mode").textContent = (session.active ? session.mode : "idle") + (session.stopping && session.active ? " (stopping…)" : "");
  const hz = session.parsed && session.parsed.rate_hz;
  $("#side-hz").textContent = session.active && hz ? hz.toFixed(0) + " Hz" : "";
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

const camsBusy = () => overview?.cameras?.[0]?.suspended_by || "";
let camsRendered = null;
// The tiles are plain MJPEG <img> streams (/api/cameras/<name>/stream). They are re-rendered only when
// the camera list or its owner changes: a session that owns the devices replaces them with a
// placeholder, and they reconnect by themselves when it hands the cameras back.
function syncCams() {
  const slot = $("#cams-slot");
  if (!slot) return;
  const key = camsBusy() + "|" + (overview?.cameras || []).map((c) => c.name).join(",");
  if (key !== camsRendered) { slot.innerHTML = camsHTML(); }
}
function camsHTML() {
  const cams = cameraNames();
  const busy = camsBusy();
  camsRendered = busy + "|" + (overview?.cameras || []).map((c) => c.name).join(",");
  return `<div class="cams">` + cams.map((c) => `
    <div class="cam" data-cam="${esc(c.name)}">
      <span class="label">${esc(c.name)}</span>
      ${c.configured && !busy
        ? `<img src="/api/cameras/${encodeURIComponent(c.name)}/stream" alt="${esc(c.name)}"
             onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'placeholder',textContent:'no signal'}))" />`
        : `<div class="placeholder">${busy ? `in use by ${esc(busy)} session` : "no camera configured in rig.yaml"}</div>`}
    </div>`).join("") + `</div>`;
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
// LeRobot encodes the episode's videos after the phase ends without printing anything, so past the
// nominal duration the clock stops at the limit and the banner says "saving".
function phaseClock(total) {
  const t = session.phase_elapsed_s;
  if (t == null) return "";
  if (total && t > total + 1) return ` · ${Math.round(total)} s / ${Math.round(total)} s · saving episode…`;
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
  stop.textContent = session.stopping && session.active ? "Stopping… arms returning home — click again to release now" : "Stop";
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
        <button id="btn-park">Park arms</button>
        <button id="btn-stop" class="danger">Stop</button>
        <button id="btn-refresh">Refresh</button>`)}
      <div class="hint">The state stream runs <code>yamkit read</code>: arms connect in gravity-compensation
        mode (motors energised but compliant — nothing moves). Park runs <code>yamkit rest</code>: every arm
        moves slowly to its home pose and is released there.</div>
      <div class="sect"><div class="sect-head">Cameras</div><div id="cams-slot">${camsHTML()}</div></div>
      <div class="sect"><div class="sect-head">Arm state</div><div class="cols cols-2" id="arm-panels"></div></div>
      <div class="sect"><div class="sect-head">Status</div><div class="st-list" id="status-list"></div>
        <div class="hint" id="bringup"></div></div>
      <div class="sect"><div class="sect-head">Session output</div>${logPaneHTML()}</div>`;
    $("#btn-read").onclick = (e) => doPost("/session/read", { hz: 5 }, e.target);
    $("#btn-park").onclick = (e) => doPost("/session/rest", {}, e.target);
    $("#btn-stop").onclick = (e) => doPost("/session/stop", {}, e.target);
    $("#btn-refresh").onclick = () => { refreshOverview(); refreshSession(); };
    this.update();
  },
  update() {
    const panels = $("#arm-panels");
    if (!panels) return;
    syncRunButtons(["#btn-read", "#btn-park"], "#btn-stop");
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
      ${pageHead("Record", "teleoperation and dataset recording", `<button id="btn-park-rec">Park arms</button><button id="btn-stop-top" class="danger">Stop</button>`)}
      <div class="sect"><div class="sect-head">Cameras</div><div id="cams-slot">${camsHTML()}</div></div>
      <div class="cols cols-2">
        <div class="sect"><div class="sect-head">Teleop</div><div class="panel pad">
          <div id="teleop-ready"></div>
          <div id="teleop-status"></div>
          <div class="toolbar" style="margin-top:12px">
            <button id="btn-teleop" class="primary">Start Teleop</button>
          </div>
          <label class="check"><input type="checkbox" id="auto-engage" /> auto-engage (follower moves to leader pose immediately)</label>
          <div class="hint">Runs <code>yamkit teleop</code>. On Start every arm first moves slowly to its home pose.
            Without auto-engage, press the teaching-handle button to engage — the follower then moves to the leader pose.
            On Stop the arms return home before being released (let go of the handles; press Stop again to release immediately).</div>
        </div></div>
        <div class="sect"><div class="sect-head">Recording</div><div class="panel pad">
          <label class="field">dataset name<input type="text" id="rec-name" placeholder="pick_cube" /></label>
          <label class="field">task instruction<input type="text" id="rec-task" placeholder="pick up the red cube and place it in the bowl" /></label>
          <div class="form-grid">
            <label class="field">episodes<input type="number" id="rec-episodes" value="10" min="1" /></label>
            <label class="field">episode duration (s)<input type="number" id="rec-episode-s" value="30" /></label>
            <label class="field">reset duration (s)<input type="number" id="rec-reset-s" value="10" /></label>
          </div>
          <div class="field" style="margin-top:12px">save to
            <div class="checks">
              <label class="check"><input type="checkbox" id="rec-local" ${hubDefault !== "hub" ? "checked" : ""} /> this computer (data/datasets)</label>
              <label class="check"><input type="checkbox" id="rec-hub" ${hubDefault !== "local" ? "checked" : ""} ${hubReady ? "" : "disabled"} />
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
          <div class="hint">Runs <code>yamkit record</code> (LeRobot <code>lerobot-record</code>) →
            <code>data/datasets/&lt;name&gt;</code>. Arms engage and move with the leaders.</div>
        </div></div>
      </div>
      <div class="sect"><div class="sect-head">Progress</div><div class="st-list" id="rec-progress"></div></div>
      <div class="sect"><div class="sect-head">Output</div>${logPaneHTML()}</div>`;
    $("#btn-teleop").onclick = (e) => doPost("/session/teleop", { auto_engage: $("#auto-engage").checked }, e.target);
    $("#btn-record").onclick = (e) => {
      const name = $("#rec-name").value.trim(), task = $("#rec-task").value.trim();
      if (!name || !task) return alert("dataset name and task instruction are required");
      const toLocal = $("#rec-local").checked, toHub = $("#rec-hub").checked && !$("#rec-hub").disabled;
      if (!toLocal && !toHub) return alert("pick at least one place to save the recording");
      doPost("/session/record", {
        name, task, to: toLocal && toHub ? "both" : toHub ? "hub" : "local",
        episodes: +$("#rec-episodes").value || 10,
        episode_s: +$("#rec-episode-s").value || 30,
        reset_s: +$("#rec-reset-s").value || 10,
        fps: Math.min(Math.max(+$("#rec-fps").value || DEFAULT_FPS, 1), maxFps),
      }, e.target);
    };
    $("#btn-stop-top").onclick = (e) => doPost("/session/stop", {}, e.target);
    $("#btn-park-rec").onclick = (e) => doPost("/session/rest", {}, e.target);
    this.update();
  },
  update() {
    const ts = $("#teleop-status");
    if (!ts) return;
    syncRunButtons(["#btn-teleop", "#btn-record", "#btn-park-rec"], "#btn-stop-top");
    syncCams();
    // pair status is only meaningful while a teleop session is actually running
    const pairs = session.active && session.mode === "teleop" ? (session.parsed?.pairs || {}) : {};
    // readiness banner: arm connection takes a few seconds after Start — show when the
    // teleop loop is actually running (status lines flowing) and when a pair is engaged
    const ready = $("#teleop-ready");
    if (ready) {
      const p = session.parsed || {};
      let html = "";
      if (session.active && session.mode === "teleop") {
        const engaged = Object.values(pairs).some((x) => x.engaged);
        if (!p.rate_hz) html = `<div class="ready-banner setup"><span class="dot"></span>Setting up — connecting arms…</div>`;
        else if (engaged) html = `<div class="ready-banner engaged"><span class="dot"></span>Engaged — follower tracking leader</div>`;
        else html = `<div class="ready-banner ready">✓ Ready — press the teaching-handle button to engage</div>`;
      } else if (session.active && session.mode === "record") {
        if (p.episode == null && !p.phase) html = `<div class="ready-banner setup"><span class="dot"></span>Setting up recorder — loading LeRobot…</div>`;
        else if (p.phase === "reset") html = `<div class="ready-banner ready">✓ Reset — put the leaders back to home and reset the scene${phaseClock(session.meta?.reset_s)}</div>`;
        else if (p.phase === "upload") html = `<div class="ready-banner ready">✓ Recording finished — uploading to the Hub (arms are parked; feeds are back)${phaseClock()}</div>`;
        else html = `<div class="ready-banner engaged"><span class="dot"></span>Recording — episode ${(p.episode ?? 0) + 1}${session.meta?.episodes ? " of " + session.meta.episodes : ""}${phaseClock(session.meta?.episode_s)}</div>`;
      }
      ready.innerHTML = html;
    }
    ts.innerHTML = Object.keys(pairs).length
      ? Object.entries(pairs).map(([n, p]) => `<div class="st-list" style="margin:4px 0">
          ${st(p.engaged, `${n}: ${p.engaged ? "ENGAGED" : "idle"}`, !p.engaged)}
          <span class="mono" style="color:var(--muted)">err ${p.error_rad != null ? p.error_rad.toFixed(3) + " rad" : "–"} · grip ${p.gripper != null ? p.gripper.toFixed(2) : "–"}</span>
        </div>`).join("")
      : `<div class="hint">${session.active && session.mode === "teleop" ? "starting…" : "no teleop session"}</div>`;
    const prog = $("#rec-progress");
    const p = session.parsed || {}, meta = session.meta || {};
    const bits = [session.active ? stN(`${session.mode} running`, true) : stN("idle")];
    if (session.active || session.mode) bits.push(stN(`elapsed ${fmtDur(session.elapsed_s)}`));
    if (p.episode != null) bits.push(stN(`episode ${p.episode + 1}${meta.episodes ? " of " + meta.episodes : ""}`));
    if (p.phase) bits.push(stN(p.phase + (session.active && session.phase_elapsed_s != null ? ` ${fmtDur(session.phase_elapsed_s)}` : "")));
    if (p.rate_hz) bits.push(stN(`${p.rate_hz.toFixed(0)} Hz`));
    if (!session.active && session.returncode != null)
      bits.push(st(session.returncode === 0, `last run: exit ${session.returncode}`, session.returncode !== 0));
    prog.innerHTML = bits.join("");
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
  let d;
  try { d = await api(`/datasets/${encodeURIComponent(name)}`); }
  catch (e) { $("#ds-detail").innerHTML = errBanner(e.message); return; }
  const eps = d.episode_list || [];
  $("#ds-detail").innerHTML = `
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
  if (ep != null) renderEpisodeViewer($("#ep-viewer"), name, d, ep);
}

async function renderEpisodeViewer(el, name, detail, ep) {
  el.innerHTML = `<div class="sect"><div class="sect-head">Episode ${ep}</div><div id="ep-body">loading…</div></div>`;
  let s;
  try { s = await api(`/datasets/${encodeURIComponent(name)}/episodes/${ep}`); }
  catch (e) { $("#ep-body").innerHTML = errBanner(e.message); return; }
  const epMeta = (detail.episode_list || []).find((e) => e.episode_index === ep) || {};
  const cams = Object.keys(epMeta.videos || {});
  const t = s.timestamp || [];
  const t0 = t.length ? t[0] : 0, t1 = t.length ? t[t.length - 1] : 1;
  const colors = seriesColors();
  const body = $("#ep-body");
  body.innerHTML = `
    ${cams.length ? `<div class="cams" style="margin-bottom:12px">` + cams.map((c) => `
      <div class="cam"><span class="label">${esc(c)}</span>
        <video id="vid-${esc(c)}" src="/api/datasets/${encodeURIComponent(name)}/video/${encodeURIComponent(c)}/${ep}" muted playsinline></video></div>`).join("") + `</div>
      <div class="toolbar"><button id="ep-play" class="primary">Play</button><span class="hint" style="margin:0">videos + charts play in sync</span></div>`
      : `<div class="toolbar"><button id="ep-play" class="primary">Play</button>
         <span class="hint" style="margin:0">no videos in this dataset — playing sweeps the cursor over the state/action charts</span></div>`}
    <input type="range" id="ep-scrub" min="${t0}" max="${t1}" step="0.01" value="${t0}" />
    <div class="legend"><span><span class="k" style="background:${colors.state}"></span>observation.state</span>
      <span><span class="k" style="background:${colors.action}"></span>action</span></div>
    <div class="charts" id="ep-charts"></div>`;

  // small multiples: one panel per state dimension, state + action series
  const names = s.names || (s["observation.state"]?.[0] || []).map((_, i) => "dim_" + i);
  const chartsEl = $("#ep-charts");
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

  const scrub = $("#ep-scrub");
  const videos = cams.map((c) => ({ el: document.getElementById("vid-" + c), meta: epMeta.videos[c] }));
  const lead = videos[0];
  let playTimer = null;
  const stopPlay = () => { if (playTimer) { clearInterval(playTimer); playTimer = null; } videos.forEach((v) => v.el.pause()); $("#ep-play").textContent = "Play"; };
  scrub.oninput = () => {
    stopPlay();
    const tc = +scrub.value;
    setCursor(tc);
    videos.forEach((v) => { v.el.currentTime = (v.meta.from_timestamp || 0) + (tc - t0); });
  };
  $("#ep-play").onclick = () => {
    if (playTimer || (lead && !lead.el.paused)) return stopPlay();
    $("#ep-play").textContent = "Pause";
    if (lead) {
      videos.forEach((v) => { v.el.currentTime = (v.meta.from_timestamp || 0) + (+scrub.value - t0); v.el.play(); });
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
  new ResizeObserver(draw).observe(canvas);
  draw();
  return { setCursor(tc) { cursorT = tc; draw(); } };
}

// ---- inference (policy runs; backend routes remain /api/deployments) ----
pages.inference = {
  async render(el, args) {
    if (args.length) return renderRunDetail(el, decodeURIComponent(args[0]));
    this._submitted = null;
    el.innerHTML = `
      ${pageHead("Inference", "check, prepare, probe, then explicitly start a policy run")}
      <div class="sect"><div class="sect-head">Policy deployment</div><div class="panel pad">
        <div class="form-grid">
          <label class="field">preset<select id="inf-preset"><option value="smolvla">SmolVLA base · forward check</option><option value="molmoact2">MolmoAct2 · bimanual YAM</option><option value="pi05">pi05 base · forward check</option><option value="custom">Custom compatible local checkpoint</option></select></label>
          <label class="field">backend<select id="inf-backend"><option value="local">Local (default)</option><option value="modal">Modal GPU</option></select></label>
          <label class="field">checkpoint<input type="text" id="inf-policy" value="smolvla" list="policy-list" /></label>
          <label class="field">task<input type="text" id="inf-task" value="pick up the object" /></label>
          <label class="field">followers<select id="inf-arms"><option value="">Both arms</option><option value="left">Left only (compatible local model)</option><option value="right">Right only (compatible local model)</option></select></label>
          <label class="field">duration (seconds)<input type="number" id="inf-duration" value="60" min="1" max="3600" /></label>
          <label class="field">local device<select id="inf-device"><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">MPS</option></select></label>
          <label class="field">Modal GPU<select id="inf-gpu"><option value="L40S">L40S · one container</option></select></label>
        </div>
        <datalist id="policy-list"></datalist>
        <div class="toolbar"><label class="check"><input type="checkbox" id="inf-rtc" /> Local RTC (policy must support guidance)</label>
          <label class="check"><input type="checkbox" id="inf-crop" /> Optional center crop to 16:9 at Modal policy boundary</label></div>
        <div id="inf-profile-note" class="hint"></div>
        <div class="hint">Modal uses unguided background chunks. Crop stays off by default and does not restore training camera geometry. Recording camera settings stay unchanged.</div>
        <div class="toolbar" style="margin-top:12px">
          <button id="btn-pc">Check (no hardware)</button><button id="btn-prepare">Prepare Modal</button>
          <button id="btn-ro" class="danger">Start rollout</button><button id="btn-inf-stop" class="danger">Stop local execution</button>
          <button id="btn-cloud-stop">Shut down owned cloud service</button>
        </div>
        <div class="hint warn">Start rollout enables motors and moves the followers. Stop halts local execution; cloud shutdown is separate. Closing this browser is not Stop.</div>
        <div class="hint">Prepared Modal GPUs scale to zero after up to 300 seconds idle (development tests: 15 seconds). Idle warm time is billable. No permanent heartbeat.</div>
      </div></div>
      <div class="sect"><div class="sect-head">Action probe · never executes predicted positions</div><div class="panel pad">
        <label class="field">saved observation (.npz path inside this repository)<input type="text" id="inf-saved" placeholder="data/probes/observation.npz" /></label>
        <div class="toolbar"><button id="btn-probe-saved">Probe saved observation</button><button id="btn-probe-live">Probe live active read</button></div>
        <div class="hint warn">Live probe is GRAVITY-COMPENSATION ACTIVE READ: motors are active and this is not guaranteed motion-free. All gripper calibrations must be valid first. A successful probe never approves motion or replays its chunk.</div>
      </div></div>
      <div class="sect"><div class="sect-head">Operation</div><div id="inf-status" class="panel pad">No operation for this selection.</div><pre id="inf-result" class="log tall"></pre></div>
      <div class="sect"><div class="sect-head">Cameras</div><div id="cams-slot">${camsHTML()}</div></div>
      <div class="sect"><div class="sect-head">Runs</div><div id="run-list">loading…</div></div>`;
    this._profiles = [];
    api("/inference/profiles").then((data) => { this._profiles = data.profiles; this.syncForm(); }).catch(() => {});
    api("/models").then((list) => {
      const dl = $("#policy-list");
      if (dl) dl.innerHTML = list.map((m) => `<option value="${esc(m.where === "cloud" ? m.repo_id : "outputs/" + m.path)}">${esc(m.policy_type ?? "")}</option>`).join("");
    }).catch(() => {});
    $("#inf-preset").onchange = () => { $("#inf-policy").value = $("#inf-preset").value === "custom" ? "" : $("#inf-preset").value; this.syncForm(); };
    ["inf-backend", "inf-policy", "inf-task", "inf-arms", "inf-duration", "inf-device", "inf-gpu", "inf-rtc", "inf-crop", "inf-saved"].forEach((id) => {
      document.getElementById(id).addEventListener("input", () => this.syncForm());
    });
    $("#btn-pc").onclick = (e) => this.launch("/session/policy-check", {}, e.target);
    $("#btn-prepare").onclick = (e) => this.launch("/session/modal-prepare", {}, e.target);
    $("#btn-ro").onclick = (e) => {
      if (!confirm("Start rollout? Motors will be enabled and the follower arms WILL move. Clear the workspace and supervise the run.")) return;
      this.launch("/session/rollout", { confirm_motion: true }, e.target);
    };
    $("#btn-probe-saved").onclick = (e) => this.launch("/session/policy-probe", { saved: $("#inf-saved").value.trim() }, e.target);
    $("#btn-probe-live").onclick = (e) => {
      if (!confirm("Approve GRAVITY-COMPENSATION ACTIVE READ? Motors will be active. This is not guaranteed motion-free. No predicted position will be executed.")) return;
      this.launch("/session/policy-probe", { live: true, confirm_active_read: true }, e.target);
    };
    $("#btn-inf-stop").onclick = (e) => doPost("/session/stop", {}, e.target);
    $("#btn-cloud-stop").onclick = (e) => doPost("/session/modal-shutdown", {}, e.target);
    this.syncForm();
    this.refreshList();
  },
  selection() {
    return { policy: $("#inf-policy").value.trim(), task: $("#inf-task").value.trim(),
      backend: $("#inf-backend").value, device: $("#inf-device").value, gpu: $("#inf-gpu").value,
      duration: Number($("#inf-duration").value), fps: 30, rtc: $("#inf-rtc").checked,
      center_crop: $("#inf-crop").checked, async_chunks: true,
      arms: $("#inf-arms").value ? [$("#inf-arms").value] : null };
  },
  syncForm() {
    if (!$("#inf-policy")) return;
    const modal = $("#inf-backend").value === "modal";
    $("#inf-device").disabled = modal;
    $("#inf-gpu").disabled = !modal;
    $("#inf-rtc").disabled = modal;
    if (modal) $("#inf-rtc").checked = false;
    $("#inf-crop").disabled = !modal;
    if (!modal) $("#inf-crop").checked = false;
    const selected = this.selection();
    const profile = this._profiles.find((p) => p.id === selected.policy || p.repo_id === selected.policy);
    $("#inf-profile-note").textContent = profile ? `Revision ${profile.revision}. ${profile.mapping_note}` : "Custom checkpoints use the existing local LeRobot path; verify their rig compatibility before motion.";
    ["btn-pc", "btn-prepare", "btn-ro", "btn-probe-saved", "btn-probe-live", "btn-cloud-stop"].forEach((id) => { document.getElementById(id).disabled = session.active || this._launching; });
    $("#btn-prepare").disabled ||= !modal;
    $("#btn-ro").disabled ||= !!profile && (!profile.mapping_verified || (!modal && profile.id === "molmoact2"));
    $("#btn-inf-stop").disabled = !session.active;
    const submitted = this._submitted;
    const matches = submitted && submitted.selection === JSON.stringify(selected) && submitted.saved === $("#inf-saved").value && submitted.id === session.meta?.operation_id;
    $("#inf-status").textContent = matches ? `${session.mode}: ${session.active ? (session.stopping ? "stopping local process…" : "running…") : session.returncode === 0 ? "completed" : "failed or stopped"} · operation ${submitted.id}` : "No completed operation for this selection. Changing options invalidates the displayed readiness result.";
    $("#inf-result").textContent = matches ? (session.parsed?.result ? JSON.stringify(session.parsed.result, null, 2) : (session.log || []).join("\n")) : "";
    syncCams();
  },
  async launch(path, extra, button) {
    if (this._launching || session.active) return;
    const selected = this.selection();
    this._launching = true;
    this.syncForm();
    try {
      const result = await post(path, { ...selected, ...extra });
      this._submitted = { id: result.meta?.operation_id, selection: JSON.stringify(selected), saved: $("#inf-saved").value };
      await refreshSession();
    } catch (e) { alert(e.message); }
    finally { this._launching = false; this.syncForm(); }
  },
  async refreshList() {
    const el = $("#run-list");
    if (!el) return;
    try {
      const list = await api("/deployments");
      el.innerHTML = `<div class="panel">` + (list.length ? `<table><tr><th>run</th><th>kind</th><th>model</th><th>task</th><th class="num">latency</th><th class="num">duration</th><th>status</th><th>termination</th></tr>` +
        list.map((d) => `<tr class="click" onclick="location.hash='#/inference/${encodeURIComponent(d.id)}'">
          <td class="mono">${esc(d.id)}</td><td>${esc(d.kind ?? "–")}</td><td class="mono">${esc(d.policy ?? "–")}</td><td>${esc(d.task ?? "–")}</td>
          <td class="num">${d.first_call_ms != null ? d.first_call_ms.toFixed(0) + " ms" : "–"}</td>
          <td class="num">${fmtDur(d.duration_s)}</td>
          <td>${st(d.status === "success", d.status ?? "?", d.status === "running" || d.status === "stopped")}</td>
          <td>${esc(d.termination ?? "–")}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no policy runs yet — run a policy check or rollout above</div>`) + `</div>`;
    } catch (e) { el.innerHTML = errBanner(e.message); }
  },
  update() {  // re-fetch only when a session starts/ends, not on every poll
    this.syncForm();
    if (this._wasActive !== session.active) { this._wasActive = session.active; this.refreshList(); }
  },
};

async function renderRunDetail(el, id) {
  el.innerHTML = `<a class="back" href="#/inference">← inference</a>${pageHead(id)}<div id="run-detail">loading…</div>`;
  let d;
  try { d = await api(`/deployments/${encodeURIComponent(id)}`); }
  catch (e) { $("#run-detail").innerHTML = errBanner(e.message); return; }
  $("#run-detail").innerHTML = `
    <div class="sect"><div class="kv panel">
      <div>status</div><div>${st(d.status === "success", d.status, d.status !== "failed")}${d.termination ? ` <span class="crumb">— ${esc(d.termination)}</span>` : ""}</div>
      <div>kind</div><div>${esc(d.kind)}</div>
      <div>model</div><div class="mono">${esc(d.policy ?? "–")}</div>
      <div>task</div><div>${esc(d.task ?? "–")}</div>
      <div>started</div><div>${fmtDate(d.started_at)}</div>
      <div>duration</div><div>${fmtDur(d.duration_s)}</div>
      <div>latency (first call)</div><div>${d.first_call_ms != null ? d.first_call_ms.toFixed(0) + " ms" : "–"}</div>
      <div>latency (next calls)</div><div>${d.step_call_ms ? d.step_call_ms.map((x) => x.toFixed(0)).join(" / ") + " ms" : "–"}</div>
      <div>exit code</div><div class="mono">${d.returncode ?? "–"}</div>
    </div></div>
    ${(d.videos || []).length ? `<div class="sect"><div class="sect-head">Replay</div><div class="cams">` + d.videos.map((v) => `
      <div class="cam"><span class="label">${esc(v)}</span>
        <video controls src="/api/deployments/${encodeURIComponent(id)}/video/${encodeURIComponent(v)}"></video></div>`).join("") + `</div></div>` : ""}
    <div class="sect"><div class="sect-head">Log</div><pre class="log tall">${esc((d.log || []).join("\n")) || "(empty)"}</pre></div>`;
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
    $("#cfg-reload").onclick = () => this.render(el);
    let c;
    try { c = await api("/config"); }
    catch (e) { $("#cfg-body").innerHTML = errBanner(e.message); return; }
    this.cfg = c;
    const ctl = c.control || {};
    const hubCfg = c.hub || {};
    const ctlFields = [
      ["teleop_hz", "teleop loop rate (Hz)"], ["sync_seconds", "engage sync move (s)"],
      ["bilateral_kp", "bilateral force-feedback gain"], ["engage_button", "engage button index"],
      ["max_joint_speed", "max joint speed (rad/s)"], ["max_gripper_speed", "max gripper speed (1/s)"],
      ["home_speed", "return-to-home speed, followers (rad/s, 0 = off)"], ["leader_home_speed", "return-to-home speed, leaders (rad/s)"],
    ];
    $("#cfg-body").innerHTML = `
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
    const note = (id, ok, msg) => { const n = $(id); n.className = "save-note " + (ok ? "ok" : "err"); n.textContent = msg; };
    const showHub = async () => {
      try {
        const h = await api("/hub");
        $("#hub-status").innerHTML = !h.logged_in ? st(false, "not signed in", true)
          : h.online ? st(true, `signed in as ${esc(h.username)}`) : st(false, `token stored, but the Hub is unreachable: ${esc(h.error || "")}`, true);
      } catch (e) { $("#hub-status").innerHTML = errBanner(e.message); }
    };
    showHub();
    $("#hub-login").onclick = async (e) => {
      const token = $("#hub-token").value.trim();
      if (!token) return note("#hub-note", false, "paste a token first");
      e.target.disabled = true;
      try { const r = await post("/hub/login", { token }); $("#hub-token").value = ""; note("#hub-note", true, `signed in as ${r.username}`); await showHub(); await refreshOverview(); }
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
function route() {
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
})();
