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
async function refreshSession() {
  try { session = await api("/session"); } catch { /* server gone */ }
  updateSidebar();
  document.dispatchEvent(new CustomEvent("session"));
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

function camsHTML() {
  const cams = cameraNames();
  const busy = overview?.cameras?.[0]?.suspended_by;
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

// show Start buttons only when idle, Stop only while a session runs
function syncRunButtons(startIds, stopId) {
  const stop = $(stopId);
  if (!stop) return;
  for (const id of startIds) {
    const b = $(id);
    if (b) b.hidden = session.active;
  }
  stop.hidden = !session.active;
  stop.disabled = !!(session.stopping && session.active);
  stop.textContent = session.stopping && session.active ? "Stopping…" : "Stop";
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
      <div class="sect"><div class="sect-head">Cameras</div>${camsHTML()}</div>
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
pages.record = {
  render(el) {
    el.innerHTML = `
      ${pageHead("Record", "teleoperation and dataset recording", `<button id="btn-stop-top" class="danger">Stop</button>`)}
      <div class="sect"><div class="sect-head">Cameras</div>${camsHTML()}</div>
      <div class="cols cols-2">
        <div class="sect"><div class="sect-head">Teleop</div><div class="panel pad">
          <div id="teleop-ready"></div>
          <div id="teleop-status"></div>
          <div class="toolbar" style="margin-top:12px">
            <button id="btn-teleop" class="primary">Start Teleop</button>
          </div>
          <label class="check"><input type="checkbox" id="auto-engage" /> auto-engage (follower moves to leader pose immediately)</label>
          <div class="hint">Runs <code>yamkit teleop</code>. Without auto-engage, press the teaching-handle
            button to engage — the follower then moves to the leader pose.</div>
        </div></div>
        <div class="sect"><div class="sect-head">Recording</div><div class="panel pad">
          <label class="field">dataset name<input type="text" id="rec-name" placeholder="pick_cube" /></label>
          <label class="field">task instruction<input type="text" id="rec-task" placeholder="pick up the red cube and place it in the bowl" /></label>
          <div class="form-grid">
            <label class="field">episodes<input type="number" id="rec-episodes" value="10" min="1" /></label>
            <label class="field">fps<input type="number" id="rec-fps" value="30" min="1" /></label>
            <label class="field">episode duration (s)<input type="number" id="rec-episode-s" value="30" /></label>
            <label class="field">reset duration (s)<input type="number" id="rec-reset-s" value="10" /></label>
          </div>
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
      doPost("/session/record", {
        name, task,
        episodes: +$("#rec-episodes").value || 10,
        episode_s: +$("#rec-episode-s").value || 30,
        reset_s: +$("#rec-reset-s").value || 10,
        fps: +$("#rec-fps").value || 30,
      }, e.target);
    };
    $("#btn-stop-top").onclick = (e) => doPost("/session/stop", {}, e.target);
    this.update();
  },
  update() {
    const ts = $("#teleop-status");
    if (!ts) return;
    syncRunButtons(["#btn-teleop", "#btn-record"], "#btn-stop-top");
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
        else if (p.phase === "reset") html = `<div class="ready-banner ready">✓ Reset — reposition the scene for the next episode</div>`;
        else html = `<div class="ready-banner engaged"><span class="dot"></span>Recording — episode ${p.episode ?? 0}${session.meta?.episodes ? " / " + session.meta.episodes : ""}</div>`;
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
    if (p.episode != null) bits.push(stN(`episode ${p.episode}${meta.episodes ? " / " + meta.episodes : ""}`));
    if (p.phase) bits.push(stN(p.phase));
    if (p.rate_hz) bits.push(stN(`${p.rate_hz.toFixed(0)} Hz`));
    if (!session.active && session.returncode != null)
      bits.push(st(session.returncode === 0, `last run: exit ${session.returncode}`, session.returncode !== 0));
    prog.innerHTML = bits.join("");
    fillLog();
  },
};

// ---- datasets ----
pages.datasets = {
  async render(el, args) {
    if (args.length) return renderDatasetDetail(el, decodeURIComponent(args[0]), args[1]);
    el.innerHTML = `${pageHead("Datasets", "<span class='mono'>data/datasets/</span>")}<div id="ds-list" class="sect">loading…</div>`;
    try {
      const list = await api("/datasets");
      $("#ds-list").innerHTML = `<div class="panel">` + (list.length ? `<table><tr><th>name</th><th class="num">episodes</th><th class="num">frames</th><th class="num">fps</th><th>robot</th><th>tasks</th><th>cameras</th><th class="num">size</th></tr>` +
        list.map((d) => `<tr class="click" onclick="location.hash='#/datasets/${encodeURIComponent(d.name)}'">
          <td class="mono">${esc(d.name)}</td><td class="num">${d.episodes ?? "–"}</td><td class="num">${d.frames ?? "–"}</td><td class="num">${d.fps ?? "–"}</td>
          <td>${esc(d.robot_type ?? "–")}</td><td>${esc((d.tasks || []).join("; ") || "–")}</td>
          <td class="mono">${esc((d.cameras || []).join(", ") || "none")}</td><td class="num">${fmtBytes(d.size_bytes)}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no datasets yet — record one from the Record page</div>`) + `</div>`;
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
    el.innerHTML = `
      ${pageHead("Inference", "policy runs launched from this UI")}
      <div class="cols cols-2">
        <div class="sect"><div class="sect-head">Policy check <span class="crumb">safe — no arm is energised</span></div><div class="panel pad">
          <label class="field" style="margin-top:0">policy (checkpoint dir or HF id)<input type="text" id="pc-policy" placeholder="outputs/train/…/pretrained_model or lerobot/smolvla_base" /></label>
          <label class="field">task<input type="text" id="pc-task" value="pick up the object" /></label>
          <div class="toolbar" style="margin-top:12px"><button id="btn-pc" class="primary">Run policy check</button></div>
        </div></div>
        <div class="sect"><div class="sect-head">Rollout <span class="crumb">moves the arms</span></div><div class="panel pad">
          <label class="field" style="margin-top:0">policy<input type="text" id="ro-policy" /></label>
          <label class="field">task<input type="text" id="ro-task" /></label>
          <div class="form-grid">
            <label class="field">duration (s)<input type="number" id="ro-duration" value="60" /></label>
            <label class="check" style="margin-top:30px"><input type="checkbox" id="ro-rtc" checked /> RTC inference</label>
          </div>
          <div class="toolbar" style="margin-top:12px"><button id="btn-ro" class="danger">Start rollout</button></div>
          <div class="hint warn">Runs <code>yamkit rollout</code>: the follower arms will move. Clear the workspace first.</div>
        </div></div>
      </div>
      <div class="sect"><div class="sect-head">Runs</div><div id="run-list">loading…</div></div>`;
    $("#btn-pc").onclick = (e) => {
      const policy = $("#pc-policy").value.trim();
      if (!policy) return alert("policy is required");
      doPost("/session/policy-check", { policy, task: $("#pc-task").value }, e.target).then(() => this.refreshList());
    };
    $("#btn-ro").onclick = (e) => {
      const policy = $("#ro-policy").value.trim(), task = $("#ro-task").value.trim();
      if (!policy || !task) return alert("policy and task are required");
      if (!confirm("Start rollout? The follower arms WILL move.")) return;
      doPost("/session/rollout", { policy, task, duration: +$("#ro-duration").value || 60, rtc: $("#ro-rtc").checked }, e.target).then(() => this.refreshList());
    };
    this.refreshList();
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
pages.models = {
  async render(el, args) {
    if (args.length) return renderModelDetail(el, decodeURIComponent(args[0]));
    el.innerHTML = `${pageHead("Models", "checkpoints under <span class='mono'>outputs/</span>")}<div id="model-list" class="sect">loading…</div>`;
    try {
      const list = await api("/models");
      $("#model-list").innerHTML = `<div class="panel">` + (list.length ? `<table><tr><th>path</th><th>type</th><th class="num">steps</th><th>dataset</th><th class="num">size</th><th>modified</th></tr>` +
        list.map((m) => `<tr class="click" onclick="location.hash='#/models/${encodeURIComponent(m.path)}'">
          <td class="mono">outputs/${esc(m.path)}</td><td>${esc(m.policy_type ?? "?")}</td>
          <td class="num">${m.steps ?? "–"}</td><td class="mono">${esc(m.dataset ?? "–")}</td>
          <td class="num">${fmtBytes(m.size_bytes)}</td><td>${fmtDate(m.modified)}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no checkpoints under outputs/ — see README §6 for training</div>`) + `</div>`;
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
    const ctlFields = [
      ["teleop_hz", "teleop loop rate (Hz)"], ["sync_seconds", "engage sync move (s)"],
      ["bilateral_kp", "bilateral force-feedback gain"], ["engage_button", "engage button index"],
      ["max_joint_speed", "max joint speed (rad/s)"], ["max_gripper_speed", "max gripper speed (1/s)"],
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
      <div class="sect"><div class="sect-head">Arms <span class="crumb">edit via YAML below</span></div><div class="panel">
        ${Object.keys(c.arms || {}).length ? `<table><tr><th>name</th><th>role</th><th>side</th><th>type</th><th>gripper</th><th>CAN serial</th><th>calibrated</th><th>rest pose</th></tr>
          ${Object.entries(c.arms).map(([n, a]) => `<tr><td class="mono">${esc(n)}</td><td>${esc(a.role)}</td><td>${esc(a.side ?? "–")}</td>
            <td>${esc(a.arm_type)}</td><td>${esc(a.gripper)}</td><td class="mono">${esc(a.can_serial ?? a.can_iface ?? "–")}</td>
            <td>${a.gripper_limits ? "gripper" : "–"}</td><td>${a.rest_pose ? "stored" : "–"}</td></tr>`).join("")}</table>`
        : `<div class="empty">no arms — run <code>yamkit discover --write</code></div>`}
      </div></div>
      <div class="sect"><div class="sect-head">Cameras</div><div class="panel">
        ${Object.keys(c.cameras || {}).length ? `<table><tr><th>name</th><th>type</th><th>device</th><th class="num">resolution</th><th class="num">fps</th></tr>
          ${Object.entries(c.cameras).map(([n, cam]) => `<tr><td class="mono">${esc(n)}</td><td>${esc(cam.type ?? "opencv")}</td>
            <td class="mono">${esc(String(cam.index_or_path ?? cam.serial_number_or_name ?? "–"))}</td>
            <td class="num">${cam.width && cam.height ? cam.width + "×" + cam.height : "–"}</td><td class="num">${cam.fps ?? "–"}</td></tr>`).join("")}</table>`
        : `<div class="empty">no cameras configured — add them under <code>cameras:</code> in the YAML below</div>`}
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
document.addEventListener("overview", () => current === pages.live && current.update && current.update());

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
