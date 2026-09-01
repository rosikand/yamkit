/* yamkit ui — vanilla JS single-page app. Talks only to /api/*; every hardware action is a
   POST that spawns the corresponding `yamkit` CLI on the host. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtBytes = (b) => b == null ? "–" : b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " kB";
const fmtDate = (t) => t == null ? "–" : new Date(t * 1000).toLocaleString();
const fmtDur = (s) => s == null ? "–" : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`;
const CHART_COLORS = { state: getComputedStyle(document.documentElement).getPropertyValue("--series-state").trim() || "#3987e5",
                       action: getComputedStyle(document.documentElement).getPropertyValue("--series-action").trim() || "#d95926" };

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

// ---------------------------------------------------------------- global state + pollers ----
let overview = null;
let session = { active: false, mode: null, parsed: {}, log: [] };
let sessionTimer = null, overviewTimer = null;

async function refreshOverview() {
  try { overview = await api("/overview"); } catch { overview = null; }
  updateHeader();
  document.dispatchEvent(new CustomEvent("overview"));
}
async function refreshSession() {
  try { session = await api("/session"); } catch { /* server gone */ }
  updateHeader();
  document.dispatchEvent(new CustomEvent("session"));
}
function updateHeader() {
  const chip = $("#mode-chip"), fps = $("#fps-chip");
  const mode = session.active ? session.mode : "idle";
  $("#mode-text").textContent = mode + (session.stopping ? " (stopping…)" : "");
  chip.className = "chip " + (session.active ? "run" : "");
  const hz = session.parsed && session.parsed.rate_hz;
  fps.hidden = !(session.active && hz);
  if (hz) $("#fps-text").textContent = hz.toFixed(0) + " Hz";
}

// -------------------------------------------------------------------------- shared views ----
function errBanner(msg) { return `<div class="error-banner">${esc(msg)}</div>`; }

function statusChip(ok, label, warn = false) {
  return `<span class="chip ${ok ? "ok" : warn ? "warn" : "err"}"><span class="dot"></span>${esc(label)}</span>`;
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

function camsRowHTML() {
  const cams = cameraNames();
  const busy = overview?.cameras?.[0]?.suspended_by;
  return `<div class="grid cols-3 section">` + cams.map((c) => `
    <div class="cam" data-cam="${esc(c.name)}">
      <span class="label">${esc(c.name)}</span>
      ${c.configured && !busy
        ? `<img src="/api/cameras/${encodeURIComponent(c.name)}/stream" alt="${esc(c.name)}"
             onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'placeholder',textContent:'no signal'}))" />`
        : `<div class="placeholder">${busy ? `in use by ${esc(busy)} session` : "no camera configured in rig.yaml"}</div>`}
    </div>`).join("") + `</div>`;
}

function jointCardHTML(armName, st) {
  const range = Math.PI; // display range ±π rad
  const rows = (st?.q || []).map((v, i) => {
    const frac = Math.max(-1, Math.min(1, v / range));
    const left = frac < 0 ? 50 + frac * 50 : 50, width = Math.abs(frac) * 50;
    return `<div class="joint"><span class="name">joint_${i + 1}</span>
      <span class="track"><span class="mid"></span><span class="fill" style="left:${left}%;width:${Math.max(width, 0.7)}%"></span></span>
      <span class="val mono">${v.toFixed(3)}</span></div>`;
  }).join("");
  const grip = st?.gripper;
  const gripRow = `<div class="joint"><span class="name">gripper</span>
      <span class="track"><span class="fill" style="left:0;width:${grip != null ? grip * 100 : 0}%"></span></span>
      <span class="val mono">${grip != null ? grip.toFixed(2) : "–"}</span></div>`;
  return `<div class="card"><h3>${esc(armName)}</h3>
    ${st ? rows + gripRow : `<div class="empty">no state — start the state stream or teleop</div>`}</div>`;
}

function logPaneHTML(id = "log") { return `<pre class="log" id="${id}"></pre>`; }
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

pages.live = {
  title: "Live",
  render(el) {
    el.innerHTML = `
      ${camsRowHTML()}
      <div class="grid cols-2 section" id="arm-cards"></div>
      <div class="grid cols-2 section">
        <div class="card"><h3>Status</h3><div id="status-chips" class="row"></div>
          <div class="hint" id="bringup"></div></div>
        <div class="card"><h3>Read-only controls</h3>
          <div class="row">
            <button id="btn-read" class="primary">Start state stream</button>
            <button id="btn-stop" class="danger">Stop</button>
            <button id="btn-refresh">Refresh</button>
          </div>
          <div class="hint">The state stream runs <code>yamkit read</code>: arms connect in
          gravity-compensation mode (motors energised but compliant — nothing moves). Opening this
          page on its own never touches the arms.</div>
        </div>
      </div>
      <div class="card section"><h3>Session output</h3>${logPaneHTML()}</div>`;
    $("#btn-read").onclick = (e) => doPost("/session/read", { hz: 5 }, e.target);
    $("#btn-stop").onclick = (e) => doPost("/session/stop", {}, e.target);
    $("#btn-refresh").onclick = () => { refreshOverview(); refreshSession(); };
    this.update();
  },
  update() {
    const cards = $("#arm-cards");
    if (!cards) return;
    const followers = Object.entries(overview?.rig?.arms || {}).filter(([, a]) => a.role === "follower").map(([n]) => n);
    const names = followers.length ? followers : Object.keys(session.parsed?.arms || {});
    cards.innerHTML = names.map((n) => jointCardHTML(n, session.parsed?.arms?.[n])).join("") ||
      `<div class="card"><div class="empty">no rig file — run <code>yamkit discover --write</code></div></div>`;
    const chips = $("#status-chips");
    if (chips) {
      const can = (overview?.can || []).map((i) => statusChip(i.up, `${i.name} ${i.up ? "UP" : "DOWN"}${i.in_rig ? "" : " (not in rig)"}`));
      if (!can.length) can.push(statusChip(false, "no CAN adapters", true));
      const cams = (overview?.cameras || []).map((c) => statusChip(c.streaming && !c.error, `cam ${c.name}${c.error ? ": " + c.error : ""}`, !c.error));
      if (!cams.length) cams.push(statusChip(false, "no cameras in rig", true));
      const rig = overview?.rig?.found
        ? statusChip(!(overview.rig.problems || []).length, `rig: ${Object.keys(overview.rig.arms).length} arms, ${overview.rig.pairs.length} pairs`)
        : statusChip(false, "rig file missing");
      const mode = statusChip(true, session.active ? `mode: ${session.mode}` : "mode: idle");
      chips.innerHTML = [rig, mode, ...can, ...cams].join(" ");
      $("#bringup").innerHTML = (overview?.can_bringup || []).length
        ? "bring interfaces up: <code>" + overview.can_bringup.map(esc).join("</code> · <code>") + "</code>" : "";
    }
    fillLog();
  },
};

pages.record = {
  title: "Record",
  render(el) {
    el.innerHTML = `
      ${camsRowHTML()}
      <div class="grid cols-2 section">
        <div class="card"><h3>Teleop</h3>
          <div id="teleop-status"></div>
          <div class="row" style="margin-top:10px">
            <button id="btn-teleop" class="primary">Start Teleop</button>
            <button id="btn-stop1" class="danger">Stop</button>
          </div>
          <label class="check"><input type="checkbox" id="auto-engage" /> auto-engage (follower moves to leader pose immediately)</label>
          <div class="hint">Runs <code>yamkit teleop</code>. Without auto-engage, press the teaching-handle
          button to engage — the follower then moves to the leader pose.</div>
        </div>
        <div class="card"><h3>Recording</h3>
          <label class="field">dataset name<input type="text" id="rec-name" placeholder="pick_cube" /></label>
          <label class="field">task instruction<input type="text" id="rec-task" placeholder="pick up the red cube and place it in the bowl" /></label>
          <div class="grid cols-2">
            <label class="field">episodes<input type="number" id="rec-episodes" value="10" min="1" /></label>
            <label class="field">fps<input type="number" id="rec-fps" value="30" min="1" /></label>
            <label class="field">episode duration (s)<input type="number" id="rec-episode-s" value="30" /></label>
            <label class="field">reset duration (s)<input type="number" id="rec-reset-s" value="10" /></label>
          </div>
          <div class="row">
            <button id="btn-record" class="primary">Start Recording</button>
            <button id="btn-stop2" class="danger">Stop</button>
          </div>
          <div class="hint">Runs <code>yamkit record</code> (LeRobot <code>lerobot-record</code>) →
          <code>data/datasets/&lt;name&gt;</code>. Arms engage and move with the leaders.</div>
        </div>
      </div>
      <div class="card section"><h3>Progress</h3><div id="rec-progress" class="row"></div></div>
      <div class="card section"><h3>Output</h3>${logPaneHTML()}</div>`;
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
    $("#btn-stop1").onclick = $("#btn-stop2").onclick = (e) => doPost("/session/stop", {}, e.target);
    this.update();
  },
  update() {
    const ts = $("#teleop-status");
    if (!ts) return;
    const pairs = session.parsed?.pairs || {};
    ts.innerHTML = Object.keys(pairs).length
      ? Object.entries(pairs).map(([n, p]) => `<div class="row" style="margin:4px 0">
          ${statusChip(p.engaged, `${n}: ${p.engaged ? "ENGAGED" : "idle"}`, !p.engaged)}
          <span class="mono" style="color:var(--text-muted)">err ${p.error_rad != null ? p.error_rad.toFixed(3) + " rad" : "–"} · grip ${p.gripper != null ? p.gripper.toFixed(2) : "–"}</span>
        </div>`).join("")
      : `<div class="hint">${session.active && session.mode === "teleop" ? "starting…" : "no teleop session"}</div>`;
    const prog = $("#rec-progress");
    const p = session.parsed || {}, meta = session.meta || {};
    const bits = [];
    bits.push(statusChip(session.active, session.active ? `${session.mode} running` : "idle", !session.active));
    if (session.active || session.mode) bits.push(`<span class="chip">elapsed ${fmtDur(session.elapsed_s)}</span>`);
    if (p.episode != null) bits.push(`<span class="chip">episode ${p.episode}${meta.episodes ? " / " + meta.episodes : ""}</span>`);
    if (p.phase) bits.push(`<span class="chip">${esc(p.phase)}</span>`);
    if (p.rate_hz) bits.push(`<span class="chip">${p.rate_hz.toFixed(0)} Hz</span>`);
    if (!session.active && session.returncode != null)
      bits.push(statusChip(session.returncode === 0, `last run: exit ${session.returncode}`, session.returncode !== 0));
    prog.innerHTML = bits.join(" ");
    fillLog();
  },
};

// ---- datasets ----
pages.datasets = {
  title: "Datasets",
  async render(el, args) {
    if (args.length) return renderDatasetDetail(el, decodeURIComponent(args[0]), args[1]);
    el.innerHTML = `<h2>Datasets <span class="hint" style="display:inline">data/datasets/</span></h2><div class="card" id="ds-list">loading…</div>`;
    try {
      const list = await api("/datasets");
      $("#ds-list").innerHTML = list.length ? `<table><tr><th>name</th><th>episodes</th><th>frames</th><th>fps</th><th>robot</th><th>tasks</th><th>cameras</th><th>size</th></tr>` +
        list.map((d) => `<tr class="click" onclick="location.hash='#/datasets/${encodeURIComponent(d.name)}'">
          <td>${esc(d.name)}</td><td>${d.episodes ?? "–"}</td><td>${d.frames ?? "–"}</td><td>${d.fps ?? "–"}</td>
          <td>${esc(d.robot_type ?? "–")}</td><td>${esc((d.tasks || []).join("; ") || "–")}</td>
          <td>${esc((d.cameras || []).join(", ") || "none")}</td><td>${fmtBytes(d.size_bytes)}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no datasets yet — record one from the Record page</div>`;
    } catch (e) { $("#ds-list").innerHTML = errBanner(e.message); }
  },
};

async function renderDatasetDetail(el, name, epArg) {
  el.innerHTML = `<a class="back" href="#/datasets">← datasets</a><h2>${esc(name)}</h2><div id="ds-detail">loading…</div>`;
  let d;
  try { d = await api(`/datasets/${encodeURIComponent(name)}`); }
  catch (e) { $("#ds-detail").innerHTML = errBanner(e.message); return; }
  const chips = [
    `<span class="chip">${d.episodes} episodes</span>`, `<span class="chip">${d.frames} frames</span>`,
    `<span class="chip">${d.fps} fps</span>`, `<span class="chip">${esc(d.robot_type || "?")}</span>`,
    `<span class="chip">${fmtBytes(d.size_bytes)}</span>`,
    ...(d.cameras || []).map((c) => `<span class="chip">📷 ${esc(c)}</span>`),
    ...(d.tasks || []).map((t) => `<span class="chip">“${esc(t)}”</span>`),
  ].join(" ");
  const eps = d.episode_list || [];
  $("#ds-detail").innerHTML = `
    <div class="row section">${chips}</div>
    <div class="card section"><h3>Episodes</h3><table><tr><th>#</th><th>frames</th><th>duration</th><th>tasks</th><th></th></tr>
      ${eps.map((e) => `<tr class="click" onclick="location.hash='#/datasets/${encodeURIComponent(name)}/${e.episode_index}'">
        <td>${e.episode_index}</td><td>${e.length ?? "–"}</td><td>${d.fps && e.length ? fmtDur(e.length / d.fps) : "–"}</td>
        <td>${esc(Array.isArray(e.tasks) ? e.tasks.join("; ") : e.tasks ?? "–")}</td><td>view →</td></tr>`).join("")}
    </table></div>
    <div id="ep-viewer"></div>`;
  const ep = epArg != null ? +epArg : (eps.length ? eps[0].episode_index : null);
  if (ep != null) renderEpisodeViewer($("#ep-viewer"), name, d, ep);
}

async function renderEpisodeViewer(el, name, detail, ep) {
  el.innerHTML = `<div class="card section"><h3>Episode ${ep}</h3><div id="ep-body">loading…</div></div>`;
  let s;
  try { s = await api(`/datasets/${encodeURIComponent(name)}/episodes/${ep}`); }
  catch (e) { $("#ep-body").innerHTML = errBanner(e.message); return; }
  const epMeta = (detail.episode_list || []).find((e) => e.episode_index === ep) || {};
  const cams = Object.keys(epMeta.videos || {});
  const t = s.timestamp || [];
  const t0 = t.length ? t[0] : 0, t1 = t.length ? t[t.length - 1] : 1;
  const body = $("#ep-body");
  body.innerHTML = `
    ${cams.length ? `<div class="grid cols-3 section">` + cams.map((c) => `
      <div class="cam"><span class="label">${esc(c)}</span>
        <video id="vid-${esc(c)}" src="/api/datasets/${encodeURIComponent(name)}/video/${encodeURIComponent(c)}/${ep}" muted playsinline></video></div>`).join("") + `</div>
      <div class="row section"><button id="ep-play" class="primary">Play</button><span class="hint">videos + charts play in sync</span></div>`
      : `<div class="row section"><button id="ep-play" class="primary">Play</button>
         <span class="hint">no videos in this dataset — playing sweeps the cursor over the state/action charts</span></div>`}
    <input type="range" id="ep-scrub" min="${t0}" max="${t1}" step="0.01" value="${t0}" />
    <div class="legend"><span><span class="k" style="background:${CHART_COLORS.state}"></span>observation.state</span>
      <span><span class="k" style="background:${CHART_COLORS.action}"></span>action</span></div>
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
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 90;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!t.length) return;
    const all = [...s1, ...s2].filter((v) => v != null && isFinite(v));
    let lo = Math.min(...all), hi = Math.max(...all);
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
    const t0 = t[0], t1 = t[t.length - 1] || 1;
    const X = (tv) => ((tv - t0) / (t1 - t0 || 1)) * (w - 8) + 4;
    const Y = (v) => h - 6 - ((v - lo) / (hi - lo)) * (h - 12);
    // recessive grid: midline only
    ctx.strokeStyle = "rgba(255,255,255,.07)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(4, Y((lo + hi) / 2)); ctx.lineTo(w - 4, Y((lo + hi) / 2)); ctx.stroke();
    for (const [series, color] of [[s1, CHART_COLORS.state], [s2, CHART_COLORS.action]]) {
      if (!series.length) continue;
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
      series.forEach((v, i) => { const x = X(t[i]), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    if (cursorT != null) {
      ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1;
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

// ---- deployments ----
pages.deployments = {
  title: "Deployments",
  async render(el, args) {
    if (args.length) return renderDeploymentDetail(el, decodeURIComponent(args[0]));
    el.innerHTML = `
      <h2>Deployments <span class="hint" style="display:inline">policy runs launched from this UI</span></h2>
      <div class="grid cols-2 section">
        <div class="card"><h3>Policy check (safe — no arm is energised)</h3>
          <label class="field">policy (checkpoint dir or HF id)<input type="text" id="pc-policy" placeholder="outputs/train/…/pretrained_model or lerobot/smolvla_base" /></label>
          <label class="field">task<input type="text" id="pc-task" value="pick up the object" /></label>
          <button id="btn-pc" class="primary">Run policy check</button>
        </div>
        <div class="card"><h3>Rollout (moves the arms)</h3>
          <label class="field">policy<input type="text" id="ro-policy" /></label>
          <label class="field">task<input type="text" id="ro-task" /></label>
          <div class="grid cols-2">
            <label class="field">duration (s)<input type="number" id="ro-duration" value="60" /></label>
            <label class="check" style="margin-top:22px"><input type="checkbox" id="ro-rtc" checked /> RTC inference</label>
          </div>
          <button id="btn-ro" class="danger">Start rollout</button>
          <div class="hint warn">Runs <code>yamkit rollout</code>: the follower arms will move. Clear the workspace first.</div>
        </div>
      </div>
      <div class="card" id="dep-list">loading…</div>`;
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
    const el = $("#dep-list");
    if (!el) return;
    try {
      const list = await api("/deployments");
      el.innerHTML = list.length ? `<table><tr><th>run</th><th>kind</th><th>model</th><th>task</th><th>latency</th><th>duration</th><th>status</th><th>termination</th></tr>` +
        list.map((d) => `<tr class="click" onclick="location.hash='#/deployments/${encodeURIComponent(d.id)}'">
          <td>${esc(d.id)}</td><td>${esc(d.kind ?? "–")}</td><td>${esc(d.policy ?? "–")}</td><td>${esc(d.task ?? "–")}</td>
          <td>${d.first_call_ms != null ? d.first_call_ms.toFixed(0) + " ms" : "–"}</td>
          <td>${fmtDur(d.duration_s)}</td>
          <td>${statusChip(d.status === "success", d.status ?? "?", d.status === "running" || d.status === "stopped")}</td>
          <td>${esc(d.termination ?? "–")}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no policy runs yet — run a policy check or rollout above</div>`;
    } catch (e) { el.innerHTML = errBanner(e.message); }
  },
  update() {  // re-fetch only when a session starts/ends, not on every poll
    if (this._wasActive !== session.active) { this._wasActive = session.active; this.refreshList(); }
  },
};

async function renderDeploymentDetail(el, id) {
  el.innerHTML = `<a class="back" href="#/deployments">← deployments</a><h2>${esc(id)}</h2><div id="dep-detail">loading…</div>`;
  let d;
  try { d = await api(`/deployments/${encodeURIComponent(id)}`); }
  catch (e) { $("#dep-detail").innerHTML = errBanner(e.message); return; }
  $("#dep-detail").innerHTML = `
    <div class="row section">
      ${statusChip(d.status === "success", d.status, d.status !== "failed")}
      <span class="chip">${esc(d.kind)}</span>
      ${d.policy ? `<span class="chip">${esc(d.policy)}</span>` : ""}
      ${d.task ? `<span class="chip">“${esc(d.task)}”</span>` : ""}
      <span class="chip">started ${fmtDate(d.started_at)}</span>
      <span class="chip">duration ${fmtDur(d.duration_s)}</span>
      ${d.first_call_ms != null ? `<span class="chip">first call ${d.first_call_ms.toFixed(0)} ms</span>` : ""}
      ${d.step_call_ms ? `<span class="chip">next calls ${d.step_call_ms.map((x) => x.toFixed(0)).join("/")} ms</span>` : ""}
      ${d.termination ? `<span class="chip">${esc(d.termination)}</span>` : ""}
    </div>
    ${(d.videos || []).length ? `<div class="grid cols-3 section">` + d.videos.map((v) => `
      <div class="cam"><span class="label">${esc(v)}</span>
        <video controls src="/api/deployments/${encodeURIComponent(id)}/video/${encodeURIComponent(v)}"></video></div>`).join("") + `</div>` : ""}
    <div class="card"><h3>Log</h3><pre class="log" style="max-height:420px">${esc((d.log || []).join("\n")) || "(empty)"}</pre></div>`;
}

// ---- models ----
pages.models = {
  title: "Models",
  async render(el) {
    el.innerHTML = `<h2>Models <span class="hint" style="display:inline">checkpoints under outputs/</span></h2><div class="card" id="model-list">loading…</div>`;
    try {
      const list = await api("/models");
      $("#model-list").innerHTML = list.length ? `<table><tr><th>path</th><th>type</th><th>steps</th><th>dataset</th><th>files</th><th>size</th><th>modified</th></tr>` +
        list.map((m) => `<tr><td class="mono">outputs/${esc(m.path)}</td><td>${esc(m.policy_type ?? "?")}</td>
          <td>${m.steps ?? "–"}</td><td>${esc(m.dataset ?? "–")}</td>
          <td class="mono">${esc((m.files || []).filter((f) => f.endsWith(".json") || f.endsWith(".safetensors")).join(", "))}</td>
          <td>${fmtBytes(m.size_bytes)}</td><td>${fmtDate(m.modified)}</td></tr>`).join("") + `</table>`
        : `<div class="empty">no checkpoints under outputs/ — see README §6 for training</div>`;
    } catch (e) { $("#model-list").innerHTML = errBanner(e.message); }
  },
};

// -------------------------------------------------------------------------------- router ----
let current = null;
function route() {
  const [page, ...args] = (location.hash.replace(/^#\//, "") || "live").split("/");
  const p = pages[page] || pages.live;
  current = p;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#/" + page));
  p.render($("#main"), args);
}
window.addEventListener("hashchange", route);

document.addEventListener("session", () => current?.update && current.update());
document.addEventListener("overview", () => current === pages.live && current.update && current.update());

(async function init() {
  await refreshOverview();
  await refreshSession();
  route();
  sessionTimer = setInterval(refreshSession, 1000);
  overviewTimer = setInterval(refreshOverview, 5000);
})();
