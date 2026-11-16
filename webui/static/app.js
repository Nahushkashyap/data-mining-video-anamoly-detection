'use strict';

const $ = (s) => document.querySelector(s);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtT = (s) => {
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r < 10 ? '0' : ''}${r.toFixed(r < 10 && m === 0 ? 1 : 0)}`.replace(/\.0$/, '');
};
const fmtRange = (a, b) => `${fmtT(a)}–${fmtT(b)}`;
const pct = (p) => `${Math.round(p * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const label = (c) => c.replace('_', ' ');

// ================================================================ status
const state = { status: null, trainedTau: null };

async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    state.status = s;
    const pill = $('#status');
    const txt = $('#status-text');
    const err = $('#model-error');
    if (s.state === 'ready') {
      const amodel = s.audio_model === 'home_head_v1' ? 'trained' : s.audio_model === 'zeroshot' ? 'zero-shot' : '';
      const audio = { ready: `audio ready${amodel ? ` (${amodel})` : ''}`, loading: 'audio loading…', error: 'audio failed', idle: 'audio off' }[s.audio_state];
      if (s.audio_state === 'ready') {
        $('#taua-hint').textContent = s.audio_model === 'home_head_v1'
          ? '1 s audio windows whose trained scream / glass-break / loud-bang / gunshot / explosion / alarm score is above τₐ go to CLAP. 0.50 = each class\'s own threshold, tuned on FSD50K val (≤2% false alarms per class on household sounds).'
          : '1 s audio windows whose zero-shot EfficientAT gunshot / explosion / scream / glass / alarm probability is above τₐ go to CLAP. Default 0.25 is calibrated on SESA train (≤5% false alarms on everyday sounds).';
      }
      pill.className = s.audio_state === 'loading' ? 'pill pill-wait' : 'pill pill-ok';
      txt.textContent = `Ready · ${s.device.toUpperCase()} · τ ${s.tau.toFixed(3)} (seed ${s.seed ?? '?'}) · ${audio}`;
      err.hidden = s.audio_state !== 'error';
      if (s.audio_state === 'error') err.innerHTML = `<b>Audio branch error:</b> ${esc(s.audio_error)}. Video detection still works.`;
      if (state.trainedTau === null) {
        state.trainedTau = s.tau;
        setTau(s.tau);
        setTauA(s.tau_a);
      }
      if (s.audio_state !== 'loading') return;
      setTimeout(pollStatus, 1500);
      return;
    }
    if (s.state === 'error') {
      pill.className = 'pill pill-err';
      txt.textContent = 'Models failed to load';
      err.hidden = false;
      err.innerHTML = `<b>Model error:</b> ${esc(s.error)} ` +
        (s.checkpoint_exists ? '' : `<br>Expected checkpoint: <code>${esc(s.checkpoint)}</code>. Copy it there, then <button class="linkbtn" id="reload">retry</button>.`);
      const rb = $('#reload');
      if (rb) rb.onclick = async () => { await fetch('/api/reload', { method: 'POST' }); setTimeout(pollStatus, 300); };
    } else {
      pill.className = 'pill pill-wait';
      txt.textContent = 'Loading models…';
    }
  } catch {
    $('#status').className = 'pill pill-err';
    $('#status-text').textContent = 'Server offline';
  }
  setTimeout(pollStatus, 1500);
}

// ================================================================ thresholds
const tauEl = $('#tau');
const supEl = $('#sup');
function setTau(v) { tauEl.value = v; $('#tau-val').textContent = Number(v).toFixed(3); }
const tauAEl = $('#taua');
function setTauA(v) { tauAEl.value = v; $('#taua-val').textContent = Number(v).toFixed(2); }
const tau = () => Number(tauEl.value);
const sup = () => Number(supEl.value);
const tauA = () => Number(tauAEl.value);
tauEl.addEventListener('input', () => { $('#tau-val').textContent = tau().toFixed(3); onThresholdInput(); });
supEl.addEventListener('input', () => { $('#sup-val').textContent = sup().toFixed(2); onThresholdInput(); });
tauAEl.addEventListener('input', () => { $('#taua-val').textContent = tauA().toFixed(2); onThresholdInput(); });
tauEl.addEventListener('change', onThresholdCommit);
supEl.addEventListener('change', onThresholdCommit);
tauAEl.addEventListener('change', onThresholdCommit);
$('#tau-reset').addEventListener('click', () => {
  if (state.trainedTau !== null) { setTau(state.trainedTau); onThresholdInput(); onThresholdCommit(); }
});

function onThresholdInput() {
  uploadChart.set({ tau: tau() });
  liveChart.set({ tau: tau() });
  audioChart.set({ tau: tauA() });
  liveAudioChart.set({ tau: tauA() });
}
function onThresholdCommit() {
  if (live.ws && live.ws.readyState === 1) live.ws.send(JSON.stringify({ type: 'params', tau: tau(), suppress: sup(), tau_a: tauA() }));
  if (upload.job && upload.job.status === 'done') rescore();
}

// ================================================================ tabs
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => { x.classList.toggle('active', x === b); x.setAttribute('aria-selected', x === b); });
  $('#panel-upload').hidden = b.dataset.tab !== 'upload';
  $('#panel-live').hidden = b.dataset.tab !== 'live';
  uploadChart.resize(); liveChart.resize(); audioChart.resize(); liveAudioChart.resize();
}));

// ================================================================ chart
class Timeline {
  constructor(canvas, tip, { onSeek, tipText, colorVar } = {}) {
    this.c = canvas; this.tip = tip; this.onSeek = onSeek; this.tipText = tipText; this.colorVar = colorVar || '--series';
    this.d = { points: [], tau: 0.5, bands: [], marks: [], xMin: 0, xMax: 10, playhead: null };
    this.hover = null;
    new ResizeObserver(() => this.resize()).observe(canvas);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.draw());
    canvas.addEventListener('mousemove', (e) => this.move(e));
    canvas.addEventListener('mouseleave', () => { this.hover = null; this.tip.hidden = true; this.draw(); });
    canvas.addEventListener('click', (e) => {
      if (!this.onSeek) return;
      const t = this.xToT(e.offsetX);
      if (t !== null) this.onSeek(t);
    });
  }
  set(patch) { Object.assign(this.d, patch); this.draw(); }
  resize() {
    const r = this.c.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    this.c.width = Math.round(r.width * dpr); this.c.height = Math.round(r.height * dpr);
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    this.draw();
  }
  get pad() { return { l: 38, r: 44, t: 22, b: 24 }; }
  tToX(t) { const p = this.pad; return p.l + (t - this.d.xMin) / Math.max(1e-6, this.d.xMax - this.d.xMin) * (this.w - p.l - p.r); }
  xToT(x) { const p = this.pad; if (x < p.l || x > this.w - p.r) return null; return this.d.xMin + (x - p.l) / (this.w - p.l - p.r) * (this.d.xMax - this.d.xMin); }
  yOf(v) { const p = this.pad; return p.t + (1 - v) * (this.h - p.t - p.b); }
  move(e) {
    const t = this.xToT(e.offsetX);
    const pts = this.d.points;
    if (t === null || !pts.length) { this.hover = null; this.tip.hidden = true; this.draw(); return; }
    let best = null; let bd = Infinity;
    for (const p of pts) { const dd = Math.abs((p.t0 + p.t1) / 2 - t); if (dd < bd) { bd = dd; best = p; } }
    this.hover = best;
    this.draw();
    if (!best) return;
    const above = best.v > this.d.tau;
    const extra = this.tipText ? this.tipText(best) : (best.extra || '');
    this.tip.innerHTML = `<div>${fmtRange(best.t0, best.t1)}</div><div>score <b>${best.v.toFixed(3)}</b> ${above ? '· above τ' : '· below τ'}</div>${extra}`;
    this.tip.hidden = false;
    const x = this.tToX((best.t0 + best.t1) / 2);
    const tw = this.tip.offsetWidth;
    this.tip.style.left = `${Math.min(this.w - tw - 4, Math.max(4, x + 12))}px`;
    this.tip.style.top = `${Math.max(0, this.yOf(best.v) - 52)}px`;
  }
  draw() {
    if (!this.w) return;
    const g = this.c.getContext('2d');
    const { l, r, t, b } = this.pad;
    const W = this.w; const H = this.h;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const C = { grid: css('--grid'), text2: css('--text-2'), muted: css('--muted'), series: css(this.colorVar), band: css('--band'), crit: css('--critical'), supp: css('--supp'), surface: css('--surface'), text: css('--text') };
    g.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';

    // bands (windows sent to CLIP)
    g.fillStyle = C.band;
    for (const bd of this.d.bands) {
      const x0 = Math.max(l, this.tToX(bd.start)); const x1 = Math.min(W - r, this.tToX(bd.end));
      if (x1 > x0) g.fillRect(x0, t, x1 - x0, H - t - b);
    }
    // grid + y labels
    g.strokeStyle = C.grid; g.lineWidth = 1; g.fillStyle = C.muted; g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      const y = Math.round(this.yOf(v)) + 0.5;
      g.beginPath(); g.moveTo(l, y); g.lineTo(W - r, y); g.stroke();
      if (v === 0 || v === 0.5 || v === 1) g.fillText(v.toFixed(1), l - 6, y);
    }
    // x ticks
    const span = this.d.xMax - this.d.xMin;
    const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const step = steps.find((s) => span / s <= 8) || 1200;
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let s = Math.ceil(this.d.xMin / step) * step; s <= this.d.xMax + 1e-6; s += step) {
      g.fillText(fmtT(s), this.tToX(s), H - b + 6);
    }
    // τ line
    const yt = Math.round(this.yOf(this.d.tau)) + 0.5;
    g.setLineDash([5, 4]); g.strokeStyle = C.text2; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(l, yt); g.lineTo(W - r, yt); g.stroke(); g.setLineDash([]);
    g.fillStyle = C.text2; g.textAlign = 'left'; g.textBaseline = 'middle';
    g.fillText(`τ ${this.d.tau.toFixed(2)}`, W - r + 5, yt);

    // score line
    const pts = this.d.points.filter((p) => p.t1 >= this.d.xMin && p.t0 <= this.d.xMax);
    if (pts.length) {
      g.save(); g.beginPath(); g.rect(l, t - 4, W - l - r, H - t - b + 8); g.clip();
      g.strokeStyle = C.series; g.lineWidth = 2; g.lineJoin = 'round'; g.lineCap = 'round';
      g.beginPath();
      pts.forEach((p, i) => { const x = this.tToX((p.t0 + p.t1) / 2); const y = this.yOf(p.v); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      g.stroke();
      if (pts.length === 1) { g.fillStyle = C.series; g.beginPath(); g.arc(this.tToX((pts[0].t0 + pts[0].t1) / 2), this.yOf(pts[0].v), 3, 0, 7); g.fill(); }
      g.restore();
    }
    // event markers on top edge
    g.textBaseline = 'bottom';
    for (const m of this.d.marks) {
      const x = this.tToX(m.t);
      if (x < l || x > W - r) continue;
      g.beginPath(); g.arc(x, t - 9, 4.5, 0, 7);
      if (m.suppressed) { g.strokeStyle = C.supp; g.lineWidth = 2; g.fillStyle = C.surface; g.fill(); g.stroke(); } else { g.fillStyle = C.crit; g.fill(); g.strokeStyle = C.surface; g.lineWidth = 2; g.stroke(); }
      if (m.label) { g.fillStyle = m.suppressed ? C.muted : C.text; g.textAlign = x > W - 90 ? 'right' : 'left'; g.fillText(m.label, x + (g.textAlign === 'left' ? 8 : -8), t - 3); }
    }
    // playhead
    if (this.d.playhead !== null && this.d.playhead >= this.d.xMin && this.d.playhead <= this.d.xMax) {
      const x = Math.round(this.tToX(this.d.playhead)) + 0.5;
      g.strokeStyle = C.text; g.lineWidth = 1; g.beginPath(); g.moveTo(x, t); g.lineTo(x, H - b); g.stroke();
    }
    // hover crosshair
    if (this.hover) {
      const x = this.tToX((this.hover.t0 + this.hover.t1) / 2); const y = this.yOf(this.hover.v);
      g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(Math.round(x) + 0.5, t); g.lineTo(Math.round(x) + 0.5, H - b); g.stroke();
      g.fillStyle = C.series; g.strokeStyle = C.surface; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill(); g.stroke();
    }
  }
}

// ================================================================ upload
const upload = { job: null, polling: null, url: null };
const player = $('#player');
const uploadChart = new Timeline($('#chart-upload'), $('#tip-upload'), {
  onSeek: (t) => {
    player.currentTime = t; player.play().catch(() => {});
    const r = upload.res;
    if (r) inspectClip(Math.max(0, Math.min(r.n_clips - 1, Math.floor(t / r.clip_sec))));
  },
});
const audioChart = new Timeline($('#chart-audio'), $('#tip-audio'), {
  colorVar: '--series-2',
  onSeek: (t) => {
    player.currentTime = t; player.play().catch(() => {});
    const a = upload.res?.audio;
    if (a?.has_audio && a.scores.length) inspectAudio(Math.max(0, Math.min(a.scores.length - 1, Math.floor(t))));
  },
});
player.addEventListener('timeupdate', () => { uploadChart.set({ playhead: player.currentTime }); audioChart.set({ playhead: player.currentTime }); });

const SRC_LABEL = { video: 'video', audio: 'audio only', 'audio+video': 'audio + video', 'audio-confirmed': 'audio-confirmed' };
function fusedChip(a, onClick) {
  const el = document.createElement('div');
  el.className = 'fchip';
  el.innerHTML = `<span class="badge badge-alert">⚠ ${esc(label(a.class))}</span><span class="time">${fmtRange(a.start_sec, a.end_sec)}</span>` +
    `<span>${pct(a.confidence)}</span><span class="src">${SRC_LABEL[a.source] || a.source}</span>` +
    (a.audio_class ? `<span class="muted small">audio: ${esc(label(a.audio_class))}</span>` : '');
  if (onClick) el.addEventListener('click', onClick);
  return el;
}
function audioCard(ev, { onClick, live: isLive = false } = {}) {
  const el = document.createElement('div');
  el.className = 'event audio-ev';
  const badge = ev.suppressed ? '<span class="badge badge-supp">○ suppressed</span>'
    : `<span class="badge ${isLive && ev.status === 'active' ? 'badge-live' : 'badge-alert'}">🔊 ${esc(label(ev.class))}</span>`;
  const status = isLive ? `<span class="muted small">${ev.status === 'active' ? 'ongoing' : 'closed'}</span>` : '';
  el.innerHTML = `<div><div class="head">${badge}<span class="time">${fmtRange(ev.start_sec, ev.end_sec)}</span>${status}</div>
    ${probBars(ev.top3)}
    <p class="summary">Trigger: ${esc(label(ev.trigger_group || ''))}${ev.trigger_peak !== undefined ? ` ${pct(ev.trigger_peak)}` : ''} · P(normal + unrelated) ${pct(ev.non_anomaly_prob)}</p></div>`;
  if (onClick) el.addEventListener('click', onClick);
  return el;
}

const drop = $('#drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) startUpload(f); });
$('#file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) startUpload(f); e.target.value = ''; });

function showProgress(name, msg, frac) {
  $('#progress').hidden = false;
  $('#progress-name').textContent = name;
  $('#progress-msg').textContent = msg;
  $('#progress-fill').style.width = `${Math.round(frac * 100)}%`;
}

function startUpload(file) {
  if (!state.status || state.status.state !== 'ready') { alertBanner('Models are not loaded yet.'); return; }
  if (upload.url) URL.revokeObjectURL(upload.url);
  upload.url = URL.createObjectURL(file);
  player.src = upload.url;
  $('#results').hidden = true;
  upload.res = null; inspState.kind = null;
  $('#insp-body').innerHTML = '<p class="muted small empty">Click a point on either score chart, an alert card, or a row above to inspect that time section.</p>';
  $('#insp-links').innerHTML = '';
  const fd = new FormData();
  fd.append('file', file); fd.append('tau', tau()); fd.append('suppress', sup()); fd.append('tau_a', tauA());
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/jobs');
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) showProgress(file.name, `Uploading ${pct(e.loaded / e.total)}`, 0.05 * e.loaded / e.total); };
  xhr.onload = () => {
    if (xhr.status !== 200) { showProgress(file.name, `Error: ${errText(xhr.responseText)}`, 0); return; }
    upload.job = JSON.parse(xhr.responseText);
    pollJob();
  };
  xhr.onerror = () => showProgress(file.name, 'Upload failed', 0);
  showProgress(file.name, 'Uploading…', 0);
  xhr.send(fd);
}

function errText(t) { try { return JSON.parse(t).detail; } catch { return t; } }

async function pollJob() {
  clearTimeout(upload.polling);
  const r = await fetch(`/api/jobs/${upload.job.id}`);
  const j = await r.json();
  upload.job = j;
  showProgress(j.name, j.status === 'error' ? `Error: ${j.error}` : j.message, 0.05 + 0.95 * j.progress);
  if (j.status === 'done') { $('#progress').hidden = true; renderResult(j.result); return; }
  if (j.status === 'error') return;
  upload.polling = setTimeout(pollJob, 400);
}

async function rescore() {
  const r = await fetch(`/api/jobs/${upload.job.id}/rescore`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tau: tau(), suppress: sup(), tau_a: tauA() }),
  });
  if (r.ok) { upload.job = await r.json(); pollJob(); }
}

function probBars(top3) {
  return `<div class="probs">${top3.map((t) => `<div class="prob"><span>${esc(label(t.class))}</span><span class="track"><span class="fill" style="width:${(t.prob * 100).toFixed(1)}%;display:block"></span></span><span class="pct">${pct(t.prob)}</span></div>`).join('')}</div>`;
}

function eventCard(ev, { onClick, live: isLive = false } = {}) {
  const el = document.createElement('div');
  el.className = 'event';
  const img = ev.peak_url || ev.peak_jpeg || '';
  const benign = ev.class === 'normal' || ev.class === 'unrelated';  // below the veto but CLIP's top guess is benign
  const badge = ev.suppressed
    ? `<span class="badge badge-supp">○ suppressed</span>`
    : benign
      ? `<span class="badge badge-supp">○ ${esc(label(ev.class))} · no alert</span>`
      : `<span class="badge ${isLive && ev.status === 'active' ? 'badge-live' : 'badge-alert'}">⚠ ${esc(label(ev.class))}</span>`;
  const status = isLive ? `<span class="muted small">${ev.status === 'active' ? 'ongoing' : 'closed'}</span>` : '';
  el.innerHTML = `<img alt="Peak frame at ${fmtT(ev.peak_ts)}" src="${img}">
    <div><div class="head">${badge}<span class="time">${fmtRange(ev.start_sec, ev.end_sec)}</span>${status}</div>
    ${probBars(ev.top3)}
    <p class="summary">Peak frame ${fmtT(ev.peak_ts)} · P(normal + unrelated) ${pct(ev.non_anomaly_prob)}</p></div>`;
  if (onClick) el.addEventListener('click', onClick);
  return el;
}

function tile(k, v, s = '') { return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`; }

// alerts actually raised: the fused list when audio ran, else video windows CLIP did not call benign
const quiet = (e) => e.suppressed || e.class === 'normal' || e.class === 'unrelated';  // shown, but not an alert

function nAlerts(res) {
  if (res.audio) return (res.fused_alerts || []).length;
  return res.events.filter((e) => e.class !== 'normal' && e.class !== 'unrelated').length;
}

function renderResult(res) {
  upload.res = res;
  $('#results').hidden = false;
  const tm = res.timing || {};
  $('#tiles').innerHTML = [
    tile('Duration', fmtT(res.duration_sec), `${res.fps} fps · ${res.n_frames} frames`),
    tile('Clips', res.n_clips, `${res.clip_sec.toFixed(1)} s each`),
    tile('Escalated to CLIP', `${res.escalated_pct}%`, `${res.flagged_clips} clips above τ`),
    tile('Alerts', nAlerts(res), `${res.windows.length} windows checked`),
    tile('Suppressed', res.suppressed_events.length, 'vetoed by CLIP'),
    res.audio ? (res.audio.has_audio
      ? tile('Audio', `${Math.round(100 * (1 - res.audio.silent_frac))}%`, `non-silent · ${res.audio.events.length} audio alert${res.audio.events.length === 1 ? '' : 's'}`)
      : tile('Audio', '—', 'no audio track'))
      : tile('X3D-S', `${tm.x3d_ms_per_clip ?? '–'} ms`, `per clip · ${res.device}`),
  ].join('');

  const box = $('#events');
  box.innerHTML = '';
  if (!res.events.length) box.innerHTML = '<p class="muted small empty">No alerts. Either nothing scored above τ, or CLIP vetoed every flagged window.</p>';
  res.events.forEach((ev) => box.appendChild(eventCard(ev, { onClick: () => { player.currentTime = ev.start_sec; player.play().catch(() => {}); inspectWindow(ev, true); } })));
  const na = nAlerts(res);
  $('#events-count').textContent = `${na} alert${na === 1 ? '' : 's'}`;
  const sbox = $('#supp');
  sbox.innerHTML = '';
  res.suppressed_events.forEach((ev) => sbox.appendChild(eventCard(ev, { onClick: () => { player.currentTime = ev.start_sec; inspectWindow(ev, true); } })));
  $('#supp-count').textContent = `(${res.suppressed_events.length})`;
  $('#supp-wrap').hidden = !res.suppressed_events.length;
  renderAudio(res);
  renderWindowCards(res);
  refreshInspector();

  const cs = res.clip_sec;
  const points = res.scores.map((v, i) => ({ t0: i * cs, t1: Math.min(res.duration_sec, (i + 1) * cs), v, raw: res.raw_scores[i] }));
  const marks = [...res.events, ...res.suppressed_events].map((ev) => ({ t: ev.peak_ts, suppressed: ev.suppressed, label: ev.suppressed ? '' : `${label(ev.class)} ${pct(ev.confidence)}` }));
  uploadChart.tipText = (p) => `<div class="muted">raw ${p.raw.toFixed(3)}</div>`;
  uploadChart.set({ points, tau: res.tau, bands: res.windows.map((w) => ({ start: w.start_sec, end: w.end_sec })), marks, xMin: 0, xMax: Math.max(res.duration_sec, cs), playhead: player.currentTime || 0 });
  uploadChart.resize();

  $('#score-table').innerHTML = `<table><thead><tr><th>Clip</th><th>Time</th><th>Smoothed</th><th>Raw</th><th>&gt; τ</th></tr></thead><tbody>${
    points.map((p, i) => `<tr><td>${i}</td><td>${fmtRange(p.t0, p.t1)}</td><td>${p.v.toFixed(3)}</td><td>${p.raw.toFixed(3)}</td><td>${p.v > res.tau ? 'yes' : ''}</td></tr>`).join('')
  }</tbody></table>`;
}

function renderAudio(res) {
  const seek = (t) => () => { player.currentTime = t; player.play().catch(() => {}); };
  const a = res.audio;
  const fbox = $('#fused'); fbox.innerHTML = '';
  (res.fused_alerts || []).forEach((f) => fbox.appendChild(fusedChip(f, seek(f.start_sec))));
  if (a && !(res.fused_alerts || []).length) fbox.innerHTML = '<p class="muted small empty">No combined alerts.</p>';
  const abox = $('#aevents'); abox.innerHTML = '';
  $('#audio-card').hidden = !a;
  if (!a) { abox.innerHTML = '<p class="muted small empty">Audio models not loaded.</p>'; return; }
  if (!a.has_audio) {
    abox.innerHTML = '<p class="muted small empty">This file has no audio track.</p>';
    $('#audio-note').textContent = 'No audio track in this file.';
    audioChart.set({ points: [], bands: [], marks: [], tau: a.tau_a, xMin: 0, xMax: Math.max(res.duration_sec, 1) });
    return;
  }
  const open = (ev) => () => { seek(ev.start_sec)(); inspectAudioEvent(ev, true); };
  a.events.forEach((ev) => abox.appendChild(audioCard(ev, { onClick: open(ev) })));
  a.suppressed_events.forEach((ev) => abox.appendChild(audioCard(ev, { onClick: open(ev) })));
  if (!a.events.length && !a.suppressed_events.length) abox.innerHTML = '<p class="muted small empty">No audio window crossed τₐ.</p>';
  const pts = a.scores.map((v, i) => ({ t0: a.times[i], t1: a.times[i] + 1, v, extra: `<div class="muted">${esc(label(a.top_group[i]))}</div>` }));
  audioChart.set({
    points: pts, tau: a.tau_a, xMin: 0, xMax: Math.max(res.duration_sec, 1), playhead: player.currentTime || 0,
    bands: a.windows.map((w) => ({ start: w.start_sec, end: w.end_sec })),
    marks: [...a.events, ...a.suppressed_events].map((e) => ({ t: (e.start_sec + e.end_sec) / 2, suppressed: e.suppressed, label: e.suppressed ? '' : `${label(e.class)} ${pct(e.confidence)}` })),
  });
  audioChart.resize();
  $('#audio-note').textContent = `${Math.round(100 * a.silent_frac)}% of 1 s windows are digitally silent (score 0). Click to seek.`;
}


// ================================================================ Stage 2 window views (22 Nov)
const NON_ANOM = new Set(['normal', 'unrelated']);
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const heatCell = (v, txt) => `<td class="cell${v > 0.55 ? ' dk' : ''}" style="background:color-mix(in srgb, var(--series) ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, transparent)">${txt ?? pct(v)}</td>`;
const orderClasses = (cls) => [...cls.filter((c) => !NON_ANOM.has(c)), ...cls.filter((c) => NON_ANOM.has(c))];

// horizontal bar chart of a model's class probabilities (fixed class order so windows compare at a glance)
function classBars(probs, cls, { accent = '--series', top = null, title = '' } = {}) {
  const desc = cls.map((c) => `${label(c)} ${pct(probs[c] ?? 0)}`).join(', ');
  return `<div class="cbars" role="img" aria-label="${esc(title)}: ${esc(desc)}">${title ? `<div class="mini-title">${esc(title)}</div>` : ''}${cls.map((c) => {
    const v = probs[c] ?? 0;
    return `<div class="cbar${c === top ? ' top' : ''}" title="${esc(label(c))}: ${(v * 100).toFixed(1)}%"><span class="cl">${esc(label(c))}</span><span class="track"><span class="fill" style="width:${Math.max(0, Math.min(100, v * 100)).toFixed(1)}%;background:var(${NON_ANOM.has(c) ? '--supp' : accent})"></span></span><span class="pct">${pct(v)}</span></div>`;
  }).join('')}</div>`;
}

// small single-series line chart (SVG) with a dashed threshold line; one point per frame / second
function miniLine(vals, { thr, thrLabel, xLabels, tips, color = '--series', title = '' }) {
  const W = 460; const H = 170; const L = 30; const R = 12; const T = 10; const B = 22;
  const n = vals.length;
  const x = (i) => (n === 1 ? (L + W - R) / 2 : L + (i * (W - L - R)) / (n - 1));
  const y = (v) => T + (1 - Math.max(0, Math.min(1, v))) * (H - T - B);
  const grid = [0, 0.5, 1].map((g) => `<line x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}" style="stroke:var(--grid)"/><text x="${L - 5}" y="${y(g) + 3.5}" text-anchor="end" class="ax">${g.toFixed(1)}</text>`).join('');
  const every = n > 10 ? Math.ceil(n / 8) : 1;
  const xl = xLabels.map((t, i) => (i % every === 0 || i === n - 1 ? `<text x="${x(i)}" y="${H - 6}" text-anchor="middle" class="ax">${esc(t)}</text>` : '')).join('');
  const th = thr === undefined ? '' : `<line x1="${L}" x2="${W - R}" y1="${y(thr)}" y2="${y(thr)}" stroke-dasharray="5 4" style="stroke:var(--text-2)" stroke-width="1.5"/><text x="${W - R}" y="${y(thr) + 13}" text-anchor="end" class="ax thr">${esc(thrLabel || '')}</text>`;
  const path = vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const dots = vals.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="4.5" style="fill:var(${color});stroke:var(--surface)" stroke-width="2"><title>${esc(tips ? tips[i] : `${xLabels[i]}: ${v.toFixed(3)}`)}</title></circle>`).join('');
  return `<div class="mini">${title ? `<div class="mini-title">${esc(title)}</div>` : ''}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${grid}${th}<path d="${path}" fill="none" style="stroke:var(${color})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>${dots}${xl}</svg></div>`;
}

function videoWindowCard(ev, res) {
  const d = ev.details;
  const cls = orderClasses(d?.classes || Object.keys(ev.mean_probs));
  const top = ev.suppressed ? (ev.clip_top_class || ev.class) : ev.class;
  const badge = ev.suppressed ? '<span class="badge badge-supp">○ vetoed · no alert</span>' : `<span class="badge badge-alert">⚠ ${esc(label(ev.class))} ${pct(ev.confidence)}</span>`;
  let line = '';
  if (d) {
    const ci = Object.fromEntries(d.classes.map((c, i) => [c, i]));
    const pa = d.frame_probs.map((r) => 1 - r[ci.normal] - r[ci.unrelated]);
    const tips = d.frame_probs.map((r, i) => {
      const best = d.classes[r.indexOf(Math.max(...r))];
      return `F${i + 1}${d.frame_ts ? ` at ${fmtT(d.frame_ts[i])}` : ''}: P(anomaly) ${pct(pa[i])}, top class ${label(best)} ${pct(Math.max(...r))}`;
    });
    line = miniLine(pa, { thr: 1 - res.suppress, thrLabel: `veto below ${(1 - res.suppress).toFixed(2)}`, xLabels: pa.map((_, i) => `F${i + 1}`), tips,
      title: 'P(anomaly) = 1 − P(normal) − P(unrelated), per frame' });
    if (d.frame_urls) line += `<div class="wthumbs">${d.frame_urls.map((u, i) => `<img src="${u}" alt="Frame ${i + 1}" class="${i === d.peak_idx ? 'peak' : ''}">`).join('')}</div>`;
  }
  const img = ev.peak_url || ev.peak_jpeg;
  const verdict = ev.suppressed
    ? `P(normal) + P(unrelated) = <b>${pct(ev.non_anomaly_prob)}</b> ≥ ${res.suppress.toFixed(2)} → CLIP says this looks normal, so the window is vetoed.`
    : `P(normal) + P(unrelated) = <b>${pct(ev.non_anomaly_prob)}</b> &lt; ${res.suppress.toFixed(2)} → alert, named by the top anomaly class <b>${esc(label(ev.class))}</b>${NON_ANOM.has(ev.clip_top_class) ? ` (CLIP's single top answer was ${esc(label(ev.clip_top_class))})` : ''}.`;
  const el = document.createElement('div');
  el.className = `wcard${ev.suppressed ? ' veto' : ''}`;
  el.innerHTML = `<div class="wc-head">${badge}<span class="time">${fmtRange(ev.start_sec, ev.end_sec)}</span><span class="muted small">clips ${ev.start_clip}–${ev.end_clip} · Stage 1 peak ${ev.stage1_peak?.toFixed(3) ?? '–'}</span><button class="linkbtn small wc-insp">Inspect →</button></div>
    <div class="wc-body">${img ? `<figure class="wc-peak"><img src="${img}" alt="Peak frame at ${fmtT(ev.peak_ts)}"><figcaption>peak frame · ${fmtT(ev.peak_ts)}</figcaption></figure>` : ''}
      ${classBars(ev.mean_probs, cls, { top, title: 'CLIP class probabilities (mean of 8 frames)' })}
      <div>${line}</div></div>
    <p class="wc-verdict">${verdict}</p>`;
  const open = () => { player.currentTime = ev.start_sec; inspectWindow(ev, true); };
  el.querySelector('.wc-insp').addEventListener('click', open);
  el.querySelector('.wc-body').addEventListener('click', open);
  return el;
}

function audioWindowCard(ev, res) {
  const a = res.audio;
  const cls = orderClasses(ev.details?.classes || Object.keys(ev.mean_probs));
  const badge = ev.suppressed ? '<span class="badge badge-supp">○ vetoed · no alert</span>'
    : `<span class="badge badge-alert">🔊 ${esc(label(ev.class))} ${pct(ev.confidence)}</span>`;
  const secs = [];
  for (let k = Math.floor(ev.start_sec); k < Math.ceil(ev.end_sec) && k < a.scores.length; k++) secs.push(k);
  const vals = secs.map((k) => a.scores[k]);
  const tips = secs.map((k) => `${fmtT(k)}: trigger ${a.scores[k].toFixed(3)} (${label(a.top_group[k])})${a.scores[k] > a.tau_a ? ', above τₐ' : ''}`);
  const line = miniLine(vals, { thr: a.tau_a, thrLabel: `τₐ ${a.tau_a.toFixed(2)}`, xLabels: secs.map((k) => fmtT(k)), tips, color: '--series-2',
    title: 'Stage 1 trigger score per second in this window' });
  const verdict = ev.suppressed
    ? `P(normal) + P(unrelated) = <b>${pct(ev.non_anomaly_prob)}</b> ≥ ${res.suppress.toFixed(2)} → CLAP says this sounds normal, so the window is vetoed.`
    : `CLAP hears <b>${esc(label(ev.class))}</b> (${pct(ev.confidence)}); P(normal) + P(unrelated) = ${pct(ev.non_anomaly_prob)} &lt; ${res.suppress.toFixed(2)} → audio event${ev.video_class && ev.video_class !== ev.class ? `, reported as video class <b>${esc(label(ev.video_class))}</b>` : ''}.`;
  const el = document.createElement('div');
  el.className = `wcard audio${ev.suppressed ? ' veto' : ''}`;
  el.innerHTML = `<div class="wc-head">${badge}<span class="time">${fmtRange(ev.start_sec, ev.end_sec)}</span><span class="muted small">Stage 1: ${esc(label(ev.trigger_group || ''))} ${ev.trigger_peak !== undefined ? ev.trigger_peak.toFixed(2) : ''}</span><button class="linkbtn small wc-insp">Inspect →</button></div>
    <div class="wc-body">${classBars(ev.mean_probs, cls, { accent: '--series-2', top: ev.class, title: 'CLAP class probabilities' })}<div>${line}</div></div>
    <p class="wc-verdict">${verdict}</p>`;
  const open = () => { player.currentTime = ev.start_sec; inspectAudioEvent(ev, true); };
  el.querySelector('.wc-insp').addEventListener('click', open);
  el.querySelector('.wc-body').addEventListener('click', open);
  return el;
}

function renderWindowCards(res) {
  const vrows = [...res.events, ...res.suppressed_events].sort((x, y) => x.start_sec - y.start_sec);
  const vbox = $('#cw-video'); vbox.innerHTML = '';
  $('#cw-video-count').textContent = vrows.length ? `· ${plural(vrows.length, 'window')} · ${plural(res.events.length, 'alert')} · ${res.suppressed_events.length} vetoed` : '';
  if (!vrows.length) vbox.innerHTML = '<p class="muted small empty">Stage 1 flagged nothing, so CLIP did not run. Lower τ to send more of the video to CLIP.</p>';
  vrows.forEach((ev) => vbox.appendChild(videoWindowCard(ev, res)));
  const a = res.audio;
  $('#cw-audio-card').hidden = !a;
  if (!a) return;
  const abox = $('#cw-audio'); abox.innerHTML = '';
  const arows = [...a.events, ...a.suppressed_events].sort((x, y) => x.start_sec - y.start_sec);
  $('#cw-audio-count').textContent = arows.length ? `· ${plural(arows.length, 'window')} · ${plural(a.events.length, 'event')} · ${a.suppressed_events.length} vetoed` : '';
  if (!a.has_audio) { abox.innerHTML = '<p class="muted small empty">This file has no audio track, so only the video branch ran.</p>'; return; }
  if (!arows.length) { abox.innerHTML = '<p class="muted small empty">No 1 s window crossed τₐ, so CLAP did not run.</p>'; return; }
  arows.forEach((ev) => abox.appendChild(audioWindowCard(ev, res)));
}

// ---------------------------------------------------------------- inspector
const bcStore = new Map();
let bcN = 0;
function barcode(values, title, extra = '') {
  const id = `bc${bcN++}`;
  bcStore.set(id, values);
  let mx = 0; let nz = 0; let neg = false;
  for (const x of values) { mx = Math.max(mx, Math.abs(x)); if (Math.abs(x) > 1e-6) nz++; if (x < 0) neg = true; }
  const key = neg ? 'blue = positive, red = negative' : 'all values ≥ 0';
  return `<div class="barcode-wrap"><div class="barcode-meta"><span>${title}</span><span>${values.length}-d · max |x| ${mx.toFixed(3)} · ${Math.round((100 * nz) / Math.max(1, values.length))}% non-zero${extra}</span></div>
    <canvas class="barcode" data-bc="${id}" aria-label="${esc(title)}: ${values.length} values"></canvas>
    <div class="barcode-meta"><span>dim 0</span><span data-bc-hover="${id}">${key}; darker = larger |value|. Hover for exact values.</span><span>dim ${values.length - 1}</span></div></div>`;
}
function hydrateBarcodes(root) {
  root.querySelectorAll('canvas.barcode').forEach((c) => {
    const v = bcStore.get(c.dataset.bc);
    if (!v) return;
    const draw = () => {
      const r = c.getBoundingClientRect(); if (!r.width) return;
      const dpr = window.devicePixelRatio || 1;
      c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
      const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
      let m = 0; for (const x of v) m = Math.max(m, Math.abs(x)); m = m || 1;
      const pos = css('--series'); const negc = css('--critical'); const w = r.width / v.length;
      for (let i = 0; i < v.length; i++) {
        const a = Math.abs(v[i]) / m; if (a < 0.02) continue;
        g.globalAlpha = Math.min(1, 0.15 + 0.85 * a); g.fillStyle = v[i] < 0 ? negc : pos;
        g.fillRect(i * w, 0, Math.max(w, 0.75), r.height);
      }
      g.globalAlpha = 1;
    };
    draw();
    new ResizeObserver(draw).observe(c);
    const out = root.querySelector(`[data-bc-hover="${c.dataset.bc}"]`);
    c.addEventListener('mousemove', (e) => {
      const i = Math.max(0, Math.min(v.length - 1, Math.floor((e.offsetX / c.getBoundingClientRect().width) * v.length)));
      if (out) out.textContent = `dim ${i} = ${v[i].toFixed(4)}`;
    });
  });
}

function heatTable(cls, cols, colHeads, mode, { peakCol = -1 } = {}) {
  // cols: array of {class -> value}; mode 'p' = probability (0..1 shading), 's' = similarity (min-max shading)
  let lo = Infinity; let hi = -Infinity;
  cols.forEach((col) => cls.forEach((c) => { lo = Math.min(lo, col[c]); hi = Math.max(hi, col[c]); }));
  const shade = (x) => (mode === 'p' ? x : (x - lo) / Math.max(1e-6, hi - lo));
  const txt = (x) => (mode === 'p' ? `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%` : x.toFixed(3));
  return `<div class="s2-table"><table class="heat"><thead><tr><th class="l">Class</th>${colHeads.map((h, i) => `<th${i === peakCol ? ' style="color:var(--critical)"' : ''}>${h}</th>`).join('')}</tr></thead><tbody>${
    cls.map((c) => `<tr><th class="l">${esc(label(c))}</th>${cols.map((col) => heatCell(shade(col[c]), txt(col[c]))).join('')}</tr>`).join('')
  }</tbody></table></div>`;
}

function segToggle(name, opts, cur) {
  return `<span class="seg" data-seg="${name}">${opts.map(([k, t]) => `<button data-v="${k}" class="${k === cur ? 'on' : ''}">${t}</button>`).join('')}</span>`;
}

const inspState = { kind: null, key: null, vmode: 'p', amode: 'p' };
function inspShow(html, links = '', scroll = false) {
  const body = $('#insp-body');
  body.innerHTML = html;
  $('#insp-links').innerHTML = links;
  hydrateBarcodes(body);
  body.querySelectorAll('[data-clip]').forEach((b) => b.addEventListener('click', () => { inspectClip(Number(b.dataset.clip)); }));
  if (scroll) $('#inspector').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function refreshInspector() {
  const { kind, key } = inspState;
  if (!kind || !upload.res) return;
  if (kind === 'clip' && key < upload.res.n_clips) inspectClip(key);
  else if (kind === 'audio' && upload.res.audio?.has_audio) inspectAudio(key);
  else if (kind === 'vwin') {
    const ev = [...upload.res.events, ...upload.res.suppressed_events].find((e) => e.start_sec === key);
    if (ev) inspectWindow(ev); else inspShow('<p class="muted small empty">That window is no longer flagged at the new thresholds. Click a chart point or a row to inspect another section.</p>');
  } else if (kind === 'awin') {
    const a = upload.res.audio;
    const ev = a && [...a.events, ...a.suppressed_events].find((e) => e.start_sec === key);
    if (ev) inspectAudioEvent(ev); else inspShow('<p class="muted small empty">That audio window is no longer flagged at the new thresholds.</p>');
  }
}

function clipWindowSection(ev, n) {
  const d = ev.details;
  const head = `<h3>${n} · CLIP on window ${fmtRange(ev.start_sec, ev.end_sec)} <span class="muted">clips ${ev.start_clip}–${ev.end_clip} · ${d ? d.frame_probs.length : 8} frames sampled evenly · ${esc(d?.embedding_model || 'CLIP ViT-B/16')}</span></h3>`;
  const verdict = ev.suppressed
    ? `<p class="small"><b>Vetoed.</b> P(normal) + P(unrelated) = <b>${pct(ev.non_anomaly_prob)}</b> ≥ ${upload.res.suppress.toFixed(2)}, so no alert. CLIP's top-1 was ${esc(label(ev.clip_top_class || ev.class))}.</p>`
    : `<p class="small"><b>Alert: ${esc(label(ev.class))} ${pct(ev.confidence)}.</b> P(normal) + P(unrelated) = ${pct(ev.non_anomaly_prob)} &lt; ${upload.res.suppress.toFixed(2)}${NON_ANOM.has(ev.clip_top_class) ? `; CLIP's top-1 was ${esc(label(ev.clip_top_class))}, so the alert is named by the most likely anomaly class` : ''}.</p>`;
  if (!d) return `<div class="insp-sec">${head}${verdict}${heatTable(orderClasses(Object.keys(ev.mean_probs)), [ev.mean_probs], ['mean'], 'p')}</div>`;
  const cls = orderClasses(d.classes);
  const ci = Object.fromEntries(d.classes.map((c, i) => [c, i]));
  const peak = d.peak_idx ?? (d.frame_ts ? d.frame_ts.indexOf(ev.peak_ts) : -1);
  const frames = d.frame_urls ? `<div class="frames">${d.frame_urls.map((u, i) => {
    const top = d.classes[d.frame_probs[i].indexOf(Math.max(...d.frame_probs[i]))];
    return `<figure class="${i === peak ? 'peak' : ''}"><img src="${u}" alt="Frame ${i + 1} at ${fmtT(d.frame_ts[i])}"><figcaption>F${i + 1} · ${fmtT(d.frame_ts[i])}<br>${esc(label(top))}</figcaption></figure>`;
  }).join('')}</div>` : '';
  const mode = inspState.vmode;
  const src = mode === 'p' ? d.frame_probs : d.frame_sims;
  const cols = src.map((row) => Object.fromEntries(d.classes.map((c) => [c, row[ci[c]]])));
  const mean = Object.fromEntries(d.classes.map((c) => [c, src.reduce((t, r) => t + r[ci[c]], 0) / src.length]));
  const heads = [...src.map((_, i) => `F${i + 1}`), 'mean'];
  return `<div class="insp-sec">${head}${frames}
    <p class="small muted">Each frame's 512-d image embedding is compared with each class's prompt embedding (cosine similarity); probability = softmax(${d.logit_scale} × similarity) over the 9 classes. The window's verdict uses the mean over frames. Red outline = peak frame.</p>
    <div class="small" style="margin:6px 0">Show ${segToggle('vmode', [['p', 'probabilities'], ['s', 'cosine similarity']], mode)}</div>
    ${heatTable(cls, [...cols, mean], heads, mode, { peakCol: peak })}
    ${verdict}
    ${barcode(d.embedding, 'CLIP image embedding (mean of the frames, L2-normalised)')}</div>`;
}

function bindSeg(root, name, fn) {
  root.querySelectorAll(`[data-seg="${name}"] button`).forEach((b) => b.addEventListener('click', () => fn(b.dataset.v)));
}

function inspectWindow(ev, scroll = false) {
  inspState.kind = 'vwin'; inspState.key = ev.start_sec;
  const r = upload.res;
  const clips = [];
  for (let k = ev.start_clip; k <= ev.end_clip && k < r.n_clips; k++) clips.push(k);
  const s1 = `<div class="insp-sec"><h3>1 · Stage 1 clips in this window <span class="muted">smoothed score per ${r.clip_sec.toFixed(1)} s clip; click one to see its X3D-S embedding</span></h3>
    <div class="flow">${clips.map((k) => `<button class="node linkbtn" style="text-decoration:none" data-clip="${k}">clip ${k} · <b>${r.scores[k].toFixed(3)}</b>${r.scores[k] > r.tau ? ' ▲' : ''}</button>`).join('')}</div>
    <p class="small muted">▲ = above τ ${r.tau.toFixed(3)}. Windows are padded by one clip on each side, so edge clips can be below τ.</p></div>`;
  inspShow(s1 + clipWindowSection(ev, 2), '', scroll);
  bindSeg($('#insp-body'), 'vmode', (v) => { inspState.vmode = v; inspectWindow(ev); });
}

async function inspectClip(k, scroll = false) {
  const r = upload.res; const job = upload.job;
  if (!r || !job) return;
  inspState.kind = 'clip'; inspState.key = k;
  $('#insp-body').innerHTML = `<p class="muted small empty">Loading clip ${k}…</p>`;
  let d;
  try {
    const resp = await fetch(`/api/jobs/${job.id}/clips/${k}`);
    if (!resp.ok) throw new Error(inspErr(errText(await resp.text())));
    d = await resp.json();
  } catch (e) { $('#insp-body').innerHTML = `<p class="muted small empty">Could not load clip ${k}: ${esc(e.message)}</p>`; return; }
  if (inspState.kind !== 'clip' || inspState.key !== k) return;
  const active = d.hidden1.filter((x) => x > 0).length;
  const ev = [...r.events, ...r.suppressed_events].find((e) => k >= e.start_clip && k <= e.end_clip);
  const verdict = d.above_tau ? `<b>above τ</b> → part of a window sent to CLIP` : (ev ? 'below τ, but inside a padded window sent to CLIP' : 'below τ → not sent to CLIP');
  const html = `<div class="insp-sec"><div class="insp-head">${d.thumb ? `<img src="${d.thumb}" alt="Middle frame of clip ${k}">` : ''}<dl class="kv">
      <dt>Time section</dt><dd>clip ${k} · ${fmtRange(d.t0, d.t1)}</dd>
      <dt>Frames</dt><dd>${d.frames[0]}–${d.frames[1]} (${d.frames_sampled} sampled, every ${d.frame_stride}th, resized to 182 px)</dd>
      <dt>Stage 1 score</dt><dd>raw ${d.raw.toFixed(4)} · smoothed ${d.smoothed.toFixed(4)} (mean of clips ${d.smooth_from[0]}–${d.smooth_from[1]})</dd>
      <dt>Verdict</dt><dd>${d.smoothed.toFixed(3)} vs τ ${d.tau.toFixed(3)}: ${verdict}</dd></dl></div></div>
    <div class="insp-sec"><h3>1 · X3D-S clip embedding <span class="muted">${esc(d.embedding_model)} · L2 norm ${d.embedding_norm}</span></h3>
      ${barcode(d.embedding, 'X3D-S features (input to the MIL head after L2-normalisation)')}</div>
    <div class="insp-sec"><h3>2 · MIL head <span class="muted">2048 → 512 → 32 → 1 (dropout off at inference)</span></h3>
      <div class="flow"><span class="node">L2-normalise <b>2048</b></span>→<span class="node">fc1 + ReLU <b>512</b> · ${active} active</span>→<span class="node">fc2 <b>32</b></span>→<span class="node">fc3 logit <b>${d.logit.toFixed(3)}</b></span>→<span class="node">sigmoid <b>${d.head_score.toFixed(4)}</b> = raw</span>→<span class="node">${d.smooth_window}-clip mean <b>${d.smoothed.toFixed(4)}</b></span>→<span class="node">${d.above_tau ? '&gt;' : '≤'} τ <b>${d.tau.toFixed(3)}</b></span></div>
      ${barcode(d.hidden1, 'fc1 activations after ReLU')}
      ${barcode(d.hidden2, 'fc2 output')}</div>
    ${ev ? clipWindowSection(ev, 3) : '<div class="insp-sec"><p class="small muted">This clip is not inside a flagged window, so CLIP did not look at it.</p></div>'}`;
  inspShow(html, `<a class="linkbtn" href="/api/jobs/${job.id}/clips/${k}" target="_blank" rel="noopener">raw numbers (JSON)</a>`, scroll);
  bindSeg($('#insp-body'), 'vmode', (v) => { inspState.vmode = v; inspectClip(k); });
}

function inspErr(t) {
  if (t === 'Not Found') return 'the app server is running an older version. Stop it (Ctrl+C) and start ./run_webui.sh again';
  if (t === 'unknown job') return 'this upload is from before the server restarted. Upload the video again';
  return t;
}

function softmax(xs, scale) { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(scale * (x - m))); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); }

function clapSection(ev, n) {
  const d = ev.details;
  const head = `<h3>${n} · CLAP on audio window ${fmtRange(ev.start_sec, ev.end_sec)} <span class="muted">${esc(d?.embedding_model || 'MS-CLAP 2023')}${d ? ` · ${plural(d.n_crops, 'crop')} of 7 s` : ''}</span></h3>`;
  const thr = upload.res.suppress;
  const verdict = ev.suppressed
    ? `<p class="small"><b>Vetoed.</b> P(normal) + P(unrelated) = <b>${pct(ev.non_anomaly_prob)}</b> ≥ ${thr.toFixed(2)}.</p>`
    : `<p class="small"><b>Audio event: ${esc(label(ev.class))} ${pct(ev.confidence)}</b>${ev.video_class && ev.video_class !== ev.class ? ` → maps to the video class <b>${esc(label(ev.video_class))}</b>` : ''}. P(normal) + P(unrelated) = ${pct(ev.non_anomaly_prob)} &lt; ${thr.toFixed(2)}.</p>`;
  if (!d) return `<div class="insp-sec">${head}${verdict}${heatTable(orderClasses(Object.keys(ev.mean_probs)), [ev.mean_probs], ['mean'], 'p')}</div>`;
  const cls = orderClasses(d.classes);
  const mode = inspState.amode;
  const rows = mode === 'p' ? d.crop_sims.map((s) => softmax(s, d.logit_scale)) : d.crop_sims;
  const cols = rows.map((row) => Object.fromEntries(d.classes.map((c, i) => [c, row[i]])));
  const mean = mode === 'p' ? ev.mean_probs : Object.fromEntries(d.classes.map((c, i) => [c, rows.reduce((t, r) => t + r[i], 0) / rows.length]));
  const heads = [...rows.map((_, i) => `crop ${i + 1}`), 'mean'];
  return `<div class="insp-sec">${head}
    <p class="small muted">The window is cut into 7 s crops (repeated if shorter). Each crop's audio embedding is compared with every prompt of a class; the class score is the best-matching prompt's cosine similarity, and probability = softmax(${d.logit_scale.toFixed(1)} × similarity).</p>
    <div class="small" style="margin:6px 0">Show ${segToggle('amode', [['p', 'probabilities'], ['s', 'cosine similarity']], mode)}</div>
    ${heatTable(cls, [...cols, mean], heads, mode)}
    ${verdict}
    ${barcode(d.embedding, 'CLAP audio embedding (mean of crops, L2-normalised)')}</div>`;
}

function inspectAudioEvent(ev, scroll = false) {
  inspState.kind = 'awin'; inspState.key = ev.start_sec;
  const a = upload.res.audio;
  const secs = [];
  for (let k = Math.floor(ev.start_sec); k < Math.ceil(ev.end_sec) && k < a.scores.length; k++) secs.push(k);
  const s1 = `<div class="insp-sec"><h3>1 · Audio Stage 1 seconds in this window <span class="muted">click one to see its EfficientAT embedding</span></h3>
    <div class="flow">${secs.map((k) => `<button class="node linkbtn" style="text-decoration:none" data-sec="${k}">${fmtT(k)} · <b>${a.scores[k].toFixed(2)}</b> ${esc(label(a.top_group[k]))}${a.scores[k] > a.tau_a ? ' ▲' : ''}</button>`).join('')}</div>
    <p class="small muted">▲ = above τₐ ${a.tau_a.toFixed(2)}. Windows are padded by 1 s on each side.</p></div>`;
  inspShow(s1 + clapSection(ev, 2), '', scroll);
  const body = $('#insp-body');
  body.querySelectorAll('[data-sec]').forEach((b) => b.addEventListener('click', () => inspectAudio(Number(b.dataset.sec))));
  bindSeg(body, 'amode', (v) => { inspState.amode = v; inspectAudioEvent(ev); });
}

async function inspectAudio(k, scroll = false) {
  const r = upload.res; const job = upload.job;
  if (!r || !job) return;
  inspState.kind = 'audio'; inspState.key = k;
  $('#insp-body').innerHTML = `<p class="muted small empty">Loading audio second ${k}…</p>`;
  let d;
  try {
    const resp = await fetch(`/api/jobs/${job.id}/audio/${k}`);
    if (!resp.ok) throw new Error(inspErr(errText(await resp.text())));
    d = await resp.json();
  } catch (e) { $('#insp-body').innerHTML = `<p class="muted small empty">Could not load audio window ${k}: ${esc(e.message)}</p>`; return; }
  if (inspState.kind !== 'audio' || inspState.key !== k) return;
  const a = r.audio;
  const ev = [...a.events, ...a.suppressed_events].find((e) => k >= e.start_sec && k < e.end_sec);
  const trained = d.model === 'home_head_v1';
  const gbars = Object.entries(d.group_scores).sort((x, y) => y[1] - x[1]).map(([g, v]) => `<div class="gbar"><span>${esc(label(g))}</span><span class="track"><span class="fill" style="width:${(Math.min(1, v) * 100).toFixed(1)}%"></span><span class="tick" style="left:${(d.tau_a * 100).toFixed(1)}%"></span></span><span class="pct">${v.toFixed(3)}</span></div>`).join('');
  const verdict = d.silent ? 'digitally silent → score forced to 0' : d.above_tau ? `<b>above τₐ</b> → part of a window sent to CLAP` : (ev ? 'below τₐ, but inside a padded window sent to CLAP' : 'below τₐ → not sent to CLAP');
  const html = `<div class="insp-sec"><dl class="kv">
      <dt>Time section</dt><dd>1 s window ${k} · ${fmtRange(d.t0, d.t1)}</dd>
      <dt>Loudness</dt><dd>${d.dbfs} dBFS${d.silent ? ' (silent)' : ''}</dd>
      <dt>Stage 1 score</dt><dd>${d.score.toFixed(4)} (max over classes; top: ${esc(label(d.top_group))})</dd>
      <dt>Verdict</dt><dd>${d.score.toFixed(3)} vs τₐ ${d.tau_a.toFixed(2)}: ${verdict}</dd></dl></div>
    <div class="insp-sec"><h3>1 · EfficientAT embedding <span class="muted">${esc(d.embedding_model)}</span></h3>
      ${d.embedding.length ? barcode(d.embedding, 'mn10 pooled embedding (input to the trained head)') : '<p class="small muted">Embedding not available.</p>'}</div>
    <div class="insp-sec"><h3>2 · ${trained ? 'Trained home-sound head: per-class score' : 'Zero-shot trigger: per-class score'} <span class="muted">${trained ? '960 → 256 → 6, calibrated so each class\'s own threshold sits at 0.50' : 'max AudioSet probability per class group'}; dashed line = τₐ</span></h3>
      <div class="gbars">${gbars}</div></div>
    ${ev ? clapSection(ev, 3) : '<div class="insp-sec"><p class="small muted">This second is not inside a flagged audio window, so CLAP did not hear it.</p></div>'}`;
  inspShow(html, `<a class="linkbtn" href="/api/jobs/${job.id}/audio/${k}" target="_blank" rel="noopener">raw numbers (JSON)</a>`, scroll);
  bindSeg($('#insp-body'), 'amode', (v) => { inspState.amode = v; inspectAudio(k); });
}

function alertBanner(msg) { const b = $('#model-error'); b.hidden = false; b.textContent = msg; setTimeout(() => { if (state.status?.state === 'ready') b.hidden = true; }, 4000); }

// ================================================================ live
const live = { ws: null, stream: null, timer: null, points: [], events: new Map(), start: 0, canvas: document.createElement('canvas'), sending: false,
  actx: null, anode: null, abuf: [], alen: 0, apoints: [], aevents: new Map(), fused: [] };
const cam = $('#cam');
const liveChart = new Timeline($('#chart-live'), $('#tip-live'));
const liveAudioChart = new Timeline($('#chart-live-audio'), $('#tip-live-audio'), { colorVar: '--series-2' });

// ---- microphone: AudioWorklet -> resample to 32 kHz -> 1 s int16 chunks prefixed with "AUD0"
const WORKLET_SRC = `class Tap extends AudioWorkletProcessor { process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; } } registerProcessor('tap', Tap);`;
async function startMic(stream) {
  const track = stream.getAudioTracks()[0];
  if (!track) return false;
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'application/javascript' })));
  const src = ctx.createMediaStreamSource(new MediaStream([track]));
  const node = new AudioWorkletNode(ctx, 'tap');
  const inRate = ctx.sampleRate;
  live.abuf = []; live.alen = 0;
  node.port.onmessage = (e) => {
    live.abuf.push(e.data); live.alen += e.data.length;
    if (live.alen < inRate) return;
    const all = new Float32Array(live.alen); let o = 0;
    for (const b of live.abuf) { all.set(b, o); o += b.length; }
    const sec = all.subarray(0, inRate); const rest = all.slice(inRate);
    live.abuf = rest.length ? [rest] : []; live.alen = rest.length;
    const out = new Int16Array(32000);
    for (let i = 0; i < 32000; i++) { // linear resample inRate -> 32 kHz
      const x = (i * inRate) / 32000; const j = Math.floor(x); const f = x - j;
      const v = (sec[j] || 0) * (1 - f) + (sec[j + 1] || 0) * f;
      out[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
    }
    const msg = new Uint8Array(4 + out.byteLength);
    msg.set([65, 85, 68, 48]); msg.set(new Uint8Array(out.buffer), 4); // "AUD0"
    if (live.ws && live.ws.readyState === 1) live.ws.send(msg.buffer);
  };
  src.connect(node);
  live.actx = ctx; live.anode = node;
  return true;
}
function stopMic() {
  try { live.anode && live.anode.disconnect(); live.actx && live.actx.close(); } catch { /* ignore */ }
  live.actx = null; live.anode = null; live.abuf = []; live.alen = 0;
}
const LIVE_SPAN = 180;

async function listCameras() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    const sel = $('#camera'); const cur = sel.value;
    sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${esc(d.label || `Camera ${i + 1}`)}</option>`).join('') || '<option value="">Default camera</option>';
    if (cur) sel.value = cur;
  } catch { /* no permission yet */ }
}

function setLiveState(kind, text) {
  const el = $('#live-state');
  el.className = `pill ${{ idle: 'pill-idle', watch: 'pill-watch', alert: 'pill-alert', wait: 'pill-wait', err: 'pill-err' }[kind]}`;
  el.lastElementChild.textContent = text;
}

async function startLive() {
  if (!state.status || state.status.state !== 'ready') { alertBanner('Models are not loaded yet.'); return; }
  $('#live-warn').hidden = true;
  try {
    const id = $('#camera').value;
    const wantMic = $('#use-mic').checked && state.status.audio_state === 'ready';
    const audio = wantMic ? { echoCancellation: false, noiseSuppression: false, autoGainControl: false } : false;
    live.stream = await navigator.mediaDevices.getUserMedia({ video: id ? { deviceId: { exact: id }, width: { ideal: 640 }, height: { ideal: 480 } } : { width: { ideal: 640 }, height: { ideal: 480 } }, audio });
  } catch (e) {
    $('#live-warn').hidden = false; $('#live-warn').textContent = `Camera unavailable: ${e.message}. Allow camera access for this page in the browser (and in System Settings → Privacy → Camera).`;
    return;
  }
  cam.srcObject = live.stream; $('#cam-empty').hidden = true;
  listCameras();
  live.points = []; live.events.clear(); $('#live-events').innerHTML = '<p class="muted small empty">Watching… the first score arrives after about 5 s.</p>'; $('#live-count').textContent = '';
  live.apoints = []; live.aevents.clear(); live.fused = [];
  liveChart.set({ points: [], bands: [], marks: [], xMin: 0, xMax: LIVE_SPAN, tau: tau() });
  liveAudioChart.set({ points: [], bands: [], marks: [], xMin: 0, xMax: LIVE_SPAN, tau: tauA() });
  liveChart.resize(); liveAudioChart.resize();
  $('#live-start').disabled = true; $('#live-stop').disabled = false;
  setLiveState('wait', 'Connecting…');

  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`);
  live.ws = ws;
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => ws.send(JSON.stringify({ type: 'start', tau: tau(), suppress: sup(), tau_a: tauA() }));
  ws.onmessage = (e) => onLiveMsg(JSON.parse(e.data));
  ws.onclose = () => { if (live.ws === ws) teardownLive('Disconnected'); };
}

function captureFrame() {
  const ws = live.ws;
  if (!ws || ws.readyState !== 1 || !cam.videoWidth || ws.bufferedAmount > 2e6) return;
  const w = Math.min(640, cam.videoWidth); const h = Math.round(cam.videoHeight * w / cam.videoWidth);
  const c = live.canvas; c.width = w; c.height = h;
  c.getContext('2d').drawImage(cam, 0, 0, w, h);
  c.toBlob((b) => { if (b && ws.readyState === 1) b.arrayBuffer().then((buf) => ws.send(buf)); }, 'image/jpeg', 0.82);
}

function onLiveMsg(m) {
  if (m.type === 'ready') {
    setLiveState('watch', 'Watching');
    live.start = performance.now();
    clearInterval(live.timer);
    live.timer = setInterval(captureFrame, 1000 / m.fps);
    if (m.audio && live.stream && live.stream.getAudioTracks().length) {
      startMic(live.stream).then((ok) => { $('#live-audio-note').textContent = ok ? 'Microphone on. Each point is 1 s of audio.' : 'No microphone track.'; })
        .catch((e) => { $('#live-audio-note').textContent = `Microphone unavailable: ${e.message}`; });
    } else {
      $('#live-audio-note').textContent = m.audio ? 'Microphone off (tick “Microphone” before starting).' : 'Audio models not loaded; video only.';
    }
  } else if (m.type === 'ascore') {
    live.apoints.push({ t0: m.t0, t1: m.t1, v: m.score, extra: `<div class="muted">${m.silent ? 'silent' : esc(label(m.group))} · ${m.dbfs} dBFS</div>` });
    const xMax = Math.max(LIVE_SPAN, m.t1 + 5);
    liveAudioChart.set({ points: live.apoints.slice(-200), xMin: xMax - LIVE_SPAN, xMax });
    live.ams = m.ms;
    updateLiveStats();
  } else if (m.type === 'aevent') {
    live.aevents.set(m.id, m);
    live.clapms = m.clap_ms;
    updateLiveStats();
    renderLiveEvents();
    if (m.status === 'active' && !m.suppressed) {
      const banner = $('#live-banner');
      banner.hidden = false; banner.className = 'live-banner alert';
      banner.textContent = `🔊 ${label(m.class).toUpperCase()} · ${pct(m.confidence)} (audio)`;
      setLiveState('alert', `Audio alert: ${label(m.class)}`);
    }
  } else if (m.type === 'fused') {
    live.fused = m.alerts;
    const last = m.alerts[m.alerts.length - 1];
    if (last && last.source !== 'video') {
      const banner = $('#live-banner');
      banner.hidden = false; banner.className = 'live-banner alert';
      banner.textContent = `⚠ ${label(last.class).toUpperCase()} · ${pct(last.confidence)} (${SRC_LABEL[last.source] || last.source})`;
    }
  } else if (m.type === 'score') {
    live.points.push({ t0: m.t0, t1: m.t1, v: m.score, raw: m.raw });
    const xMax = Math.max(LIVE_SPAN, m.t1 + 5);
    liveChart.tipText = (p) => `<div class="muted">raw ${p.raw.toFixed(3)}</div>`;
    liveChart.set({ points: live.points.slice(-200), xMin: xMax - LIVE_SPAN, xMax });
    if (!liveHasActiveAlert()) setLiveState('watch', m.flagged ? 'Above τ · checking with CLIP' : `Watching · score ${m.score.toFixed(2)}`);
  } else if (m.type === 'event') {
    live.events.set(m.id, m);
    renderLiveEvents();
    const banner = $('#live-banner');
    if (m.status === 'active') {
      banner.hidden = false;
      banner.className = `live-banner ${quiet(m) ? 'supp' : 'alert'}`;
      banner.textContent = quiet(m) ? `Flagged by Stage 1 · CLIP says ${label(m.class)} (${pct(m.non_anomaly_prob)} normal/unrelated)` : `⚠ ${label(m.class).toUpperCase()} · ${pct(m.confidence)}`;
      setLiveState(quiet(m) ? 'watch' : 'alert', m.suppressed ? 'Flagged · vetoed by CLIP' : quiet(m) ? `Flagged · CLIP says ${label(m.class)}` : `Alert: ${label(m.class)}`);
    } else {
      setTimeout(() => { if (!liveHasActiveAlert()) $('#live-banner').hidden = true; }, 2500);
      setLiveState('watch', 'Watching');
    }
  } else if (m.type === 'timing') {
    if (m.x3d_ms !== undefined) live.x3d = m.x3d_ms;
    if (m.clip_ms !== undefined) live.clip = m.clip_ms;
    if (m.backlog !== undefined) live.backlog = m.backlog;
    updateLiveStats();
  } else if (m.type === 'warning' || m.type === 'error') {
    $('#live-warn').hidden = false; $('#live-warn').textContent = m.message;
    if (m.type === 'error' && !live.timer) teardownLive('Error');
  } else if (m.type === 'stopped') {
    teardownLive('Stopped');
  }
}

function updateLiveStats() {
  const audio = live.ams !== undefined ? ` · audio trigger ${live.ams} ms/s · CLAP ${live.clapms ?? '–'} ms` : '';
  $('#live-stats').textContent = `X3D-S ${live.x3d ?? '–'} ms/clip · CLIP ${live.clip ?? '–'} ms/window${audio} · backlog ${live.backlog ?? 0}`;
}

function liveHasActiveAlert() {
  for (const e of live.events.values()) if (e.status === 'active' && !quiet(e)) return true;
  for (const e of live.aevents.values()) if (e.status === 'active' && !e.suppressed) return true;
  return false;
}

function renderLiveEvents() {
  const evs = [...live.events.values()].sort((a, b) => b.id - a.id);
  const aevs = [...live.aevents.values()];
  const all = [...evs.map((e) => ({ e, kind: 'v' })), ...aevs.map((e) => ({ e, kind: 'a' }))].sort((x, y) => y.e.start_sec - x.e.start_sec);
  const box = $('#live-events'); box.innerHTML = '';
  all.forEach(({ e, kind }) => box.appendChild(kind === 'v' ? eventCard(e, { live: true }) : audioCard(e, { live: true })));
  const n = evs.filter((e) => !quiet(e)).length;
  const na = aevs.filter((e) => !e.suppressed).length;
  $('#live-count').textContent = `${n} video · ${na} audio alert${n + na === 1 ? '' : 's'} · ${evs.length - n + aevs.length - na} suppressed`;
  liveAudioChart.set({
    bands: aevs.map((e) => ({ start: e.start_sec, end: e.end_sec })),
    marks: aevs.map((e) => ({ t: (e.start_sec + e.end_sec) / 2, suppressed: e.suppressed, label: e.suppressed ? '' : label(e.class) })),
  });
  liveChart.set({
    bands: evs.map((e) => ({ start: e.start_sec, end: e.end_sec })),
    marks: evs.map((e) => ({ t: e.peak_ts, suppressed: e.suppressed, label: e.suppressed ? '' : label(e.class) })),
  });
}

function stopLive() {
  clearInterval(live.timer); live.timer = null;
  $('#live-stop').disabled = true;
  if (live.ws && live.ws.readyState === 1) {
    setLiveState('wait', 'Finishing last clips…');
    live.ws.send(JSON.stringify({ type: 'stop' }));
    setTimeout(() => teardownLive('Stopped'), 6000);
  } else teardownLive('Stopped');
}

function teardownLive(text) {
  clearInterval(live.timer); live.timer = null;
  const ws = live.ws; live.ws = null;
  if (ws && ws.readyState <= 1) ws.close();
  stopMic();
  if (live.stream) live.stream.getTracks().forEach((t) => t.stop());
  live.stream = null; cam.srcObject = null; $('#cam-empty').hidden = false;
  $('#live-banner').hidden = true;
  $('#live-start').disabled = false; $('#live-stop').disabled = true;
  setLiveState('idle', text);
}

$('#live-start').addEventListener('click', startLive);
$('#live-stop').addEventListener('click', stopLive);
navigator.mediaDevices?.addEventListener?.('devicechange', listCameras);
listCameras();
pollStatus();
