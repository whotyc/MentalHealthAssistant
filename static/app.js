(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const EMOS = ['happy','neutral','sad','angry','fearful','disgusted','surprised'];
const COLORS = {happy:'#b08d3e',neutral:'#8c8a80',sad:'#6d86a0',angry:'#b5654f',
                fearful:'#8b7aa0',disgusted:'#84905a',surprised:'#c08a52'};
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
const DURATION = 12000;
// [ключ перевода, ключ метрики, нижний порог, верхний порог, знаков после запятой]
const METRICS = [
  ['m_ear','ear',0.2,0.32,2], ['m_blink','blinkRate',8,25,0], ['m_mouth','mar',0.05,0.3,2],
  ['m_smile','lift',-0.005,0.02,3], ['m_brow','brow',0.08,0.14,3],
  ['m_furrow','furrow',0.15,0.21,3], ['m_tilt','tilt',3,10,1]];

const mean = a => a.reduce((s, x) => s + x, 0) / (a.length || 1);
const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const dot = e => { const s = h('span', 'dot'); s.style.background = COLORS[e]; return s; };

let lang = localStorage.getItem('lang') || 'ru';
if (!I18N[lang]) lang = 'ru';
let token = localStorage.getItem('token');
let user = null, items = [], lastResult = null, authMode = 'login', curView = 'auth', charts = {};

const t = k => (I18N[lang] && I18N[lang][k]) || I18N.ru[k] || k;
const errText = c => t(I18N.ru['err_' + c] ? 'err_' + c : 'err_generic');

function toast(msg) {
  const el = $('#toast'); el.textContent = msg; el.classList.add('show');
  clearTimeout(toast.id); toast.id = setTimeout(() => el.classList.remove('show'), 3800);
}

async function api(path, method = 'GET', body) {
  const r = await fetch('/api' + path, {
    method,
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: 'Bearer ' + token } : {}) },
    body: body ? JSON.stringify(body) : undefined });
  let data = {}; try { data = await r.json(); } catch (_) {}
  if (!r.ok) {
    if ((r.status === 401 || r.status === 422) && data.msg) logout();
    throw Object.assign(new Error(data.error || data.msg || 'error'), { status: r.status, data });
  }
  return data;
}

/* ---------- Навигация ---------- */
function showView(name) {
  if (name !== 'scan') stopCamera();
  curView = name;
  $$('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name));
  $$('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  document.body.classList.toggle('noauth', name === 'auth');
  window.scrollTo(0, 0);
  if (name === 'home') renderHome();
  if (name === 'profile') renderProfile();
  if (name === 'chat') loadChat();
}

function applyI18n() {
  document.documentElement.lang = lang;
  $$('[data-i18n]').forEach(e => e.textContent = t(e.dataset.i18n));
  $$('[data-i18n-ph]').forEach(e => e.placeholder = t(e.dataset.i18nPh));
  $('#lang').value = lang;
  renderAuth(); updateScanButtons(); renderChips(); renderResult();
  if (user) { renderHome(); renderProfile(); if (curView === 'chat') renderChat(); }
}

/* ---------- Вход ---------- */
function renderAuth() {
  const reg = authMode === 'register';
  $('#a-name').style.display = reg ? '' : 'none';
  $('#a-submit').textContent = t(reg ? 'auth_register' : 'auth_login');
  $('#a-switch-text').textContent = t(reg ? 'auth_have_acc' : 'auth_no_acc');
  $('#a-switch').textContent = t(reg ? 'auth_login' : 'auth_register');
  $('#a-pass').autocomplete = reg ? 'new-password' : 'current-password';
}

async function enterApp() { await loadItems(); showView('home'); }

function logout() {
  token = null; user = null; items = []; lastResult = null; chatLog = [];
  localStorage.removeItem('token'); stopCamera(); $('#result').replaceChildren();
  showView('auth');
}

async function loadItems() { try { items = await api('/measurements?limit=60'); } catch (_) {} }

/* ---------- Главная ---------- */
const isToday = iso => new Date(iso).toDateString() === new Date().toDateString();

function renderHome() {
  if (!user) return;
  $('#h-hello').textContent = `${t('hello')}, ${user.name}`;
  const today = items.find(i => isToday(i.created_at));
  const box = $('#h-today'); box.replaceChildren();
  if (today) {
    box.append(dot(today.emotion), document.createTextNode(t('emo_' + today.emotion)));
    $('#h-today-sub').textContent = `${t('res_stress')}: ${today.stress}/100`;
  } else { box.textContent = t('not_analyzed'); $('#h-today-sub').textContent = ''; }
  const week = items.filter(i => Date.now() - new Date(i.created_at) < 7 * 864e5);
  $('#h-week').textContent = week.length;
  $('#h-week-sub').textContent = week.length ? `${t('avg_stress')}: ${Math.round(mean(week.map(i => i.stress)))}` : t('week_empty');
  const tips = $('#h-tips'); tips.replaceChildren();
  [0, 1, 2].forEach(i => tips.append(h('div', 'tip', t('tip' + i))));
}

/* ---------- Камера и анализ ---------- */
const video = $('#video'), canvas = $('#overlay');
const scan = { stream: null, loop: false, rec: false, samples: [], t0: 0, voice: null, timer: null, voiceRequested: false };
let modelsReady = null;

function ensureModels() {
  if (typeof faceapi === 'undefined') return Promise.reject(new Error('face-api not loaded'));
  if (!modelsReady) {
    modelsReady = Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
    ]).catch(e => { modelsReady = null; throw e; });
  }
  return modelsReady;
}

function updateScanButtons() {
  $('#btn-cam').textContent = t(scan.stream ? 'scan_stop_cam' : 'scan_enable_cam');
  $('#btn-analyze').textContent = t('scan_start');
  $('#btn-analyze').disabled = !scan.stream || scan.rec;
}

async function startCamera() {
  const msg = $('#stage-msg');
  msg.textContent = t('scan_loading');
  try {
    await ensureModels();
    scan.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (err) {
    console.error(err); msg.textContent = ''; toast(t('cam_error')); return;
  }
  video.srcObject = scan.stream;
  await video.play();
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  $('#stage').classList.add('on'); msg.textContent = t('scan_look');
  scan.loop = true; updateScanButtons(); detectLoop();
}

function closeVoice(v) {
  clearInterval(v.timer);
  v.stream.getTracks().forEach(x => x.stop());
  v.ac.close().catch(() => {});
}

function stopCamera() {
  scan.loop = false; scan.rec = false; clearInterval(scan.timer);
  if (scan.stream) { scan.stream.getTracks().forEach(x => x.stop()); scan.stream = null; }
  if (scan.voice) { closeVoice(scan.voice); scan.voice = null; }
  video.srcObject = null;
  $('#stage').classList.remove('on'); $('#progress').classList.remove('on');
  $('#stage-msg').textContent = ''; $('#live').textContent = '';
  updateScanButtons();
}

function landmarkMetrics(P) {
  const faceW = dist(P[0], P[16]);
  const ear = e => (dist(P[e + 1], P[e + 5]) + dist(P[e + 2], P[e + 4])) / (2 * dist(P[e], P[e + 3]));
  const mouthCenterY = (P[51].y + P[57].y) / 2, cornersY = (P[48].y + P[54].y) / 2;
  const eyeTopY = (P[37].y + P[38].y + P[43].y + P[44].y) / 4;
  const browY = (P[19].y + P[24].y) / 2;
  return {
    ear: (ear(36) + ear(42)) / 2,
    mar: dist(P[62], P[66]) / dist(P[60], P[64]),
    lift: (mouthCenterY - cornersY) / faceW,
    brow: (eyeTopY - browY) / faceW,
    furrow: dist(P[21], P[22]) / faceW,
    tilt: Math.abs(Math.atan2(P[45].y - P[36].y, P[45].x - P[36].x) * 180 / Math.PI)
  };
}

function drawFace(ctx, det) {
  const b = det.detection.box;
  ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1.5; ctx.strokeRect(b.x, b.y, b.width, b.height);
  ctx.fillStyle = 'rgba(255,255,255,.7)';
  det.landmarks.positions.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, 1.5, 0, 7); ctx.fill(); });
}

async function detectLoop() {
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
  const ctx = canvas.getContext('2d');
  while (scan.loop) {
    let det = null;
    try { det = await faceapi.detectSingleFace(video, opts).withFaceLandmarks().withFaceExpressions(); }
    catch (e) { console.error(e); }
    if (!scan.loop) break;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const live = $('#live');
    if (det) {
      drawFace(ctx, det);
      const ex = {}; EMOS.forEach(e => ex[e] = det.expressions[e]);
      const top = EMOS.reduce((a, b) => ex[a] > ex[b] ? a : b);
      live.textContent = `${t('emo_' + top)}, ${Math.round(ex[top] * 100)}%`;
      if (scan.rec) scan.samples.push({ t: performance.now(), ex, m: landmarkMetrics(det.landmarks.positions) });
    } else live.textContent = t('no_face_live');
    await new Promise(r => requestAnimationFrame(r));
  }
}

/* --- голос --- */
function detectPitch(buf, sr) {
  let energy = 0; for (let i = 0; i < buf.length; i++) energy += buf[i] * buf[i];
  const minLag = Math.floor(sr / 400), maxLag = Math.floor(sr / 70);
  let best = 0, bestLag = -1;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0; for (let i = 0; i < buf.length - lag; i++) s += buf[i] * buf[i + lag];
    if (s > best) { best = s; bestLag = lag; }
  }
  return (bestLag < 0 || best / energy < 0.3) ? -1 : sr / bestLag;
}

async function startVoice() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false } });
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  await ac.resume();
  const an = ac.createAnalyser(); an.fftSize = 2048;
  ac.createMediaStreamSource(stream).connect(an);
  const v = { stream, ac, frames: 0, rms: [], pitch: [], cent: [], timer: null };
  const td = new Float32Array(an.fftSize), fd = new Uint8Array(an.frequencyBinCount);
  const binHz = ac.sampleRate / an.fftSize;
  v.timer = setInterval(() => {
    an.getFloatTimeDomainData(td); an.getByteFrequencyData(fd);
    v.frames++;
    let e = 0; for (let i = 0; i < td.length; i++) e += td[i] * td[i];
    const rms = Math.sqrt(e / td.length);
    if (rms < 0.015) return;
    v.rms.push(rms);
    const p = detectPitch(td, ac.sampleRate); if (p > 0) v.pitch.push(p);
    let num = 0, den = 0; for (let i = 0; i < fd.length; i++) { num += i * binHz * fd[i]; den += fd[i]; }
    if (den > 0) v.cent.push(num / den);
  }, 60);
  return v;
}

function voiceSummary(v) {
  closeVoice(v);
  if (v.rms.length < 15 || v.pitch.length < 10) return null;
  const sorted = [...v.pitch].sort((a, b) => a - b), med = sorted[Math.floor(sorted.length / 2)];
  const pitch = v.pitch.filter(p => p > med * 0.6 && p < med * 1.6);
  if (pitch.length < 8) return null;
  const pm = mean(pitch), pcv = sd(pitch) / pm, rm = mean(v.rms), rcv = sd(v.rms) / rm, cm = mean(v.cent);
  const stress = 100 * (0.4 * clamp((pcv - 0.08) / 0.25) + 0.3 * clamp((rcv - 0.3) / 0.9) + 0.3 * clamp((cm - 1200) / 2000));
  return { pitch: Math.round(pm), pitchCV: +pcv.toFixed(2), loudCV: +rcv.toFixed(2),
           centroid: Math.round(cm), speech: +(v.rms.length / v.frames).toFixed(2), stress: Math.round(stress) };
}

async function startAnalysis() {
  if (!scan.stream || scan.rec) return;
  const useVoice = $('#use-voice').checked;
  scan.voice = null; scan.voiceRequested = useVoice;
  $('#btn-analyze').disabled = true;
  if (useVoice) { try { scan.voice = await startVoice(); } catch (e) { scan.voiceRequested = false; toast(t('mic_denied')); } }
  scan.samples = []; scan.rec = true; scan.t0 = performance.now();
  $('#stage-msg').textContent = t(scan.voice ? 'scan_speak' : 'scan_look');
  $('#progress').classList.add('on');
  scan.timer = setInterval(() => {
    const p = clamp((performance.now() - scan.t0) / DURATION);
    $('#progress-bar').style.width = (p * 100) + '%';
    if (p >= 1) { clearInterval(scan.timer); finishAnalysis(); }
  }, 100);
}

/* ---------- Выводы без внешних сервисов: простые правила ---------- */
const levelOf = s => s < 34 ? 'low' : s < 67 ? 'mid' : 'high';

function observations(r) {
  const p = [t('ins_emo_' + r.emotion), t('ins_stress_' + levelOf(r.stress))];
  if (r.avg != null) {
    const d = r.stress - r.avg;
    p.push(t(d >= 10 ? 'ins_vs_up' : d <= -10 ? 'ins_vs_down' : 'ins_vs_same').replace('{n}', r.avg));
  }
  if (r.tags.furrow === 'low') p.push(t('ins_brow'));
  if (r.tags.blinkRate === 'high') p.push(t('ins_blink'));
  if (r.voice && r.voice.pitchCV > 0.2) p.push(t('ins_voice'));
  return p.join(' ');
}

function stepKeys(r) {
  if (r.stress >= 67) return ['st_breath', 'st_rest', 'st_walk'];
  if (r.emotion === 'sad' || r.emotion === 'fearful') return ['st_talk', 'st_write', 'st_breath'];
  if (r.emotion === 'angry' || r.emotion === 'disgusted') return ['st_breath', 'st_walk', 'st_water'];
  if (r.emotion === 'happy') return ['st_keep', 'st_gratitude', 'st_talk'];
  return ['st_water', 'st_write', 'st_gratitude'];
}

async function finishAnalysis() {
  scan.rec = false;
  const voice = scan.voice ? voiceSummary(scan.voice) : null; scan.voice = null;
  const S = scan.samples;
  $('#progress').classList.remove('on'); $('#progress-bar').style.width = '0';
  $('#stage-msg').textContent = t('scan_look'); updateScanButtons();
  if (S.length < 8) { toast(t('no_face')); return; }

  const avg = {}; EMOS.forEach(e => avg[e] = S.reduce((s, x) => s + x.ex[e], 0) / S.length);
  const emotion = EMOS.reduce((a, b) => avg[a] > avg[b] ? a : b);
  const m = {}; ['ear','mar','lift','brow','furrow','tilt'].forEach(k => m[k] = mean(S.map(x => x.m[k])));
  const ears = S.map(x => x.m.ear), sortedE = [...ears].sort((a, b) => a - b);
  const base = sortedE[Math.floor(sortedE.length * 0.8)];
  let blinks = 0, closed = false;
  ears.forEach(e => { if (e < base * 0.7) { if (!closed) { blinks++; closed = true; } } else closed = false; });
  const secs = Math.max(1, (S[S.length - 1].t - S[0].t) / 1000);
  m.blinkRate = Math.round(blinks / secs * 60);

  const neg = avg.angry + avg.fearful + 0.7 * avg.sad + 0.6 * avg.disgusted + 0.2 * avg.surprised - 0.3 * avg.happy;
  let face = clamp(neg) * 100;
  if (m.furrow < 0.15) face += 10;
  if (m.blinkRate > 25) face += 10;
  face = Math.min(100, Math.round(face));
  const stress = voice ? Math.round(0.4 * face + 0.6 * voice.stress) : face;

  const tags = {}, metrics = {};
  METRICS.forEach(([, key, lo, hi, dec]) => {
    metrics[key] = +m[key].toFixed(dec + 1);
    tags[key] = m[key] < lo ? 'low' : m[key] > hi ? 'high' : 'norm';
  });
  const emotions = {}; EMOS.forEach(e => emotions[e] = +avg[e].toFixed(3));

  const recent = items.filter(i => Date.now() - new Date(i.created_at) < 30 * 864e5);
  const personalAvg = recent.length >= 2 ? Math.round(mean(recent.map(i => i.stress))) : null;

  lastResult = { emotion, emotions, stress, face_stress: face, voice, metrics, tags,
                 voiceRequested: scan.voiceRequested, avg: personalAvg };
  renderResult();
  $('#result').scrollIntoView({ behavior: 'smooth' });

  const insight = observations(lastResult) + '\n\n' + stepKeys(lastResult).map(k => '— ' + t(k)).join('\n');
  try {
    await api('/measurements', 'POST', { emotion, emotions, stress, insight,
      details: { face_stress: face, voice, metrics, tags } });
    await loadItems();
  } catch (e) { console.error(e); }
}

function renderResult() {
  const box = $('#result'); box.replaceChildren();
  const r = lastResult; if (!r) return;
  const level = levelOf(r.stress);

  const b1 = h('div', 'res-block');
  const head = h('div', 'result-head'); head.append(dot(r.emotion), h('h2', '', t('emo_' + r.emotion)));
  b1.append(head, h('p', 'muted', `${t('res_confidence')}: ${Math.round(r.emotions[r.emotion] * 100)}%`),
    h('div', 'sec-title', t('res_stress')));
  const g = h('div', 'gauge'), f = h('div', 'gauge-fill ' + level); f.style.width = r.stress + '%';
  g.append(f); b1.append(g, h('p', '', `${r.stress}/100, ${t('res_stress_' + level)}`));
  let parts = `${t('res_face_part')}: ${r.face_stress}`;
  if (r.voice) parts += `  ·  ${t('res_voice_part')}: ${r.voice.stress}`;
  b1.append(h('p', 'muted small', parts));
  if (r.voiceRequested && !r.voice) b1.append(h('p', 'muted small', t('voice_none')));
  box.append(b1);

  const b2 = h('div', 'res-block'); b2.append(h('h3', '', t('res_obs')));
  b2.append(h('p', 'prose', observations(r)));
  b2.append(h('div', 'sec-title', t('res_steps')));
  const ul = h('ul', 'steps'); stepKeys(r).forEach(k => ul.append(h('li', '', t(k)))); b2.append(ul);
  b2.append(h('p', 'muted small', t('res_note')));
  box.append(b2);

  const b3 = h('div', 'res-block'); b3.append(h('h3', '', t('res_emotions')));
  [...EMOS].sort((a, b) => r.emotions[b] - r.emotions[a]).forEach(e => {
    const row = h('div', 'bar-row'), tr = h('div', 'track'), fl = h('div', 'fill');
    fl.style.width = Math.round(r.emotions[e] * 100) + '%'; fl.style.background = COLORS[e];
    tr.append(fl); row.append(h('span', '', t('emo_' + e)), tr, h('span', 'muted', Math.round(r.emotions[e] * 100) + '%'));
    b3.append(row);
  });
  box.append(b3);

  const b4 = h('div', 'res-block'); b4.append(h('h3', '', t('res_metrics')));
  METRICS.forEach(([label, key, , , dec]) => {
    const row = h('div', 'kv'), right = h('span');
    right.append(document.createTextNode(Number(r.metrics[key]).toFixed(dec)));
    right.append(h('span', 'tag ' + r.tags[key], t('tag_' + r.tags[key])));
    row.append(h('span', '', t(label)), right); b4.append(row);
  });
  box.append(b4);

  if (r.voice) {
    const b5 = h('div', 'res-block'); b5.append(h('h3', '', t('res_voice')));
    [['v_pitch', r.voice.pitch], ['v_var', r.voice.pitchCV], ['v_loud', r.voice.loudCV], ['v_bright', r.voice.centroid]]
      .forEach(([k, v]) => { const row = h('div', 'kv'); row.append(h('span', '', t(k)), h('span', '', String(v))); b5.append(row); });
    box.append(b5);
  }
}

/* ---------- Разговор (локальный помощник, история в браузере) ---------- */
let chatBusy = false, chatLog = [], lastReply = '';
const chatKey = () => 'chat_' + (user ? user.id : 'guest');
const saveChat = () => localStorage.setItem(chatKey(), JSON.stringify(chatLog.slice(-100)));

function addMsg(who, text) {
  const box = $('#chat-box'), m = h('div', 'msg ' + who, text);
  box.append(m); box.scrollTop = box.scrollHeight; return m;
}
function renderChat() {
  const box = $('#chat-box'); box.replaceChildren();
  addMsg('ai', t('chat_welcome'));
  chatLog.forEach(m => addMsg(m.who, m.text));
}
function loadChat() {
  try { chatLog = JSON.parse(localStorage.getItem(chatKey()) || '[]'); } catch (_) { chatLog = []; }
  renderChat();
}
function renderChips() {
  const c = $('#chips'); c.replaceChildren();
  [1, 2, 3].forEach(i => { const b = h('button', 'chip', t('chip' + i)); b.type = 'button'; b.onclick = () => sendChat(t('chip' + i)); c.append(b); });
}
function pick(arr) {
  const pool = arr.filter(x => x !== lastReply);
  const r = (pool.length ? pool : arr)[Math.floor(Math.random() * (pool.length || arr.length))];
  lastReply = r; return r;
}
function chatReply(text) {
  const s = text.toLowerCase(), c = CHAT[lang] || CHAT.ru;
  if (CRISIS_RE.test(s)) return c.crisis;
  for (const tp of c.topics) if (new RegExp(tp.re, 'i').test(s)) return pick(tp.r);
  return pick(c.fallback);
}
async function sendChat(text) {
  text = (text || '').trim(); if (!text || chatBusy) return;
  chatBusy = true; $('#chat-input').value = '';
  chatLog.push({ who: 'user', text }); saveChat(); addMsg('user', text);
  const typing = h('div', 'msg ai typing', '···'); $('#chat-box').append(typing); $('#chat-box').scrollTop = 1e9;
  await new Promise(r => setTimeout(r, 700 + Math.random() * 700));
  typing.remove();
  const reply = chatReply(text);
  chatLog.push({ who: 'ai', text: reply }); saveChat(); addMsg('ai', reply);
  chatBusy = false;
}

/* ---------- Дневник ---------- */
function renderProfile() {
  if (!user) return;
  $('#p-name').textContent = user.name; $('#p-email').textContent = user.email;
  renderCharts();
  const box = $('#p-history'); box.replaceChildren();
  if (!items.length) { box.append(h('p', 'muted', t('prof_empty'))); return; }
  items.slice(0, 20).forEach(i => {
    const d = h('details', 'hist'), s = h('summary');
    s.append(dot(i.emotion), h('span', '', t('emo_' + i.emotion)),
      h('span', 'muted small', new Date(i.created_at).toLocaleString(lang, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })),
      h('span', 'pill', i.stress + '/100'));
    d.append(s, h('p', 'prose', i.insight || '—')); box.append(d);
  });
}

function renderCharts() {
  Object.values(charts).forEach(c => c.destroy()); charts = {};
  const data = [...items].reverse().slice(-30);
  $('#p-nodata').style.display = data.length >= 2 ? 'none' : '';
  if (typeof Chart === 'undefined' || !data.length) return;
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const grid = dark ? 'rgba(255,255,255,.08)' : 'rgba(0,0,0,.06)', txt = dark ? '#a39a8c' : '#7b7368';
  const line = dark ? '#86a894' : '#5f7f6b';
  if (data.length >= 2) {
    charts.s = new Chart($('#ch-stress'), { type: 'line',
      data: { labels: data.map(i => new Date(i.created_at).toLocaleDateString(lang, { day: 'numeric', month: 'short' })),
        datasets: [{ data: data.map(i => i.stress), borderColor: line, backgroundColor: 'transparent',
          tension: .25, pointRadius: 3, pointBackgroundColor: line, borderWidth: 2 }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 100, grid: { color: grid }, ticks: { color: txt } },
                  x: { grid: { display: false }, ticks: { color: txt, maxTicksLimit: 6 } } } } });
  }
  const counts = EMOS.map(e => items.filter(i => i.emotion === e).length);
  charts.e = new Chart($('#ch-emo'), { type: 'doughnut',
    data: { labels: EMOS.map(e => t('emo_' + e)), datasets: [{ data: counts, backgroundColor: EMOS.map(e => COLORS[e]), borderWidth: 0 }] },
    options: { maintainAspectRatio: false, cutout: '62%', plugins: { legend: { position: 'right', labels: { color: txt, boxWidth: 10 } } } } });
}

/* ---------- События ---------- */
$('#lang').value = lang;
$('#lang').addEventListener('change', e => { lang = e.target.value; localStorage.setItem('lang', lang); applyI18n(); });
$('#a-switch').addEventListener('click', e => { e.preventDefault(); authMode = authMode === 'login' ? 'register' : 'login'; $('#a-err').textContent = ''; renderAuth(); });
$('#auth-form').addEventListener('submit', async e => {
  e.preventDefault(); $('#a-err').textContent = '';
  const body = { email: $('#a-email').value.trim(), password: $('#a-pass').value };
  if (authMode === 'register') body.name = $('#a-name').value.trim();
  try {
    const r = await api(authMode === 'register' ? '/register' : '/login', 'POST', body);
    token = r.token; localStorage.setItem('token', token); user = r.user; $('#a-pass').value = '';
    await enterApp();
  } catch (err) { $('#a-err').textContent = errText(err.data && err.data.error); }
});
$$('.nav-btn').forEach(b => b.addEventListener('click', () => showView(b.dataset.view)));
$('#h-start').addEventListener('click', () => showView('scan'));
$('#btn-cam').addEventListener('click', () => scan.stream ? stopCamera() : startCamera());
$('#btn-analyze').addEventListener('click', startAnalysis);
$('#chat-form').addEventListener('submit', e => { e.preventDefault(); sendChat($('#chat-input').value); });
$('#chat-clear').addEventListener('click', () => { chatLog = []; localStorage.removeItem(chatKey()); renderChat(); });
$('#p-logout').addEventListener('click', logout);
$('#p-delete').addEventListener('click', async () => {
  if (!confirm(t('prof_delete_confirm'))) return;
  try { await api('/measurements', 'DELETE'); items = []; lastResult = null; $('#result').replaceChildren(); renderProfile(); renderHome(); } catch (_) {}
});
window.addEventListener('beforeunload', stopCamera);

/* ---------- Старт ---------- */
(async function boot() {
  applyI18n();
  if (token) { try { user = await api('/me'); await enterApp(); return; } catch (_) { logout(); } }
  showView('auth');
})();
})();