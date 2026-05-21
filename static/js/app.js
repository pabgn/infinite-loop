// ──────────────────────────────────────────────
// STATE
// ──────────────────────────────────────────────
let graph = null;
let currentBeat = 0;
let jumpCount = 0;
let lastSimilarity = 0;
let isPlaying = false;
let pollInterval = null;
let engineInterval = null;
let currentHash = null;
let totalTime = 0; // accumulated playback seconds

const audio = document.getElementById('audio-el');

// ── Web Audio — AudioBufferSourceNode playback ──────────────────────────────
// The entire audio file is decoded into an AudioBuffer once. Jumps launch a new
// BufferSourceNode at the target offset while fading out the old one — both play
// simultaneously during the crossfade, so there is never a gap or seek glitch.
let audioCtx  = null;
let audioBuffer = null;      // decoded PCM for the full track
let activeSource = null;     // current BufferSourceNode
let activeGain   = null;     // its GainNode
let ctxStartTime   = 0;      // audioCtx.currentTime when current segment started
let offsetAtStart  = 0;      // audio-file offset (seconds) at that moment
const CF = 0.07;             // crossfade duration (70 ms)

function ensureAudioCtx() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
}

function getCurrentTime() {
  if (!audioCtx || !isPlaying) return offsetAtStart;
  return Math.min(offsetAtStart + (audioCtx.currentTime - ctxStartTime),
                  audioBuffer ? audioBuffer.duration : 0);
}

function _startSource(offset) {
  const src  = audioCtx.createBufferSource();
  src.buffer = audioBuffer;
  const gain = audioCtx.createGain();
  src.connect(gain);
  gain.connect(audioCtx.destination);
  src.start(0, Math.max(0, offset));
  ctxStartTime  = audioCtx.currentTime;
  offsetAtStart = offset;
  activeSource  = src;
  activeGain    = gain;
}

function startPlayback(offset = 0) {
  ensureAudioCtx();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  if (activeSource) { try { activeSource.stop(); } catch (_) {} }
  if (audioBuffer) _startSource(offset);
}

function pausePlayback() {
  offsetAtStart = getCurrentTime();  // save position before stopping
  if (activeSource) { try { activeSource.stop(); } catch (_) {} activeSource = null; }
}

// User-initiated seek: immediate crossfade to a file offset.
function seekTo(targetOffset) {
  if (!audioCtx || !audioBuffer) return;
  if (!isPlaying) { offsetAtStart = targetOffset; return; }
  const now = audioCtx.currentTime;
  if (activeGain) {
    activeGain.gain.cancelScheduledValues(now);
    activeGain.gain.setValueAtTime(activeGain.gain.value, now);
    activeGain.gain.linearRampToValueAtTime(0, now + CF);
  }
  const oldSrc = activeSource;
  if (oldSrc) setTimeout(() => { try { oldSrc.stop(); } catch (_) {} }, (CF + 0.05) * 1000);
  const src  = audioCtx.createBufferSource();
  src.buffer = audioBuffer;
  const gain = audioCtx.createGain();
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(1, now + CF);
  src.connect(gain);
  gain.connect(audioCtx.destination);
  src.start(now, Math.max(0, targetOffset));
  ctxStartTime  = now;
  offsetAtStart = targetOffset;
  activeSource  = src;
  activeGain    = gain;
}

// Engine jump: beat-aligned crossfade using the Web Audio scheduler.
// We schedule src.start() at the exact future wall-clock time of the beat
// boundary — the scheduler is sample-accurate, so there is no JS timing drift.
function seamlessJump(destBeatStart, srcNextBeatTime) {
  if (!audioCtx || !audioBuffer) return;
  const now = audioCtx.currentTime;

  // How far until the source beat boundary from this precise moment
  const srcNow = offsetAtStart + (now - ctxStartTime);
  const remaining = Math.max(0, srcNextBeatTime - srcNow);
  const switchAt = now + remaining;  // exact wall-clock of the beat boundary

  // Old source fades out ending at switchAt + CF
  if (activeGain) {
    activeGain.gain.cancelScheduledValues(now);
    activeGain.gain.setValueAtTime(activeGain.gain.value, now);
    activeGain.gain.linearRampToValueAtTime(0, switchAt + CF);
  }
  const oldSrc = activeSource;
  if (oldSrc) setTimeout(() => { try { oldSrc.stop(); } catch (_) {} }, (remaining + CF + 0.1) * 1000);

  // New source scheduled to start at switchAt from destBeatStart — sample-accurate
  const src  = audioCtx.createBufferSource();
  src.buffer = audioBuffer;
  const gain = audioCtx.createGain();
  gain.gain.setValueAtTime(0, switchAt);
  gain.gain.linearRampToValueAtTime(1, switchAt + CF);
  src.connect(gain);
  gain.connect(audioCtx.destination);
  src.start(switchAt, destBeatStart);

  // getCurrentTime() after switchAt: destBeatStart + (audioCtx.currentTime - switchAt)
  ctxStartTime  = switchAt;
  offsetAtStart = destBeatStart;
  activeSource  = src;
  activeGain    = gain;
}
// ────────────────────────────────────────────────────────────────────────────

// Jump engine parameters
let minSim = 0.88;            // only near-identical segments
let minBeatGap = 128;         // beats between jumps (~1 min at 120 BPM)

let lastJumpTime = -Infinity;  // getCurrentTime() value at the last jump
let progressionPhase = 0;     // 0-1, real position in song (currentTime / duration)
let jumpDestOffset = 0;       // manual destination fine-trim (seconds)
let lastEligibleJumpBeat = -1; // cached; -1 = needs recompute

// ──────────────────────────────────────────────
// SLIDERS
// ──────────────────────────────────────────────
document.getElementById('sl-sim').addEventListener('input', e => {
  minSim = e.target.value / 100;
  document.getElementById('val-sim').textContent = e.target.value + '%';
  lastEligibleJumpBeat = -1;
  buildCircleStatic(); renderCircle();
});
document.getElementById('sl-gap').addEventListener('input', e => {
  minBeatGap = parseInt(e.target.value);
  document.getElementById('val-gap').textContent = e.target.value + ' beats';
});
document.getElementById('sl-offset').addEventListener('input', e => {
  jumpDestOffset = parseInt(e.target.value) / 1000;
  const v = parseInt(e.target.value);
  document.getElementById('val-offset').textContent = (v >= 0 ? '+' : '') + v + 'ms';
});

// ──────────────────────────────────────────────
// ANALYZE
// ──────────────────────────────────────────────
document.getElementById('analyze-btn').addEventListener('click', startAnalysis);
document.getElementById('url-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') startAnalysis();
});

async function startAnalysis() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;

  setError('');
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('player-section').style.display = 'none';
  document.getElementById('progress-section').style.display = 'block';
  setProgress(0, 'Enviando al servidor...');

  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url})
  });
  const data = await res.json();

  if (data.error) { setError(data.error); return; }

  currentHash = data.hash;

  if (data.status === 'done') {
    await loadPlayer(currentHash);
  } else {
    pollStatus(currentHash);
  }
}

function pollStatus(hash) {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    const res = await fetch(`/api/status/${hash}`);
    const data = await res.json();
    setProgress(data.progress || 0, data.message || '...');
    if (data.status === 'done') {
      clearInterval(pollInterval);
      await loadPlayer(hash);
    } else if (data.status === 'error') {
      clearInterval(pollInterval);
      setError(data.message);
      document.getElementById('analyze-btn').disabled = false;
    }
  }, 800);
}

async function loadPlayer(hash) {
  const res = await fetch(`/api/graph/${hash}`);
  graph = await res.json();

  // Set UI
  document.getElementById('track-title').textContent = graph.meta?.title || 'Unknown';
  document.getElementById('track-info').textContent =
    `${formatTime(graph.duration)} · ${graph.n_beats} beats · ${Math.round(graph.tempo)} BPM`;
  document.getElementById('thumb').src = graph.meta?.thumbnail || '';
  document.getElementById('stat-tempo').textContent = Math.round(graph.tempo) + ' BPM';

  // Decode audio into memory for gap-free playback
  ensureAudioCtx();
  setProgress(98, 'Decodificando audio...');
  const audioResp = await fetch(`/api/audio/${hash}`);
  const audioAB   = await audioResp.arrayBuffer();
  audioBuffer = await audioCtx.decodeAudioData(audioAB);
  offsetAtStart = 0;

  // Draw waveform
  drawWaveform();

  document.getElementById('progress-section').style.display = 'none';
  document.getElementById('player-section').style.display = 'block';
  document.getElementById('analyze-btn').disabled = false;

  currentBeat = 0;
  jumpCount = 0;
  totalTime = 0;
  progressionPhase = 0;
  lastJumpTime = -Infinity;
  lastEligibleJumpBeat = -1;

  // Init circular jump map
  disabledJumps.clear();
  recentJumps.length = 0;
  resizeCircle();
  buildCircleStatic();
  renderCircle();
}

// ──────────────────────────────────────────────
// ENGINE — The heart of the system
// ──────────────────────────────────────────────
document.getElementById('play-btn').addEventListener('click', togglePlay);

function togglePlay() {
  if (!graph) return;
  if (isPlaying) {
    pausePlayback();
    isPlaying = false;
    document.getElementById('play-btn').textContent = '▶';
    if (engineInterval) clearInterval(engineInterval);
    renderCircle();
  } else {
    isPlaying = true;
    startPlayback(offsetAtStart);
    document.getElementById('play-btn').textContent = '⏸';
    startEngine();
    startCircleLoop();
  }
}

function startEngine() {
  if (engineInterval) clearInterval(engineInterval);

  // Align to current beat start if drifted
  const beatTime = graph.beat_times[currentBeat];
  if (Math.abs(getCurrentTime() - beatTime) > 0.5) {
    startPlayback(beatTime);
  }

  engineInterval = setInterval(engineTick, 40);
}

// How many seconds ahead of the beat boundary we fire the seek.
// Compensates for setInterval jitter + browser audio seek latency (~60-80ms).
const SEEK_LOOKAHEAD_S = 0.08;

// Anti-recency memory — track recently visited beat zones
const recentlyVisited = [];  // ring buffer of beat indices
const RECENCY_WINDOW = 20;   // remember last 20 jumps

// Disabled jumps — Set of "from:to" strings
const disabledJumps = new Set();
function jumpKey(from, to) { return `${from}:${to}`; }
function isJumpDisabled(from, to) { return disabledJumps.has(jumpKey(from, to)); }

function engineTick() {
  if (!graph || !isPlaying) return;

  const now = getCurrentTime();
  totalTime = now;

  // Real song position (0 = start, 1 = end)
  progressionPhase = graph.duration > 0 ? Math.min(1, now / graph.duration) : 0;

  // Find current beat
  const times = graph.beat_times;
  let cb = currentBeat;
  while (cb < times.length - 1 && times[cb + 1] <= now) cb++;
  currentBeat = cb;

  const nextBeatTime = times[Math.min(cb + 1, times.length - 1)];
  const timeToNext = nextBeatTime - now;
  if (timeToNext > SEEK_LOOKAHEAD_S) { updateUI(); return; }

  // ── Jump decision ─────────────────────────────────────────────────────
  // Two controls:
  //   minSim    → quality gate (only near-identical segments)
  //   minBeatGap → minimum seconds between jumps (gap in real time, not ticks)
  const isDownbeat = graph.is_downbeat ? graph.is_downbeat[cb] : (cb % 4 === 0);
  const nearBoundary = isNearSegmentBoundary(cb, 4);

  const jumps = graph.jump_graph[cb] || [];
  const srcType = getSegmentType(cb);
  const bestJump = jumps
    .filter(j => j.similarity >= minSim && j.to < cb)   // backward jumps only
    .sort((a, b) => {
      // Boost cross-type destinations: landing in a different section feels
      // like musical progression rather than repetition.
      const bonus = (t) => (t.segment !== srcType && srcType >= 0) ? 0.15 : 0;
      return (b.similarity + bonus(b)) - (a.similarity + bonus(a));
    })[0];
  const validJump = (bestJump && !isJumpDisabled(cb, bestJump.to)) ? bestJump : null;

  // Convert minBeatGap (beats) → seconds using actual tempo
  const beatDuration = 60 / (graph.tempo || 120);
  const minGapSeconds = minBeatGap * beatDuration;
  const timeSinceLastJump = now - lastJumpTime;

  // Structural gate: quality conditions that must always hold
  const structuralGate = isDownbeat && nearBoundary && validJump !== null;

  // Find (and cache) the highest-indexed eligible jump source beat.
  // Only THAT beat bypasses the time-gap and probability — so if there are
  // two jump lines near the end, only the last one always fires.
  if (lastEligibleJumpBeat < 0) {
    const isDB = graph.is_downbeat || [];
    let last = -1;
    for (const fromStr of Object.keys(graph.jump_graph)) {
      const from = parseInt(fromStr);
      if (!isDB[from] || !isNearSegmentBoundary(from, 4)) continue;
      if ((graph.jump_graph[from] || []).some(j => j.similarity >= minSim && j.to < from) && from > last)
        last = from;
    }
    lastEligibleJumpBeat = last;
  }
  const isLastJump = lastEligibleJumpBeat >= 0 && Math.abs(cb - lastEligibleJumpBeat) <= 2;

  // Probabilistic gate: lets the song sometimes pass through a jump point
  // so it can reach later sections. Each recent visit to the destination
  // zone reduces the probability, down to a floor of 15%.
  let jumpProb = 0.75;
  if (validJump) {
    const destZone = Math.min(validJump.to + 1, graph.beat_times.length - 1);
    const recentHits = recentlyVisited.filter(v => Math.abs(v - destZone) < 8).length;
    jumpProb = Math.max(0.15, 0.75 - recentHits * 0.25);
  }

  const canJump = structuralGate && (
    isLastJump ||                                       // last jump line always fires
    (timeSinceLastJump >= minGapSeconds && Math.random() < jumpProb)
  );

  if (canJump) {
    performJump(validJump);
    lastJumpTime = now;
  }

  updateUI();
}

// Returns the segment archetype index (0=INTRO,1=VERSE,2=CHORUS,3=BRIDGE) for a beat,
// or -1 if not found. Matches the `segment` field stored on each jump object.
function getSegmentType(beat) {
  const segs = graph.segments || [];
  for (let i = 0; i < segs.length; i++) {
    if (beat >= segs[i].start && beat < segs[i].end) return i % 4;
  }
  return -1;
}

function isNearSegmentBoundary(beat, tolerance) {
  const boundaries = graph.segment_boundaries;
  if (!boundaries || boundaries.length === 0) {
    // No structural data: fall back to every 16 beats (4 bars)
    return beat % 16 < tolerance || beat % 16 > 16 - tolerance;
  }
  return boundaries.some(b => Math.abs(b - beat) <= tolerance);
}


function performJump(jump) {
  const fromTime = getCurrentTime();
  const fromBeat = currentBeat;
  const srcNextBeatTime = graph.beat_times[Math.min(fromBeat + 1, graph.beat_times.length - 1)];

  // The graph matches beat cb ≅ jump.to. But we cross into cb+1, so we land at
  // jump.to+1 (which sounds like cb+1 because the context windows are aligned).
  const destIndex = Math.min(jump.to + 1, graph.beat_times.length - 1);
  const destBeatStart = (graph.beat_times[destIndex] || 0) + jumpDestOffset;

  seamlessJump(Math.max(0, destBeatStart), srcNextBeatTime);
  currentBeat = destIndex;
  jumpCount++;
  lastSimilarity = jump.similarity;

  // Animate this jump on the circular map
  recentJumps.push({ from: fromBeat, to: jump.to, ts: performance.now() });
  startCircleLoop();

  // Update anti-recency ring buffer
  recentlyVisited.push(jump.to);
  if (recentlyVisited.length > RECENCY_WINDOW) recentlyVisited.shift();

  logJump(fromTime, targetTime, jump.similarity, jump.is_downbeat);
  document.getElementById('jump-count').textContent = jumpCount;
  document.getElementById('stat-sim').textContent = Math.round(jump.similarity * 100) + '%';
}

function logJump(from, to, sim) {
  const log = document.getElementById('jump-log');
  const entry = document.createElement('div');
  entry.className = 'entry';
  entry.innerHTML = `
    <span class="ts">${formatTime(from)} →</span>
    <span>${formatTime(to)}</span>
    <span class="sim">${Math.round(sim*100)}%</span>
    <span style="color:#444;margin-left:auto">${getModeLabel()}</span>
  `;
  log.insertBefore(entry, log.firstChild);
  if (log.children.length > 40) log.removeChild(log.lastChild);
}

function getModeLabel() {
  // Use actual segment data if available
  if (graph && graph.segments && graph.segments.length > 0) {
    const segs = graph.segments;
    for (const seg of segs) {
      if (currentBeat >= seg.start && currentBeat < seg.end) {
        return seg.label || '—';
      }
    }
  }
  if (progressionPhase < 0.25) return 'INTRO';
  if (progressionPhase < 0.55) return 'DEVELOPMENT';
  if (progressionPhase < 0.80) return 'LOOP';
  return 'INFINITE';
}

// ──────────────────────────────────────────────
// UI UPDATE
// ──────────────────────────────────────────────
function _origUpdateUI() {
  if (!graph) return;
  const now = getCurrentTime();
  document.getElementById('time-display').textContent = formatTime(now);
  document.getElementById('stat-beat').textContent =
    currentBeat + 1 + ' / ' + graph.n_beats;
  document.getElementById('stat-mode').textContent = getModeLabel();
  drawWaveformPosition(now);
}

// ──────────────────────────────────────────────
// CANVAS — Beat/energy waveform
// ──────────────────────────────────────────────
function drawWaveform() {
  if (!graph) return;
  const canvas = document.getElementById('beat-canvas');
  const W = canvas.width = canvas.offsetWidth * window.devicePixelRatio;
  const H = canvas.height = 80 * window.devicePixelRatio;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, W, H);

  const rms = graph.beat_rms;
  const energy = graph.energy_arc || rms;
  const maxRms = Math.max(...rms) || 1;
  const maxE = Math.max(...energy) || 1;
  const n = rms.length;
  const isDownbeats = graph.is_downbeat || [];
  const segments = graph.segments || [];

  // Draw segment backgrounds
  for (const seg of segments) {
    const segColors = ['rgba(100,30,150,0.08)','rgba(30,80,150,0.08)','rgba(150,80,30,0.08)','rgba(30,150,100,0.08)'];
    const sx = (seg.start / n) * W;
    const sw = ((seg.end - seg.start) / n) * W;
    ctx.fillStyle = segColors[segments.indexOf(seg) % 4];
    ctx.fillRect(sx, 0, sw, H);
  }

  // Draw RMS bars
  for (let i = 0; i < n; i++) {
    const x = (i / n) * W;
    const w = Math.max(1, (W / n) - 0.5);
    const h = (rms[i] / maxRms) * (H * 0.7);
    const y = H - h;
    // Downbeat marker
    const isDB = isDownbeats[i];
    ctx.fillStyle = isDB ? 'rgba(232,255,71,0.08)' : '#0f0f0f';
    ctx.fillRect(x, 0, w, H);
    ctx.fillStyle = isDB ? 'rgba(232,255,71,0.5)' : 'rgba(232,255,71,0.22)';
    ctx.fillRect(x, y, w, h);
  }

  // Draw energy arc on top
  if (graph.energy_arc) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,77,109,0.6)';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < n; i++) {
      const x = (i / n) * W + W / n / 2;
      const y = H - (energy[i] / maxE) * H * 0.85;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function drawWaveformPosition(currentTime) {
  if (!graph) return;
  const canvas = document.getElementById('beat-canvas');
  const W = canvas.width;
  const H = canvas.height;
  const ctx = canvas.getContext('2d');

  drawWaveform();

  // Highlight current position
  const pos = (currentTime / graph.duration) * W;
  ctx.fillStyle = '#fff';
  ctx.fillRect(pos - 1, 0, 2, H);

  // Show jump graph for current beat — highlight reachable beats
  const jumps = graph.jump_graph[currentBeat] || [];
  for (const j of jumps) {
    const jt = j.time || graph.beat_times[j.to] || 0;
    const jx = (jt / graph.duration) * W;
    const alpha = j.similarity * 0.7;
    ctx.fillStyle = `rgba(255,77,109,${alpha})`;
    ctx.fillRect(jx - 1, 0, 2, H);
  }
}

// Click on canvas to seek
document.getElementById('beat-canvas').addEventListener('click', (e) => {
  if (!graph) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const ratio = (e.clientX - rect.left) / rect.width;
  seekTo(ratio * graph.duration);
  syncBeatToTime();
  lastJumpTime = -Infinity;
});

// ──────────────────────────────────────────────
// UTILS
// ──────────────────────────────────────────────
function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2,'0')}`;
}

function setProgress(pct, msg) {
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-msg').textContent = msg;
}

function setError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
  if (msg) document.getElementById('analyze-btn').disabled = false;
}

window.addEventListener('resize', () => {
  if (!graph) return;
  drawWaveform();
  resizeCircle();
  buildCircleStatic();
  renderCircle();
});

// ──────────────────────────────────────────────
// CIRCULAR JUMP MAP
// ──────────────────────────────────────────────
// The whole song is a circle: start at the top (12 o'clock), running clockwise
// to the end. A playhead travels the rim as it plays. Every possible jump is a
// chord linking its origin point to its destination point on the rim, so you
// can see, enable/disable and watch jumps happen. Click a chord to toggle it;
// click the rim to seek.

let circleCtx = null;
let circleDims = null;        // {dpr, size, cx, cy, R}
let circleStatic = null;      // cached offscreen canvas (rim + segments + chords)
let circleRAF = null;
const recentJumps = [];       // {from, to, ts} — "jump in progress" animation
const JUMP_ANIM_MS = 900;

function angleForTime(t) {
  const d = graph.duration || 1;
  return -Math.PI / 2 + 2 * Math.PI * (t / d);
}
function angleForBeat(i) {
  return angleForTime(graph.beat_times[i] ?? 0);
}
function ptOnRim(cx, cy, R, a) {
  return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
}
function chordControl(cx, cy, x1, y1, x2, y2) {
  // Control point pulled toward the center so chords bow inward.
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  return [cx + (mx - cx) * 0.35, cy + (my - cy) * 0.35];
}

// Arrowhead at the destination end of a quadratic bezier (P0→CP→P2).
// The tangent at t=1 is the direction from CP to P2.
function drawArrow(ctx, cpx, cpy, x2, y2, size) {
  const dx = x2 - cpx, dy = y2 - cpy;
  const len = Math.hypot(dx, dy);
  if (len < 0.001) return;
  const ux = dx / len, uy = dy / len;   // unit tangent
  const px = -uy, py = ux;              // perpendicular
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - ux * size + px * size * 0.45, y2 - uy * size + py * size * 0.45);
  ctx.lineTo(x2 - ux * size - px * size * 0.45, y2 - uy * size - py * size * 0.45);
  ctx.closePath();
  ctx.fill();
}
function quadPoint(p0, cp, p1, t) {
  const u = 1 - t;
  return [
    u * u * p0[0] + 2 * u * t * cp[0] + t * t * p1[0],
    u * u * p0[1] + 2 * u * t * cp[1] + t * t * p1[1],
  ];
}

function resizeCircle() {
  const canvas = document.getElementById('circle-canvas');
  if (!canvas) return;
  const wrap = canvas.parentElement;
  const cssSize = Math.max(280, Math.min((wrap.clientWidth || 480), 560));
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = cssSize + 'px';
  canvas.style.height = cssSize + 'px';
  canvas.width = Math.round(cssSize * dpr);
  canvas.height = Math.round(cssSize * dpr);
  circleCtx = canvas.getContext('2d');
  circleDims = { dpr, size: cssSize, cx: cssSize / 2, cy: cssSize / 2, R: cssSize / 2 - 38 };
}

const SEG_RING_COLORS = [
  'rgba(180,110,255,0.6)', 'rgba(90,150,255,0.6)',
  'rgba(255,150,80,0.6)', 'rgba(80,220,160,0.6)',
];

function buildCircleStatic() {
  if (!graph) return;
  if (!circleDims) resizeCircle();
  const { dpr, size, cx, cy, R } = circleDims;
  const off = document.createElement('canvas');
  off.width = size * dpr;
  off.height = size * dpr;
  const ctx = off.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const n = graph.n_beats;

  // Segment arcs coloured on the rim
  const segs = graph.segments || [];
  ctx.lineCap = 'butt';
  for (let si = 0; si < segs.length; si++) {
    const seg = segs[si];
    ctx.beginPath();
    ctx.strokeStyle = SEG_RING_COLORS[si % SEG_RING_COLORS.length];
    ctx.lineWidth = 6;
    ctx.arc(cx, cy, R, angleForBeat(seg.start), angleForBeat(Math.min(seg.end, n - 1)));
    ctx.stroke();
  }

  // Base rim
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Time ticks + labels every 30s
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.fillStyle = '#555';
  ctx.font = '9px "Space Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const dur = graph.duration || 0;
  for (let t = 0; t <= dur; t += 30) {
    const a = angleForTime(t);
    const [ix, iy] = ptOnRim(cx, cy, R - 4, a);
    const [ox, oy] = ptOnRim(cx, cy, R + 4, a);
    ctx.beginPath(); ctx.moveTo(ix, iy); ctx.lineTo(ox, oy); ctx.stroke();
    const [lx, ly] = ptOnRim(cx, cy, R + 16, a);
    ctx.fillText(formatTime(t), lx, ly);
  }

  // Only draw chords that the engine can actually fire:
  // source beat must be a downbeat near a segment boundary, similarity >= minSim.
  drawEngineChords(ctx, cx, cy, R);

  circleStatic = off;
}

function drawEngineChords(ctx, cx, cy, R) {
  // Mirror exactly the engine's canJump conditions (minus the runtime counters).
  // A beat is a valid jump source only if it's a downbeat AND near a boundary.
  const isDB = graph.is_downbeat || [];
  const jg   = graph.jump_graph;

  for (const [fromStr, jumps] of Object.entries(jg)) {
    const from = parseInt(fromStr);

    // Must be a downbeat near a segment boundary — same gate as the engine
    if (!isDB[from]) continue;
    if (!isNearSegmentBoundary(from, 4)) continue;

    const [x1, y1] = ptOnRim(cx, cy, R, angleForBeat(from));

    const srcType = getSegmentType(from);
    const best = jumps
      .filter(j => j.similarity >= minSim && j.to < from)
      .sort((a, b) => {
        const bonus = (t) => (t.segment !== srcType && srcType >= 0) ? 0.15 : 0;
        return (b.similarity + bonus(b)) - (a.similarity + bonus(a));
      })[0];

    if (!best) continue;

    const [x2, y2] = ptOnRim(cx, cy, R, angleForBeat(best.to));
    const [cpx, cpy] = chordControl(cx, cy, x1, y1, x2, y2);
    const color = isJumpDisabled(from, best.to)
      ? 'rgba(160,160,160,0.45)'
      : `rgba(232,255,71,${0.2 + best.similarity * 0.7})`;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(cpx, cpy, x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = isJumpDisabled(from, best.to) ? 1.2 : 1.5;
    ctx.stroke();
    ctx.fillStyle = color;
    drawArrow(ctx, cpx, cpy, x2, y2, 7);
  }
}

function renderCircle() {
  if (!graph || !circleCtx || !circleStatic) return;
  const ctx = circleCtx;
  const { dpr, size, cx, cy, R } = circleDims;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  // Cached static layer (rim + segments + all chords)
  ctx.drawImage(circleStatic, 0, 0, size, size);

  // Candidate jumps from the current beat — only shown when the engine could fire
  const isDB = graph.is_downbeat || [];
  const canFireNow = isDB[currentBeat] && isNearSegmentBoundary(currentBeat, 4);
  const [x1, y1] = ptOnRim(cx, cy, R, angleForBeat(currentBeat));
  if (canFireNow) for (const j of (graph.jump_graph[currentBeat] || [])) {
    if (isJumpDisabled(currentBeat, j.to) || j.similarity < minSim || j.to >= currentBeat) continue;
    const [x2, y2] = ptOnRim(cx, cy, R, angleForBeat(j.to));
    const [cpx, cpy] = chordControl(cx, cy, x1, y1, x2, y2);
    const col = `rgba(255,77,109,${0.25 + j.similarity * 0.6})`;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(cpx, cpy, x2, y2);
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.fillStyle = col;
    drawArrow(ctx, cpx, cpy, x2, y2, 6);
  }

  // Jump-in-progress animations
  const now = performance.now();
  for (let i = recentJumps.length - 1; i >= 0; i--) {
    const rj = recentJumps[i];
    const p = (now - rj.ts) / JUMP_ANIM_MS;
    if (p >= 1) { recentJumps.splice(i, 1); continue; }
    const pf = ptOnRim(cx, cy, R, angleForBeat(rj.from));
    const pt = ptOnRim(cx, cy, R, angleForBeat(rj.to));
    const cp = chordControl(cx, cy, pf[0], pf[1], pt[0], pt[1]);
    ctx.beginPath();
    ctx.moveTo(pf[0], pf[1]);
    ctx.quadraticCurveTo(cp[0], cp[1], pt[0], pt[1]);
    ctx.strokeStyle = `rgba(255,255,255,${0.9 * (1 - p)})`;
    ctx.lineWidth = 2;
    ctx.stroke();
    const [dx, dy] = quadPoint(pf, cp, pt, p);
    ctx.beginPath();
    ctx.arc(dx, dy, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
  }

  // Playhead on the rim
  const [px, py] = ptOnRim(cx, cy, R, angleForTime(getCurrentTime()));
  ctx.beginPath();
  ctx.arc(px, py, 8, 0, 2 * Math.PI);
  ctx.fillStyle = 'rgba(255,77,109,0.18)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
  ctx.fillStyle = '#ff4d6d';
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#fff';
  ctx.stroke();

  // Center readout
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#e8ff47';
  ctx.font = '22px "Bebas Neue", sans-serif';
  ctx.fillText(formatTime(getCurrentTime()), cx, cy - 8);
  ctx.fillStyle = '#666';
  ctx.font = '10px "Space Mono", monospace';
  ctx.fillText(`${jumpCount} saltos`, cx, cy + 12);
}

function circleLoop() {
  renderCircle();
  if (isPlaying || recentJumps.length) {
    circleRAF = requestAnimationFrame(circleLoop);
  } else {
    circleRAF = null;
  }
}
function startCircleLoop() {
  if (!circleRAF) circleRAF = requestAnimationFrame(circleLoop);
}

function nearestChord(mx, my) {
  const { cx, cy, R } = circleDims;
  const isDB = graph.is_downbeat || [];
  let best = null, bestD = 12;  // px hit threshold
  for (const [fromStr, jumps] of Object.entries(graph.jump_graph)) {
    const from = parseInt(fromStr);
    if (!isDB[from] || !isNearSegmentBoundary(from, 4)) continue;
    // Only the single best chord is drawn — only that one is clickable
    const srcType = getSegmentType(from);
    const j = jumps
      .filter(x => x.similarity >= minSim && x.to < from)
      .sort((a, b) => {
        const bonus = (t) => (t.segment !== srcType && srcType >= 0) ? 0.15 : 0;
        return (b.similarity + bonus(b)) - (a.similarity + bonus(a));
      })[0];
    if (!j) continue;
    const pf = ptOnRim(cx, cy, R, angleForBeat(from));
    const pt = ptOnRim(cx, cy, R, angleForBeat(j.to));
    const cp = chordControl(cx, cy, pf[0], pf[1], pt[0], pt[1]);
    for (let s = 1; s < 12; s++) {
      const [qx, qy] = quadPoint(pf, cp, pt, s / 12);
      const d = Math.hypot(qx - mx, qy - my);
      if (d < bestD) { bestD = d; best = { from, to: j.to }; }
    }
  }
  return best;
}

function syncBeatToTime() {
  const times = graph.beat_times;
  let cb = 0;
  while (cb < times.length - 1 && times[cb + 1] <= getCurrentTime()) cb++;
  currentBeat = cb;
}

// Click: toggle a chord, or seek on the rim
document.getElementById('circle-canvas')?.addEventListener('click', (e) => {
  if (!graph || !circleDims) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (circleDims.size / rect.width);
  const my = (e.clientY - rect.top) * (circleDims.size / rect.height);
  const { cx, cy, R } = circleDims;
  const dist = Math.hypot(mx - cx, my - cy);

  // Rim click → seek
  if (Math.abs(dist - R) < 16) {
    let frac = (Math.atan2(my - cy, mx - cx) + Math.PI / 2) / (2 * Math.PI);
    frac = ((frac % 1) + 1) % 1;
    seekTo(frac * graph.duration);
    syncBeatToTime();
    lastJumpTime = -Infinity;
    renderCircle();
    return;
  }

  // Chord click → toggle that jump
  const hit = nearestChord(mx, my);
  if (!hit) return;
  const key = jumpKey(hit.from, hit.to);
  if (disabledJumps.has(key)) disabledJumps.delete(key);
  else disabledJumps.add(key);
  buildCircleStatic();
  renderCircle();

  const j = (graph.jump_graph[hit.from] || []).find(x => x.to === hit.to);
  const state = disabledJumps.has(key) ? '🔴 desactivado' : '🟢 activado';
  document.getElementById('pe-info').innerHTML =
    `Beat <strong>${hit.from}</strong> (${formatTime(graph.beat_times[hit.from] || 0)}) → ` +
    `<strong>${hit.to}</strong> (${formatTime(graph.beat_times[hit.to] || 0)}) · ` +
    `Sim <strong>${Math.round((j ? j.similarity : 0) * 100)}%</strong> · ${state}`;
});

// Hover tooltip over chords
document.getElementById('circle-canvas')?.addEventListener('mousemove', (e) => {
  if (!graph || !circleDims) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (circleDims.size / rect.width);
  const my = (e.clientY - rect.top) * (circleDims.size / rect.height);
  const hit = nearestChord(mx, my);
  e.currentTarget.style.cursor = hit ? 'pointer' : 'crosshair';
  if (!hit) return;
  const j = (graph.jump_graph[hit.from] || []).find(x => x.to === hit.to);
  const dis = isJumpDisabled(hit.from, hit.to) ? ' · <span style="color:#ff4d6d">DESACTIVADO</span>' : '';
  document.getElementById('pe-info').innerHTML =
    `Beat <strong>${hit.from}</strong> (${formatTime(graph.beat_times[hit.from] || 0)}) → ` +
    `<strong>${hit.to}</strong> (${formatTime(graph.beat_times[hit.to] || 0)}) · ` +
    `Sim <strong>${Math.round((j ? j.similarity : 0) * 100)}%</strong>${dis}`;
});

function peEnableAll() {
  disabledJumps.clear();
  buildCircleStatic();
  renderCircle();
  document.getElementById('pe-info').textContent = 'Todos los saltos activados.';
}

function peDisableAll() {
  if (!graph) return;
  for (const [from, jumps] of Object.entries(graph.jump_graph)) {
    for (const j of jumps) disabledJumps.add(jumpKey(parseInt(from), j.to));
  }
  buildCircleStatic();
  renderCircle();
  document.getElementById('pe-info').textContent =
    'Todos los saltos desactivados. El motor seguirá hacia adelante sin saltar.';
}

// Redraw circle on each engine tick (cheap — the heavy layer is cached)
function updateUI() {
  _origUpdateUI();
  if (!isPlaying) renderCircle();
}
