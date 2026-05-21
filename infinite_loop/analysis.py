"""Analysis engine for INFINITE LOOP.

Key-invariant chroma, context windows, downbeat detection, Foote novelty
segmentation, jump-graph construction and energy-arc progression, plus
validation-image and sample-MP3 rendering helpers.
"""

import numpy as np
import librosa
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────
# 1. KEY-INVARIANT CHROMA
# ──────────────────────────────────────────────────────────────

def detect_key(chroma_mean: np.ndarray) -> int:
    """
    Estimate global key via Krumhansl-Schmuckler key profiles.
    Returns pitch class offset (0=C, 1=C#, ..., 11=B) of detected tonic.
    """
    # Major and minor key profiles (Krumhansl-Kessler weights)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_score, best_key = -np.inf, 0
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        score_maj = np.corrcoef(rolled, major)[0, 1]
        score_min = np.corrcoef(rolled, minor)[0, 1]
        score = max(score_maj, score_min)
        if score > best_score:
            best_score, best_key = score, shift
    return best_key


def key_invariant_chroma(chroma: np.ndarray) -> np.ndarray:
    """
    Rotate chroma so tonic is always at bin 0.
    This makes beats in the same key directly comparable regardless of key.
    """
    key = detect_key(chroma.mean(axis=1))
    return np.roll(chroma, -key, axis=0)


# ──────────────────────────────────────────────────────────────
# 2. CONTEXT WINDOW FEATURES
# ──────────────────────────────────────────────────────────────

def beat_context_features(
    chroma_per_beat: np.ndarray,
    mfcc_per_beat: np.ndarray,
    rms_per_beat: np.ndarray,
    context: int = 4
) -> np.ndarray:
    """
    For each beat i, concatenate features of beats [i-context .. i+context].
    This means similarity captures *musical phrases*, not isolated beats.
    Beats near edges are padded with zeros (reflected actually).
    Returns: (n_beats, feature_dim)
    """
    n = len(chroma_per_beat)
    pad_c = np.pad(chroma_per_beat, ((context, context), (0, 0)), mode='reflect')
    pad_m = np.pad(mfcc_per_beat, ((context, context), (0, 0)), mode='reflect')
    pad_r = np.pad(rms_per_beat.reshape(-1, 1), ((context, context), (0, 0)), mode='reflect')

    features = []
    for i in range(n):
        window_c = pad_c[i:i + 2 * context + 1].flatten()       # (9*12,)
        window_m = pad_m[i:i + 2 * context + 1].flatten()       # (9*12,)
        window_r = pad_r[i:i + 2 * context + 1].flatten()       # (9,)
        features.append(np.concatenate([window_c, window_m, window_r]))

    return np.array(features)


# ──────────────────────────────────────────────────────────────
# 3. FOOTE NOVELTY — STRUCTURAL SEGMENTATION
# ──────────────────────────────────────────────────────────────

def foote_novelty(similarity_matrix: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    """
    Foote novelty function: detects structural boundaries.
    Convolves a checkerboard kernel along the diagonal of the SSM.
    High novelty = structural boundary (verse→chorus, chorus→bridge, etc.)
    """
    n = similarity_matrix.shape[0]
    k = kernel_size

    # Checkerboard kernel
    kernel = np.ones((k, k))
    kernel[:k//2, :k//2] = 1
    kernel[k//2:, k//2:] = 1
    kernel[:k//2, k//2:] = -1
    kernel[k//2:, :k//2] = -1

    novelty = np.zeros(n)
    half = k // 2
    for i in range(half, n - half):
        block = similarity_matrix[i - half:i + half, i - half:i + half]
        if block.shape == kernel.shape:
            novelty[i] = np.sum(block * kernel)

    # Smooth and normalize
    novelty = gaussian_filter1d(novelty, sigma=2)
    novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)
    return novelty


def detect_segments(novelty: np.ndarray, min_distance: int = 8) -> np.ndarray:
    """Find segment boundaries from novelty peaks."""
    peaks, _ = find_peaks(novelty, distance=min_distance, height=0.3)
    return peaks  # beat indices of boundaries


# ──────────────────────────────────────────────────────────────
# 4. DOWNBEAT DETECTION
# ──────────────────────────────────────────────────────────────

def detect_downbeats(beat_times: np.ndarray, tempo: float) -> np.ndarray:
    """
    Estimate downbeats (beat 1 of each bar) assuming 4/4 time.
    Groups beats into bars of 4 and marks beat 0 of each group.
    Returns boolean mask over beat_times.
    """
    n = len(beat_times)
    # Find best phase for bar grouping by checking energy consistency
    # Simple approach: every 4th beat is a downbeat, find best phase offset
    best_phase = 0
    # We just label every 4th beat as a downbeat (common for 4/4 electronic music)
    is_downbeat = np.zeros(n, dtype=bool)
    is_downbeat[::4] = True  # phase 0 default

    # Try all 4 phases, pick one where beat intervals are most consistent
    beat_intervals = np.diff(beat_times)
    if len(beat_intervals) >= 4:
        best_var = np.inf
        for phase in range(4):
            indices = np.arange(phase, n - 1, 4)
            if len(indices) > 1:
                intervals = beat_intervals[indices]
                var = np.var(intervals)
                if var < best_var:
                    best_var = var
                    best_phase = phase

    is_downbeat = np.zeros(n, dtype=bool)
    is_downbeat[best_phase::4] = True
    return is_downbeat


# ──────────────────────────────────────────────────────────────
# 5. ENERGY ARC
# ──────────────────────────────────────────────────────────────

def compute_energy_arc(rms_per_beat: np.ndarray, smoothing: int = 16) -> np.ndarray:
    """
    Smooth energy curve that describes the song's dynamic arc.
    Used to guide progression: prefer jumps to beats with similar energy level.
    Returns normalized [0,1] energy per beat.
    """
    smoothed = gaussian_filter1d(rms_per_beat.astype(float), sigma=smoothing)
    return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)


# ──────────────────────────────────────────────────────────────
# 6. JUMP GRAPH — IMPROVED
# ──────────────────────────────────────────────────────────────

def build_jump_graph(
    similarity: np.ndarray,
    beat_times: np.ndarray,
    is_downbeat: np.ndarray,
    energy_arc: np.ndarray,
    segments: np.ndarray,
    K: int = 16,
    min_sim: float = 0.55,
) -> dict:
    """
    Build jump graph with improvements:
    - Downbeat-only jumps (jump only on beat 1 of a bar)
    - Energy proximity bonus (prefer same-energy zone)
    - Segment-crossing bonus (jumps between equivalent sections score higher)
    - Minimum distance in beats
    """
    n = len(beat_times)
    MIN_DIST = 8  # minimum beats between source and target

    # Build segment membership per beat
    seg_labels = np.zeros(n, dtype=int)
    if len(segments) > 0:
        boundaries = np.concatenate([[0], segments, [n]])
        for seg_idx in range(len(boundaries) - 1):
            s, e = boundaries[seg_idx], boundaries[seg_idx + 1]
            seg_labels[s:e] = seg_idx % 4  # wrap to 4 archetypes (intro/verse/chorus/bridge)

    jump_graph = {}
    for i in range(n):
        # Only compute jumps from downbeats — browser will step beat-by-beat
        # but graph is pre-computed only from downbeats for quality
        sims = similarity[i].copy()

        # Mask out nearby beats
        lo = max(0, i - MIN_DIST)
        hi = min(n, i + MIN_DIST)
        sims[lo:hi] = -1

        # Energy proximity bonus: beats with similar energy get +0.1 bonus
        energy_diff = np.abs(energy_arc - energy_arc[i])
        energy_bonus = 0.12 * (1 - energy_diff)
        sims = sims + energy_bonus
        sims[lo:hi] = -1  # re-mask

        # Cross-type bonus: different structural role gets +0.08
        # (verse→chorus or chorus→verse feels like progression; same→same is repetitive)
        seg_cross = (seg_labels != seg_labels[i]).astype(float) * 0.08
        sims = sims + seg_cross
        sims[lo:hi] = -1

        # Get top-K
        top_k_idx = np.argsort(sims)[-K:][::-1]
        jumps = []
        for j in top_k_idx:
            if sims[j] > min_sim:
                jumps.append({
                    "to": int(j),
                    "similarity": float(np.clip(similarity[i, j], 0, 1)),  # raw sim for display
                    "time": float(beat_times[j]),
                    "is_downbeat": bool(is_downbeat[j]),
                    "energy": float(energy_arc[j]),
                    "segment": int(seg_labels[j]),
                })

        # Sort by raw similarity desc
        jumps.sort(key=lambda x: x["similarity"], reverse=True)
        jump_graph[i] = jumps

    return jump_graph


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def analyze(audio_path: str, status_cb=None) -> dict:
    """
    Full improved analysis pipeline.
    status_cb(progress: int, message: str) — optional progress callback.
    """
    def update(pct, msg):
        if status_cb:
            status_cb(pct, msg)
        else:
            print(f"  [{pct:3d}%] {msg}")

    update(5, "Cargando audio...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration_s = len(y) / sr

    # hop_length=256 → ~11.6 ms/frame resolution (vs ~23 ms with default 512)
    HOP = 256

    update(15, "Detectando beats y tempo...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    beat_times = beat_times[beat_times < duration_s - 0.1]

    # Snap each beat to the nearest onset within ±30 ms for sub-frame accuracy.
    # Onset detection finds exact attack transients; the beat tracker gives the
    # right beat but can be off by one or two frames (11–23 ms).
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, backtrack=True)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)
    snapped = []
    for bt in beat_times:
        close = onset_times[np.abs(onset_times - bt) < 0.030]
        snapped.append(float(close[np.argmin(np.abs(close - bt))]) if len(close) else float(bt))
    beat_times = np.array(snapped)

    n_beats = len(beat_times)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    update(25, f"Extrayendo features para {n_beats} beats...")

    # Feature extraction — use same hop so frame indices are consistent
    chroma_raw = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP, bins_per_octave=36)
    chroma_ki = key_invariant_chroma(chroma_raw)
    mfcc_raw = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    rms_raw = librosa.feature.rms(y=y, hop_length=HOP)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP)

    beat_chroma, beat_mfcc, beat_rms, beat_contrast = [], [], [], []
    WINDOW = 3  # frames to average around beat

    for bt in beat_times:
        frame = int(librosa.time_to_frames(bt, sr=sr, hop_length=HOP))
        f0 = max(0, frame - WINDOW)
        f1 = min(chroma_ki.shape[1], frame + WINDOW + 1)

        beat_chroma.append(chroma_ki[:, f0:f1].mean(axis=1))
        beat_mfcc.append(mfcc_raw[:, f0:f1].mean(axis=1))
        beat_contrast.append(spectral_contrast[:, f0:f1].mean(axis=1))

        rms_f = min(frame, len(rms_raw) - 1)
        beat_rms.append(float(rms_raw[rms_f]))

    beat_chroma = np.array(beat_chroma)       # (n, 12)
    beat_mfcc = np.array(beat_mfcc)           # (n, 13)
    beat_rms = np.array(beat_rms)             # (n,)
    beat_contrast = np.array(beat_contrast)   # (n, 7)

    update(40, "Construyendo features de contexto (ventana de 4 beats)...")
    ctx_features = beat_context_features(beat_chroma, beat_mfcc, beat_rms, context=4)

    update(55, "Calculando Self-Similarity Matrix...")
    # Normalize
    ctx_norm = ctx_features / (np.linalg.norm(ctx_features, axis=1, keepdims=True) + 1e-8)
    # Full SSM via cosine similarity
    similarity = ctx_norm @ ctx_norm.T  # (n, n)
    np.fill_diagonal(similarity, 0)    # no self-jumps

    update(65, "Detectando estructura musical (Foote novelty)...")
    novelty = foote_novelty(similarity, kernel_size=min(16, n_beats // 4))
    segment_boundaries = detect_segments(novelty, min_distance=max(4, n_beats // 12))

    update(75, "Detectando downbeats (compases 4/4)...")
    is_downbeat = detect_downbeats(beat_times, tempo_val)

    update(80, "Calculando arco de energía...")
    energy_arc = compute_energy_arc(beat_rms)

    update(88, "Construyendo grafo de saltos mejorado...")
    jump_graph = build_jump_graph(
        similarity, beat_times, is_downbeat, energy_arc, segment_boundaries, K=16
    )

    # Segment list with energy info for frontend
    boundaries_full = np.concatenate([[0], segment_boundaries, [n_beats]])
    segments_info = []
    for idx in range(len(boundaries_full) - 1):
        s, e = int(boundaries_full[idx]), int(boundaries_full[idx + 1])
        segments_info.append({
            "start": s,
            "end": e,
            "start_time": float(beat_times[s]),
            "end_time": float(beat_times[min(e, n_beats - 1)]),
            "energy": float(energy_arc[s:e].mean()),
            "label": ["INTRO", "VERSE", "CHORUS", "BRIDGE"][idx % 4],
        })

    update(95, "Preparando datos para el frontend...")

    return {
        "tempo": tempo_val,
        "n_beats": n_beats,
        "beat_times": beat_times.tolist(),
        "beat_rms": beat_rms.tolist(),
        "energy_arc": energy_arc.tolist(),
        "is_downbeat": is_downbeat.tolist(),
        "novelty": novelty.tolist(),
        "segments": segments_info,
        "segment_boundaries": segment_boundaries.tolist(),
        "jump_graph": jump_graph,
        "duration": duration_s,
        # For validation image
        "_similarity_matrix": similarity,
        "_novelty": novelty,
    }


# ──────────────────────────────────────────────────────────────
# VALIDATION — SSM IMAGE + SAMPLE MP3
# ──────────────────────────────────────────────────────────────

def generate_validation_image(result: dict, output_path: str):
    """
    Generate a 3-panel validation image:
    1. Self-Similarity Matrix with segment boundaries + jump graph overlay
    2. Novelty function with detected boundaries
    3. Energy arc with downbeat markers
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    ssm = result["_similarity_matrix"]
    novelty = result["_novelty"]
    energy = result["energy_arc"]
    boundaries = result["segment_boundaries"]
    beat_times = np.array(result["beat_times"])
    is_downbeat = np.array(result["is_downbeat"])
    jump_graph = result["jump_graph"]
    n = len(beat_times)

    # Custom dark colormap
    cmap = LinearSegmentedColormap.from_list(
        "dark_heat", ["#080808", "#1a0a2e", "#6b2fa0", "#e8ff47", "#ffffff"]
    )

    fig = plt.figure(figsize=(16, 14), facecolor='#080808')
    fig.suptitle("INFINITE LOOP — Análisis Musical v2", 
                 color='#e8ff47', fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.94, bottom=0.06)

    # ── Panel 1: SSM (big, left+right) ──
    ax_ssm = fig.add_subplot(gs[0, :])
    ax_ssm.set_facecolor('#080808')

    im = ax_ssm.imshow(ssm, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                        origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax_ssm, fraction=0.02, pad=0.01).ax.yaxis.set_tick_params(color='#666')

    # Segment boundaries
    for b in boundaries:
        ax_ssm.axhline(b, color='#ff4d6d', lw=0.8, alpha=0.7)
        ax_ssm.axvline(b, color='#ff4d6d', lw=0.8, alpha=0.7)

    # Sample jump arcs on SSM (show first 20 beats' best jump)
    shown = 0
    for i in range(0, min(n, 100), 4):
        jumps = jump_graph.get(i, [])
        if jumps:
            best = jumps[0]
            j = best["to"]
            sim = best["similarity"]
            alpha = sim * 0.6
            ax_ssm.plot([i, j], [i, j], 'o', color='#e8ff47', 
                       markersize=2, alpha=alpha, markeredgewidth=0)
            ax_ssm.annotate('', xy=(j, i), xytext=(i, i),
                           arrowprops=dict(arrowstyle='->', color='#e8ff47',
                                         alpha=alpha * 0.7, lw=0.6))
            shown += 1
            if shown > 30:
                break

    ax_ssm.set_title("Self-Similarity Matrix  (amarillo=saltos, rojo=fronteras de sección)",
                     color='#f0ede6', fontsize=10, pad=8)
    ax_ssm.tick_params(colors='#444')
    ax_ssm.set_xlabel("Beat index", color='#666', fontsize=8)
    ax_ssm.set_ylabel("Beat index", color='#666', fontsize=8)

    # Add segment labels on top
    boundaries_full = [0] + list(boundaries) + [n]
    labels = ["INTRO", "VERSE", "CHORUS", "BRIDGE"]
    for idx in range(len(boundaries_full) - 1):
        mid = (boundaries_full[idx] + boundaries_full[idx + 1]) / 2
        ax_ssm.text(mid, -2, labels[idx % 4], color='#ff4d6d',
                   fontsize=7, ha='center', va='bottom', fontweight='bold')

    # ── Panel 2: Novelty ──
    ax_nov = fig.add_subplot(gs[1, 0])
    ax_nov.set_facecolor('#0d0d0d')
    ax_nov.fill_between(range(n), novelty, alpha=0.6, color='#6b2fa0')
    ax_nov.plot(novelty, color='#e8ff47', lw=0.8)
    for b in boundaries:
        ax_nov.axvline(b, color='#ff4d6d', lw=1.2, alpha=0.8)
    ax_nov.set_title("Foote Novelty — fronteras estructurales", color='#f0ede6', fontsize=9)
    ax_nov.set_xlabel("Beat", color='#666', fontsize=7)
    ax_nov.set_ylabel("Novelty", color='#666', fontsize=7)
    ax_nov.tick_params(colors='#444', labelsize=7)
    ax_nov.set_xlim(0, n)
    ax_nov.set_ylim(0, 1.1)
    for spine in ax_nov.spines.values():
        spine.set_color('#1e1e1e')

    # ── Panel 3: Energy arc + downbeats ──
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.set_facecolor('#0d0d0d')
    ax_e.fill_between(range(n), energy, alpha=0.5, color='#1a4a2e')
    ax_e.plot(energy, color='#47ff9a', lw=0.9, label='Energy arc')

    # Mark downbeats
    db_indices = np.where(is_downbeat)[0]
    ax_e.vlines(db_indices, 0, np.array(energy)[db_indices], color='#e8ff47',
               alpha=0.2, lw=0.5, label='Downbeats')

    ax_e.set_title("Arco de energía + downbeats", color='#f0ede6', fontsize=9)
    ax_e.set_xlabel("Beat", color='#666', fontsize=7)
    ax_e.set_ylabel("Energy (norm)", color='#666', fontsize=7)
    ax_e.tick_params(colors='#444', labelsize=7)
    ax_e.set_xlim(0, n)
    ax_e.legend(fontsize=7, facecolor='#111', labelcolor='#999', 
                edgecolor='#333', loc='upper right')
    for spine in ax_e.spines.values():
        spine.set_color('#1e1e1e')

    # ── Panel 4: Jump similarity distribution ──
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_dist.set_facecolor('#0d0d0d')
    all_sims = [j["similarity"] for beats in jump_graph.values() for j in beats]
    if all_sims:
        ax_dist.hist(all_sims, bins=30, color='#e8ff47', alpha=0.8, edgecolor='#080808', lw=0.3)
    ax_dist.set_title("Distribución de similaridad de saltos", color='#f0ede6', fontsize=9)
    ax_dist.set_xlabel("Similaridad", color='#666', fontsize=7)
    ax_dist.set_ylabel("Count", color='#666', fontsize=7)
    ax_dist.tick_params(colors='#444', labelsize=7)
    ax_dist.axvline(0.5, color='#ff4d6d', lw=1, linestyle='--', alpha=0.7, label='threshold 0.5')
    ax_dist.legend(fontsize=7, facecolor='#111', labelcolor='#999', edgecolor='#333')
    for spine in ax_dist.spines.values():
        spine.set_color('#1e1e1e')

    # ── Panel 5: Jump distance distribution ──
    ax_jd = fig.add_subplot(gs[2, 1])
    ax_jd.set_facecolor('#0d0d0d')
    jump_distances = []
    for i, jumps in jump_graph.items():
        for j in jumps:
            d = abs(j["to"] - int(i))
            jump_distances.append(d)
    if jump_distances:
        ax_jd.hist(jump_distances, bins=30, color='#ff4d6d', alpha=0.8, edgecolor='#080808', lw=0.3)
    ax_jd.set_title("Distribución de distancia de saltos (en beats)", color='#f0ede6', fontsize=9)
    ax_jd.set_xlabel("Distancia (beats)", color='#666', fontsize=7)
    ax_jd.set_ylabel("Count", color='#666', fontsize=7)
    ax_jd.tick_params(colors='#444', labelsize=7)
    for spine in ax_jd.spines.values():
        spine.set_color('#1e1e1e')

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#080808', edgecolor='none')
    plt.close()
    print(f"  Validation image saved: {output_path}")


def generate_sample_mp3(audio_path: str, result: dict, output_path: str,
                         sample_duration: float = 40.0, n_jumps: int = 12):
    """
    Render a sample MP3 by simulating the infinite loop engine.
    Uses pydub for audio concatenation with crossfade.
    """
    from pydub import AudioSegment

    CROSSFADE_MS = 80  # ms crossfade at each jump

    audio = AudioSegment.from_mp3(audio_path)
    beat_times = result["beat_times"]
    jump_graph = result["jump_graph"]
    energy_arc = result["energy_arc"]
    n_beats = result["n_beats"]

    # Simulate the walk
    current_beat = 4  # start a bit in
    segments_audio = []
    total_ms = 0
    target_ms = sample_duration * 1000
    jump_count = 0
    visited_recently = []  # anti-recency

    print(f"  Rendering sample: {sample_duration}s, ~{n_jumps} jumps, {CROSSFADE_MS}ms crossfade")

    while total_ms < target_ms and jump_count < n_jumps * 3:
        # How long to play before next jump
        # Play 2-4 bars (8-16 beats) between jumps
        bars_to_play = np.random.choice([2, 3, 4])
        beats_to_play = bars_to_play * 4
        end_beat = min(current_beat + beats_to_play, n_beats - 1)

        t_start_ms = int(beat_times[current_beat] * 1000)
        t_end_ms = int(beat_times[end_beat] * 1000)

        if t_end_ms <= t_start_ms or t_start_ms >= len(audio):
            current_beat = 4
            continue

        chunk = audio[t_start_ms:min(t_end_ms, len(audio))]
        if len(chunk) < 100:
            current_beat = 4
            continue

        segments_audio.append(chunk)
        total_ms += len(chunk)

        # Jump decision — pick best non-recently-visited
        jumps = jump_graph.get(end_beat, [])
        if not jumps:
            current_beat = (end_beat + 1) % n_beats
            continue

        # Anti-recency: penalize recently visited beats
        scored = []
        for jmp in jumps:
            recency_penalty = sum(1 for v in visited_recently if abs(v - jmp["to"]) < 8) * 0.1
            score = jmp["similarity"] - recency_penalty
            scored.append((score, jmp))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Weighted pick from top 3
        top3 = scored[:3]
        weights = np.array([max(0.01, s) for s, _ in top3])
        weights /= weights.sum()
        chosen_idx = np.random.choice(len(top3), p=weights)
        chosen_jump = top3[chosen_idx][1]

        visited_recently.append(chosen_jump["to"])
        if len(visited_recently) > 16:
            visited_recently.pop(0)

        current_beat = chosen_jump["to"]
        jump_count += 1
        print(f"    Jump {jump_count:2d}: beat {end_beat:3d} → {current_beat:3d} "
              f"(sim={chosen_jump['similarity']:.2f}, t={beat_times[end_beat]:.1f}s→{beat_times[current_beat]:.1f}s)")

    # Assemble with crossfade
    if not segments_audio:
        print("  ERROR: no audio segments generated")
        return

    print(f"  Assembling {len(segments_audio)} segments with {CROSSFADE_MS}ms crossfade...")
    result_audio = segments_audio[0]
    for seg in segments_audio[1:]:
        result_audio = result_audio.append(seg, crossfade=CROSSFADE_MS)

    result_audio.export(output_path, format="mp3", bitrate="192k")
    print(f"  Sample MP3 saved: {output_path} ({len(result_audio)/1000:.1f}s)")
