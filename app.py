import sys
import os
import glob
import numpy as np
import soundfile as sf

# librosa imports mínimos (sin sklearn)
from librosa.core.audio import load as lr_load
from librosa.core.pitch import pyin as lr_pyin
from librosa.core.convert import note_to_hz as lr_note_to_hz
from librosa.core.spectrum import stft as lr_stft, istft as lr_istft

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
)

# ----------------- DEFAULTS -----------------

FMIN_NOTE = "C2"
FMAX_NOTE = "C7"

DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 256

WT_FRAME_SIZE = 2048
WT_MIP_LEVELS = 8

WT_DIR_DEFAULT = r"D:\WAVETABLE"

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a")


# ----------------- WAVETABLE OSC -----------------

def _to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.astype(np.float32, copy=False)


def _fade_edges(frame: np.ndarray, fade: int = 8) -> np.ndarray:
    if fade <= 0 or 2 * fade >= len(frame):
        return frame
    w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out = frame.copy()
    out[:fade] *= w
    out[-fade:] *= w[::-1]
    return out


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(frame)) + 1e-12
    return (frame / m).astype(np.float32, copy=False)


def _linear_resample(x: np.ndarray, new_len: int) -> np.ndarray:
    n = len(x)
    if new_len == n:
        return x.astype(np.float32, copy=False)
    src = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, new_len, endpoint=False, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32, copy=False)


def load_wavetable_wav(
    path: str,
    frame_size: int = WT_FRAME_SIZE,
    normalize_each_frame: bool = True,
    edge_fade: int = 8,
) -> np.ndarray:
    audio, _sr = sf.read(path, always_2d=False)
    audio = _to_mono_float(audio)

    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))

    n_frames = max(1, len(audio) // frame_size)
    use_len = n_frames * frame_size
    audio = audio[:use_len]

    frames = audio.reshape(n_frames, frame_size).copy()

    for i in range(n_frames):
        f = frames[i]
        f = f - np.mean(f)
        f = _fade_edges(f, edge_fade)
        if normalize_each_frame:
            f = _normalize_frame(f)
        frames[i] = f

    return frames.astype(np.float32, copy=False)


def build_wavetable_mipmaps(frames: np.ndarray, levels: int = WT_MIP_LEVELS):
    frames = np.asarray(frames, dtype=np.float32)
    n_frames, frame_size = frames.shape

    mipmaps = []
    cur = frames
    cur_size = frame_size

    for _lvl in range(levels):
        mipmaps.append(cur)

        next_size = max(32, cur_size // 2)
        if next_size == cur_size:
            break

        nxt = np.zeros((n_frames, next_size), dtype=np.float32)
        for fi in range(n_frames):
            nxt[fi] = _linear_resample(cur[fi], next_size)

        cur = nxt
        cur_size = next_size

    return mipmaps


def _lerp(a, b, t):
    return a + (b - a) * t


def _table_read_linear(table_1d: np.ndarray, phase: np.ndarray) -> np.ndarray:
    n = len(table_1d)
    idx = phase * n
    i0 = np.floor(idx).astype(np.int32)
    frac = idx - i0
    i1 = (i0 + 1) % n
    return (1.0 - frac) * table_1d[i0] + frac * table_1d[i1]


def render_wavetable_osc(
    f0_hz: np.ndarray,
    sr: int,
    mipmaps: list,
    position: float = 0.0,
    phase0: float = 0.0,
    mip_strength: float = 1.0,
):
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    n_samples = len(f0_hz)
    n_levels = len(mipmaps)

    base_frames = mipmaps[0]
    n_frames = base_frames.shape[0]

    pos = float(np.clip(position, 0.0, 1.0))
    fidx = pos * (n_frames - 1)
    f0i = int(np.floor(fidx))
    ft = float(fidx - f0i)
    f1i = min(f0i + 1, n_frames - 1)

    phase = np.empty(n_samples, dtype=np.float32)
    ph = float(phase0 % 1.0)
    for i in range(n_samples):
        phase[i] = ph
        ph += float(f0_hz[i]) / float(sr)
        ph -= np.floor(ph)

    # mip selector simple
    f_ref = 55.0
    ratio = np.maximum(f0_hz / f_ref, 1e-6)
    lvl_float = np.log2(ratio) * float(np.clip(mip_strength, 0.0, 1.0))
    lvl = np.clip(np.floor(lvl_float).astype(np.int32), 0, n_levels - 1)

    out = np.zeros(n_samples, dtype=np.float32)
    for L in range(n_levels):
        mask = (lvl == L)
        if not np.any(mask):
            continue
        tables_L = mipmaps[L]
        t0 = tables_L[f0i]
        t1 = tables_L[f1i]
        table = _lerp(t0, t1, ft)
        out[mask] = _table_read_linear(table, phase[mask])

    return out, float(ph)


def synth_wavetable_from_f0_env(
    f0_frames_hz: np.ndarray,
    env_frames: np.ndarray,
    sr: int,
    n_samples: int,
    frame_length: int,
    hop_length: int,
    mipmaps: list,
    position: float,
    mip_strength: float,
):
    f0_frames_hz = np.asarray(f0_frames_hz, dtype=np.float32)
    env_frames = np.asarray(env_frames, dtype=np.float32)

    centers = (np.arange(len(f0_frames_hz)) * hop_length + frame_length / 2.0)
    centers = np.clip(centers, 0, max(0, n_samples - 1))

    t = np.arange(n_samples, dtype=np.float32)
    f0 = np.interp(t, centers.astype(np.float32), f0_frames_hz).astype(np.float32)
    env = np.interp(t, centers.astype(np.float32), env_frames).astype(np.float32)

    f0 = np.clip(f0, 1.0, sr / 2.0)

    osc, _ = render_wavetable_osc(
        f0_hz=f0,
        sr=sr,
        mipmaps=mipmaps,
        position=position,
        phase0=0.0,
        mip_strength=mip_strength,
    )

    out = osc * env
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


# ----------------- ANALYSIS -----------------

def load_mono(path: str):
    y, sr = lr_load(path, sr=None, mono=True)
    return y.astype(np.float32, copy=False), int(sr)


def one_pole_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


# ----------------- NEW HELPERS (NORMAL vs FLUIDO) -----------------

def compute_activity(env_bands: np.ndarray, voiced: np.ndarray, thr: float = 0.02) -> np.ndarray:
    """
    Decide si hay 'algo sonando' por frame.
    Usamos energía total (promedio de bandas) + voiced como respaldo.
    """
    env_total = np.mean(env_bands, axis=0).astype(np.float32)
    act = env_total > float(thr)
    if voiced is not None:
        act = act | voiced.astype(bool)
    return act.astype(bool)


def hold_last_active_specenv(spec_env: np.ndarray, active: np.ndarray) -> np.ndarray:
    """
    Si no hay actividad, congelar el timbre al último frame activo.
    """
    spec_env = np.asarray(spec_env, np.float32).copy()
    n = spec_env.shape[1]
    if n == 0:
        return spec_env
    last = 0
    for t in range(n):
        if bool(active[t]):
            last = t
        else:
            spec_env[:, t] = spec_env[:, last]
    return spec_env


def make_fluid_env(
    env: np.ndarray,
    active: np.ndarray,
    floor: float,
    influence: float,
    alpha: float = 0.05,
    unvoiced_influence_mul: float = 0.25,
) -> np.ndarray:
    """
    Env 'fluido' (azul):
    - Nunca cae a 0 (usa floor)
    - Sigue un poco la dinámica real (influence)
    - En frames no-activos, baja la influencia para no "respirar" de más.
    """
    env = np.asarray(env, np.float32)
    active = np.asarray(active, bool)

    e = one_pole_smooth(env, alpha=alpha).astype(np.float32)

    infl = float(np.clip(influence, 0.0, 1.0))
    flo = float(np.clip(floor, 0.0, 1.0))

    infl_vec = np.full_like(e, infl, dtype=np.float32)
    infl_vec[~active] *= float(np.clip(unvoiced_influence_mul, 0.0, 1.0))

    out = flo + infl_vec * e
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    # un pelín más de suavizado para que no haya escalones
    out = one_pole_smooth(out, alpha=alpha).astype(np.float32)
    return out


def smooth_freq_logmag(logmag: np.ndarray, smooth_bins: int) -> np.ndarray:
    smooth_bins = int(max(1, smooth_bins))
    if smooth_bins <= 1:
        return logmag

    k = smooth_bins
    kernel = np.ones(k, dtype=np.float32) / float(k)
    pad = k // 2
    padded = np.pad(logmag, ((pad, pad), (0, 0)), mode="reflect")

    out = np.empty_like(logmag, dtype=np.float32)
    for t in range(logmag.shape[1]):
        out[:, t] = np.convolve(padded[:, t], kernel, mode="valid").astype(np.float32)
    return out


def extract_f0_multiband_env_and_specenv(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    f0_smooth_alpha: float,
    env_smooth_alpha: float,
    gate_unvoiced: bool,
    spec_smooth_bins: int,
):
    # ---- F0 (pyin) ----
    f0, voiced_flag, _ = lr_pyin(
        y,
        fmin=lr_note_to_hz(FMIN_NOTE),
        fmax=lr_note_to_hz(FMAX_NOTE),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = np.asarray(voiced_flag, dtype=bool)

    valid = np.where(~np.isnan(f0))[0]
    if len(valid) == 0:
        raise RuntimeError("No se pudo detectar pitch (F0) en el audio fuente.")

    # hold-last pitch (ya)
    f0_clean = f0.copy()
    last = f0_clean[valid[0]]
    for i in range(len(f0_clean)):
        if np.isnan(f0_clean[i]):
            f0_clean[i] = last
        else:
            last = f0_clean[i]

    f0_clean = one_pole_smooth(f0_clean, alpha=f0_smooth_alpha)

    # ---- STFT magnitude ----
    n_fft = frame_length
    S = lr_stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop_length, win_length=frame_length, center=True)
    mag = np.abs(S).astype(np.float32)
    n_bins, n_frames = mag.shape

    # ---- multiband env (4 bandas) ----
    freqs = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float32)
    edges = [
        (20.0, 140.0),     # low
        (140.0, 700.0),    # lowmid
        (700.0, 4000.0),   # highmid
        (4000.0, min(16000.0, sr / 2.0 - 1.0)),  # high
    ]

    env_bands = np.zeros((4, n_frames), dtype=np.float32)
    for bi, (lo, hi) in enumerate(edges):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        band_mag = mag[mask, :]
        env = np.sqrt(np.mean(band_mag * band_mag, axis=0) + 1e-12).astype(np.float32)
        env = env / (np.max(env) + 1e-12)
        env_bands[bi] = env

    # ---- spectral envelope (timbre) ----
    logmag = np.log(mag + 1e-8).astype(np.float32)
    spec_env_log = smooth_freq_logmag(logmag, smooth_bins=spec_smooth_bins)
    spec_env = np.exp(spec_env_log).astype(np.float32)

    # ---- ALIGN ----
    n = min(len(f0_clean), len(voiced), n_frames)
    f0_clean = f0_clean[:n]
    voiced = voiced[:n]
    env_bands = env_bands[:, :n]
    spec_env = spec_env[:, :n]

    if gate_unvoiced:
        v = voiced.astype(np.float32)
        env_bands = env_bands * v[None, :]

    for bi in range(4):
        env_bands[bi] = one_pole_smooth(env_bands[bi], alpha=env_smooth_alpha)

    # ✅ actividad + freeze timbre en silencio
    active = compute_activity(env_bands, voiced, thr=0.02)
    spec_env = hold_last_active_specenv(spec_env, active)

    return f0_clean.astype(np.float32), voiced, env_bands.astype(np.float32), spec_env.astype(np.float32), active


def spectral_envelope_match(
    y_syn: np.ndarray,
    sr: int,
    spec_env_src: np.ndarray,   # (bins, frames)
    frame_length: int,
    hop_length: int,
    spec_smooth_bins: int,
    strength: float,
    clamp_lo: float,
    clamp_hi: float,
):
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return y_syn

    n_fft = frame_length
    S = lr_stft(y_syn.astype(np.float32), n_fft=n_fft, hop_length=hop_length, win_length=frame_length, center=True)
    mag = np.abs(S).astype(np.float32)
    phase = np.angle(S).astype(np.float32)

    logmag = np.log(mag + 1e-8).astype(np.float32)
    spec_env_syn_log = smooth_freq_logmag(logmag, smooth_bins=spec_smooth_bins)
    spec_env_syn = np.exp(spec_env_syn_log).astype(np.float32)

    bins = min(spec_env_src.shape[0], spec_env_syn.shape[0], mag.shape[0])
    frames = min(spec_env_src.shape[1], spec_env_syn.shape[1], mag.shape[1])

    src = spec_env_src[:bins, :frames]
    syn = spec_env_syn[:bins, :frames]
    mag_use = mag[:bins, :frames]
    ph_use = phase[:bins, :frames]

    G = src / (syn + 1e-12)
    G = np.clip(G, float(clamp_lo), float(clamp_hi)).astype(np.float32)

    mag_out = mag_use * np.power(G, strength).astype(np.float32)

    S_out = mag_out * (np.cos(ph_use) + 1j * np.sin(ph_use))
    y_out = lr_istft(S_out, hop_length=hop_length, win_length=frame_length, center=True, length=len(y_syn))
    y_out = np.asarray(y_out, dtype=np.float32)
    return np.clip(y_out, -1.0, 1.0).astype(np.float32, copy=False)


# ----------------- FILE LISTING -----------------

def list_wav_files(folder: str, recursive: bool = True):
    if not folder or not os.path.isdir(folder):
        return []
    pattern = "**/*.wav" if recursive else "*.wav"
    files = glob.glob(os.path.join(folder, pattern), recursive=recursive)
    files += glob.glob(os.path.join(folder, pattern.upper()), recursive=recursive)
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: p.lower())
    return files


def list_audio_files(folder: str):
    files = []
    for ext in AUDIO_EXTS:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
        files += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: p.lower())
    return files


def make_out_roots_for_single(out_target: str, input_file: str):
    """
    Devuelve root sin extensión, para crear:
      root + "__normal.wav"
      root + "__fluido.wav"
    Acepta:
      - out_target carpeta existente: usa base del input
      - out_target ruta a .wav: usa esa ruta sin .wav como root
      - out_target sin .wav: lo trata como carpeta y la crea
    """
    out_target = out_target.strip()
    base = os.path.splitext(os.path.basename(input_file))[0]

    if os.path.isdir(out_target):
        root = os.path.join(out_target, base)
    elif out_target.lower().endswith(".wav"):
        root = out_target[:-4]
    else:
        # lo tratamos como carpeta
        os.makedirs(out_target, exist_ok=True)
        root = os.path.join(out_target, base)

    return root


# ----------------- WORKER -----------------

class AudioWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        wt_dir: str,
        seed: int,               # 0 => azar total
        pos_min: float,
        pos_max: float,
        mip_min: float,
        mip_max: float,
        hop_length: int,
        frame_length: int,
        env_alpha: float,
        f0_alpha: float,
        gate_unvoiced: bool,
        output_gain: float,
        enable_spec_match: bool,
        spec_strength: float,
        spec_smooth_bins: int,
        spec_clamp_lo: float,
        spec_clamp_hi: float,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.wt_dir = wt_dir
        self.seed = int(seed)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)
        self.mip_min = float(mip_min)
        self.mip_max = float(mip_max)
        self.hop_length = int(hop_length)
        self.frame_length = int(frame_length)
        self.env_alpha = float(env_alpha)
        self.f0_alpha = float(f0_alpha)
        self.gate_unvoiced = bool(gate_unvoiced)
        self.output_gain = float(output_gain)

        self.enable_spec_match = bool(enable_spec_match)
        self.spec_strength = float(spec_strength)
        self.spec_smooth_bins = int(spec_smooth_bins)
        self.spec_clamp_lo = float(spec_clamp_lo)
        self.spec_clamp_hi = float(spec_clamp_hi)

        self._mipmap_cache = {}

        # FLUID preset (hardcode como pediste)
        self._fluid_floor = 0.18
        self._fluid_influence = 0.35
        self._fluid_alpha = 0.06
        self._fluid_unvoiced_mul = 0.15

    def _load_mipmaps_cached(self, wt_path: str):
        wt_path = os.path.abspath(wt_path)
        mm = self._mipmap_cache.get(wt_path)
        if mm is not None:
            return mm
        frames = load_wavetable_wav(wt_path, frame_size=WT_FRAME_SIZE)
        mipmaps = build_wavetable_mipmaps(frames, levels=WT_MIP_LEVELS)
        self._mipmap_cache[wt_path] = mipmaps
        return mipmaps

    def _process_one(self, src_file: str, out_root: str, wavetable_files: list, rng: np.random.Generator):
        out_normal = out_root + "__normal.wav"
        out_fluido = out_root + "__fluido.wav"

        self.log.emit(f"Fuente: {os.path.basename(src_file)}")
        y, sr = load_mono(src_file)
        self.log.emit(f"  len={len(y)} sr={sr}")

        self.log.emit("  Analizando: F0 + multiband env + spectral envelope...")
        f0_frames, voiced, env_bands, spec_env, active = extract_f0_multiband_env_and_specenv(
            y=y,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            f0_smooth_alpha=self.f0_alpha,
            env_smooth_alpha=self.env_alpha,
            gate_unvoiced=self.gate_unvoiced,
            spec_smooth_bins=self.spec_smooth_bins,
        )

        # Elegimos 4 wavetables UNA vez (mismo set para normal y fluido)
        picks = rng.choice(wavetable_files, size=4, replace=(len(wavetable_files) < 4))

        layers_normal = []
        layers_fluido = []

        for i in range(4):
            wt = str(picks[i])
            mipmaps = self._load_mipmaps_cached(wt)

            pos = float(rng.uniform(self.pos_min, self.pos_max))
            mip = float(rng.uniform(self.mip_min, self.mip_max))

            env_normal = env_bands[i]
            env_fluid = make_fluid_env(
                env=env_bands[i],
                active=active,
                floor=self._fluid_floor,
                influence=self._fluid_influence,
                alpha=self._fluid_alpha,
                unvoiced_influence_mul=self._fluid_unvoiced_mul,
            )

            layer_n = synth_wavetable_from_f0_env(
                f0_frames_hz=f0_frames,
                env_frames=env_normal,
                sr=sr,
                n_samples=len(y),
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                mipmaps=mipmaps,
                position=pos,
                mip_strength=mip,
            )
            layer_f = synth_wavetable_from_f0_env(
                f0_frames_hz=f0_frames,
                env_frames=env_fluid,
                sr=sr,
                n_samples=len(y),
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                mipmaps=mipmaps,
                position=pos,
                mip_strength=mip,
            )

            layers_normal.append(layer_n)
            layers_fluido.append(layer_f)

            self.log.emit(
                f"  Layer {i+1}: {os.path.basename(wt)} | pos={pos:.2f} mip={mip:.2f}"
                f" | fluid(floor={self._fluid_floor:.2f}, infl={self._fluid_influence:.2f})"
            )

        # Mix NORMAL
        mix_n = np.sum(np.stack(layers_normal, axis=0), axis=0).astype(np.float32)
        mx = float(np.max(np.abs(mix_n)) + 1e-12)
        mix_n = (mix_n / mx * 0.95).astype(np.float32)

        # Mix FLUIDO
        mix_f = np.sum(np.stack(layers_fluido, axis=0), axis=0).astype(np.float32)
        mx = float(np.max(np.abs(mix_f)) + 1e-12)
        mix_f = (mix_f / mx * 0.95).astype(np.float32)

        if self.enable_spec_match:
            self.log.emit("  Aplicando spectral envelope match (timbre) a NORMAL...")
            mix_n = spectral_envelope_match(
                y_syn=mix_n,
                sr=sr,
                spec_env_src=spec_env,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                spec_smooth_bins=self.spec_smooth_bins,
                strength=self.spec_strength,
                clamp_lo=self.spec_clamp_lo,
                clamp_hi=self.spec_clamp_hi,
            )
            self.log.emit("  Aplicando spectral envelope match (timbre) a FLUIDO...")
            mix_f = spectral_envelope_match(
                y_syn=mix_f,
                sr=sr,
                spec_env_src=spec_env,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                spec_smooth_bins=self.spec_smooth_bins,
                strength=self.spec_strength,
                clamp_lo=self.spec_clamp_lo,
                clamp_hi=self.spec_clamp_hi,
            )

        if self.output_gain != 1.0:
            mix_n = np.clip(mix_n * float(self.output_gain), -1.0, 1.0).astype(np.float32)
            mix_f = np.clip(mix_f * float(self.output_gain), -1.0, 1.0).astype(np.float32)

        # crear carpeta si hace falta
        for out_file in (out_normal, out_fluido):
            out_dir = os.path.dirname(out_file)
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        sf.write(out_normal, mix_n, sr)
        sf.write(out_fluido, mix_f, sr)
        self.log.emit(f"  Guardado NORMAL: {out_normal}")
        self.log.emit(f"  Guardado FLUIDO: {out_fluido}")

    def run(self):
        try:
            inp = self.input_path
            outp = self.output_path

            wavetable_files = list_wav_files(self.wt_dir, recursive=True)
            if len(wavetable_files) == 0:
                raise RuntimeError("No se encontraron .wav en la carpeta de wavetables (incl. subcarpetas).")

            # RNG: si seed=0 => azar total (entropy del sistema)
            rng = np.random.default_rng(None if self.seed == 0 else self.seed)

            is_batch = os.path.isdir(inp)

            if not is_batch:
                # SINGLE
                if not os.path.isfile(inp):
                    raise RuntimeError("Input single debe ser un archivo.")

                out_root = make_out_roots_for_single(outp, inp)

                self.progress.emit(5)
                self.log.emit("=== PROCESO SINGLE ===")
                self.log.emit(
                    f"Wavetable dir: {self.wt_dir} | count={len(wavetable_files)} | seed={'RANDOM' if self.seed==0 else self.seed}"
                )
                self._process_one(inp, out_root, wavetable_files, rng)
                self.progress.emit(100)
                self.finished.emit()
                return

            # BATCH
            in_dir = inp
            out_dir = outp
            if os.path.isfile(out_dir):
                out_dir = os.path.dirname(out_dir)

            if not out_dir:
                raise RuntimeError("Output batch debe ser una carpeta válida.")
            os.makedirs(out_dir, exist_ok=True)

            files = list_audio_files(in_dir)
            if not files:
                raise RuntimeError("No se encontraron audios en la carpeta input.")

            self.log.emit("=== PROCESO BATCH ===")
            self.log.emit(f"Input: {in_dir}")
            self.log.emit(f"Output: {out_dir}")
            self.log.emit(
                f"Wavetable dir: {self.wt_dir} | count={len(wavetable_files)} | seed={'RANDOM' if self.seed==0 else self.seed}"
            )
            self.progress.emit(1)

            total = len(files)
            for idx, src_file in enumerate(files):
                base = os.path.splitext(os.path.basename(src_file))[0]
                out_root = os.path.join(out_dir, base)

                self.log.emit(f"\n[{idx+1}/{total}]")
                p = int(5 + 90 * (idx / max(1, total)))
                self.progress.emit(p)

                # RNG continuo: completamente random a lo largo del batch
                self._process_one(src_file, out_root, wavetable_files, rng)

            self.progress.emit(100)
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


# ----------------- UI -----------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Restaurador por Síntesis (4 bandas + timbre match) — NORMAL/FLUIDO")
        self.resize(980, 720)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Input / Output (con botones separados archivo/carpeta)
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()

        btn_in_file = QPushButton("Input archivo…")
        btn_in_dir = QPushButton("Input carpeta…")
        btn_out_file = QPushButton("Output archivo…")
        btn_out_dir = QPushButton("Output carpeta…")

        btn_in_file.clicked.connect(self.pick_input_file)
        btn_in_dir.clicked.connect(self.pick_input_dir)
        btn_out_file.clicked.connect(self.pick_output_file)
        btn_out_dir.clicked.connect(self.pick_output_dir)

        row_in = QHBoxLayout()
        row_in.addWidget(QLabel("Input:"))
        row_in.addWidget(self.in_edit, stretch=1)
        row_in.addWidget(btn_in_file)
        row_in.addWidget(btn_in_dir)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output:"))
        row_out.addWidget(self.out_edit, stretch=1)
        row_out.addWidget(btn_out_file)
        row_out.addWidget(btn_out_dir)

        layout.addLayout(row_in)
        layout.addLayout(row_out)

        # Wavetables
        gb_wt = QGroupBox("Wavetables (random desde carpeta)")
        g = QVBoxLayout(gb_wt)

        self.wt_dir_edit = QLineEdit(WT_DIR_DEFAULT)
        btn_wt_dir = QPushButton("Carpeta…")
        btn_wt_dir.clicked.connect(self.pick_wt_dir)

        row_wtdir = QHBoxLayout()
        row_wtdir.addWidget(QLabel("Carpeta wavetables:"))
        row_wtdir.addWidget(self.wt_dir_edit, stretch=1)
        row_wtdir.addWidget(btn_wt_dir)
        g.addLayout(row_wtdir)

        row_rand = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(0)
        row_rand.addWidget(QLabel("Seed (0 = RANDOM SIEMPRE):"))
        row_rand.addWidget(self.seed_spin)

        self.pos_min = QDoubleSpinBox()
        self.pos_min.setRange(0.0, 1.0)
        self.pos_min.setSingleStep(0.01)
        self.pos_min.setValue(0.0)

        self.pos_max = QDoubleSpinBox()
        self.pos_max.setRange(0.0, 1.0)
        self.pos_max.setSingleStep(0.01)
        self.pos_max.setValue(1.0)

        self.mip_min = QDoubleSpinBox()
        self.mip_min.setRange(0.0, 1.0)
        self.mip_min.setSingleStep(0.05)
        self.mip_min.setValue(0.6)

        self.mip_max = QDoubleSpinBox()
        self.mip_max.setRange(0.0, 1.0)
        self.mip_max.setSingleStep(0.05)
        self.mip_max.setValue(1.0)

        row_rand.addSpacing(10)
        row_rand.addWidget(QLabel("WT pos min:"))
        row_rand.addWidget(self.pos_min)
        row_rand.addWidget(QLabel("max:"))
        row_rand.addWidget(self.pos_max)

        row_rand.addSpacing(10)
        row_rand.addWidget(QLabel("WT mip min:"))
        row_rand.addWidget(self.mip_min)
        row_rand.addWidget(QLabel("max:"))
        row_rand.addWidget(self.mip_max)

        row_rand.addStretch()
        g.addLayout(row_rand)

        layout.addWidget(gb_wt)

        # Analysis
        gb_an = QGroupBox("Análisis")
        a = QHBoxLayout(gb_an)

        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(64, 4096)
        self.hop_spin.setSingleStep(64)
        self.hop_spin.setValue(DEFAULT_HOP_LENGTH)

        self.env_alpha = QDoubleSpinBox()
        self.env_alpha.setRange(0.01, 1.0)
        self.env_alpha.setSingleStep(0.05)
        self.env_alpha.setValue(0.25)

        self.f0_alpha = QDoubleSpinBox()
        self.f0_alpha.setRange(0.01, 1.0)
        self.f0_alpha.setSingleStep(0.05)
        self.f0_alpha.setValue(0.20)

        self.gate_check = QCheckBox("Mutear unvoiced")
        self.gate_check.setChecked(True)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 3.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1.0)

        a.addWidget(QLabel("Hop:"))
        a.addWidget(self.hop_spin)
        a.addSpacing(10)
        a.addWidget(QLabel("Env α:"))
        a.addWidget(self.env_alpha)
        a.addSpacing(10)
        a.addWidget(QLabel("F0 α:"))
        a.addWidget(self.f0_alpha)
        a.addSpacing(12)
        a.addWidget(self.gate_check)
        a.addSpacing(10)
        a.addWidget(QLabel("Gain:"))
        a.addWidget(self.gain_spin)
        a.addStretch()

        layout.addWidget(gb_an)

        # Spectral match
        gb_sm = QGroupBox("Spectral envelope match (timbre / EQ dinámica)")
        s = QHBoxLayout(gb_sm)

        self.spec_enable = QCheckBox("Activar")
        self.spec_enable.setChecked(True)

        self.spec_strength = QDoubleSpinBox()
        self.spec_strength.setRange(0.0, 1.0)
        self.spec_strength.setSingleStep(0.05)
        self.spec_strength.setValue(0.7)

        self.spec_smooth_bins = QSpinBox()
        self.spec_smooth_bins.setRange(1, 256)
        self.spec_smooth_bins.setValue(25)

        self.spec_clamp_lo = QDoubleSpinBox()
        self.spec_clamp_lo.setRange(0.01, 1.0)
        self.spec_clamp_lo.setSingleStep(0.05)
        self.spec_clamp_lo.setValue(0.25)

        self.spec_clamp_hi = QDoubleSpinBox()
        self.spec_clamp_hi.setRange(1.0, 20.0)
        self.spec_clamp_hi.setSingleStep(0.5)
        self.spec_clamp_hi.setValue(4.0)

        s.addWidget(self.spec_enable)
        s.addSpacing(10)
        s.addWidget(QLabel("Strength:"))
        s.addWidget(self.spec_strength)
        s.addSpacing(10)
        s.addWidget(QLabel("Smooth bins:"))
        s.addWidget(self.spec_smooth_bins)
        s.addSpacing(10)
        s.addWidget(QLabel("Clamp lo:"))
        s.addWidget(self.spec_clamp_lo)
        s.addWidget(QLabel("hi:"))
        s.addWidget(self.spec_clamp_hi)
        s.addStretch()

        layout.addWidget(gb_sm)

        # Process
        self.btn_process = QPushButton("Procesar (exporta __normal.wav y __fluido.wav)")
        self.btn_process.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_process)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        layout.addWidget(self.logs, stretch=1)

        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        footer = QLabel("© 2025 Gabriel Golker")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self.thread = None
        self.worker = None

    def log(self, msg: str):
        self.logs.append(msg)

    # --------- pickers ---------
    def pick_input_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar audio fuente", "", "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff *.m4a);;Todos (*.*)"
        )
        if path:
            self.in_edit.setText(path)

    def pick_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta input")
        if folder:
            self.in_edit.setText(folder)

    def pick_output_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar salida (root)", "resultado.wav", "WAV (*.wav);;Todos (*.*)"
        )
        if path:
            self.out_edit.setText(path)

    def pick_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta output")
        if folder:
            self.out_edit.setText(folder)

    def pick_wt_dir(self):
        start = self.wt_dir_edit.text().strip() or WT_DIR_DEFAULT
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de wavetables", start)
        if folder:
            self.wt_dir_edit.setText(folder)

    # --------- run ---------
    def start_processing(self):
        inp = self.in_edit.text().strip()
        outp = self.out_edit.text().strip()
        wt_dir = self.wt_dir_edit.text().strip()

        if not inp or not outp:
            QMessageBox.warning(self, "Falta info", "Completa input y output.")
            return
        if not wt_dir or not os.path.isdir(wt_dir):
            QMessageBox.warning(self, "Wavetables", "La carpeta de wavetables no existe.")
            return

        pos_min = float(self.pos_min.value())
        pos_max = float(self.pos_max.value())
        if pos_min > pos_max:
            pos_min, pos_max = pos_max, pos_min

        mip_min = float(self.mip_min.value())
        mip_max = float(self.mip_max.value())
        if mip_min > mip_max:
            mip_min, mip_max = mip_max, mip_min

        self.logs.clear()
        self.progress.setValue(0)
        self.btn_process.setEnabled(False)

        self.thread = QThread()
        self.worker = AudioWorker(
            input_path=inp,
            output_path=outp,
            wt_dir=wt_dir,
            seed=int(self.seed_spin.value()),
            pos_min=pos_min,
            pos_max=pos_max,
            mip_min=mip_min,
            mip_max=mip_max,
            hop_length=int(self.hop_spin.value()),
            frame_length=DEFAULT_FRAME_LENGTH,
            env_alpha=float(self.env_alpha.value()),
            f0_alpha=float(self.f0_alpha.value()),
            gate_unvoiced=bool(self.gate_check.isChecked()),
            output_gain=float(self.gain_spin.value()),
            enable_spec_match=bool(self.spec_enable.isChecked()),
            spec_strength=float(self.spec_strength.value()),
            spec_smooth_bins=int(self.spec_smooth_bins.value()),
            spec_clamp_lo=float(self.spec_clamp_lo.value()),
            spec_clamp_hi=float(self.spec_clamp_hi.value()),
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_done)
        self.worker.error.connect(self.on_err)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def on_done(self):
        self.btn_process.setEnabled(True)
        QMessageBox.information(self, "Listo", "Proceso completado (NORMAL + FLUIDO).")

    def on_err(self, msg: str):
        self.btn_process.setEnabled(True)
        self.log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    import qdarkstyle
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
