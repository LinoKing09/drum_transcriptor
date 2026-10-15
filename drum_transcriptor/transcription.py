"""Drum onset transcription.

The drum stem is decomposed with a non-negative matrix factor deconvolution
(NMFD): a library of spectro-temporal templates – short spectrogram snippets
of real kick, snare, hi-hat, cymbal and tom hits from several drum kits – is
fitted to the song's spectrogram. Each template gets an activation curve; the
curves of one instrument class are summed and peak-picked into hits.
Because the model is additive, simultaneous hits (hi-hat + snare + kick) are
explained jointly instead of masking each other, and the temporal shape of the
templates separates e.g. a kick's pitch drop from a tom's sustained tone.

The templates are learned by benchmark/learn_templates.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks

SR = 44100
HOP = 441  # 10 ms frames
N_FFT = 2048
N_MELS = 128
FMIN, FMAX = 25.0, 18000.0
FPS = SR / HOP
T_LEN = 10  # template length in frames
PAD = 10  # frames of silence prepended before analysis

CLASSES = ["kick", "snare", "hihat", "hihat_open", "crash", "ride", "tom"]

# Peak-picking parameters, tuned on MDB Drums with benchmark/tune.py.
# A peak must exceed an absolute height (relative to the loudest instrument of
# the kit) ...
DEFAULT_THRESHOLDS = {
    "kick": 0.01, "snare": 0.06, "hihat": 0.015, "hihat_open": 0.22,
    "crash": 0.17, "ride": 0.015, "tom": 0.5,
}
# ... and a fraction of the typical (90th percentile) peak of its own class.
DEFAULT_OWN_FRAC = {
    "kick": 0.3, "snare": 0.0, "hihat": 0.2, "hihat_open": 0.0,
    "crash": 0.0, "ride": 0.0, "tom": 0.4,
}
# A peak of class A is dropped when class B peaks at the same time with more
# than RATIO times its activation: B's sound simply leaks into A's templates.
# Toms are deliberately conservative: a wrong tom hurts a chart more than a
# missing one.
RIVALS = {
    "tom": [("snare", 0.4), ("kick", 3.0)],
    "hihat_open": [("crash", 1.5)],
    "crash": [("snare", 1.5)],
    "ride": [("crash", 5.0), ("hihat", 0.7)],
    "hihat": [("ride", 2.0)],
}
MIN_ATTACK = 0.15
ADAPT = 0.6  # how far templates may adapt to the song's kit


@dataclass
class Hit:
    time: float
    instrument: str  # kick, snare, hihat, hihat_open, crash, ride, tom_high, tom_mid, tom_floor
    velocity: float  # 0..1 relative to the loud hits of that instrument
    ghost: bool = False


def spectrogram(y: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=1.0,
    )


@lru_cache(maxsize=1)
def load_templates() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(__file__).with_name("templates.npz"))
    return data["templates"].astype(np.float64), data["labels"]


def _shift(X: np.ndarray, t: int) -> np.ndarray:
    """Shift columns right by t (zero padded); negative t shifts left."""
    if t == 0:
        return X
    out = np.zeros_like(X)
    if t > 0:
        out[:, t:] = X[:, :-t]
    else:
        out[:, :t] = X[:, -t:]
    return out


def nmfd(V: np.ndarray, W: np.ndarray, iters: int = 30, adapt: float = 0.0):
    """KL-NMFD (K templates of F x T) returning activations H (K x N).

    With adapt > 0 the templates are updated too, but blended back towards
    the library templates (weight `adapt` at most) so they can follow the
    sound of the song's kit without losing their identity.
    """
    eps = 1e-10
    K, _, Tl = W.shape
    W0 = W
    W = W.copy()
    H = np.full((K, V.shape[1]), V.sum(axis=0).mean() / K + eps)
    for it in range(iters):
        Wt = [W[:, :, t] for t in range(Tl)]  # each K x F
        norm = W.sum(axis=(1, 2))[:, None] + eps
        R = V / (sum(Wt[t].T @ _shift(H, t) for t in range(Tl)) + eps)
        H *= sum(Wt[t] @ _shift(R, -t) for t in range(Tl)) / norm
        if adapt > 0 and it >= iters // 3:
            R = V / (sum(Wt[t].T @ _shift(H, t) for t in range(Tl)) + eps)
            W_upd = np.empty_like(W)
            for t in range(Tl):
                Hs = _shift(H, t)
                W_upd[:, :, t] = W[:, :, t] * (R @ Hs.T).T / (Hs.sum(axis=1)[:, None] + eps)
            W_upd /= W_upd.sum(axis=(1, 2), keepdims=True) + eps
            alpha = adapt * (it - iters // 3 + 1) / (iters - iters // 3)
            W = (1 - alpha) * W0 + alpha * W_upd
    return H


def _reference(acts: dict[str, np.ndarray]) -> float:
    """Loudness reference of the kit: typical peak of the strongest class."""
    return max(np.percentile(a, 99.5) for a in acts.values()) + 1e-12


def _split_toms(hits: list[Hit], tom_pitch: list[float]) -> None:
    """Label toms high/mid/floor by their (relative) pitch within the song."""
    if not hits:
        return
    p = np.log2(np.asarray(tom_pitch))
    lo, hi = np.percentile(p, 10), np.percentile(p, 90)
    spread = hi - lo
    if spread < 0.3:  # roughly one drum
        names = ["tom_high"] if np.median(p) > np.log2(140) else ["tom_floor"]
        edges: list[float] = []
    elif spread < 0.8:
        names, edges = ["tom_floor", "tom_high"], [float(np.median(p))]
    else:
        names, edges = ["tom_floor", "tom_mid", "tom_high"], [lo + spread / 3, lo + 2 * spread / 3]
    for hit, pitch in zip(hits, p):
        hit.instrument = names[int(np.searchsorted(edges, pitch))]


def _toms_are_kick(tom_hits: list[Hit], hits: list[Hit]) -> bool:
    """Real toms show up in fills. If nearly every "tom" coincides with a
    kick, it is the body of a boomy (often electronic) kick instead."""
    kicks = np.array([h.time for h in hits if h.instrument == "kick"])
    if len(tom_hits) < 8 or not len(kicks):
        return False
    together = [np.min(np.abs(kicks - t.time)) < 0.03 for t in tom_hits]
    return float(np.mean(together)) > 0.6


def _decay_valleys(an: "Analysis", frames: list[int]) -> np.ndarray:
    """High-frequency energy left between consecutive cymbal hits, relative to
    the hit: ~0 for a closed hi-hat, clearly higher for ride or open hi-hat."""
    hb = an.V[an.freqs > 6000].sum(axis=0)
    out = np.full(len(frames), np.nan)
    for i, (p, nxt) in enumerate(zip(frames, frames[1:])):
        if nxt - p >= 18:  # a closed hi-hat needs ~180 ms to die away
            out[i] = hb[p + 8 : nxt - 1].min() / (hb[p : p + 3].max() + 1e-9)
    return out


def _timekeeping(an: "Analysis", hits: list[Hit], frames: dict[int, int],
                 window: float = 2.0) -> None:
    """Decide hi-hat vs. ride per passage and spot open hi-hats.

    Drummers keep time on one cymbal per passage. Whether that is a closed
    hi-hat or a ride shows in how much the sound rings between the hits; where
    that is unclear the local majority of the spectral labels decides.
    """
    keep = sorted((h for h in hits if h.instrument in ("hihat", "ride", "hihat_open")),
                  key=lambda h: h.time)
    if not keep:
        return
    valley = _decay_valleys(an, [frames[id(h)] for h in keep])
    times = np.array([h.time for h in keep])
    is_ride = np.array([h.instrument == "ride" for h in keep])
    new = []
    for i, h in enumerate(keep):
        near = np.abs(times - h.time) <= window
        v = valley[near & ~np.isnan(valley)]
        ring = float(np.median(v)) if len(v) >= 3 else np.nan
        rides, total = is_ride[near].sum(), near.sum()
        if ring < 0.06:  # dry passage: closed hi-hat, a ringing single hit is an open one
            open_hat = h.instrument == "hihat_open" or valley[i] > 0.15
            new.append("hihat_open" if open_hat else "hihat")
        elif ring > 0.09 and h.instrument != "hihat_open":
            new.append("ride")
        elif total >= 4 and rides >= 0.65 * total:
            new.append("ride")
        elif total >= 4 and rides <= 0.35 * total and h.instrument == "ride":
            new.append("hihat")
        else:
            new.append(h.instrument)
    for h, name in zip(keep, new):
        h.instrument = name


@dataclass
class Analysis:
    V: np.ndarray  # mel magnitude spectrogram
    acts: dict[str, np.ndarray]  # class activations, normalised to the kit loudness
    attack: np.ndarray  # broadband onset strength (0..~1)
    freqs: np.ndarray


def analyze(drums: np.ndarray, adapt: float = ADAPT) -> Analysis:
    """The expensive part: decompose the drum signal into class activations."""
    # leading silence, so that a hit right at t=0 still forms a peak
    V = spectrogram(np.concatenate([np.zeros(PAD * HOP, dtype=drums.dtype), drums]))
    W, labels = load_templates()
    H = nmfd(V, W, adapt=adapt)
    acts = {c: H[labels == c].sum(axis=0) for c in CLASSES}
    ref = _reference(acts)
    acts = {c: a / ref for c, a in acts.items()}
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(V, ref=np.max), sr=SR,
                                        hop_length=HOP)
    flux = flux / (np.percentile(flux, 99) + 1e-9)
    freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return Analysis(V, acts, maximum_filter1d(flux, size=5), freqs)


def pick_hits(an: Analysis, sensitivity: float = 1.0, thresholds: dict | None = None,
              own_frac: dict | None = None, rivals: dict | None = None,
              min_attack: float = MIN_ATTACK) -> list[Hit]:
    """Turn class activations into hits.

    A peak counts when it exceeds both an absolute threshold (relative to the
    loudest instrument of the kit) and a fraction of the typical peak height of
    its own class, and when it is not just leakage of a much stronger rival.
    """
    thr = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    own = dict(DEFAULT_OWN_FRAC, **(own_frac or {}))
    rivals = RIVALS if rivals is None else rivals
    tom_band = (an.freqs > 60) & (an.freqs < 450)
    hits: list[Hit] = []
    frames: dict[int, int] = {}
    tom_hits, tom_pitch = [], []
    for name in CLASSES:
        a = an.acts[name]
        cand, _ = find_peaks(a, height=thr[name] / sensitivity, distance=int(0.05 * FPS))
        typical = np.percentile(a[cand], 90) if len(cand) else 0.0
        height = max(thr[name], own[name] * typical) / sensitivity
        peaks, _ = find_peaks(a, height=height, prominence=0.5 * height,
                              distance=int(0.05 * FPS))
        for p in peaks:
            if an.attack[p] < min_attack:
                continue
            win = slice(max(p - 2, 0), p + 3)
            if any(an.acts[r][win].max() > ratio * a[p] for r, ratio in rivals.get(name, [])):
                continue
            hit = Hit((p - PAD) / FPS, name, float(min(a[p] / (typical + 1e-12), 1.0)))
            frames[id(hit)] = p
            if name == "tom":
                seg = an.V[tom_band, p : p + 6].mean(axis=1)
                tom_pitch.append(float(an.freqs[tom_band][np.argmax(seg)]))
                tom_hits.append(hit)
            hits.append(hit)
    if _toms_are_kick(tom_hits, hits):
        drop = set(map(id, tom_hits))
    else:
        # A tom and a snare at the same instant are one drum ringing into the
        # other's templates: keep whichever is stronger relative to its class.
        snares = [h for h in hits if h.instrument == "snare"]
        drop = set()
        for t in tom_hits:
            for sn in snares:
                if abs(sn.time - t.time) < 0.03:
                    drop.add(id(t) if sn.velocity >= t.velocity else id(sn))
    keep = [i for i, t in enumerate(tom_hits) if id(t) not in drop]
    tom_hits, tom_pitch = [tom_hits[i] for i in keep], [tom_pitch[i] for i in keep]
    hits = [h for h in hits if id(h) not in drop]
    _split_toms(tom_hits, tom_pitch)
    _timekeeping(an, hits, frames)

    snare_v = [h.velocity for h in hits if h.instrument == "snare"]
    if len(snare_v) > 8:
        med = float(np.median(snare_v))
        for h in hits:
            if h.instrument == "snare" and h.velocity < 0.4 * med:
                h.ghost = True
    hits.sort(key=lambda h: h.time)
    return hits


def transcribe(drums: np.ndarray, sensitivity: float = 1.0) -> list[Hit]:
    """Transcribe a mono drum signal (sampled at SR) into a list of hits."""
    return pick_hits(analyze(drums), sensitivity)


def load_mono(path, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    return y
