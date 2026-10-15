"""Synthetic drum kit for tests: renders a known pattern to audio."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100
rng = np.random.default_rng(1)


def _env(n, decay):
    return np.exp(-np.arange(n) / (decay * SR))


def _noise(n, lo=None, hi=None):
    x = rng.standard_normal(n)
    if lo and hi:
        x = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x)
    elif lo:
        x = sosfilt(butter(4, lo, btype="high", fs=SR, output="sos"), x)
    return x / (np.abs(x).max() + 1e-9)


def _tone(n, f0, f1, decay):
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, decay)


def sounds():
    n = int(0.6 * SR)
    return {
        "kick": 0.9 * _tone(n, 150, 50, 0.12) + 0.05 * _noise(n, 2000, 5000) * _env(n, 0.005),
        "snare": 0.4 * _tone(n, 240, 190, 0.05) + 0.5 * _noise(n, 1000, 10000) * _env(n, 0.07),
        "hihat": 0.25 * _noise(n, 7000) * _env(n, 0.025),
        "hihat_open": 0.25 * _noise(n, 7000) * _env(n, 0.25),
        "crash": 0.35 * _noise(n * 3, 3000) * _env(n * 3, 0.6),
        "tom_high": 0.6 * _tone(n, 260, 200, 0.18) + 0.1 * _noise(n, 1000, 4000) * _env(n, 0.02),
        "tom_floor": 0.7 * _tone(n, 130, 95, 0.25) + 0.1 * _noise(n, 800, 3000) * _env(n, 0.02),
    }


def rock_song(bpm=112, bars=12):
    """Groove with hi-hat eighths, a tom fill every 4th bar and crashes."""
    beat = 60.0 / bpm
    six = beat / 4
    events = []
    for bar in range(bars):
        t0 = 1.0 + bar * 4 * beat
        fill = bar % 4 == 3
        if bar % 4 == 0:
            events.append((t0, "crash"))
        for s in range(16):
            t = t0 + s * six
            if fill and s >= 8:
                inst = ["snare", "snare", "tom_high", "tom_high",
                        "tom_floor", "tom_floor", "tom_floor", "kick"][s - 8]
                events.append((t, inst))
                continue
            if s % 2 == 0 and not (bar % 4 == 0 and s == 0):
                events.append((t, "hihat_open" if (s == 14 and bar % 2) else "hihat"))
            if s in (0, 8, 10) or (s == 7 and bar % 2):
                events.append((t, "kick"))
            if s in (4, 12):
                events.append((t, "snare"))
    total = 1.0 + bars * 4 * beat + 2.0
    y = np.zeros(int(total * SR) + SR * 2)
    snd = sounds()
    for t, inst in events:
        i = int(t * SR)
        s = snd[inst]
        y[i : i + len(s)] += s
    y /= np.abs(y).max()
    return y.astype(np.float32), sorted(events), [1.0 + k * beat for k in range(bars * 4 + 1)]
