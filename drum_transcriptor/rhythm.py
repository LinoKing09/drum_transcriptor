"""Beat tracking and quantisation of drum hits onto a metrical grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from .transcription import HOP, SR, Hit


@dataclass
class Beat:
    subdivision: int  # 4 = sixteenths, 3 = eighth-note triplets
    slots: list[set[str]] = field(default_factory=list)  # instrument names (+ "ghost:snare")

    def is_empty(self) -> bool:
        return not any(self.slots)


@dataclass
class Score:
    bars: list[list[Beat]]
    tempo: int
    beats_per_bar: int
    bar_times: list[float] = field(default_factory=list)  # start time of every bar


def track_beats(drums: np.ndarray, bpm: float | None = None) -> np.ndarray:
    """Return beat times (s) of the drum signal."""
    env = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP)
    _, beats = librosa.beat.beat_track(
        onset_envelope=env, sr=SR, hop_length=HOP, start_bpm=bpm or 120.0, bpm=bpm,
        tightness=400 if bpm else 100, units="time",
    )
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4:
        period = 60.0 / (bpm or 120.0)
        beats = np.arange(0, len(drums) / SR + period, period)
    return beats


def _extend(beats: np.ndarray, start: float, end: float) -> np.ndarray:
    """Extrapolate the beat grid so that it covers [start, end]."""
    period = float(np.median(np.diff(beats)))
    head = []
    t = beats[0] - period
    while t > start - period:
        head.append(t)
        t -= period
    tail = []
    t = beats[-1] + period
    while t < end + period:
        tail.append(t)
        t += period
    return np.concatenate([head[::-1], beats, tail])


def _downbeat_offset(grid: list[Beat], beats_per_bar: int) -> int:
    """Pick the bar phase that best fits common drum idioms: kick and crash on
    the one, snare on the back beats and fills leading into the one."""
    if beats_per_bar < 2:
        return 0
    back = {1, 3} if beats_per_bar == 4 else {beats_per_bar // 2}
    toms = {"tom_high", "tom_mid", "tom_floor"}
    n_bars = max(len(grid) / beats_per_bar, 1.0)
    first = next((i for i, b in enumerate(grid) if not b.is_empty()), 0)
    best, best_score = 0, -np.inf
    for off in range(beats_per_bar):
        score = 0.0
        for i, beat in enumerate(grid):
            pos = (i - off) % beats_per_bar
            head = beat.slots[0]
            kick, snare, crash = "kick" in head, "snare" in head, "crash" in head
            if pos == 0:
                score += 1.0 * kick + 2.0 * crash - 1.0 * snare
            elif pos in back:
                score += 1.0 * snare - 0.3 * kick
            if pos == beats_per_bar - 1:  # fills happen at the end of the bar
                score += 0.3 * sum(1 for s in beat.slots[1:] for t in s
                                   if t in toms or t == "snare")
        score /= n_bars
        if (first - off) % beats_per_bar == 0:  # tie-breaker: drums enter on the one
            score += 1e-3
        if score > best_score + 1e-9:
            best, best_score = off, score
    return best


def _build_grid(hits: list[Hit], beats: np.ndarray, triplets: bool) -> list[Beat]:
    """Snap every hit to the nearest sixteenth (or triplet eighth) of its beat."""
    per_beat: list[list[tuple[float, Hit]]] = [[] for _ in range(len(beats))]
    idx = np.searchsorted(beats, [h.time for h in hits], side="right") - 1
    for h, i in zip(hits, idx):
        i = int(np.clip(i, 0, len(beats) - 2))
        frac = (h.time - beats[i]) / (beats[i + 1] - beats[i])
        per_beat[i].append((frac, h))

    grid: list[Beat] = []
    carry: list[tuple[float, Hit]] = []
    for i in range(len(beats) - 1):
        items = carry + per_beat[i]
        carry = []
        sub = 4
        if triplets and len(items) >= 2:
            fr = np.array([f for f, _ in items])
            err4 = np.abs(fr * 4 - np.round(fr * 4)) / 4
            err3 = np.abs(fr * 3 - np.round(fr * 3)) / 3
            off_grid = np.sum(err4 > 0.07) >= 2
            if off_grid and err3.sum() < 0.55 * err4.sum():
                sub = 3
        beat = Beat(sub, [set() for _ in range(sub)])
        for f, h in items:
            slot = int(round(f * sub))
            if slot >= sub:  # belongs to the next beat
                carry.append((f - 1.0, h))
                continue
            beat.slots[max(slot, 0)].add(f"ghost:{h.instrument}" if h.ghost else h.instrument)
        grid.append(beat)

    # a ghost note sharing a slot with a real snare hit is just the snare
    for beat in grid:
        for s in beat.slots:
            if "snare" in s:
                s.discard("ghost:snare")
            if "hihat_open" in s:
                s.discard("hihat")
    return grid


def _on_offbeats(grid: list[Beat]) -> bool:
    """True when kick and snare sit on the '&' instead of the beat, i.e. the
    beat tracker locked onto the off-beat eighths."""
    on = off = 0
    for beat in grid:
        if beat.subdivision != 4:
            continue
        on += sum(1 for t in beat.slots[0] if t in ("kick", "snare"))
        off += sum(1 for t in beat.slots[2] if t in ("kick", "snare"))
    return off >= 8 and off > 1.5 * on


def quantize(
    hits: list[Hit],
    beats: np.ndarray,
    beats_per_bar: int = 4,
    triplets: bool = True,
    duration: float | None = None,
) -> Score:
    if not hits:
        return Score(bars=[], tempo=int(round(60 / np.median(np.diff(beats)))),
                     beats_per_bar=beats_per_bar)
    end = max(duration or 0.0, hits[-1].time)
    beats = _extend(beats, min(0.0, hits[0].time), end)
    # tempo from a line fit over all beats (robust against frame jitter)
    period = float(np.polyfit(np.arange(len(beats)), beats, 1)[0])
    tempo = int(round(60.0 / period))

    grid = _build_grid(hits, beats, triplets)
    if _on_offbeats(grid):
        beats = beats - period / 2
        grid = _build_grid(hits, beats, triplets)

    off = _downbeat_offset(grid, beats_per_bar)
    times = list(beats[:-1])
    if off:  # pad the start so that the detected downbeat begins bar 2
        pad = beats_per_bar - off
        grid = [Beat(4, [set() for _ in range(4)]) for _ in range(pad)] + grid
        times = [beats[0] - (pad - k) * period for k in range(pad)] + times

    # drop trailing silence, pad the last bar
    while grid and grid[-1].is_empty():
        grid.pop()
    while len(grid) % beats_per_bar:
        grid.append(Beat(4, [set() for _ in range(4)]))
    bars = [grid[i : i + beats_per_bar] for i in range(0, len(grid), beats_per_bar)]
    bar_times = [float(t) for t in times[: len(grid) : beats_per_bar]]
    # bars that lie entirely before the start of the audio are just padding
    while len(bars) > 1 and bar_times[0] + beats_per_bar * period <= 0.05 and all(
        b.is_empty() for b in bars[0]
    ):
        bars.pop(0)
        bar_times.pop(0)
    return Score(bars=bars, tempo=tempo, beats_per_bar=beats_per_bar, bar_times=bar_times)
