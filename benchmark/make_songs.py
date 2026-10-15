"""Render benchmark songs with known drum ground truth.

Writes, per song, a MIDI file, a drums-only WAV, a full-mix WAV (drums + bass +
piano) and a JSON with the ground-truth hits. Needs `fluidsynth` and a GM
soundfont (e.g. Debian/Ubuntu: `apt install fluidsynth fluid-soundfont-gm`).

    python benchmark/make_songs.py out_dir
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import mido

SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
GM = {"kick": 36, "snare": 38, "hihat": 42, "hihat_open": 46, "crash": 49, "ride": 51,
      "tom_high": 50, "tom_mid": 47, "tom_floor": 43}
TPB = 480


def _groove(name: str, bar: int, rng: random.Random):
    """Yield (position in 16ths or triplet-8ths, instrument, velocity) for one bar."""
    fill = bar % 4 == 3
    if name == "rock":
        for s in range(0, 16, 2):
            if not (bar % 4 == 0 and s == 0):
                yield s, "hihat_open" if (s == 14 and bar % 4 == 1) else "hihat", 80
        for s in (0, 6, 8) + ((10,) if bar % 2 else ()):
            yield s, "kick", 110
        for s in (4, 12):
            if not (fill and s == 12):
                yield s, "snare", 110
        if bar % 4 == 0:
            yield 0, "crash", 115
        if fill:
            for s, inst in zip(range(8, 16), ["snare", "snare", "tom_high", "tom_high",
                                              "tom_mid", "tom_mid", "tom_floor", "tom_floor"]):
                yield s, inst, 105
    elif name == "funk":
        for s in range(16):
            yield s, "hihat", 90 if s % 4 == 0 else 60
        for s in (0, 3, 10) + ((7,) if bar % 2 else ()):
            yield s, "kick", 110
        for s in (4, 12):
            yield s, "snare", 115
        for s in (7, 9, 15):
            if rng.random() < 0.7:
                yield s, "snare", 35  # ghost notes
    elif name == "ride":
        for s in range(0, 16, 2):
            yield s, "ride", 85
        for s in (0, 8, 11):
            yield s, "kick", 110
        for s in (4, 12):
            yield s, "snare", 110
        if bar % 8 == 0:
            yield 0, "crash", 115
    elif name == "shuffle":  # positions are triplet eighths (12 per bar)
        for s in range(0, 12, 3):
            yield s, "hihat", 85
            yield s + 2, "hihat", 65
        for s in (0, 6, 8):
            yield s, "kick", 110
        for s in (3, 9):
            yield s, "snare", 110


SONGS = {"rock": 104, "funk": 96, "ride": 132, "shuffle": 88}


def make(name: str, bpm: int, out: Path, bars: int = 16) -> None:
    rng = random.Random(name)
    per_bar = 12 if name == "shuffle" else 16
    step = TPB * 4 // per_bar
    mid = mido.MidiFile(ticks_per_beat=TPB)
    meta, drums, bass, keys = (mido.MidiTrack() for _ in range(4))
    mid.tracks += [meta, drums, bass, keys]
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    lead_in = TPB * 4  # one empty bar before the drums start

    events, truth = [], []
    for bar in range(bars):
        for pos, inst, vel in _groove(name, bar, rng):
            tick = lead_in + bar * TPB * 4 + pos * step
            events.append((tick, GM[inst], vel))
            truth.append({"time": mido.tick2second(tick, TPB, mido.bpm2tempo(bpm)),
                          "instrument": inst, "ghost": vel < 50})

    def write(track, evs, channel, length):
        msgs = []
        for tick, note, vel in evs:
            msgs.append((tick, mido.Message("note_on", note=note, velocity=vel, channel=channel)))
            msgs.append((tick + length, mido.Message("note_off", note=note, channel=channel)))
        msgs.sort(key=lambda m: (m[0], m[1].type == "note_on"))
        now = 0
        for tick, msg in msgs:
            track.append(msg.copy(time=tick - now))
            now = tick

    write(drums, events, 9, 60)
    roots = [40, 45, 43, 38]
    bass_ev = [(lead_in + b * TPB * 4 + q * TPB * 2, roots[b % 4], 90)
               for b in range(bars) for q in range(2)]
    keys_ev = [(lead_in + b * TPB * 4 + q * TPB, roots[b % 4] + 24 + iv, 60)
               for b in range(bars) for q in range(4) for iv in (0, 4, 7)]
    bass.append(mido.Message("program_change", program=33, channel=0))
    keys.append(mido.Message("program_change", program=4, channel=1))
    write(bass, bass_ev, 0, TPB * 2 - 20)
    write(keys, keys_ev, 1, TPB - 20)

    out.mkdir(parents=True, exist_ok=True)
    full_mid = out / f"{name}.mid"
    mid.save(full_mid)
    drum_only = mido.MidiFile(ticks_per_beat=TPB)
    drum_only.tracks += [meta, drums]
    drum_mid = out / f"{name}_drums.mid"
    drum_only.save(drum_mid)
    for src, wav in ((full_mid, out / f"{name}_mix.wav"), (drum_mid, out / f"{name}_drums.wav")):
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", "44100", "-F", str(wav),
                        SOUNDFONT, str(src)], check=True, capture_output=True)
    (out / f"{name}.json").write_text(json.dumps({"bpm": bpm, "hits": truth}, indent=1))


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_data")
    for song, tempo in SONGS.items():
        make(song, tempo, target)
        print("wrote", song)
