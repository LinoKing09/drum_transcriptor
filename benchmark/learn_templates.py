"""Learn spectro-temporal drum templates from isolated soundfont hits.

Renders every instrument of several GM drum kits at a few velocities, cuts the
spectrogram right at each hit and stores one template per (kit, instrument)
in drum_transcriptor/templates.npz.

    python benchmark/learn_templates.py [soundfont.sf2 ...]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drum_transcriptor import transcription as T  # noqa: E402

# instrument class -> GM notes that represent it
CLASSES = {
    "kick": [35, 36],
    "snare": [38, 40, 37],
    "hihat": [42, 44],
    "hihat_open": [46],
    "crash": [49, 57, 52, 55],
    "ride": [51, 59, 53],
    "tom": [41, 43, 45, 47, 48, 50],
}
TRAIN_KITS = [8, 16, 32, 40]  # Room, Power, Jazz, Brush (Standard = 0 is held out)
VELOCITIES = [50, 85, 120]
TOM_VELOCITY = 85  # one template per tom pitch
GAP = 2.0  # seconds between hits


def render(sf2: str, kit: int, notes: list[tuple[int, int]], wav: Path) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60)))
    tr.append(mido.Message("program_change", program=kit, channel=9))
    step = int(GAP * 480)
    tr.append(mido.Message("note_on", note=notes[0][0], velocity=notes[0][1], channel=9, time=step))
    for (n0, _), (n1, v1) in zip(notes, notes[1:] + [(notes[-1][0], 0)]):
        tr.append(mido.Message("note_off", note=n0, channel=9, time=60))
        if v1:
            tr.append(mido.Message("note_on", note=n1, velocity=v1, channel=9, time=step - 60))
    midi = wav.with_suffix(".mid")
    mid.save(midi)
    subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", str(T.SR), "-F", str(wav),
                    sf2, str(midi)], check=True, capture_output=True)


def main(soundfonts: list[str]) -> None:
    names = list(CLASSES)
    templates, labels = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for sf2 in soundfonts:
            for kit in TRAIN_KITS:
                notes = [(n, v) for c in names for n in CLASSES[c] for v in VELOCITIES]
                wav = Path(tmp) / f"kit{kit}.wav"
                render(sf2, kit, notes, wav)
                S = T.spectrogram(T.load_mono(wav))
                per_class: dict[str, list[np.ndarray]] = {c: [] for c in names}
                for k, (note, vel) in enumerate(notes):
                    f0 = int(round((k + 1) * GAP * T.FPS))
                    seg = S[:, f0 - 3 : f0 + 3 + T.T_LEN]
                    # align on the strongest attack near the nominal onset
                    on = int(np.argmax(np.diff(seg[:, :8].sum(axis=0)))) + 1
                    tpl = seg[:, on : on + T.T_LEN]
                    if tpl.sum() <= 0:
                        continue
                    cls = next(c for c in names if note in CLASSES[c])
                    per_class[cls].append(tpl / tpl.sum())
                    if cls == "tom" and vel == TOM_VELOCITY:  # keep every tom pitch as its own template
                        templates.append(tpl / tpl.sum())
                        labels.append("tom")
                for c in names:
                    if per_class[c] and c != "tom":
                        templates.append(np.mean(per_class[c], axis=0))
                        labels.append(c)
    out = Path(T.__file__).with_name("templates.npz")
    np.savez_compressed(out, templates=np.stack(templates).astype(np.float32),
                        labels=np.array(labels))
    print(f"{len(labels)} templates -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["/usr/share/sounds/sf2/FluidR3_GM.sf2"])
