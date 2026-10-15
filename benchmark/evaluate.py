"""Score the transcription against ground truth (onset F-measure, 30 ms tolerance).

    python benchmark/evaluate.py data_dir [--mix]        # songs from make_songs.py
    python benchmark/evaluate.py "MDB Drums" --mdb [--mix]  # MDB Drums dataset

--mix runs the full mix through Demucs first instead of using the drum-only audio.
MDB Drums: https://github.com/CarlSouthall/MDBDrums
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drum_transcriptor import audio, transcription  # noqa: E402

GROUP = {"tom_high": "tom", "tom_mid": "tom", "tom_floor": "tom", "hihat_open": "hihat",
         "ride": "cymbal", "crash": "cymbal"}
MDB = {"KD": "kick", "SD": "snare", "HH": "hihat", "TT": "tom", "CY": "cymbal"}
TOL = 0.03


def songs(data: Path, mdb: bool, mix: bool):
    """Yield (name, truth hits, audio path) for every song of a dataset."""
    if not mdb:
        for truth_file in sorted(data.glob("*.json")):
            name = truth_file.stem
            truth = json.loads(truth_file.read_text())["hits"]
            yield name, [(h["time"], h["instrument"]) for h in truth], data / (
                f"{name}_mix.wav" if mix else f"{name}_drums.wav")
        return
    for ann in sorted((data / "annotations" / "class").glob("*.txt")):
        name = ann.stem.replace("_class", "")
        truth = []
        for line in ann.read_text().splitlines():
            t, cls = line.split()
            if cls in MDB:
                truth.append((float(t), MDB[cls]))
        sub = "full_mix" if mix else "drum_only"
        suffix = "_MIX.wav" if mix else "_Drum.wav"
        yield name, truth, data / "audio" / sub / (name + suffix)


def score(ref: list[float], est: list[float]) -> tuple[int, int, int]:
    ref = sorted(ref)
    used = [False] * len(ref)
    tp = 0
    for t in sorted(est):
        for i, r in enumerate(ref):
            if not used[i] and abs(r - t) <= TOL:
                used[i] = True
                tp += 1
                break
    return tp, len(est), len(ref)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--mix", action="store_true")
    p.add_argument("--mdb", action="store_true")
    p.add_argument("--sensitivity", type=float, default=1.0)
    args = p.parse_args()
    totals = defaultdict(lambda: [0, 0, 0])
    for name, truth, wav in songs(args.data, args.mdb, args.mix):
        if args.mix:
            work = args.data / "sep" / name
            work.mkdir(parents=True, exist_ok=True)
            src = work / "song.wav"
            if not src.exists():
                src.write_bytes(wav.read_bytes())
            wav = audio.separate_drums(src)
        hits = transcription.transcribe(transcription.load_mono(wav), args.sensitivity)
        ref, est = defaultdict(list), defaultdict(list)
        for t, inst in truth:
            ref[GROUP.get(inst, inst)].append(t)
        for h in hits:
            est[GROUP.get(h.instrument, h.instrument)].append(h.time)
        line = []
        for k in sorted(set(ref) | set(est)):
            tp, ne, nr = score(ref[k], est[k])
            for i, v in enumerate((tp, ne, nr)):
                totals[k][i] += v
            f = 2 * tp / max(ne + nr, 1)
            line.append(f"{k}={f:.2f}({tp}/{nr},{ne - tp}fp)")
        print(f"{name[:22]:22s} " + "  ".join(line))
    print("-" * 60)
    for k, (tp, ne, nr) in sorted(totals.items()):
        print(f"{k:8s} P={tp / max(ne, 1):.2f} R={tp / max(nr, 1):.2f} F={2 * tp / max(ne + nr, 1):.2f}")


if __name__ == "__main__":
    main()
