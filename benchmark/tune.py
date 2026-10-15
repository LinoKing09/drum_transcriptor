"""Tune the peak-picking parameters on MDB Drums (odd songs) and report on the
held-out even songs.

    python benchmark/tune.py "MDB Drums" [--mix]
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import GROUP, score, songs  # noqa: E402

from drum_transcriptor import audio, transcription as T  # noqa: E402

EVAL_GROUPS = ["kick", "snare", "hihat", "tom", "cymbal"]


def load(data: Path, mix: bool, adapt: float = 0.0):
    cache = data / f"analyses{'_mix' if mix else ''}_a{adapt:g}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    items = []
    for name, truth, wav in songs(data, True, mix):
        if mix:
            work = data / "sep" / name
            work.mkdir(parents=True, exist_ok=True)
            src = work / "song.wav"
            if not src.exists():
                src.write_bytes(wav.read_bytes())
            wav = audio.separate_drums(src)
        print("analysing", name, flush=True)
        items.append((name, truth, T.analyze(T.load_mono(wav), adapt)))
    cache.write_bytes(pickle.dumps(items))
    return items


def evaluate(items, params) -> dict[str, tuple[int, int, int]]:
    tot = defaultdict(lambda: [0, 0, 0])
    for _, truth, an in items:
        hits = T.pick_hits(an, **params)
        ref, est = defaultdict(list), defaultdict(list)
        for t, inst in truth:
            ref[GROUP.get(inst, inst)].append(t)
        for h in hits:
            est[GROUP.get(h.instrument, h.instrument)].append(h.time)
        for k in EVAL_GROUPS:
            for i, v in enumerate(score(ref[k], est[k])):
                tot[k][i] += v
    return tot


def fscore(tot) -> dict[str, float]:
    return {k: 2 * tp / max(ne + nr, 1) for k, (tp, ne, nr) in tot.items()}


def objective(items, params) -> float:
    f = fscore(evaluate(items, params))
    return sum(f.values()) / len(f)


def report(label, items, params):
    tot = evaluate(items, params)
    f = fscore(tot)
    print(f"{label:10s} " + "  ".join(f"{k}={f[k]:.2f}" for k in EVAL_GROUPS)
          + f"  mean={sum(f.values()) / len(f):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--adapt", type=float, default=0.0)
    args = ap.parse_args()
    items = load(args.data, args.mix, args.adapt)
    tune, hold = items[1::2], items[0::2]

    params = {
        "thresholds": dict(T.DEFAULT_THRESHOLDS),
        "own_frac": dict(T.DEFAULT_OWN_FRAC),
        "rivals": copy.deepcopy(T.RIVALS),
        "min_attack": 0.1,
    }
    report("before/tune", tune, params)
    report("before/hold", hold, params)
    best = objective(tune, params)
    for _ in range(args.rounds):
        for c in T.CLASSES:
            for key, grid in (("thresholds", [0.0005, 0.001, 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1,
                                              0.13, 0.17, 0.22, 0.3]),
                              ("own_frac", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])):
                for v in grid:
                    trial = copy.deepcopy(params)
                    trial[key][c] = v
                    o = objective(tune, trial)
                    if o > best + 1e-4:
                        best, params = o, trial
        for c, rl in params["rivals"].items():
            for i, (r, _) in enumerate(rl):
                for v in [0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 100.0]:
                    trial = copy.deepcopy(params)
                    trial["rivals"][c][i] = (r, v)
                    o = objective(tune, trial)
                    if o > best + 1e-4:
                        best, params = o, trial
        for v in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
            trial = dict(params, min_attack=v)
            o = objective(tune, trial)
            if o > best + 1e-4:
                best, params = o, trial
        report("after/tune", tune, params)
    report("after/hold", hold, params)
    print(params)


if __name__ == "__main__":
    main()
