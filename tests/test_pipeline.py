import shutil

import numpy as np
import pytest

from drum_transcriptor import cli, notation, rhythm, transcription
from drum_transcriptor.rhythm import Beat, Score
from drum_transcriptor.transcription import Hit

from synth import rock_song


def _f1(ref, est, tol=0.03):
    ref = list(ref)
    tp = 0
    for t in est:
        match = [r for r in ref if abs(r - t) <= tol]
        if match:
            ref.remove(match[0])
            tp += 1
    return 2 * tp / max(len(est) + len(ref) + tp, 1)


def test_transcribes_synthetic_groove():
    y, events, _ = rock_song()
    hits = transcription.transcribe(y)
    for inst in ("kick", "snare"):
        ref = [t for t, i in events if i == inst]
        est = [h.time for h in hits if h.instrument == inst]
        assert _f1(ref, est) > 0.85, inst
    ref = [t for t, i in events if i.startswith("hihat")]
    est = [h.time for h in hits if h.instrument.startswith("hihat")]
    assert _f1(ref, est) > 0.8


def test_quantize_rock_beat_and_downbeat():
    period = 0.5
    beats = np.arange(0, 8.5, period)
    hits = []
    # the "one" of the groove is the third tracked beat
    for bar in range(3):
        t0 = 1.0 + bar * 4 * period
        hits += [Hit(t0, "kick", 1), Hit(t0 + period, "snare", 1),
                 Hit(t0 + 2 * period, "kick", 1), Hit(t0 + 3 * period, "snare", 1)]
        hits += [Hit(t0 + k * period / 2 + 0.01, "hihat", 1) for k in range(8)]
    hits.sort(key=lambda h: h.time)
    score = rhythm.quantize(hits, beats, beats_per_bar=4, triplets=False)
    assert score.tempo == 120
    one = next(bar for bar in score.bars if not all(b.is_empty() for b in bar))
    assert "kick" in one[0].slots[0] and "snare" in one[1].slots[0]
    assert "hihat" in one[0].slots[2]  # the "and"
    assert one[0].slots[1] == set() and one[0].slots[3] == set()


def test_quantize_detects_triplets():
    period = 0.6
    beats = np.arange(0, 6.1, period)
    hits = [Hit(1.2 + k * period / 3, "hihat", 1) for k in range(9)]
    score = rhythm.quantize(hits, beats, beats_per_bar=4)
    subdivisions = {b.subdivision for bar in score.bars for b in bar if not b.is_empty()}
    assert subdivisions == {3}


def _score():
    rock = [Beat(4, [{"kick", "hihat"}, set(), {"hihat"}, set()]),
            Beat(4, [{"snare", "hihat"}, set(), {"hihat"}, set()])] * 2
    fill = [Beat(4, [{"snare"}, {"snare"}, {"tom_high"}, {"tom_floor"}]),
            Beat(3, [{"ghost:snare"}, {"snare"}, {"kick", "crash"}]),
            Beat(4, [set()] * 4), Beat(4, [{"hihat_open"}, set(), set(), set()])]
    empty = [Beat(4, [set()] * 4) for _ in range(4)]
    return Score(bars=[empty, empty, rock, rock, rock, fill], tempo=100, beats_per_bar=4)


def test_lilypond_source():
    ly = notation.to_lilypond(_score(), 'My "Song"', "Band")
    assert r'title = "My \"Song\""' in ly
    assert r"\repeat percent 3" in ly
    assert r"\compressMMRests { R4*8 }" in ly
    assert r"\tuplet 3/2" in ly and r"\parenthesize sn" in ly
    assert "<hh sn>8" in ly and "bd4" in ly  # kick lives in the lower voice
    assert r"\tempo 4 = 100" in ly


@pytest.mark.skipif(shutil.which("lilypond") is None, reason="LilyPond not installed")
def test_renders_pdf_and_midi(tmp_path):
    pdf, midi = notation.render(notation.to_lilypond(_score(), "Test"), tmp_path / "test")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert midi.exists()


def test_time_argument():
    assert cli._time("75") == 75
    assert cli._time("1:15") == 75
    assert cli._time("1:02:03") == 3723
