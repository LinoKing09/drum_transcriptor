"""End-to-end pipeline: YouTube link -> drum sheet music (PDF + MIDI)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import audio, notation, rhythm, transcription

Progress = Callable[[str, float], None]


@dataclass
class Options:
    separate: bool = True  # isolate drums with Demucs first
    sensitivity: float = 1.0
    bpm: float | None = None
    beats_per_bar: int = 4
    triplets: bool = True
    start: float | None = None
    end: float | None = None
    cookies_from_browser: str | None = None
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "drum_transcriptor")


@dataclass
class Result:
    pdf: Path
    midi: Path
    ly: Path
    title: str
    tempo: int
    bars: int
    hits: int


def slug(text: str) -> str:
    return re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")[:80] or "drums"


def run(source: str, out_dir: Path, opts: Options, progress: Progress | None = None) -> Result:
    def step(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    if not notation.lilypond_available():
        raise RuntimeError("LilyPond ist nicht installiert (https://lilypond.org, "
                           "z. B. 'sudo apt install lilypond' oder 'brew install lilypond').")

    step("Lade Audio herunter …", 0.02)
    if Path(source).exists():
        wav, song = audio.local_audio(Path(source), opts.cache_dir)
    else:
        wav, song = audio.download_audio(source, opts.cache_dir, opts.cookies_from_browser)
    wav = audio.trim(wav, opts.start, opts.end)

    if opts.separate and audio.demucs_available():
        step("Isoliere das Schlagzeug (Demucs) – das dauert etwas …", 0.15)
        drum_wav = audio.separate_drums(wav)
    else:
        if opts.separate:
            step("Demucs nicht installiert – nutze den kompletten Mix.", 0.15)
        drum_wav = wav

    step("Erkenne Schläge …", 0.6)
    y = transcription.load_mono(drum_wav)
    hits = transcription.transcribe(y, sensitivity=opts.sensitivity)

    step("Erkenne Tempo und Takt …", 0.75)
    beats = rhythm.track_beats(y, bpm=opts.bpm)
    score = rhythm.quantize(
        hits, beats, beats_per_bar=opts.beats_per_bar, triplets=opts.triplets,
        duration=len(y) / transcription.SR,
    )

    step("Setze die Noten (LilyPond) …", 0.85)
    ly = notation.to_lilypond(
        score, song.title, song.artist,
        source=f"youtu.be/{song.video_id}" if not song.video_id.startswith("local-") else "",
    )
    name = slug(f"{song.artist} {song.title}" if song.artist else song.title)
    pdf, midi = notation.render(ly, out_dir / name)
    step("Fertig!", 1.0)
    return Result(pdf=pdf, midi=midi, ly=pdf.with_suffix(".ly"), title=song.title,
                  tempo=score.tempo, bars=len(score.bars), hits=len(hits))
