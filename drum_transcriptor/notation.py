"""Render a quantised Score as LilyPond drum notation and compile it to PDF/MIDI."""

from __future__ import annotations

import shutil
import subprocess
from itertools import groupby
from pathlib import Path

from .rhythm import Beat, Score

LILY = {
    "kick": "bd",
    "snare": "sn",
    "hihat": "hh",
    "hihat_open": "hho",
    "crash": "cymc",
    "ride": "cymr",
    "tom_high": "tomh",
    "tom_mid": "tommh",
    "tom_floor": "tomfh",
}
LEGEND_ORDER = ["crash", "ride", "hihat", "hihat_open", "tom_high", "tom_mid", "snare",
                "tom_floor", "kick"]
LEGEND_NAMES = {
    "kick": "Bass Drum", "snare": "Snare", "hihat": "Hi-Hat", "hihat_open": "Hi-Hat offen",
    "crash": "Crash", "ride": "Ride", "tom_high": "Tom 1", "tom_mid": "Tom 2",
    "tom_floor": "Standtom",
}
FEET = {"kick"}

DUR4 = {1: "16", 2: "8", 3: "8.", 4: "4"}
DUR3 = {1: "8", 2: "4", 3: "4"}


def _base(token: str) -> str:
    return token.split(":", 1)[-1]


def _note(tokens: set[str], dur: str) -> str:
    names = []
    for tok in sorted(tokens, key=lambda t: LEGEND_ORDER.index(_base(t))):
        name = LILY[_base(tok)]
        names.append(rf"\parenthesize {name}" if tok.startswith("ghost:") else name)
    if len(names) == 1:
        return names[0] + dur
    return "<" + " ".join(names) + ">" + dur


def _beat(beat: Beat, voice: set[str]) -> str | None:
    slots = [{t for t in s if _base(t) in voice} for s in beat.slots]
    if not any(slots):
        return None
    sub = beat.subdivision
    if sub == 3 and not any(slots[1:]):  # a single note on the beat needs no tuplet
        return _note(slots[0], "4")
    durs = DUR4 if sub == 4 else DUR3
    out = []
    onsets = [i for i, s in enumerate(slots) if s] + [sub]
    if onsets[0] > 0:
        out.append("r" + durs[onsets[0]])
    for a, b in zip(onsets, onsets[1:]):
        out.append(_note(slots[a], durs[b - a]))
    text = " ".join(out)
    return rf"\tuplet 3/2 {{ {text} }}" if sub == 3 else text


def _voice(bar: list[Beat], voice: set[str], spacer_if_empty: bool) -> str:
    beats = [_beat(b, voice) for b in bar]
    n = len(bar)
    if all(b is None for b in beats):
        return f"s4*{n}" if spacer_if_empty else f"R4*{n}"
    out, i = [], 0
    while i < n:
        if beats[i] is not None:
            out.append(beats[i])
            i += 1
        elif i % 2 == 0 and i + 1 < n and beats[i + 1] is None:
            out.append("r2")
            i += 2
        else:
            out.append("r4")
            i += 1
    return " ".join(out)


def _bar(bar: list[Beat]) -> str | None:
    if all(b.is_empty() for b in bar):
        return None
    hands = set(LILY) - FEET
    up = _voice(bar, hands, spacer_if_empty=False)
    down = _voice(bar, FEET, spacer_if_empty=True)
    return rf"<< {{ {up} }} \\ {{ {down} }} >>"


def _music(score: Score) -> str:
    n = score.beats_per_bar
    lines = []
    for text, group in groupby(_bar(b) for b in score.bars):
        count = len(list(group))
        if text is None:
            lines.append(rf"\compressMMRests {{ R4*{n * count} }}")
        elif count == 1:
            lines.append(f"{text} |")
        else:
            lines.append(rf"\repeat percent {count} {{ {text} | }}")
    return "\n    ".join(lines)


def _legend(used: set[str]) -> str:
    items = [i for i in LEGEND_ORDER if i in used]
    notes = " ".join(rf'{LILY[i]}4_\markup {{ \tiny "{LEGEND_NAMES[i]}" }}' for i in items)
    return rf"""\markup \fill-line {{ \score {{
  \new DrumStaff \with {{
    \remove Time_signature_engraver
    \remove Bar_engraver
    \override StaffSymbol.staff-space = #0.8
  }} \drummode {{ \cadenzaOn {notes} }}
  \layout {{ indent = 0 ragged-right = ##t \context {{ \Score \remove Bar_number_engraver }} }}
}} }}"""


def _str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def to_lilypond(score: Score, title: str, artist: str = "", source: str = "") -> str:
    used = {_base(t) for bar in score.bars for b in bar for s in b.slots for t in s}
    music = _music(score)
    return rf"""\version "2.24.0"
\pointAndClickOff

\header {{
  title = "{_str(title)}"
  subtitle = "Schlagzeug"
  composer = "{_str(artist)}"
  tagline = "Automatisch transkribiert mit drum_transcriptor{' · ' + _str(source) if source else ''}"
}}

\paper {{
  #(set-paper-size "a4")
  top-margin = 12\mm
  bottom-margin = 12\mm
  left-margin = 14\mm
  right-margin = 14\mm
  markup-system-spacing.padding = #3
}}

music = \drummode {{
    \tempo 4 = {score.tempo}
    \time {score.beats_per_bar}/4
    {music}
    \bar "|."
}}

{_legend(used)}

\score {{
  \new DrumStaff \with {{ instrumentName = "Drums" }} \music
  \layout {{
    \context {{
      \Score
      countPercentRepeats = ##t
      repeatCountVisibility = #(every-nth-repeat-count-visible 1)
    }}
  }}
}}

\score {{
  \new DrumStaff \unfoldRepeats \music
  \midi {{ }}
}}
"""


def lilypond_available() -> bool:
    return shutil.which("lilypond") is not None


def render(ly_source: str, out_base: Path) -> tuple[Path, Path]:
    """Compile LilyPond source; returns (pdf, midi)."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    ly = out_base.with_suffix(".ly")
    ly.write_text(ly_source, encoding="utf-8")
    proc = subprocess.run(
        ["lilypond", "-s", "-o", str(out_base), str(ly)], capture_output=True, text=True
    )
    pdf, midi = out_base.with_suffix(".pdf"), out_base.with_suffix(".midi")
    if proc.returncode != 0 or not pdf.exists():
        raise RuntimeError("LilyPond-Fehler:\n" + proc.stderr[-3000:])
    return pdf, midi
