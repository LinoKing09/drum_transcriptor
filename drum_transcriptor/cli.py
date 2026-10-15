"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline


def _time(value: str) -> float:
    """Accept seconds ('75') or mm:ss ('1:15')."""
    parts = [float(p) for p in value.split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="drum-transcriptor",
        description="Transkribiert das Schlagzeug eines Songs (YouTube-Link oder Audiodatei) "
                    "als PDF-Noten.",
    )
    p.add_argument("source", nargs="?", help="YouTube-Link oder Pfad zu einer Audiodatei")
    p.add_argument("-o", "--out", type=Path, default=Path("."), help="Ausgabeordner")
    p.add_argument("--start", type=_time, help="Startzeit, z. B. 0:30")
    p.add_argument("--end", type=_time, help="Endzeit, z. B. 2:15")
    p.add_argument("--bpm", type=float, help="Tempo vorgeben statt automatisch erkennen")
    p.add_argument("--time", default="4/4", choices=["2/4", "3/4", "4/4", "5/4", "6/4"],
                   help="Taktart (Standard: 4/4)")
    p.add_argument("--sensitivity", type=float, default=1.0,
                   help="Empfindlichkeit, >1 erkennt mehr (leise) Schläge, <1 weniger")
    p.add_argument("--no-triplets", action="store_true", help="Keine Triolen verwenden")
    p.add_argument("--no-separation", action="store_true",
                   help="Schlagzeug nicht mit Demucs isolieren (schneller, ungenauer)")
    p.add_argument("--cookies-from-browser", metavar="BROWSER",
                   help="Browser-Anmeldung für YouTube nutzen (firefox, chrome, edge, …)")
    p.add_argument("--web", action="store_true", help="Weboberfläche starten")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.web:
        from .web import serve
        serve(port=args.port)
        return 0
    if not args.source:
        p.error("Bitte einen YouTube-Link oder eine Audiodatei angeben (oder --web).")

    opts = pipeline.Options(
        separate=not args.no_separation,
        sensitivity=args.sensitivity,
        bpm=args.bpm,
        beats_per_bar=int(args.time.split("/")[0]),
        triplets=not args.no_triplets,
        start=args.start,
        end=args.end,
        cookies_from_browser=args.cookies_from_browser,
    )

    def progress(msg: str, frac: float) -> None:
        bar = "█" * int(frac * 24) + "░" * (24 - int(frac * 24))
        print(f"\r{bar} {msg:<60}", end="\n" if frac >= 1 else "", flush=True)

    try:
        res = pipeline.run(args.source, args.out, opts, progress)
    except Exception as exc:  # noqa: BLE001 - show a friendly message
        print(f"\nFehler: {exc}", file=sys.stderr)
        return 1
    print(f"\n🥁  {res.title}: {res.bars} Takte, {res.hits} Schläge, ♩ = {res.tempo}")
    print(f"    PDF:  {res.pdf}")
    print(f"    MIDI: {res.midi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
