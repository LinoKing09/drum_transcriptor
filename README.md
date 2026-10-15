# 🥁 drum_transcriptor

Ein kleines Programm, das aus einem **YouTube-Link** (oder einer Audiodatei) das
**Schlagzeug als PDF-Noten** transkribiert – inklusive MIDI-Datei zum Gegenhören.

```
drum-transcriptor "https://www.youtube.com/watch?v=…"
```

…oder bequem im Browser:

```
drum-transcriptor --web
```

![Pipeline](https://img.shields.io/badge/YouTube-→%20Demucs%20→%20NMFD%20→%20LilyPond%20→%20PDF-d1462f)

## Was dabei herauskommt

- **PDF** in Standard-Schlagzeugnotation: zwei Stimmen (Hände oben, Bassdrum unten),
  Legende der verwendeten Instrumente, Tempoangabe, Taktzahlen
- gleiche Takte hintereinander werden als **Wiederholungszeichen** (mit Anzahl) gesetzt,
  Pausen ohne Schlagzeug als **Mehrtaktpausen**
- **Triolen** (z. B. Shuffle) werden pro Schlag erkannt, **Ghost Notes** auf der Snare
  eingeklammert, offene Hi-Hats markiert
- **MIDI**-Datei und der **LilyPond-Quelltext** zum Nachbearbeiten

Erkannte Instrumente: Bassdrum, Snare, Hi-Hat (geschlossen/offen), Crash, Ride,
Toms (hoch/mittel/Standtom).

## Installation

Voraussetzungen: Python ≥ 3.10, [FFmpeg](https://ffmpeg.org) und [LilyPond](https://lilypond.org)
(für den Notensatz).

```bash
# Debian/Ubuntu
sudo apt install ffmpeg lilypond
# macOS
brew install ffmpeg lilypond
# Windows: ffmpeg und LilyPond installieren und zum PATH hinzufügen

pip install .
```

Das installiert auch [Demucs](https://github.com/facebookresearch/demucs) (PyTorch), mit dem
das Schlagzeug vorab aus dem Mix herausgetrennt wird. Beim ersten Lauf lädt Demucs sein
Modell (~80 MB) herunter.

## Benutzung

```bash
drum-transcriptor "https://youtu.be/…"                    # ganzer Song
drum-transcriptor "https://youtu.be/…" --start 0:45 --end 1:30   # nur ein Ausschnitt
drum-transcriptor song.mp3 -o noten/                       # lokale Audiodatei
drum-transcriptor "https://youtu.be/…" --bpm 96 --time 3/4 # Tempo/Taktart vorgeben
drum-transcriptor --web                                    # Weboberfläche auf http://127.0.0.1:8000
```

| Option | Bedeutung |
| --- | --- |
| `-o, --out` | Ausgabeordner (Standard: aktueller Ordner) |
| `--start`, `--end` | Ausschnitt in Sekunden oder `m:ss` |
| `--bpm` | Tempo vorgeben, falls die automatische Erkennung danebenliegt (z. B. halbes/doppeltes Tempo) |
| `--time` | Taktart `2/4`, `3/4`, `4/4` (Standard), `5/4`, `6/4` |
| `--sensitivity` | > 1 erkennt mehr leise Schläge, < 1 weniger Fehlalarme |
| `--no-triplets` | nur gerade Unterteilungen verwenden |
| `--no-separation` | ohne Demucs (schneller, aber deutlich ungenauer) |
| `--cookies-from-browser firefox` | falls YouTube „Sign in to confirm you're not a bot“ meldet |

Heruntergeladene Songs und getrennte Spuren werden in `~/.cache/drum_transcriptor` zwischengespeichert,
ein zweiter Lauf (z. B. mit anderer Empfindlichkeit) ist daher schnell.

## Wie es funktioniert

1. **Download** der Tonspur mit [yt-dlp](https://github.com/yt-dlp/yt-dlp).
2. **Trennung** des Schlagzeugs vom Rest des Songs mit Demucs (`htdemucs`).
3. **Erkennung der Schläge** mit einer *Non-negative Matrix Factor Deconvolution* (NMFD):
   Eine Bibliothek kurzer Spektrogramm-Ausschnitte („Templates“) echter Kick-, Snare-,
   Hi-Hat-, Becken- und Tom-Schläge aus mehreren Drumkits wird an das Spektrogramm des Songs
   angepasst; die Templates dürfen sich dabei an das Kit des Songs angleichen. Weil das Modell
   additiv ist, werden gleichzeitige Schläge (Hi-Hat + Snare + Kick) gemeinsam erklärt.
   Danach greifen musikalische Regeln: Hi-Hat vs. Ride wird abschnittsweise am Ausklingen
   entschieden, Toms die nur „mitklingen“ (mit Kick oder Snare) werden verworfen usw.
4. **Rhythmus**: Beat-Tracking ([librosa](https://librosa.org)), Quantisierung auf Sechzehntel
   bzw. Triolen pro Schlag, Korrektur wenn der Tracker auf den Off-Beats liegt, und Wahl des
   Taktanfangs (Kick/Crash auf der Eins, Snare auf 2 und 4, Fills am Taktende).
5. **Notensatz** mit [LilyPond](https://lilypond.org) → PDF + MIDI.

## Wie gut ist das?

Die Erkennung ist automatisch und **nicht perfekt** – das Ergebnis ist ein guter Startpunkt,
den man beim Gegenhören korrigiert (am bequemsten im mitgelieferten `.ly`- oder MIDI-File).
Gemessen auf [MDB Drums](https://github.com/CarlSouthall/MDBDrums) (23 echte Songs, kompletter
Mix → Demucs → Transkription, F-Maß bei 30 ms Toleranz):

| Bassdrum | Snare | Hi-Hat | Becken | Toms |
| --- | --- | --- | --- | --- |
| 0,90 | 0,68 | 0,65 | 0,39 | 0,29 |

Tempo wird bei 21 von 23 Songs korrekt erkannt (die anderen beiden: halbes Tempo bei 220 BPM),
der Taktanfang bei 15 von 23. Am besten funktionieren klar produzierte Pop-/Rock-Grooves;
Jazz-Becken, Toms und sehr leise Ghost Notes sind am schwierigsten.

## Entwicklung

```bash
pip install -e ".[dev]"
pytest

# Benchmarks (brauchen fluidsynth + einen GM-Soundfont, bzw. den MDB-Drums-Datensatz)
python benchmark/make_songs.py bench/          # synthetische Songs mit Ground Truth rendern
python benchmark/evaluate.py bench/ [--mix]
python benchmark/evaluate.py "MDB Drums" --mdb [--mix]
python benchmark/tune.py "MDB Drums" [--mix]   # Peak-Picking-Parameter tunen (Train/Test-Split)
python benchmark/learn_templates.py            # Templates aus Soundfont-Drumkits neu lernen
```

Die mitgelieferten Templates (`drum_transcriptor/templates.npz`) wurden aus den Drumkits
Room, Power, Jazz und Brush des FluidR3-GM-Soundfonts gelernt.
