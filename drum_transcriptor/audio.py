"""Audio acquisition: YouTube download and drum stem separation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SongInfo:
    title: str
    artist: str
    video_id: str
    duration: float | None = None


def _clean_title(title: str, artist: str) -> tuple[str, str]:
    """Turn 'Artist - Title (Official Video)' into ('Title', 'Artist')."""
    title = re.sub(
        r"\s*[\(\[][^\)\]]*(official|video|audio|lyric|hd|4k|remaster|visualizer)[^\)\]]*[\)\]]",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    if " - " in title:
        left, right = title.split(" - ", 1)
        return right.strip(), left.strip()
    artist = re.sub(r"\s*-\s*Topic$|VEVO$", "", artist).strip()
    return title, artist


BOT_HINT = (
    "YouTube verlangt eine Anmeldung („Sign in to confirm you're not a bot“). "
    "Probiere es mit --cookies-from-browser firefox (oder chrome, edge, safari …), "
    "damit die Anmeldung deines Browsers verwendet wird."
)


def download_audio(
    url: str, workdir: Path, cookies_from_browser: str | None = None
) -> tuple[Path, SongInfo]:
    """Download the audio track of a YouTube video as WAV (cached per video ID)."""
    import yt_dlp

    workdir.mkdir(parents=True, exist_ok=True)
    base = {"quiet": True, "no_warnings": True, "noplaylist": True}
    if cookies_from_browser:
        base["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        if "not a bot" in str(exc) or "Sign in" in str(exc):
            raise RuntimeError(BOT_HINT) from exc
        raise
    if info.get("entries"):  # search result or playlist: take the first video
        info = next(e for e in info["entries"] if e)
        url = info.get("webpage_url") or info["id"]

    title, artist = _clean_title(
        info.get("track") or info.get("title") or "Unbekannter Song",
        info.get("artist") or info.get("uploader") or "",
    )
    song = SongInfo(title=title, artist=artist, video_id=info["id"], duration=info.get("duration"))

    song_dir = workdir / song.video_id
    wav = song_dir / "song.wav"
    if not wav.exists():
        song_dir.mkdir(parents=True, exist_ok=True)
        opts = {
            **base,
            "format": "bestaudio/best",
            "outtmpl": str(song_dir / "download.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        (song_dir / "download.wav").rename(wav)
    (song_dir / "info.json").write_text(json.dumps(song.__dict__, ensure_ascii=False, indent=2))
    return wav, song


def local_audio(path: Path, workdir: Path) -> tuple[Path, SongInfo]:
    """Use a local audio file instead of a YouTube link."""
    song_dir = workdir / ("local-" + re.sub(r"[^\w-]", "_", path.stem))
    song_dir.mkdir(parents=True, exist_ok=True)
    wav = song_dir / "song.wav"
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), str(wav)], check=True
        )
    return wav, SongInfo(title=path.stem, artist="", video_id=song_dir.name)


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
    except ImportError:
        return False
    return True


def separate_drums(wav: Path, model: str = "htdemucs") -> Path:
    """Isolate the drum stem with Demucs. Result is cached next to the input."""
    out = wav.parent / f"{wav.stem}_drums.wav"
    if out.exists():
        return out
    sep_dir = wav.parent / "separated"
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "drums", "-n", model,
         "-o", str(sep_dir), str(wav)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    (sep_dir / model / wav.stem / "drums.wav").rename(out)
    shutil.rmtree(sep_dir, ignore_errors=True)
    return out


def trim(wav: Path, start: float | None, end: float | None) -> Path:
    """Cut a section out of the song (seconds)."""
    if start is None and end is None:
        return wav
    tag = f"{start or 0:g}-{end if end is not None else 'end'}"
    out = wav.parent / f"{wav.stem}_{tag}.wav"
    if not out.exists():
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav)]
        if start:
            cmd += ["-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        subprocess.run(cmd + [str(out)], check=True)
    return out
