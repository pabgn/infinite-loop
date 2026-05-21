"""Download + analysis pipeline (runs in a background thread)."""

import json
import os
import tempfile

from .analysis import analyze
from .config import CACHE_DIR, analysis_status

# Path to a persistent cookies file written once at startup from the env var.
_COOKIES_FILE: str | None = None


def _ensure_cookies_file() -> str | None:
    """
    Return a path to a Netscape-format cookies file, or None if not configured.

    Reads YOUTUBE_COOKIES env var (full cookies.txt content) once and writes it
    to a temp file that persists for the lifetime of the process.
    """
    global _COOKIES_FILE
    if _COOKIES_FILE is not None:
        return _COOKIES_FILE

    # Explicit file path takes priority (useful when mounting a secret file)
    path = os.environ.get("YOUTUBE_COOKIES_PATH", "").strip()
    if path and os.path.exists(path):
        _COOKIES_FILE = path
        return _COOKIES_FILE

    # Inline cookies content stored as an env var
    content = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if content:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="yt_cookies_"
        )
        f.write(content)
        f.close()
        _COOKIES_FILE = f.name
        return _COOKIES_FILE

    return None


def analyze_track(url: str, url_hash: str):
    """Full pipeline v2: download → analyze → build jump graph"""
    try:
        import yt_dlp

        status = analysis_status[url_hash]

        # 1. Download
        status.update({"progress": 5, "message": "Descargando audio de YouTube..."})
        audio_path = CACHE_DIR / f"{url_hash}.mp3"
        graph_path = CACHE_DIR / f"{url_hash}.json"

        if not audio_path.exists():
            ydl_opts = {
                # Permissive chain: audio-only → any combined → absolute worst
                # (ffmpeg extracts audio regardless of what we download)
                "format": "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best[ext=mp4]/best/worst",
                "outtmpl": str(CACHE_DIR / f"{url_hash}.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                # tv_embedded bypasses format restrictions on datacenter IPs;
                # web_creator is the fallback if embedded is unavailable.
                "extractor_args": {
                    "youtube": {"player_client": ["tv_embedded", "web_creator", "web"]}
                },
                "quiet": True,
                "no_warnings": True,
            }

            cookies_file = _ensure_cookies_file()
            if cookies_file:
                ydl_opts["cookiefile"] = cookies_file

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "Unknown")
                duration = info.get("duration", 0)
                thumbnail = info.get("thumbnail", "")
            status["meta"] = {"title": title, "duration": duration, "thumbnail": thumbnail}
        else:
            if graph_path.exists():
                status["meta"] = json.loads(graph_path.read_text()).get("meta", {})
            else:
                status["meta"] = {}

        # 2. Run v2 analysis
        def progress_cb(pct, msg):
            mapped = 20 + int(pct * 0.75)
            status.update({"progress": mapped, "message": msg})

        result = analyze(str(audio_path), status_cb=progress_cb)

        # 3. Strip internal numpy fields, add meta
        result.pop("_similarity_matrix", None)
        result.pop("_novelty", None)
        result["meta"] = status["meta"]

        # 4. Save
        graph_path.write_text(json.dumps(result))

        status.update({
            "status": "done",
            "progress": 100,
            "message": "¡Análisis completo!",
        })

    except Exception as e:
        import traceback
        analysis_status[url_hash].update({
            "status": "error",
            "message": f"Error: {str(e)}",
            "traceback": traceback.format_exc(),
        })
