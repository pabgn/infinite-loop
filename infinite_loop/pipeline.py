"""Download + analysis pipeline (runs in a background thread)."""

import json

from .analysis import analyze
from .config import CACHE_DIR, analysis_status


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
                "format": "bestaudio/best",
                "outtmpl": str(CACHE_DIR / f"{url_hash}.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "quiet": True,
                "no_warnings": True,
            }
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
            # Map 0-100 into 20-95 range (download already done)
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

