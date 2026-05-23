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


def _find_executable(name: str, extra_dirs: list) -> str | None:
    """Return the full path to an executable, searching PATH + extra_dirs."""
    import shutil, glob
    found = shutil.which(name)
    if found:
        return found
    for d in extra_dirs:
        # Support nvm-style versioned dirs: ~/.nvm/versions/node/*/bin/node
        for candidate in glob.glob(os.path.join(d, "*", "bin", name)) + [os.path.join(d, name)]:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _browser_available(browser: str) -> bool:
    """Return True if the browser's cookie store exists on this machine."""
    import platform
    home = os.path.expanduser("~")
    paths = {
        "safari":   [f"{home}/Library/Cookies/Cookies.binarycookies"],
        "chrome":   [f"{home}/Library/Application Support/Google/Chrome",
                     f"{home}/.config/google-chrome"],
        "firefox":  [f"{home}/Library/Application Support/Firefox/Profiles",
                     f"{home}/.mozilla/firefox"],
        "chromium": [f"{home}/Library/Application Support/Chromium",
                     f"{home}/.config/chromium"],
    }
    return any(os.path.exists(p) for p in paths.get(browser, []))


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
            base_opts = {
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

            # Strategy 1: android client — no n-challenge or signature solving needed,
            # works on datacenter IPs for public videos. Must NOT have cookies set
            # (yt-dlp skips android entirely when any cookiefile/cookiesfrombrowser is present).
            android_opts = {**base_opts, "extractor_args": {"youtube": {"player_client": ["android"]}}}

            # Strategy 2: web client with cookies — for age-restricted / sign-in-required videos.
            web_opts = {**base_opts}
            cookies_file = _ensure_cookies_file()
            if cookies_file:
                web_opts["cookiefile"] = cookies_file
            elif not os.environ.get("YOUTUBE_COOKIES") and not os.environ.get("YOUTUBE_COOKIES_PATH"):
                for browser in ("safari", "chrome", "firefox", "chromium"):
                    if _browser_available(browser):
                        web_opts["cookiesfrombrowser"] = (browser,)
                        break
            _node = _find_executable("node", [
                os.path.expanduser("~/.nvm/versions/node"),
                "/usr/local/bin", "/opt/homebrew/bin", "/usr/bin",
            ])
            if _node:
                web_opts["js_runtimes"] = {"node": {"path": _node}}
            web_opts["remote_components"] = ["ejs:github"]

            info = None
            for attempt, ydl_opts in enumerate([android_opts, web_opts]):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                    break
                except Exception as e:
                    if attempt == 0:
                        # Clean up any partial file before retrying
                        for f in CACHE_DIR.glob(f"{url_hash}.*"):
                            try: f.unlink()
                            except: pass
                        continue
                    raise
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
