"""Shared configuration and runtime state for INFINITE LOOP."""

import tempfile
from pathlib import Path

# Directory where downloaded audio + analysis graphs are cached.
# Override with the INFINITE_LOOP_CACHE environment variable.
import os

CACHE_DIR = Path(os.environ.get("INFINITE_LOOP_CACHE",
                                Path(tempfile.gettempdir()) / "infinite_loop_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Server settings (overridable via environment).
HOST = os.environ.get("INFINITE_LOOP_HOST", "0.0.0.0")
PORT = int(os.environ.get("INFINITE_LOOP_PORT") or os.environ.get("PORT") or "8149")
DEBUG = os.environ.get("INFINITE_LOOP_DEBUG", "").lower() in ("1", "true", "yes")

# In-memory analysis progress tracker: url_hash -> {status, progress, message, ...}
analysis_status: dict = {}
