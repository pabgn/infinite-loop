#!/usr/bin/env python3
"""INFINITE LOOP — Flask application entry point.

Run with:
    python app.py
or in development:
    flask --app app run --debug
"""

import hashlib
import json
import threading

from flask import Flask, jsonify, render_template, request, send_file, Response

from infinite_loop.config import CACHE_DIR, DEBUG, HOST, PORT, analysis_status
from infinite_loop.pipeline import analyze_track

app = Flask(__name__)


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    url = request.json.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL requerida"}), 400

    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    graph_path = CACHE_DIR / f"{url_hash}.json"

    # Already analyzed?
    if graph_path.exists() and url_hash not in analysis_status:
        graph = json.loads(graph_path.read_text())
        return jsonify({"status": "done", "hash": url_hash, "cached": True, "meta": graph.get("meta", {})})

    if url_hash in analysis_status:
        s = analysis_status[url_hash]
        return jsonify({"status": s.get("status", "analyzing"), "hash": url_hash, **s})

    # Start analysis
    analysis_status[url_hash] = {"status": "analyzing", "progress": 0, "message": "Iniciando..."}
    t = threading.Thread(target=analyze_track, args=(url, url_hash), daemon=True)
    t.start()
    return jsonify({"status": "analyzing", "hash": url_hash})


@app.route("/api/status/<url_hash>")
def api_status(url_hash):
    if url_hash not in analysis_status:
        graph_path = CACHE_DIR / f"{url_hash}.json"
        if graph_path.exists():
            return jsonify({"status": "done", "progress": 100})
        return jsonify({"status": "not_found"}), 404
    return jsonify(analysis_status[url_hash])


@app.route("/api/graph/<url_hash>")
def api_graph(url_hash):
    graph_path = CACHE_DIR / f"{url_hash}.json"
    if not graph_path.exists():
        return jsonify({"error": "Not found"}), 404
    return Response(graph_path.read_text(), mimetype="application/json")


@app.route("/api/audio/<url_hash>")
def api_audio(url_hash):
    audio_path = CACHE_DIR / f"{url_hash}.mp3"
    if not audio_path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(audio_path), mimetype="audio/mpeg")


def main():
    print("\n" + "═" * 50)
    print("  INFINITE LOOP — Motor de Música Eterna")
    print("═" * 50)
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Abre en tu navegador: http://localhost:{PORT}")
    print("═" * 50 + "\n")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)


if __name__ == "__main__":
    main()
