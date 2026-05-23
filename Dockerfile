FROM python:3.12-slim

# ffmpeg (yt-dlp + pydub), libsndfile1 (librosa/soundfile), gcc (some scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the yt-dlp EJS challenge solver so it's cached in the image
RUN yt-dlp --remote-components ejs:github -o /dev/null -- "https://www.youtube.com/watch?v=jNQXAC9IVRw" 2>&1 || true

COPY . .

ENV INFINITE_LOOP_CACHE=/tmp/infinite_loop_cache

EXPOSE 8000

# 1 worker (analysis_status is in-memory — multi-worker would fragment it)
# 4 threads so status polls don't block while analysis runs
# 300s timeout covers long analysis jobs
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} \
             --workers 1 \
             --threads 4 \
             --timeout 300 \
             app:app
