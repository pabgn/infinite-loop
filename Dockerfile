FROM python:3.12-slim

# ffmpeg (yt-dlp + pydub), libsndfile1 (librosa/soundfile), gcc (some scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
