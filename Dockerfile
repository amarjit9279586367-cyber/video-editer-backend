FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 1 worker is required: job state is kept in memory. Threads handle concurrency.
CMD gunicorn main:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-10000}
