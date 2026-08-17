# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for faster, reproducible installs (cached layer)
RUN pip install --no-cache-dir uv==0.4.30

# Install Python dependencies (only rebuild if requirements.txt changes)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -r requirements.txt

# SAM3 weights baked into the image (large but stable layer)
COPY sam3 /app/sam3

# Copy application code (changes most frequently, smallest layer)
COPY app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
