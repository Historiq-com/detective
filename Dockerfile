FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for faster, reproducible installs
RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# SAM3 weights baked into the image (downloaded in Cloud Build step)
COPY sam3 /app/sam3

COPY app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
