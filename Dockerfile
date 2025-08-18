FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the SAM 2.1 model file to avoid downloading it every time
COPY sam2.1_b.pt /app/sam2.1_b.pt

# Copy the OWLv2 model files to avoid downloading from Hugging Face
COPY owlv2-model/ /app/owlv2-model/

COPY app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
