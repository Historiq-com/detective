


docker run --rm -p 8080:8080 \
  -v $HOME/.config/gcloud:/root/.config/gcloud:ro \
  -e GOOGLE_CLOUD_PROJECT="vlgo-site-567f8" \
  -e GOOGLE_CLOUD_LOCATION="us-central1" \
  subjobj-sam2

  gcloud run deploy seg-api-dev \
  --source . \
  --region us-central1 \
  --service-account vertex-ai-app@vlgo-site-567f8.iam.gserviceaccount.com \
  --cpu 4 --memory 8Gi \
  --timeout 1200 \
  --no-allow-unauthenticated

## What this service does
- FastAPI endpoint `/predict` takes an image file.
- Gemini 3 (Vertex AI) proposes subjects/objects as text prompts.
- Meta SAM 3 runs detection and segmentation for each prompt (multi-instance), returning boxes, scores, and masks (base64 PNG).

## Local run (CPU)
```bash
cd /Users/dserrentino/detective
# Install deps (uses uv; falls back to pip if you prefer)
uv pip install -r requirements.txt

# Point to local SAM3 weights
export SAM3_LOCAL_PATH=/Users/dserrentino/sam3

# Start API (pick a port)
uv run --python .venv/bin/python uvicorn app:app --host 0.0.0.0 --port 8000
```

## Quick test
```bash
# Health
curl -i http://localhost:8000/healthz
# Predict
curl -X POST http://localhost:8000/predict \
  -F "image=@testfile.png;type=image/png" \
  -o resp.json
```

## Env vars that matter
- `SAM3_LOCAL_PATH` (required for local) or `SAM3_GCS_URI` (for Cloud Run) points to the SAM3 model folder.
- `GOOGLE_CLOUD_PROJECT` (defaults to `vlgo-site-567f8`) for Gemini auth; location is fixed to `global`.
