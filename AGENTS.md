# Repository Instructions

## Service boundaries

- This is a Python 3.11 FastAPI service centered on `app.py`. `/predict` combines Gemini prompt generation with SAM segmentation; `/healthz` is the health endpoint.
- Keep model weights out of Git. Local development points `SAM3_LOCAL_PATH` at an existing checkout; Cloud Build stages weights into the Docker build context from `SAM3_GCS_URI`.
- Keep Cloud Run private. Preserve authenticated invocation and the service-account/identity-token flow used by the manual remote smoke script.
- Preserve concurrency-one GPU deployment assumptions unless model memory and request isolation have been revalidated.

## Development and verification

- Install with `uv pip install -r requirements.txt` (or an equivalent isolated pip environment) and run locally with `uvicorn app:app --host 0.0.0.0 --port 8000`.
- Smoke-test `GET /healthz` first, then `POST /predict` with a small image as shown in `Readme.md`.
- The repository has no formal automated test suite or test configuration. `test.py` is an authenticated, stateful remote smoke client with a configured service origin and service account; do not run it or change its target without explicit authorization.
- The Docker image expects a `sam3/` directory in the build context. Normal deployment should use `cloudbuild.yaml`, which stages that directory before building.
- When changing dependencies or model-loading behavior, validate both a CPU/local startup path and the Cloud Build/Docker path where practical.

## Secrets and artifacts

- Do not commit model weights, application-default credentials, or identity tokens. Do not add or overwrite response artifacts unless the task explicitly updates a manual fixture.
- Treat the tracked `resp*.json`, `result.json`, and test image as manual fixtures/artifacts; do not use them as assertions for a passing automated suite.
