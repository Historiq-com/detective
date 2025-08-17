#!/usr/bin/env python3
import os, json, time, sys, subprocess, logging, requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from google.auth import default
from google.auth import impersonated_credentials

# ---- CONFIG (edit these) -----------------------------------------------------
# The Cloud Run SERVICE URL **origin only** (no path, no trailing slash):
BASE_ORIGIN = "https://seg-api-dev-268616946422.us-central1.run.app"

# Endpoint paths exactly as your app defines them:
HEALTH_PATH  = "/healthz"   # GET
PREDICT_PATH = "/predict"   # POST

IMAGE_PATH = "testfile.png"
CALLER_SA  = "vertex-ai-app@vlgo-site-567f8.iam.gserviceaccount.com"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---- ID token helpers --------------------------------------------------------
import time, subprocess, logging
from google.auth import default
from google.auth import impersonated_credentials
from google.auth.transport.requests import Request
from google.oauth2 import id_token

CALLER_SA = "vertex-ai-app@vlgo-site-567f8.iam.gserviceaccount.com"
_ID_TOKEN_CACHE = {"token": None, "exp": 0}

def _id_token_via_metadata(aud: str) -> str:
    # Works in Cloud Run (metadata server)
    return id_token.fetch_id_token(Request(), aud)

def _id_token_via_impersonation(aud: str) -> str:
    # Version-agnostic: build IDTokenCredentials from impersonated access token creds
    # Requires your USER to have roles/iam.serviceAccountTokenCreator on CALLER_SA
    src, _ = default()  # from `gcloud auth application-default login` locally
    target = impersonated_credentials.Credentials(
        source_credentials=src,
        target_principal=CALLER_SA,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        lifetime=3600,
    )
    id_creds = impersonated_credentials.IDTokenCredentials.from_credentials(
        target, target_audience=aud
    )
    id_creds.refresh(Request())
    return id_creds.token

def _id_token_via_gcloud(aud: str) -> str:
    # Fallback for local dev if needed; impersonate the same SA
    out = subprocess.run(
        ["gcloud", "auth", "print-identity-token",
         f"--audiences={aud}",
         f"--impersonate-service-account={CALLER_SA}"],
        capture_output=True, text=True, check=True
    )
    return out.stdout.strip()

def get_id_token(aud: str) -> str:
    now = time.time()
    if _ID_TOKEN_CACHE["token"] and (_ID_TOKEN_CACHE["exp"] - now > 300):
        return _ID_TOKEN_CACHE["token"]
    for fn in (_id_token_via_metadata, _id_token_via_impersonation, _id_token_via_gcloud):
        try:
            tok = fn(aud)
            _ID_TOKEN_CACHE.update(token=tok, exp=now + 55 * 60)
            return tok
        except Exception as e:
            logging.warning(f"{fn.__name__} failed: {e}")
    raise RuntimeError("All ID token methods failed")

def auth_headers(aud: str) -> dict:
    return {"Authorization": f"Bearer {get_id_token(aud)}"}

# ---- Calls -------------------------------------------------------------------
# ... keep your auth helpers as-is ...

BASE_ORIGIN = "https://seg-api-dev-268616946422.us-central1.run.app"
PREDICT_PATH = "/predict"
IMAGE_PATH = "testfile.png"

def post_image(path: str, image_path: str, timeout=600):
    import os, requests
    url = f"{BASE_ORIGIN}{path}"
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        r = requests.post(url, files=files, headers=auth_headers(BASE_ORIGIN), timeout=timeout)
    r.raise_for_status()
    return r

if __name__ == "__main__":
    resp = post_image(PREDICT_PATH, IMAGE_PATH)
    print("✅ Predict OK:", resp.status_code)
    print(resp.text[:500])

