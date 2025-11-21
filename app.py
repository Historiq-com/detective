# app.py
import os, io, json, base64, logging, threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterable

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse

# --- Gemini 3 Pro via Vertex AI (service account / ADC) ---
from google import genai
from google.genai.types import GenerateContentConfig, Part
from google.cloud import storage

# --- Meta SAM 3 (detection + segmentation) ---
from transformers import Sam3Model, Sam3Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM3_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Project/location for Vertex AI
PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "vlgo-site-567f8")
# Gemini 3 lives in the global multi-region; keep this fixed for simplicity.
LOCATION = "global"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

# SAM 3 model storage
SAM3_LOCAL_PATH = os.getenv("SAM3_LOCAL_PATH", "/app/sam3")
SAM3_LOCAL_FALLBACK = "/Users/dserrentino/sam3"
SAM3_GCS_URI = os.getenv("SAM3_GCS_URI", "gs://historiq-sam3")
SAM3_SCORE_THRESH = float(os.getenv("SAM3_SCORE_THRESH", "0.05"))
SAM3_MASK_THRESH = float(os.getenv("SAM3_MASK_THRESH", "0.5"))

@dataclass
class Item:
    name: str
    context: str = ""  # optional

@dataclass
class Instance:
    score: float
    bbox: Tuple[float, float, float, float]  # [x0,y0,x1,y1]
    mask: Optional[np.ndarray] = None

app = FastAPI(title="Gemini 3 + SAM 3 (Meta) inference API", version="2.0")

# ---------------- Utilities ----------------
def mask_to_png_b64(mask_bool: np.ndarray) -> str:
    """Encode a boolean mask as transparent PNG (alpha channel) → base64."""
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = (mask_bool.astype(np.uint8) * 255)
    im = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------- Storage helpers ----------------
def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[len("gs://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")

def _download_gcs_directory(uri: str, dest_dir: Path) -> None:
    """Mirror a small directory of blobs from GCS into dest_dir.
    This is just getting the model weights from GCS into the container 
    so we're not downloading the entire model from huggingface live every time."""
    bucket_name, prefix = _parse_gs_uri(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    dest_dir.mkdir(parents=True, exist_ok=True)

    found_any = False
    for blob in blobs:
        if not blob.name or blob.name.endswith("/"):
            continue
        found_any = True
        rel = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
        out_path = dest_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"[SAM3] Downloading gs://{bucket_name}/{blob.name} -> {out_path}")
        blob.download_to_filename(str(out_path))

    if not found_any:
        raise FileNotFoundError(f"No blobs found under {uri}")

def _ensure_sam3_weights() -> Path:
    """Return a path with SAM3 weights, downloading from GCS if needed."""
    candidates: Iterable[Path] = [
        Path(SAM3_LOCAL_PATH),
        Path(SAM3_LOCAL_FALLBACK),
    ]
    for cand in candidates:
        if cand.exists() and (cand / "model.safetensors").exists():
            return cand
        if cand.exists() and (cand / "sam3.pt").exists():
            return cand

    dest = Path(SAM3_LOCAL_PATH)
    if SAM3_GCS_URI:
        _download_gcs_directory(SAM3_GCS_URI, dest)
        return dest
    raise FileNotFoundError("SAM3 weights not found locally and no GCS URI provided")


def _load_sam3_if_needed():
    """Lazy-load SAM3 once, thread-safe."""
    if app.state.sam3_model is not None and app.state.sam3_processor is not None:
        return
    with app.state._sam3_lock:
        if app.state.sam3_model is not None and app.state.sam3_processor is not None:
            return
        sam3_path = _ensure_sam3_weights()
        logging.info(f"[SAM3] Loading model from {sam3_path} on {DEVICE} (dtype={SAM3_DTYPE})")
        app.state.sam3_model = Sam3Model.from_pretrained(
            str(sam3_path),
            torch_dtype=SAM3_DTYPE,
        ).to(DEVICE).eval()
        app.state.sam3_processor = Sam3Processor.from_pretrained(str(sam3_path))
        app.state.sam3_path = str(sam3_path)

# ---------------- Startup: load models ----------------
@app.on_event("startup")
def _load_models():
    # Gemini via Vertex AI using service account / ADC
    app.state.gemini = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
    )

    # Lazy-load SAM3 on first request to keep startup fast
    app.state.sam3_model = None
    app.state.sam3_processor = None
    app.state.sam3_path = None
    app.state._sam3_lock = threading.Lock()

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok", 200)

# ---------------- Step 1: Gemini subjects/objects (SAM3-ready phrases) ----------------
def step1_gemini_subjects_objects(img_bytes: bytes, mime: str) -> Dict[str, List[Item]]:
    # Minimal schema with optional context; phrases constrained for SAM3 text prompts
    schema = {
        "type": "object",
        "properties": {
            "subjects": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type":"string", "minLength":2, "maxLength":64},
                    "context": {"type":"string"}
                },
                "required": ["name"],
                "additionalProperties": False
            }},
            "objects": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type":"string", "minLength":2, "maxLength":64},
                    "context": {"type":"string"}
                },
                "required": ["name"],
                "additionalProperties": False
            }},
        },
        "required": ["subjects","objects"],
        "additionalProperties": False
    }

    prompt = (
        "Return ONLY JSON matching the schema.\n"
        "Goal: produce short SAM3 text prompts (2–6 words) that uniquely identify visible people (subjects) "
        "and notable non-human items (objects).\n\n"
        "Rules:\n"
        "- Keep each 'name' to 2–6 words, no commas, no 'and'.\n"
        "- Be specific: add ONE helpful disambiguator (color/material/action/location).\n"
        "  Good: 'woman sweeping with broom', 'man eating sandwich', 'blue enamel mug', 'wooden dining chair'.\n"
        "  Bad: 'cup', 'person with thing'.\n"
        "- Use 'man'/'woman' only if obvious; otherwise 'person'.\n"
        "- 'context' is optional; include only if meaningful for archivists.\n"
        "- No counts/coords/masks. No text outside JSON."
    )

    part = Part.from_bytes(data=img_bytes, mime_type=mime or "image/jpeg")
    cfg = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=2048,
    )
    resp = app.state.gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[part, prompt],
        config=cfg,
    )
    if not resp.text:
        # Check if response was truncated due to token limit
        if (hasattr(resp, 'candidates') and resp.candidates and 
            resp.candidates[0].finish_reason and 
            str(resp.candidates[0].finish_reason) == 'FinishReason.MAX_TOKENS'):
            raise ValueError("Response truncated due to token limit. Consider increasing max_output_tokens.")
        raise ValueError(f"Empty response from Gemini API. Response: {resp}")
    data = json.loads(resp.text)

    # Cleanup: trim, dedup (case-insensitive), keep optional context
    def _clean(items):
        seen = set()
        out = []
        for x in items or []:
            name = " ".join((x.get("name","")).split())
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(Item(name=name, context=x.get("context","") or ""))
        return out

    subs = _clean(data.get("subjects", []))
    objs = _clean(data.get("objects", []))
    return {"subjects": subs, "objects": objs}

# ---------------- Step 2: SAM 3 detection + segmentation ----------------
def _move_to_device(obj, device: str, float_dtype: torch.dtype):
    """
    Recursively move/cast tensors in nested structures to the target device/dtype.
    Handles dicts, lists, tuples, and tensors.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.to(device=device, dtype=float_dtype, non_blocking=True)
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, float_dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_move_to_device(v, device, float_dtype) for v in obj]
        return type(obj)(seq)
    return obj

def _sam3_segment(image: Image.Image, items: List[Item]) -> List[List[Instance]]:
    """
    For each item, run SAM3 with a text prompt and return a list of instances.
    """
    _load_sam3_if_needed()
    proc: Sam3Processor = app.state.sam3_processor
    model: Sam3Model = app.state.sam3_model

    all_results: List[List[Instance]] = []
    for it in items:
        prompt = it.name.strip()
        batch = proc(images=image, text=prompt, return_tensors="pt")
        batch = _move_to_device(batch, DEVICE, SAM3_DTYPE)

        with torch.inference_mode():
            outputs = model(**batch)

        processed = proc.post_process_instance_segmentation(
            outputs,
            threshold=SAM3_SCORE_THRESH,
            mask_threshold=SAM3_MASK_THRESH,
            target_sizes=batch.get("original_sizes").tolist(),
        )[0]

        masks = processed.get("masks")
        boxes = processed.get("boxes")
        scores = processed.get("scores")

        if masks is None:
            masks_np = []
        else:
            masks_np = masks.cpu().numpy()
            if masks_np.dtype != bool:
                masks_np = masks_np > 0.5

        instances: List[Instance] = []
        boxes_seq = boxes if boxes is not None else []
        scores_seq = scores if scores is not None else []

        for idx, score in enumerate(scores_seq):
            if len(boxes_seq) > idx:
                box_tensor = boxes_seq[idx]
                if hasattr(box_tensor, "detach"):
                    bbox = box_tensor.detach().cpu().tolist()
                else:
                    bbox = list(box_tensor)
            else:
                bbox = [0, 0, 0, 0]

            if hasattr(score, "detach"):
                score_val = float(score.detach().cpu())
            else:
                score_val = float(score)
            mask_arr = masks_np[idx] if len(masks_np) > idx else None
            instances.append(Instance(
                score=score_val,
                bbox=(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
                mask=mask_arr
            ))
        all_results.append(instances)
    return all_results

# ---------------- API ----------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    mime = image.content_type or "image/jpeg"
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 1) Gemini → labels (+ optional context)
    lists = step1_gemini_subjects_objects(img_bytes, mime)
    subjects: List[Item] = lists["subjects"]
    objects:  List[Item] = lists["objects"]

    # 2) SAM 3 → detection + segmentation for each label (multi-instance aware)
    subj_instances = _sam3_segment(pil, subjects)
    obj_instances  = _sam3_segment(pil, objects)

    def pack(items: List[Item], instances: List[List[Instance]]) -> List[Dict]:
        out = []
        for it, inst_list in zip(items, instances):
            inst_payload = []
            for inst in inst_list:
                mask_b64 = None
                area = None
                if inst.mask is not None:
                    mask_b64 = mask_to_png_b64(inst.mask)
                    area = int(np.sum(inst.mask))
                inst_payload.append({
                    "score": inst.score,
                    "bbox": [int(round(v)) for v in inst.bbox],
                    "mask_png_base64": mask_b64,
                    "mask_area": area,
                })
            out.append({
                "name": it.name,
                "context": it.context,
                "instances": inst_payload,
            })
        return out

    resp = {
        "image_size": {"width": pil.width, "height": pil.height},
        "sam3": {
            "model_path": app.state.sam3_path,
            "score_threshold": SAM3_SCORE_THRESH,
            "mask_threshold": SAM3_MASK_THRESH,
            "device": DEVICE,
        },
        "subjects": pack(subjects, subj_instances),
        "objects":  pack(objects, obj_instances)
    }
    return JSONResponse(resp)
