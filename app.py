# app.py
import os, io, json, base64
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import cv2
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse

# --- Gemini 2.5 Flash via Vertex AI (service account / ADC) ---
from google import genai
from google.genai import types as gtypes

# --- OWLv2 (open-vocab detector) ---
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --- Ultralytics SAM 2.1 ---
from ultralytics import SAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Project/location for Vertex AI
PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "vlgo-site-567f8")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

OWL_REPO = os.getenv("OWL_REPO", "google/owlv2-base-patch16")
OWL_SCORE_THRESH = float(os.getenv("OWL_SCORE_THRESH", "0.1"))

# Ultralytics weight id (auto-downloads on first use)
ULTRA_SAM_WEIGHTS = os.getenv("ULTRA_SAM_WEIGHTS", "sam2.1_b.pt")

# Helpful normalization for person labels coming from Gemini
PERSON_SYNONYMS = {"person","people","man","woman","boy","girl","human","men","women","person(s)"}

@dataclass
class Item:
    name: str
    context: str = ""  # optional

@dataclass
class Det:
    label: str
    score: float
    bbox: Tuple[float, float, float, float]  # [x0,y0,x1,y1]

app = FastAPI(title="Gemini (Vertex AI) + OWLv2 + SAM 2.1 (Ultralytics)", version="1.1")

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

def normalize_label(s: str) -> str:
    s2 = s.strip().lower()
    return "person" if s2 in PERSON_SYNONYMS else s.strip()

def _expand_people_queries(vocab: List[str]) -> List[str]:
    """Add a 'person' variant for gendered people phrases to boost OWL recall."""
    expanded = []
    for v in vocab:
        expanded.append(v)
        low = v.lower()
        for term in ["woman","women","man","men","girl","boy","people","person(s)"]:
            if term in low:
                expanded.append(low.replace(term, "person"))
                break
    # preserve order, drop dups
    return list(dict.fromkeys(expanded))

# ---------------- Startup: load models ----------------
@app.on_event("startup")
def _load_models():
    # Gemini via Vertex AI using service account / ADC
    app.state.gemini = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
    )

    # OWLv2
    app.state.owl_proc = AutoProcessor.from_pretrained(OWL_REPO)
    app.state.owl = AutoModelForZeroShotObjectDetection.from_pretrained(OWL_REPO).to(DEVICE).eval()

    # Ultralytics SAM 2.1 (auto-downloads weights like "sam2.1_b.pt")
    app.state.sam = SAM(ULTRA_SAM_WEIGHTS)

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok", 200)

# ---------------- Step 1: Gemini subjects/objects (OWLv2-ready phrases) ----------------
def step1_gemini_subjects_objects(img_bytes: bytes, mime: str) -> Dict[str, List[Item]]:
    # Minimal schema with optional context; phrases constrained for OWLv2
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
        "Goal: produce short OWLv2-ready phrases (2–6 words) that uniquely identify visible people (subjects) "
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

    part = gtypes.Part.from_bytes(data=img_bytes, mime_type=mime or "image/jpeg")
    resp = app.state.gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[part, prompt],
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0, top_p=1.0, max_output_tokens=2048
        ),
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

# ---------------- Step 2: OWLv2 boxes for labels ----------------
def step2_owl_boxes(image: Image.Image, label_vocab: List[str]) -> List[Det]:
    proc = app.state.owl_proc
    model = app.state.owl

    vocab = [normalize_label(x) for x in label_vocab]
    texts = [[v] for v in vocab]
    inputs = proc(images=image, text=texts, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image.height, image.width]], device=DEVICE)
    results = proc.post_process_object_detection(
        outputs=outputs,
        threshold=OWL_SCORE_THRESH,
        target_sizes=target_sizes
    )[0]

    boxes = results["boxes"].cpu().numpy().tolist()
    scores = results["scores"].cpu().numpy().tolist()
    labels_idx = results["labels"].cpu().numpy().tolist()

    dets: List[Det] = []
    for b, s, li in zip(boxes, scores, labels_idx):
        dets.append(Det(
            label=vocab[int(li)],
            score=float(s),
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        ))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets

def assign_detections_to_items(items: List[Item], dets: List[Det]) -> List[Optional[Det]]:
    """
    For each item, consume the highest-score unused detection whose label
    matches the normalized item name (many-to-many handled by popping).
    """
    buckets: Dict[str, List[Det]] = {}
    for d in dets:
        buckets.setdefault(d.label.lower(), []).append(d)
    for v in buckets.values():
        v.sort(key=lambda d: d.score, reverse=True)

    assigned: List[Optional[Det]] = []
    for it in items:
        key = normalize_label(it.name).lower()
        lst = buckets.get(key, [])
        det = lst.pop(0) if lst else None
        assigned.append(det)
    return assigned

# ---------------- Step 3: SAM 2.1 masks (Ultralytics) ----------------
def step3_sam_masks_ultralytics(image: Image.Image, boxes_xyxy: List[Tuple[float,float,float,float]]) -> List[np.ndarray]:
    """
    Use Ultralytics SAM 2.1 with bbox prompts to produce boolean masks (H,W).
    Ultralytics returns masks aligned to input size.
    """
    if not boxes_xyxy:
        return []

    results = app.state.sam.predict(
        source=image,
        bboxes=[list(map(float, b)) for b in boxes_xyxy],
        verbose=False
    )
    r = results[0]

    masks_bool: List[np.ndarray] = []
    if r.masks is not None and hasattr(r.masks, "data"):
        m = r.masks.data  # torch.Tensor
        m = (m > 0.5).cpu().numpy().astype(bool)
        for i in range(m.shape[0]):
            masks_bool.append(m[i])
    else:
        W, H = image.size
        for _ in boxes_xyxy:
            masks_bool.append(np.zeros((H, W), dtype=bool))
    return masks_bool

# ---------------- API ----------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    mime = image.content_type or "image/jpeg"
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 1) Gemini → OWLv2-ready phrases (+ optional context)
    lists = step1_gemini_subjects_objects(img_bytes, mime)
    subjects: List[Item] = lists["subjects"]
    objects:  List[Item] = lists["objects"]

    # 2) OWLv2 → boxes for union of labels (with 'person' fallbacks for gendered phrases)
    vocab = [it.name for it in subjects] + [it.name for it in objects]
    vocab = _expand_people_queries(vocab)
    dets = step2_owl_boxes(pil, vocab)

    # Optional debug on total miss
    if not dets:
        print("[OWL] No detections for queries:", vocab)

    subj_dets = assign_detections_to_items(subjects, dets)
    obj_dets  = assign_detections_to_items(objects, dets)

    # Collect all boxes in order (so mask indices line up)
    all_boxes: List[Tuple[float,float,float,float]] = []
    index_map: List[Tuple[str,int]] = []  # ("subject"/"object", idx)
    for i, d in enumerate(subj_dets):
        if d is not None:
            all_boxes.append(d.bbox)
            index_map.append(("subject", i))
    for i, d in enumerate(obj_dets):
        if d is not None:
            all_boxes.append(d.bbox)
            index_map.append(("object", i))

    # 3) SAM 2.1 → masks for those boxes
    masks: List[np.ndarray] = []
    if all_boxes:
        masks = step3_sam_masks_ultralytics(pil, all_boxes)

    # Build response
    def pack(items: List[Item], det_list: List[Optional[Det]]) -> List[Dict]:
        out = []
        for idx, (it, det) in enumerate(zip(items, det_list)):
            bbox = [int(round(v)) for v in det.bbox] if det else None
            mask_b64 = None
            if det:
                # find its mask by its position in index_map
                for k, (kind, j) in enumerate(index_map):
                    if (kind == "subject" and items is subjects and j == idx) or \
                       (kind == "object" and items is objects and j == idx):
                        mask_b64 = mask_to_png_b64(masks[k])
                        break
            out.append({
                "name": it.name,
                "context": it.context,
                "bbox": bbox,
                "mask_png_base64": mask_b64
            })
        return out

    resp = {
        "image_size": {"width": pil.width, "height": pil.height},
        "subjects": pack(subjects, subj_dets),
        "objects":  pack(objects, obj_dets)
    }
    return JSONResponse(resp)
