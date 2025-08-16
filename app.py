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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OWL_REPO = os.getenv("OWL_REPO", "google/owlv2-base-patch16")
OWL_SCORE_THRESH = float(os.getenv("OWL_SCORE_THRESH", "0.25"))

# Ultralytics weight id (auto-downloads on first use)
ULTRA_SAM_WEIGHTS = os.getenv("ULTRA_SAM_WEIGHTS", "sam2.1_b.pt")

# Helpful normalization for person labels coming from Gemini
PERSON_SYNONYMS = {"person","people","man","woman","boy","girl","human","men","women","person(s)"}

@dataclass
class Item:
    name: str
    context: str

@dataclass
class Det:
    label: str
    score: float
    bbox: Tuple[float, float, float, float]  # [x0,y0,x1,y1]

app = FastAPI(title="Gemini (Vertex AI) + OWLv2 + SAM 2.1 (Ultralytics)", version="1.0")

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

# ---------------- Step 1: Gemini subjects/objects + context ----------------
def step1_gemini_subjects_objects(img_bytes: bytes, mime: str) -> Dict[str, List[Item]]:
    schema = {
        "type": "object",
        "properties": {
            "subjects": {"type": "array", "items": {
                "type": "object",
                "properties": {"name":{"type":"string"}, "context":{"type":"string"}},
                "required": ["name","context"]
            }},
            "objects": {"type": "array", "items": {
                "type": "object",
                "properties": {"name":{"type":"string"}, "context":{"type":"string"}},
                "required": ["name","context"]
            }},
        },
        "required": ["subjects","objects"]
    }

    prompt = (
        "List visible entities in two groups:\n"
        "1) subjects = the primary human actors.\n"
        "2) objects = notable non-human items.\n"
        "For each item return JSON fields:\n"
        "- name: Be extremely descriptive with the fewest number of words possible to describe the item. This description witll be used by an OWLv2 model to detect the item in the image.\n"
        "- context: One sentence explaining its importance of the itemin the historical context of the image. Your audience is archivists and historians.\n"
        "No coordinates or masks."
    )

    part = gtypes.Part.from_bytes(data=img_bytes, mime_type=mime or "image/jpeg")
    resp = app.state.gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[part, prompt],
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema
        ),
    )
    data = json.loads(resp.text)
    subs = [Item(**x) for x in data.get("subjects", [])]
    objs = [Item(**x) for x in data.get("objects", [])]
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
    # sort by score desc
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

    # Ultralytics expects a list of bboxes per image:
    # predict(source, bboxes=[[x1,y1,x2,y2], ...], imgsz=<opt>)
    results = app.state.sam.predict(
        source=image,
        bboxes=[list(map(float, b)) for b in boxes_xyxy],
        verbose=False
    )
    # results is a list per image; we passed one image → results[0]
    r = results[0]

    # r.masks.data is a tensor [N, H, W] (boolean-ish 0/1)
    masks_bool: List[np.ndarray] = []
    if r.masks is not None and hasattr(r.masks, "data"):
        m = r.masks.data  # torch.Tensor
        m = (m > 0.5).cpu().numpy().astype(bool)
        for i in range(m.shape[0]):
            masks_bool.append(m[i])
    else:
        # No masks found for the given boxes (rare); return empties aligned to image size
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

    # 1) Gemini → names + one-sentence context
    lists = step1_gemini_subjects_objects(img_bytes, mime)
    subjects: List[Item] = lists["subjects"]
    objects:  List[Item] = lists["objects"]

    # 2) OWLv2 → boxes for union of labels
    vocab = [it.name for it in subjects] + [it.name for it in objects]
    dets = step2_owl_boxes(pil, vocab)

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
