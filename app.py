"""
Crop Disease Identification Web Application (Flask Backend)
===========================================================
Web UI and REST API for real-time crop disease diagnosis
using Ultralytics YOLO11s-cls.
"""

import io
import os
import time
from pathlib import Path
from typing import Tuple

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best.pt"))
if not WEIGHTS_PATH.exists():
    candidate_weights = [
        BASE_DIR / "runs" / "classify" / "crop_disease_yolo11" / "yolo11m_plantdoc_field" / "weights" / "best.pt",
        BASE_DIR / "runs" / "classify" / "crop_disease_yolo11" / "yolo11s_plantdoc_field" / "weights" / "best.pt",
    ]
    for cw in candidate_weights:
        if cw.exists():
            WEIGHTS_PATH = cw
            break

# Target device
DEVICE = "0" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print(f"[*] Loading model from: {WEIGHTS_PATH}")
print(f"[*] Accelerator: {DEVICE} ({DEVICE_NAME})")
model = YOLO(str(WEIGHTS_PATH))
print(f"[+] Loaded model with {len(model.names)} classes successfully!")


def parse_disease_label(raw_label: str) -> Tuple[str, str, str]:
    """
    Dynamically parses any class label into (Crop, Disease Condition, Health Status)
    without using any hardcoded lookup tables.
    """
    clean = raw_label.replace("_", " ").strip()
    words = clean.split()
    if not words:
        return ("Unknown", clean, "UNKNOWN")

    # Crop name extraction (handles compound crop names like 'Bell pepper')
    if len(words) > 1 and words[0].lower() == "bell" and words[1].lower() == "pepper":
        crop = "Bell Pepper"
        subwords = words[2:]
    else:
        crop = words[0].capitalize()
        subwords = words[1:]

    # Remove generic anatomical tokens like 'leaf'
    filtered = [w for w in subwords if w.lower() != "leaf"]

    # Detect health status dynamically
    if not filtered or "healthy" in clean.lower():
        return (crop, "Healthy Leaf", "HEALTHY")

    condition = " ".join(filtered).title()
    return (crop, condition, "DISEASED")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def get_info():
    """Returns dynamic model metadata."""
    return jsonify({
        "classes_count": len(model.names),
        "device": DEVICE_NAME,
        "input_resolution": "448 x 448 px",
        "classes": model.names,
    })


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """Receives image via multipart upload and runs real-time diagnosis."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_bytes = uploaded_file.read()
    source_name = uploaded_file.filename

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    # Run YOLO inference at 448px for micro-lesion pathology detection
    t0 = time.perf_counter()
    results = model.predict(source=pil_image, imgsz=448, device=DEVICE, verbose=False)
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    probs = results[0].probs
    top_indices = probs.top5[:min(5, len(model.names))]
    primary_idx = top_indices[0]
    primary_label = model.names[primary_idx]
    primary_conf = float(probs.data[primary_idx]) * 100.0

    crop, condition, health_status = parse_disease_label(primary_label)

    # Top differential candidates directly from model output
    candidates = []
    for rank, idx in enumerate(top_indices, 1):
        c_label = model.names[idx]
        c_crop, c_cond, c_stat = parse_disease_label(c_label)
        candidates.append({
            "rank": rank,
            "label": c_label,
            "crop": c_crop,
            "condition": c_cond,
            "status": c_stat,
            "confidence": round(float(probs.data[idx]) * 100.0, 2)
        })

    print(f"\n[AI Inference] {source_name} -> {primary_label} ({primary_conf:.1f}%) [{health_status}] in {inference_time_ms:.1f}ms")

    return jsonify({
        "success": True,
        "source": source_name,
        "primary": {
            "label": primary_label,
            "crop": crop,
            "condition": condition,
            "status": health_status,
            "confidence": round(primary_conf, 1),
        },
        "candidates": candidates,
        "inference_time_ms": round(inference_time_ms, 1),
        "device": DEVICE_NAME,
    })


if __name__ == "__main__":
    print("\n=======================================================")
    print("  Crop Disease Diagnosis Web Server Running")
    print("  Local Address: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
