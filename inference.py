"""
Crop Disease Identification Tool (Inference Pipeline)
======================================================
Loads the trained Ultralytics YOLO11 model and diagnoses crop diseases
from leaf photographs taken in the field.

Usage:
    python inference.py --image "path/to/leaf.jpg"
    python inference.py "path/to/leaf.jpg"
    python inference.py --image "path/to/leaf.jpg" --save
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from ultralytics import YOLO

# Default weights path (looks in clean models/ directory first)
DEFAULT_WEIGHTS = (
    Path("models/best.pt")
    if Path("models/best.pt").exists()
    else Path("runs/classify/crop_disease_yolo11/yolo11s_plantdoc_field/weights/best.pt")
)


def parse_disease_label(raw_label: str) -> Tuple[str, str, str]:
    """
    Dynamically parses class label into (Crop, Disease Condition, Health Status)
    without using any hardcoded class lists.
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


def run_inference(
    image_path: Path,
    weights_path: Path = DEFAULT_WEIGHTS,
    top_k: int = 3,
    imgsz: int = 448,
    device: str = "auto",
    augment: bool = False,
    save_annotated: bool = False,
    output_dir: Path = Path("predictions"),
) -> List[Dict]:
    """
    Runs classification inference on an input image and prints diagnostic results.
    """
    if not image_path.exists():
        print(f"\n[ERROR] Image not found: {image_path}")
        sys.exit(1)

    if not weights_path.exists():
        print(f"\n[ERROR] Model weights not found at: {weights_path.resolve()}")
        print("Please check your weights path or train the model first.")
        sys.exit(1)

    # Device selection
    if device == "auto":
        target_device = "0" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    # Load Model
    model = YOLO(str(weights_path))

    # Predict (with optional Test-Time Augmentation)
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=target_device,
        augment=augment,
        verbose=False,
    )
    probs = results[0].probs

    top_indices = probs.top5[:top_k]
    ranked_results = []

    print("\n" + "=" * 65)
    print(f"               CROP DISEASE DIAGNOSIS REPORT")
    print("=" * 65)
    print(f" Image Source : {image_path.resolve()}")
    print(f" Model Engine : YOLO11 ({weights_path.name})")
    print(f" Hardware     : {'GPU (CUDA:0)' if target_device == '0' else 'CPU'}")
    print("-" * 65)

    primary_label = model.names[top_indices[0]]
    primary_crop, primary_condition, primary_status = parse_disease_label(primary_label)
    primary_conf = float(probs.data[top_indices[0]]) * 100

    # Primary Diagnosis Card
    status_tag = "[ HEALTHY ]" if primary_status == "HEALTHY" else "[ DISEASED ]"
    print(f" PRIMARY DIAGNOSIS : {status_tag} {primary_label}")
    print(f" Crop Type         : {primary_crop}")
    print(f" Pathology Status  : {primary_condition}")
    print(f" Confidence Score  : {primary_conf:.2f}%\n")

    # Ranked Top-K Breakdown
    print(f" Top-{top_k} Differential Diagnosis Ranking:")
    print("-" * 65)
    print(f"  {'Rank':<5} | {'Category Label':<35} | {'Confidence':<10}")
    print("-" * 65)

    for rank, idx in enumerate(top_indices, 1):
        class_name = model.names[idx]
        confidence = float(probs.data[idx]) * 100.0
        ranked_results.append({
            "rank": rank,
            "class": class_name,
            "confidence": confidence,
        })
        print(f"  #{rank:<4} | {class_name:<35} | {confidence:6.2f}%")

    print("=" * 65)

    # Optional: Save image with diagnosis banner
    if save_annotated:
        output_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(image_path))
        if img is not None:
            banner_text = f"{primary_label} ({primary_conf:.1f}%)"
            color = (40, 180, 40) if primary_status == "HEALTHY" else (30, 30, 220)

            # Draw banner overlay
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(img, banner_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            save_file = output_dir / f"pred_{image_path.name}"
            cv2.imwrite(str(save_file), img)
            print(f" Annotated image saved to: {save_file.resolve()}\n")

    return ranked_results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict crop disease from an input leaf image using trained YOLO11 model"
    )
    # Positional or flagged image argument
    parser.add_argument(
        "pos_image",
        nargs="?",
        default=None,
        help="Path to leaf image file (positional)",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to leaf image file (flagged)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to model weights (.pt). Default: {DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of top disease candidates to display (default: 3)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=448,
        help="Inference image resolution (default: 448)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'auto', '0', 'cpu'",
    )
    parser.add_argument(
        "--augment",
        "-a",
        action="store_true",
        help="Enable Test-Time Augmentation (TTA) for higher accuracy on difficult field images",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save annotated image with prediction overlay in 'predictions/' directory",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine image path from either positional or flagged arg
    target_image = args.image or args.pos_image

    if not target_image:
        print("\n" + "=" * 55)
        print("       PLANTDOC CROP DISEASE INFERENCE CLI")
        print("=" * 55)
        # Interactive prompt if executed without parameters
        target_image = input("Enter path to leaf image: ").strip().strip('"\'')

    if not target_image:
        print("[ERROR] No image path provided. Exiting.")
        sys.exit(1)

    run_inference(
        image_path=Path(target_image),
        weights_path=Path(args.weights),
        top_k=args.top_k,
        imgsz=args.imgsz,
        device=args.device,
        augment=args.augment,
        save_annotated=args.save,
    )


if __name__ == "__main__":
    main()
