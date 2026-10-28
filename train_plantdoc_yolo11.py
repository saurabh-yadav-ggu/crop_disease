"""
Production-Ready Multi-Crop Disease Classification Pipeline on PlantDoc Dataset
================================================================================
Architecture: Ultralytics YOLO11 (yolo11s-cls.pt)
Target: Real-world in-the-wild agricultural field condition classification

Key Pathology-Specific Optimizations:
1. High input resolution (imgsz=320) to capture minute fungal/bacterial lesions amidst farm clutter.
2. Pathology-safe augmentations: restricted HSV hue jitter (hsv_h=0.005) preserving chlorosis/necrosis signals,
   increased brightness variance (hsv_v=0.30) for outdoor farm sunlight/shadows, and random erasing (0.2)
   to prevent overfitting to background soil, hands, and sky.
3. Stratified Train/Val split (85%/15%) preserving independent holdout test split.
4. Deep image scrubber validating image buffers (Image.verify + Image.load).
5. Cross-platform archive-based extractor to handle Windows NTFS invalid characters (e.g. '?') in web-scraped filenames.
6. Top-3 disease inference and dual edge-export (ONNX & TFLite).
"""

import argparse
import io
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from ultralytics import YOLO

# Allow truncated images to be identified and handled
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PlantDoc-YOLO11")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# 1. Dataset Acquisition, Windows NTFS Sanitization, & Structuring
# =============================================================================

def sanitize_filename(filename: str) -> str:
    """Replaces Windows NTFS invalid characters (< > : \" / \\ | ? *) with an underscore."""
    # Replace query strings or illegal chars
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Strip any trailing dots or spaces which Windows also disallows
    cleaned = cleaned.strip(". ")
    return cleaned if cleaned else "unnamed_image.jpg"


def verify_image_buffer(file_bytes: bytes) -> bool:
    """Verifies that in-memory image bytes represent a valid, uncorrupted, decodable raster image."""
    if not file_bytes or len(file_bytes) == 0:
        return False
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.load()
        return True
    except Exception:
        return False


def extract_and_structure_from_tar(
    tar_path: Path,
    output_dir: Path,
    train_val_split_ratio: float = 0.85,
    seed: int = 42,
) -> Path:
    """
    Extracts directly from a Git archive tarball into an Ultralytics YOLO-compliant structure:
        output_dir/
            ├── train/ (85% of train)
            ├── val/   (15% of train)
            └── test/  (100% of test - holdout set)

    Sanitizes invalid Windows characters and cleans corrupted files on-the-fly.
    """
    random.seed(seed)
    prep_train = output_dir / "train"
    prep_val = output_dir / "val"
    prep_test = output_dir / "test"

    logger.info(f"Extracting and sanitizing dataset from archive '{tar_path}'...")
    corrupted_count = 0
    sanitized_count = 0

    # Group train images by class in memory first for stratified splitting
    train_data: Dict[str, List[Tuple[str, bytes]]] = {}
    test_data: Dict[str, List[Tuple[str, bytes]]] = {}

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            parts = Path(member.name).parts
            if len(parts) < 3:
                continue

            split_name = parts[0].lower()  # 'train' or 'test'
            if split_name not in ("train", "test"):
                continue

            class_name = parts[1]
            raw_filename = parts[2]
            clean_filename = sanitize_filename(raw_filename)

            if clean_filename != raw_filename:
                sanitized_count += 1

            # Check extension
            ext = Path(clean_filename).suffix.lower()
            if ext not in VALID_EXTENSIONS:
                continue

            # Read file stream
            f = tar.extractfile(member)
            if f is None:
                continue
            file_bytes = f.read()

            if not verify_image_buffer(file_bytes):
                corrupted_count += 1
                continue

            if split_name == "train":
                train_data.setdefault(class_name, []).append((clean_filename, file_bytes))
            else:
                test_data.setdefault(class_name, []).append((clean_filename, file_bytes))

    logger.info(
        f"Archive scanned: {len(train_data)} train classes, {len(test_data)} test classes. "
        f"Sanitized {sanitized_count} Windows-incompatible filenames. Filtered out {corrupted_count} corrupted images."
    )

    # 1. Stratified Train / Val partitioning
    for class_name, items in train_data.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_val_split_ratio)
        train_items = items[:split_idx]
        val_items = items[split_idx:]

        cls_train_dir = prep_train / class_name
        cls_val_dir = prep_val / class_name
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        for filename, data in train_items:
            (cls_train_dir / filename).write_bytes(data)

        for filename, data in val_items:
            (cls_val_dir / filename).write_bytes(data)

    # 2. Holdout Test Set
    for class_name, items in test_data.items():
        cls_test_dir = prep_test / class_name
        cls_test_dir.mkdir(parents=True, exist_ok=True)
        for filename, data in items:
            (cls_test_dir / filename).write_bytes(data)

    train_total = sum(len(list((prep_train / c).glob("*"))) for c in prep_train.iterdir() if c.is_dir())
    val_total = sum(len(list((prep_val / c).glob("*"))) for c in prep_val.iterdir() if c.is_dir())
    test_total = sum(len(list((prep_test / c).glob("*"))) for c in prep_test.iterdir() if c.is_dir())

    logger.info(
        f"Dataset partitioning complete: Train={train_total}, Val={val_total}, Test={test_total} "
        f"across {len(train_data)} categories."
    )
    return output_dir


def prepare_plantdoc_pipeline(
    raw_dir: Path,
    output_dir: Path,
    train_val_split_ratio: float = 0.85,
    seed: int = 42,
) -> Path:
    """Ensures PlantDoc dataset is cloned, extracted, sanitized for Windows NTFS, and structured."""
    prep_train = output_dir / "train"
    prep_val = output_dir / "val"
    prep_test = output_dir / "test"

    if (
        prep_train.exists()
        and prep_val.exists()
        and prep_test.exists()
        and any(prep_train.iterdir())
        and any(prep_val.iterdir())
        and any(prep_test.iterdir())
    ):
        logger.info(f"Structured dataset already prepared at '{output_dir.resolve()}'. Skipping generation.")
        return output_dir

    tar_path = Path("plantdoc.tar")
    if not tar_path.exists():
        # Check if PlantDoc-Dataset/.git exists
        git_dir = raw_dir / ".git"
        if git_dir.exists():
            logger.info("Generating plantdoc.tar directly from local Git tree...")
            subprocess.run(
                ["git", "-c", "core.protectNTFS=false", "archive", "-o", str(tar_path.resolve()), "HEAD"],
                cwd=str(raw_dir),
                check=True,
            )
        else:
            logger.info("Cloning repository tree without checkout to bypass NTFS filename limits...")
            subprocess.run(
                ["git", "clone", "-n", "--depth", "1", "https://github.com/Pratikkayal/PlantDoc-Dataset.git", str(raw_dir)],
                check=True,
            )
            subprocess.run(
                ["git", "-c", "core.protectNTFS=false", "archive", "-o", str(tar_path.resolve()), "HEAD"],
                cwd=str(raw_dir),
                check=True,
            )

    return extract_and_structure_from_tar(
        tar_path=tar_path,
        output_dir=output_dir,
        train_val_split_ratio=train_val_split_ratio,
        seed=seed,
    )


# =============================================================================
# 2. Real-World Field Tuned YOLO11-cls Training
# =============================================================================

def train_plantdoc_model(
    data_dir: Path,
    weights: str = "yolo11m-cls.pt",
    epochs: int = 75,
    imgsz: int = 448,
    batch: int = 16,
    device: str = "0",
    project: str = "crop_disease_yolo11",
    name: str = "yolo11m_plantdoc_field",
    lr0: float = 3e-4,
    freeze: Optional[int] = None,
) -> Path:
    """
    Trains Ultralytics YOLO11m-cls with Precision Ag hyperparameter profile:
    - imgsz=448: High resolution for resolving subtle lesions amidst farm clutter.
    - yolo11m-cls: Medium model with ~10M parameters for higher multi-crop capacity.
    - AdamW + lr0=3e-4 + cos_lr: Smooth cosine annealing convergence.
    - dropout=0.3 + label_smoothing=0.1: Counteracts memorization of farm backgrounds.
    - mixup=0.12: Image blending to prevent background overfitting.
    - Pathology-Safe Augmentations:
        - auto_augment=None: Inhibits random color inversions destroying chlorosis signals.
        - hsv_h=0.008: Preserves diagnostic chlorophyll green vs yellow/brown necrosis.
        - hsv_s=0.20 & hsv_v=0.35: Simulates harsh direct sunlight and deep cloud/canopy shadows.
        - scale=0.4 & erasing=0.25: Enforces invariant representation across camera distance & occlusions.
    """
    # Verify GPU availability
    if device in ("0", 0, "cuda", "cuda:0"):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Target Hardware Acceleration: GPU [CUDA:0] -> {gpu_name} ({vram_gb:.2f} GB VRAM)")
            target_device = "0"
        else:
            logger.warning(
                "Requested GPU training ('--device 0'), but torch.cuda.is_available() is False. "
                "Falling back to CPU."
            )
            target_device = "cpu"
    else:
        target_device = device

    logger.info(f"Initializing YOLO11 classification model from checkpoint: {weights}...")
    model = YOLO(weights)

    train_kwargs = dict(
        data=str(data_dir.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=target_device,
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.01,
        cos_lr=True,
        patience=15,
        dropout=0.3,
        label_smoothing=0.1,
        weight_decay=0.0005,
        warmup_epochs=5.0,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    logger.info(f"Commencing training ({epochs} epochs, imgsz={imgsz}, batch={batch}, lr0={lr0}, device={target_device})...")
    train_results = model.train(
        **train_kwargs,

        # Plant-Pathology Safe Augmentations
        auto_augment=None,      # Disables RandAugment color distortion
        hsv_h=0.008,            # Minimal hue jitter: preserves diagnostic chlorosis vs tissue
        hsv_s=0.20,             # Mild saturation variations
        hsv_v=0.35,             # Substantial brightness shifts for harsh sun & canopy shadows
        degrees=45.0,           # Handheld mobile angle variation
        fliplr=0.5,             # Horizontal leaf flip
        flipud=0.5,             # Vertical leaf flip
        scale=0.4,              # Zoom variance for near/far foliage
        shear=0.0,
        perspective=0.0,
        erasing=0.25,           # Random patch erasure forces focus away from background soil/hands
        mixup=0.12,             # Blend images to force model to learn lesion micro-features

        # Outputs
        project=project,
        name=name,
        pretrained=True,
        plots=True,
        save=True,
        exist_ok=True,
        verbose=True,
    )

    best_checkpoint = Path(project) / name / "weights" / "best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = Path(train_results.save_dir) / "weights" / "best.pt"

    logger.info(f"Training completed. Best weights saved to: {best_checkpoint.resolve()}")
    return best_checkpoint


# =============================================================================
# 3. Evaluation on Holdout Test Split
# =============================================================================

def evaluate_on_test_split(
    checkpoint_path: Path,
    data_dir: Path,
    imgsz: int = 448,
    device: str = "0",
) -> Dict[str, float]:
    """Evaluates the trained model on the holdout 'test' split and logs Top-1 & Top-5 accuracy."""
    target_device = "0" if (device in ("0", 0, "cuda") and torch.cuda.is_available()) else "cpu"
    logger.info(f"Evaluating {checkpoint_path.name} on holdout test partition (device={target_device})...")
    model = YOLO(str(checkpoint_path))

    metrics = model.val(
        data=str(data_dir.resolve()),
        split="test",
        imgsz=imgsz,
        device=target_device,
        plots=True,
    )

    top1 = float(metrics.top1)
    top5 = float(metrics.top5)

    print("\n" + "=" * 55)
    print("       PLANTDOC HOLDOUT TEST EVALUATION RESULTS      ")
    print("=" * 55)
    print(f"  Top-1 Accuracy: {top1 * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5 * 100:.2f}%")
    print("=" * 55 + "\n")

    return {"top1": top1, "top5": top5}


# =============================================================================
# 4. Field Image Top-3 Disease Inference
# =============================================================================

def predict_crop_disease(
    model_or_path: Path,
    image_path: Path,
    top_k: int = 3,
    imgsz: int = 320,
) -> List[Tuple[str, float]]:
    """Performs inference on a single field photograph and returns top-k predictions."""
    model = YOLO(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path

    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return []

    results = model.predict(source=str(image_path), imgsz=imgsz, verbose=False)
    result = results[0]

    probs = result.probs
    top_indices = probs.top5[:top_k]

    predictions: List[Tuple[str, float]] = []
    print("\n" + "-" * 58)
    print(f" DIAGNOSIS FOR: {Path(image_path).name}")
    print("-" * 58)

    for rank, idx in enumerate(top_indices, 1):
        class_name = model.names[idx]
        confidence = float(probs.data[idx]) * 100.0
        predictions.append((class_name, confidence))
        print(f"  Rank {rank}: {class_name:<36} | {confidence:6.2f}%")

    print("-" * 58 + "\n")
    return predictions


# =============================================================================
# 5. Production Edge Export (ONNX & TFLite)
# =============================================================================

def export_edge_models(
    checkpoint_path: Path,
    imgsz: int = 320,
    formats: Tuple[str, ...] = ("onnx", "tflite"),
) -> None:
    """Exports model to target runtime formats for mobile/edge IoT farm devices."""
    model = YOLO(str(checkpoint_path))

    for fmt in formats:
        logger.info(f"Exporting model to {fmt.upper()} (imgsz={imgsz})...")
        try:
            exported_path = model.export(format=fmt, imgsz=imgsz)
            logger.info(f"Successfully exported {fmt.upper()} model to: {exported_path}")
        except Exception as e:
            logger.warning(
                f"Export to {fmt.upper()} skipped or failed: {e}. "
                f"(To enable, ensure export dependencies are installed)."
            )


# =============================================================================
# CLI Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Production PlantDoc Multi-Crop Disease Classification using YOLO11s-cls"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="PlantDoc-Dataset",
        help="Path to cloned PlantDoc-Dataset repository",
    )
    parser.add_argument(
        "--prepared-data-dir",
        type=str,
        default="dataset_prepared",
        help="Directory where structured train/val/test splits will be created",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11m-cls.pt",
        help="Pretrained YOLO11 checkpoint (default: yolo11m-cls.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        help="Number of training epochs (default: 75)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=448,
        help="Input image resolution (default: 448 for lesion micro-features)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16; optimized for 6GB VRAM at 448px)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Target acceleration: '0' for NVIDIA GPU, or 'cpu'",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.85,
        help="Train/val split ratio from raw train set (default: 0.85)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="crop_disease_yolo11",
        help="Project output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11m_plantdoc_field",
        help="Run identifier",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=3e-4,
        help="Initial learning rate (default: 3e-4 with cosine decay)",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Number of layers to freeze for transfer learning (e.g. 10)",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Path to an optional field test image for immediate disease diagnosis",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX and TFLite edge model exports",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir = Path(args.raw_data_dir)
    prepared_dir = Path(args.prepared_data_dir)

    # Step 1 & 2: Process PlantDoc Dataset (sanitizing NTFS names & building 85/15 train/val + test)
    prep_data_path = prepare_plantdoc_pipeline(
        raw_dir=raw_dir,
        output_dir=prepared_dir,
        train_val_split_ratio=args.split_ratio,
    )

    # Step 3: Train YOLO11s-cls with Precision Ag hyperparameter profile on GPU
    best_weights = train_plantdoc_model(
        data_dir=prep_data_path,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        freeze=args.freeze,
    )

    # Step 4: Validate on Holdout Test Split
    evaluate_on_test_split(
        checkpoint_path=best_weights,
        data_dir=prep_data_path,
        imgsz=args.imgsz,
        device=args.device,
    )

    # Step 5: Run sample inference if requested (or find an image from test set)
    sample_img = args.test_image
    if not sample_img:
        test_images = list((prep_data_path / "test").glob("*/*.[jJ][pP]*[gG]"))
        if test_images:
            sample_img = str(test_images[0])

    if sample_img:
        predict_crop_disease(best_weights, Path(sample_img), top_k=3, imgsz=args.imgsz)

    # Step 6: Export for Edge Deployment
    if not args.skip_export:
        export_edge_models(best_weights, imgsz=args.imgsz, formats=("onnx", "tflite"))

    logger.info("Pipeline executed successfully!")


if __name__ == "__main__":
    main()
