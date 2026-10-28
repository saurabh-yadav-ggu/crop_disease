# Multi-Crop Disease Classification System using Ultralytics YOLO11

A deep learning precision agriculture system engineered to identify and diagnose crop diseases under real-world, in-the-wild agricultural field conditions using **Ultralytics YOLO11s-cls**.

---

## 📌 Project Overview

Traditional crop disease classification models are often trained on clean, laboratory-photographed datasets (e.g. PlantVillage) where single leaves are placed against plain backgrounds. These models severely degrade in real-world farms due to:
- Complex field backgrounds (soil, weeds, shadows, sky, farmer hands).
- Variable lighting (harsh solar glare, cloud shadows).
- Minute fungal/bacterial pustules occupying only 5–15% of the frame.

This project addresses these challenges by training a **YOLO11s-cls** architecture on the **PlantDoc Dataset** using domain-specific pathology safeguards.

---

## 📊 Benchmark Results

Trained on an **NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)** using **YOLO11m-cls (448×448 px)**:

| Metric | Score | Note | Improvement |
| :--- | :---: | :--- | :---: |
| **Holdout Test Top-1 Accuracy** | **74.15%** | Evaluated on 236 unseen real-world field images | **+7.20%** 🚀 |
| **Holdout Test Top-5 Accuracy** | **94.92%** | True condition among top-5 predictions | **+0.43%** |
| **Validation Top-1 Accuracy** | **76.37%** | In-training stratified validation | **+2.17%** |
| **Validation Top-5 Accuracy** | **96.15%** | Top-5 accuracy across validation split | **+0.55%** |
| **Inference Latency** | **24.5 ms** | Real-time GPU execution at 448px resolution | Ready for field edge use |

---

## 📂 Project Structure

```text
crop_disease/
│
├── dataset_prepared/         # Structured, verified dataset splits (28 classes)
│   ├── train/                # 1,973 images (Stratified 85%)
│   ├── val/                  # 364 images (Stratified 15%)
│   └── test/                 # 236 images (Holdout test set)
│
├── models/                   # Production-ready model checkpoints
│   ├── best.pt               # Trained PyTorch YOLO11s model
│   └── best.onnx             # Slimmed ONNX export (20.9 MB)
│
├── sample_images/            # Example field images for instant testing
│   ├── sample_1.png
│   ├── sample_2.png
│   └── sample_potato_blight.png
│
├── runs/                     # Detailed training logs, plots, & confusion matrices
│   └── classify/crop_disease_yolo11/yolo11s_plantdoc_field/
│       ├── confusion_matrix.png
│       └── results.png
│
├── inference.py              # CLI & programmatic inference tool
├── train_plantdoc_yolo11.py  # Production training pipeline script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation

Activate your virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

*(Ensure PyTorch with CUDA is installed if running on an NVIDIA GPU)*.

### 2. Run Disease Diagnosis (Inference)

To diagnose any crop leaf image:

```bash
# Direct image prediction
python inference.py sample_images/sample_potato_blight.png

# Predict and save annotated overlay to predictions/ folder
python inference.py sample_images/sample_potato_blight.png --save

# Interactive mode (prompts you for an image path)
python inference.py
```

**Example Output:**
```text
=================================================================
               CROP DISEASE DIAGNOSIS REPORT
=================================================================
 Image Source : sample_images/sample_potato_blight.png
 Model Engine : YOLO11 (best.pt)
 Hardware     : GPU (CUDA:0)
-----------------------------------------------------------------
 PRIMARY DIAGNOSIS : [ DISEASED ] Potato leaf late blight
 Crop Type         : Potato
 Pathology Status  : Leaf Late Blight
 Confidence Score  : 65.49%

 Top-3 Differential Diagnosis Ranking:
-----------------------------------------------------------------
  Rank  | Category Label                      | Confidence
-----------------------------------------------------------------
  #1    | Potato leaf late blight             |  65.49%
  #2    | Potato leaf early blight            |  28.56%
  #3    | Tomato Early blight leaf            |   3.25%
=================================================================
```

### 3. Re-training the Model

To train or fine-tune the pipeline:

```bash
python train_plantdoc_yolo11.py --weights yolo11m-cls.pt --imgsz 448 --epochs 75 --batch 16 --device 0
```

---

## 🌿 Supported Disease Categories (28 Classes)

The system covers 28 disease and healthy categories across 13 major agricultural crops:
- **Apple**: Apple Scab, Apple Rust, Healthy Leaf
- **Bell Pepper**: Bacterial Spot, Healthy Leaf
- **Blueberry**: Healthy Leaf
- **Cherry**: Healthy Leaf
- **Corn (Maize)**: Gray Leaf Spot, Leaf Blight, Common Rust
- **Grape**: Black Rot, Healthy Leaf
- **Peach**: Healthy Leaf
- **Potato**: Early Blight, Late Blight
- **Raspberry**: Healthy Leaf
- **Soybean**: Healthy Leaf
- **Squash**: Powdery Mildew
- **Strawberry**: Healthy Leaf
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mites, Mosaic Virus, Yellow Leaf Curl Virus, Healthy Leaf
