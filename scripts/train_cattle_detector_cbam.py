"""
scripts/train_cattle_detector_cbam.py

Trains the custom YOLOv8s-CBAM architecture (models/yolov8-cbam.yaml) on the
cattle dataset, and — for comparison purposes, since your professor asked for
evaluation across architectures — also gives you an easy way to train a
plain YOLOv8s baseline on the SAME dataset for a fair side-by-side.

USAGE:
    python scripts/train_cattle_detector_cbam.py
    python scripts/train_cattle_detector_cbam.py --baseline   # trains plain YOLOv8s instead, for comparison
"""

import argparse
import sys
from pathlib import Path

# Let this script import utils/cbam.py from the project root.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.cbam import register_cbam

from ultralytics import YOLO

DATA_YAML = "datasets/cattle/data.yaml"
CBAM_ARCHITECTURE = "models/yolov8-cbam.yaml"
CBAM_MODEL_OUT = "models/cattle_detector_cbam.pt"
BASELINE_MODEL_OUT = "models/cattle_detector_yolov8s_baseline.pt"


def train_cbam():
    # MUST happen before YOLO() builds anything from a yaml that references "CBAM".
    register_cbam()

    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"Could not find {DATA_YAML}. Make sure the cattle dataset is in datasets/cattle/.")
        return

    print("Building YOLOv8s-CBAM from custom architecture yaml...")
    model = YOLO(CBAM_ARCHITECTURE)

    # Transfer whatever weights DO match from the official pretrained YOLOv8s.
    # Layers that don't exist in the stock model (our new CBAM block, and the
    # Detect head sized for nc=1 instead of 80 COCO classes) are simply left
    # at their random initialization — everything else starts from pretrained
    # weights rather than from scratch, which trains much faster and better.
    print("Transferring compatible pretrained weights from yolov8s.pt...")
    model.load("yolov8s.pt")

    results = model.train(
        data=str(data_path),
        epochs=50,          # a bit more than before — CBAM adds capacity that benefits from more epochs
        imgsz=640,
        batch=8,
        name="cattle_detector_cbam",
        project="results/runs",
    )

    best_weights = Path("results/runs/cattle_detector_cbam/weights/best.pt")
    if best_weights.exists():
        Path("models").mkdir(exist_ok=True)
        best_weights_copy = Path(CBAM_MODEL_OUT)
        best_weights_copy.write_bytes(best_weights.read_bytes())
        print(f"\nTraining complete. Model saved to {CBAM_MODEL_OUT}")
    else:
        print(f"\nTraining finished but couldn't find best.pt — check results/runs/cattle_detector_cbam/ manually.")


def train_baseline():
    """Plain YOLOv8s on the SAME dataset — gives you an apples-to-apples comparison point."""
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"Could not find {DATA_YAML}. Make sure the cattle dataset is in datasets/cattle/.")
        return

    print("Training baseline YOLOv8s (no CBAM) for comparison...")
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640,
        batch=8,
        name="cattle_detector_yolov8s_baseline",
        project="results/runs",
    )

    best_weights = Path("results/runs/cattle_detector_yolov8s_baseline/weights/best.pt")
    if best_weights.exists():
        Path("models").mkdir(exist_ok=True)
        best_weights_copy = Path(BASELINE_MODEL_OUT)
        best_weights_copy.write_bytes(best_weights.read_bytes())
        print(f"\nTraining complete. Model saved to {BASELINE_MODEL_OUT}")
    else:
        print(f"\nTraining finished but couldn't find best.pt — check results/runs/cattle_detector_yolov8s_baseline/ manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Train plain YOLOv8s instead of YOLOv8-CBAM")
    args = parser.parse_args()

    if args.baseline:
        train_baseline()
    else:
        train_cbam()
