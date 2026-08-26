"""
scripts/test_baseline_on_image.py

Quick standalone test: runs a single image through the baseline YOLOv8s
model (no CBAM) and reports the detected count.

USAGE:
    python scripts/test_baseline_on_image.py path/to/photo.jpg
"""

import sys
from pathlib import Path
from ultralytics import YOLO

BASELINE_MODEL = "models/cattle_detector_yolov8s_baseline.pt"


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_baseline_on_image.py path/to/photo.jpg")
        return

    image_path = sys.argv[1]
    if not Path(BASELINE_MODEL).exists():
        print(f"Baseline model not found at {BASELINE_MODEL}")
        return

    model = YOLO(BASELINE_MODEL)
    results = model(image_path)
    count = len(results[0].boxes)

    print(f"\nBaseline YOLOv8s detected: {count} cattle")

    out_path = f"baseline_prediction_{Path(image_path).stem}.jpg"
    results[0].save(filename=out_path)
    print(f"Annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
