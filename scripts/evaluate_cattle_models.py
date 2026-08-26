"""
scripts/evaluate_cattle_models.py

Runs Ultralytics' built-in validation on the trained cattle detector(s) and
prints standard object-detection metrics. If both the CBAM model AND a
baseline YOLOv8s model exist, prints them side by side for direct
comparison — exactly the architecture comparison your professor asked for.

USAGE:
    python scripts/evaluate_cattle_models.py
"""

from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.cbam import register_cbam

DATA_YAML = "datasets/cattle/data.yaml"
CBAM_MODEL = "models/cattle_detector_cbam.pt"
BASELINE_MODEL = "models/cattle_detector_yolov8s_baseline.pt"


def evaluate_model(model_path: str, name: str, is_cbam: bool):
    from ultralytics import YOLO

    if is_cbam:
        register_cbam()

    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
    model = YOLO(model_path)

    metrics = model.val(data=DATA_YAML, project="results/eval_runs", name=name.replace(" ", "_"))

    start = time.time()
    model.val(data=DATA_YAML, project="results/eval_runs", name=f"{name.replace(' ', '_')}_timing", verbose=False)
    elapsed = time.time() - start

    print(f"\n{name} results:")
    print(f"  mAP50:          {metrics.box.map50:.3f}")
    print(f"  mAP50-95:       {metrics.box.map:.3f}")
    print(f"  Precision:      {metrics.box.mp:.3f}")
    print(f"  Recall:         {metrics.box.mr:.3f}")
    print(f"  Validation time: {elapsed:.1f}s (rough proxy for inference cost)")

    return {
        "name": name,
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "val_time": elapsed,
    }


def main():
    results = []

    if Path(CBAM_MODEL).exists():
        results.append(evaluate_model(CBAM_MODEL, "YOLOv8-CBAM", is_cbam=True))
    else:
        print(f"CBAM model not found at {CBAM_MODEL} — train it first.")

    if Path(BASELINE_MODEL).exists():
        results.append(evaluate_model(BASELINE_MODEL, "YOLOv8s Baseline", is_cbam=False))
    else:
        print(f"\nBaseline model not found at {BASELINE_MODEL}.")
        print("Train it for comparison with: python scripts/train_cattle_detector_cbam.py --baseline")

    if len(results) == 2:
        print(f"\n{'='*60}\nSIDE-BY-SIDE COMPARISON\n{'='*60}")
        print(f"{'Metric':<20}{'YOLOv8-CBAM':<15}{'YOLOv8s Baseline':<15}")
        for key, label in [("map50", "mAP50"), ("map50_95", "mAP50-95"),
                             ("precision", "Precision"), ("recall", "Recall")]:
            print(f"{label:<20}{results[0][key]:<15.3f}{results[1][key]:<15.3f}")
        print(f"{'Val time (s)':<20}{results[0]['val_time']:<15.1f}{results[1]['val_time']:<15.1f}")


if __name__ == "__main__":
    main()
