"""
verification/herd.py

Module 2 backend — Herd Verification.

Wraps the trained cattle detector (plain YOLOv8s baseline) with the
declared-vs-detected comparison logic. The dashboard page calls
run_herd_verification() and gets back everything it needs to display AND
everything trust_score.py needs to score it.
"""

from pathlib import Path

CATTLE_MODEL_PATH = Path("models/cattle_detector_yolov8s_baseline.pt")

_model_cache = {"model": None}


def _load_model():
    """Loads the baseline YOLOv8s model once and reuses it."""
    if _model_cache["model"] is None:
        if not CATTLE_MODEL_PATH.exists():
            return None
        from ultralytics import YOLO
        _model_cache["model"] = YOLO(str(CATTLE_MODEL_PATH))
    return _model_cache["model"]


def run_herd_verification(image, declared_count: int):
    """
    Runs detection on a herd image and compares against the declared count.

    Returns a dict with everything the dashboard and trust score need:
        detected_count, average_confidence, max_confidence, min_confidence,
        status ('verified' / 'manual_review' / 'rejected'),
        annotated_image (numpy array with boxes drawn), difference
    """
    model = _load_model()
    if model is None:
        return None  # caller should show "model not trained yet" message

    results = model(image)
    boxes = results[0].boxes

    detected_count = len(boxes)
    confidences = boxes.conf.tolist() if detected_count > 0 else []

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    max_confidence = max(confidences) if confidences else 0.0
    min_confidence = min(confidences) if confidences else 0.0

    difference = abs(declared_count - detected_count)
    if difference == 0:
        status = "verified"
    elif difference <= 1:
        status = "manual_review"
    else:
        status = "rejected"

    annotated_image = results[0].plot()

    return {
        "detected_count": detected_count,
        "average_confidence": average_confidence,
        "max_confidence": max_confidence,
        "min_confidence": min_confidence,
        "status": status,
        "annotated_image": annotated_image,
        "difference": difference,
        "model_used": "yolov8s_baseline",
    }
