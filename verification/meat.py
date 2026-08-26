"""
verification/meat.py

Module 3 backend — Meat Freshness Verification.

Wraps the trained RADAM feature extractor + SVM classifier. Note this model
outputs class PROBABILITIES (via SVC(probability=True)) rather than a Grad-CAM
heatmap — RADAM doesn't have a single "last conv layer" the way a standard
CNN classifier does, since it aggregates across MULTIPLE layers of a frozen
backbone, so Grad-CAM in its usual form doesn't directly apply here. This is
worth naming explicitly in your thesis as a trade-off of the RADAM approach:
you gain training efficiency and don't need to fine-tune a large model, but
you lose the single-heatmap explainability your MobileNetV3+Grad-CAM version
had. If explainability matters more than efficiency for your final choice of
architecture, that's a legitimate reason to prefer the CNN-based version —
this is exactly the kind of trade-off your evaluation should surface.
"""

import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.radam import RadamFeatureExtractor

PROJECTIONS_PATH = Path("models/radam_projections.pt")
SVM_MODEL_PATH = Path("models/freshness_svm_radam.joblib")
CLASSES = ["fresh", "half_fresh", "spoiled"]

_cache = {"extractor": None, "svm": None}


def _load_models():
    if _cache["extractor"] is None or _cache["svm"] is None:
        if not PROJECTIONS_PATH.exists() or not SVM_MODEL_PATH.exists():
            return None, None
        _cache["extractor"] = RadamFeatureExtractor.load(str(PROJECTIONS_PATH))
        _cache["svm"] = joblib.load(SVM_MODEL_PATH)
    return _cache["extractor"], _cache["svm"]


def run_meat_verification(image):
    """
    Runs RADAM feature extraction + SVM classification on a meat image.

    Returns a dict with everything the dashboard and trust score need:
        label, confidence, prob_fresh, prob_half_fresh, prob_spoiled
    """
    extractor, svm = _load_models()
    if extractor is None or svm is None:
        return None  # caller should show "model not trained yet" message

    features = extractor.extract(image).reshape(1, -1)
    probabilities = svm.predict_proba(features)[0]

    prob_dict = dict(zip(CLASSES, probabilities))
    predicted_label = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[predicted_label]

    return {
        "label": predicted_label,
        "confidence": float(confidence),
        "prob_fresh": float(prob_dict["fresh"]),
        "prob_half_fresh": float(prob_dict["half_fresh"]),
        "prob_spoiled": float(prob_dict["spoiled"]),
        "model_used": "dcnn_radam",
    }
