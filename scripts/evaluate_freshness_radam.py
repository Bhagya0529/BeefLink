"""
scripts/evaluate_freshness_radam.py

Produces a confusion matrix and per-class precision/recall/F1 report for
the RADAM+SVM freshness classifier, using the same train/val split logic
as training (same random_state) so results are reproducible.

USAGE:
    python scripts/evaluate_freshness_radam.py
"""

import sys
from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.radam import RadamFeatureExtractor

DATA_DIR = Path("datasets/freshness")
CLASSES = ["fresh", "half_fresh", "spoiled"]
PROJECTIONS_PATH = "models/radam_projections.pt"
SVM_MODEL_PATH = "models/freshness_svm_radam.joblib"
OUT_IMAGE = "results/freshness_radam_confusion_matrix.png"


def load_image_paths_and_labels():
    paths, labels = [], []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            continue
        for img_path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")):
            paths.append(img_path)
            labels.append(class_idx)
    return paths, labels


def main():
    if not Path(SVM_MODEL_PATH).exists() or not Path(PROJECTIONS_PATH).exists():
        print("RADAM model not found. Train it first with scripts/train_freshness_radam.py")
        return

    paths, labels = load_image_paths_and_labels()

    _, val_paths, _, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Evaluating on {len(val_paths)} held-out validation images")

    extractor = RadamFeatureExtractor.load(PROJECTIONS_PATH)
    svm = joblib.load(SVM_MODEL_PATH)

    val_features = []
    for i, path in enumerate(val_paths):
        image = Image.open(path)
        val_features.append(extractor.extract(image))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(val_paths)} done")
    val_features = np.stack(val_features)

    predictions = svm.predict(val_features)

    print("\nPer-class report:")
    print(classification_report(val_labels, predictions, target_names=CLASSES))

    cm = confusion_matrix(val_labels, predictions)

    Path("results").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("DCNN+RADAM Freshness Classifier — Confusion Matrix")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_IMAGE, dpi=150)
    print(f"\nConfusion matrix saved to {OUT_IMAGE}")


if __name__ == "__main__":
    main()
