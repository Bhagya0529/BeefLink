"""
scripts/train_freshness_radam.py

Extracts RADAM features for every image in datasets/freshness/, trains an
SVM classifier on top of those features, and reports evaluation metrics.

Unlike training a CNN, this doesn't run epochs — the backbone never changes,
so feature extraction happens once per image, then the SVM trains in
seconds. Expect the FEATURE EXTRACTION step to be the slow part (one forward
pass per image through ResNet50), not the classifier training itself.

USAGE:
    python scripts/train_freshness_radam.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.radam import RadamFeatureExtractor

DATA_DIR = Path("datasets/freshness")
CLASSES = ["fresh", "half_fresh", "spoiled"]

PROJECTIONS_OUT = "models/radam_projections.pt"
SVM_MODEL_OUT = "models/freshness_svm_radam.joblib"


def load_image_paths_and_labels():
    paths, labels = [], []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue
        for img_path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")):
            paths.append(img_path)
            labels.append(class_idx)
    return paths, labels


def main():
    print("Loading image list...")
    paths, labels = load_image_paths_and_labels()
    print(f"Found {len(paths)} images across {len(CLASSES)} classes")

    if len(paths) == 0:
        print("No images found. Check datasets/freshness/ has fresh/half_fresh/spoiled folders with images.")
        return

    # Stratified split so class proportions are preserved in both sets.
    # A plain random split is appropriate here (unlike the old Roboflow
    # dataset) since these are distinct original photos with no augmented
    # near-duplicates to worry about.
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_paths)} images — Validation: {len(val_paths)} images")

    print("\nBuilding RADAM feature extractor (pretrained ResNet50 backbone, untrained random projections)...")
    extractor = RadamFeatureExtractor(seed=42)
    extractor.save(PROJECTIONS_OUT)  # save immediately so training and inference always match
    print(f"Random projections saved to {PROJECTIONS_OUT}")

    print("\nExtracting features for training set (this is the slow part — one forward pass per image)...")
    train_features = []
    for i, path in enumerate(train_paths):
        image = Image.open(path)
        train_features.append(extractor.extract(image))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(train_paths)} done")
    train_features = np.stack(train_features)

    print("\nExtracting features for validation set...")
    val_features = []
    for i, path in enumerate(val_paths):
        image = Image.open(path)
        val_features.append(extractor.extract(image))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(val_paths)} done")
    val_features = np.stack(val_features)

    print("\nTraining SVM classifier on extracted features...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(train_features, train_labels)

    print("\nEvaluating on held-out validation set...")
    predictions = svm.predict(val_features)
    print("\nPer-class report:")
    print(classification_report(val_labels, predictions, target_names=CLASSES))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(CLASSES)
    print(confusion_matrix(val_labels, predictions))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(svm, SVM_MODEL_OUT)
    print(f"\nSVM model saved to {SVM_MODEL_OUT}")
    print(f"Random projections saved to {PROJECTIONS_OUT}")
    print("\nBoth files are needed together for inference — the dashboard will load both.")


if __name__ == "__main__":
    main()
