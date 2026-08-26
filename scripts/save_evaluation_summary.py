"""
scripts/save_evaluation_summary.py

Writes your CBAM-vs-baseline detection comparison and the RADAM freshness
report into a single markdown file under results/, so these numbers exist
as an actual saved artifact rather than only living in terminal scrollback.

USAGE:
    python scripts/save_evaluation_summary.py
"""

from pathlib import Path
from datetime import date

SUMMARY = f"""# BeefLink — Evaluation Summary

Generated: {date.today().isoformat()}

## 1. Cattle Detection — YOLOv8-CBAM vs. YOLOv8s Baseline

Both models trained on the same dataset (WRKusuma Cow Detection, 1,039 images,
single-class), 50 epochs each, identical hyperparameters except architecture.

| Metric | YOLOv8-CBAM | YOLOv8s Baseline |
|---|---|---|
| mAP50 | 0.864 | 0.869 |
| mAP50-95 | 0.535 | 0.555 |
| Precision | 0.893 | 0.881 |
| Recall | 0.771 | 0.791 |
| Validation time (s) | 52.1 | 52.2 |

**Finding:** Adding CBAM attention did not produce a measurable improvement
over the plain YOLOv8s baseline on this dataset — CBAM shows a small
precision gain offset by a small recall decrease, with mAP50 and mAP50-95
both marginally lower than the baseline. Inference/validation time was
effectively identical between the two.

**Discussion points for thesis:**
- The dataset (1,039 images) may be too small for CBAM's added parameters
  to show a clear benefit — attention mechanisms often need more data.
- A single CBAM insertion point (after SPPF) is more minimal than some
  published multi-stage CBAM placements — a stronger effect might appear
  with CBAM inserted after each backbone stage instead.
- Single-class cattle detection may already be a comparatively "easy" task
  with limited headroom for attention-based refinement to improve on.

## 2. Meat Freshness — DCNN+RADAM

Trained on the Mendeley "Meat Species and Hourly Freshness" dataset
(4,500 real market/lab photos, bucketed by elapsed time: 0-12hr -> fresh,
24hr -> half_fresh, 36-48hr -> spoiled). RADAM feature extraction (frozen
pretrained ResNet50 + random projection aggregation) + SVM classifier.

Overall accuracy: **96%** (900 held-out validation images)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| fresh | 0.97 | 0.97 | 0.97 | 360 |
| half_fresh | 0.93 | 0.89 | 0.91 | 180 |
| spoiled | 0.97 | 0.99 | 0.98 | 360 |

Confusion matrix (rows = actual, columns = predicted):
fresh  half_fresh  spoiled
**Finding:** Strong, genuinely trustworthy result — no data leakage risk
(distinct real photos, no augmentation duplicates), confirmed with a
real-world stress-test photo from outside the dataset (previously
misclassified by an earlier MobileNetV3 model trained on a less diverse
dataset; correctly handled by this RADAM model).

**Weakest class:** half_fresh (91% F1) — has half the training data of the
other two classes (900 vs. 1,800 images) and sits at the genuinely
ambiguous boundary between fresh and spoiled. Expected and explainable,
not a flaw.

**Trade-off vs. the earlier MobileNetV3+Grad-CAM approach:** RADAM doesn't
support a single Grad-CAM-style heatmap, since it aggregates features
across multiple layers of a frozen backbone rather than having one
identifiable final conv layer. This is a genuine explainability trade-off
worth naming — RADAM trains faster and needs less data, but loses the
single-image visual explanation the CNN+Grad-CAM version had.

## 3. Verification Engine — Trust Score

Combines both models' outputs into a weighted 0-100 trust score (40 pts
count match, 30 pts detection confidence, 30 pts freshness), compared
against a simpler rule-based binary decision from the reference
architecture. Tested end-to-end through the full dashboard pipeline
(Register -> Herd Verification -> Meat Verification -> Report).

Example result: Batch BATCH-D8C09777 — Trust Score 90/100, Final Decision
"Verified", vs. rule-based decision "Review Required" for the same batch —
demonstrating the graded approach can reach a different (and arguably more
informative) conclusion than a hard rule cutoff.
"""


def main():
    Path("results").mkdir(exist_ok=True)
    out_path = Path("results/evaluation_summary.md")
    out_path.write_text(SUMMARY)
    print(f"Evaluation summary saved to {out_path}")
    print("Edit this file directly if any numbers need updating.")


if __name__ == "__main__":
    main()
