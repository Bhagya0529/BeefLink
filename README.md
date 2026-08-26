# BeefLink
A computer vision verification layer for a farmer-owned organic beef coordination network. BeefLink independently verifies two claims a farmer makes when registering a supply listing — declared herd count and meat freshness — and combines both into a single, graded trust score, without requiring a prior relationship between the farmer and buyer.

Built as an individual MEng technical prototype (Computer Vision & Artificial Intelligence, University of Limerick), developed from a Digital Futures Lab group project, Mid-West Beef Link.

What it does
Module	What it checks	How
Herd Verification	Does the declared cattle count match the photo?	YOLOv8s object detection
Meat Verification	Is the meat fresh, half-fresh, or spoiled?	RADAM feature extraction + SVM classifier
Verification Engine	Combines both checks	Weighted trust score (0–100) + a separate rule-based decision for comparison

Every verification produces a trust score (Verified / Review Required / Rejected) and a downloadable PDF report, shown to the buyer before a transaction proceeds.

Project structure
BeefLink/
├── app/
│   ├── dashboard.py              # Main entry point
│   └── pages/
│       ├── 1_Register.py         # Supply registration
│       ├── 2_Herd_Verification.py
│       ├── 3_Meat_Verification.py
│       └── 4_Report.py           # Combined trust score + PDF/share export
├── database/
│   └── schema.py                 # SQLite schema (batches, herd_verification,
│                                  # meat_verification, reports)
├── datasets/
│   ├── cattle/                   # WRKusuma "Cow Detection" (Roboflow)
│   └── freshness/                # Mendeley meat freshness dataset,
│       ├── fresh/                # bucketed by post-slaughter time
│       ├── half_fresh/
│       └── spoiled/
├── models/
│   ├── cattle_detector_yolov8s_baseline.pt   # production model
│   ├── cattle_detector_cbam.pt               # comparison model
│   ├── yolov8-cbam.yaml                      # custom CBAM architecture
│   ├── radam_projections.pt                  # RADAM feature projections
│   └── freshness_svm_radam.joblib            # trained SVM classifier
├── results/
│   ├── evaluation_summary.md
│   └── freshness_radam_confusion_matrix.png
├── scripts/                      # Training & evaluation scripts
├── utils/
│   ├── cbam.py                   # CBAM attention module
│   ├── radam.py                  # RADAM feature extractor
│   └── theme.py                  # Dashboard visual theme
├── verification/
│   ├── herd.py                   # Herd verification logic
│   ├── meat.py                   # Meat verification logic
│   ├── trust_score.py            # Trust score & rule-based decision
│   ├── verification_engine.py    # Combines both checks
│   └── report_export.py          # PDF report + share options
└── .streamlit/
    └── config.toml               # Theme colors
Setup
Requirements
Python 3.10+
~2 GB free disk space for models and datasets
No GPU required — everything runs on CPU (training was done on a laptop CPU; inference is fast enough for interactive use)
1. Clone and create a virtual environment
bash
git clone <your-repo-url>
cd BeefLink
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

Note: if you ever rename or move this folder after creating the venv, delete and recreate venv/ — the activation scripts hard-code the original path and will break silently otherwise.

2. Install dependencies
bash
pip install --upgrade pip
pip install streamlit torch torchvision ultralytics pillow numpy scikit-learn matplotlib joblib reportlab
3. Run the dashboard
bash
streamlit run app/dashboard.py

Open the URL Streamlit prints (typically http://localhost:8501).

Using the dashboard
Register Supply — enter a farm ID and declared herd count, upload a herd photo (and, separately, a meat photo when available)
Herd Verification — enter the Batch ID, run detection; compares YOLOv8s's detected count against the declared count
Meat Verification — run the RADAM + SVM classifier on the uploaded meat photo
Report — view the combined trust score, breakdown by component, and final decision (Verified / Review Required / Rejected); download as PDF or share via email/copy
How the trust score works

The trust score is out of 100, split across three components:

Component	Points	Rule
Count match	40	Full points for an exact match, scaling to 0 at a difference of 3+ animals
Detection confidence	30	Scales directly with the detector's average confidence
Freshness	30	"Spoiled" always scores 0 regardless of confidence; "fresh"/"half_fresh" scale by confidence

Final decision: ≥85 → Verified · ≥60 → Review Required · <60 → Rejected

These thresholds are an initial specification, not tuned against real-world labelled outcomes — see Limitations.

Model details
Cattle detection — YOLOv8s baseline vs. YOLOv8-CBAM

Both architectures were trained on identical data (1,039 images, WRKusuma Cow Detection dataset) for a controlled comparison. The plain YOLOv8s baseline is the model used in the dashboard — it matched or slightly outperformed the CBAM-augmented version in formal evaluation (mAP50 0.869 vs. 0.864), so the simpler model was kept as the production one. The CBAM comparison itself is reported as an evaluation result, not discarded.

Meat freshness — RADAM + SVM

Extracts features from a frozen, pre-trained ResNet50 backbone at multiple depths (no fine-tuning), aggregates them into a fixed-length descriptor, and classifies with an RBF-kernel SVM. Trained on the Mendeley "Meat Species and Hourly Freshness" dataset (4,500 images), bucketed by post-slaughter time into fresh / half-fresh / spoiled. Achieves 96% accuracy on a held-out, leakage-free validation set. Note: because RADAM aggregates across multiple backbone layers rather than relying on a single final convolutional layer, it does not support Grad-CAM-style visual explanations.

Retraining the models
bash
# Cattle detector (baseline)
python scripts/train_cattle_detector_cbam.py --baseline

# Cattle detector (CBAM)
python scripts/train_cattle_detector_cbam.py

# Meat freshness classifier
python scripts/train_freshness_radam.py

# Evaluate freshness classifier (produces confusion matrix + report)
python scripts/evaluate_freshness_radam.py

Training the cattle detectors takes roughly 12–13 hours per model on CPU (50 epochs, 640×640, batch size 8).

Limitations & future work
Trust score weights (40/30/30) and decision thresholds (85/60) are an initial specification, not validated against real labelled transaction outcomes
Cattle detection dataset (1,039 images) is modest in size; the CBAM-vs-baseline comparison would benefit from repeat runs across multiple seeds and a larger dataset
RADAM's simplified feature aggregation (fixed random projection) trades some representational richness for CPU-friendly training speed, versus the original paper's trained-autoencoder approach
No individual animal identification — the system verifies internal consistency of a listing, not cross-referencing against national livestock records (e.g. Ireland's AIM system)
Single-machine prototype (SQLite, no authentication/role separation) — not a multi-user production deployment
Datasets
Cattle detection: WRKusuma "Cow Detection", Roboflow Universe
Meat freshness: "A Comprehensive Image Dataset for Accurate Authentication of Meat Species and Hourly Freshness Using Artificial Intelligence in Food Safety", Mendeley Data
Acknowledgements

Developed from the Mid-West Beef Link concept, a Digital Futures Lab group project at the University of Limerick. Supervised by Dr. Arash Joorabchi.
