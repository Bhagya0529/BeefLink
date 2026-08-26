"""
app/pages/3_Meat_Verification.py

Meat Freshness Verification. Enter a Batch ID, upload a
photo, and this runs the RADAM+SVM classifier.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from database.schema import init_db, get_connection, insert_meat_verification
from verification.meat import run_meat_verification
from utils.theme import apply_theme, eyebrow

init_db()

st.set_page_config(page_title="Meat Verification — BeefLink", layout="wide", page_icon="🐄")
apply_theme()

st.title("Meat Freshness Verification")

batch_id = st.text_input("Batch ID", placeholder="e.g. BATCH-A1B2C3D4")

if batch_id:
    conn = get_connection()
    batch = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    conn.close()

    if batch is None:
        st.error("No batch found with that ID. Register a batch first on the Register Supply page.")
    else:
        st.write(f"**Farm:** {batch['farm_id']} — **Batch:** {batch_id}")

        carcass_image = st.file_uploader("Upload meat photo", type=["jpg", "jpeg", "png"])

        if carcass_image is not None:
            image = Image.open(carcass_image).convert("RGB")
            st.image(image, caption="Uploaded photo", width="stretch")

            if st.button("Run freshness verification"):
                with st.spinner("Extracting RADAM features and classifying..."):
                    result = run_meat_verification(image)

                if result is None:
                    st.warning("No trained freshness model found yet. Train it first (Phase 3): "
                               "models/radam_projections.pt and models/freshness_svm_radam.joblib are both required.")
                else:
                    st.subheader(f"Prediction: {result['label']} ({result['confidence']*100:.1f}% confidence)")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fresh", f"{result['prob_fresh']*100:.1f}%")
                    col2.metric("Half Fresh", f"{result['prob_half_fresh']*100:.1f}%")
                    col3.metric("Spoiled", f"{result['prob_spoiled']*100:.1f}%")

    

                    insert_meat_verification(
                        batch_id=batch_id,
                        label=result["label"],
                        confidence=result["confidence"],
                        prob_fresh=result["prob_fresh"],
                        prob_half_fresh=result["prob_half_fresh"],
                        prob_spoiled=result["prob_spoiled"],
                        model_used=result["model_used"],
                    )
                    st.info("Result saved. Continue to Verification Report next.")