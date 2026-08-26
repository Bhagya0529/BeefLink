"""
app/pages/2_Herd_Verification.py

Herd Verification. Enter a Batch ID (from registration), and
this runs the YOLOv8s (baseline) detector against the herd photo that was uploaded
at registration time, comparing it to the declared count.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from database.schema import init_db, get_connection, insert_herd_verification
from verification.herd import run_herd_verification
from utils.theme import apply_theme, eyebrow, status_badge

init_db()

st.set_page_config(page_title="Herd Verification — BeefLink", layout="wide", page_icon="🐄")
apply_theme()

st.title("Herd Verification")

batch_id = st.text_input("Batch ID", placeholder="e.g. BATCH-A1B2C3D4")

if batch_id:
    conn = get_connection()
    batch = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    conn.close()

    if batch is None:
        st.error("No batch found with that ID. Register a batch first on the Register Supply page.")
    else:
        st.write(f"**Farm:** {batch['farm_id']} — **Declared count:** {batch['declared_count']}")

        image = Image.open(batch["image_path"]).convert("RGB")
        st.image(image, caption="Herd photo from registration", width="stretch")

        if st.button("Run herd verification"):
            with st.spinner("Running YOLOv8s detection..."):
                result = run_herd_verification(image, batch["declared_count"])

            if result is None:
                st.warning("No trained cattle detector found yet at models/cattle_detector_yolov8s_baseline.pt. Train it first (Phase 2).")
            else:
                st.image(result["annotated_image"], caption=f"Detected {result['detected_count']} cattle", width="stretch")

                col1, col2, col3 = st.columns(3)
                col1.metric("Declared", batch["declared_count"])
                col2.metric("Detected", result["detected_count"])
                col3.metric("Difference", result["difference"])

                status_display = {
                    "verified": "Verified",
                    "manual_review": "Review Required",
                    "rejected": "Rejected",
                }
                st.markdown(status_badge(status_display[result["status"]]), unsafe_allow_html=True)

                st.caption(f"Average detection confidence: {result['average_confidence']*100:.1f}%")

                insert_herd_verification(
                    batch_id=batch_id,
                    detected_count=result["detected_count"],
                    avg_conf=result["average_confidence"],
                    max_conf=result["max_confidence"],
                    min_conf=result["min_confidence"],
                    status=result["status"],
                    model_used=result["model_used"],
                )
                st.info("Result saved. Continue to Meat Verification next.")