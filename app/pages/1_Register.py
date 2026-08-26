"""
app/pages/1_Register.py

Supply Registration.

The farmer enters batch details and uploads a herd image. No AI runs here —
this just captures the declaration that later modules will verify against.

"""

import sys
from pathlib import Path
from datetime import date
import uuid

import streamlit as st
from PIL import Image

# Let this page import from database/ and utils/ at the project root.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from database.schema import init_db, insert_batch
from utils.theme import apply_theme, eyebrow

init_db()

st.set_page_config(page_title="Register Supply — BeefLink", layout="wide", page_icon="🐄")
apply_theme()

st.title("Register Supply")
st.caption("Enter batch details and upload a herd photo.")

with st.form("register_form"):
    col1, col2 = st.columns(2)
    with col1:
        farm_id = st.text_input("Farm ID", placeholder="e.g. FARM-CLARE-01")
        batch_date = st.date_input("Date", value=date.today(), format="DD/MM/YYYY")
    with col2:
        declared_count = st.number_input("Number of cattle declared", min_value=1, step=1)

    herd_image = st.file_uploader("Upload herd image", type=["jpg", "jpeg", "png"])

    submitted = st.form_submit_button("Register batch")

    if submitted:
        if not farm_id:
            st.error("Farm ID is required.")
        elif herd_image is None:
            st.error("Please upload a herd image.")
        else:
            batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"

            # Save the uploaded image to disk so later modules can load it by path.
            image_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "uploads"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{batch_id}_herd.jpg"
            Image.open(herd_image).convert("RGB").save(image_path)

            insert_batch(
                batch_id=batch_id,
                farm_id=farm_id,
                declared_count=int(declared_count),
                image_path=str(image_path),
                date=str(batch_date),
            )

            st.success(f"Batch registered: **{batch_id}**")
            st.info("Copy this Batch ID — you'll need it on the Herd Verification page next.")
            st.code(batch_id)