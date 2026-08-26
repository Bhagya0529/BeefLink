import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.schema import init_db, get_all_batches
from utils.theme import apply_theme, hero, eyebrow, workflow_step

init_db()

st.set_page_config(page_title="BeefLink", layout="wide", page_icon="🐄")
apply_theme()

st.markdown(
    hero(
        "BeefLink",
        "A decision-support system that independently verifies herd "
        "count and meat freshness.",
    ),
    unsafe_allow_html=True,
)

st.markdown(eyebrow("Workflow Pipeline"), unsafe_allow_html=True)

# Interactive workflow visualization
col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    st.markdown(
        workflow_step(
            "1",
            "📋",
            "Register Supply",
            "Enter the batch details and upload a herd photo to create the declaration."
        ),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        workflow_step(
            "2",
            "🔍",
            "Herd Verification",
            "YOLOv8s detector analyzes the herd image and validates the declared cattle count."
        ),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        workflow_step(
            "3",
            "🍖",
            "Meat Verification",
            "RADAM + SVM classifier evaluates meat freshness (fresh / half-fresh / spoiled)."
        ),
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        workflow_step(
            "4",
            "✅",
            "Final Report",
            "System combines both verifications into a trust score and buyer recommendation."
        ),
        unsafe_allow_html=True
    )

st.caption("💡 Each batch gets a unique **Batch ID** that tracks it through all four steps. Use the sidebar to navigate.")

st.divider()
st.markdown(eyebrow("Recent activity"), unsafe_allow_html=True)
st.subheader("Recent batches")

batches = get_all_batches()
if not batches:
    st.info("No batches registered yet. Start on the Register Supply page.")
else:
    for b in batches[:10]:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{b['batch_id']}**")
                st.caption(f"Farm: {b['farm_id']}  ·  Declared: {b['declared_count']}  ·  {b['date']}")
            with col2:
                st.caption(b['date'])