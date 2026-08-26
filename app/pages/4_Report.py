"""
app/pages/4_Report.py

Verification Report. Pulls together the herd and meat
verification results for a batch, runs the Verification Engine
to get a trust score and final decision, and displays/saves the report.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from database.schema import init_db, get_batch_full_record, insert_report
from verification.verification_engine import run_verification
from verification.report_export import generate_pdf_report, generate_share_text
from utils.theme import apply_theme, eyebrow, status_badge

init_db()

st.set_page_config(page_title="Verification Report — BeefLink", layout="wide", page_icon="🐄")
apply_theme()

st.title("Verification Report")
st.caption("Combines herd and meat verification into one trust score and final decision.")

batch_id = st.text_input("Batch ID", placeholder="e.g. BATCH-A1B2C3D4")

if batch_id:
    record = get_batch_full_record(batch_id)

    if record["batch"] is None:
        st.error("No batch found with that ID.")
    elif record["herd"] is None:
        st.warning("No herd verification found for this batch yet. Complete Module 2 first.")
    elif record["meat"] is None:
        st.warning("No meat verification found for this batch yet. Complete Module 3 first.")
    else:
        batch = record["batch"]
        herd = record["herd"]
        meat = record["meat"]

        herd_result = {
            "detected_count": herd["detected_count"],
            "average_confidence": herd["average_confidence"],
            "difference": abs(batch["declared_count"] - herd["detected_count"]),
            "status": herd["status"],
        }
        meat_result = {
            "label": meat["freshness_label"],
            "confidence": meat["freshness_confidence"],
        }

        outcome = run_verification(batch["declared_count"], herd_result, meat_result)

        st.subheader(f"Batch: {batch_id} — Farm: {batch['farm_id']}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Trust Score", f"{outcome['trust_score']} / 100")
            st.markdown(f"**Final Decision:** {status_badge(outcome['final_decision'])}", unsafe_allow_html=True)
        with col2:
            st.caption("Trust score breakdown:")
            for component, points in outcome["trust_score_breakdown"].items():
                st.write(f"- {component.replace('_', ' ').title()}: {points['earned']} / {points['max']}")

        st.divider()
        st.write(f"**Declared count:** {batch['declared_count']} — **Detected count:** {herd['detected_count']}")
        st.write(f"**Freshness:** {meat['freshness_label']} ({meat['freshness_confidence']*100:.1f}% confidence)")

        st.divider()
        st.subheader("Export this report")

        col_pdf, col_share = st.columns(2)

        with col_pdf:
            pdf_bytes = generate_pdf_report(batch, herd, meat, outcome, batch_id)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"beeflink_report_{batch_id}.pdf",
                mime="application/pdf",
                width="stretch",
            )

        with col_share:
            share_text = generate_share_text(batch, herd, meat, outcome, batch_id)
            email_subject = f"BeefLink Verification Report — {batch_id}"
            # mailto links need URL-encoded spaces/newlines
            import urllib.parse
            mailto_url = (
                f"mailto:?subject={urllib.parse.quote(email_subject)}"
                f"&body={urllib.parse.quote(share_text)}"
            )
            st.link_button("✉️ Share via email", mailto_url, width="stretch")

        with st.expander("Or copy the report as text"):
            st.code(share_text, language=None)
            st.caption("Select the text above and copy it — useful for WhatsApp, Slack, or pasting into any message.")

        st.divider()
        st.subheader("Buyer decision")
        st.caption(
            f"The system recommends '{outcome['final_decision']}' based on the trust score above. "
        )

        col_accept, col_reject = st.columns(2)
        with col_accept:
            if st.button("✅ Accept this lot", width="stretch"):
                insert_report(batch_id, outcome["trust_score"], outcome["final_decision"], buyer_decision="Accepted")
                st.success("Lot accepted. Report saved.")
        with col_reject:
            if st.button("❌ Reject this lot", width="stretch"):
                insert_report(batch_id, outcome["trust_score"], outcome["final_decision"], buyer_decision="Rejected")
                st.error("Lot rejected. Report saved.")