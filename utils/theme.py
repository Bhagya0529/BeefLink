"""
utils/theme.py

Shared visual theme for the BeefLink dashboard. Call apply_theme() near the
top of every page (after st.set_page_config) to get consistent fonts,
colors, and component styling across the whole app, matching the palette
used in the thesis and panel presentation (warm parchment background,
deep forest green, mono/serif type pairing).

USAGE:
    from utils.theme import apply_theme, status_badge, workflow_step

    st.set_page_config(page_title="...", layout="wide")
    apply_theme()
    ...
    st.markdown(status_badge("Verified"), unsafe_allow_html=True)
"""

import streamlit as st


THEME_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');


:root {
    --bl-bg: #FAF6EC;
    --bl-panel: #F1EAD6;
    --bl-ink: #201D16;
    --bl-ink-soft: #52493A;
    --bl-forest: #1F3D2B;
    --bl-forest-soft: #2B5138;
    --bl-verified: #2D6A4F;
    --bl-review: #A9752B;
    --bl-rejected: #8B3A3A;
    --bl-line: #DDD2AE;
}


/* =========================================================
   BASE APP
   ========================================================= */

.stApp {
    background-color: var(--bl-bg);
    color: var(--bl-ink);
    font-family: 'Inter', sans-serif;
}


/* =========================================================
   REMOVE STREAMLIT UI CONTROLS & TOOLTIPS
   ========================================================= */

/* Remove the sidebar collapse / expand button and its tooltip */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stSidebarCollapseButton"] button {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Hide all tooltips and keyboard hints */
[role="tooltip"],
div[aria-label*="keyboard"],
span[aria-label*="keyboard"],
div[title*="keyboard"],
span[title*="keyboard"] {
    display: none !important;
    visibility: hidden !important;
}

/* Remove Streamlit's top-right toolbar */
[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}

/* Remove Streamlit header */
header {
    display: none !important;
    visibility: hidden !important;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: var(--bl-forest) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background-color: var(--bl-panel);
    border-right: 1px solid var(--bl-line);
}

[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
    background-color: var(--bl-forest) !important;
    color: #FAF6EC !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: background-color 0.15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > a:hover {
    background-color: var(--bl-forest-soft) !important;
    color: #FAF6EC !important;
}


/* =========================================================
   TEXT INPUTS, NUMBER INPUTS, SELECTS
   ========================================================= */

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox > div {
    border-radius: 4px !important;
    border-color: var(--bl-line) !important;
    font-family: 'Inter', sans-serif !important;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--bl-forest) !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--bl-ink-soft) !important;
}


/* =========================================================
   METRIC / INFO CARDS
   ========================================================= */

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--bl-panel);
    border-color: var(--bl-line) !important;
    border-radius: 6px !important;
}


/* =========================================================
   CAPTIONS
   ========================================================= */

[data-testid="stCaptionContainer"],
.stCaption {
    color: var(--bl-ink-soft) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}


/* =========================================================
   ALERTS
   ========================================================= */

[data-testid="stAlertContentSuccess"] {
    color: var(--bl-verified) !important;
}

[data-testid="stAlertContentError"] {
    color: var(--bl-rejected) !important;
}

[data-testid="stAlertContentWarning"] {
    color: var(--bl-review) !important;
}

div[data-baseweb="notification"] {
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
}


/* =========================================================
   PROGRESS BAR
   ========================================================= */

.stProgress > div > div > div {
    background-color: var(--bl-forest) !important;
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {
    border-color: var(--bl-line) !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bl-panel) !important;
    border: 1px dashed var(--bl-line) !important;
    border-radius: 6px !important;
}


/* =========================================================
   TABS
   ========================================================= */

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--bl-ink-soft) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--bl-forest) !important;
}


/* =========================================================
   HERO BANNER
   ========================================================= */

.bl-hero {
    background-color: var(--bl-forest);
    color: #FAF6EC;
    padding: 2.2rem 2.4rem;
    border-radius: 8px;
    margin-bottom: 1.6rem;
}

.bl-hero h1 {
    color: #FAF6EC !important;
    margin: 0 0 0.4rem 0 !important;
    font-size: 2.2rem !important;
}

.bl-hero p {
    color: #E3DFC8;
    font-family: 'Inter', sans-serif;
    font-size: 1.02rem;
    margin: 0;
    max-width: 60ch;
}


/* =========================================================
   SECTION EYEBROW
   ========================================================= */

.bl-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--bl-forest-soft);
    margin-bottom: 0.3rem;
}

</style>
"""


def apply_theme():
    """Injects the shared BeefLink CSS theme into the current page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """
    Returns an HTML pill badge for a status string, styled consistently
    with the rest of the app. Pass to st.markdown(..., unsafe_allow_html=True).

    status: one of "Verified", "Review Required", "Rejected"
            (also accepts the lowercase dashboard-internal forms:
            "verified", "manual_review", "rejected")
    """

    normalized = status.strip().lower()

    mapping = {
        "verified": ("Verified", "#2D6A4F"),
        "review required": ("Review Required", "#A9752B"),
        "manual_review": ("Review Required", "#A9752B"),
        "rejected": ("Rejected", "#8B3A3A"),
    }

    label, color = mapping.get(
        normalized,
        (status, "#52493A")
    )

    return f"""
    <span style="
        display:inline-block;
        font-family:'IBM Plex Mono', monospace;
        font-size:0.72rem;
        letter-spacing:0.06em;
        text-transform:uppercase;
        padding:0.28rem 0.8rem;
        border-radius:100px;
        background-color:{color};
        color:#FAF6EC;
    ">{label}</span>
    """


def eyebrow(text: str) -> str:
    """Returns a small mono-font eyebrow label, e.g. above a section title."""
    return f'<div class="bl-eyebrow">{text}</div>'


def hero(title: str, subtitle: str) -> str:
    """Returns a full-width forest-green hero banner for a page's top."""
    return f"""
    <div class="bl-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """


def workflow_step(step_num: str, emoji: str, title: str, description: str) -> str:
    """Returns a styled workflow step card with emoji and description."""
    return f"""
    <div style="
        background-color: var(--bl-panel);
        border: 1px solid var(--bl-line);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
    ">
        <div style="font-size: 3rem; margin-bottom: 0.8rem; line-height: 1;">{emoji}</div>
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--bl-forest);
            margin-bottom: 0.6rem;
            font-weight: 600;
        ">▸ Step {step_num}</div>
        <h3 style="
            font-family: 'Fraunces', serif;
            color: var(--bl-forest);
            margin: 0 0 0.8rem 0;
            font-size: 1.2rem;
            font-weight: 600;
        ">{title}</h3>
        <p style="
            font-family: 'Inter', sans-serif;
            color: var(--bl-ink-soft);
            font-size: 0.9rem;
            margin: 0;
            line-height: 1.6;
        ">{description}</p>
    </div>
    """