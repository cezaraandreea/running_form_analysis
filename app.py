"""
app.py — RunAnalyzer: analiza tehnicii de alergare prin viziune artificială.
"""

import streamlit as st
from src.ui.pages import home_page, upload_page, analysis_page, results_page

st.set_page_config(
    page_title="RunAnalyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("src/ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.25rem 0.5rem 1rem;">
            <div style="font-size:1.3rem;font-weight:700;color:#e6edf3;letter-spacing:-0.3px;">
                🏃 RunAnalyzer
            </div>
            <div style="font-size:0.75rem;color:#6e7681;margin-top:2px;">
                Analiza tehnicii de alergare
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    nav_items = [
        ("Acasă",         "home",     "○"),
        ("Încarcă video", "upload",   "○"),
        ("Analiză",       "analysis", "○"),
        ("Rezultate",     "results",  "○"),
    ]

    current = st.session_state.page
    for label, key, _ in nav_items:
        active = current == key
        btn_style = (
            "background:rgba(47,129,247,0.12);color:#2f81f7;font-weight:600;"
            if active else ""
        )
        st.markdown(
            f'<div style="margin:2px 0;border-radius:6px;{btn_style}">'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    # Show progress indicator when a video is loaded
    st.markdown("---")
    has_video   = bool(st.session_state.video_path)
    has_results = bool(st.session_state.analysis_results)
    steps = [
        ("Video încărcat",  has_video),
        ("Analiză rulată",  has_results),
        ("Rezultate gata",  has_results),
    ]
    st.markdown(
        '<div style="font-size:0.72rem;color:#6e7681;text-transform:uppercase;'
        'letter-spacing:0.07em;margin-bottom:0.5rem;">Progres</div>',
        unsafe_allow_html=True,
    )
    for step_label, done in steps:
        icon  = "✓" if done else "·"
        color = "#3fb950" if done else "#6e7681"
        st.markdown(
            f'<div style="font-size:0.82rem;color:{color};padding:2px 0;">'
            f'{icon}&nbsp; {step_label}</div>',
            unsafe_allow_html=True,
        )

# ── Router ───────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "home":
    home_page()
elif page == "upload":
    upload_page()
elif page == "analysis":
    analysis_page()
elif page == "results":
    results_page()
