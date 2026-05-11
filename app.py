"""
app.py - Punctul de intrare principal al aplicației Streamlit
Sistem de analiză a tehnicii de alergare folosind viziune artificială
"""

import streamlit as st
from src.ui.pages import home_page, upload_page, analysis_page, results_page

# ── Configurare pagină ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RunAnalyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global (importat din style.css) ─────────────────────────────────────
with open("src/ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── State management ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ── Sidebar navigație ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏃 RunAnalyzer")
    st.markdown("---")

    pages = {
        "🏠 Acasă":      "home",
        "📤 Încarcă video": "upload",
        "🔬 Analiză":    "analysis",
        "📊 Rezultate":  "results",
    }

    for label, page_key in pages.items():
        if st.button(label, use_container_width=True):
            st.session_state.page = page_key

    st.markdown("---")
    st.markdown("**Status sistem:**")
    st.markdown("✅ MediaPipe loaded")
    st.markdown("✅ OpenCV ready")

# ── Router pagini ─────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "home":
    home_page()
elif page == "upload":
    upload_page()
elif page == "analysis":
    analysis_page()
elif page == "results":
    results_page()