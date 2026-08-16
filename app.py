"""
app.py — RunAnalyzer: analiza tehnicii de alergare prin viziune artificială.
"""

import streamlit as st

from src.persistence.auth import init_db
from src.ui.auth_pages import auth_landing
from src.ui.pages import (
    analysis_page,
    history_page,
    home_page,
    results_page,
    upload_page,
)

st.set_page_config(
    page_title="RunAnalyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("src/ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Init DB si stare sesiune ──────────────────────────────────────────────────
init_db()

if "page"             not in st.session_state:
    st.session_state.page             = "home"
if "video_path"       not in st.session_state:
    st.session_state.video_path       = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "user"             not in st.session_state:
    st.session_state.user             = None   # dict {id, username, email} sau None
if "is_guest"         not in st.session_state:
    st.session_state.is_guest         = False
if "analysis_saved"   not in st.session_state:
    st.session_state.analysis_saved   = False

# ── Auth gate ─────────────────────────────────────────────────────────────────
_logged_in = bool(st.session_state.user) or st.session_state.is_guest
if not _logged_in:
    auth_landing()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.25rem 0.5rem 1rem;">
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

    # ── Info utilizator ───────────────────────────────────────────────────────
    user = st.session_state.user
    if user:
        initials = "".join(w[0].upper() for w in user["username"].split(".")[:2]) or user["username"][0].upper()
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;
                        padding:0.5rem 0.5rem 0.75rem;border-bottom:1px solid #2d2a40;">
                <div style="width:34px;height:34px;border-radius:50%;
                            background:linear-gradient(135deg,#8b5cf6,#6d28d9);
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.85rem;font-weight:700;color:#fff;flex-shrink:0;">
                    {initials}
                </div>
                <div>
                    <div style="font-size:0.85rem;font-weight:600;color:#ece8f5;">
                        {user['username']}
                    </div>
                    <div style="font-size:0.72rem;color:#6b6388;">{user['email']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="padding:0.4rem 0.5rem 0.75rem;border-bottom:1px solid #2d2a40;">
                <span style="font-size:0.8rem;color:#6b6388;">👤 Mod invitat</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    # ── Navigare ──────────────────────────────────────────────────────────────
    nav_items = [
        ("Acasă",         "home"),
        ("Încarcă video", "upload"),
        ("Analiză",       "analysis"),
        ("Rezultate",     "results"),
    ]
    if user:
        nav_items.append(("Istoric", "history"))

    current = st.session_state.page
    for label, key in nav_items:
        if current == key:
            st.markdown(
                f'<div class="nav-active"><span>{label}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    # ── Progres ───────────────────────────────────────────────────────────────
    st.markdown("---")
    has_video   = bool(st.session_state.video_path)
    has_results = bool(st.session_state.analysis_results)
    steps = [
        ("Video încărcat", has_video),
        ("Analiză rulată", has_results),
        ("Rezultate gata", has_results),
    ]
    st.markdown(
        '<div style="font-size:0.72rem;color:#6b6388;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:0.5rem;">Progres</div>',
        unsafe_allow_html=True,
    )
    for step_label, done in steps:
        icon  = "✓" if done else "·"
        color = "#34d399" if done else "#6b6388"
        st.markdown(
            f'<div style="font-size:0.82rem;color:{color};padding:2px 0;">'
            f'{icon}&nbsp; {step_label}</div>',
            unsafe_allow_html=True,
        )

    # ── Deconectare / Schimba cont ────────────────────────────────────────────
    st.markdown("---")
    if user:
        if st.button("Deconectare", use_container_width=True):
            st.session_state.user             = None
            st.session_state.is_guest         = False
            st.session_state.page             = "home"
            st.session_state.analysis_results = None
            st.session_state.video_path       = None
            st.session_state.analysis_saved   = False
            st.rerun()
    else:
        if st.button("Conectare / Cont nou", use_container_width=True):
            st.session_state.is_guest = False
            st.rerun()

# ── Router ────────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "home":
    home_page()
elif page == "upload":
    upload_page()
elif page == "analysis":
    analysis_page()
elif page == "results":
    results_page()
elif page == "history":
    history_page()
