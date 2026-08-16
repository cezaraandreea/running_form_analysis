"""
src/ui/auth_pages.py
Pagina de autentificare: alegere vizitator / conectare / cont nou.
"""

from __future__ import annotations

import streamlit as st

from src.persistence.auth import login_user, register_user


def auth_landing() -> None:
    """
    Pagina de start afisata inainte de autentificare.
    Utilizatorul alege intre: vizitator, conectare, cont nou.
    """
    st.markdown(
        """
        <div style="max-width:460px;margin:60px auto 0;text-align:center;padding:0 1rem;">
            <h1 style="font-size:2.1rem;font-weight:800;color:#e6edf3;
                       margin-bottom:0.4rem;letter-spacing:-0.5px;">
                RunAnalyzer
            </h1>
            <p style="color:#8b949e;font-size:1rem;margin-bottom:2rem;">
                Analizează-ți tehnica de alergare cu ajutorul viziunii artificiale.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_login, tab_register = st.tabs(["Conectare", "Cont nou"])

        with tab_login:
            _login_form()

        with tab_register:
            _register_form()

        st.markdown(
            "<div style='text-align:center;color:#6b6388;margin:1.2rem 0 0.6rem;'>"
            "— sau —</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Continuă ca vizitator",
            use_container_width=True,
            key="guest_btn",
        ):
            _set_guest()

        st.markdown(
            "<p style='text-align:center;color:#6b6388;font-size:0.78rem;margin-top:0.4rem;'>"
            "Ca vizitator, analizele nu sunt salvate și nu poți urmări progresul în timp.</p>",
            unsafe_allow_html=True,
        )


# ── Formulare ─────────────────────────────────────────────────────────────────

def _login_form() -> None:
    with st.form("login_form"):
        username = st.text_input("Utilizator sau email", placeholder="ion.popescu")
        password = st.text_input("Parolă", type="password")
        submitted = st.form_submit_button(
            "Conectare", use_container_width=True, type="primary"
        )

    if not submitted:
        return

    if not username or not password:
        st.error("Completează toate câmpurile.")
        return

    user = login_user(username, password)
    if user:
        _set_user(user)
    else:
        st.error("Utilizator sau parolă incorectă.")


def _register_form() -> None:
    with st.form("register_form"):
        username  = st.text_input("Nume utilizator", placeholder="ion.popescu", key="reg_u")
        email     = st.text_input("Email", placeholder="ion@exemplu.com",       key="reg_e")
        password  = st.text_input("Parolă (min. 6 caractere)", type="password", key="reg_p")
        password2 = st.text_input("Confirmă parola",           type="password", key="reg_p2")
        submitted = st.form_submit_button(
            "Creează cont", use_container_width=True, type="primary"
        )

    if not submitted:
        return

    if password != password2:
        st.error("Parolele nu coincid.")
        return

    ok, msg = register_user(username, email, password)
    if not ok:
        st.error(msg)
        return

    # Login automat dupa inregistrare
    user = login_user(username, password)
    if user:
        st.success(f"Cont creat! Bine ai venit, {user['username']}.")
        _set_user(user)


# ── Helpers de stare ──────────────────────────────────────────────────────────

def _set_user(user: dict) -> None:
    st.session_state.user     = user
    st.session_state.is_guest = False
    st.session_state.page     = "home"
    st.rerun()


def _set_guest() -> None:
    st.session_state.user     = None
    st.session_state.is_guest = True
    st.session_state.page     = "home"
    st.rerun()
