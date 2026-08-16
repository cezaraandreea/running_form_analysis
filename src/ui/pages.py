"""
Pagini Streamlit pentru aplicatia RunAnalyzer.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import streamlit as st

from src.services.analysis import Severity
from src.services.email_sender import EmailConfigError, is_email_configured, send_analysis_email
from src.persistence.history import get_analysis_detail, get_user_analyses, save_analysis
from src.services.report import generate_pdf_report
from src.services.video_processor import VideoProcessor
from src.pipeline.visualization import (
    plot_elbow_angles,
    plot_foot_strike_chart,
    plot_knee_angles,
    plot_knee_drive_symmetry,
    plot_overstride_chart,
    plot_phase_contact_symmetry,
    plot_score_gauge,
    plot_trunk_lean,
    plot_vertical_oscillation,
)


UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/output")


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def home_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Analizează-ți<br>tehnica de alergare</h1>
            <p>Încarcă un videoclip filmat din lateral și obține în câteva secunde
            o analiză completă a biomecanicii tale — de la cadență și lungimea pasului
            până la simetria genunchilor și tipul de aterizare.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("📐", "Biomecanică completă",
         "Unghiuri genunchi, trunchi și coate analizate cadru cu cadru pe tot parcursul filmării."),
        ("⚖️", "Simetrie stânga–dreapta",
         "Detectăm diferențele dintre piciorul stâng și drept la contact și la ridicarea genunchiului."),
        ("📋", "Feedback personalizat",
         "Scor general 0–100 și recomandări specifice bazate pe parametrii biomecanici măsurați."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        col.markdown(
            f'<div class="feature-card"><div class="icon">{icon}</div>'
            f'<h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    cards2 = [
        ("👟", "Tip de aterizare",
         "Identifică dacă alergătorul aterizează pe călcâi, mijlocul tălpii sau vârful piciorului."),
        ("📏", "Lungime pas & cadență",
         "Calculează lungimea medie a pasului în metri și cadența în pași pe minut."),
        ("📄", "Export PDF",
         "Descarcă un raport complet cu toate metricile și graficele analizei."),
    ]
    for col, (icon, title, desc) in zip([c4, c5, c6], cards2):
        col.markdown(
            f'<div class="feature-card"><div class="icon">{icon}</div>'
            f'<h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**Cerință video:** filmează din lateral, cu alergătorul complet vizibil în cadru "
        "și o singură persoană în imagine."
    )
    if st.button("Începe analiza →", type="primary"):
        st.session_state.page = "upload"
        st.rerun()


def _nav(back_label: str, back_page: str, forward_label: str | None = None, forward_page: str | None = None) -> None:
    """Header bar de navigare cu butoane mici în colțurile paginii."""
    st.markdown('<span id="nav-marker"></span>', unsafe_allow_html=True)
    if forward_label:
        c_back, _, c_fwd = st.columns([2, 7, 2])
    else:
        c_back, _ = st.columns([2, 9])
    with c_back:
        if st.button(f"← {back_label}", key=f"nb_{back_page}_{forward_label}"):
            st.session_state.page = back_page
            st.rerun()
    if forward_label and forward_page:
        with c_fwd:
            if st.button(f"{forward_label} →", key=f"nf_{forward_page}", type="primary"):
                st.session_state.page = forward_page
                st.rerun()


def _step_bar(active: int) -> None:
    """Bara de progres cu 3 pași: 1=upload, 2=analiză, 3=rezultate."""
    steps = ["Încarcă video", "Configurare", "Rezultate"]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        step_num = i + 1
        if step_num < active:
            color, dot = "#34d399", "✓"
        elif step_num == active:
            color, dot = "#8b5cf6", str(step_num)
        else:
            color, dot = "#2d2a40", str(step_num)
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:22px;height:22px;border-radius:50%;background:{color};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:10px;font-weight:700;color:#fff;flex-shrink:0;">{dot}</div>'
            f'<span style="font-size:0.82rem;color:{"#ece8f5" if step_num <= active else "#6b6388"};'
            f'font-weight:{"600" if step_num == active else "400"};">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)


def upload_page() -> None:
    _ensure_dirs()
    _nav("Acasă", "home")
    _step_bar(1)
    st.header("Încarcă videoclipul")
    uploaded_file = st.file_uploader(
        "Alege un fișier video (.mp4 / .avi / .mov)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if not uploaded_file:
        return

    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.video_path = str(save_path)
    st.success(f"Video încărcat: **{save_path.name}**")
    st.video(str(save_path))

    if st.button("Continuă spre analiză", type="primary", use_container_width=True):
        st.session_state.page = "analysis"
        st.rerun()


def analysis_page() -> None:
    _nav("Înapoi la upload", "upload")
    _step_bar(2)
    st.header("Detalii despre alergător")
    video_path = st.session_state.get("video_path")
    if not video_path:
        st.warning("Încarcă mai întâi un videoclip.")
        return

    col1, col2 = st.columns(2)
    with col1:
        runner_height = st.number_input(
            "Înălțime (cm)", min_value=140, max_value=220, value=175,
            help="Folosită pentru a calcula lungimea pasului și oscilația verticală în metri."
        )
    with col2:
        runner_sex = st.selectbox(
            "Sex", options=["Masculin", "Feminin"],
            help="Influențează raportul lungime picior / înălțime și pragurile de cadență."
        )

    st.session_state.runner_height_cm = int(runner_height)
    st.session_state.runner_sex = runner_sex

    if not st.button("Analizează", type="primary", use_container_width=True):
        return

    progress = st.progress(0.0)
    status   = st.empty()

    with st.spinner("Procesez videoclipul..."):
        with VideoProcessor(skip_frames=1) as processor:
            result = processor.process(
                video_path,
                progress_callback=lambda p: progress.progress(min(max(float(p), 0.0), 1.0)),
                runner_height_cm=st.session_state.get("runner_height_cm", 175),
                runner_sex=st.session_state.get("runner_sex", "Masculin"),
            )

            annotated_path = OUTPUT_DIR / f"annotated_{Path(video_path).stem}.mp4"
            processor.generate_annotated_video(
                video_path=video_path,
                output_path=str(annotated_path),
                pose_frames=result.pose_frames,
                bio_frames=result.bio_frames,
            )

    try:
        ffmpeg_ok = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        ffmpeg_ok = False

    status.success("Analiză finalizată.")
    st.session_state.analysis_results = {
        "processing": result,
        "annotated_video_path": str(annotated_path),
        "ffmpeg_available": ffmpeg_ok,
    }
    st.session_state.page = "results"
    st.rerun()


def results_page() -> None:
    _nav("Configurare", "analysis", "Analiză nouă", "upload")
    _step_bar(3)
    st.header("Rezultatele analizei")
    data = st.session_state.get("analysis_results")
    if not data:
        st.warning("Nu există încă rezultate. Rulează mai întâi analiza.")
        return

    # ── Salvare automata (o singura data per analiza, doar pt. utilizatori autentificati) ──
    user = st.session_state.get("user")
    if user and not st.session_state.get("analysis_saved", False):
        processing = data["processing"]
        if processing.analysis_result:
            save_analysis(
                user_id=user["id"],
                video_filename=st.session_state.get("video_path", ""),
                analysis_result=processing.analysis_result,
                height_cm=st.session_state.get("runner_height_cm", 175),
                runner_sex=st.session_state.get("runner_sex", "Masculin"),
            )
            st.session_state.analysis_saved = True
            st.toast("Analiză salvată în contul tău.", icon="✅")

    processing  = data["processing"]
    gait        = processing.gait_analysis
    height_cm   = st.session_state.get("runner_height_cm", 175)
    runner_sex  = st.session_state.get("runner_sex", "Masculin")
    leg_ratio   = 0.49 if runner_sex == "Masculin" else 0.46
    ideal_cad   = "170–185" if runner_sex == "Masculin" else "174–190"

    # ── Metrici principale ────────────────────────────────────────────────────
    if gait:
        col1, col2, col3, col4 = st.columns(4)

        if gait.phase_symmetry_contact_deg is not None:
            col1.metric(
                "Simetrie la contact",
                f"{gait.phase_symmetry_contact_deg:.1f}°",
                help="Diferența medie a unghiului genunchiului între piciorul stâng și drept la aterizare. Ideal < 15°.",
            )

        if gait.knee_drive_symmetry_deg is not None:
            col2.metric(
                "Simetrie ridicare genunchi",
                f"{gait.knee_drive_symmetry_deg:.1f}°",
                help="Diferența medie a unghiului genunchiului între stânga și dreapta la ridicare. Ideal < 15°.",
            )

        if gait.left_drive_knee_mean_deg is not None and gait.right_drive_knee_mean_deg is not None:
            col3.metric(
                "Unghi ridicare genunchi (S / D)",
                f"{gait.left_drive_knee_mean_deg:.0f}° / {gait.right_drive_knee_mean_deg:.0f}°",
                help="Unghiul mediu al genunchiului la ridicare. Valori mai mici = ridicare mai înaltă.",
            )

        step_vals = [v for v in [gait.left_step_length_mean, gait.right_step_length_mean] if v is not None]
        if step_vals and gait.leg_length_normalized:
            leg_len_m   = height_cm * leg_ratio / 100
            scale       = leg_len_m / gait.leg_length_normalized
            avg_step_m  = sum(step_vals) / len(step_vals) * scale
            col4.metric("Lungime medie pas", f"{avg_step_m:.2f} m")

        col5, col6 = st.columns(2)

        if gait.cadence_spm is not None:
            col5.metric(
                "Cadență",
                f"{gait.cadence_spm:.0f} pași/min",
                help=f"Ideal pentru {'bărbați' if runner_sex == 'Masculin' else 'femei'}: {ideal_cad} pași/min.",
            )

        if gait.vertical_oscillation_norm is not None and gait.leg_length_normalized:
            leg_len_m = height_cm * leg_ratio / 100
            scale     = leg_len_m / gait.leg_length_normalized
            vo_cm     = gait.vertical_oscillation_norm * scale * 100
            col6.metric(
                "Oscilație verticală",
                f"{vo_cm:.1f} cm",
                help="Amplitudinea verticală a mișcării șoldului. Ideal < 8 cm; valori mari indică energie risipită.",
            )

    # ── Feedback automat ──────────────────────────────────────────────────────
    analysis = processing.analysis_result
    if analysis:
        st.divider()
        st.subheader("Feedback tehnică de alergare")

        col_gauge, col_summary = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(plot_score_gauge(analysis["score"]), use_container_width=True)
        with col_summary:
            st.markdown(analysis["summary"])

        issues   = analysis.get("issues", [])
        errors   = [i for i in issues if i.severity == Severity.ERROR]
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        oks      = [i for i in issues if i.severity == Severity.OK]

        if errors:
            with st.expander(f"🔴 Probleme critice ({len(errors)})", expanded=True):
                for item in errors:
                    st.error(f"**{item.category}** — {item.message}")
                    if item.detail:
                        st.caption(f"→ {item.detail}")

        if warnings:
            with st.expander(f"🟡 Avertizări ({len(warnings)})", expanded=True):
                for item in warnings:
                    st.warning(f"**{item.category}** — {item.message}")
                    if item.detail:
                        st.caption(f"→ {item.detail}")

        if oks:
            with st.expander(f"🟢 Aspecte pozitive ({len(oks)})", expanded=False):
                for item in oks:
                    st.success(f"**{item.category}** — {item.message}")

    # ── Grafice ───────────────────────────────────────────────────────────────
    if processing.bio_frames:
        st.divider()
        st.subheader("Grafice")

        st.plotly_chart(plot_knee_angles(processing.bio_frames),          use_container_width=True)
        st.plotly_chart(plot_trunk_lean(processing.bio_frames),           use_container_width=True)
        st.plotly_chart(plot_vertical_oscillation(processing.bio_frames), use_container_width=True)
        if gait:
            st.plotly_chart(plot_foot_strike_chart(gait),        use_container_width=True)
            st.plotly_chart(plot_phase_contact_symmetry(gait),   use_container_width=True)
            st.plotly_chart(plot_knee_drive_symmetry(gait),      use_container_width=True)
            st.plotly_chart(plot_overstride_chart(gait),         use_container_width=True)
        st.plotly_chart(plot_elbow_angles(processing.bio_frames), use_container_width=True)

    # ── Video adnotat ─────────────────────────────────────────────────────────
    annotated_video_path = data.get("annotated_video_path")
    if annotated_video_path and Path(annotated_video_path).exists():
        st.divider()
        st.subheader("Video adnotat")
        with open(annotated_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)

    # ── Export PDF ────────────────────────────────────────────────────────────
    st.divider()
    with st.spinner("Pregătesc raportul PDF..."):
        pdf_bytes = generate_pdf_report(
            bio_frames=processing.bio_frames,
            gait=gait,
            analysis=processing.analysis_result,
            height_cm=height_cm,
            runner_sex=runner_sex,
        )
    import datetime
    filename = f"RunAnalyzer_{datetime.date.today().strftime('%Y%m%d')}.pdf"
    st.download_button(
        label="Descarcă raport PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

    # ── Trimite pe email ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Trimite raportul pe email")

    if not is_email_configured():
        st.info(
            "Trimiterea prin email nu este configurată. "
            "Adaugă secțiunea `[email]` în `.streamlit/secrets.toml` pentru a activa această funcție.",
            icon="ℹ️",
        )
    else:
        user        = st.session_state.get("user")
        default_email = user["email"] if user else ""

        with st.form("email_form"):
            email_to = st.text_input(
                "Adresă email destinatar",
                value=default_email,
                placeholder="exemplu@gmail.com",
            )
            send_btn = st.form_submit_button(
                "Trimite raportul PDF", use_container_width=True
            )

        if send_btn:
            if not email_to or "@" not in email_to:
                st.error("Introdu o adresă de email validă.")
            else:
                analysis    = processing.analysis_result or {}
                stats       = analysis.get("statistics", {})
                score_val   = analysis.get("score")
                pdf_fname   = f"RunAnalyzer_{__import__('datetime').date.today().strftime('%Y%m%d')}.pdf"

                with st.spinner(f"Se trimite emailul către {email_to}..."):
                    try:
                        send_analysis_email(
                            to_email    = email_to,
                            pdf_bytes   = pdf_bytes,
                            score       = score_val,
                            stats       = stats,
                            runner_sex  = runner_sex,
                            pdf_filename= pdf_fname,
                        )
                        st.success(f"Raportul a fost trimis cu succes la **{email_to}**.")
                    except EmailConfigError as exc:
                        st.error(f"Eroare configurare email: {exc}")
                    except Exception as exc:
                        st.error(f"Trimiterea a eșuat: {exc}")

    st.divider()
    if st.button("Analizează alt videoclip", use_container_width=True):
        st.session_state.page             = "upload"
        st.session_state.analysis_results = None
        st.session_state.analysis_saved   = False
        st.rerun()


# ── Pagina Istoric ────────────────────────────────────────────────────────────

def history_page() -> None:
    _nav("Acasă", "home", "Analiză nouă", "upload")
    st.header("Istoricul analizelor tale")

    user = st.session_state.get("user")
    if not user:
        st.info("Conectează-te pentru a vedea istoricul analizelor.")
        return

    analyses = get_user_analyses(user["id"])

    if not analyses:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 1rem;color:#6b6388;">
                <div style="font-size:2.5rem;margin-bottom:0.75rem;">📊</div>
                <p style="font-size:1rem;">Nu ai nicio analiză salvată încă.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Începe prima analiză →", type="primary"):
            st.session_state.page = "upload"
            st.rerun()
        return

    st.caption(f"{len(analyses)} {'analiză' if len(analyses) == 1 else 'analize'} salvate")

    _SEVERITY_COLOR = {"error": "#f87171", "warning": "#fbbf24", "ok": "#34d399"}
    _SEVERITY_ICON  = {"error": "🔴", "warning": "🟡", "ok": "🟢"}

    for row in analyses:
        # Formatare data
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(row["created_at"])
            date_str = dt.strftime("%d %b %Y, %H:%M")
        except Exception:
            date_str = row["created_at"][:16]

        score       = row.get("score")
        cadence     = row.get("cadence_spm")
        vo          = row.get("vertical_oscillation_cm")
        foot_strike = row.get("foot_strike_type") or "—"
        video_name  = row.get("video_filename") or "necunoscut"
        sym         = row.get("symmetry_contact_deg")

        score_str = f"{int(score)}" if score is not None else "—"

        with st.expander(f"{date_str}  |  Scor: {score_str}/100  |  {video_name}"):
            # Metrici rapide
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Scor", f"{score_str}/100")
            c2.metric("Cadență", f"{cadence:.0f} p/min" if cadence else "—")
            c3.metric("Oscilație vert.", f"{vo:.1f} cm" if vo else "—")
            c4.metric("Aterizare", foot_strike)

            if sym is not None:
                st.caption(f"Simetrie la contact: **{sym:.1f}°**")

            runner_info = (
                f"{row.get('runner_height_cm', '?')} cm · "
                f"{row.get('runner_sex', '?')}"
            )
            st.caption(f"Alergător: {runner_info}")

            # Feedback
            import json
            try:
                feedback = json.loads(row.get("feedback_json") or "[]")
            except Exception:
                feedback = []

            if feedback:
                st.markdown("**Feedback:**")
                for item in feedback:
                    sev   = item.get("severity", "ok")
                    icon  = _SEVERITY_ICON.get(sev, "·")
                    color = _SEVERITY_COLOR.get(sev, "#ece8f5")
                    msg   = item.get("message", "")
                    cat   = item.get("category", "")
                    detail = item.get("detail", "")
                    st.markdown(
                        f"<div style='padding:4px 0;font-size:0.88rem;'>"
                        f"{icon} <strong style='color:{color}'>{cat}</strong> — {msg}"
                        + (f"<br><span style='color:#6b6388;margin-left:1.5rem;font-size:0.82rem;'>"
                           f"→ {detail}</span>" if detail else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()
    if st.button("Analiză nouă →", type="primary"):
        st.session_state.page           = "upload"
        st.session_state.analysis_saved = False
        st.rerun()
