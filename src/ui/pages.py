"""
Pagini Streamlit pentru aplicatia RunAnalyzer.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import streamlit as st

from src.analysis import Severity
from src.report import generate_pdf_report
from src.video_processor import VideoProcessor
from src.visualization import (
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
    st.title("RunAnalyzer")
    st.markdown(
        """
        **Analizează-ți tehnica de alergare** printr-un simplu videoclip filmat din lateral.

        Aplicația detectează automat:
        - unghiurile genunchilor și simetria la contactul cu solul
        - ridicarea genunchiului (knee drive) și simetria acesteia
        - tipul de aterizare: călcâi, mijlocul tălpii sau vârf
        - supraalungirea pasului (overstriding)
        - lungimea medie a pasului, cadența și oscilația verticală
        - un scor general al tehnicii și recomandări personalizate
        """
    )
    st.info(
        "**Cerințe video:** filmează din lateral, cu alergătorul vizibil complet în cadru "
        "și o singură persoană în imagine."
    )


def upload_page() -> None:
    _ensure_dirs()
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
    st.header("Rezultatele analizei")
    data = st.session_state.get("analysis_results")
    if not data:
        st.warning("Nu există încă rezultate. Rulează mai întâi analiza.")
        return

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

    st.divider()
    if st.button("Analizează alt videoclip", use_container_width=True):
        st.session_state.page = "upload"
        st.session_state.analysis_results = None
        st.rerun()
