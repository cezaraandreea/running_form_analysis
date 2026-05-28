"""
Pagini Streamlit pentru aplicatia RunAnalyzer.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.video_processor import VideoProcessor
from src.analysis import RunningAnalyzer, Severity
from src.visualization import (
    plot_elbow_angles,
    plot_knee_angles,
    plot_overview_dashboard,
    plot_score_gauge,
    plot_symmetry,
    plot_trunk_lean,
)


UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/output")


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def home_page() -> None:
    st.title("Sistem de analiza a tehnicii de alergare")
    st.markdown(
        """
        Aplicatia proceseaza videoclipuri de alergare si extrage:
        - keypoints anatomice (umeri, coate, solduri, genunchi, glezne),
        - parametri biomecanici,
        - feedback automat si scor general.
        """
    )
    st.info("Pasii recomandati: incarca video -> ruleaza analiza -> verifica rezultatele.")


def upload_page() -> None:
    _ensure_dirs()
    st.header("Incarcare video")
    uploaded_file = st.file_uploader(
        "Alege un fisier video (.mp4/.avi/.mov)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if not uploaded_file:
        st.caption("Nu ai incarcat inca un fisier.")
        return

    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.video_path = str(save_path)
    st.success(f"Video incarcat: {save_path.name}")
    st.video(str(save_path))

    if st.button("Mergi la analiza", type="primary", use_container_width=True):
        st.session_state.page = "analysis"
        st.rerun()


def analysis_page() -> None:
    st.header("Analiza tehnicii de alergare")
    video_path = st.session_state.get("video_path")
    if not video_path:
        st.warning("Incarca mai intai un videoclip din pagina de upload.")
        return

    col1, col2 = st.columns(2)
    with col1:
        skip_frames = st.slider("Skip frame-uri (viteza)", min_value=0, max_value=6, value=1)
    with col2:
        max_frames = st.number_input("Max frame-uri analizate (0 = toate)", min_value=0, value=0)

    run_btn = st.button("Ruleaza analiza", type="primary", use_container_width=True)
    if not run_btn:
        st.caption("Apasa butonul pentru procesare.")
        return

    progress = st.progress(0.0)
    status = st.empty()
    max_frames_val = int(max_frames) if int(max_frames) > 0 else None

    with st.spinner("Procesez videoclipul..."):
        with VideoProcessor(skip_frames=skip_frames, max_frames=max_frames_val) as processor:
            result = processor.process(
                video_path,
                progress_callback=lambda p: progress.progress(min(max(float(p), 0.0), 1.0)),
            )
            analyzer = RunningAnalyzer()
            analysis_result = analyzer.analyze_sequence(result.bio_frames)

            annotated_path = OUTPUT_DIR / f"annotated_{Path(video_path).stem}.mp4"
            processor.generate_annotated_video(
                video_path=video_path,
                output_path=str(annotated_path),
                pose_frames=result.pose_frames,
                bio_frames=result.bio_frames,
            )

    status.success("Analiza finalizata.")
    st.session_state.analysis_results = {
        "processing": result,
        "analysis": analysis_result,
        "annotated_video_path": str(annotated_path),
    }
    st.session_state.page = "results"
    st.rerun()


def _severity_label(severity: Severity) -> str:
    if severity == Severity.ERROR:
        return "eroare"
    if severity == Severity.WARNING:
        return "atentie"
    return "ok"


def results_page() -> None:
    st.header("Rezultate")
    data = st.session_state.get("analysis_results")
    if not data:
        st.warning("Nu exista inca rezultate. Ruleaza analiza mai intai.")
        return

    processing = data["processing"]
    analysis = data["analysis"]
    score = analysis.get("score", 0.0)
    stats = analysis.get("statistics", {})
    issues = analysis.get("issues", [])
    summary = analysis.get("summary", "Fara sumar.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Frame-uri analizate", len(processing.bio_frames))
    c2.metric("Detection rate", f"{processing.detection_rate:.1f}%")
    c3.metric("Scor general", f"{score:.0f}/100")

    st.plotly_chart(plot_score_gauge(score), use_container_width=True)
    st.plotly_chart(plot_overview_dashboard(stats, score), use_container_width=True)

    st.subheader("Feedback automat")
    st.markdown(summary)
    for item in issues:
        st.write(f"- [{_severity_label(item.severity)}] {item.category}: {item.message}")

    st.subheader("Grafice")
    if processing.bio_frames:
        st.plotly_chart(plot_knee_angles(processing.bio_frames), use_container_width=True)
        st.plotly_chart(plot_trunk_lean(processing.bio_frames), use_container_width=True)
        st.plotly_chart(plot_symmetry(processing.bio_frames), use_container_width=True)
        st.plotly_chart(plot_elbow_angles(processing.bio_frames), use_container_width=True)
    else:
        st.warning("Nu exista frame-uri cu date biomecanice.")

    annotated_video_path = data.get("annotated_video_path")
    if annotated_video_path and Path(annotated_video_path).exists():
        st.subheader("Video adnotat")
        st.video(annotated_video_path)

    if st.button("Analizeaza alt videoclip", use_container_width=True):
        st.session_state.page = "upload"
        st.session_state.analysis_results = None
        st.rerun()
