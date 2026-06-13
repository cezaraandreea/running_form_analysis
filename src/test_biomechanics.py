"""
Test rapid pentru pasul 2: calculul parametrilor biomecanici.

Ruleaza pipeline-ul pe un video, apoi afiseaza statistici agregate
si exporta un CSV cu metricile pe frame.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from src.video_processor import VideoProcessor
except ModuleNotFoundError:
    from video_processor import VideoProcessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "run_test4.mp4"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "output" / "biomechanics_metrics.csv"


def mean_of(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def print_stat(label: str, value: float | None, suffix: str = "") -> None:
    if value is None:
        print(f"- {label}: N/A")
        return
    print(f"- {label}: {value:.3f}{suffix}")


def export_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test biomecanica alergare din video.")
    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help="Calea catre fisierul video.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Proceseaza 1 din (skip_frames + 1) frame-uri.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Numarul maxim de frame-uri procesate (0 = toate).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Fisier CSV output pentru metrici per frame.",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Fisierul video nu exista: {video_path}")

    max_frames = args.max_frames if args.max_frames > 0 else None
    with VideoProcessor(skip_frames=max(0, args.skip_frames), max_frames=max_frames) as processor:
        result = processor.process(str(video_path))

    bio = result.bio_frames
    print("=== Rezultate biomecanice ===")
    print(f"- Video: {video_path.name}")
    print(f"- Frame-uri cu pose detectat: {len(result.pose_frames)}")
    print(f"- Detection rate: {result.detection_rate:.2f}%")
    print(f"- Frame-uri biomecanice: {len(bio)}")

    print_stat("Knee angle left mean", mean_of([f.knee_angle_left for f in bio]), " deg")
    print_stat("Knee angle right mean", mean_of([f.knee_angle_right for f in bio]), " deg")
    print_stat("Trunk lean mean", mean_of([f.trunk_lean_angle for f in bio]), " deg")
    print_stat("Elbow angle left mean", mean_of([f.elbow_angle_left for f in bio]), " deg")
    print_stat("Elbow angle right mean", mean_of([f.elbow_angle_right for f in bio]), " deg")
    print_stat("Diferenta L-R instantanee (medie)", mean_of([f.knee_symmetry_diff for f in bio]), " deg")

    from src.gait_analysis import analyze_gait_phases
    gait = analyze_gait_phases(result.pose_frames, result.bio_frames, fps=result.metadata.fps)
    print_stat("Simetrie la contact", gait.phase_symmetry_contact_deg, " deg")
    print(f"- Contacte estimate: stanga={len(gait.left_strikes)}, dreapta={len(gait.right_strikes)}")
    print_stat("Step length mean", mean_of([f.step_length for f in bio]), " norm")
    print_stat("Knee ROM proxy mean", mean_of([f.knee_rom_proxy for f in bio]), " norm")

    csv_out = Path(args.csv_out).expanduser().resolve()
    rows = [f.to_dict() for f in bio]
    export_csv(csv_out, rows)
    print(f"- CSV export: {csv_out}")


if __name__ == "__main__":
    main()
