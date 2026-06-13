"""
src/video_processor.py
Pipeline complet de procesare video:
  1. Citește frame-urile dintr-un videoclip
  2. Rulează pose estimation pe fiecare frame
  3. Calculează parametrii biomecanici
  4. Returnează toate datele + video adnotat
"""

import cv2
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, Optional, Callable
from dataclasses import dataclass

from src.pose_estimation import PoseEstimator, PoseFrame


def _reencode_h264(input_path: str, output_path: str) -> bool:
    """
    Re-encodează input_path la H.264/MP4 folosind ffmpeg.
    Returnează True dacă a reușit, False dacă ffmpeg nu este disponibil sau a eșuat.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",   # compatibil cu toate browserele
                "-an",                   # fără audio (nu există în videoclipul original)
                output_path,
            ],
            capture_output=True,
            timeout=600,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
from src.biomechanics import BiomechanicsCalculator, BiomechanicsFrame
from src.gait_analysis import GaitAnalysisResult, analyze_gait_phases
from src.analysis import RunningAnalyzer


@dataclass
class VideoMetadata:
    """Informații despre fișierul video."""
    path:       str
    fps:        float
    width:      int
    height:     int
    total_frames: int
    duration_s: float


@dataclass
class ProcessingResult:
    """Rezultatul complet al procesării unui videoclip."""
    metadata:      VideoMetadata
    pose_frames:   list[PoseFrame]
    bio_frames:    list[BiomechanicsFrame]
    detection_rate: float   # % frame-uri cu detecție reușită
    gait_analysis:   Optional[GaitAnalysisResult] = None
    analysis_result: Optional[dict] = None


class VideoProcessor:
    """
    Orchestrează întregul pipeline de procesare:
    video → frames → pose → biomecanică.
    """

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence:  float = 0.5,
        skip_frames: int = 0,        # procesează 1 din (skip_frames+1) frame-uri
        max_frames:  Optional[int] = None,   # limitează numărul de frame-uri
    ):
        self.detection_confidence = detection_confidence
        self.tracking_confidence  = tracking_confidence
        self.skip_frames = skip_frames
        self.max_frames  = max_frames

        self.pose_estimator = PoseEstimator(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.bio_calculator = BiomechanicsCalculator()

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Citește metadatele videoclipului fără a-l procesa."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Nu pot deschide videoclipul: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return VideoMetadata(
            path=video_path,
            fps=fps,
            width=width,
            height=height,
            total_frames=total,
            duration_s=total / fps if fps > 0 else 0,
        )

    def process(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
        runner_height_cm: int = 175,
        runner_sex: str = "Masculin",
    ) -> ProcessingResult:
        """
        Procesează complet videoclipul.
        progress_callback(0.0–1.0) este apelat după fiecare frame (opțional).
        """
        metadata = self.get_metadata(video_path)
        cap      = cv2.VideoCapture(video_path)

        pose_frames: list[PoseFrame]   = []
        bio_frames:  list[BiomechanicsFrame] = []

        frame_idx    = 0
        processed    = 0
        total_to_process = metadata.total_frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Saritură frame-uri (opțional, pentru viteză)
            if self.skip_frames > 0 and frame_idx % (self.skip_frames + 1) != 0:
                frame_idx += 1
                continue

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Pose estimation
            pose_frame = self.pose_estimator.process_frame(frame, frame_idx, timestamp_ms)

            if pose_frame is not None:
                pose_frames.append(pose_frame)

                # Biomecanică
                bio_frame = self.bio_calculator.calculate(pose_frame)
                bio_frames.append(bio_frame)

            processed += 1
            frame_idx += 1

            # Callback progres
            if progress_callback and total_to_process > 0:
                progress_callback(frame_idx / total_to_process)

            # Limită opțională
            if self.max_frames and processed >= self.max_frames:
                break

        cap.release()

        detection_rate = len(pose_frames) / max(processed, 1) * 100.0
        gait = analyze_gait_phases(pose_frames, bio_frames, fps=metadata.fps)
        analysis_result = RunningAnalyzer().analyze_sequence(
            bio_frames, gait,
            runner_height_cm=runner_height_cm,
            runner_sex=runner_sex,
        )

        return ProcessingResult(
            metadata=metadata,
            pose_frames=pose_frames,
            bio_frames=bio_frames,
            detection_rate=detection_rate,
            gait_analysis=gait,
            analysis_result=analysis_result,
        )

    def generate_annotated_video(
        self,
        video_path: str,
        output_path: str,
        pose_frames: list[PoseFrame],
        bio_frames:  list[BiomechanicsFrame],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """
        Generează un videoclip nou cu scheletul și metricile desenate pe fiecare frame.
        Scrie mai întâi cu mp4v într-un fișier temporar, apoi re-encodează la H.264
        cu ffmpeg (necesar pentru redare directă în browser).
        Returnează calea fișierului output.
        """
        cap  = cv2.VideoCapture(video_path)
        meta = self.get_metadata(video_path)

        # Scriem într-un fișier temporar cu codec mp4v (OpenCV nativ)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix="_raw.mp4")
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, meta.fps, (meta.width, meta.height))

        pose_by_idx = {pf.frame_index: pf for pf in pose_frames}
        bio_by_idx  = {bf.frame_index: bf for bf in bio_frames}

        frame_idx = 0
        total     = meta.total_frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pf = pose_by_idx.get(frame_idx)
            if pf is not None:
                frame = self.pose_estimator.draw_skeleton(frame, pf)
                bf = bio_by_idx.get(frame_idx)
                if bf is not None:
                    frame = self._draw_metrics_overlay(frame, bf)

            writer.write(frame)

            if progress_callback and total > 0:
                progress_callback(frame_idx / total)

            frame_idx += 1

        cap.release()
        writer.release()

        # Re-encodare H.264 cu ffmpeg pentru redare în browser
        if not _reencode_h264(tmp_path, output_path):
            # ffmpeg indisponibil — servim fișierul mp4v original
            import shutil
            shutil.move(tmp_path, output_path)
        else:
            os.remove(tmp_path)

        return output_path

    def _draw_metrics_overlay(self, frame: np.ndarray, bf: BiomechanicsFrame) -> np.ndarray:
        """Desenează metricile biomecanice ca text pe frame."""
        overlay = frame.copy()
        h, w    = frame.shape[:2]

        metrics = []
        if bf.knee_angle_left is not None:
            metrics.append(f"Knee L: {bf.knee_angle_left:.0f}deg")
        if bf.knee_angle_right is not None:
            metrics.append(f"Knee R: {bf.knee_angle_right:.0f}deg")
        if bf.trunk_lean_angle is not None:
            metrics.append(f"Trunk: {bf.trunk_lean_angle:.0f}deg")

        bg_h = len(metrics) * 22 + 10
        cv2.rectangle(overlay, (5, 5), (200, bg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        for i, text in enumerate(metrics):
            cv2.putText(
                frame, text,
                (10, 22 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 128), 1, cv2.LINE_AA
            )

        return frame

    def close(self):
        self.pose_estimator.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()