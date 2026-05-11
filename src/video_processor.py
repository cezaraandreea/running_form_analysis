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
from pathlib import Path
from typing import Generator, Optional, Callable
from dataclasses import dataclass

from src.pose_estimation import PoseEstimator, PoseFrame
from src.biomechanics import BiomechanicsCalculator, BiomechanicsFrame


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

        return ProcessingResult(
            metadata=metadata,
            pose_frames=pose_frames,
            bio_frames=bio_frames,
            detection_rate=detection_rate,
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
        Returnează calea fișierului output.
        """
        cap      = cv2.VideoCapture(video_path)
        meta     = self.get_metadata(video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path, fourcc, meta.fps, (meta.width, meta.height)
        )

        # Indexăm pose_frames după frame_index
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
                # Desenează scheletul
                frame = self.pose_estimator.draw_skeleton(frame, pf)

                # Overlay metrici
                bf = bio_by_idx.get(frame_idx)
                if bf is not None:
                    frame = self._draw_metrics_overlay(frame, bf)

            writer.write(frame)

            if progress_callback and total > 0:
                progress_callback(frame_idx / total)

            frame_idx += 1

        cap.release()
        writer.release()
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