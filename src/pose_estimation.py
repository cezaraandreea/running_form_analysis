"""
src/pose_estimation.py
Detectarea corpului uman și extragerea punctelor-cheie (keypoints)
folosind Google MediaPipe Pose.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Keypoint:
    """Un punct anatomic detectat într-un frame."""
    name: str
    x: float          # coordonată normalizată [0, 1]
    y: float          # coordonată normalizată [0, 1]
    z: float          # adâncime estimată (relativă)
    visibility: float # încrederea detecției [0, 1]

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convertește coordonatele normalizate în pixeli."""
        return int(self.x * width), int(self.y * height)


@dataclass
class PoseFrame:
    """Rezultatul estimării de postură pentru un singur frame."""
    frame_index: int
    timestamp_ms: float
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    raw_landmarks: Optional[object] = field(default=None, repr=False)

    def get(self, name: str) -> Optional[Keypoint]:
        return self.keypoints.get(name)


# ── Constante: numele articulațiilor relevante din MediaPipe ──────────────────

RELEVANT_LANDMARKS = {
    # Umeri
    "LEFT_SHOULDER":  11,
    "RIGHT_SHOULDER": 12,
    # Coate
    "LEFT_ELBOW":     13,
    "RIGHT_ELBOW":    14,
    # Încheieturi mâini
    "LEFT_WRIST":     15,
    "RIGHT_WRIST":    16,
    # Șolduri
    "LEFT_HIP":       23,
    "RIGHT_HIP":      24,
    # Genunchi
    "LEFT_KNEE":      25,
    "RIGHT_KNEE":     26,
    # Glezne
    "LEFT_ANKLE":     27,
    "RIGHT_ANKLE":    28,
    # Picioare (vârf)
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# Conexiuni pentru desenarea scheletului
SKELETON_CONNECTIONS = [
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER",  "LEFT_ELBOW"),
    ("LEFT_ELBOW",     "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW",    "RIGHT_WRIST"),
    ("LEFT_SHOULDER",  "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP",       "RIGHT_HIP"),
    ("LEFT_HIP",       "LEFT_KNEE"),
    ("LEFT_KNEE",      "LEFT_ANKLE"),
    ("RIGHT_HIP",      "RIGHT_KNEE"),
    ("RIGHT_KNEE",     "RIGHT_ANKLE"),
    ("LEFT_ANKLE",     "LEFT_FOOT_INDEX"),
    ("RIGHT_ANKLE",    "RIGHT_FOOT_INDEX"),
]


# ── Clasa principală ──────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Wrapper peste MediaPipe Pose pentru extragerea keypoints-urilor
    din frame-uri video.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,       # 0=Lite, 1=Full, 2=Heavy
    ):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Versiunea instalată de mediapipe nu expune API-ul 'mp.solutions'. "
                "Instalează o versiune compatibilă, de ex.: "
                "'python -m pip install --upgrade --force-reinstall mediapipe==0.10.14'"
            )

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_ms: float) -> Optional[PoseFrame]:
        """
        Procesează un singur frame BGR și returnează un PoseFrame
        cu keypoints-urile detectate, sau None dacă nu s-a detectat nicio persoană.
        """
        # MediaPipe lucrează cu RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        keypoints = {}

        for name, idx in RELEVANT_LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            keypoints[name] = Keypoint(
                name=name,
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            keypoints=keypoints,
            raw_landmarks=results.pose_landmarks,
        )

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose_frame: PoseFrame,
        color_joint: tuple = (0, 255, 128),
        color_bone:  tuple = (255, 200, 0),
        joint_radius: int = 6,
        bone_thickness: int = 2,
    ) -> np.ndarray:
        """
        Desenează scheletul (articulații + conexiuni) pe frame-ul primit
        și returnează frame-ul adnotat (copie).
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # Desenează conexiunile (oasele)
        for start_name, end_name in SKELETON_CONNECTIONS:
            kp_start = pose_frame.get(start_name)
            kp_end   = pose_frame.get(end_name)
            if kp_start and kp_end:
                if kp_start.visibility > 0.3 and kp_end.visibility > 0.3:
                    pt1 = kp_start.to_pixel(w, h)
                    pt2 = kp_end.to_pixel(w, h)
                    cv2.line(output, pt1, pt2, color_bone, bone_thickness, cv2.LINE_AA)

        # Desenează articulațiile (cercuri)
        for kp in pose_frame.keypoints.values():
            if kp.visibility > 0.3:
                px, py = kp.to_pixel(w, h)
                cv2.circle(output, (px, py), joint_radius, color_joint, -1, cv2.LINE_AA)
                cv2.circle(output, (px, py), joint_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        return output

    def close(self):
        """Eliberează resursele MediaPipe."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()