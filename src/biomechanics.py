"""
src/biomechanics.py
Calculul parametrilor biomecanici din keypoints-urile extrase.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.pose_estimation import PoseFrame, Keypoint


# ── Funcții geometrice de bază ────────────────────────────────────────────────

def angle_between_three_points(a: tuple, b: tuple, c: tuple) -> float:
    """
    Calculează unghiul ABC (în grade) unde B este vârful unghiului.
    a, b, c sunt tuple-uri (x, y) sau (x, y, z).
    """
    a = np.array(a[:2])   # folosim doar x, y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def angle_with_vertical(top: tuple, bottom: tuple) -> float:
    """
    Calculează unghiul unui segment față de axa verticală (în grade).
    0° = perfect vertical, 90° = orizontal.
    """
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]   # y crește în jos în coord. imagine
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return float(angle)


def euclidean_distance(a: tuple, b: tuple) -> float:
    """Distanța euclidiană 2D între două puncte normalizate."""
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def kp_to_tuple(kp: Optional[Keypoint]) -> Optional[tuple]:
    """Convertește un Keypoint la tuple (x, y, z) sau None."""
    if kp is None:
        return None
    return (kp.x, kp.y, kp.z)


# ── Dataclass pentru parametrii unui frame ────────────────────────────────────

@dataclass
class BiomechanicsFrame:
    """Toți parametrii biomecanici calculați pentru un frame."""
    frame_index: int
    timestamp_ms: float

    # Unghiuri genunchi [grade]
    knee_angle_left:  Optional[float] = None
    knee_angle_right: Optional[float] = None

    # Unghi trunchi față de verticală [grade]
    trunk_lean_angle: Optional[float] = None

    # Unghiuri coate [grade]
    elbow_angle_left:  Optional[float] = None
    elbow_angle_right: Optional[float] = None

    # Poziție gleznă față de șold (detecție overstriding)
    foot_strike_offset_left:  Optional[float] = None  # x_ankle - x_hip (normalizat)
    foot_strike_offset_right: Optional[float] = None

    # Înălțimea centrului de masă (șold mediu, normalizat)
    hip_height: Optional[float] = None

    # Diferență instantanee L-R (nu e simetrie de fază; picioarele sunt în faze diferite)
    knee_symmetry_diff: Optional[float] = None

    # Lungime pas aproximată (distanța dintre glezne, normalizat)
    step_length: Optional[float] = None

    # Amplitudine instantanee genunchi (diferența dintre unghiuri, normalizat la 180)
    knee_rom_proxy: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "frame_index":             self.frame_index,
            "timestamp_ms":            self.timestamp_ms,
            "knee_angle_left":         self.knee_angle_left,
            "knee_angle_right":        self.knee_angle_right,
            "trunk_lean_angle":        self.trunk_lean_angle,
            "elbow_angle_left":        self.elbow_angle_left,
            "elbow_angle_right":       self.elbow_angle_right,
            "foot_strike_offset_left": self.foot_strike_offset_left,
            "foot_strike_offset_right":self.foot_strike_offset_right,
            "hip_height":              self.hip_height,
            "knee_symmetry_diff":      self.knee_symmetry_diff,
            "step_length":             self.step_length,
            "knee_rom_proxy":          self.knee_rom_proxy,
        }


# ── Clasa principală ──────────────────────────────────────────────────────────

class BiomechanicsCalculator:
    """
    Primește un PoseFrame și calculează toți parametrii biomecanici relevanți.
    """

    def __init__(self, min_visibility: float = 0.2):
        self.min_visibility = min_visibility

    def calculate(self, pose_frame: PoseFrame) -> BiomechanicsFrame:
        kp = pose_frame.keypoints
        bf = BiomechanicsFrame(
            frame_index=pose_frame.frame_index,
            timestamp_ms=pose_frame.timestamp_ms,
        )

        # Helper local
        def get(name):
            point = kp.get(name)
            if point is None:
                return None
            if point.visibility < self.min_visibility:
                return None
            return kp_to_tuple(point)

        # ── Unghi genunchi stâng: șold → genunchi → gleznă ───────────────────
        lh = get("LEFT_HIP");    lk = get("LEFT_KNEE");  la = get("LEFT_ANKLE")
        if all(v is not None for v in [lh, lk, la]):
            bf.knee_angle_left = angle_between_three_points(lh, lk, la)

        # ── Unghi genunchi drept ──────────────────────────────────────────────
        rh = get("RIGHT_HIP");   rk = get("RIGHT_KNEE"); ra = get("RIGHT_ANKLE")
        if all(v is not None for v in [rh, rk, ra]):
            bf.knee_angle_right = angle_between_three_points(rh, rk, ra)

        # ── Diferență instantanee unghi genunchi L-R (per frame) ──────────────
        if bf.knee_angle_left is not None and bf.knee_angle_right is not None:
            bf.knee_symmetry_diff = abs(bf.knee_angle_left - bf.knee_angle_right)
            bf.knee_rom_proxy = bf.knee_symmetry_diff / 180.0

        # ── Unghi trunchi față de verticală: umăr mediu → șold mediu ─────────
        ls = get("LEFT_SHOULDER"); rs = get("RIGHT_SHOULDER")
        if all(v is not None for v in [ls, rs, lh, rh]):
            shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            hip_mid      = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            bf.trunk_lean_angle = angle_with_vertical(shoulder_mid, hip_mid)
            bf.hip_height = hip_mid[1]   # y normalizat (0=sus, 1=jos)

        # ── Unghi cot stâng: umăr → cot → încheietură ────────────────────────
        le = get("LEFT_ELBOW");  lw = get("LEFT_WRIST")
        if all(v is not None for v in [ls, le, lw]):
            bf.elbow_angle_left = angle_between_three_points(ls, le, lw)

        # ── Unghi cot drept ───────────────────────────────────────────────────
        re = get("RIGHT_ELBOW"); rw = get("RIGHT_WRIST")
        if all(v is not None for v in [rs, re, rw]):
            bf.elbow_angle_right = angle_between_three_points(rs, re, rw)

        # ── Overstriding: distanța orizontală gleznă față de șold ────────────
        # Dacă glezna e mult în fața șoldului → overstride
        if lh is not None and la is not None:
            bf.foot_strike_offset_left  = la[0] - lh[0]   # pozitiv = piciorul în față
        if rh is not None and ra is not None:
            bf.foot_strike_offset_right = ra[0] - rh[0]

        # ── Lungime pas aproximată: distanța 2D dintre glezne ────────────────
        if la is not None and ra is not None:
            bf.step_length = euclidean_distance(la, ra)

        return bf