"""
Detectie faze de mers/alergare si simetrie la evenimente echivalente (contact cu solul si ridicare genunchi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.pipeline.biomechanics import BiomechanicsFrame
from src.pipeline.pose_estimation import PoseFrame

# Prag pentru simetrie la contact (comparatie L vs R la acelasi tip de eveniment)
PHASE_CONTACT_SYMMETRY_MAX = 15.0

# Prag pentru simetrie la ridicarea genunchiului (knee drive)
KNEE_DRIVE_SYMMETRY_MAX = 15.0

# Diferenta minima calcan vs varf (coord. normalizate) pentru a clasifica tipul de contact
FOOT_STRIKE_DELTA_THRESHOLD = 0.03


@dataclass
class FootStrikeEvent:
    frame_index: int
    timestamp_ms: float
    side: str
    knee_angle: Optional[float]
    ankle_y: float
    foot_strike_offset: Optional[float] = None  # x_ankle - x_hip; pozitiv = picior in fata soldului
    strike_type: Optional[str] = None           # "heel" | "midfoot" | "forefoot"
    step_length: Optional[float] = None         # distanta 2D intre glezne la momentul contactului


@dataclass
class KneeDriveEvent:
    """Eveniment de ridicare a genunchiului (faza de swing, genunchiul la punctul cel mai sus)."""
    frame_index: int
    timestamp_ms: float
    side: str
    knee_angle: Optional[float]
    knee_y: float  # y minim = genunchi cel mai sus in cadru


@dataclass
class GaitAnalysisResult:
    left_strikes: list[FootStrikeEvent]
    right_strikes: list[FootStrikeEvent]
    phase_symmetry_contact_deg: Optional[float]
    left_contact_knee_mean_deg: Optional[float]
    right_contact_knee_mean_deg: Optional[float]
    # Knee drive (ridicare genunchi)
    left_knee_drives: list[KneeDriveEvent] = field(default_factory=list)
    right_knee_drives: list[KneeDriveEvent] = field(default_factory=list)
    knee_drive_symmetry_deg: Optional[float] = None
    left_drive_knee_mean_deg: Optional[float] = None
    right_drive_knee_mean_deg: Optional[float] = None
    # Overstride la contact (media offset-ului gleznă-șold per picior)
    left_overstride_mean: Optional[float] = None
    right_overstride_mean: Optional[float] = None
    # Lungime pas la contact per picior
    left_step_length_mean: Optional[float] = None
    right_step_length_mean: Optional[float] = None
    # Lungimea medie sold-glezna la contact (coord. normalizate) — folosita ca referinta de scala
    leg_length_normalized: Optional[float] = None
    # Cadenta (pasi pe minut)
    cadence_spm: Optional[float] = None
    # Oscilatie verticala (amplitudine normalizata 0-1 a miscarilor soldului)
    vertical_oscillation_norm: Optional[float] = None

    @property
    def strike_count(self) -> tuple[int, int]:
        return len(self.left_strikes), len(self.right_strikes)

    @property
    def drive_count(self) -> tuple[int, int]:
        return len(self.left_knee_drives), len(self.right_knee_drives)


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _pick_peaks(y: np.ndarray, min_distance: int, min_prominence: float) -> list[int]:
    """Indici locali maximi (gleznă jos în cadru = y mare)."""
    candidates: list[tuple[int, float]] = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_prominence:
            candidates.append((i, float(y[i])))

    candidates.sort(key=lambda x: -x[1])
    selected: list[int] = []
    for idx, _ in candidates:
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)
    return sorted(selected)


def _pick_valleys(y: np.ndarray, min_distance: int, max_ceiling: float) -> list[int]:
    """Indici locali minimi (genunchi sus în cadru = y mic)."""
    candidates: list[tuple[int, float]] = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1] and y[i] <= max_ceiling:
            candidates.append((i, float(y[i])))

    candidates.sort(key=lambda x: x[1])  # cele mai joase valori y (cel mai sus in cadru) primele
    selected: list[int] = []
    for idx, _ in candidates:
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)
    return sorted(selected)


def _classify_strike(
    pose_frame: PoseFrame,
    heel_key: str,
    foot_index_key: str,
) -> Optional[str]:
    """
    Clasifica tipul de contact pe baza pozitiei calcanului vs varfului la momentul aterizarii.
    In coordonate imagine Y creste in jos, deci y mai mare = mai aproape de sol.
    delta = heel.y - foot_index.y:
      > +threshold  => calcaneu mai jos => heel strike
      < -threshold  => varf mai jos     => forefoot strike
      altfel        => midfoot strike
    """
    heel = pose_frame.keypoints.get(heel_key)
    toe  = pose_frame.keypoints.get(foot_index_key)
    if heel is None or toe is None:
        return None
    if heel.visibility < 0.2 or toe.visibility < 0.2:
        return None
    delta = heel.y - toe.y
    if delta > FOOT_STRIKE_DELTA_THRESHOLD:
        return "heel"
    if delta < -FOOT_STRIKE_DELTA_THRESHOLD:
        return "forefoot"
    return "midfoot"


def _detect_strikes_for_side(
    pose_frames: list[PoseFrame],
    bio_by_idx: dict[int, BiomechanicsFrame],
    side: str,
    ankle_key: str,
    knee_attr: str,
    offset_attr: str,
    heel_key: str,
    foot_index_key: str,
    min_distance: int,
) -> list[FootStrikeEvent]:
    # Index pose frames for O(1) lookup during classification
    pose_by_idx = {pf.frame_index: pf for pf in pose_frames}

    series: list[tuple[int, float, float]] = []
    for pf in pose_frames:
        ankle = pf.keypoints.get(ankle_key)
        if ankle is None or ankle.visibility < 0.2:
            continue
        series.append((pf.frame_index, pf.timestamp_ms, ankle.y))

    if len(series) < 8:
        return []

    ankle_ys = np.array([s[2] for s in series], dtype=float)
    smoothed = _smooth(ankle_ys, window=5)
    prominence = float(np.median(smoothed) + 0.015)

    peak_local = _pick_peaks(smoothed, min_distance=min_distance, min_prominence=prominence)
    events: list[FootStrikeEvent] = []
    for pi in peak_local:
        frame_idx, ts, ay = series[pi]
        bf  = bio_by_idx.get(frame_idx)
        pf  = pose_by_idx.get(frame_idx)
        knee     = getattr(bf, knee_attr,   None) if bf else None
        offset   = getattr(bf, offset_attr, None) if bf else None
        step_len = bf.step_length if bf else None
        stype    = _classify_strike(pf, heel_key, foot_index_key) if pf else None
        events.append(
            FootStrikeEvent(
                frame_index=frame_idx,
                timestamp_ms=ts,
                side=side,
                knee_angle=knee,
                ankle_y=ay,
                foot_strike_offset=offset,
                strike_type=stype,
                step_length=step_len,
            )
        )
    return events


def _detect_knee_drives_for_side(
    pose_frames: list[PoseFrame],
    bio_by_idx: dict[int, BiomechanicsFrame],
    side: str,
    knee_key: str,
    knee_attr: str,
    min_distance: int,
) -> list[KneeDriveEvent]:
    """
    Detecteaza momentele de flexie maxima a genunchiului in faza de swing (knee drive).
    Urmarim unghiul genunchiului (nu pozitia spatiala) si gasim minimele locale —
    unghi minim = genunchi cel mai flexat = momentul de drive maxim.
    180° = picior drept; unghi mic (ex. 90°) = genunchi ridicat si flexat bine.
    """
    series: list[tuple[int, float, float, float]] = []  # (frame_idx, ts, knee_angle, knee_y)
    for pf in pose_frames:
        knee_kp = pf.keypoints.get(knee_key)
        if knee_kp is None or knee_kp.visibility < 0.2:
            continue
        bf = bio_by_idx.get(pf.frame_index)
        angle = getattr(bf, knee_attr, None) if bf else None
        if angle is None:
            continue
        series.append((pf.frame_index, pf.timestamp_ms, angle, knee_kp.y))

    if len(series) < 8:
        return []

    angles = np.array([s[2] for s in series], dtype=float)
    smoothed = _smooth(angles, window=5)
    # Pragul: unghiul trebuie sa fie mai mic decat quartila 40 ca sa fie considerat "flexat"
    ceiling = float(np.percentile(smoothed, 40))

    valley_local = _pick_valleys(smoothed, min_distance=min_distance, max_ceiling=ceiling)
    events: list[KneeDriveEvent] = []
    for vi in valley_local:
        frame_idx, ts, angle, ky = series[vi]
        events.append(
            KneeDriveEvent(
                frame_index=frame_idx,
                timestamp_ms=ts,
                side=side,
                knee_angle=angle,
                knee_y=ky,
            )
        )
    return events


def _avg_leg_length_normalized(
    strikes: list[FootStrikeEvent],
    pose_by_idx: dict[int, PoseFrame],
    hip_key: str,
    ankle_key: str,
) -> Optional[float]:
    """Distanta medie sold-glezna la contactele detectate (coord. normalizate 0-1)."""
    lengths = []
    for event in strikes:
        pf = pose_by_idx.get(event.frame_index)
        if pf is None:
            continue
        hip   = pf.keypoints.get(hip_key)
        ankle = pf.keypoints.get(ankle_key)
        if hip and ankle and hip.visibility > 0.2 and ankle.visibility > 0.2:
            d = float(np.sqrt((ankle.x - hip.x) ** 2 + (ankle.y - hip.y) ** 2))
            lengths.append(d)
    return float(np.mean(lengths)) if lengths else None


def _estimate_cadence_spm(bio_frames: list[BiomechanicsFrame]) -> Optional[float]:
    """
    Estimeaza cadenta din FFT-ul semnalului de unghi al genunchiului.
    Unghiul unui genunchi face un ciclu complet per stridă (contact stâng → contact stâng).
    Frecvența dominantă = frecvența stride-ului; cadența = freq_stride × 2 × 60.
    Intervalul căutat: 0.67–1.83 Hz = 80–220 pași/min.
    """
    estimates: list[float] = []
    for attr in ("knee_angle_left", "knee_angle_right"):
        pairs = [
            (f.timestamp_ms, getattr(f, attr))
            for f in bio_frames
            if getattr(f, attr) is not None
        ]
        if len(pairs) < 30:
            continue

        times_ms = np.array([p[0] for p in pairs], dtype=float)
        values   = np.array([p[1] for p in pairs], dtype=float)

        duration_s = (times_ms[-1] - times_ms[0]) / 1000.0
        if duration_s < 1.5:
            continue

        effective_fps = (len(values) - 1) / duration_s
        values = values - np.mean(values)

        spectrum = np.abs(np.fft.rfft(values))
        freqs    = np.fft.rfftfreq(len(values), d=1.0 / effective_fps)

        # Frecventa stride-ului pentru 80–220 pasi/min: 0.67–1.83 Hz
        mask = (freqs >= 0.67) & (freqs <= 1.83)
        if not np.any(mask):
            continue

        stride_freq = float(freqs[mask][int(np.argmax(spectrum[mask]))])
        estimates.append(stride_freq * 2.0 * 60.0)

    if not estimates:
        return None
    return float(np.mean(estimates))


def analyze_gait_phases(
    pose_frames: list[PoseFrame],
    bio_frames: list[BiomechanicsFrame],
    fps: float = 30.0,
) -> GaitAnalysisResult:
    """
    Detecteaza contacte estimate (gleznă la punctul cel mai jos) si ridicari de genunchi
    (genunchi la punctul cel mai sus) per picior, si calculeaza simetria la ambele faze.
    """
    if not pose_frames or not bio_frames:
        return GaitAnalysisResult([], [], None, None, None)

    bio_by_idx = {bf.frame_index: bf for bf in bio_frames}
    # ~0.25s intre contacte pe acelasi picior (ajustat la fps efectiv)
    min_distance = max(4, int(fps * 0.22))

    left = _detect_strikes_for_side(
        pose_frames, bio_by_idx, "left", "LEFT_ANKLE", "knee_angle_left",
        "foot_strike_offset_left", "LEFT_HEEL", "LEFT_FOOT_INDEX", min_distance
    )
    right = _detect_strikes_for_side(
        pose_frames, bio_by_idx, "right", "RIGHT_ANKLE", "knee_angle_right",
        "foot_strike_offset_right", "RIGHT_HEEL", "RIGHT_FOOT_INDEX", min_distance
    )

    left_angles = [e.knee_angle for e in left if e.knee_angle is not None]
    right_angles = [e.knee_angle for e in right if e.knee_angle is not None]

    left_mean = float(np.mean(left_angles)) if left_angles else None
    right_mean = float(np.mean(right_angles)) if right_angles else None

    phase_sym = None
    if left_mean is not None and right_mean is not None:
        phase_sym = abs(left_mean - right_mean)

    left_offsets = [e.foot_strike_offset for e in left if e.foot_strike_offset is not None]
    right_offsets = [e.foot_strike_offset for e in right if e.foot_strike_offset is not None]
    left_overstride_mean = float(np.mean(left_offsets)) if left_offsets else None
    right_overstride_mean = float(np.mean(right_offsets)) if right_offsets else None

    left_step_lens = [e.step_length for e in left if e.step_length is not None]
    right_step_lens = [e.step_length for e in right if e.step_length is not None]
    left_step_length_mean = float(np.mean(left_step_lens)) if left_step_lens else None
    right_step_length_mean = float(np.mean(right_step_lens)) if right_step_lens else None

    pose_by_idx = {pf.frame_index: pf for pf in pose_frames}
    leg_lens = [
        v for v in [
            _avg_leg_length_normalized(left,  pose_by_idx, "LEFT_HIP",  "LEFT_ANKLE"),
            _avg_leg_length_normalized(right, pose_by_idx, "RIGHT_HIP", "RIGHT_ANKLE"),
        ] if v is not None
    ]
    leg_length_normalized = float(np.mean(leg_lens)) if leg_lens else None

    # --- Cadence via FFT on hip height signal (robust vs. contact counting) ---
    cadence_spm: Optional[float] = _estimate_cadence_spm(bio_frames)

    # --- Vertical oscillation from hip height signal ---
    hip_heights = [f.hip_height for f in bio_frames if f.hip_height is not None]
    vertical_oscillation_norm: Optional[float] = None
    if len(hip_heights) >= 10:
        arr_h = np.array(hip_heights, dtype=float)
        vertical_oscillation_norm = float(np.percentile(arr_h, 90) - np.percentile(arr_h, 10))

    # --- Knee drive detection ---
    left_drives = _detect_knee_drives_for_side(
        pose_frames, bio_by_idx, "left", "LEFT_KNEE", "knee_angle_left", min_distance
    )
    right_drives = _detect_knee_drives_for_side(
        pose_frames, bio_by_idx, "right", "RIGHT_KNEE", "knee_angle_right", min_distance
    )

    left_drive_angles = [e.knee_angle for e in left_drives if e.knee_angle is not None]
    right_drive_angles = [e.knee_angle for e in right_drives if e.knee_angle is not None]

    left_drive_mean = float(np.mean(left_drive_angles)) if left_drive_angles else None
    right_drive_mean = float(np.mean(right_drive_angles)) if right_drive_angles else None

    drive_sym = None
    if left_drive_mean is not None and right_drive_mean is not None:
        drive_sym = abs(left_drive_mean - right_drive_mean)

    return GaitAnalysisResult(
        left_strikes=left,
        right_strikes=right,
        phase_symmetry_contact_deg=phase_sym,
        left_contact_knee_mean_deg=left_mean,
        right_contact_knee_mean_deg=right_mean,
        left_knee_drives=left_drives,
        right_knee_drives=right_drives,
        knee_drive_symmetry_deg=drive_sym,
        left_drive_knee_mean_deg=left_drive_mean,
        right_drive_knee_mean_deg=right_drive_mean,
        left_overstride_mean=left_overstride_mean,
        right_overstride_mean=right_overstride_mean,
        left_step_length_mean=left_step_length_mean,
        right_step_length_mean=right_step_length_mean,
        leg_length_normalized=leg_length_normalized,
        cadence_spm=cadence_spm,
        vertical_oscillation_norm=vertical_oscillation_norm,
    )
