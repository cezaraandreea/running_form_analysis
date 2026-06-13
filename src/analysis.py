"""
src/analysis.py
Analizează parametrii biomecanici și detectează probleme de tehnică.
Generează feedback automat și un scor general.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

from src.biomechanics import BiomechanicsFrame
from src.gait_analysis import GaitAnalysisResult


# ── Tipuri de feedback ────────────────────────────────────────────────────────

class Severity(Enum):
    OK      = "ok"
    WARNING = "warning"
    ERROR   = "error"


@dataclass
class FeedbackItem:
    """Un item de feedback pentru o problemă detectată."""
    category:    str        # ex: "Genunchi", "Trunchi", "Brațe"
    message:     str        # mesajul afișat utilizatorului
    severity:    Severity
    detail:      str = ""   # explicație tehnică
    value:       Optional[float] = None   # valoarea măsurată


# ── Praguri biomecanice ───────────────────────────────────────────────────────
# Valorile sunt orientative; pot fi ajustate pe baza literaturii de specialitate.

class Thresholds:
    # Unghi genunchi la contactul cu solul (ideal 140-170°)
    KNEE_ANGLE_MIN = 140.0      # sub acest unghi = flexie excesivă
    KNEE_ANGLE_MAX = 175.0      # peste acest unghi = genunchi prea drept

    # Înclinarea trunchiului față de verticală (ideal 5-10°)
    TRUNK_LEAN_MIN = 3.0        # sub = prea drept / aplecat pe spate
    TRUNK_LEAN_MAX = 15.0       # peste = aplecat excesiv înainte

    # Simetrie la contact (comparatie L vs R la evenimente echivalente)
    PHASE_CONTACT_SYMMETRY_MAX = 15.0

    # Overstriding: offset gleznă față de șold (normalizat, ideal ≤ 0)
    OVERSTRIDE_THRESHOLD = 0.05  # peste = pas prea lung

    # Unghi cot (ideal 85-95°)
    ELBOW_MIN = 70.0
    ELBOW_MAX = 110.0

    # Unghi genunchi la ridicare (knee drive); unghi mai mic = genunchi mai sus
    KNEE_DRIVE_ANGLE_MAX = 145.0   # peste = ridicare insuficienta a genunchiului
    KNEE_DRIVE_SYMMETRY_MAX = 15.0

    # Tipul de contact cu solul
    HEEL_STRIKE_PCT_WARNING = 60.0  # % contacte pe calcan — peste = avertizare

    # Cadenta (pasi pe minut) — valori implicite (barbati)
    CADENCE_LOW_ERROR   = 150.0
    CADENCE_LOW_WARN    = 165.0
    CADENCE_HIGH        = 185.0

    # Cadenta specifica femeilor (talie mai mica → rotatie mai rapida)
    CADENCE_LOW_ERROR_F = 155.0
    CADENCE_LOW_WARN_F  = 170.0
    CADENCE_HIGH_F      = 190.0

    # Oscilatie verticala (cm)
    VERTICAL_OSC_WARN_CM = 8.0   # peste = energie risipita
    VERTICAL_OSC_HIGH_CM = 10.0  # peste = problema semnificativa


def _heel_pct(strikes: list) -> Optional[float]:
    typed = [e for e in strikes if e.strike_type is not None]
    if not typed:
        return None
    return sum(1 for e in typed if e.strike_type == "heel") / len(typed) * 100.0


def _leg_ratio(runner_sex: str) -> float:
    """Raportul lungime picior / inaltime corporala, specific sexului."""
    return 0.49 if runner_sex == "Masculin" else 0.46


def _vo_cm(gait: Optional[GaitAnalysisResult], height_cm: int, runner_sex: str) -> Optional[float]:
    if not gait or gait.vertical_oscillation_norm is None or gait.leg_length_normalized is None:
        return None
    leg_len_m = height_cm * _leg_ratio(runner_sex) / 100
    scale_m = leg_len_m / gait.leg_length_normalized
    return gait.vertical_oscillation_norm * scale_m * 100


# ── Clasa de analiză ──────────────────────────────────────────────────────────

class RunningAnalyzer:
    """
    Analizează o secvență de BiomechanicsFrame-uri și produce:
    - lista de probleme detectate
    - statistici agregate
    - scor general (0–100)
    - feedback text complet
    """

    def analyze_sequence(
        self,
        frames: list[BiomechanicsFrame],
        gait: Optional[GaitAnalysisResult] = None,
        runner_height_cm: int = 175,
        runner_sex: str = "Masculin",
    ) -> dict:
        """
        Primește lista tuturor frame-urilor procesate și returnează
        un dict complet cu rezultatele analizei.
        """
        if not frames:
            return {}

        stats   = self._compute_statistics(frames, gait, runner_height_cm, runner_sex)
        issues  = self._detect_issues(stats, runner_sex)
        score   = self._compute_score(issues, stats)
        summary = self._generate_summary(issues, score)

        return {
            "statistics":  stats,
            "issues":      issues,
            "score":       score,
            "summary":     summary,
            "frame_count": len(frames),
        }

    # ── Statistici ────────────────────────────────────────────────────────────

    def _compute_statistics(
        self,
        frames: list[BiomechanicsFrame],
        gait: Optional[GaitAnalysisResult] = None,
        runner_height_cm: int = 175,
        runner_sex: str = "Masculin",
    ) -> dict:
        """Calculează medii, min, max și std pentru fiecare parametru."""

        def safe_mean(vals):
            v = [x for x in vals if x is not None]
            return float(np.mean(v)) if v else None

        def safe_std(vals):
            v = [x for x in vals if x is not None]
            return float(np.std(v)) if v else None

        knee_l  = [f.knee_angle_left   for f in frames]
        knee_r  = [f.knee_angle_right  for f in frames]
        trunk   = [f.trunk_lean_angle  for f in frames]
        elbow_l = [f.elbow_angle_left  for f in frames]
        elbow_r = [f.elbow_angle_right for f in frames]
        sym_inst = [f.knee_symmetry_diff for f in frames]
        os_l    = [f.foot_strike_offset_left  for f in frames]
        os_r    = [f.foot_strike_offset_right for f in frames]

        return {
            "knee_angle_left_mean":    safe_mean(knee_l),
            "knee_angle_right_mean":   safe_mean(knee_r),
            "knee_angle_left_std":     safe_std(knee_l),
            "knee_angle_right_std":    safe_std(knee_r),
            "trunk_lean_mean":         safe_mean(trunk),
            "trunk_lean_std":          safe_std(trunk),
            "elbow_angle_left_mean":   safe_mean(elbow_l),
            "elbow_angle_right_mean":  safe_mean(elbow_r),
            "symmetry_diff_mean":      safe_mean(sym_inst),
            "phase_symmetry_contact_deg": (
                gait.phase_symmetry_contact_deg if gait else None
            ),
            "left_contact_knee_mean": (
                gait.left_contact_knee_mean_deg if gait else None
            ),
            "right_contact_knee_mean": (
                gait.right_contact_knee_mean_deg if gait else None
            ),
            # overstride: folosim mediile de la evenimentele de contact (mai precise)
            "overstride_left_mean":  (gait.left_overstride_mean  if gait else safe_mean(os_l)),
            "overstride_right_mean": (gait.right_overstride_mean if gait else safe_mean(os_r)),
            "step_length_mean":        safe_mean([f.step_length for f in frames]),
            "knee_rom_proxy_mean":     safe_mean([f.knee_rom_proxy for f in frames]),
            "knee_drive_symmetry_deg": (gait.knee_drive_symmetry_deg   if gait else None),
            "left_drive_knee_mean":    (gait.left_drive_knee_mean_deg  if gait else None),
            "right_drive_knee_mean":   (gait.right_drive_knee_mean_deg if gait else None),
            # Tipul de contact: % heel strike per picior
            "heel_strike_pct_left":  _heel_pct(gait.left_strikes  if gait else []),
            "heel_strike_pct_right": _heel_pct(gait.right_strikes if gait else []),
            # Cadenta si oscilatie verticala
            "cadence_spm":            (gait.cadence_spm if gait else None),
            "vertical_oscillation_cm": _vo_cm(gait, runner_height_cm, runner_sex),
            "runner_sex":              runner_sex,
        }

    # ── Detecție probleme ─────────────────────────────────────────────────────

    def _detect_issues(self, stats: dict, runner_sex: str = "Masculin") -> list[FeedbackItem]:
        issues = []
        T = Thresholds
        female = runner_sex == "Feminin"

        # --- Genunchi ---
        for side, key in [("stâng", "knee_angle_left_mean"), ("drept", "knee_angle_right_mean")]:
            val = stats.get(key)
            if val is not None:
                if val < T.KNEE_ANGLE_MIN:
                    issues.append(FeedbackItem(
                        category="Genunchi",
                        message=f"Flexie excesivă a genunchiului {side} ({val:.1f}°)",
                        severity=Severity.WARNING,
                        detail="Unghiul genunchiului este prea mic. Relaxează piciorul la contactul cu solul.",
                        value=val,
                    ))
                elif val > T.KNEE_ANGLE_MAX:
                    issues.append(FeedbackItem(
                        category="Genunchi",
                        message=f"Genunchiul {side} prea rigid la contact ({val:.1f}°)",
                        severity=Severity.WARNING,
                        detail="Menține o ușoară flexie a genunchiului pentru absorbția șocului.",
                        value=val,
                    ))
                else:
                    issues.append(FeedbackItem(
                        category="Genunchi",
                        message=f"Unghi genunchi {side} corect ({val:.1f}°)",
                        severity=Severity.OK,
                        value=val,
                    ))

        # --- Trunchi ---
        trunk = stats.get("trunk_lean_mean")
        if trunk is not None:
            if trunk < T.TRUNK_LEAN_MIN:
                issues.append(FeedbackItem(
                    category="Trunchi",
                    message=f"Trunchi prea vertical sau aplecat pe spate ({trunk:.1f}°)",
                    severity=Severity.WARNING,
                    detail="O ușoară înclinare înainte (5-10°) îmbunătățește eficiența alergării.",
                    value=trunk,
                ))
            elif trunk > T.TRUNK_LEAN_MAX:
                issues.append(FeedbackItem(
                    category="Trunchi",
                    message=f"Înclinare excesivă a trunchiului înainte ({trunk:.1f}°)",
                    severity=Severity.ERROR,
                    detail="Excesul de aplecare poate cauza suprasolicitarea spatelui. Ridică privirea.",
                    value=trunk,
                ))
            else:
                issues.append(FeedbackItem(
                    category="Trunchi",
                    message=f"Postură corectă a trunchiului ({trunk:.1f}°)",
                    severity=Severity.OK,
                    value=trunk,
                ))

        # --- Simetrie la contact (nu folosim diferența instantanee per frame) ---
        sym = stats.get("phase_symmetry_contact_deg")
        if sym is not None:
            if sym > T.PHASE_CONTACT_SYMMETRY_MAX:
                issues.append(FeedbackItem(
                    category="Simetrie",
                    message=f"Asimetrie la contact cu solul ({sym:.1f}°)",
                    severity=Severity.WARNING,
                    detail="Diferența medie a unghiului genunchiului la contact stânga vs. dreapta depășește pragul orientativ de 15°.",
                    value=sym,
                ))
            else:
                issues.append(FeedbackItem(
                    category="Simetrie",
                    message=f"Simetrie bună la contact ({sym:.1f}°)",
                    severity=Severity.OK,
                    value=sym,
                ))

        # --- Simetrie la ridicarea genunchiului (knee drive) ---
        drive_sym = stats.get("knee_drive_symmetry_deg")
        if drive_sym is not None:
            if drive_sym > T.KNEE_DRIVE_SYMMETRY_MAX:
                issues.append(FeedbackItem(
                    category="Simetrie",
                    message=f"Asimetrie la ridicarea genunchiului ({drive_sym:.1f}°)",
                    severity=Severity.WARNING,
                    detail="Diferența medie a unghiului genunchiului la ridicare stânga vs. dreapta depășește 15°. Încearcă să ridici ambii genunchi la același nivel.",
                    value=drive_sym,
                ))
            else:
                issues.append(FeedbackItem(
                    category="Simetrie",
                    message=f"Simetrie bună la ridicarea genunchiului ({drive_sym:.1f}°)",
                    severity=Severity.OK,
                    value=drive_sym,
                ))

        # --- Ridicare genunchi (unghi la knee drive) ---
        for side, key in [("stâng", "left_drive_knee_mean"), ("drept", "right_drive_knee_mean")]:
            val = stats.get(key)
            if val is not None and val > T.KNEE_DRIVE_ANGLE_MAX:
                issues.append(FeedbackItem(
                    category="Genunchi",
                    message=f"Ridicare insuficientă a genunchiului {side} la pas ({val:.1f}°)",
                    severity=Severity.WARNING,
                    detail="Genunchiul nu este suficient de flexat la ridicare. Ridică mai mult genunchiul în faza de swing pentru o propulsie mai eficientă.",
                    value=val,
                ))

        # --- Tipul de contact cu solul (heel / midfoot / forefoot) ---
        for side, key in [("stâng", "heel_strike_pct_left"), ("drept", "heel_strike_pct_right")]:
            pct = stats.get(key)
            if pct is not None:
                if pct >= T.HEEL_STRIKE_PCT_WARNING:
                    issues.append(FeedbackItem(
                        category="Contact",
                        message=f"Aterizare predominant pe călcâi — piciorul {side} ({pct:.0f}% heel strike)",
                        severity=Severity.WARNING,
                        detail="Aterizarea pe călcâi crește forțele de impact asupra articulațiilor. Încearcă să aterizezi pe mijlocul tălpii sau pe vârf.",
                        value=pct,
                    ))
                else:
                    issues.append(FeedbackItem(
                        category="Contact",
                        message=f"Aterizare corectă — piciorul {side} ({pct:.0f}% heel strike)",
                        severity=Severity.OK,
                        value=pct,
                    ))

        # --- Overstriding ---
        for side, key in [("stâng", "overstride_left_mean"), ("drept", "overstride_right_mean")]:
            val = stats.get(key)
            if val is not None and val > T.OVERSTRIDE_THRESHOLD:
                issues.append(FeedbackItem(
                    category="Pas",
                    message=f"Pas prea lung detectat - piciorul {side}",
                    severity=Severity.WARNING,
                    detail="Aterizarea cu piciorul mult în fața corpului crește riscul de accidentare. Scurtează pasul.",
                    value=val,
                ))

        # --- Cadenta ---
        cadence = stats.get("cadence_spm")
        if cadence is not None:
            c_err  = T.CADENCE_LOW_ERROR_F if female else T.CADENCE_LOW_ERROR
            c_warn = T.CADENCE_LOW_WARN_F  if female else T.CADENCE_LOW_WARN
            c_high = T.CADENCE_HIGH_F      if female else T.CADENCE_HIGH
            ideal  = "170–185" if not female else "174–190"
            if cadence < c_err:
                issues.append(FeedbackItem(
                    category="Cadență",
                    message=f"Cadență foarte redusă ({cadence:.0f} pași/min)",
                    severity=Severity.ERROR,
                    detail=f"O cadență sub {c_err:.0f} pași/min indică pași prea lungi și ineficienți. Crește ritmul și scurtează pasul.",
                    value=cadence,
                ))
            elif cadence < c_warn:
                issues.append(FeedbackItem(
                    category="Cadență",
                    message=f"Cadență sub optim ({cadence:.0f} pași/min)",
                    severity=Severity.WARNING,
                    detail=f"Cadența ideală pentru {'femei' if female else 'bărbați'} este {ideal} pași/min. Încearcă să crești ritmul pașilor.",
                    value=cadence,
                ))
            else:
                label = "excelentă" if cadence >= c_high else "bună"
                issues.append(FeedbackItem(
                    category="Cadență",
                    message=f"Cadență {label} ({cadence:.0f} pași/min)",
                    severity=Severity.OK,
                    value=cadence,
                ))

        # --- Oscilatie verticala ---
        vo = stats.get("vertical_oscillation_cm")
        if vo is not None:
            if vo > T.VERTICAL_OSC_HIGH_CM:
                issues.append(FeedbackItem(
                    category="Oscilație verticală",
                    message=f"Oscilație verticală excesivă ({vo:.1f} cm)",
                    severity=Severity.WARNING,
                    detail="Un salt vertical mare (>10 cm) consumă energie inutil. Concentrează-te pe propulsie înainte, nu în sus.",
                    value=vo,
                ))
            elif vo > T.VERTICAL_OSC_WARN_CM:
                issues.append(FeedbackItem(
                    category="Oscilație verticală",
                    message=f"Oscilație verticală ridicată ({vo:.1f} cm)",
                    severity=Severity.WARNING,
                    detail="Oscilația verticală optimă este sub 8 cm. Încearcă să menții corpul mai jos și să aluneci înainte.",
                    value=vo,
                ))
            else:
                issues.append(FeedbackItem(
                    category="Oscilație verticală",
                    message=f"Oscilație verticală eficientă ({vo:.1f} cm)",
                    severity=Severity.OK,
                    value=vo,
                ))

        # --- Brațe ---
        for side, key in [("stâng", "elbow_angle_left_mean"), ("drept", "elbow_angle_right_mean")]:
            val = stats.get(key)
            if val is not None:
                if val < T.ELBOW_MIN or val > T.ELBOW_MAX:
                    issues.append(FeedbackItem(
                        category="Brațe",
                        message=f"Unghi incorect al cotului {side} ({val:.1f}°)",
                        severity=Severity.WARNING,
                        detail="Menține cotul la ~90° pentru un balans eficient al brațelor.",
                        value=val,
                    ))

        return issues

    # ── Scor general ──────────────────────────────────────────────────────────

    def _compute_score(self, issues: list[FeedbackItem], stats: dict) -> float:
        """
        Calculează un scor 0–100 pe baza problemelor detectate.
        100 = tehnică perfectă, 0 = probleme severe multiple.
        """
        score = 100.0
        for item in issues:
            if item.severity == Severity.ERROR:
                score -= 15.0
            elif item.severity == Severity.WARNING:
                score -= 7.0
        return max(0.0, min(100.0, score))

    # ── Sumar text ────────────────────────────────────────────────────────────

    def _generate_summary(self, issues: list[FeedbackItem], score: float) -> str:
        errors   = [i for i in issues if i.severity == Severity.ERROR]
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        oks      = [i for i in issues if i.severity == Severity.OK]

        if score >= 85:
            overall = "✅ Tehnică de alergare excelentă!"
        elif score >= 65:
            overall = "⚠️ Tehnică bună cu câteva aspecte de îmbunătățit."
        else:
            overall = "❌ Sunt detectate probleme semnificative de tehnică."

        lines = [overall, ""]

        if errors:
            lines.append("**Probleme critice:**")
            for e in errors:
                lines.append(f"  🔴 {e.message}")
                if e.detail:
                    lines.append(f"     → {e.detail}")
            lines.append("")

        if warnings:
            lines.append("**Atenție:**")
            for w in warnings:
                lines.append(f"  🟡 {w.message}")
                if w.detail:
                    lines.append(f"     → {w.detail}")
            lines.append("")

        if oks:
            lines.append("**Aspecte pozitive:**")
            for o in oks:
                lines.append(f"  🟢 {o.message}")

        return "\n".join(lines)