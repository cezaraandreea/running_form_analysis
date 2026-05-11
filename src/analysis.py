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

    # Simetrie genunchi (diferența stânga/dreapta, ideal < 10°)
    SYMMETRY_MAX   = 15.0       # peste = dezechilibru semnificativ

    # Overstriding: offset gleznă față de șold (normalizat, ideal ≤ 0)
    OVERSTRIDE_THRESHOLD = 0.05  # peste = pas prea lung

    # Unghi cot (ideal 85-95°)
    ELBOW_MIN = 70.0
    ELBOW_MAX = 110.0


# ── Clasa de analiză ──────────────────────────────────────────────────────────

class RunningAnalyzer:
    """
    Analizează o secvență de BiomechanicsFrame-uri și produce:
    - lista de probleme detectate
    - statistici agregate
    - scor general (0–100)
    - feedback text complet
    """

    def analyze_sequence(self, frames: list[BiomechanicsFrame]) -> dict:
        """
        Primește lista tuturor frame-urilor procesate și returnează
        un dict complet cu rezultatele analizei.
        """
        if not frames:
            return {}

        stats   = self._compute_statistics(frames)
        issues  = self._detect_issues(stats)
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

    def _compute_statistics(self, frames: list[BiomechanicsFrame]) -> dict:
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
        sym     = [f.knee_symmetry_diff for f in frames]
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
            "symmetry_diff_mean":      safe_mean(sym),
            "overstride_left_mean":    safe_mean(os_l),
            "overstride_right_mean":   safe_mean(os_r),
        }

    # ── Detecție probleme ─────────────────────────────────────────────────────

    def _detect_issues(self, stats: dict) -> list[FeedbackItem]:
        issues = []
        T = Thresholds

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

        # --- Simetrie ---
        sym = stats.get("symmetry_diff_mean")
        if sym is not None:
            if sym > T.SYMMETRY_MAX:
                issues.append(FeedbackItem(
                    category="Simetrie",
                    message=f"Dezechilibru între membre detectat ({sym:.1f}°)",
                    severity=Severity.ERROR,
                    detail="Diferența semnificativă între piciorul stâng și drept poate indica o asimetrie musculară.",
                    value=sym,
                ))
            else:
                issues.append(FeedbackItem(
                    category="Simetrie",
                    message=f"Simetrie bună între picioare ({sym:.1f}°)",
                    severity=Severity.OK,
                    value=sym,
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