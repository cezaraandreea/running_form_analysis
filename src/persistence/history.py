"""
src/history.py
Salvarea si recuperarea istoricului analizelor per utilizator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.persistence.auth import _get_conn


def _derive_foot_strike(stats: dict) -> Optional[str]:
    """Determina tipul predominant de aterizare din procentele heel strike."""
    pcts = [
        v for v in [stats.get("heel_strike_pct_left"), stats.get("heel_strike_pct_right")]
        if v is not None
    ]
    if not pcts:
        return None
    avg = sum(pcts) / len(pcts)
    if avg >= 60:
        return "Calcan (heel)"
    if avg <= 20:
        return "Varf (forefoot)"
    return "Mijloc talpa (midfoot)"


def save_analysis(
    user_id: int,
    video_filename: str,
    analysis_result: dict,
    height_cm: int,
    runner_sex: str,
) -> int:
    """
    Salveaza un rezultat de analiza pentru utilizatorul autentificat.
    Returneaza id-ul randului nou creat.
    """
    from datetime import datetime

    stats    = analysis_result.get("statistics", {})
    score    = analysis_result.get("score")
    issues   = analysis_result.get("issues", [])

    cadence  = stats.get("cadence_spm")
    vo_cm    = stats.get("vertical_oscillation_cm")
    sym_deg  = stats.get("phase_symmetry_contact_deg")
    foot_str = _derive_foot_strike(stats)

    # Lungimea pasului nu e in statistics direct; o lasam None (e afisata in UI dar nu stocata ca atribut distinct)
    step_m: Optional[float] = None

    feedback_list = [
        {
            "category": f.category,
            "message":  f.message,
            "severity": f.severity.value,
            "detail":   f.detail or "",
        }
        for f in issues
    ]

    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses
              (user_id, created_at, video_filename, score, runner_height_cm, runner_sex,
               cadence_spm, vertical_oscillation_cm, step_length_m, foot_strike_type,
               symmetry_contact_deg, feedback_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                user_id,
                datetime.utcnow().isoformat(),
                Path(video_filename).name if video_filename else None,
                round(score) if score is not None else None,
                height_cm,
                runner_sex,
                round(cadence, 1)  if cadence is not None else None,
                round(vo_cm,  1)   if vo_cm   is not None else None,
                round(step_m, 2)   if step_m   is not None else None,
                foot_str,
                round(sym_deg, 1)  if sym_deg  is not None else None,
                json.dumps(feedback_list, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_user_analyses(user_id: int) -> list[dict]:
    """Returneaza toate analizele unui utilizator, cele mai recente primele."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_analysis_detail(analysis_id: int, user_id: int) -> Optional[dict]:
    """
    Returneaza detaliile unei analize specifice.
    Verifica ca analiza apartine utilizatorului (securitate).
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["feedback"] = json.loads(result.pop("feedback_json", "[]"))
    return result
