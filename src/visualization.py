"""
src/visualization.py
Generarea graficelor și vizualizărilor pentru rezultatele analizei.
Folosește matplotlib și plotly pentru grafice interactive.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.biomechanics import BiomechanicsFrame


# ── Culori consistente ────────────────────────────────────────────────────────
COLOR_LEFT  = "#00C896"   # verde
COLOR_RIGHT = "#FF6B35"   # portocaliu
COLOR_OK    = "#2ECC71"
COLOR_WARN  = "#F39C12"
COLOR_ERR   = "#E74C3C"
BG_DARK     = "#0F1117"


# ── Grafice Plotly (interactive, pentru Streamlit) ────────────────────────────

def plot_knee_angles(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    """Grafic temporal al unghiurilor genunchilor."""
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    left  = [f.knee_angle_left  for f in bio_frames]
    right = [f.knee_angle_right for f in bio_frames]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps, y=left,
        mode="lines", name="Genunchi stâng",
        line=dict(color=COLOR_LEFT, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=right,
        mode="lines", name="Genunchi drept",
        line=dict(color=COLOR_RIGHT, width=2),
    ))

    # Zone de referință (unghi normal)
    fig.add_hrect(y0=140, y1=175, fillcolor="rgba(46,204,113,0.1)",
                  line_width=0, annotation_text="Zonă normală")

    fig.update_layout(
        title="Unghiuri genunchi în timp",
        xaxis_title="Timp (s)",
        yaxis_title="Unghi (°)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_trunk_lean(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    """Grafic al înclinării trunchiului."""
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    trunk = [f.trunk_lean_angle for f in bio_frames]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=trunk,
        mode="lines", name="Înclinare trunchi",
        line=dict(color="#9B59B6", width=2),
        fill="tozeroy",
        fillcolor="rgba(155,89,182,0.15)",
    ))
    fig.add_hrect(y0=3, y1=15, fillcolor="rgba(46,204,113,0.1)",
                  line_width=0, annotation_text="Ideal 3–15°")

    fig.update_layout(
        title="Înclinarea trunchiului față de verticală",
        xaxis_title="Timp (s)",
        yaxis_title="Unghi (°)",
        template="plotly_dark",
    )
    return fig


def plot_symmetry(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    """Grafic de simetrie stânga/dreapta."""
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    sym = [f.knee_symmetry_diff for f in bio_frames]

    colors = [COLOR_OK if (s or 0) < 10 else COLOR_WARN if (s or 0) < 15 else COLOR_ERR
              for s in sym]

    fig = go.Figure(go.Bar(
        x=timestamps,
        y=sym,
        marker_color=colors,
        name="Diferență simetrie",
    ))
    fig.add_hline(y=15, line_dash="dash", line_color=COLOR_ERR,
                  annotation_text="Prag problematic")

    fig.update_layout(
        title="Simetria genunchilor (diferența stânga–dreapta)",
        xaxis_title="Timp (s)",
        yaxis_title="Diferență (°)",
        template="plotly_dark",
    )
    return fig


def plot_elbow_angles(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    """Grafic unghiuri coate – balans brațe."""
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    left  = [f.elbow_angle_left  for f in bio_frames]
    right = [f.elbow_angle_right for f in bio_frames]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=left,  mode="lines",
                             name="Cot stâng",  line=dict(color=COLOR_LEFT)))
    fig.add_trace(go.Scatter(x=timestamps, y=right, mode="lines",
                             name="Cot drept", line=dict(color=COLOR_RIGHT)))
    fig.add_hrect(y0=70, y1=110, fillcolor="rgba(46,204,113,0.1)",
                  line_width=0, annotation_text="Ideal 70–110°")

    fig.update_layout(
        title="Unghiuri coate – balans brațe",
        xaxis_title="Timp (s)",
        yaxis_title="Unghi (°)",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


def plot_overview_dashboard(stats: dict, score: float) -> go.Figure:
    """
    Dashboard radial (spider chart) cu toți parametrii principali,
    pentru o privire de ansamblu rapidă.
    """
    categories = ["Genunchi", "Trunchi", "Simetrie", "Brațe", "Pas"]

    # Normalizăm fiecare parametru la 0–100 (100 = perfect)
    def normalize_knee(angle):
        if angle is None:
            return 50
        if 140 <= angle <= 175:
            return 100
        return max(0, 100 - abs(angle - 157) * 2)

    def normalize_trunk(angle):
        if angle is None:
            return 50
        if 3 <= angle <= 15:
            return 100
        return max(0, 100 - abs(angle - 9) * 5)

    def normalize_symmetry(diff):
        if diff is None:
            return 50
        return max(0, 100 - diff * 5)

    def normalize_elbow(angle):
        if angle is None:
            return 50
        if 70 <= angle <= 110:
            return 100
        return max(0, 100 - abs(angle - 90) * 2)

    knee_score = (
        normalize_knee(stats.get("knee_angle_left_mean")) +
        normalize_knee(stats.get("knee_angle_right_mean"))
    ) / 2

    values = [
        knee_score,
        normalize_trunk(stats.get("trunk_lean_mean")),
        normalize_symmetry(stats.get("symmetry_diff_mean")),
        (normalize_elbow(stats.get("elbow_angle_left_mean")) +
         normalize_elbow(stats.get("elbow_angle_right_mean"))) / 2,
        100,  # placeholder pentru pas (overstride detection mai complexă)
    ]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(0,200,150,0.2)",
        line=dict(color=COLOR_LEFT, width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Scor general: {score:.0f}/100",
        template="plotly_dark",
    )
    return fig


# ── Grafic scor cu gauge ──────────────────────────────────────────────────────

def plot_score_gauge(score: float) -> go.Figure:
    color = COLOR_OK if score >= 75 else COLOR_WARN if score >= 50 else COLOR_ERR

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Scor tehnică alergare", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  50], "color": "rgba(231,76,60,0.3)"},
                {"range": [50, 75], "color": "rgba(243,156,18,0.3)"},
                {"range": [75,100], "color": "rgba(46,204,113,0.3)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(template="plotly_dark", height=300)
    return fig