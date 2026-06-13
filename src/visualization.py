"""
src/visualization.py
Generarea graficelor si vizualizarilor pentru rezultate.
"""

import numpy as np
import plotly.graph_objects as go

from src.biomechanics import BiomechanicsFrame
from src.gait_analysis import GaitAnalysisResult, PHASE_CONTACT_SYMMETRY_MAX, KNEE_DRIVE_SYMMETRY_MAX


# ── Culori consistente ────────────────────────────────────────────────────────
COLOR_LEFT = "#00C896"   # verde
COLOR_RIGHT = "#FF6B35"  # portocaliu
COLOR_OK = "#2ECC71"
COLOR_WARN = "#F39C12"
COLOR_ERR = "#E74C3C"
BG_DARK = "#0F1117"


def _preprocess_series(
    values: list[float | None],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    smooth_window: int = 9,
) -> tuple[list[float | None], float]:
    """
    Curata seria pentru afisare:
    - elimina outlierii (IQR),
    - aplica limite fiziologice,
    - aplica smoothing pe vecinatate.
    """
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    valid_ratio = float(np.mean(~np.isnan(arr))) if len(arr) else 0.0

    if np.sum(~np.isnan(arr)) >= 8:
        valid = arr[~np.isnan(arr)]
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        arr[(arr < low) | (arr > high)] = np.nan

    if lower_bound is not None:
        arr[arr < lower_bound] = np.nan
    if upper_bound is not None:
        arr[arr > upper_bound] = np.nan

    if np.sum(~np.isnan(arr)) >= max(5, smooth_window):
        radius = max(1, smooth_window // 2)
        smoothed = arr.copy()
        for i in range(len(arr)):
            start, end = max(0, i - radius), min(len(arr), i + radius + 1)
            window_vals = arr[start:end]
            valid_vals = window_vals[~np.isnan(window_vals)]
            if len(valid_vals) >= max(3, radius):
                smoothed[i] = float(np.mean(valid_vals))
        arr = smoothed

    return [None if np.isnan(v) else float(v) for v in arr], valid_ratio


def compute_metric_quality(bio_frames: list[BiomechanicsFrame]) -> dict[str, float]:
    """Returneaza procentul de frame-uri valide pentru fiecare metrica."""
    if not bio_frames:
        return {}

    metrics = {
        "knee_angle_left": [f.knee_angle_left for f in bio_frames],
        "knee_angle_right": [f.knee_angle_right for f in bio_frames],
        "trunk_lean_angle": [f.trunk_lean_angle for f in bio_frames],
        "elbow_angle_left": [f.elbow_angle_left for f in bio_frames],
        "elbow_angle_right": [f.elbow_angle_right for f in bio_frames],
        "knee_symmetry_diff": [f.knee_symmetry_diff for f in bio_frames],
        "step_length": [f.step_length for f in bio_frames],
    }
    return {name: float(np.mean([v is not None for v in vals]) * 100.0) for name, vals in metrics.items()}


# ── Grafice Plotly (interactive, pentru Streamlit) ───────────────────────────

def plot_knee_angles(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    left, left_valid = _preprocess_series(
        [f.knee_angle_left for f in bio_frames], lower_bound=30, upper_bound=185
    )
    right, right_valid = _preprocess_series(
        [f.knee_angle_right for f in bio_frames], lower_bound=30, upper_bound=185
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=left, mode="lines",
        name=f"Genunchi stang ({left_valid * 100:.0f}% valid)",
        line=dict(color=COLOR_LEFT, width=2), connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=right, mode="lines",
        name=f"Genunchi drept ({right_valid * 100:.0f}% valid)",
        line=dict(color=COLOR_RIGHT, width=2), connectgaps=False,
    ))
    fig.add_hrect(y0=140, y1=175, fillcolor="rgba(46,204,113,0.1)", line_width=0, annotation_text="Zona de referinta")
    fig.update_layout(
        title="Unghiuri genunchi in timp (filtrat + smooth)",
        xaxis_title="Timp (s)",
        yaxis_title="Unghi (grade)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_trunk_lean(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    trunk, trunk_valid = _preprocess_series(
        [f.trunk_lean_angle for f in bio_frames], lower_bound=0, upper_bound=65
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=trunk, mode="lines",
        name=f"Inclinare trunchi ({trunk_valid * 100:.0f}% valid)",
        line=dict(color="#9B59B6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.15)", connectgaps=False,
    ))
    fig.add_hrect(y0=3, y1=15, fillcolor="rgba(46,204,113,0.1)", line_width=0, annotation_text="Ideal 3-15")
    fig.update_layout(
        title="Inclinarea trunchiului fata de verticala",
        xaxis_title="Timp (s)",
        yaxis_title="Unghi (grade)",
        template="plotly_dark",
    )
    return fig


def plot_knee_lr_diff_instantaneous(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    """
    Diferența |unghi stâng − unghi drept| în același frame.
    Valori mari sunt normale la alergare (picioare în faze diferite).
    """
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    diff, diff_valid = _preprocess_series(
        [f.knee_symmetry_diff for f in bio_frames], lower_bound=0, upper_bound=180
    )

    fig = go.Figure(go.Bar(
        x=timestamps,
        y=diff,
        marker_color="rgba(0, 200, 150, 0.55)",
        name=f"Diferenta L-R ({diff_valid * 100:.0f}% valid)",
    ))
    fig.update_layout(
        title="Diferență unghi genunchi L–R (instantanee, per frame)",
        xaxis_title="Timp (s)",
        yaxis_title="Diferență (grade)",
        template="plotly_dark",
        annotations=[dict(
            text="Diferențe mari sunt așteptate: picioarele sunt în faze diferite.",
            xref="paper", yref="paper", x=0, y=1.12, showarrow=False,
            font=dict(size=11, color="#94a3b8"),
        )],
    )
    return fig


def plot_symmetry(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    """Alias retrocompatibil."""
    return plot_knee_lr_diff_instantaneous(bio_frames)


def _symmetry_bar(
    left_vals: list[float],
    right_vals: list[float],
    sym_deg: float | None,
    sym_max: float,
    title: str,
    yaxis_label: str,
    ideal_low: float,
    ideal_high: float,
    yaxis_reversed: bool = False,
    no_data_msg: str = "Nu s-au detectat suficiente date.",
) -> go.Figure:
    """
    Doua bare simple (stâng / drept) care arata media unghiului per picior.
    Bara mai inalta = unghi mai mare. Bare egale = simetrie perfecta.
    Bara de eroare arata variatia intre pasi.
    """
    left_mean  = float(np.mean(left_vals))  if left_vals  else None
    right_mean = float(np.mean(right_vals)) if right_vals else None
    left_std   = float(np.std(left_vals))   if len(left_vals)  > 1 else 0.0
    right_std  = float(np.std(right_vals))  if len(right_vals) > 1 else 0.0

    sym_text = f"{sym_deg:.1f}°" if sym_deg is not None else "N/A"
    sym_color = (
        COLOR_OK   if sym_deg is not None and sym_deg <= sym_max
        else COLOR_WARN if sym_deg is not None and sym_deg <= sym_max + 10
        else COLOR_ERR
    )
    if sym_deg is not None:
        status = "✓ simetrie bună" if sym_deg <= sym_max else ("⚠ asimetrie moderată" if sym_deg <= sym_max + 10 else "✗ asimetrie marcată")
    else:
        status = ""

    fig = go.Figure()

    bars = [
        ("Stâng",  left_mean,  left_std,  len(left_vals),  COLOR_LEFT,  "rgba(0,200,150,0.75)"),
        ("Drept",  right_mean, right_std, len(right_vals), COLOR_RIGHT, "rgba(255,107,53,0.75)"),
    ]
    for label, mean, std, n, border, fill in bars:
        if mean is None:
            continue
        fig.add_trace(go.Bar(
            x=[f"{label}<br><sub>({n} pași)</sub>"],
            y=[mean],
            error_y=dict(type="data", array=[std], visible=True, color=border, thickness=2, width=8),
            marker=dict(color=fill, line=dict(color=border, width=2)),
            name=label,
            text=[f"<b>{mean:.1f}°</b>"],
            textposition="outside",
            textfont=dict(size=16, color=border),
            width=0.45,
        ))

    # Zona ideala
    fig.add_hrect(y0=ideal_low, y1=ideal_high,
                  fillcolor="rgba(46,204,113,0.10)", line_width=0,
                  annotation_text=f"Ideal {ideal_low:.0f}–{ideal_high:.0f}°",
                  annotation_position="top right",
                  annotation_font=dict(color=COLOR_OK, size=11))

    fig.update_layout(
        title=dict(text=f"{title} — {sym_text}  {status}", font=dict(size=15)),
        yaxis_title=yaxis_label,
        yaxis=dict(autorange="reversed" if yaxis_reversed else True),
        xaxis=dict(tickfont=dict(size=14)),
        template="plotly_dark",
        showlegend=False,
        bargap=0.35,
        annotations=[dict(
            text=f"Diferenta dintre picioare: <b>{sym_text}</b>  (prag ≤{sym_max:.0f}°)",
            xref="paper", yref="paper", x=0.5, y=1.10,
            xanchor="center", showarrow=False,
            font=dict(size=13, color=sym_color if sym_deg is not None else "#94a3b8"),
        )],
    )

    if left_mean is None and right_mean is None:
        fig.add_annotation(
            text=no_data_msg,
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8"),
        )

    return fig


def plot_phase_contact_symmetry(gait: GaitAnalysisResult) -> go.Figure:
    """Doua bare: unghi mediu al genunchiului la contact per picior."""
    left_ang  = [e.knee_angle for e in gait.left_strikes  if e.knee_angle is not None]
    right_ang = [e.knee_angle for e in gait.right_strikes if e.knee_angle is not None]
    return _symmetry_bar(
        left_ang, right_ang,
        sym_deg=gait.phase_symmetry_contact_deg,
        sym_max=PHASE_CONTACT_SYMMETRY_MAX,
        title="Unghi genunchi la contactul cu solul",
        yaxis_label="Unghi genunchi (grade)",
        ideal_low=140, ideal_high=175,
        no_data_msg="Nu s-au detectat suficiente contacte. Folosește video lateral, corp complet în cadru.",
    )


def plot_knee_drive_symmetry(gait: GaitAnalysisResult) -> go.Figure:
    """Doua bare: unghi mediu al genunchiului la ridicare per picior. Bara mai mica = genunchi mai sus."""
    left_ang  = [e.knee_angle for e in gait.left_knee_drives  if e.knee_angle is not None]
    right_ang = [e.knee_angle for e in gait.right_knee_drives if e.knee_angle is not None]
    return _symmetry_bar(
        left_ang, right_ang,
        sym_deg=gait.knee_drive_symmetry_deg,
        sym_max=KNEE_DRIVE_SYMMETRY_MAX,
        title="Unghi genunchi la ridicare (knee drive)",
        yaxis_label="Unghi genunchi (grade) — bara mai mică = genunchi mai sus",
        ideal_low=90, ideal_high=130,
        yaxis_reversed=True,
        no_data_msg="Nu s-au detectat ridicari de genunchi. Asigura-te ca intregul corp este vizibil in cadru.",
    )


def plot_foot_strike_chart(gait: GaitAnalysisResult) -> go.Figure:
    """
    Grafic tip bara 100% care arata proportia heel / midfoot / forefoot per picior.
    Bazat pe diferenta calcan.y - varf.y la momentul contactului detectat.
    """
    COLORS = {
        "heel":     ("#E74C3C", "rgba(231,76,60,0.75)"),
        "midfoot":  ("#F39C12", "rgba(243,156,18,0.75)"),
        "forefoot": ("#2ECC71", "rgba(46,204,113,0.75)"),
    }
    LABELS = {"heel": "Călcâi (heel)", "midfoot": "Mijloc talpă", "forefoot": "Vârf (forefoot)"}

    def counts(events: list) -> dict:
        c = {"heel": 0, "midfoot": 0, "forefoot": 0}
        for e in events:
            if e.strike_type in c:
                c[e.strike_type] += 1
        return c

    left_c  = counts(gait.left_strikes)
    right_c = counts(gait.right_strikes)
    left_n  = sum(left_c.values())
    right_n = sum(right_c.values())

    fig = go.Figure()

    legs   = []
    if left_n  > 0: legs.append(("Stâng",  left_c,  left_n))
    if right_n > 0: legs.append(("Drept",  right_c, right_n))

    for strike_key in ["heel", "midfoot", "forefoot"]:
        border, fill = COLORS[strike_key]
        xs, ys, texts = [], [], []
        for leg_label, c, n in legs:
            pct = c[strike_key] / n * 100
            xs.append(leg_label)
            ys.append(pct)
            texts.append(f"{pct:.0f}%" if pct >= 8 else "")

        fig.add_trace(go.Bar(
            name=LABELS[strike_key],
            x=xs,
            y=ys,
            text=texts,
            textposition="inside",
            textfont=dict(size=14, color="white"),
            marker=dict(color=fill, line=dict(color=border, width=1.5)),
        ))

    # Rezumat dominant per picior
    def dominant(c, n):
        if n == 0:
            return "N/A"
        key = max(c, key=c.get)
        return f"{LABELS[key]} ({c[key]/n*100:.0f}%)"

    subtitle_parts = []
    if left_n  > 0: subtitle_parts.append(f"Stâng: {dominant(left_c,  left_n)}")
    if right_n > 0: subtitle_parts.append(f"Drept: {dominant(right_c, right_n)}")
    subtitle = "   |   ".join(subtitle_parts)

    fig.update_layout(
        barmode="stack",
        title=dict(text="Tipul de contact cu solul (heel / midfoot / forefoot)", font=dict(size=15)),
        yaxis=dict(title="% din contacte detectate", range=[0, 100], ticksuffix="%"),
        xaxis=dict(tickfont=dict(size=14)),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.35,
        annotations=[dict(
            text=subtitle,
            xref="paper", yref="paper", x=0.5, y=1.13,
            xanchor="center", showarrow=False,
            font=dict(size=12, color="#94a3b8"),
        )],
    )

    if not legs:
        fig.add_annotation(
            text="Nu s-au detectat contacte. Folosește video lateral, corp complet în cadru.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8"),
        )

    return fig


def plot_overstride_chart(gait: GaitAnalysisResult) -> go.Figure:
    """
    Bara per picior aratand cat de mult aterizeaza piciorul in fata soldului la contact.
    0 = picior direct sub sold (ideal).  >0.05 = overstride (risc de accidentare).
    """
    THRESHOLD = 0.05

    left_val  = gait.left_overstride_mean
    right_val = gait.right_overstride_mean

    labels, values, colors, borders = [], [], [], []
    for label, val, fill, border in [
        ("Stâng", left_val,  "rgba(0,200,150,0.75)",  "#00C896"),
        ("Drept", right_val, "rgba(255,107,53,0.75)", "#FF6B35"),
    ]:
        if val is None:
            continue
        labels.append(label)
        values.append(val)
        over = val > THRESHOLD
        colors.append("rgba(231,76,60,0.80)" if over else fill)
        borders.append("#E74C3C" if over else border)

    fig = go.Figure()

    if labels:
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color=borders, width=2),
            ),
            text=[f"<b>{v:+.3f}</b>" for v in values],
            textposition="outside",
            textfont=dict(size=16),
            width=0.45,
            showlegend=False,
        ))

    # Linie la 0: aterizare perfecta sub sold
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=1)

    # Zona de overstride
    fig.add_hrect(
        y0=THRESHOLD, y1=max(0.25, (max(values) if values else 0) + 0.05),
        fillcolor="rgba(231,76,60,0.08)", line_width=0,
        annotation_text="Overstride (>0.05)", annotation_position="top right",
        annotation_font=dict(color="#E74C3C", size=11),
    )
    fig.add_hline(
        y=THRESHOLD, line_dash="dash", line_color="#E74C3C", line_width=1.5,
        annotation_text="Prag overstride", annotation_position="bottom right",
        annotation_font=dict(color="#E74C3C", size=10),
    )

    # Zona ideala (sub 0)
    fig.add_hrect(
        y0=-0.25, y1=0,
        fillcolor="rgba(46,204,113,0.06)", line_width=0,
        annotation_text="Aterizare sub sold (ideal)", annotation_position="bottom right",
        annotation_font=dict(color="#2ECC71", size=11),
    )

    any_over = any(v > THRESHOLD for v in values) if values else False
    status = "✗ overstride detectat" if any_over else "✓ aterizare corectă"
    status_color = "#E74C3C" if any_over else "#2ECC71"

    fig.update_layout(
        title=dict(
            text=f"Overstride la contact — {status}",
            font=dict(size=15),
        ),
        yaxis_title="Offset gleznă față de șold (coordonate normalizate)",
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(range=[-0.15, max(0.2, (max(values) if values else 0) + 0.06)]),
        template="plotly_dark",
        bargap=0.35,
        annotations=[dict(
            text=(
                "0 = picior direct sub șold.  "
                "Valoare pozitivă = picior în față = risc de suprasolicitare articulară."
            ),
            xref="paper", yref="paper", x=0.5, y=1.10,
            xanchor="center", showarrow=False,
            font=dict(size=12, color=status_color),
        )],
    )

    if not labels:
        fig.add_annotation(
            text="Nu s-au detectat contacte cu solul. Folosește video lateral, corp complet în cadru.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8"),
        )

    return fig


def plot_elbow_angles(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    timestamps = [f.timestamp_ms / 1000.0 for f in bio_frames]
    left, left_valid = _preprocess_series(
        [f.elbow_angle_left for f in bio_frames], lower_bound=20, upper_bound=185
    )
    right, right_valid = _preprocess_series(
        [f.elbow_angle_right for f in bio_frames], lower_bound=20, upper_bound=185
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=left, mode="lines",
        name=f"Cot stang ({left_valid * 100:.0f}% valid)",
        line=dict(color=COLOR_LEFT), connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=right, mode="lines",
        name=f"Cot drept ({right_valid * 100:.0f}% valid)",
        line=dict(color=COLOR_RIGHT), connectgaps=False,
    ))
    fig.add_hrect(y0=70, y1=110, fillcolor="rgba(46,204,113,0.1)", line_width=0, annotation_text="Ideal 70-110")
    fig.update_layout(
        title="Unghiuri coate - balans brate",
        xaxis_title="Timp (s)",
        yaxis_title="Unghi (grade)",
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
        normalize_symmetry(stats.get("phase_symmetry_contact_deg")),
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

def plot_vertical_oscillation(bio_frames: list[BiomechanicsFrame]) -> go.Figure:
    """Grafic oscilatie verticala: inaltimea soldului in timp, cu banda de amplitudine."""
    pairs = [
        (f.timestamp_ms / 1000.0, f.hip_height)
        for f in bio_frames
        if f.hip_height is not None
    ]
    if len(pairs) < 10:
        fig = go.Figure()
        fig.update_layout(title="Oscilație verticală — date insuficiente", template="plotly_dark", height=280)
        return fig

    times = [p[0] for p in pairs]
    hip_y = np.array([p[1] for p in pairs], dtype=float)

    # Simple moving-average smoothing
    window = min(9, max(3, len(hip_y) // 15))
    smoothed = np.convolve(hip_y, np.ones(window) / window, mode="same")

    p10 = float(np.percentile(hip_y, 10))
    p90 = float(np.percentile(hip_y, 90))
    amplitude = p90 - p10

    fig = go.Figure()
    fig.add_hrect(
        y0=p10, y1=p90,
        fillcolor="rgba(33,150,243,0.12)",
        line_width=0,
        annotation_text=f"Amplitudine: {amplitude:.3f}",
        annotation_position="top right",
        annotation_font_color="#90CAF9",
    )
    fig.add_trace(go.Scatter(
        x=times, y=smoothed.tolist(),
        mode="lines",
        name="Înălțime șold (filtrat)",
        line=dict(color="#2196F3", width=1.5),
    ))
    # Invert Y so that "up" on chart = runner physically higher
    fig.update_yaxes(autorange="reversed", title="Poziție șold (Y norm.)")
    fig.update_xaxes(title="Timp (s)")
    fig.update_layout(
        title="Oscilație verticală (înălțime șold)",
        template="plotly_dark",
        height=280,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    return fig


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