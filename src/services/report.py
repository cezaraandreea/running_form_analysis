"""
src/report.py
Generare raport PDF cu rezultatele analizei de alergare.
"""
from __future__ import annotations

import datetime
import os
import tempfile
from typing import Optional

from fpdf import FPDF

from src.services.analysis import Severity
from src.pipeline.biomechanics import BiomechanicsFrame
from src.pipeline.gait_analysis import GaitAnalysisResult
from src.pipeline.visualization import (
    plot_elbow_angles,
    plot_foot_strike_chart,
    plot_knee_angles,
    plot_knee_drive_symmetry,
    plot_overstride_chart,
    plot_phase_contact_symmetry,
    plot_trunk_lean,
    plot_vertical_oscillation,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _d(text: str) -> str:
    """Normalizeaza textul pentru codificarea Latin-1 folosita de fonturile PDF standard."""
    for src, dst in [
        ('ă','a'),('â','a'),('î','i'),('ș','s'),('ț','t'),
        ('Ă','A'),('Â','A'),('Î','I'),('Ș','S'),('Ț','T'),
        ('ş','s'),('ţ','t'),('Ş','S'),('Ţ','T'),
        ('—','-'),('–','-'),   # em dash, en dash
        ('“','"'),('”','"'),   # ghilimele culbute
        ('‘',"'"),('’',"'"),   # apostrof curbat
        ('…','...'),                # ellipsis
    ]:
        text = text.replace(src, dst)
    # Orice caracter ramas in afara Latin-1 e inlocuit cu '?'
    return text.encode('latin-1', errors='replace').decode('latin-1')


def _fig_png(fig, width: int = 750, height: int = 320) -> Optional[bytes]:
    try:
        import plotly.io as pio
        return pio.to_image(fig, format='png', width=width, height=height, scale=1.5)
    except Exception:
        return None


# ── PDF class ─────────────────────────────────────────────────────────────────

class _RunReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 6, 'RunAnalyzer - Raport analiza alergare', align='R')
        self.ln(2)
        self.set_draw_color(220, 220, 220)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 6, f'Pagina {self.page_no()}', align='C')

    def section_title(self, text: str) -> None:
        self.ln(2)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, _d(text), ln=True)
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def metric_row(self, label: str, value: str, note: str, shaded: bool) -> None:
        if shaded:
            self.set_fill_color(246, 248, 250)
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(60, 60, 60)
        self.cell(80, 7, _d(label), fill=shaded)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(20, 20, 20)
        self.cell(40, 7, value, fill=shaded)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 7, _d(note), fill=shaded, ln=True)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf_report(
    bio_frames: list[BiomechanicsFrame],
    gait: Optional[GaitAnalysisResult],
    analysis: Optional[dict],
    height_cm: int,
    runner_sex: str,
) -> bytes:
    """
    Construieste un PDF complet cu metrici, feedback si grafice.
    Returneaza continutul PDF ca bytes.
    """
    leg_ratio   = 0.49 if runner_sex == 'Masculin' else 0.46
    ideal_cad   = '170-185' if runner_sex == 'Masculin' else '174-190'
    date_str    = datetime.date.today().strftime('%d.%m.%Y')

    pdf = _RunReport(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Titlu ─────────────────────────────────────────────────────────────────
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 12, 'Raport analiza alergare', ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f'Generat: {date_str}', ln=True)
    pdf.ln(5)

    # ── Info alergator ────────────────────────────────────────────────────────
    pdf.set_fill_color(240, 244, 250)
    pdf.set_draw_color(200, 215, 235)
    y0 = pdf.get_y()
    pdf.rect(pdf.l_margin, y0, pdf.epw, 16, style='FD')
    pdf.set_y(y0 + 2)
    half = pdf.epw / 2
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(40, 60, 100)
    pdf.set_x(pdf.l_margin + 4)
    pdf.cell(half, 6, _d(f'Inaltime: {height_cm} cm'), ln=False)
    pdf.cell(half, 6, _d(f'Sex: {runner_sex}'), ln=True)
    pdf.set_x(pdf.l_margin + 4)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(80, 100, 140)
    pdf.cell(half, 6, _d(f'Raport lungime picior/inaltime: {leg_ratio}'), ln=False)
    pdf.cell(half, 6, _d(f'Cadenta ideala: {ideal_cad} pasi/min'), ln=True)
    pdf.ln(8)

    # ── Scor general ──────────────────────────────────────────────────────────
    if analysis:
        score = analysis.get('score', 0)
        if score >= 75:
            score_rgb = (39, 174, 96)
            score_label = 'Excelent'
        elif score >= 50:
            score_rgb = (211, 130, 20)
            score_label = 'Mediu'
        else:
            score_rgb = (192, 57, 43)
            score_label = 'Necesita imbunatatiri'

        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(50, 9, 'Scor tehnica de alergare:', ln=False)
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(*score_rgb)
        pdf.cell(25, 9, f'{score:.0f}/100', ln=False)
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(*score_rgb)
        pdf.cell(0, 9, f'({score_label})', ln=True)
        pdf.ln(4)

    # ── Metrici principale ────────────────────────────────────────────────────
    pdf.section_title('Metrici principale')

    metrics: list[tuple[str, str, str]] = []
    if gait:
        if gait.cadence_spm is not None:
            metrics.append(('Cadenta', f'{gait.cadence_spm:.0f} pasi/min', f'Ideal: {ideal_cad}'))
        if gait.phase_symmetry_contact_deg is not None:
            metrics.append(('Simetrie la contact', f'{gait.phase_symmetry_contact_deg:.1f}°', 'Ideal < 15°'))
        if gait.knee_drive_symmetry_deg is not None:
            metrics.append(('Simetrie ridicare genunchi', f'{gait.knee_drive_symmetry_deg:.1f}°', 'Ideal < 15°'))
        if gait.left_drive_knee_mean_deg is not None and gait.right_drive_knee_mean_deg is not None:
            metrics.append((
                'Unghi ridicare genunchi (S/D)',
                f'{gait.left_drive_knee_mean_deg:.0f}° / {gait.right_drive_knee_mean_deg:.0f}°',
                'Mai mic = ridicare mai inalta',
            ))
        step_vals = [v for v in [gait.left_step_length_mean, gait.right_step_length_mean] if v is not None]
        if step_vals and gait.leg_length_normalized:
            scale    = (height_cm * leg_ratio / 100) / gait.leg_length_normalized
            avg_step = sum(step_vals) / len(step_vals) * scale
            metrics.append(('Lungime medie pas', f'{avg_step:.2f} m', ''))
        if gait.vertical_oscillation_norm is not None and gait.leg_length_normalized:
            scale = (height_cm * leg_ratio / 100) / gait.leg_length_normalized
            vo_cm = gait.vertical_oscillation_norm * scale * 100
            metrics.append(('Oscilatie verticala', f'{vo_cm:.1f} cm', 'Ideal < 8 cm'))

    if metrics:
        for i, (label, value, note) in enumerate(metrics):
            pdf.metric_row(label, value, note, shaded=(i % 2 == 0))
    else:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(130, 130, 130)
        pdf.cell(0, 7, 'Date insuficiente pentru calculul metricilor.', ln=True)
    pdf.ln(4)

    # ── Feedback ──────────────────────────────────────────────────────────────
    if analysis:
        issues = analysis.get('issues', [])
        pdf.section_title('Feedback tehnica')

        for item in issues:
            if item.severity == Severity.ERROR:
                marker, rgb = '[!]', (192, 57, 43)
            elif item.severity == Severity.WARNING:
                marker, rgb = '[~]', (180, 110, 20)
            else:
                marker, rgb = '[OK]', (39, 174, 96)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(*rgb)
            pdf.cell(12, 6, marker, ln=False)
            pdf.set_text_color(30, 30, 30)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 6, _d(f'{item.category}: {item.message}'), ln=True)
            if item.detail:
                pdf.set_x(pdf.l_margin + 12)
                pdf.set_font('Helvetica', 'I', 8)
                pdf.set_text_color(110, 110, 110)
                pdf.multi_cell(0, 5, _d(f'  {item.detail}'))
            pdf.ln(1)

    # ── Grafice ───────────────────────────────────────────────────────────────
    chart_defs = [
        ('Unghiuri genunchi', plot_knee_angles(bio_frames)),
        ('Inclinare trunchi', plot_trunk_lean(bio_frames)),
        ('Oscilatie verticala', plot_vertical_oscillation(bio_frames)),
    ]
    if gait:
        chart_defs += [
            ('Tip de aterizare', plot_foot_strike_chart(gait)),
            ('Simetrie la contact', plot_phase_contact_symmetry(gait)),
            ('Simetrie ridicare genunchi', plot_knee_drive_symmetry(gait)),
            ('Overstride', plot_overstride_chart(gait)),
        ]
    chart_defs.append(('Unghiuri coate', plot_elbow_angles(bio_frames)))

    with tempfile.TemporaryDirectory() as tmp:
        charts_added = 0
        for title, fig in chart_defs:
            png = _fig_png(fig)
            if png is None:
                continue
            img_path = os.path.join(tmp, f'{charts_added}.png')
            with open(img_path, 'wb') as fh:
                fh.write(png)

            # New page for first chart, or when near the bottom
            chart_h = 65   # mm
            if charts_added == 0:
                pdf.add_page()
                pdf.section_title('Grafice')
            elif pdf.get_y() + chart_h + 12 > pdf.h - pdf.b_margin:
                pdf.add_page()

            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 6, _d(title), ln=True)
            pdf.image(img_path, x=pdf.l_margin, w=pdf.epw, h=chart_h)
            pdf.ln(5)
            charts_added += 1

    return bytes(pdf.output())
