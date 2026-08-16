"""
src/email_sender.py
Trimitere email cu raportul PDF atasat, prin SMTP.
Configurat prin .streamlit/secrets.toml (sectiunea [email]).
Foloseste exclusiv biblioteca standard Python.
"""

from __future__ import annotations

import smtplib
from datetime import date
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import streamlit as st


class EmailConfigError(Exception):
    """Ridicata cand lipseste configuratia SMTP din secrets.toml."""


def _get_smtp_config() -> dict:
    """Citeste configuratia SMTP din st.secrets. Ridica EmailConfigError daca lipseste."""
    try:
        cfg = st.secrets["email"]
        return {
            "host":     cfg["smtp_host"],
            "port":     int(cfg["smtp_port"]),
            "user":     cfg["smtp_user"],
            "password": cfg["smtp_password"],
            "sender":   cfg.get("sender_name", "RunAnalyzer"),
        }
    except (KeyError, FileNotFoundError):
        raise EmailConfigError(
            "Configuratia SMTP lipseste. Adauga sectiunea [email] in .streamlit/secrets.toml."
        )


def is_email_configured() -> bool:
    """Returneaza True daca sectiunea [email] exista in secrets.toml."""
    try:
        _get_smtp_config()
        return True
    except EmailConfigError:
        return False


def _build_body(score: Optional[float], stats: dict, runner_sex: str) -> str:
    """Construieste corpul HTML al emailului."""
    score_str = f"{int(score)}/100" if score is not None else "N/A"
    cadence   = stats.get("cadence_spm")
    vo_cm     = stats.get("vertical_oscillation_cm")
    sym       = stats.get("phase_symmetry_contact_deg")

    cadence_str = f"{cadence:.0f} pași/min" if cadence else "N/A"
    vo_str      = f"{vo_cm:.1f} cm"         if vo_cm   else "N/A"
    sym_str     = f"{sym:.1f}°"             if sym     else "N/A"

    today = date.today().strftime("%d.%m.%Y")

    return f"""
<!DOCTYPE html>
<html lang="ro">
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;background:#f6f8fa;margin:0;padding:0;">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr><td align="center" style="padding:32px 16px;">
      <table width="560" style="background:#fff;border-radius:10px;
             box-shadow:0 2px 8px rgba(0,0,0,.08);overflow:hidden;">

        <!-- Header -->
        <tr>
          <td style="background:#0d1117;padding:24px 32px;">
            <span style="font-size:1.5rem;font-weight:800;color:#e6edf3;">
              🏃 RunAnalyzer
            </span><br>
            <span style="color:#6e7681;font-size:0.85rem;">
              Raport analiză tehnică de alergare · {today}
            </span>
          </td>
        </tr>

        <!-- Scor -->
        <tr>
          <td style="padding:24px 32px 0;">
            <p style="color:#24292f;font-size:1rem;margin:0 0 8px;">
              Raportul tău de analiză este atașat acestui email.
            </p>
            <div style="display:inline-block;background:#f3eeff;border:2px solid #8b5cf6;
                        border-radius:8px;padding:12px 24px;margin:12px 0;">
              <span style="font-size:1.8rem;font-weight:800;color:#8b5cf6;">
                {score_str}
              </span>
              <span style="color:#57606a;font-size:0.9rem;margin-left:8px;">scor tehnică</span>
            </div>
          </td>
        </tr>

        <!-- Metrici -->
        <tr>
          <td style="padding:16px 32px;">
            <table width="100%" cellpadding="8" cellspacing="0"
                   style="border:1px solid #e8eaed;border-radius:6px;font-size:0.9rem;">
              <tr style="background:#f6f8fa;">
                <td style="color:#57606a;">Cadență</td>
                <td style="font-weight:600;color:#24292f;">{cadence_str}</td>
              </tr>
              <tr>
                <td style="color:#57606a;">Oscilație verticală</td>
                <td style="font-weight:600;color:#24292f;">{vo_str}</td>
              </tr>
              <tr style="background:#f6f8fa;">
                <td style="color:#57606a;">Simetrie la contact</td>
                <td style="font-weight:600;color:#24292f;">{sym_str}</td>
              </tr>
              <tr>
                <td style="color:#57606a;">Sex</td>
                <td style="font-weight:600;color:#24292f;">{runner_sex}</td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:16px 32px 28px;color:#6e7681;font-size:0.8rem;
                     border-top:1px solid #e8eaed;margin-top:16px;">
            Raportul PDF complet (cu grafice și feedback detaliat) este atașat acestui mesaj.
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
""".strip()


def send_analysis_email(
    to_email: str,
    pdf_bytes: bytes,
    score: Optional[float],
    stats: dict,
    runner_sex: str = "Masculin",
    pdf_filename: str = "RunAnalyzer_raport.pdf",
) -> None:
    """
    Trimite emailul cu raportul PDF atasat.
    Ridica EmailConfigError daca SMTP nu e configurat.
    Ridica smtplib.SMTPException / OSError la erori de retea/autentificare.
    """
    cfg = _get_smtp_config()

    msg = MIMEMultipart("mixed")
    msg["From"]    = f"{cfg['sender']} <{cfg['user']}>"
    msg["To"]      = to_email
    msg["Subject"] = "Raportul tău de analiză RunAnalyzer"

    # Corp HTML
    html_body = _build_body(score, stats, runner_sex)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # Atasament PDF
    attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
    attachment.add_header(
        "Content-Disposition", "attachment", filename=pdf_filename
    )
    msg.attach(attachment)

    # Trimitere prin SMTP cu STARTTLS
    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=15) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(cfg["user"], cfg["password"])
        server.sendmail(cfg["user"], to_email, msg.as_string())
