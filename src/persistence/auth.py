"""
src/auth.py
Autentificare utilizatori cu SQLite + PBKDF2-SHA256.
Nu necesita dependente externe — foloseste exclusiv biblioteca standard.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/runanalyzer.db")


# ── Conexiune ─────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Initializare schema ───────────────────────────────────────────────────────

def init_db() -> None:
    """Creeaza tabelele daca nu exista. Apeleaza la pornirea aplicatiei."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT    NOT NULL UNIQUE,
                email        TEXT    NOT NULL UNIQUE,
                password_hash TEXT   NOT NULL,
                salt         TEXT    NOT NULL,
                created_at   TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                 INTEGER NOT NULL,
                created_at              TEXT    NOT NULL,
                video_filename          TEXT,
                score                   REAL,
                runner_height_cm        INTEGER,
                runner_sex              TEXT,
                cadence_spm             REAL,
                vertical_oscillation_cm REAL,
                step_length_m           REAL,
                foot_strike_type        TEXT,
                symmetry_contact_deg    REAL,
                feedback_json           TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()


# ── Hash parola ───────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Returneaza (hash_hex, salt_hex). PBKDF2-HMAC-SHA256, 260000 iteratii."""
    if salt is None:
        salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000)
    return dk.hex(), salt


# ── Inregistrare ─────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Inregistreaza un utilizator nou.
    Returneaza (True, "") la succes sau (False, mesaj_eroare) la esec.
    """
    username = username.strip()
    email    = email.strip().lower()

    if not username or not email or not password:
        return False, "Completează toate câmpurile."
    if len(username) < 2:
        return False, "Numele de utilizator trebuie să aibă cel puțin 2 caractere."
    if len(password) < 6:
        return False, "Parola trebuie să aibă cel puțin 6 caractere."
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Adresă de email invalidă."

    pw_hash, salt = _hash_password(password)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?,?,?,?,?)",
                (username, email, pw_hash, salt, datetime.utcnow().isoformat()),
            )
            conn.commit()
        return True, ""
    except sqlite3.IntegrityError as exc:
        if "username" in str(exc):
            return False, "Numele de utilizator este deja folosit."
        return False, "Adresa de email este deja înregistrată."


# ── Autentificare ─────────────────────────────────────────────────────────────

def login_user(username_or_email: str, password: str) -> Optional[dict]:
    """
    Verifica credentialele.
    Returneaza dict cu {id, username, email} la succes, None la esec.
    """
    val = username_or_email.strip()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (val, val.lower()),
        ).fetchone()

    if not row:
        return None
    pw_hash, _ = _hash_password(password, row["salt"])
    if pw_hash != row["password_hash"]:
        return None
    return {"id": row["id"], "username": row["username"], "email": row["email"]}
