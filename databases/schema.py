"""
database/schema.py

Defines the SQLite schema for BeefLink and provides simple helper functions
to initialize the database and insert/query records. Kept deliberately
plain (raw sqlite3, no ORM) so it's easy to inspect and reason about for
your thesis write-up.

Tables:
    batches            — one row per supply listing a farmer registers
    herd_verification  — one row per herd photo verification (Module 2)
    meat_verification  — one row per meat freshness verification (Module 3)
    reports            — one row per final verification report (Module 4-6)

USAGE:
    from database.schema import init_db, get_connection
    init_db()  # call once at app startup — safe to call repeatedly
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sqlite.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    farm_id TEXT NOT NULL,
    declared_count INTEGER NOT NULL,
    image_path TEXT,
    date TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS herd_verification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    detected_count INTEGER NOT NULL,
    average_confidence REAL,
    max_confidence REAL,
    min_confidence REAL,
    status TEXT,                -- 'verified' / 'manual_review' / 'rejected'
    model_used TEXT,            -- e.g. 'yolov8n', 'yolov8s', 'yolov8-cbam'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
);

CREATE TABLE IF NOT EXISTS meat_verification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    freshness_label TEXT,       -- 'fresh' / 'half_fresh' / 'spoiled'
    freshness_confidence REAL,
    prob_fresh REAL,
    prob_half_fresh REAL,
    prob_spoiled REAL,
    model_used TEXT,            -- e.g. 'mobilenetv3', 'dcnn_radam'
    gradcam_path TEXT,          -- path to saved heatmap image, if applicable
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    trust_score INTEGER,
    decision TEXT,               -- 'Verified' / 'Review Required' / 'Rejected' (system-computed)
    buyer_decision TEXT,         -- 'Accepted' / 'Rejected' (the buyer's actual choice)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
);
"""


def get_connection():
    """Returns a sqlite3 connection with row access by column name."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates all tables if they don't already exist. Safe to call anytime."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def insert_batch(batch_id: str, farm_id: str, declared_count: int, image_path: str, date: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO batches (batch_id, farm_id, declared_count, image_path, date) VALUES (?, ?, ?, ?, ?)",
        (batch_id, farm_id, declared_count, image_path, date),
    )
    conn.commit()
    conn.close()


def insert_herd_verification(batch_id: str, detected_count: int, avg_conf: float,
                               max_conf: float, min_conf: float, status: str, model_used: str):
    conn = get_connection()
    conn.execute(
        """INSERT INTO herd_verification
           (batch_id, detected_count, average_confidence, max_confidence, min_confidence, status, model_used)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (batch_id, detected_count, avg_conf, max_conf, min_conf, status, model_used),
    )
    conn.commit()
    conn.close()


def insert_meat_verification(batch_id: str, label: str, confidence: float,
                               prob_fresh: float, prob_half_fresh: float, prob_spoiled: float,
                               model_used: str, gradcam_path: str = None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO meat_verification
           (batch_id, freshness_label, freshness_confidence, prob_fresh, prob_half_fresh, prob_spoiled, model_used, gradcam_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (batch_id, label, confidence, prob_fresh, prob_half_fresh, prob_spoiled, model_used, gradcam_path),
    )
    conn.commit()
    conn.close()


def insert_report(batch_id: str, trust_score: int, decision: str, buyer_decision: str = None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO reports (batch_id, trust_score, decision, buyer_decision) VALUES (?, ?, ?, ?)",
        (batch_id, trust_score, decision, buyer_decision),
    )
    conn.commit()
    conn.close()


def get_batch_full_record(batch_id: str):
    """Joins all tables for a single batch — used by the Report page."""
    conn = get_connection()
    batch = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    herd = conn.execute(
        "SELECT * FROM herd_verification WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)
    ).fetchone()
    meat = conn.execute(
        "SELECT * FROM meat_verification WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)
    ).fetchone()
    report = conn.execute(
        "SELECT * FROM reports WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)
    ).fetchone()
    conn.close()
    return {"batch": batch, "herd": herd, "meat": meat, "report": report}


def get_all_batches():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows
