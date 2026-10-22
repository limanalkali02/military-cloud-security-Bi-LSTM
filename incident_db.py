"""
incident_db.py

Persistent storage for confirmed intrusion incidents, backed by SQLite.

This is what your project was missing: every time the Bi-LSTM model
(not a placeholder rule) classifies a flow as malicious above the
decision threshold, a record is written here with full attribution
(timestamp, source IP, destination IP, protocol, attack type,
model confidence, mitigation action taken). This is your "report"
of what happened and where it came from.
"""

import sqlite3
import csv
import os
import threading

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "security_incidents.db")

_lock = threading.Lock()


def init_db(db_path: str = DB_PATH):
    with _lock:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                destination_ip TEXT NOT NULL,
                protocol TEXT,
                attack_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                action_taken TEXT
            )
        """)
        conn.commit()
        conn.close()


def log_incident(timestamp: str, source_ip: str, destination_ip: str,
                  protocol: str, attack_type: str, confidence: float,
                  action_taken: str = "IP flagged for blocking",
                  db_path: str = DB_PATH):
    with _lock:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO incidents (timestamp, source_ip, destination_ip, protocol, "
            "attack_type, confidence, action_taken) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, source_ip, destination_ip, protocol, attack_type, confidence, action_taken)
        )
        conn.commit()
        conn.close()


def fetch_recent_incidents(limit: int = 50, db_path: str = DB_PATH):
    with _lock:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def count_incidents(db_path: str = DB_PATH) -> int:
    with _lock:
        conn = sqlite3.connect(db_path)
        n = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        conn.close()
    return n


def count_by_attack_type(db_path: str = DB_PATH):
    with _lock:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT attack_type, COUNT(*) FROM incidents GROUP BY attack_type"
        ).fetchall()
        conn.close()
    return dict(rows)


def export_csv(out_path: str, db_path: str = DB_PATH):
    incidents = fetch_recent_incidents(limit=100000, db_path=db_path)
    if not incidents:
        fieldnames = ["id", "timestamp", "source_ip", "destination_ip",
                      "protocol", "attack_type", "confidence", "action_taken"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return out_path
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(incidents[0].keys()))
        writer.writeheader()
        writer.writerows(incidents)
    return out_path


# Ensure the table exists as soon as this module is imported
init_db()
