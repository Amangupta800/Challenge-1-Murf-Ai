import sqlite3
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# DB will live at: project-root/shared-data/day6_fraud.db
DB_PATH = Path(__file__).parent.parent / "shared-data" / "day6_fraud.db"


def _get_conn():
    """Open a SQLite connection with row_factory set to Row."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # returns dict-like rows
    return conn


def init_fraud_db():
    """Create table and one demo case if not exists."""
    conn = _get_conn()
    cur = conn.cursor()

    # Create table if needed
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userName TEXT,
            securityIdentifier TEXT,
            cardEnding TEXT,
            transactionAmount TEXT,
            transactionName TEXT,
            transactionLocation TEXT,
            transactionTime TEXT,
            transactionCategory TEXT,
            transactionSource TEXT,
            verificationQuestion TEXT,
            verificationAnswer TEXT,
            status TEXT,
            outcomeNote TEXT,
            lastUpdated TEXT
        );
        """
    )

    # If table is empty, insert one fake case
    cur.execute("SELECT COUNT(*) AS c FROM fraud_cases;")
    row = cur.fetchone()
    if row is None or row["c"] == 0:
        logger.info("fraud_cases empty, inserting demo row")
        cur.execute(
            """
            INSERT INTO fraud_cases (
                userName,
                securityIdentifier,
                cardEnding,
                transactionAmount,
                transactionName,
                transactionLocation,
                transactionTime,
                transactionCategory,
                transactionSource,
                verificationQuestion,
                verificationAnswer,
                status,
                outcomeNote,
                lastUpdated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Aman Gupta",
                "SEC-12345",
                "4242",
                "₹4,999.00",
                "ABC Industries",
                "Mumbai, IN",
                "2025-11-26T10:15:00",
                "e-commerce",
                "alibaba.com",
                "What is your favorite color?",
                "blue",
                "pending_review",
                "",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    conn.commit()
    conn.close()


def load_fraud_case() -> dict:
    """Load a single fraud case (for MVP we just take the first row)."""
    # ensure DB & demo row exist
    init_fraud_db()

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM fraud_cases ORDER BY id LIMIT 1;")
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise RuntimeError("No fraud cases found in SQLite DB")

    return dict(row)


def save_fraud_case(case: dict) -> None:
    """Write updates back to the same row."""
    if "id" not in case:
        raise ValueError("Case dict must contain 'id'")

    conn = _get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE fraud_cases
        SET
            userName = ?,
            securityIdentifier = ?,
            cardEnding = ?,
            transactionAmount = ?,
            transactionName = ?,
            transactionLocation = ?,
            transactionTime = ?,
            transactionCategory = ?,
            transactionSource = ?,
            verificationQuestion = ?,
            verificationAnswer = ?,
            status = ?,
            outcomeNote = ?,
            lastUpdated = ?
        WHERE id = ?
        """,
        (
            case.get("userName"),
            case.get("securityIdentifier"),
            case.get("cardEnding"),
            case.get("transactionAmount"),
            case.get("transactionName"),
            case.get("transactionLocation"),
            case.get("transactionTime"),
            case.get("transactionCategory"),
            case.get("transactionSource"),
            case.get("verificationQuestion"),
            case.get("verificationAnswer"),
            case.get("status"),
            case.get("outcomeNote"),
            case.get("lastUpdated", datetime.now().isoformat(timespec="seconds")),
            case["id"],
        ),
    )

    conn.commit()
    conn.close()
