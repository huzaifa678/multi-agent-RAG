import os
import sqlite3
from sqlite3 import Connection
from utils.logger import get_logger

logger = get_logger("memory-db")

DB_NAME = "app.db"


def get_db_connection() -> Connection:

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_long_term_memory():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def create_chat_history():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model_used TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    create_long_term_memory()


def insert_message(
    session_id: str, role: str, content: str, model_used: str | None = None
):
    try:
        logger.info("insert_message CALLED")
        logger.info(f"session_id={session_id}, role={role}, model_used={model_used}")

        conn = get_db_connection()

        logger.info(f"DB PATH: {conn}")

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chat_history (session_id, role, content, model_used)
            VALUES (?, ?, ?, ?)
        """,
            (session_id, role, content, model_used or "unknown"),
        )

        conn.commit()

        logger.info("INSERT COMMITTED")

        cursor.execute("SELECT changes()")
        changes = cursor.fetchone()[0]
        logger.info(f"ROWS INSERTED: {changes}")

        conn.close()
        logger.info("DB CONNECTION CLOSED")

    except Exception as e:
        logger.exception("insert_message FAILED with error: %s", str(e))
        raise


def get_chat_history(session_id: str, limit: int = 10):
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ?
    """,
        (session_id, limit),
    )

    rows = cursor.fetchall()
    conn.close()

    return [{"role": r["role"], "content": r["content"]} for r in rows]


def insert_long_term_memory(session_id: str, content: str, source: str | None = None):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO long_term_memory (session_id, content, source)
        VALUES (?, ?, ?)
    """,
        (session_id, content, source),
    )

    conn.commit()
    conn.close()


def get_long_term_memory(session_id: str, limit: int = 20):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT content, source
        FROM long_term_memory
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ?
    """,
        (session_id, limit),
    )

    rows = cursor.fetchall()
    conn.close()

    return [{"content": r["content"], "source": r["source"]} for r in rows]


create_chat_history()
