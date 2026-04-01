import sqlite3
from sqlite3 import Connection

DB_NAME = "app.db"


def get_db_connection() -> Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


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


def insert_message(session_id: str, role: str, content: str, model_used: str | None = None):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history (session_id, role, content, model_used)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, model_used))

    conn.commit()
    conn.close()


def get_chat_history(session_id: str, limit: int = 10):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, content
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ?
    """, (session_id, limit))

    rows = cursor.fetchall()
    conn.close()

    return [{"role": r["role"], "content": r["content"]} for r in rows]


create_chat_history()