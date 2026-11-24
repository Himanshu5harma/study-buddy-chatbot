"""Persistent conversation memory backed by SQLite.

LangChain concept: `BaseChatMessageHistory` is the standard interface for
storing a conversation (`messages`, `add_messages`, `clear`). LangChain ships
ready-made implementations (Redis, Postgres, SQL...), but writing a small one
yourself shows exactly how memory works: it's just rows of serialized messages
keyed by a session id.

To move to Postgres/Redis in production, swap this class for the matching
LangChain integration; nothing else in the app has to change.
"""

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


class SQLiteStore:
    """Owns the SQLite connection; hands out per-session history objects."""

    def __init__(self, db_path: str):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI may call us from different threads, so allow cross-thread use
        # and serialize writes with a lock.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def history(self, session_id: str) -> "SQLiteChatMessageHistory":
        return SQLiteChatMessageHistory(self, session_id)

    def close(self) -> None:
        self._conn.close()

    # --- low-level operations used by SQLiteChatMessageHistory ---
    def _load(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT message FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return messages_from_dict([json.loads(row[0]) for row in rows])

    def _append(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO messages (session_id, message) VALUES (?, ?)",
                [(session_id, json.dumps(message_to_dict(m))) for m in messages],
            )
            self._conn.commit()

    def _clear(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.commit()


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, store: SQLiteStore, session_id: str):
        self._store = store
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        return self._store._load(self.session_id)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self._store._append(self.session_id, messages)

    def clear(self) -> None:
        self._store._clear(self.session_id)
