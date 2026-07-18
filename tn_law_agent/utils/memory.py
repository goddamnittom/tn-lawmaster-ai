"""
TN-LawMaster — Conversation Memory (SQLite)
=============================================
Lightweight, zero-dependency multi-turn conversation memory backed by
SQLite (stdlib only).

The module stores Q&A pairs keyed by *session_id* so that the
LangGraph pipeline can inject prior conversation turns as context for
the current query.

Usage::

    from tn_law_agent.utils.memory import ConversationMemory

    mem = ConversationMemory(db_path="./memory.db", max_turns=6)
    mem.save("session-abc", user_msg="What is DUI?", assistant_msg="DUI is ...")
    history = mem.load("session-abc")
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    mem.clear("session-abc")
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import List

logger = logging.getLogger(__name__)

# DDL executed once at startup
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_session_turns ON conversation_turns (session_id, id);
"""


class ConversationMemory:
    """
    SQLite-backed conversation history store.

    Thread-safe: uses a per-instance ``threading.Lock`` to serialise
    writes.  Reads are safe under Python's default serialised
    ``check_same_thread=False`` mode with WAL journal.

    Args:
        db_path:   Path to the SQLite file.  Defaults to ``./memory.db``.
                   Directories in the path are created automatically.
        max_turns: Maximum number of conversation *turns* (user + assistant
                   pairs) retained per session.  Older turns are pruned on
                   every ``save()`` call.  Defaults to ``6``.
    """

    def __init__(self, db_path: str = "./memory.db", max_turns: int = 6) -> None:
        self.db_path = db_path
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._init_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, session_id: str) -> List[dict]:
        """
        Load the most recent conversation turns for *session_id*.

        Returns a flat list of message dicts ordered oldest → newest::

            [
                {"role": "user",      "content": "What is DUI?"},
                {"role": "assistant", "content": "DUI is defined by TCA § 55-10-401 ..."},
                ...
            ]

        At most ``max_turns * 2`` rows are returned (each turn = 1 user +
        1 assistant message).
        """
        max_rows = self.max_turns * 2  # user + assistant per turn
        sql = """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (session_id, max_rows)).fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]
        except Exception as exc:
            logger.error("[ConversationMemory.load] session=%s error=%s", session_id, exc)
            return []

    def save(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """
        Persist a single Q&A turn and prune older turns beyond *max_turns*.

        Args:
            session_id:    Identifier for the conversation session.
            user_msg:      The user's question / prompt.
            assistant_msg: The assistant's reply.
        """
        insert_sql = """
            INSERT INTO conversation_turns (session_id, role, content)
            VALUES (?, ?, ?)
        """
        # Prune: keep only the most recent (max_turns * 2) rows
        prune_sql = """
            DELETE FROM conversation_turns
            WHERE session_id = ?
              AND id NOT IN (
                  SELECT id FROM conversation_turns
                  WHERE session_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
        """
        keep_rows = self.max_turns * 2  # will become max_turns after the two new inserts

        try:
            with self._lock:
                with self._connect() as conn:
                    conn.execute(insert_sql, (session_id, "user", user_msg))
                    conn.execute(insert_sql, (session_id, "assistant", assistant_msg))
                    # Prune after inserting to keep the freshest turns
                    conn.execute(prune_sql, (session_id, session_id, keep_rows))
            logger.debug(
                "[ConversationMemory.save] session=%s saved turn", session_id
            )
        except Exception as exc:
            logger.error(
                "[ConversationMemory.save] session=%s error=%s", session_id, exc
            )

    def clear(self, session_id: str) -> None:
        """
        Delete all conversation history for *session_id*.

        Args:
            session_id: The session whose history should be erased.
        """
        sql = "DELETE FROM conversation_turns WHERE session_id = ?"
        try:
            with self._lock:
                with self._connect() as conn:
                    conn.execute(sql, (session_id,))
            logger.info("[ConversationMemory.clear] cleared session=%s", session_id)
        except Exception as exc:
            logger.error(
                "[ConversationMemory.clear] session=%s error=%s", session_id, exc
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return a connection to the SQLite database with WAL mode enabled."""
        import os
        # Ensure parent directories exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the schema if it doesn't already exist."""
        try:
            with self._connect() as conn:
                conn.executescript(_CREATE_TABLE_SQL)
            logger.debug(
                "[ConversationMemory] DB initialised at %s", self.db_path
            )
        except Exception as exc:
            logger.error("[ConversationMemory._init_db] %s", exc)
            raise
