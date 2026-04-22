"""Layered SQLite memory.

Several memory tables:
  1. messages              — raw history of all exchanges (role, content, ts)
  2. summaries             — rolling summaries of older message windows
  3. facts                 — durable things to remember
  4. autonomous_messages   — log of bot-initiated messages (for daily quota)
  5. stats                 — interaction counter (for intimacy level)
  6. daily_mood            — mood picked once per day
  7. events                — recent off-chat events (decay after ~7 days)
  8. pic_log               — generated-photo log (for rate limit + cooldown)

The `Memory` class is intentionally thread-unsafe: this is a single-chat bot,
and blocking SQLite calls are isolated via `asyncio.to_thread` from the
`AsyncMemory` wrapper below.
"""
from __future__ import annotations

import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


log = logging.getLogger("diana-bot.memory")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('user','assistant')),
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    rolled_up  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, created_at);

CREATE TABLE IF NOT EXISTS summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    summary     TEXT    NOT NULL,
    covers_upto INTEGER NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_chat ON summaries(chat_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    fact       TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_chat ON facts(chat_id);

CREATE TABLE IF NOT EXISTS autonomous_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    message    TEXT    NOT NULL,
    sent_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auto_chat_sent ON autonomous_messages(chat_id, sent_at);

-- Interaction counter used to compute intimacy level.
CREATE TABLE IF NOT EXISTS stats (
    chat_id           INTEGER PRIMARY KEY,
    interaction_count INTEGER NOT NULL DEFAULT 0
);

-- Mood picked once per day (randomly from those defined in the YAML).
CREATE TABLE IF NOT EXISTS daily_mood (
    chat_id     INTEGER NOT NULL,
    date        TEXT    NOT NULL,
    mood_name   TEXT    NOT NULL,
    mood_desc   TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (chat_id, date)
);

-- Recent off-chat events added via the /event command.
-- Injected into the system prompt for ~7 days, then rotate out.
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_chat_ts ON events(chat_id, created_at);

-- Generated-photo log (manual /pic, reactive, autonomous).
-- Used for daily rate limit + cooldown.
CREATE TABLE IF NOT EXISTS pic_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    trigger_type  TEXT    NOT NULL CHECK (trigger_type IN ('manual','reactive','autonomous')),
    scene_prompt  TEXT,
    caption       TEXT,
    sent_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_piclog_chat_sent ON pic_log(chat_id, sent_at);
"""


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Memory:
    def __init__(self, db_path: Path, chat_id: int) -> None:
        self.db_path = Path(db_path)
        self.chat_id = chat_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)
        log.info("Memory ready: %s chat_id=%s", self.db_path, self.chat_id)

    # ---- messages ----

    def save_message(self, role: str, content: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO messages(chat_id, role, content, created_at) VALUES(?,?,?,?)",
                (self.chat_id, role, content, _now_iso()),
            )
            return int(cur.lastrowid)

    def recent_messages(self, limit: int = 10, include_rolled_up: bool = False) -> list[dict]:
        """Last N messages in chronological order (oldest → newest)."""
        where = "chat_id = ?"
        if not include_rolled_up:
            where += " AND rolled_up = 0"
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT id, role, content, created_at FROM messages
                WHERE {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.chat_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def last_assistant_text(self) -> str:
        """Return the content of the most recent assistant message, or '' if none."""
        with self._conn() as c:
            row = c.execute(
                "SELECT content FROM messages WHERE chat_id=? AND role='assistant' "
                "ORDER BY id DESC LIMIT 1",
                (self.chat_id,),
            ).fetchone()
        return row["content"] if row else ""

    def count_non_rolled(self) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE chat_id=? AND rolled_up=0",
                (self.chat_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def mark_rolled_up(self, up_to_id: int) -> int:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE messages SET rolled_up=1 WHERE chat_id=? AND id<=? AND rolled_up=0",
                (self.chat_id, up_to_id),
            )
            return int(cur.rowcount)

    # ---- summaries ----

    def save_summary(self, summary: str, covers_upto: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO summaries(chat_id, summary, covers_upto, created_at) VALUES(?,?,?,?)",
                (self.chat_id, summary, covers_upto, _now_iso()),
            )

    def latest_summary(self) -> Optional[str]:
        with self._conn() as c:
            row = c.execute(
                "SELECT summary FROM summaries WHERE chat_id=? ORDER BY id DESC LIMIT 1",
                (self.chat_id,),
            ).fetchone()
        return row["summary"] if row else None

    # ---- facts ----

    def save_fact(self, fact: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO facts(chat_id, fact, created_at) VALUES(?,?,?)",
                (self.chat_id, fact, _now_iso()),
            )
            return int(cur.lastrowid)

    def list_facts(self, limit: int = 50) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT fact FROM facts WHERE chat_id=? ORDER BY id ASC LIMIT ?",
                (self.chat_id, limit),
            ).fetchall()
        return [r["fact"] for r in rows]

    # ---- autonomous ----

    def log_autonomous(self, message: str) -> int:
        ts = _now_iso()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO autonomous_messages(chat_id, message, sent_at) VALUES(?,?,?)",
                (self.chat_id, message, ts),
            )
            c.execute(
                "INSERT INTO messages(chat_id, role, content, created_at) VALUES(?,?,?,?)",
                (self.chat_id, "assistant", message, ts),
            )
            return int(cur.lastrowid)

    def autonomous_count_today(self) -> int:
        """Number of autonomous messages sent today (local date)."""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM autonomous_messages WHERE chat_id=? AND substr(sent_at,1,10)=?",
                (self.chat_id, today),
            ).fetchone()
        return int(row["n"]) if row else 0

    def last_any_message_at(self) -> Optional[datetime]:
        """Timestamp of the last message (user or assistant) in this chat."""
        with self._conn() as c:
            row = c.execute(
                "SELECT created_at FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT 1",
                (self.chat_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return datetime.fromisoformat(row["created_at"])
        except ValueError:
            return None

    # ---- stats ----

    def increment_interaction(self) -> int:
        """Increment the interaction counter and return the new value."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO stats(chat_id, interaction_count) VALUES(?, 1) "
                "ON CONFLICT(chat_id) DO UPDATE SET interaction_count = interaction_count + 1",
                (self.chat_id,),
            )
            row = c.execute(
                "SELECT interaction_count FROM stats WHERE chat_id=?",
                (self.chat_id,),
            ).fetchone()
        return int(row["interaction_count"]) if row else 0

    def get_interaction_count(self) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT interaction_count FROM stats WHERE chat_id=?",
                (self.chat_id,),
            ).fetchone()
        return int(row["interaction_count"]) if row else 0

    # ---- daily mood ----

    def get_today_mood(self) -> Optional[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._conn() as c:
            row = c.execute(
                "SELECT mood_name, mood_desc FROM daily_mood WHERE chat_id=? AND date=?",
                (self.chat_id, today),
            ).fetchone()
        if row is None:
            return None
        return {"name": row["mood_name"], "desc": row["mood_desc"]}

    def set_today_mood(self, name: str, desc: str) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO daily_mood(chat_id, date, mood_name, mood_desc, created_at) "
                "VALUES(?,?,?,?,?)",
                (self.chat_id, today, name, desc, _now_iso()),
            )

    # ---- events ----

    def save_event(self, text: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO events(chat_id, text, created_at) VALUES(?,?,?)",
                (self.chat_id, text, _now_iso()),
            )
            return int(cur.lastrowid)

    def recent_events(self, days: int = 7, limit: int = 50) -> list[dict]:
        """Events from the last `days` days, newest first."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, text, created_at FROM events "
                "WHERE chat_id=? AND created_at >= ? ORDER BY id DESC LIMIT ?",
                (self.chat_id, cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- pic_log (photo rate limit) ----

    def log_pic(self, trigger_type: str, scene_prompt: str, caption: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO pic_log(chat_id, trigger_type, scene_prompt, caption, sent_at) "
                "VALUES(?,?,?,?,?)",
                (self.chat_id, trigger_type, scene_prompt, caption, _now_iso()),
            )
            return int(cur.lastrowid)

    def pic_count_today(self) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM pic_log WHERE chat_id=? AND substr(sent_at,1,10)=?",
                (self.chat_id, today),
            ).fetchone()
        return int(row["n"]) if row else 0

    def last_pic_at(self) -> Optional[datetime]:
        with self._conn() as c:
            row = c.execute(
                "SELECT sent_at FROM pic_log WHERE chat_id=? ORDER BY id DESC LIMIT 1",
                (self.chat_id,),
            ).fetchone()
        if not row:
            return None
        try:
            return datetime.fromisoformat(row["sent_at"])
        except ValueError:
            return None


# Thin async wrapper: blocking SQLite calls run in a thread pool.
class AsyncMemory:
    def __init__(self, mem: Memory) -> None:
        self._m = mem

    @property
    def sync(self) -> Memory:
        return self._m

    async def save_message(self, role: str, content: str) -> int:
        return await asyncio.to_thread(self._m.save_message, role, content)

    async def recent_messages(self, limit: int = 10) -> list[dict]:
        return await asyncio.to_thread(self._m.recent_messages, limit)

    async def last_assistant_text(self) -> str:
        return await asyncio.to_thread(self._m.last_assistant_text)

    async def list_facts(self, limit: int = 50) -> list[str]:
        return await asyncio.to_thread(self._m.list_facts, limit)

    async def save_fact(self, fact: str) -> int:
        return await asyncio.to_thread(self._m.save_fact, fact)

    async def latest_summary(self) -> Optional[str]:
        return await asyncio.to_thread(self._m.latest_summary)

    async def log_autonomous(self, message: str) -> int:
        return await asyncio.to_thread(self._m.log_autonomous, message)

    async def autonomous_count_today(self) -> int:
        return await asyncio.to_thread(self._m.autonomous_count_today)

    async def last_any_message_at(self) -> Optional[datetime]:
        return await asyncio.to_thread(self._m.last_any_message_at)

    async def increment_interaction(self) -> int:
        return await asyncio.to_thread(self._m.increment_interaction)

    async def get_interaction_count(self) -> int:
        return await asyncio.to_thread(self._m.get_interaction_count)

    async def get_today_mood(self) -> Optional[dict]:
        return await asyncio.to_thread(self._m.get_today_mood)

    async def set_today_mood(self, name: str, desc: str) -> None:
        return await asyncio.to_thread(self._m.set_today_mood, name, desc)

    async def save_event(self, text: str) -> int:
        return await asyncio.to_thread(self._m.save_event, text)

    async def recent_events(self, days: int = 7, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._m.recent_events, days, limit)

    async def log_pic(self, trigger_type: str, scene_prompt: str, caption: str) -> int:
        return await asyncio.to_thread(self._m.log_pic, trigger_type, scene_prompt, caption)

    async def pic_count_today(self) -> int:
        return await asyncio.to_thread(self._m.pic_count_today)

    async def last_pic_at(self) -> Optional[datetime]:
        return await asyncio.to_thread(self._m.last_pic_at)
