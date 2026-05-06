"""Persistent turn log.

We do NOT inject prior turns into the system prompt — each program start is
a fresh conversation. This module exists so the user can browse past
conversations after the fact (`/memory` slash command, or open the SQLite
file with any tool).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import aiosqlite

from . import config


class TurnLog:
    """SQLite-backed log of (timestamp, session_id, role, text) tuples."""

    def __init__(self, db_path: Path = config.MEMORY_DB_PATH):
        self.db_path = Path(db_path)

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL
                )
            """)
            await db.commit()

    async def append(self, session_id: str, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO turns (ts, session_id, role, text) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), session_id, role, text),
            )
            await db.commit()

    async def recent(self, n: int = 50) -> list[tuple[str, str, str, str]]:
        """Return [(ts, session_id, role, text), ...] oldest-first."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT ts, session_id, role, text FROM turns ORDER BY id DESC LIMIT ?",
                (n,),
            ) as cur:
                rows = await cur.fetchall()
        return list(reversed(rows))

    async def reset(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM turns")
            await db.commit()

    async def format_recent(self, n: int = 50) -> str:
        rows = await self.recent(n)
        if not rows:
            return "[memory empty]"
        lines = []
        for ts, sid, role, text in rows:
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "…"
            lines.append(f"[{ts}] [{sid[:6]}] {role}: {snippet}")
        return "\n".join(lines)
