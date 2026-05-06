"""Multilingual term translation tool.

Calls a small / cheap Gemini text model under the hood, asks for strict JSON
output, and caches results in SQLite (translations of stable terms don't change).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import aiosqlite
from google import genai
from google.genai import types

from .. import config


_cache_db: Optional[Path] = None
_client: Optional[genai.Client] = None


def init_cache(db_path: Path) -> None:
    global _cache_db
    _cache_db = Path(db_path)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.read_api_key())
    return _client


async def _ensure_translation_cache_table() -> None:
    if _cache_db is None:
        return
    async with aiosqlite.connect(_cache_db) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS translation_cache (
                term      TEXT NOT NULL,
                domain    TEXT NOT NULL,
                data      TEXT NOT NULL,
                cached_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (term, domain)
            )
        """)
        await db.commit()


async def translate_term(args: dict) -> dict:
    term = str(args["term"]).strip()
    domain = str(args.get("domain", "general")).strip() or "general"
    if not term:
        return {"error": "empty term"}

    key_term = term.lower()
    key_domain = domain.lower()

    # cache lookup
    if _cache_db is not None:
        await _ensure_translation_cache_table()
        async with aiosqlite.connect(_cache_db) as db:
            async with db.execute(
                "SELECT data FROM translation_cache WHERE term=? AND domain=?",
                (key_term, key_domain),
            ) as cur:
                row = await cur.fetchone()
            if row:
                data = json.loads(row[0])
                data["_cache"] = "hit"
                return data

    prompt = (
        "You are a precise translator across Simplified Chinese, English, and Japanese.\n"
        f'TERM: "{term}"\n'
        f"DOMAIN: {domain}\n\n"
        "Detect the source language, then output STRICT JSON only (no commentary, "
        "no markdown, no surrounding text), with these exact keys:\n"
        "  source_language: one of 'zh' / 'en' / 'ja'\n"
        "  zh: Simplified Chinese rendering (no pinyin)\n"
        "  en: English rendering\n"
        "  ja: Japanese rendering, using kanji where appropriate\n"
        "  ja_reading: hiragana or katakana reading of the Japanese\n"
        "  notes: short disambiguation if multiple senses; empty string if not needed\n\n"
        "Pick the sense matching the domain. For chemistry, prefer the IUPAC / standard "
        "scientific term."
    )

    last_err = None
    for attempt in range(4):
        try:
            client = _get_client()
            resp = await client.aio.models.generate_content(
                model=config.TRANSLATE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            data = json.loads(resp.text or "{}")
            if not data:
                raise ValueError("empty translation response")

            if _cache_db is not None:
                async with aiosqlite.connect(_cache_db) as db:
                    await db.execute(
                        "INSERT OR REPLACE INTO translation_cache "
                        "(term, domain, data) VALUES (?, ?, ?)",
                        (key_term, key_domain,
                         json.dumps(data, ensure_ascii=False)),
                    )
                    await db.commit()
            return data
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = ("503" in msg or "429" in msg
                         or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg)
            if not transient or attempt == 3:
                break
            await asyncio.sleep(2 ** attempt + 1)
    return {"error": f"translation failed: {type(last_err).__name__}: {last_err}"}


# ---------- declarations + dispatch ----------
S = types.Schema
T = types.Type

DECLARATIONS = [
    types.FunctionDeclaration(
        name="translate_term",
        description=("Translate a single word or short phrase among Simplified "
                     "Chinese (zh), English (en), and Japanese (ja). "
                     "Use whenever the user asks 'how do you say X in Y' or "
                     "for a multilingual lookup of a technical term. Returns "
                     "all three languages plus a Japanese reading."),
        parameters=S(type=T.OBJECT,
                     properties={
                         "term": S(type=T.STRING,
                                   description="The word or phrase to translate."),
                         "domain": S(type=T.STRING,
                                     description="Optional: 'chemistry', 'cooking', 'general' (default)."),
                     },
                     required=["term"]),
    ),
]

DISPATCH = {
    "translate_term": translate_term,
}
