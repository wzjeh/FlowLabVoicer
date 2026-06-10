"""Tools exposed to the LLM via Live API function-calling.

Each domain module (timer, chemistry, translation) defines:
  - async functions that take an args dict and return a result dict
  - DISPATCH:    {name: function}
  - DECLARATIONS: list[FunctionDeclaration]

This `__init__` aggregates them so the live session can pass a single
unified Tool with all FunctionDeclarations.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from google.genai import types

from . import chemistry, music, system, timer, translation

# Aggregate dispatch table
DISPATCH: dict[str, Callable[..., Awaitable[dict]]] = {
    **timer.DISPATCH,
    **chemistry.DISPATCH,
    **translation.DISPATCH,
    **system.DISPATCH,
    **music.DISPATCH,
}


def get_tool_declarations() -> list[types.Tool]:
    """All tools for the Live session: Google Search grounding + one Tool
    object containing every local FunctionDeclaration.

    google_search gives the model live web data (weather, news, prices,
    recent events) so general questions get real answers instead of
    refusals — verified to coexist with function_declarations on
    gemini-2.5-flash-native-audio (2026-06-10). Search runs server-side;
    nothing to dispatch locally.
    """
    decls: list[types.FunctionDeclaration] = []
    decls.extend(timer.DECLARATIONS)
    decls.extend(chemistry.DECLARATIONS)
    decls.extend(translation.DECLARATIONS)
    decls.extend(system.DECLARATIONS)
    decls.extend(music.DECLARATIONS)
    return [
        types.Tool(google_search=types.GoogleSearch()),
        types.Tool(function_declarations=decls),
    ]


def init_caches(db_path) -> None:
    """Wire the SQLite path into modules that need a cache (chemistry, translation)."""
    chemistry.init_cache(db_path)
    translation.init_cache(db_path)
