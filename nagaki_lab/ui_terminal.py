"""Terminal UI adapter — reads stdin lines, dispatches ENTER + slash commands.

Decoupled from ConversationLoop: receives a loop instance plus a turn_log
reference, calls public methods (`on_user_action`, `request_exit`).

Slash commands recognised:
    /exit /quit       leave
    /memory           show recent persisted turns
    /reset            wipe memory.db
    /timers           list active timers
    /handle           print current resumption handle
    /state            print current state machine state
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

from . import conversation as conv_mod
from .memory import TurnLog
from .tools import timer as timer_mod


class StdinReader:
    """One thread reads stdin; lines pushed to an async queue."""

    def __init__(self):
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stop = False

    def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_event_loop()

        def _reader():
            while not self._stop:
                line = sys.stdin.readline()
                if line == "":
                    asyncio.run_coroutine_threadsafe(self.queue.put("__EOF__"), loop)
                    break
                asyncio.run_coroutine_threadsafe(
                    self.queue.put(line.rstrip("\n")), loop
                )

        self._task = asyncio.create_task(asyncio.to_thread(_reader))

    async def get(self) -> str:
        return await self.queue.get()

    def stop(self) -> None:
        self._stop = True


class TerminalUI:
    def __init__(self, loop: conv_mod.ConversationLoop, turn_log: TurnLog):
        self.loop = loop
        self.turn_log = turn_log
        self.stdin = StdinReader()

    def _prompt(self) -> str:
        s = self.loop.state
        if s == conv_mod.ConversationLoop.STATE_CAPTURING:
            return ">>> [recording — ENTER to send]  "
        if s == conv_mod.ConversationLoop.STATE_RESPONDING:
            return ">>> [waiting for reply…]  "
        if s == conv_mod.ConversationLoop.STATE_WAKE_LISTENING:
            return ">>> [wake-listening]  "
        return ">>>  "

    async def run(self) -> None:
        self.stdin.start()
        print('[ENTER = capture / send / interrupt. '
              '/exit /memory /reset /timers /handle /state]\n')
        try:
            while True:
                print(f"\n{self._prompt()}", end="", flush=True)
                line = await self.stdin.get()
                cmd = line.strip()

                if cmd == "__EOF__" or cmd in ("/exit", "/quit"):
                    print("[bye]")
                    await self.loop.request_exit()
                    return
                if cmd == "":
                    try:
                        await self.loop.on_user_action()
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        raise
                    except Exception as e:
                        print(f"\n[ui caught error: {type(e).__name__}: {e}]")
                    continue
                if cmd == "/memory":
                    print(await self.turn_log.format_recent(50))
                    continue
                if cmd == "/reset":
                    await self.turn_log.reset()
                    print("[memory cleared]")
                    continue
                if cmd == "/timers":
                    r = await timer_mod.list_active_timers({})
                    print(json.dumps(r, indent=2, ensure_ascii=False))
                    continue
                if cmd == "/handle":
                    h = self.loop.live.handle
                    print(f"[handle: {h[:32] + '…' if h else 'none yet'}]")
                    continue
                if cmd == "/state":
                    print(f"[state: {self.loop.state}]")
                    continue
                print(f"[unknown command: {cmd}]")
        finally:
            self.stdin.stop()
