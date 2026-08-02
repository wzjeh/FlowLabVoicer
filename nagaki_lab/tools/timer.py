"""Lab-bench countdown timers.

Timers run in-memory only — they do not persist across program restarts.
When a timer fires it triple-beeps through the USB speaker via aplay.
"""
from __future__ import annotations

import asyncio
import math
import struct
import subprocess
import time
from datetime import datetime
from typing import Any

from google.genai import types

from .. import config


# ---------- beep generator ----------
def _make_beep_pcm(freq: float = 880.0, duration: float = 0.4,
                   sample_rate: int = config.OUTPUT_RATE,
                   gain: float = 0.4) -> bytes:
    n = int(sample_rate * duration)
    fade = max(1, int(0.05 * sample_rate))
    out = []
    for i in range(n):
        if i < fade:
            env = i / fade
        elif i > n - fade:
            env = (n - i) / fade
        else:
            env = 1.0
        v = int(gain * env * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        out.append(v)
    return struct.pack(f"<{n}h", *out)


_BEEP = _make_beep_pcm()
_GAP = b"\x00\x00" * int(config.OUTPUT_RATE * 0.15)
_BEEP_SEQ = (_BEEP + _GAP) * 3

# Short, higher-pitched single tick for wake-word acknowledgement —
# distinct from the timer's 880 Hz triple-beep so the user can tell them
# apart by ear. Used by ConversationLoop._on_wake_detected.
_TICK = _make_beep_pcm(freq=1320.0, duration=0.12, gain=0.30)

# Two-tone descending "uh-oh" for turn failure (upload 1011, mid-turn
# disconnect). Low + descending so it reads as "error / try again",
# clearly distinct from the wake tick (high, single) and the timer beep
# (mid, triple). Used by ConversationLoop when a turn produces no reply —
# the whole point is that the user hears *something* instead of silence.
_ERR_GAP = b"\x00\x00" * int(config.OUTPUT_RATE * 0.08)
_ERROR_SEQ = (_make_beep_pcm(freq=440.0, duration=0.16, gain=0.35)
              + _ERR_GAP
              + _make_beep_pcm(freq=330.0, duration=0.22, gain=0.35))


def _aplay_pcm_blocking(pcm: bytes) -> None:
    """Pipe a raw PCM blob through aplay synchronously, swallowing aplay's
    underrun / overrun stderr chatter that otherwise spams the console."""
    proc = subprocess.Popen(
        ["aplay", "-D", config.SPEAKER_DEVICE, "-q",
         "-r", str(config.OUTPUT_RATE), "-c", "1", "-f", "S16_LE", "-t", "raw"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    proc.communicate(pcm)


def play_beep_blocking() -> None:
    """Play the triple beep synchronously (timer expiry)."""
    _aplay_pcm_blocking(_BEEP_SEQ)


def play_tick_blocking() -> None:
    """Play a single short higher-pitch tick (wake-word ack)."""
    _aplay_pcm_blocking(_TICK)


def play_error_blocking() -> None:
    """Play the two-tone descending error cue (turn failed, please retry)."""
    _aplay_pcm_blocking(_ERROR_SEQ)


# ---------- timer registry ----------
_timers: dict[int, dict[str, Any]] = {}
_timer_seq = 0
_timer_lock = asyncio.Lock()


async def _timer_task(tid: int, seconds: float, label: str) -> None:
    try:
        await asyncio.sleep(seconds)
        info = _timers.pop(tid, None)
        if info is not None:
            print(f"\n*** TIMER FIRED [{tid}] {label or '(no label)'} ***")
            await asyncio.to_thread(play_beep_blocking)
    except asyncio.CancelledError:
        pass


# ---------- tool implementations ----------
async def set_timer(args: dict) -> dict:
    global _timer_seq
    seconds = float(args["seconds"])
    label = str(args.get("label", "")).strip()
    async with _timer_lock:
        _timer_seq += 1
        tid = _timer_seq
        task = asyncio.create_task(_timer_task(tid, seconds, label))
        _timers[tid] = {
            "id": tid, "label": label, "seconds": seconds,
            "started_at": time.time(), "task": task,
        }
    fires_at = datetime.fromtimestamp(time.time() + seconds).strftime("%H:%M:%S")
    return {"timer_id": tid, "label": label, "seconds": seconds, "fires_at": fires_at}


async def list_active_timers(args: dict) -> dict:
    now = time.time()
    out = []
    for t in _timers.values():
        out.append({
            "timer_id": t["id"], "label": t["label"],
            "remaining_seconds": round(t["started_at"] + t["seconds"] - now, 1),
        })
    return {"active_timers": out}


async def cancel_timer(args: dict) -> dict:
    tid = int(args["timer_id"])
    info = _timers.pop(tid, None)
    if info is None:
        return {"cancelled": False, "reason": "no such timer"}
    info["task"].cancel()
    return {"cancelled": True, "timer_id": tid, "label": info["label"]}


async def current_time(args: dict) -> dict:
    return {
        "iso": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(time.time()),
    }


# ---------- declarations + dispatch ----------
S = types.Schema
T = types.Type

DECLARATIONS = [
    types.FunctionDeclaration(
        name="set_timer",
        description=("Start a countdown timer that beeps when expired. "
                     "Use for lab time-keeping (reaction quenches, holds, residence-time waits). "
                     "Convert minutes/hours to seconds before calling."),
        parameters=S(type=T.OBJECT,
                     properties={
                         "seconds": S(type=T.NUMBER, description="Duration in seconds."),
                         "label": S(type=T.STRING,
                                    description="Short tag e.g. 'reaction quench'."),
                     },
                     required=["seconds"]),
    ),
    types.FunctionDeclaration(
        name="list_active_timers",
        description="List all currently running timers with remaining seconds.",
        parameters=S(type=T.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="cancel_timer",
        description="Cancel a running timer by its ID.",
        parameters=S(type=T.OBJECT,
                     properties={
                         "timer_id": S(type=T.INTEGER, description="The id returned by set_timer."),
                     },
                     required=["timer_id"]),
    ),
    types.FunctionDeclaration(
        name="current_time",
        description="Get the current local wall-clock time on the device.",
        parameters=S(type=T.OBJECT, properties={}),
    ),
]

DISPATCH = {
    "set_timer": set_timer,
    "list_active_timers": list_active_timers,
    "cancel_timer": cancel_timer,
    "current_time": current_time,
}
