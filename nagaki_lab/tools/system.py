"""System-level controls callable by the LLM.

Currently: speaker volume control via PipeWire's `wpctl`. Operates on the
default audio sink, which on this device is the USB Speaker (set up at
startup by the bluetooth helper). Mute and absolute-level control included
so the model can react to phrases like '音量大一点', '调到 60', 'mute',
'静音', 'unmute', 'set volume to 30 percent', etc.

Future system tools (brightness, mic mute, restart timer service, etc.) go
in this same module.
"""
from __future__ import annotations

import re
import subprocess

from google.genai import types


# wpctl symbolic name for the current default sink
_DEFAULT_SINK = "@DEFAULT_AUDIO_SINK@"


def _wpctl(*args: str, timeout: float = 2.0) -> tuple[int, str]:
    """Run wpctl and return (returncode, combined-output)."""
    try:
        r = subprocess.run(
            ["wpctl", *args], capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return 127, "wpctl not found (PipeWire utilities not installed)"
    except subprocess.TimeoutExpired:
        return 124, "wpctl timed out"
    except Exception as e:
        return 1, f"wpctl error: {e}"


def _read_volume() -> tuple[float, bool, str | None]:
    """Returns (level_0_to_1, is_muted, error_or_None)."""
    rc, out = _wpctl("get-volume", _DEFAULT_SINK)
    if rc != 0:
        return 0.0, False, out.strip()
    m = re.search(r"Volume:\s*([\d.]+)", out)
    if not m:
        return 0.0, False, f"could not parse wpctl output: {out!r}"
    return float(m.group(1)), ("MUTED" in out), None


def _write_volume(level_0_to_1: float) -> str | None:
    rc, out = _wpctl("set-volume", _DEFAULT_SINK, f"{max(0.0, min(1.5, level_0_to_1)):.2f}")
    return None if rc == 0 else out.strip()


def _write_mute(muted: bool) -> str | None:
    rc, out = _wpctl("set-mute", _DEFAULT_SINK, "1" if muted else "0")
    return None if rc == 0 else out.strip()


# ---------- tool implementation ----------
async def control_volume(args: dict) -> dict:
    action = str(args.get("action", "")).strip().lower()
    if action not in ("up", "down", "set", "mute", "unmute", "get"):
        return {"error": "action must be one of: up, down, set, mute, unmute, get"}

    cur_lvl, cur_muted, err = _read_volume()
    if err is not None:
        return {"error": err}
    cur_pct = round(cur_lvl * 100)

    if action == "get":
        return {"volume_percent": cur_pct, "muted": cur_muted}

    if action == "mute":
        if cur_muted:
            return {"volume_percent": cur_pct, "muted": True, "note": "already muted"}
        e = _write_mute(True)
        if e:
            return {"error": e}
        return {"volume_percent": cur_pct, "muted": True}

    if action == "unmute":
        if not cur_muted:
            return {"volume_percent": cur_pct, "muted": False, "note": "already unmuted"}
        e = _write_mute(False)
        if e:
            return {"error": e}
        return {"volume_percent": cur_pct, "muted": False}

    # up / down / set: need percent
    try:
        percent = int(args.get("percent", 10))
    except (TypeError, ValueError):
        return {"error": "percent must be an integer"}

    if action == "up":
        new_lvl = min(1.0, cur_lvl + percent / 100.0)
    elif action == "down":
        new_lvl = max(0.0, cur_lvl - percent / 100.0)
    else:  # set
        percent = max(0, min(100, percent))
        new_lvl = percent / 100.0

    e = _write_volume(new_lvl)
    if e:
        return {"error": e}

    # If setting volume on a muted device, helpfully unmute too.
    if cur_muted and new_lvl > 0:
        _write_mute(False)
        cur_muted = False

    return {
        "action": action,
        "before_percent": cur_pct,
        "after_percent": round(new_lvl * 100),
        "delta_percent": round(new_lvl * 100) - cur_pct,
        "muted": cur_muted,
    }


# ---------- declarations + dispatch ----------
S = types.Schema
T = types.Type

DECLARATIONS = [
    types.FunctionDeclaration(
        name="control_volume",
        description=(
            "Adjust or query the speaker output volume of this device. "
            "Use when the user asks to change volume ('声音大一点', '小声一点', "
            "'mute', '静音', '调到 50%', 'lower the volume', etc.). "
            "Actions: 'up' (raise by `percent`, default 10), 'down' (lower by "
            "`percent`, default 10), 'set' (set to absolute `percent`, 0-100), "
            "'mute', 'unmute', 'get' (just query current level). "
            "After calling, briefly tell the user the new level (e.g. '已调到 60'). "
            "If the user just says 'a bit louder/softer' without a number, use "
            "the default 10. If they say 'much louder/softer', use 25-30."
        ),
        parameters=S(type=T.OBJECT,
                     properties={
                         "action": S(type=T.STRING,
                                     description="One of: up, down, set, mute, unmute, get."),
                         "percent": S(type=T.INTEGER,
                                      description="0-100. For 'up'/'down' = delta from current "
                                                  "(default 10). For 'set' = absolute level. "
                                                  "Ignored for mute/unmute/get."),
                     },
                     required=["action"]),
    ),
]

DISPATCH = {
    "control_volume": control_volume,
}
