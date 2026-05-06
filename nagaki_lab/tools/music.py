"""Music playback tool.

Plays audio files from ~/voice/music/ via ffplay (lightweight, supports
mp3 / m4a / flac / wav / ogg / opus, no GUI window).

Usage by the model:
  play_music(title="一笑江湖")    # default if no title given
  play_music()                    # plays DEFAULT_TITLE
  stop_music()
  list_music()

Music files: drop them into ~/voice/music/ named like
    一笑江湖.mp3
    YourSong.m4a
File-stem case-insensitive substring match is used, so "一笑" → "一笑江湖.mp3".

Single-playback state: starting a new song stops any previous one.
"""
from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Optional

from google.genai import types

from .. import config


MUSIC_DIR = config.DATA_DIR / "music"
MUSIC_EXTENSIONS = (".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus")
DEFAULT_TITLE = "一笑江湖"

# Single-playback state.
_current_proc: Optional[subprocess.Popen] = None
_current_path: Optional[Path] = None


def _list_files() -> list[Path]:
    if not MUSIC_DIR.exists():
        return []
    return sorted(p for p in MUSIC_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in MUSIC_EXTENSIONS)


def _find(title: str) -> Optional[Path]:
    """Case-insensitive: exact stem match first, then substring."""
    needle = title.strip().lower()
    files = _list_files()
    if not files:
        return None
    for p in files:
        if p.stem.lower() == needle:
            return p
    for p in files:
        if needle in p.stem.lower():
            return p
    return None


def _stop_current() -> None:
    global _current_proc, _current_path
    if _current_proc is not None and _current_proc.poll() is None:
        # If the process is currently SIGSTOPped (e.g. paused while the model
        # is speaking), SIGTERM cannot be delivered until we SIGCONT it first.
        try:
            _current_proc.send_signal(signal.SIGCONT)
        except Exception:
            pass
        try:
            _current_proc.terminate()
            _current_proc.wait(timeout=1.0)
        except Exception:
            try:
                _current_proc.kill()
            except Exception:
                pass
    _current_proc = None
    _current_path = None


# ---------- module-level pause/resume (called by the conversation loop
# to duck music while the model is speaking) ----------
def is_playing() -> bool:
    return _current_proc is not None and _current_proc.poll() is None


def pause_current() -> bool:
    """Pause the currently playing track (SIGSTOP). True if a process was paused."""
    if not is_playing():
        return False
    try:
        _current_proc.send_signal(signal.SIGSTOP)  # type: ignore[union-attr]
        return True
    except Exception:
        return False


def resume_current() -> bool:
    """Resume a previously SIGSTOPped track. True if a process was resumed."""
    if not is_playing():
        return False
    try:
        _current_proc.send_signal(signal.SIGCONT)  # type: ignore[union-attr]
        return True
    except Exception:
        return False


# ---------- tool implementations ----------
async def play_music(args: dict) -> dict:
    global _current_proc, _current_path

    title = str(args.get("title") or DEFAULT_TITLE).strip() or DEFAULT_TITLE

    # Default to looping forever — kitchen / lab background music is meant
    # to keep going. The user can ask "play once" to disable, or "stop music".
    loop_arg = args.get("loop", True)
    if isinstance(loop_arg, str):
        loop = loop_arg.strip().lower() not in ("false", "0", "no", "off", "")
    else:
        loop = bool(loop_arg)

    if not MUSIC_DIR.exists():
        return {
            "error": f"music directory does not exist yet: {MUSIC_DIR}. "
                     f"Create it (`mkdir -p {MUSIC_DIR}`) and drop "
                     ".mp3/.m4a/.wav files there.",
        }

    target = _find(title)
    if target is None:
        return {
            "error": f"no song matching {title!r} in {MUSIC_DIR}",
            "available_tracks": [p.stem for p in _list_files()],
        }

    # Stop any previous playback before starting new
    if _current_proc is not None and _current_proc.poll() is None:
        _stop_current()

    # ffplay -loop 0 = infinite repeat. -loop 1 = play once.
    # NB: -autoexit stays in either case so a one-shot play exits cleanly.
    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    if loop:
        cmd += ["-loop", "0"]
    cmd.append(str(target))

    try:
        _current_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _current_path = target
    except FileNotFoundError:
        return {"error": "ffplay not found (install with: sudo apt install ffmpeg)"}
    except Exception as e:
        return {"error": f"playback failed: {type(e).__name__}: {e}"}

    return {
        "playing": target.stem,
        "file": str(target),
        "looping": loop,
        "note": "playback started; will play through the USB speaker. "
                + ("Will loop until user asks to stop. "
                   if loop else "Will play once and end. ")
                + "Tell the user briefly. They can ask to 'stop music' anytime.",
    }


async def stop_music(args: dict) -> dict:
    global _current_proc, _current_path
    if _current_proc is None or _current_proc.poll() is not None:
        return {"stopped": False, "note": "no music was playing"}
    title = _current_path.stem if _current_path else "unknown"
    _stop_current()
    return {"stopped": True, "title": title}


async def list_music(args: dict) -> dict:
    files = _list_files()
    return {
        "music_directory": str(MUSIC_DIR),
        "track_count": len(files),
        "tracks": [p.stem for p in files],
        "default_title": DEFAULT_TITLE,
    }


# ---------- declarations + dispatch ----------
S = types.Schema
T = types.Type

DECLARATIONS = [
    types.FunctionDeclaration(
        name="play_music",
        description=(
            "Play a song from the device's local music library at "
            f"{MUSIC_DIR}. Default song is '{DEFAULT_TITLE}' if no title given. "
            "Matching is case-insensitive substring on the filename stem, so "
            "the user can ask for a partial title (e.g. '一笑' matches "
            "'一笑江湖.mp3'). By DEFAULT the track loops forever — only set "
            "loop=false if the user explicitly says 'play once' / '只播一遍' / "
            "'一回だけ'. After calling, briefly tell the user what is "
            "playing in their language."
        ),
        parameters=S(type=T.OBJECT,
                     properties={
                         "title": S(type=T.STRING,
                                    description=f"Song title or part of it. "
                                                f"Default: '{DEFAULT_TITLE}'."),
                         "loop": S(type=T.BOOLEAN,
                                   description="Whether to repeat the song "
                                               "indefinitely. Default true. "
                                               "Set false only when the user "
                                               "asks for a single play."),
                     }),
    ),
    types.FunctionDeclaration(
        name="stop_music",
        description="Stop any music currently being played by play_music.",
        parameters=S(type=T.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="list_music",
        description="List the songs available in the device's music library.",
        parameters=S(type=T.OBJECT, properties={}),
    ),
]

DISPATCH = {
    "play_music": play_music,
    "stop_music": stop_music,
    "list_music": list_music,
}
