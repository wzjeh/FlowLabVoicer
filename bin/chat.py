#!/usr/bin/env python3
"""bin/chat.py — voice conversation entry point.

Usage:
    python bin/chat.py
    python bin/chat.py --voice Puck
    python bin/chat.py --wake alexa
    python bin/chat.py --skip-bt-check

Wires up: bluetooth setup, mic, speaker, leds, live session, tools, memory,
conversation loop, terminal UI. Then runs.
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Make sibling package importable when running from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nagaki_lab import config, prompts
from nagaki_lab.audio.input import MicCapture
from nagaki_lab.audio.output import SpeakerPlayback
from nagaki_lab.audio import bluetooth as bt
from nagaki_lab.leds import LEDStatus
from nagaki_lab.live import LiveSession
from nagaki_lab.memory import TurnLog
from nagaki_lab.tools import get_tool_declarations, init_caches
from nagaki_lab.conversation import ConversationLoop
from nagaki_lab.ui_terminal import TerminalUI


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Nagaki Lab voice assistant")
    ap.add_argument("--voice", default=config.DEFAULT_VOICE,
                    choices=config.AVAILABLE_VOICES,
                    help="TTS voice")
    ap.add_argument("--wake", default=None,
                    help=("Wake-word name (default OFF). "
                          "Pre-trained options: alexa, hey_jarvis, hey_mycroft, hey_marvin, timer, weather. "
                          "Specify the empty arg to enable with the configured default "
                          f"({config.DEFAULT_WAKE_WORD})."))
    ap.add_argument("--wake-threshold", type=float, default=None,
                    help=(f"Wake detection threshold (0.0-1.0). Default {config.WAKE_WORD_THRESHOLD}. "
                          "Lower = more sensitive (more false positives). Try 0.3 if BT mic CVSD lowers scores."))
    ap.add_argument("--skip-bt-check", action="store_true",
                    help="Skip auto-HFP setup at startup")
    ap.add_argument("--bt-name", default=config.BT_HEADSET_NAME_SUBSTRING,
                    help="BT headset name substring (for HFP profile selection)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-event timing log lines")
    return ap.parse_args()


async def amain(args: argparse.Namespace) -> None:
    if not args.skip_bt_check:
        try:
            bt.ensure_hfp(args.bt_name)
        except Exception as e:
            print(f"[bt] ensure_hfp warning: {e}")

    init_caches(config.MEMORY_DB_PATH)

    print("[fresh session — no prior context loaded]")

    wake_detector = None
    if args.wake is not None:
        wake_name = args.wake or config.DEFAULT_WAKE_WORD
        threshold = args.wake_threshold if args.wake_threshold is not None \
            else config.WAKE_WORD_THRESHOLD
        print(f"[loading wake word '{wake_name}'…]")
        from nagaki_lab.wake import WakeWordDetector
        wake_detector = WakeWordDetector(wake_name, threshold=threshold)
        print(f"[wake detector ready (threshold {wake_detector.threshold})]")

    mic = MicCapture()
    speaker = SpeakerPlayback()
    leds = LEDStatus()
    turn_log = TurnLog(config.MEMORY_DB_PATH)
    live = LiveSession(
        system_prompt=prompts.SYSTEM_PROMPT,
        tools=get_tool_declarations(),
        voice=args.voice,
    )

    loop = ConversationLoop(
        mic=mic, speaker=speaker, live_session=live,
        turn_log=turn_log, leds=leds,
        wake_detector=wake_detector,
        verbose=not args.quiet,
    )

    print(f"[session {loop.session_id} voice={args.voice} model={config.MODEL}]")

    ui = TerminalUI(loop, turn_log)

    try:
        await asyncio.gather(
            loop.run(),
            ui.run(),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[interrupted]")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
