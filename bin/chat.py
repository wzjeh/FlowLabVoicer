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
from nagaki_lab.button import ButtonListener
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
    ap.add_argument("--no-button", action="store_true",
                    help="Disable the GPIO pushbutton (default: enabled if "
                         f"BUTTON_GPIO_PIN={config.BUTTON_GPIO_PIN} is set)")
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

    # Physical pushbutton input — same callback the keyboard ENTER triggers,
    # so it's a third input surface alongside ENTER and the wake word. The
    # only one of the three that survives systemd-daemon mode (no TTY).
    button = ButtonListener(
        gpio_pin=None if args.no_button else config.BUTTON_GPIO_PIN,
        on_press=loop.on_user_action,
    )
    await button.start()
    if button.available:
        print(f"[button: ready on BCM {button.gpio_pin}]")
    elif not args.no_button and config.BUTTON_GPIO_PIN is not None:
        print(f"[button: disabled — {button.error}]")

    # TerminalUI reads stdin for ENTER + slash commands, but only makes
    # sense when stdin is an actual terminal. Under systemd-daemon mode
    # stdin is /dev/null and readline() instantly returns EOF, which
    # TerminalUI treats as `/exit` and shuts the whole process down.
    # Detect that case and run loop-only; the wake word and the button
    # are still functional input paths in daemon mode.
    has_tty = sys.stdin.isatty()
    ui = TerminalUI(loop, turn_log) if has_tty else None
    if not has_tty:
        print("[no TTY — running in daemon mode; "
              "trigger via wake word or pushbutton]")

    try:
        if ui is not None:
            await asyncio.gather(loop.run(), ui.run())
        else:
            await loop.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[interrupted]")
    finally:
        await button.stop()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
