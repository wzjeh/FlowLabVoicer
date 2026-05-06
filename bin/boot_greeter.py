#!/usr/bin/env python3
"""bin/boot_greeter.py — boot-time greeter run by systemd.

Plays a triple-beep through the USB speaker and (if LEDs are wired) sets
them to the 'idle' steady pattern, indicating the device is up and ready.
The script exits after running; APA102 LEDs latch their last value so they
remain lit at idle.
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nagaki_lab.leds import LEDStatus
from nagaki_lab.tools.timer import play_beep_blocking


async def amain() -> int:
    leds = LEDStatus()
    if leds.available:
        print("[boot] LEDs available")
        await leds.set_state("idle")
        await asyncio.sleep(0.2)
    else:
        print(f"[boot] LEDs unavailable: {leds.error}", file=sys.stderr)
    try:
        await asyncio.to_thread(play_beep_blocking)
    except Exception as e:
        print(f"[boot] beep failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
