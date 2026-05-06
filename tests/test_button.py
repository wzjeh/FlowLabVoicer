"""Standalone hardware test for the OBSF-30-K (or any momentary GPIO button).

Wire the button to BUTTON_GPIO_PIN (default BCM 17, header pin 11) and any
GND, then::

    .venv/bin/python tests/test_button.py

Each press should print one ``[BUTTON] press #N at t+...s`` line.
Ctrl+C to exit; final line shows total presses.

If start fails (missing gpiozero, pin in use by another process, wrong
pin), the script prints an error and exits without hanging.
"""
import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nagaki_lab import config
from nagaki_lab.button import ButtonListener


async def main() -> int:
    count = 0
    t0 = time.monotonic()

    async def on_press() -> None:
        nonlocal count
        count += 1
        print(f"[BUTTON] press #{count} at t+{time.monotonic() - t0:.2f}s")

    print(f"[test] watching BCM {config.BUTTON_GPIO_PIN} (header pin "
          f"{_bcm_to_header(config.BUTTON_GPIO_PIN)})")
    print("[test] press the button. Ctrl+C to quit.\n")

    async with ButtonListener(on_press=on_press) as button:
        if not button.available:
            print(f"[ERROR] button setup failed: {button.error}")
            print("\nDiagnostics:")
            print(f"  - configured pin: BCM {config.BUTTON_GPIO_PIN}")
            print("  - is gpiozero installed?  .venv/bin/pip show gpiozero")
            print("  - is the pin already used by another driver?")
            return 1
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

    print(f"\n[test] total presses: {count}")
    return 0


def _bcm_to_header(bcm: int) -> str:
    """Approximate BCM-to-header-pin lookup for the common GPIO pins,
    just to make the diagnostic message friendlier. Falls back to '?'."""
    table = {
        2: "3",  3: "5",  4: "7",  17: "11",  27: "13",  22: "15",
        10: "19", 9: "21", 11: "23", 0: "27", 5: "29",  6: "31",
        13: "33", 19: "35", 26: "37",
        14: "8", 15: "10", 18: "12", 23: "16", 24: "18", 25: "22",
        8: "24", 7: "26", 1: "28", 12: "32", 16: "36", 20: "38", 21: "40",
    }
    return table.get(bcm, "?")


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
