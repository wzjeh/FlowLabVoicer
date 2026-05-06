"""Mic loopback sanity test — record, show level, play back."""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from nagaki_lab import config
from nagaki_lab.audio.input import MicCapture


def play_back(samples: bytes) -> None:
    proc = subprocess.Popen(
        ["aplay", "-D", config.SPEAKER_DEVICE, "-q",
         "-r", str(config.INPUT_RATE), "-c", "1", "-f", "S16_LE", "-t", "raw"],
        stdin=subprocess.PIPE,
    )
    proc.communicate(samples)


async def main():
    seconds = 5.0
    mic = MicCapture()

    print(f"\n>>> SPEAK NOW for {seconds:.0f}s — counting down…")
    for i in range(3, 0, -1):
        print(f"  {i}…", flush=True)
        time.sleep(1)
    print("  RECORDING")

    await mic.start()
    await asyncio.sleep(seconds)
    pcm = await mic.stop()

    arr = np.frombuffer(pcm, dtype=np.int16)
    peak = int(np.max(np.abs(arr))) if len(arr) else 0
    rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if len(arr) else 0
    peak_db = 20 * np.log10(peak / 32768) if peak else -120
    rms_db = 20 * np.log10(rms / 32768) if rms else -120

    print(f"\n  samples: {len(arr)} ({mic.duration_s:.2f}s)")
    print(f"  peak   : {peak}/32767  ({peak_db:+.1f} dBFS)")
    print(f"  rms    : {rms:.0f}/32767  ({rms_db:+.1f} dBFS)")
    print()
    if peak < 200:
        print("  >>> SILENT or mic not capturing. Check BT HFP / mic mute.")
    elif peak < 5000:
        print("  >>> quiet but workable.")
    elif peak < 25000:
        print("  >>> good level.")
    else:
        print("  >>> hot — consider lowering mic gain.")
    print()

    print(">>> playing back through USB speaker…")
    play_back(pcm)
    print(">>> done.\n")


if __name__ == "__main__":
    asyncio.run(main())
