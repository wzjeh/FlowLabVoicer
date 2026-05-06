"""Speaker playback via aplay, with pre-buffer to suppress underruns.

`SpeakerPlayback` wraps an `aplay` subprocess. When chunks arrive faster than
they play (network jitter, model token-by-token TTS) aplay's internal buffer
can drain while we wait for the next chunk → "underrun!!! 4.2 sec" prints.

Fix: accumulate the FIRST `prebuffer_ms` of received audio in our process
before opening aplay. Once we hit the threshold, open aplay and dump the
prebuffer in one go (filling aplay's internal queue), then stream subsequent
chunks normally. This gives aplay enough head-start to absorb later jitter.

API:
  - write(data): forward a chunk; opens aplay once prebuffer is full
  - close():     close stdin, flush any pending prebuffer, wait for aplay to drain
  - abort():     immediate stop — kill aplay, drop any pending prebuffer
"""
from __future__ import annotations

import subprocess
from typing import Optional

from .. import config


class SpeakerPlayback:

    DEFAULT_PREBUFFER_MS = 200   # accumulate this much before opening aplay

    def __init__(
        self,
        device: str = config.SPEAKER_DEVICE,
        sample_rate: int = config.OUTPUT_RATE,
        channels: int = config.OUTPUT_CHANNELS,
        prebuffer_ms: int = DEFAULT_PREBUFFER_MS,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self._prebuffer_target_bytes = int(
            sample_rate * channels * 2 * prebuffer_ms / 1000
        )
        self._proc: Optional[subprocess.Popen] = None
        self._prebuffer = bytearray()

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _open_proc(self) -> None:
        # aplay still prints "underrun!!! ..." and similar non-fatal warnings
        # to stderr even with -q. Those underruns happen routinely when the
        # network stalls a chunk and don't mean playback failed; redirect to
        # /dev/null so the chat console stays clean. (If you ever need to
        # debug audio, replace DEVNULL with subprocess.PIPE temporarily.)
        self._proc = subprocess.Popen(
            ["aplay", "-D", self.device, "-q",
             "-r", str(self.sample_rate), "-c", str(self.channels),
             "-f", "S16_LE", "-t", "raw"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _write_raw(self, pcm: bytes) -> None:
        if not pcm or self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(pcm)
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if self._proc is None:
            self._prebuffer.extend(pcm)
            if len(self._prebuffer) >= self._prebuffer_target_bytes:
                self._open_proc()
                self._write_raw(bytes(self._prebuffer))
                self._prebuffer.clear()
            return
        # already streaming
        self._write_raw(pcm)

    def close(self) -> None:
        """Close stdin and wait for aplay to drain. If the response was so
        short that we never hit the prebuffer threshold, open aplay just to
        play the remaining buffer."""
        if self._proc is None and self._prebuffer:
            self._open_proc()
            self._write_raw(bytes(self._prebuffer))
            self._prebuffer.clear()
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10.0)
        except Exception:
            pass
        self._proc = None

    def abort(self) -> None:
        """Kill aplay immediately. Drop any pending prebuffer (user wants silence now)."""
        self._prebuffer.clear()
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=1.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
