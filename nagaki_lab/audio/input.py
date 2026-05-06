"""Microphone capture, two flavours.

  MicCapture  — one-shot: open / record / close, returns full PCM buffer.
                Used for the manual ENTER-to-ENTER push-to-talk flow.

  MicStream   — continuous: open once, async-iterate int16 chunks.
                Used for wake-word mode where the mic must stay live for
                wake detection AND can also feed a capture buffer when
                CAPTURING state is active.

Both wrap sounddevice's `InputStream`. They do not know anything about wake
words, capture buffering, or the Live API — those are orchestrator concerns.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional

import numpy as np
import sounddevice as sd

from .. import config


class MicCapture:
    def __init__(
        self,
        sample_rate: int = config.INPUT_RATE,
        channels: int = config.INPUT_CHANNELS,
        block_ms: int = config.INPUT_BLOCK_MS,
        max_seconds: float = config.MAX_CAPTURE_SECONDS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_ms = block_ms
        self.max_seconds = max_seconds

        self._buf = bytearray()
        self._peak_rms = 0.0
        self._last_chunk_rms = 0.0
        self._stop_evt: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._capping = False  # True if we hit max_seconds

    @property
    def captured_pcm(self) -> bytes:
        return bytes(self._buf)

    @property
    def peak_rms(self) -> float:
        return self._peak_rms

    @property
    def last_chunk_rms(self) -> float:
        return self._last_chunk_rms

    @property
    def duration_s(self) -> float:
        return len(self._buf) / 2 / self.sample_rate / self.channels

    @property
    def hit_cap(self) -> bool:
        return self._capping

    async def start(self) -> None:
        if self._task is not None:
            return
        self._buf = bytearray()
        self._peak_rms = 0.0
        self._last_chunk_rms = 0.0
        self._capping = False
        self._stop_evt = asyncio.Event()
        self._task = asyncio.create_task(self._record())

    async def stop(self) -> bytes:
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            self._task = None
        return self.captured_pcm

    async def _record(self) -> None:
        block = int(self.sample_rate * self.block_ms / 1000)
        buf = self._buf

        def callback(indata, frames, time_info, status):
            try:
                arr = indata.flatten() if hasattr(indata, "flatten") else \
                    np.frombuffer(indata, dtype=np.int16)
                if len(arr) == 0:
                    return
                sq = arr.astype(np.float32) ** 2
                rms = float(np.sqrt(sq.mean()))
                self._last_chunk_rms = rms
                if rms > self._peak_rms:
                    self._peak_rms = rms
                buf.extend(arr.tobytes())
            except Exception:
                pass

        try:
            with sd.InputStream(
                samplerate=self.sample_rate, channels=self.channels,
                dtype="int16", blocksize=block, callback=callback,
            ):
                t0 = time.monotonic()
                while not self._stop_evt.is_set():
                    await asyncio.sleep(0.05)
                    if time.monotonic() - t0 > self.max_seconds:
                        self._capping = True
                        return
        except asyncio.CancelledError:
            pass


class MicStream:
    """Continuous mic source as an async iterator of int16 ndarray chunks.

    Use as an async context manager:

        async with MicStream() as mic:
            async for chunk in mic.chunks():
                ...               # state-dispatch what to do with the chunk

    Chunks are int16 ndarray of shape (block,) at the configured sample rate.
    The InputStream is held open for the whole `async with`. Producer (the
    sounddevice callback thread) pushes into an asyncio.Queue; consumer pulls
    via `chunks()`. If the consumer falls behind, oldest queued chunks are
    dropped silently — better than blocking the audio thread.
    """

    def __init__(
        self,
        sample_rate: int = config.INPUT_RATE,
        channels: int = config.INPUT_CHANNELS,
        block_ms: int = config.INPUT_BLOCK_MS,
        queue_size: int = 200,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_ms = block_ms
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._open = False

    async def __aenter__(self) -> "MicStream":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def open(self) -> None:
        if self._open:
            return
        block = int(self.sample_rate * self.block_ms / 1000)
        self._loop = asyncio.get_event_loop()
        queue = self._queue

        def callback(indata, frames, time_info, status):
            try:
                arr = indata.flatten() if hasattr(indata, "flatten") else \
                    np.frombuffer(indata, dtype=np.int16)
                if len(arr) == 0:
                    return
                self._loop.call_soon_threadsafe(
                    lambda: queue.put_nowait(arr) if not queue.full() else None
                )
            except Exception:
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=self.channels,
            dtype="int16", blocksize=block, callback=callback,
        )
        self._stream.start()
        self._open = True

    async def close(self) -> None:
        if not self._open:
            return
        self._open = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        # signal any waiting iterator to exit
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        """Yield int16 chunks until close() is called."""
        while self._open:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item is None:
                return
            yield item
