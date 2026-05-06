"""Conversation orchestrator — the heart of the assistant.

Coordinates: mic capture → live session upload → response receive → playback.
Maintains the user-facing state machine. Drives LED status. Persists turns
to memory. Knows nothing about stdin or terminal UI; the UI calls action
methods on this loop.

State machine:

  WAKE_LISTENING (only if a wake detector is configured)
       │  wake detected, OR user presses ENTER
       ▼
  CAPTURING ──── ENTER ────▶ RESPONDING
       ▲                          │
       │                          │ turn_complete  OR  user_abort
       │                          ▼
       └─────────────────── IDLE  (or WAKE_LISTENING)

If no wake detector: starts in IDLE; the user drives every turn manually
with ENTER.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

import numpy as np

from . import config, live as live_mod, prompts
from .audio.input import MicCapture, MicStream
from .audio.output import SpeakerPlayback
from .leds import LEDStatus
from .memory import TurnLog
from .tools import DISPATCH as TOOL_DISPATCH
from .tools import get_tool_declarations
from .tools.music import (
    pause_current as _music_pause,
    resume_current as _music_resume,
)
from .tools.timer import play_beep_blocking
from .wake import WakeWordDetector
from google.genai import types


class ConversationLoop:

    STATE_IDLE = "idle"
    STATE_WAKE_LISTENING = "wake_listening"
    STATE_CAPTURING = "capturing"
    STATE_RESPONDING = "responding"

    def __init__(
        self,
        mic: MicCapture,
        speaker: SpeakerPlayback,
        live_session: live_mod.LiveSession,
        turn_log: TurnLog,
        leds: LEDStatus,
        wake_detector: Optional[WakeWordDetector] = None,
        verbose: bool = True,
        on_state_change: Optional[Callable[[str, str], None]] = None,
    ):
        self.mic = mic
        self.speaker = speaker
        self.live = live_session
        self.turn_log = turn_log
        self.leds = leds
        self.wake_detector = wake_detector
        self.verbose = verbose
        self._on_state_change = on_state_change

        self.session_id = uuid.uuid4().hex[:8]
        self._state = (self.STATE_WAKE_LISTENING if wake_detector
                       else self.STATE_IDLE)
        self._exit = False

        # turn-scoped state
        self._t0: Optional[float] = None
        self._abort_flag = False
        self._last_server_activity: Optional[float] = None
        self._capturing_user_chunks: list[str] = []
        self._capturing_asst_chunks: list[str] = []
        self._first_audio_marked = False
        # True while music is SIGSTOPped because the model is currently
        # speaking. Cleared (and SIGCONT issued) when the turn ends via any
        # path — turn_complete / recv_error / abort / watchdog reconnect.
        self._music_paused_for_turn = False

        # session resource: held during run()
        self._session_obj = None
        self._turn_task: Optional[asyncio.Task] = None
        self._needs_reconnect = False

        # Subscribe our orchestration logic to LiveSession events. The session
        # itself does not know about us; any other consumer (a future display,
        # a logger) can subscribe to the same events independently.
        L = live_mod.LiveSession
        self.live.on(L.EVENT_ANY,                 self._on_any_event)
        self.live.on(L.EVENT_USER_TRANSCRIPT,     self._on_user_transcript)
        self.live.on(L.EVENT_ASSISTANT_TRANSCRIPT, self._on_assistant_transcript)
        self.live.on(L.EVENT_AUDIO_CHUNK,         self._on_audio_chunk)
        self.live.on(L.EVENT_TOOL_CALL,           self._on_tool_call)
        self.live.on(L.EVENT_TURN_COMPLETE,       self._on_turn_complete)
        self.live.on(L.EVENT_GO_AWAY,             self._on_go_away)
        self.live.on(L.EVENT_RECV_ERROR,          self._on_recv_error)

        # wake-mode-only state. The wake dispatcher accumulates speech here
        # while state == CAPTURING. Untouched in non-wake mode.
        self._wake_capture_buf = bytearray()
        self._wake_capture_peak: float = 0.0
        self._wake_capture_started_at: Optional[float] = None
        self._wake_recent_rms: list[float] = []   # rolling window for silence detect
        self._wake_chunks_remaining: int = 0      # how many chunks to discard right after wake

    # tunables for wake-driven auto-end-of-capture
    WAKE_CAPTURE_MAX_S = 15.0       # hard cap on auto capture
    WAKE_SILENCE_WINDOW_S = 1.0     # how much trailing silence triggers end
    WAKE_SILENCE_RMS = 250.0        # RMS below this counts as silence
    WAKE_GRACE_S = 0.6              # min capture before silence detection arms
    WAKE_PRE_DISCARD_MS = 500       # drop this many ms after wake (the wake word itself)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_busy(self) -> bool:
        return self._state in (self.STATE_CAPTURING, self.STATE_RESPONDING)

    # ---------- timing helpers ----------
    def _t_mark(self, label: str) -> None:
        if self._t0 is None:
            self._t0 = time.monotonic()
            dt = 0.0
        else:
            dt = time.monotonic() - self._t0
        if self.verbose:
            print(f"  [t+{dt:5.2f}s] {label}")

    def _t_reset(self) -> None:
        self._t0 = None

    # ---------- state transitions ----------
    _BUSY_STATES = ("capturing", "responding")  # = (STATE_CAPTURING, STATE_RESPONDING)

    async def _set_state(self, new_state: str) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state

        # Duck music whenever we transition into a "busy" state (mic capturing
        # OR model responding) and un-duck when returning to rest. This is the
        # single chokepoint — every entry into CAPTURING/RESPONDING and every
        # exit back to IDLE/WAKE_LISTENING flows through here.
        # Why both states: if music plays during capture it bleeds into the
        # mic and ruins server VAD's end-of-turn detection (no real trailing
        # silence) — observed as 25s server-idle hangs.
        was_busy = old in self._BUSY_STATES
        now_busy = new_state in self._BUSY_STATES
        if not was_busy and now_busy:
            if not self._music_paused_for_turn and _music_pause():
                self._music_paused_for_turn = True
                if self.verbose:
                    print("  [music paused]")
        elif was_busy and not now_busy:
            if self._music_paused_for_turn:
                _music_resume()
                self._music_paused_for_turn = False
                if self.verbose:
                    print("  [music resumed]")

        if self._on_state_change is not None:
            try:
                self._on_state_change(old, new_state)
            except Exception:
                pass
        led_state = {
            self.STATE_IDLE: "idle",
            self.STATE_WAKE_LISTENING: "idle",
            self.STATE_CAPTURING: "listening",
            self.STATE_RESPONDING: "thinking",
        }.get(new_state, "idle")
        await self.leds.set_state(led_state)

    # ---------- public actions (called by UI) ----------
    async def on_user_action(self) -> None:
        """The user pressed ENTER (or a similar trigger). Dispatch by state.

        Wraps the dispatch in a top-level try so an unexpected error in any
        capture / send / abort path can never crash the UI loop. Any failure
        flags the session for reconnect and returns the user to idle.
        """
        try:
            if self._state in (self.STATE_IDLE, self.STATE_WAKE_LISTENING):
                await self._start_capture()
            elif self._state == self.STATE_CAPTURING:
                await self._end_capture()
            elif self._state == self.STATE_RESPONDING:
                await self._abort_response()
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as e:
            print(f"\n[conversation error in user action: "
                  f"{type(e).__name__}: {e}]")
            self.speaker.abort()
            self._needs_reconnect = True
            try:
                await self._return_to_resting_state()
            except Exception:
                pass

    async def request_exit(self) -> None:
        self._exit = True
        if self._state == self.STATE_CAPTURING:
            await self._end_capture()

    # ---------- run ----------
    async def run(self) -> None:
        """Open the live session and keep it alive until exit. Per-turn receive
        tasks are spawned by _end_capture and run to completion; the run() loop
        itself just keeps the websocket open and the watchdog ticking.

        We DO NOT have a long-running receive task. The SDK's session.receive()
        is per-turn (it breaks at turn_complete), so the canonical pattern is
        send-then-receive per turn — exactly what test_live_minimal.py does.
        """
        await self.turn_log.init()
        if self.leds.available:
            print("[LEDs: enabled]")
        else:
            print(f"[LEDs: disabled — {self.leds.error}]")
        await self.leds.set_state("idle")

        backoff = config.RECONNECT_BACKOFF_INITIAL_S
        try:
            while not self._exit:
                try:
                    async with self.live as session:
                        self._session_obj = session
                        if self.live.handle is not None:
                            print(f"[reconnected with handle {self.live.handle[:12]}…]")
                        backoff = config.RECONNECT_BACKOFF_INITIAL_S

                        self._needs_reconnect = False
                        watch = asyncio.create_task(self._watchdog_loop())
                        wake_task = None
                        if self.wake_detector is not None:
                            wake_task = asyncio.create_task(self._wake_dispatcher_loop())

                        try:
                            while not self._exit and not self._needs_reconnect:
                                await asyncio.sleep(0.2)
                            if self._needs_reconnect:
                                # Trigger the outer reconnect path so we get a
                                # clean session (preserves session_resumption
                                # handle, so context survives).
                                raise ConnectionAbortedError(
                                    "watchdog forced reconnect after server idle timeout"
                                )
                        finally:
                            for t in (watch, wake_task):
                                if t is None:
                                    continue
                                t.cancel()
                                try:
                                    await t
                                except (asyncio.CancelledError, Exception):
                                    pass
                            if self._turn_task is not None and not self._turn_task.done():
                                self._turn_task.cancel()
                                try:
                                    await self._turn_task
                                except (asyncio.CancelledError, Exception):
                                    pass
                        return
                except (KeyboardInterrupt, asyncio.CancelledError):
                    raise
                except Exception as e:
                    print(f"\n[disconnected: {type(e).__name__}: {e}]")
                    if self._state in (self.STATE_CAPTURING, self.STATE_RESPONDING):
                        await self._set_state(self.STATE_IDLE)
                    print(f"[reconnecting in {backoff:.1f}s — Ctrl+C to give up]")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, config.RECONNECT_BACKOFF_MAX_S)
        finally:
            self.speaker.abort()
            await self.leds.stop()

    # ---------- capture ----------
    async def _start_capture(self) -> None:
        """Begin a capture turn. In non-wake mode this opens MicCapture; in
        wake mode the dispatcher loop is already producing chunks and we just
        flip a flag so it starts buffering them."""
        self._t0 = None
        self._t_mark("capture_start")
        await self._set_state(self.STATE_CAPTURING)
        if self.wake_detector is None:
            await self.mic.start()
        else:
            # wake mode: dispatcher will start buffering when it sees CAPTURING
            self._wake_capture_buf = bytearray()
            self._wake_capture_peak = 0.0
            self._wake_capture_started_at = time.monotonic()
            self._wake_recent_rms = []
            self._wake_chunks_remaining = 0
        if self.verbose:
            print("  [recording — speak now; press ENTER to send]")

    async def _end_capture(self, *, auto: bool = False) -> None:
        if self._state != self.STATE_CAPTURING:
            return
        if self.wake_detector is None:
            pcm = await self.mic.stop()
            duration = self.mic.duration_s
            peak = self.mic.peak_rms
        else:
            pcm = bytes(self._wake_capture_buf)
            duration = len(pcm) / 2 / config.INPUT_RATE
            peak = self._wake_capture_peak
            self._wake_capture_buf = bytearray()
            self._wake_capture_peak = 0.0
            self._wake_capture_started_at = None
            self._wake_recent_rms = []
        reason = "auto-silence" if auto else "enter_pressed"
        self._t_mark(f"{reason} (captured {duration:.2f}s, peak RMS {peak:.0f})")

        # quality gate
        if duration < config.MIN_CAPTURE_SECONDS:
            print(f"  [too short ({duration:.2f}s) — press ENTER and speak again]")
            await self._return_to_resting_state()
            return
        if peak < config.MIN_PEAK_RMS:
            print(f"  [no speech detected (peak RMS {peak:.0f}) — speak louder/closer]")
            await self._return_to_resting_state()
            return

        # ship it
        await self._set_state(self.STATE_RESPONDING)
        self._abort_flag = False
        self._last_server_activity = time.monotonic()
        self._first_audio_marked = False
        self._capturing_user_chunks = []
        self._capturing_asst_chunks = []

        # We bracket the upload with explicit activity_start / activity_end
        # markers because server VAD is disabled in _build_config. Without
        # those markers the server has no signal that the user's turn ended
        # and just sits silent until our 25s watchdog fires.
        # The trailing silence is no longer strictly required (activity_end
        # tells the server we're done) but we keep it small as a safety
        # cushion against tail-clipping by the encoder.
        silence = bytes(int(config.INPUT_RATE * 2 * config.TRAILING_SILENCE_S))
        upload = pcm + silence

        try:
            await self.live.send_activity_start()
            n = 0
            for i in range(0, len(upload), config.UPLOAD_CHUNK_BYTES):
                piece = upload[i:i + config.UPLOAD_CHUNK_BYTES]
                await self.live.send_audio_chunk(piece)
                n += 1
            await self.live.send_activity_end()
            self._t_mark(
                f"audio_uploaded ({n} chunks, {len(upload)} bytes "
                f"= {duration:.2f}s speech + {config.TRAILING_SILENCE_S:.1f}s silence) "
                f"[activity_start..end]"
            )
        except Exception as e:
            # Common case: 1011 keepalive ping timeout, or any other
            # websocket-level failure during the upload burst. Don't crash —
            # flag the session for reconnect and return to a clean idle state.
            self._t_mark(f"upload FAILED: {type(e).__name__}: {e}")
            print(f"  [upload failed; will reconnect session]")
            self._needs_reconnect = True
            await self._return_to_resting_state()
            return
        if self.verbose:
            print("  [sent — waiting for reply (ENTER to interrupt)]")

        # Spawn the per-turn receive task NOW (post-send). It will run to
        # completion at TurnComplete, transitioning state back to IDLE.
        self._turn_task = asyncio.create_task(self._receive_one_turn())

    async def _abort_response(self) -> None:
        self._t_mark("user_abort")
        self._abort_flag = True
        self.speaker.abort()
        # The receive task will exit naturally (it sees abort_flag and drains
        # until turn_complete, OR we cancel it if it's stuck).
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
            self._turn_task = None
        await self._return_to_resting_state()
        if self.verbose:
            print("  [interrupted — back to idle]")

    async def _return_to_resting_state(self) -> None:
        self._last_server_activity = None
        # Music un-duck happens automatically inside _set_state when we
        # transition out of a busy state.
        target = (self.STATE_WAKE_LISTENING if self.wake_detector
                  else self.STATE_IDLE)
        await self._set_state(target)

    # ---------- receive (per-turn) ----------
    async def _receive_one_turn(self) -> None:
        """Drive the per-turn receive loop. The actual event handling happens
        in the handlers we registered in __init__ — this method just kicks off
        LiveSession.run_one_turn() and cleans up afterwards.

        Matches the proven test_live_minimal.py pattern: send → receive →
        TurnComplete → cleanup. The next turn spawns a fresh task because the
        SDK's session.receive() is itself per-turn.
        """
        try:
            await self.live.run_one_turn()
        except asyncio.CancelledError:
            self.speaker.abort()
            raise
        finally:
            self._abort_flag = False
            self._last_server_activity = None
            self._first_audio_marked = False
            self._capturing_user_chunks = []
            self._capturing_asst_chunks = []
            if not self._exit:
                await self._return_to_resting_state()

    # ---------- per-event handlers (registered with self.live in __init__) ----------
    async def _on_any_event(self, name, event) -> None:
        # Cross-cutting: any server message keeps the watchdog quiet.
        self._last_server_activity = time.monotonic()

    async def _on_user_transcript(self, ev) -> None:
        if self._abort_flag:
            return
        if not self._capturing_user_chunks:
            self._t_mark("first_user_transcript")
        self._capturing_user_chunks.append(ev.text)

    async def _on_assistant_transcript(self, ev) -> None:
        if self._abort_flag:
            return
        self._capturing_asst_chunks.append(ev.text)

    async def _on_audio_chunk(self, ev) -> None:
        if self._abort_flag:
            return
        if not self._first_audio_marked:
            self._t_mark("first_audio_chunk")
            self._first_audio_marked = True
            await self.leds.set_state("speaking")
            # (Music ducking happens in _set_state on the IDLE→CAPTURING
            # transition, before any audio is even captured.)
        self.speaker.write(ev.pcm)

    async def _on_tool_call(self, ev) -> None:
        await self._handle_tool_calls(ev.function_calls)

    async def _on_go_away(self, ev) -> None:
        print(f"\n  [server goAway: {ev.time_left}]")

    async def _on_turn_complete(self, ev) -> None:
        self._t_mark("turn_complete")
        self.speaker.close()
        user_text = "".join(self._capturing_user_chunks).strip()
        asst_text = "".join(self._capturing_asst_chunks).strip()
        if not self._abort_flag:
            if user_text:
                print(f"\n<<< [you: {user_text}]")
                await self.turn_log.append(self.session_id, "user", user_text)
            if asst_text:
                print(f"<<< {asst_text}")
                await self.turn_log.append(self.session_id, "assistant", asst_text)
            elif user_text:
                print("<<< [model returned no audio — please ask again]")
        self._t_reset()

    async def _on_recv_error(self, ev) -> None:
        # Most common: 1008 policy violation, 1011 keepalive timeout. The
        # main run() loop polls _needs_reconnect every 200 ms and triggers
        # a clean reconnect via session_resumption.
        print(f"\n[recv error: {type(ev.exception).__name__}: {ev.exception}]")
        print("[server dropped — auto-reconnecting in <1s, next turn will resume]")
        self.speaker.abort()
        self._needs_reconnect = True

    async def _handle_tool_calls(self, function_calls: list) -> None:
        """Dispatch each tool call with a hard timeout. If a tool hangs (slow
        upstream API, rate-limit retry storm), we still send back an error
        response so the model can recover instead of leaving the server
        waiting forever (which would trip our watchdog and force a reconnect)."""
        responses = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            args_str = repr(args) if len(repr(args)) < 200 else repr(args)[:200] + "…"
            print(f"  [tool: {fc.name}({args_str})]")
            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    TOOL_DISPATCH[fc.name](args),
                    timeout=config.TOOL_CALL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                result = {
                    "error": f"tool '{fc.name}' timed out after "
                             f"{config.TOOL_CALL_TIMEOUT_S:.0f}s — "
                             "upstream service may be slow or rate-limited"
                }
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            dt_ms = (time.monotonic() - t0) * 1000
            preview = repr(result)
            if len(preview) > 200:
                preview = preview[:200] + "…"
            print(f"  [→ {preview}]  ({dt_ms:.0f}ms)")
            responses.append(types.FunctionResponse(
                id=fc.id, name=fc.name, response={"result": result},
            ))
        await self.live.send_tool_responses(responses)

    # ---------- watchdog ----------
    # ---------- wake-mode mic dispatcher ----------
    async def _wake_dispatcher_loop(self) -> None:
        """Wake-mode mic loop. Holds a single MicStream open for the whole
        session and dispatches each chunk based on the current state:

          WAKE_LISTENING  → feed wake_detector; on detection, transition to
                            CAPTURING (chime, init buffer, transient discard
                            window for the wake utterance itself)
          CAPTURING       → append to capture buffer, track peak + silence,
                            auto-end when 1 s of trailing silence accumulates
                            after a 0.6 s grace period (or 15 s hard cap)
          IDLE/RESPONDING → drop chunk
        """
        last_status_t = time.monotonic()
        rms_window: list[float] = []   # for status print only
        try:
            async with MicStream() as stream:
                async for chunk in stream.chunks():
                    state = self._state
                    if state == self.STATE_WAKE_LISTENING:
                        try:
                            detected = self.wake_detector.feed(chunk)
                        except Exception as e:
                            print(f"\n[wake detector error: {e}]")
                            continue

                        # diagnostic: log non-trivial scores so the user can
                        # see how close the model is to firing
                        score = self.wake_detector.last_score
                        if score > 0.05:
                            print(f"  [wake score: {score:.2f} (threshold {self.wake_detector.threshold:.2f})]")

                        # rolling RMS for the periodic status line
                        if len(rms_window) < 100:
                            sq = chunk.astype(np.float32) ** 2
                            rms_window.append(float(np.sqrt(sq.mean())))
                        else:
                            sq = chunk.astype(np.float32) ** 2
                            rms_window.pop(0)
                            rms_window.append(float(np.sqrt(sq.mean())))

                        if detected:
                            rms_window.clear()
                            await self._on_wake_detected()
                            last_status_t = time.monotonic()
                            continue

                        now = time.monotonic()
                        if now - last_status_t >= 5.0:
                            avg = sum(rms_window) / len(rms_window) if rms_window else 0.0
                            mx = max(rms_window) if rms_window else 0.0
                            print(f"  [wake-listening: avg RMS={avg:.0f}, "
                                  f"peak RMS={mx:.0f}, last score={score:.2f}]")
                            last_status_t = now

                    elif state == self.STATE_CAPTURING:
                        await self._wake_consume_capture_chunk(chunk)
                    # else: drop
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"\n[wake dispatcher error: {type(e).__name__}: {e}]")

    async def _on_wake_detected(self) -> None:
        print(f"\n[wake! '{self.wake_detector.wakeword}' "
              f"score={self.wake_detector.last_score:.2f}]")
        # short non-blocking chime so user knows we're listening
        asyncio.create_task(asyncio.to_thread(self._chime))
        # Transition into capture, with a brief discard window so the wake
        # utterance itself does not bleed into the audio sent to the model.
        self._t0 = None
        self._t_mark("capture_start (wake-driven)")
        await self._set_state(self.STATE_CAPTURING)
        self._wake_capture_buf = bytearray()
        self._wake_capture_peak = 0.0
        self._wake_capture_started_at = time.monotonic()
        self._wake_recent_rms = []
        self._wake_chunks_remaining = max(
            1, self.WAKE_PRE_DISCARD_MS // config.INPUT_BLOCK_MS
        )
        if self.verbose:
            print("  [recording — speak now; will auto-end on silence (or ENTER)]")

    @staticmethod
    def _chime() -> None:
        try:
            play_beep_blocking()
        except Exception:
            pass

    async def _wake_consume_capture_chunk(self, chunk: np.ndarray) -> None:
        # discard the first few chunks (wake-word echo)
        if self._wake_chunks_remaining > 0:
            self._wake_chunks_remaining -= 1
            return

        sq = chunk.astype(np.float32) ** 2
        rms = float(np.sqrt(sq.mean()))
        if rms > self._wake_capture_peak:
            self._wake_capture_peak = rms
        self._wake_capture_buf.extend(chunk.tobytes())
        self._wake_recent_rms.append(rms)
        # window size = WAKE_SILENCE_WINDOW_S worth of chunks
        max_window = max(1, int(self.WAKE_SILENCE_WINDOW_S * 1000
                                / config.INPUT_BLOCK_MS))
        if len(self._wake_recent_rms) > max_window:
            self._wake_recent_rms.pop(0)

        elapsed = time.monotonic() - (self._wake_capture_started_at or time.monotonic())

        # hard cap
        if elapsed >= self.WAKE_CAPTURE_MAX_S:
            await self._end_capture(auto=True)
            return

        # silence-driven end
        if (elapsed >= self.WAKE_GRACE_S
                and len(self._wake_recent_rms) >= max_window
                and all(r < self.WAKE_SILENCE_RMS for r in self._wake_recent_rms)):
            await self._end_capture(auto=True)
            return

    # ---------- watchdog ----------
    async def _watchdog_loop(self) -> None:
        try:
            while not self._exit:
                await asyncio.sleep(0.5)

                # ENTER-mode CAPTURING: did the user blow past
                # MAX_CAPTURE_SECONDS without pressing ENTER again? The mic
                # task self-stops at the cap but silently — without this
                # check the user would think they were still recording while
                # nothing further was being captured. Auto-send what we have
                # so the partial speech gets a response instead of nothing.
                if (self._state == self.STATE_CAPTURING
                        and self.wake_detector is None
                        and self.mic.hit_cap):
                    print(f"\n  [recording cap of "
                          f"{config.MAX_CAPTURE_SECONDS:.0f}s hit — "
                          "auto-sending what was captured]")
                    try:
                        await self._end_capture(auto=True)
                    except Exception as e:
                        print(f"  [auto-send failed: "
                              f"{type(e).__name__}: {e}]")
                    continue

                # RESPONDING: server-idle reconnect (existing behaviour).
                if self._state != self.STATE_RESPONDING:
                    continue
                if self._last_server_activity is None:
                    continue
                idle_for = time.monotonic() - self._last_server_activity
                if idle_for > config.SERVER_IDLE_TIMEOUT_S:
                    print(f"\n[server silent for {idle_for:.0f}s — "
                          "forcing session reconnect for a clean state]")
                    self.speaker.abort()
                    if self._turn_task is not None and not self._turn_task.done():
                        self._turn_task.cancel()
                        try:
                            await self._turn_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        self._turn_task = None
                    await self._return_to_resting_state()
                    # Signal the main run() loop to drop this websocket and
                    # reconnect. The session_resumption handle is preserved.
                    self._needs_reconnect = True
                    return
        except asyncio.CancelledError:
            raise
