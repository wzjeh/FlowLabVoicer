"""Wrapper around the Gemini Live API session, with an event-bus interface.

`LiveSession` exposes the SDK's incoming protocol as a stream of typed events
that consumers subscribe to via `on(event_name, handler)`. This decouples the
conversation orchestrator from the SDK and lets multiple consumers (e.g. a
future display, a logger, a metrics sink) attach to the same session without
any one of them needing to know about the others.

Event names (also exposed as `LiveSession.EVENT_*` constants for typo-safe use):

  user_transcript        UserTranscript      partial / cumulative transcript of user audio
  assistant_transcript   AssistantTranscript partial / cumulative transcript of model speech
  audio_chunk            AudioOut            one chunk of int16 PCM (24 kHz mono) to play
  tool_call              ToolCallEvent       model wants to call one or more tools
  turn_complete          TurnComplete        end of one model turn (the iterator stops)
  go_away                GoAway              server is about to close the connection
  resumption_update      ResumptionUpdate    new session-resumption handle was issued
  recv_error             RecvError           an unhandled exception during receive
  any                    (name, event)       fires for EVERY event above; use for cross-cutting
                                             concerns such as "track last server activity"

Usage:

    live = LiveSession(system_prompt=..., tools=...)
    live.on(LiveSession.EVENT_AUDIO_CHUNK, lambda ev: speaker.write(ev.pcm))
    live.on(LiveSession.EVENT_TURN_COMPLETE, lambda ev: print("done"))
    async with live as session:
        # send some audio …
        await live.run_one_turn()   # drives receive + dispatches events
                                     # returns after TurnComplete or recv_error

Handlers may be sync `def f(event)` or async `async def f(event)`. The
`'any'` handler is called as `handler(event_name, event)`.
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional, Union

from google import genai
from google.genai import types

from . import config


# ---------- typed events ----------
@dataclass
class UserTranscript:
    text: str

@dataclass
class AssistantTranscript:
    text: str

@dataclass
class AudioOut:
    pcm: bytes

@dataclass
class ToolCallEvent:
    function_calls: list  # list of types.FunctionCall

@dataclass
class TurnComplete:
    pass

@dataclass
class GoAway:
    time_left: Optional[str]

@dataclass
class ResumptionUpdate:
    handle: Optional[str]

@dataclass
class RecvError:
    exception: BaseException


Event = Union[UserTranscript, AssistantTranscript, AudioOut, ToolCallEvent,
              TurnComplete, GoAway, ResumptionUpdate, RecvError]

Handler = Callable[..., Union[None, Awaitable[None]]]


# ---------- session wrapper ----------
class LiveSession:
    # Event-name constants for typo-safe subscription
    EVENT_USER_TRANSCRIPT = "user_transcript"
    EVENT_ASSISTANT_TRANSCRIPT = "assistant_transcript"
    EVENT_AUDIO_CHUNK = "audio_chunk"
    EVENT_TOOL_CALL = "tool_call"
    EVENT_TURN_COMPLETE = "turn_complete"
    EVENT_GO_AWAY = "go_away"
    EVENT_RESUMPTION_UPDATE = "resumption_update"
    EVENT_RECV_ERROR = "recv_error"
    EVENT_ANY = "any"

    KNOWN_EVENTS = (
        EVENT_USER_TRANSCRIPT, EVENT_ASSISTANT_TRANSCRIPT, EVENT_AUDIO_CHUNK,
        EVENT_TOOL_CALL, EVENT_TURN_COMPLETE, EVENT_GO_AWAY,
        EVENT_RESUMPTION_UPDATE, EVENT_RECV_ERROR, EVENT_ANY,
    )

    def __init__(
        self,
        system_prompt: str,
        tools: list[types.Tool],
        voice: str = config.DEFAULT_VOICE,
        model: str = config.MODEL,
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.voice = voice
        self.model = model
        self.handle: Optional[str] = None

        self._client = genai.Client(
            api_key=config.read_api_key(),
            http_options={"api_version": config.API_VERSION},
        )
        self._session = None
        self._cm = None
        self._handlers: dict[str, list[Handler]] = {n: [] for n in self.KNOWN_EVENTS}

    # ---------- subscription API ----------
    def on(self, event_name: str, handler: Handler) -> None:
        if event_name not in self._handlers:
            raise ValueError(
                f"unknown event '{event_name}'. Known: {self.KNOWN_EVENTS}"
            )
        self._handlers[event_name].append(handler)

    def off(self, event_name: str, handler: Handler) -> None:
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass

    async def _call_handler(self, handler: Handler, *args) -> None:
        """Call a handler, awaiting it if it's a coroutine."""
        try:
            res = handler(*args)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            print(f"  [handler error: {type(e).__name__}: {e}]")

    async def _emit(self, event_name: str, event: Event) -> None:
        # 'any' handlers see every event, with the name as first arg
        for h in self._handlers.get(self.EVENT_ANY, ()):
            await self._call_handler(h, event_name, event)
        # specific handlers
        for h in self._handlers.get(event_name, ()):
            await self._call_handler(h, event)

    # ---------- connection lifecycle ----------
    def _build_config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                ),
            ),
            system_instruction=types.Content(parts=[types.Part(text=self.system_prompt)]),
            tools=self.tools,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(handle=self.handle),
            # Disable server-side VAD. We use manual activity_start /
            # activity_end markers around each buffered upload (see
            # ConversationLoop._end_capture). Reason: with server VAD on,
            # the server occasionally fails to detect end-of-speech for
            # short / quiet utterances, leaving the turn open and the model
            # silent — observed as repeated 25s server-idle reconnects.
            # The legacy day4_voice.py used this same configuration and was
            # the most reliable mode after iterating through several VAD
            # variants; it was lost during the nagaki_lab/ refactor.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True,
                ),
            ),
        )

    async def __aenter__(self) -> "LiveSession":
        self._cm = self._client.aio.live.connect(model=self.model, config=self._build_config())
        self._session = await self._cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._cm is not None:
            try:
                return await self._cm.__aexit__(exc_type, exc, tb)
            finally:
                self._cm = None
                self._session = None

    # ---------- send ----------
    async def send_activity_start(self) -> None:
        """Mark the beginning of a user utterance. Required when server VAD
        is disabled (see _build_config) — without this the server will not
        treat subsequent audio as in-turn input."""
        await self._session.send_realtime_input(activity_start=types.ActivityStart())

    async def send_activity_end(self) -> None:
        """Mark the end of a user utterance. Triggers the server to generate
        a response. Without this the server keeps the turn open indefinitely
        (no transcript, no audio reply, eventual 25s watchdog reconnect)."""
        await self._session.send_realtime_input(activity_end=types.ActivityEnd())

    async def send_audio_chunk(self, pcm: bytes) -> None:
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={config.INPUT_RATE}"),
        )

    async def send_text(self, text: str) -> None:
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )

    async def send_tool_responses(self, function_responses: list) -> None:
        await self._session.send_tool_response(function_responses=function_responses)

    # ---------- receive ----------
    def _parse_message(self, msg) -> Iterable[tuple[str, Event]]:
        """Convert one raw SDK message into (event_name, event_obj) pairs."""
        upd = getattr(msg, "session_resumption_update", None)
        if upd is not None:
            new_handle = getattr(upd, "new_handle", None)
            if new_handle:
                self.handle = new_handle
            yield self.EVENT_RESUMPTION_UPDATE, ResumptionUpdate(handle=new_handle)
            return

        go = getattr(msg, "go_away", None)
        if go is not None:
            yield self.EVENT_GO_AWAY, GoAway(time_left=getattr(go, "time_left", None))
            return

        if getattr(msg, "tool_call", None):
            yield self.EVENT_TOOL_CALL, ToolCallEvent(
                function_calls=list(msg.tool_call.function_calls)
            )
            return

        sc = getattr(msg, "server_content", None)
        if sc is None:
            return

        it = getattr(sc, "input_transcription", None)
        if it is not None and getattr(it, "text", None):
            yield self.EVENT_USER_TRANSCRIPT, UserTranscript(text=it.text)

        ot = getattr(sc, "output_transcription", None)
        if ot is not None and getattr(ot, "text", None):
            yield self.EVENT_ASSISTANT_TRANSCRIPT, AssistantTranscript(text=ot.text)

        mt = getattr(sc, "model_turn", None)
        if mt is not None:
            for part in (mt.parts or []):
                data = getattr(getattr(part, "inline_data", None), "data", None)
                if data:
                    yield self.EVENT_AUDIO_CHUNK, AudioOut(pcm=data)

        if getattr(sc, "turn_complete", False):
            yield self.EVENT_TURN_COMPLETE, TurnComplete()

    async def run_one_turn(self) -> None:
        """Drive the receive loop for ONE turn. Dispatches events to registered
        handlers. Returns when TurnComplete fires.

        Why one turn at a time: the SDK's `session.receive()` generator
        deliberately ends after each turn_complete (see google.genai.live
        AsyncSession.receive — it `break`s the generator at turn_complete).
        Callers loop this method per turn.

        Receive errors do NOT raise out of this method — they are emitted as
        `EVENT_RECV_ERROR` events. Subscribers (typically the orchestrator)
        decide how to react (reconnect, abort, etc).
        `asyncio.CancelledError` IS re-raised so cancellation propagates.
        """
        try:
            async for msg in self._session.receive():
                for name, event in self._parse_message(msg):
                    await self._emit(name, event)
                    if name == self.EVENT_TURN_COMPLETE:
                        return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._emit(self.EVENT_RECV_ERROR, RecvError(exception=e))
