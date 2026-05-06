"""Minimal multi-turn diagnostic for the Gemini Live API.

This script is deliberately stripped of every piece of orchestration we built
on top of the SDK: no mic, no UI, no LEDs, no state machine, no tools.
It records ONE wav once (then re-uses it from cache), opens a Live session,
and sends that same audio THREE times in a row, watching whether each turn
gets a response.

Outcome interpretation:
  - All 3 turns get a response → SDK multi-turn works. The bug is in our
                                 orchestration code (ConversationLoop etc).
  - Turn 1 OK, turn 2 hangs    → SDK / model / send-pattern problem.
                                 Try a different model or a different way
                                 of signalling end-of-turn.

Usage:
    python tests/test_live_minimal.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from google import genai
from google.genai import types

from nagaki_lab import config

TEST_WAV = config.DATA_DIR / "test_input.wav"
DURATION_S = 4.0


def load_or_record() -> bytes:
    if TEST_WAV.exists():
        with wave.open(str(TEST_WAV), "rb") as w:
            n = w.getnframes()
            assert w.getnchannels() == 1
            assert w.getframerate() == config.INPUT_RATE
            return w.readframes(n)

    import sounddevice as sd
    print(f"\nNo cached input found at {TEST_WAV}.")
    print(f"About to record {DURATION_S:.0f}s — say something useful (e.g. "
          f"'你好你是谁' or 'hello, who are you?').")
    for i in range(3, 0, -1):
        print(f"  recording in {i}…")
        time.sleep(1)
    print("  RECORDING")
    audio = sd.rec(int(DURATION_S * config.INPUT_RATE),
                   samplerate=config.INPUT_RATE, channels=1, dtype="int16")
    sd.wait()
    pcm = audio.flatten().tobytes()
    with wave.open(str(TEST_WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.INPUT_RATE)
        w.writeframes(pcm)
    print(f"  saved to {TEST_WAV}\n")
    return pcm


async def send_one_turn(session, audio_pcm: bytes, label: str,
                        timeout_s: float = 30.0) -> dict:
    print(f"\n========== {label} ==========")
    silence = bytes(int(config.INPUT_RATE * 2 * config.TRAILING_SILENCE_S))
    upload = audio_pcm + silence

    t0 = time.monotonic()
    chunk_size = config.UPLOAD_CHUNK_BYTES
    n = 0
    for i in range(0, len(upload), chunk_size):
        await session.send_realtime_input(
            audio=types.Blob(data=upload[i:i + chunk_size],
                             mime_type=f"audio/pcm;rate={config.INPUT_RATE}"),
        )
        n += 1
    print(f"[t+{time.monotonic()-t0:5.2f}s] uploaded {n} chunks ({len(upload)} bytes)")

    user_text = []
    asst_text = []
    audio_chunks = 0
    audio_bytes = 0
    first_msg_at = None
    first_audio_at = None
    turn_complete = False
    last_activity = time.monotonic()

    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            try:
                msg = await asyncio.wait_for(session.receive().__anext__(),
                                             timeout=2.0)
            except (asyncio.TimeoutError, StopAsyncIteration):
                # check absolute idle
                if time.monotonic() - last_activity > timeout_s:
                    break
                continue
            last_activity = time.monotonic()
            if first_msg_at is None:
                first_msg_at = time.monotonic() - t0
                print(f"[t+{first_msg_at:5.2f}s] first server message")

            sc = getattr(msg, "server_content", None)
            if sc is None:
                continue
            it = getattr(sc, "input_transcription", None)
            if it and getattr(it, "text", None):
                user_text.append(it.text)
            ot = getattr(sc, "output_transcription", None)
            if ot and getattr(ot, "text", None):
                asst_text.append(ot.text)
            mt = getattr(sc, "model_turn", None)
            if mt:
                for part in (mt.parts or []):
                    data = getattr(getattr(part, "inline_data", None), "data", None)
                    if data:
                        audio_chunks += 1
                        audio_bytes += len(data)
                        if first_audio_at is None:
                            first_audio_at = time.monotonic() - t0
                            print(f"[t+{first_audio_at:5.2f}s] first audio chunk")
            if getattr(sc, "turn_complete", False):
                turn_complete = True
                print(f"[t+{time.monotonic()-t0:5.2f}s] turn_complete")
                break
    except Exception as e:
        print(f"[t+{time.monotonic()-t0:5.2f}s] receive error: {type(e).__name__}: {e}")

    user_s = "".join(user_text).strip()
    asst_s = "".join(asst_text).strip()
    print(f"  user transcript: {user_s!r}")
    print(f"  assistant transcript: {asst_s!r}")
    print(f"  audio chunks: {audio_chunks}, audio bytes: {audio_bytes}")
    return {
        "label": label,
        "ok": turn_complete and (audio_chunks > 0 or asst_s),
        "turn_complete": turn_complete,
        "first_msg_at": first_msg_at,
        "first_audio_at": first_audio_at,
        "audio_bytes": audio_bytes,
        "user_text": user_s,
        "asst_text": asst_s,
    }


async def main() -> None:
    audio = load_or_record()
    print(f"using audio: {len(audio)} bytes = {len(audio)/2/config.INPUT_RATE:.2f}s\n")

    client = genai.Client(api_key=config.read_api_key(),
                          http_options={"api_version": config.API_VERSION})
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        system_instruction=types.Content(parts=[types.Part(
            text="Reply in ONE short sentence in the user's language."
        )]),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    results = []
    async with client.aio.live.connect(model=config.MODEL, config=cfg) as session:
        for i in (1, 2, 3):
            r = await send_one_turn(session, audio, f"TURN {i}")
            results.append(r)
            await asyncio.sleep(1.0)

    print("\n\n========== SUMMARY ==========")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        first_audio = (f"{r['first_audio_at']:.2f}s"
                       if r["first_audio_at"] is not None else "(no audio)")
        print(f"  {r['label']}: {status:4}  first_audio_at={first_audio}  audio_bytes={r['audio_bytes']}")

    n_ok = sum(1 for r in results if r["ok"])
    print()
    if n_ok == 3:
        print("✓ ALL 3 TURNS WORKED at the SDK level.")
        print("  → Multi-turn bug we've been chasing is in our orchestration "
              "code, not in the SDK / model / send pattern.")
    elif n_ok >= 1:
        print(f"✗ {n_ok}/3 turns worked. SDK multi-turn is unstable for this "
              "model and send pattern.")
        print("  → Try a different model (e.g. gemini-live-2.5-flash-preview), "
              "or a different way of marking turn boundaries (audio_stream_end, "
              "activity_start/end, send_client_content for end).")
    else:
        print("? Even turn 1 failed. Bigger issue — check API key, network, model name.")


if __name__ == "__main__":
    asyncio.run(main())
