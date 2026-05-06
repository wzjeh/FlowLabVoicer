#!/usr/bin/env python3
"""bin/say.py — one-shot text → speech via Live API.

Usage:
    python bin/say.py "你好，介绍一下你自己"
    python bin/say.py "Tell me a joke" --voice Puck
"""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from google import genai
from google.genai import types

from nagaki_lab import config


async def speak(text: str, voice: str) -> None:
    client = genai.Client(api_key=config.read_api_key(),
                          http_options={"api_version": config.API_VERSION})
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    aplay = subprocess.Popen(
        ["aplay", "-D", config.SPEAKER_DEVICE, "-q",
         "-r", str(config.OUTPUT_RATE), "-c", "1", "-f", "S16_LE", "-t", "raw"],
        stdin=subprocess.PIPE,
    )
    bytes_recv = 0
    print(f"-> {text!r}")
    async with client.aio.live.connect(model=config.MODEL, config=cfg) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )
        async for msg in session.receive():
            sc = getattr(msg, "server_content", None)
            if sc is None:
                continue
            mt = getattr(sc, "model_turn", None)
            if mt:
                for part in (mt.parts or []):
                    data = getattr(getattr(part, "inline_data", None), "data", None)
                    if data:
                        try:
                            aplay.stdin.write(data)
                            aplay.stdin.flush()
                        except (BrokenPipeError, ValueError):
                            pass
                        bytes_recv += len(data)
            if getattr(sc, "turn_complete", False):
                break
    print(f"<- {bytes_recv} bytes ({bytes_recv/config.OUTPUT_RATE/2:.1f}s of audio)")
    try:
        aplay.stdin.close()
    except Exception:
        pass
    aplay.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--voice", default=config.DEFAULT_VOICE,
                    choices=config.AVAILABLE_VOICES)
    args = ap.parse_args()
    asyncio.run(speak(" ".join(args.text), args.voice))


if __name__ == "__main__":
    main()
