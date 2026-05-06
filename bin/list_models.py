#!/usr/bin/env python3
"""bin/list_models.py — utility: list models the API key has access to,
flag those that support Live API (bidiGenerateContent)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from google import genai
from nagaki_lab import config


def main() -> None:
    api_key = config.read_api_key()
    for api_version in ("v1beta", "v1alpha"):
        print(f"\n=========== {api_version} ===========")
        try:
            client = genai.Client(api_key=api_key,
                                  http_options={"api_version": api_version})
            for m in client.models.list():
                actions = getattr(m, "supported_actions", None) or []
                if "bidiGenerateContent" in actions:
                    print(f"  LIVE  {m.name}   actions={actions}")
                else:
                    print(f"        {m.name}")
        except Exception as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
