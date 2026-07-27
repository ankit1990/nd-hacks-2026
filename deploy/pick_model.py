#!/usr/bin/env python3
"""Choose a chat-capable model id from an OpenAI-compatible /v1/models response.

    curl -s https://inference.local/v1/models | deploy/pick_model.py [preferred-id]

Do not just take data[0]: on an OpenAI-backed gateway that is often an embedding model
(text-embedding-ada-002), which cannot serve chat completions and makes every extraction
fail. Prefer the model the sandbox is configured with, then the first id that is not
obviously a non-chat model.

Prints nothing and exits 1 when no usable id is found, so callers can branch on it.
"""

import json
import re
import sys

NOT_CHAT = re.compile(
    r"embed|whisper|tts|dall-e|moderation|audio|image|realtime|transcribe|search|similarity|edit",
    re.I,
)


def main() -> int:
    preferred = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    try:
        payload = json.load(sys.stdin)
        ids = [m["id"] for m in payload["data"]]
    except Exception:
        return 1

    if not ids:
        return 1
    if preferred and preferred in ids:
        print(preferred)
        return 0

    chat = [i for i in ids if not NOT_CHAT.search(i)]
    if not chat:
        return 1
    print(chat[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
