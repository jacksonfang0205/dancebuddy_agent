"""Talk to the agent in the terminal, or give it one request.

  python -m dancebuddy chat          (or python -m dancebuddy.chat)
  python -m dancebuddy go "the dance is data/videos/dance.mp4, the plush is assets/plush_soft: dance along"

Each conversation is saved as runs/sessions/<chat|go>-<timestamp>.json: every message, tool call and result.
"""

from __future__ import annotations

import sys
import time

from dancebuddy.agent.session import AgentSession
from dancebuddy.sandbox import ROOT
from dancebuddy.user import ConsoleUser


def _trace(kind: str):
    return ROOT / "runs" / "sessions" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.json"


def go(request: str, model: str | None = None) -> str:
    s = AgentSession(model=model)
    reply = s.send(request)
    s.save(_trace("go"))
    print(f"\n{reply}\n\n({s.calls} LLM calls, ${s.cost:.3f})")
    return reply


def chat(model: str | None = None, first: str | None = None) -> None:
    s = AgentSession(model=model, user=ConsoleUser())
    trace = _trace("chat")
    print("DanceBuddy chat. Tell me where the dance video is and which plush to use. Empty line or 'exit' to stop.")
    text = first
    while True:
        if text is None:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        reply = s.send(text)
        s.save(trace)
        print(f"\nDanceBuddy: {reply}\n({s.calls} LLM calls so far, ${s.cost:.3f})")
        text = None
    s.save(trace)
    print(f"conversation saved to {trace}")


if __name__ == "__main__":
    chat(first=" ".join(sys.argv[1:]) or None)
