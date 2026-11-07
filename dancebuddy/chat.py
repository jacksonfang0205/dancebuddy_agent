"""Talk to the agent in the terminal, or give it one request.

  python -m dancebuddy chat          (or python -m dancebuddy.chat)
  python -m dancebuddy go "the dance is data/videos/dance.mp4, the plush is assets/plush_soft: dance along"

The terminal shows only the conversation. Tool calls, commands and Blender / MediaPipe output go to
runs/sessions/<chat|go>-<timestamp>.log (add --verbose to see them live); the whole conversation, with every tool
call and result, is saved next to it as .json.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

from dancebuddy.agent.session import AgentSession
from dancebuddy.sandbox import ROOT
from dancebuddy.user import ConsoleUser


def _paths(kind: str):
    base = ROOT / "runs" / "sessions" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".json"), base.with_suffix(".log")


class Quiet:
    """Send everything written to the terminal (prints, subprocesses such as Blender) to a log file."""

    def __init__(self, log_path, enabled: bool = True):
        self.enabled = enabled
        self.file = open(log_path, "a") if enabled else None
        self.saved = None

    def _on(self):
        if self.enabled and self.saved is None:
            sys.stdout.flush(), sys.stderr.flush()
            self.saved = (os.dup(1), os.dup(2))
            os.dup2(self.file.fileno(), 1)
            os.dup2(self.file.fileno(), 2)

    def _off(self):
        if self.saved is not None:
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(self.saved[0], 1)
            os.dup2(self.saved[1], 2)
            os.close(self.saved[0]), os.close(self.saved[1])
            self.saved = None

    @contextmanager
    def working(self):
        self._on()
        try:
            yield self
        finally:
            self._off()

    @contextmanager
    def paused(self):
        was_on = self.saved is not None
        self._off()
        try:
            yield
        finally:
            if was_on:
                self._on()


class QuietConsoleUser(ConsoleUser):
    def __init__(self, quiet: Quiet):
        self.quiet = quiet

    def ask(self, question: str) -> str:
        with self.quiet.paused():
            print(f"\rDanceBuddy: {question}")
            return input("you> ").strip()


def go(request: str, model: str | None = None, verbose: bool = False) -> str:
    trace, log = _paths("go")
    quiet = Quiet(log, enabled=not verbose)
    s = AgentSession(model=model)
    print("DanceBuddy is working on it...", flush=True)
    with quiet.working():
        reply = s.send(request)
    s.save(trace)
    print(f"\nDanceBuddy: {reply}\n\n({s.calls} model calls, ${s.cost:.2f}; details in {log.relative_to(ROOT)})")
    return reply


def chat(model: str | None = None, first: str | None = None, verbose: bool = False) -> None:
    trace, log = _paths("chat")
    quiet = Quiet(log, enabled=not verbose)
    s = AgentSession(model=model, user=QuietConsoleUser(quiet))
    print("DanceBuddy: Hi! Tell me where the dance video is and which plush to use (a rigged character or a .glb),\n"
          "and anything special you'd like. Press Enter on an empty line to stop.")
    text = first
    while True:
        if text is None:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        print("(working...)", flush=True)
        with quiet.working():
            reply = s.send(text)
        s.save(trace)
        print(f"\nDanceBuddy: {reply}")
        text = None
    s.save(trace)
    print(f"\nBye! ({s.calls} model calls, ${s.cost:.2f}. Conversation: {trace.relative_to(ROOT)}, "
          f"details: {log.relative_to(ROOT)})")


if __name__ == "__main__":
    chat(first=" ".join(a for a in sys.argv[1:] if a != "--verbose") or None, verbose="--verbose" in sys.argv)
