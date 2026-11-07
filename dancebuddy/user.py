"""The person on the other side of the agent, as the ask_user tool sees them.

  ConsoleUser   asks in the terminal (chat)
  NoUser        nobody to ask (go, the evaluation): the agent decides itself
  ScriptedUser  fixed answers, for tests
"""

from __future__ import annotations


class NoUser:
    can_answer = False

    def ask(self, question: str) -> str:
        return "nobody can answer in this run; decide yourself and say what you chose"


class ConsoleUser:
    can_answer = True

    def ask(self, question: str) -> str:
        print(f"\nDanceBuddy asks: {question}")
        return input("you> ").strip()


class ScriptedUser:
    can_answer = True

    def __init__(self, answers: list[str]):
        self.answers = list(answers)

    def ask(self, question: str) -> str:
        return self.answers.pop(0) if self.answers else ""
