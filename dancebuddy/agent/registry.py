"""The tool registry: the tools offered on one turn, their JSON schemas for the model, and the dispatch.

A Tool pairs an OpenAI-style function spec with a handler that takes the parsed arguments and returns a JSON-able
dict. ToolRegistry.call never raises: a failing tool comes back to the model as {"error": ...} so it can react.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    spec: dict
    fn: Callable[[dict], dict]

    @property
    def name(self) -> str:
        return self.spec["function"]["name"]


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[dict]:
        return [t.spec for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)

    def call(self, name: str, args: dict, missing: Callable[[str], str] | None = None) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": missing(name) if missing else f"unknown tool {name}"}
        try:
            return tool.fn(args)
        except SystemExit as e:
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001  reported to the model, not fatal
            return {"error": f"{type(e).__name__}: {e}"}
