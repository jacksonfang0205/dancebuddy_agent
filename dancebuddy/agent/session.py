"""The agent loop: one AgentSession holds a conversation with the model and runs the tools it calls.

Each turn the model gets the conversation and the tools offered now (agent/tools: build_tools); it answers with
tool calls, which run through the registry, or with a reply, which ends the turn. A later user message continues
the same session, so "now make it spin" revises the same dance.

  AgentSession(user=ConsoleUser())    the full agent (chat.py: go, chat)
  plan_dance(...)                     the same loop with only the dance tools, on a dance that is already perceived
                                      and a character that is already rigged: how the evaluation runs it, and the
                                      ablations switch tools off
"""

from __future__ import annotations

import json
from pathlib import Path

from dancebuddy import llm
from dancebuddy.agent.prompts import CHAT_RULE, DANCE_PLANNING, DIRECTOR, GO_RULE
from dancebuddy.agent.tools import build_tools
from dancebuddy.agent.tools.character import Characters
from dancebuddy.agent.tools.dance import SPECS as DANCE_SPECS, Toolbox
from dancebuddy.agent.tools.workspace import resolve
from dancebuddy.sandbox import ROOT
from dancebuddy.user import NoUser


class AgentSession:
    VIEW_SPEC = llm.tool_spec(
        "view_image", "Look at an image file (a character view, a rendered frame). It is shown to you after the "
        "tool results.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
    ASK_SPEC = llm.tool_spec(
        "ask_user", "Ask the user a question and wait for the answer.",
        {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]})

    def __init__(self, model: str | None = None, user=None, log=print, max_turns: int = 60,
                 dance_only: bool = False):
        self.model, self.log, self.max_turns, self.dance_only = model, log, max_turns, dance_only
        self.user = user or NoUser()
        self.box: Toolbox | None = None
        self.run_dir: Path | None = None
        self.char: Path | None = None
        self.characters = Characters()
        self.cost, self.calls, self.snippets = 0.0, 0, 0
        self.pending_images: list[str] = []
        self.tool_log: list[dict] = []
        system = (DANCE_PLANNING if dance_only else DIRECTOR.format(
            root=ROOT, chat_rule=CHAT_RULE if self.user.can_answer else GO_RULE, dance=DANCE_PLANNING))
        self.messages: list[dict] = [{"role": "system", "content": system}]

    # ------------------------------------------------------------------ the loop
    def send(self, text: str) -> str:
        """One user message: the agent works until it replies. Returns the reply."""
        self.messages.append({"role": "user", "content": text})
        if self.box is not None and not self.dance_only:  # a fresh budget of plans and looks for each request
            self.box.max_attempts = len(self.box.attempts) + 6
            self.box.max_looks = len(self.box.looks) + 4
        for _ in range(self.max_turns):
            tools = build_tools(self)
            resp = llm.call(self.messages, tools=tools.specs(), model=self.model)
            self.calls += 1
            self.cost += float((resp.get("usage") or {}).get("cost") or 0.0)
            calls = resp["tool_calls"]
            msg = {"role": "assistant", "content": resp["text"] or None}
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                     for c in calls]
            self.messages.append(msg)
            if not calls:
                return resp["text"]
            if resp["text"]:
                self.log(f"  {resp['text'].strip()[:300]}")
            for c in calls:
                self.messages.append({"role": "tool", "tool_call_id": c["id"],
                                      "content": self.call(c["name"], c["arguments"], tools)})
                tools = build_tools(self)  # open_dance adds the dance tools for the next call
            if self.pending_images:  # images go in a user message after the tool results
                content = [{"type": "text", "text": "The images you asked to view: " + ", ".join(self.pending_images)}]
                content += [llm.image_block(p) for p in self.pending_images]
                self.messages.append({"role": "user", "content": content})
                self.pending_images = []
        return "(stopped: too many steps without a reply)"

    def call(self, name: str, args: dict, tools=None) -> str:
        self.log(f"> {name} {json.dumps(args)[:200]}")
        if "_parse_error" in args:
            res = {"error": "arguments were not valid JSON", "details": args["_parse_error"]}
        else:
            tools = tools or build_tools(self)
            res = tools.call(name, args, missing=lambda n: f"{n} needs an open dance: run open_dance first"
                             if n in DANCE_SPECS else f"unknown tool {n}")
        if isinstance(res, dict) and "error" in res:
            self.log(f"  ! {str(res['error'])[:300]}")
        self.tool_log.append({"tool": name, "args": args, "error": res.get("error") if isinstance(res, dict) else None})
        return json.dumps(res, default=str)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"messages": self.messages, "tools": self.tool_log, "llm_calls": self.calls,
                                    "cost_usd": round(self.cost, 4), "model": self.model or llm.DEFAULT_MODEL},
                                   indent=1, default=str))

    # ------------------------------------------------------------------ tools that live in the session
    def view_image(self, a: dict) -> dict:
        p = resolve(a["path"])
        if not p.exists() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            return {"error": f"no png/jpg image at {p}"}
        self.pending_images.append(str(p))
        return {"ok": True, "note": "the image follows the tool results"}

    def ask_user(self, a: dict) -> dict:
        return {"answer": self.user.ask(a["question"])}


def plan_dance(kp: dict, score: dict, rig: dict, spec: dict, out_dir: str | Path, max_attempts: int = 4,
               max_turns: int = 24, model: str | None = None, blend: str | None = None, source: str | None = None,
               max_looks: int = 4, use_looks: bool = True, use_geometry: bool = True, use_moves: bool = True) -> dict:
    """The agent with only the dance tools (the evaluation's setting). Writes trace.json, summary.json, every
    attempt and the best plan / curves / checks to out_dir."""
    from dancebuddy.agent.tools.dance.checks import rank
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = AgentSession(model=model, max_turns=max_turns, dance_only=True, log=lambda *_: None)
    s.box = Toolbox(kp, score, rig, spec, out, blend=blend, source=source, model=model, max_attempts=max_attempts,
                    max_looks=max_looks, use_looks=use_looks, use_geometry=use_geometry, use_moves=use_moves)
    missing = s.box.missing()
    note = f" In this run you do not have {', '.join(missing)}; skip those steps." if missing else ""
    reply = s.send("Make the character dance this dance. Start by reading the dance and the character." + note)
    (out / "trace.json").write_text(json.dumps(s.messages, indent=1))
    attempts, looks = s.box.attempts, s.box.looks
    if not attempts:
        raise RuntimeError("the agent never submitted a valid plan; see trace.json")
    best = max(attempts, key=lambda x: rank(x["result"]))
    for f in ("plan.json", "curves.json", "checks.json"):
        (out / f).write_text((Path(best["dir"]) / f).read_text())
    summary = {"attempts": len(attempts), "best_attempt": best["attempt"], "passed": best["result"]["passed"],
               "history": [{"attempt": x["attempt"], "passed": x["result"]["passed"],
                            "problems": sum(len(u["problems"]) for u in x["result"]["units"])} for x in attempts],
               "looks": [{"attempt": x["attempt"], "frames": x["frames"], "picture": x["picture"]} for x in looks],
               "final_message": reply, "model": model or llm.DEFAULT_MODEL, "capabilities": s.box.use,
               "llm_calls": s.calls + len(looks),
               "cost_usd": round(s.cost + sum(x.get("cost_usd", 0.0) for x in looks), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"agent: {len(attempts)} attempt(s), best = {best['attempt']}, passed = {best['result']['passed']}")
    return summary
