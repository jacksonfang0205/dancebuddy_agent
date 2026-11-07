"""The agent with only the dance tools, on a dance that is already perceived and a character that is already
rigged. The evaluation runs it this way (and with tools switched off for the ablations), so every system starts
from the same inputs. The full agent (director.py) uses the same dance tools after it has found and prepared the
inputs itself."""

from __future__ import annotations

import json
from pathlib import Path

from dancebuddy.agent import llm
from dancebuddy.tools.dance.checks import rank
from dancebuddy.agent.prompts import DANCE_PLANNING as SYSTEM
from dancebuddy.tools.dance import Toolbox


def run_agent(kp: dict, score: dict, rig: dict, spec: dict, out_dir: str | Path, max_attempts: int = 4,
              max_turns: int = 24, model: str | None = None, blend: str | None = None,
              source: str | None = None, max_looks: int = 4, use_looks: bool = True, use_geometry: bool = True,
              use_moves: bool = True, new_character: str | None = None, character_request: str = "",
              tripo=None) -> dict:
    """Run the agent on one dance. blend: the rigged character (geometry check, looks); source: the dance video
    (looks). use_looks / use_geometry / use_moves switch a capability off (ablations): its tool is not offered and
    the agent is told. new_character: the agent first makes a character with this name (from character_request)
    and then dances with it; rig may be None. Writes trace.json, summary.json, the best plan / curves / checks and
    every attempt."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    box = Toolbox(kp, score, rig, spec, out, blend=blend, source=source, model=model, max_attempts=max_attempts,
                  max_looks=max_looks, use_looks=use_looks, use_geometry=use_geometry, use_moves=use_moves,
                  new_character=new_character, tripo=tripo)
    missing = box.missing()
    note = f" In this run you do not have {', '.join(missing)}; skip those steps." if missing else ""
    ask = ("Make the character dance this dance. Start by reading the dance and the character." if not new_character
           else f"Make a new character ({character_request}) and then make it dance this dance. Start by making "
                "the character.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask + note}]
    final_text = ""
    calls_made, cost = 0, 0.0
    for _ in range(max_turns):
        resp = llm.call(messages, tools=box.specs(), model=model)
        calls_made += 1
        cost += float((resp.get("usage") or {}).get("cost") or 0.0)
        calls = resp["tool_calls"]
        assistant = {"role": "assistant", "content": resp["text"] or None}
        if calls:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                       for c in calls]
        messages.append(assistant)
        if not calls:
            final_text = resp["text"]
            break
        for c in calls:
            a = c["arguments"]
            result = (json.dumps({"error": "arguments were not valid JSON", "details": a["_parse_error"]})
                      if "_parse_error" in a else box.call(c["name"], a))
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    (out / "trace.json").write_text(json.dumps(messages, indent=1))
    attempts, looks = box.attempts, box.looks
    if not attempts:
        raise RuntimeError("the agent never submitted a valid plan; see trace.json")
    best = max(attempts, key=lambda x: rank(x["result"]))
    for f in ("plan.json", "curves.json", "checks.json"):
        (out / f).write_text((Path(best["dir"]) / f).read_text())
    summary = {"attempts": len(attempts), "best_attempt": best["attempt"], "passed": best["result"]["passed"],
               "history": [{"attempt": x["attempt"], "passed": x["result"]["passed"],
                            "problems": sum(len(u["problems"]) for u in x["result"]["units"])} for x in attempts],
               "looks": [{"attempt": x["attempt"], "frames": x["frames"], "picture": x["picture"]} for x in looks],
               "final_message": final_text, "model": model or llm.DEFAULT_MODEL,
               "capabilities": box.use, "character": str(box.char_prefix) if box.char_prefix else None,
               "character_steps": box.charbox.made if box.charbox else [], "llm_calls": calls_made + len(looks),
               "cost_usd": round(cost + sum(x.get("cost_usd", 0.0) for x in looks), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"agent: {len(attempts)} attempt(s), best = {best['attempt']}, passed = {best['result']['passed']}")
    return summary
