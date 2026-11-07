"""Thin cached wrapper around OpenRouter (OpenAI-compatible chat completions).

Set OPENROUTER_API_KEY (in .env or your shell). Pick the model with
DANCEBUDDY_MODEL, using OpenRouter's names, e.g. "openai/gpt-6-sol",
"openai/gpt-6-luna" or an "anthropic/..." model; see https://openrouter.ai/models.

Every request is hashed (model + messages + tools + settings) and the reply
is stored under .cache/llm/. Re-running the same step is free and gives the
same answer, which is what makes the README quick start reproducible: commit
the cache for the example clip and a reader can replay the agent run without
spending anything. A ledger (.cache/llm/ledger.jsonl) records tokens and cost.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("DANCEBUDDY_CACHE", ROOT / ".cache" / "llm"))
DEFAULT_MODEL = os.environ.get("DANCEBUDDY_MODEL", "openai/gpt-6-sol")
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def image_block(path: str | Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def tool_spec(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def call(messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
         max_tokens: int = 8000) -> dict:
    """messages in OpenAI chat format. Returns
    {"text": str, "tool_calls": [{"id", "name", "arguments": dict}], "finish_reason", "usage"}."""
    model = model or DEFAULT_MODEL
    effort = os.environ.get("DANCEBUDDY_REASONING_EFFORT", "")
    payload = {"model": model, "messages": messages, "tools": tools or [], "max_tokens": max_tokens, "effort": effort}
    key = _key(payload)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; put it in .env or export it in your shell")
    client = OpenAI(base_url=OPENROUTER_URL, api_key=api_key)
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
    extra = {"usage": {"include": True}}  # OpenRouter reports the exact cost
    if effort:
        extra["reasoning"] = {"effort": effort}
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs, extra_body=extra)
    choice = resp.choices[0]
    msg = choice.message
    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as e:
            args = {"_parse_error": str(e), "_raw": tc.function.arguments}
        calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
    usage = resp.usage.model_dump() if resp.usage else {}
    out = {"text": msg.content or "", "tool_calls": calls, "finish_reason": choice.finish_reason,
           "usage": usage, "model": model}
    path.write_text(json.dumps(out))
    with open(CACHE / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"key": key, "model": model, "seconds": round(time.time() - t0, 2),
                            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                            "cost_usd": usage.get("cost")}) + "\n")
    return out


def spent() -> float:
    """Total USD recorded in the ledger."""
    p = CACHE / "ledger.jsonl"
    if not p.exists():
        return 0.0
    return sum((json.loads(l).get("cost_usd") or 0.0) for l in p.read_text().splitlines() if l.strip())


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a reply (tolerates ```json fences)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj
