"""Dance tools: what the agent uses to plan, once a dance and a character are open.

  read      get_dance_score, get_dance_signals, get_leg_events, get_character_spec, get_plan_format
  act       submit_plan: the plan (plan.py, moves.py) becomes motion curves (retarget.py, legs.py) and is checked
            in numbers (checks.py)
  check     check_arm_geometry, look_at_frames (inspect.py: the posed rig, and frames reviewed by a vision model)

A Toolbox holds one dance's state (its attempts and looks) and can switch tools off for ablations. open_dance makes
one for a perceived run and a rigged character; keep_attempt saves the chosen attempt as a version to render.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from dancebuddy import llm
from dancebuddy.agent.tools.dance import moves as moves_mod
from dancebuddy.agent.tools.dance.plan import EMPHASIS_MOVES, KNOBS, LAYERS, PLAN_SCHEMA, default_plan
from dancebuddy.agent.tools.dance.inspect import check_arm_geometry, look_at_frames, pick_frames
from dancebuddy.agent.tools.dance.checks import check_curves
from dancebuddy.agent.tools.dance.retarget import retarget
from dancebuddy.agent.tools.character import spec as character_kb
from dancebuddy.agent.tools.perception.signals import compute_signals, unit_activity

FRAMES_ARGS = {"type": "object", "properties": {"attempt": {"type": "integer"}, "unit_id": {"type": "integer"},
                                                 "frames": {"type": "array", "items": {"type": "integer"}}}}
SPECS = {
    "get_dance_score": llm.tool_spec(
        "get_dance_score", "Move units (time, label, which body regions move), accents and beats, as frames and seconds.",
        {"type": "object", "properties": {}}),
    "get_dance_signals": llm.tool_spec(
        "get_dance_signals", "The dancer's continuous signals (arm raise, torso lean/side/twist, head nod/tilt, hip "
        "bounce/sway, foot lift): meaning, typical range, and which move most in each unit.",
        {"type": "object", "properties": {}}),
    "get_leg_events": llm.tool_spec(
        "get_leg_events", "The leg moves found (jump, kick, step) with their index, time, side and size, and how each "
        "is translated by default.", {"type": "object", "properties": {}}),
    "get_character_spec": llm.tool_spec(
        "get_character_spec", "How this character moves: its parts and how they connect, its pose vocabulary, its "
        "range, and how human parts translate.", {"type": "object", "properties": {}}),
    "get_plan_format": llm.tool_spec(
        "get_plan_format", "The plan's knobs, their ranges and defaults, the emphasis moves, and how to write your "
        "own moves.", {"type": "object", "properties": {}}),
    "submit_plan": llm.tool_spec(
        "submit_plan", "Submit a plan. It is carried out on the character and checked; returns the checks per unit.",
        PLAN_SCHEMA),
    "check_arm_geometry": llm.tool_spec(
        "check_arm_geometry", "Check the arms on the posed character (no rendering, a few seconds): where an arm "
        "folds or sinks into the body. On the latest attempt unless `attempt` is given; on `frames` (at most 24) or on "
        "a unit's key frames.", FRAMES_ARGS),
    "look_at_frames": llm.tool_spec(
        "look_at_frames", "Render up to 8 frames of an attempt small, next to the dancer at the same moments, and get "
        "a reviewer's notes per frame (copy 0-2, problems, a fix). About half a minute; limited number of looks per "
        "run. On the latest attempt unless `attempt` is given; on `frames` or on a unit's accents and beats.",
        FRAMES_ARGS),
}

LEG_DEFAULTS = {"jump": "squash to wind up, stretch and hop, squash on landing",
                "kick": "a big lean away from the kicking leg with a small hop",
                "step": "a small bob with a lean onto the stepping side"}


def score_view(score: dict) -> dict:
    fps = float(score["fps"])
    return {"fps": fps, "duration_s": score["duration_s"], "n_beats": len(score["beats"]),
            "accents": [{"frame": a, "time_s": round(a / fps, 2)} for a in score["accents"]],
            "units": [{k: u[k] for k in ("id", "start", "end", "start_s", "end_s", "label", "dominant_regions",
                                         "region_share", "features", "accents") if k in u} for u in score["units"]]}


class Toolbox:
    """One run's tools. blend: the rigged character (geometry check, looks); source: the dance video (looks)."""

    def __init__(self, kp: dict, score: dict, rig: dict, spec: dict | None, out: Path, *,
                 blend: str | None = None, source: str | None = None, model: str | None = None,
                 max_attempts: int = 4, max_looks: int = 4, use_looks: bool = True, use_geometry: bool = True,
                 use_moves: bool = True):
        self.kp, self.score, self.out = kp, score, Path(out)
        self.source, self.model = source, model
        self.max_attempts, self.max_looks = max_attempts, max_looks
        self.use = {"looks": use_looks, "geometry": use_geometry, "moves": use_moves}
        self.validator = Draft202012Validator(PLAN_SCHEMA)
        self.signals = compute_signals(kp)
        self.rig, self.spec, self.blend = rig, spec or {}, blend
        self.events = retarget(kp, rig, signals=self.signals, beats=score["beats"], score=score)["leg_events"]
        self.attempts: list[dict] = []
        self.looks: list[dict] = []

    def specs(self) -> list[dict]:
        off = {"look_at_frames": not self.use["looks"], "check_arm_geometry": not self.use["geometry"]}
        return [s for name, s in SPECS.items() if not off.get(name, False)]

    def missing(self) -> list[str]:
        return [n for n, k in (("looking at rendered frames", "looks"), ("the arm geometry check", "geometry"),
                               ("writing your own moves", "moves")) if not self.use[k]]

    def call_dict(self, name: str, args: dict) -> dict:
        if name not in {s["function"]["name"] for s in self.specs()}:
            return {"error": f"unknown tool {name}"}
        return getattr(self, "t_" + name)(args)

    def call(self, name: str, args: dict) -> str:
        return json.dumps(self.call_dict(name, args))

    # ---- read
    def t_get_dance_score(self, args):
        return score_view(self.score)

    def t_get_dance_signals(self, args):
        return {"signals": self.score.get("signal_stats"), "units": unit_activity(self.score)}

    def t_get_leg_events(self, args):
        fps = float(self.score["fps"])
        return [{"index": i, "type": e["type"], "side": e["side"], "time_s": round(e["peak"] / fps, 2),
                 "size_torso_lengths": round(e["size"], 2), "default": LEG_DEFAULTS[e["type"]]}
                for i, e in enumerate(self.events)]

    def t_get_character_spec(self, args):
        return {k: v for k, v in self.spec.items() if k not in ("physics", "rig")}

    def t_get_plan_format(self, args):
        fmt = {"unit_knobs": {k: {"default": v[0], "range": [v[1], v[2]], "meaning": v[3]} for k, v in KNOBS.items()},
               "legs": "per leg event index: skip (bool) and size (x default translation, 0.3-1.6)",
               "emphasis_moves": {k: v[0] for k, v in EMPHASIS_MOVES.items()},
               "emphasis_size": "0.1-1.0 of the move's full size", "default_plan": default_plan(self.score),
               "layers": {"how": "plan.layers: motions that are not in the dance, each over [start_s, end_s] "
                                 "(seconds). Use them when the user asks for something other than the copy, e.g. "
                                 "'spin the whole time' = a spin layer over the whole clip. They add onto the copy; "
                                 "add a still layer over the same window to drop the copy there (layers only). "
                                 "submit_plan then reports checks.requests: whether each layer showed.",
                          "types": {k: v[0] + (f" (size default {v[1]} {v[2]})" if v[1] is not None else "")
                                    for k, v in LAYERS.items()}}}
        if self.use["moves"]:
            fmt["moves"] = {"how": moves_mod.__doc__.split("\n\n", 1)[1],
                            "poses": {k: dict(zip(("raise_", "forward", "bend", "fist"), v))
                                      for k, v in moves_mod.POSES.items()}}
        return fmt

    # ---- act
    def t_submit_plan(self, args):
        if len(self.attempts) >= self.max_attempts:
            return {"error": "no attempts left; stop and summarise"}
        errs = [e.message for e in self.validator.iter_errors(args)]
        if errs:
            return {"error": "plan does not match the schema", "details": errs[:5]}
        if not self.use["moves"] and args.get("moves"):
            return {"error": "own moves are not available in this run; use the knobs and emphasis"}
        try:
            moves_mod.parse(args.get("moves", ""))
        except moves_mod.MoveError as e:
            return {"error": "moves do not parse", "details": str(e)}
        curves = retarget(self.kp, self.rig, signals=self.signals, beats=self.score["beats"], plan=args,
                          score=self.score)
        res = check_curves(curves, self.score, self.rig, self.spec, plan=args)
        k = len(self.attempts) + 1
        d = self.out / f"attempt_{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "plan.json").write_text(json.dumps(args, indent=1))
        (d / "curves.json").write_text(json.dumps(curves))
        (d / "checks.json").write_text(json.dumps(res, indent=1))
        self.attempts.append({"attempt": k, "dir": str(d), "result": res})
        print(f"  attempt {k}: passed={res['passed']} "
              + " ".join(f"{n}={'ok' if c['passed'] else 'FAIL'}" for n, c in res["checks"].items()))
        return {"attempt": k, "attempts_left": self.max_attempts - k,
                **{kk: v for kk, v in res.items() if kk != "all_units"}}

    # ---- check
    def _attempt(self, args):
        k = int(args.get("attempt") or len(self.attempts))
        return next((x for x in self.attempts if x["attempt"] == k), self.attempts[-1])

    def t_check_arm_geometry(self, args):
        if not self.attempts:
            return {"error": "submit a plan first"}
        if not self.blend:
            return {"error": "this check is not available in this run"}
        att, n = self._attempt(args), len(self.kp["image"])
        frames = pick_frames(self.score, n, args.get("frames"), args.get("unit_id"), most=24)
        if not args.get("frames") and args.get("unit_id") is None:
            frames = list(range(0, n, max(1, n // 24)))[:24]
        res = check_arm_geometry(Path(att["dir"]) / "curves.json", self.blend, self.score, frames,
                                 Path(att["dir"]) / "geometry")
        print(f"  geometry on attempt {att['attempt']}: {len(res['problems'])} problems in {len(frames)} frames")
        return {"attempt": att["attempt"], **res}

    def t_look_at_frames(self, args):
        if not self.attempts:
            return {"error": "submit a plan first"}
        if not (self.blend and self.source):
            return {"error": "this check is not available in this run"}
        if len(self.looks) >= self.max_looks:
            return {"error": "no looks left; decide from what you have"}
        att, n = self._attempt(args), len(self.kp["image"])
        frames = pick_frames(self.score, n, args.get("frames"), args.get("unit_id"), most=8)
        res = look_at_frames(Path(att["dir"]) / "curves.json", self.blend, self.score, self.spec, self.source, frames,
                             Path(att["dir"]) / f"look_{len(self.looks) + 1}", self.model)
        self.looks.append({"attempt": att["attempt"], **res})
        print(f"  look {len(self.looks)} at attempt {att['attempt']}, frames {frames}")
        return {"attempt": att["attempt"], "looks_left": self.max_looks - len(self.looks), "frames": frames,
                "notes": res["notes"]}


SESSION_SPECS = {
    "open_dance": llm.tool_spec(
        "open_dance", "Open a perceived run (runs/NAME with dance_score.json) with a rigged character for planning. "
        "The dance tools (get_dance_score ... submit_plan, look_at_frames) work on it afterwards. Attempts are "
        "written to runs/NAME/<workdir>.",
        {"type": "object", "properties": {"run": {"type": "string"}, "character": {"type": "string"},
                                          "workdir": {"type": "string", "description": "default agent_work"},
                                          "attempts": {"type": "integer", "description": "plans allowed (default 6)"},
                                          "own_moves": {"type": "boolean", "description": "allow plan.moves (default "
                                                        "false: they blend in within a few frames and look like "
                                                        "snapping; only when the user asks for a specific pose)"}},
         "required": ["run", "character"]}),
    "keep_attempt": llm.tool_spec(
        "keep_attempt", "Copy an attempt (default: the best by the checks) to runs/NAME/<system> so it can be "
        "rendered.", {"type": "object", "properties": {"attempt": {"type": "integer"}, "system": {"type": "string"}}}),
}


def open_dance(a: dict, model: str | None = None) -> tuple[Toolbox | None, Path | None, Path | None, dict]:
    """(toolbox, run dir, character prefix, result for the model)."""
    from dancebuddy.agent.tools.perception.video import load_keypoints
    from dancebuddy.agent.tools.workspace import rel, resolve
    run, char = resolve(a["run"]), resolve(a["character"])
    if not (run / "dance_score.json").exists():
        return None, None, None, {"error": f"{run} has no dance_score.json: run perceive_dance first"}
    if not Path(str(char) + "_rig.json").exists():
        return None, None, None, {"error": f"no rigged character at {char} (see list_characters)"}
    score = json.loads((run / "dance_score.json").read_text())
    src = (run / "source.txt").read_text().strip() if (run / "source.txt").exists() else score.get("source")
    out = run / (a.get("workdir") or "agent_work")
    box = Toolbox(load_keypoints(run / "keypoints.npz"), score, character_kb.rig(char), character_kb.spec(char), out,
                  blend=character_kb.blend(char), source=src if src and Path(src).exists() else None, model=model,
                  max_attempts=int(a.get("attempts") or 6), max_looks=4, use_moves=bool(a.get("own_moves", False)))
    return box, run, char, {"run": rel(run), "character": rel(char), "duration_s": score["duration_s"],
                            "units": len(score["units"]), "accents": len(score["accents"]), "attempts_dir": rel(out),
                            "can_look": bool(box.blend and box.source),
                            "note": "the dance tools are now available: read the dance and the character, then plan"}


def keep_attempt(box: Toolbox | None, run_dir: Path | None, char: Path | None, a: dict) -> dict:
    import re
    import shutil
    from dancebuddy.agent.tools.dance.checks import rank
    from dancebuddy.agent.tools.workspace import rel
    if box is None or not box.attempts:
        return {"error": "no attempts yet: open_dance and submit_plan first"}
    att = (next((x for x in box.attempts if x["attempt"] == int(a["attempt"])), None) if a.get("attempt")
           else max(box.attempts, key=lambda x: rank(x["result"])))
    if att is None:
        return {"error": f"no attempt {a['attempt']}; there are {len(box.attempts)}"}
    system = re.sub(r"[^A-Za-z0-9_-]", "", a.get("system") or "agent") or "agent"
    dest = run_dir / system
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("plan.json", "curves.json", "checks.json"):
        shutil.copy(Path(att["dir"]) / f, dest / f)
    (dest / "summary.json").write_text(json.dumps({
        "attempt": att["attempt"], "passed": att["result"]["passed"], "character": str(char),
        "checks": {k: v.get("passed") for k, v in att["result"]["checks"].items()}}, indent=1))
    return {"kept": att["attempt"], "to": rel(dest), "passed": att["result"]["passed"],
            "next": f"render_video(system={system!r})"}
