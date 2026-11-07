"""The director: one agent that runs the whole DanceBuddy pipeline from a request in plain words.

It finds the inputs (the dancer's video, the character's GLB or an imported character), runs each stage with
shell and Python tools (perception, character import and rigging, rendering, the side-by-side video), and plans
the dance itself with the dance tools of tools.py once a run is open. It works one-shot (`go`) or as a
conversation (`chat`), where later requests ("now make it spin the whole time", "bigger arms in the chorus")
revise the same run.

  tools  find_files, run_shell, run_python, view_image            (workspace)
         list_characters, import_character                        (characters; Tripo tools when TRIPO_API_KEY is set)
         open_dance                                               (load a perceived run + a character)
         get_dance_score ... submit_plan, check_arm_geometry, look_at_frames   (tools.py, once a dance is open)
         keep_attempt                                             (write the chosen attempt for rendering)
         ask_user                                                 (chat only)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dancebuddy.agent import llm
from dancebuddy.tools.character.make import CHARACTER_SPECS, CharacterTools
from dancebuddy.tools.dance.checks import rank
from dancebuddy.tools.character import spec as character_kb
from dancebuddy.agent.prompts import CHAT_RULE, DANCE_PLANNING as DANCE_SYSTEM, DIRECTOR, GO_RULE
from dancebuddy.tools.dance import SPECS as DANCE_SPECS, Toolbox

ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
MODEL_EXT = {".glb", ".gltf", ".fbx", ".obj"}

# commands the agent may not run: they delete, publish, install or reach outside the project
BLOCKED = [r"\brm\s+-[a-z]*r", r"\brm\s+-[a-z]*f", r"\bsudo\b", r"\bgit\s+(push|reset|clean|checkout|rebase)",
           r"\b(curl|wget|scp|ssh)\b", r"\b(pip|uv)\s+(install|add|remove)\b", r"\bbrew\b", r">\s*/dev/sd",
           r"\bmkfs\b", r"\bchmod\s+-R\b", r"\bkill(all)?\b", r"\.env\b"]


WORK_SPECS = {
    "find_files": llm.tool_spec(
        "find_files", "List files under a folder whose names match a glob (e.g. '*.mp4'), newest first, with size "
        "and duration for videos. kind='video' or 'model' lists all videos / 3D models.",
        {"type": "object", "properties": {"folder": {"type": "string", "description": "default: the project"},
                                          "pattern": {"type": "string"},
                                          "kind": {"type": "string", "enum": ["video", "model", "any"]},
                                          "recursive": {"type": "boolean"}}}),
    "run_shell": llm.tool_spec(
        "run_shell", "Run a shell command in the project folder (zsh/bash, the project's python first on PATH). "
        "Returns the exit code and the end of the output. Destructive commands are refused.",
        {"type": "object", "properties": {"command": {"type": "string"},
                                          "timeout_s": {"type": "integer", "description": "default 1800"}},
         "required": ["command"]}),
    "run_python": llm.tool_spec(
        "run_python", "Run a Python snippet with the project's Python (numpy, json, the dancebuddy package). Print "
        "what you want back.", {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}),
    "view_image": llm.tool_spec(
        "view_image", "Look at an image file (a character view, a rendered frame). It is shown to you after the "
        "tool results.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    "perceive_dance": llm.tool_spec(
        "perceive_dance", "Run perception on a dance video: MediaPipe pose, then beats, accents, move units and the "
        "dancer's signals, written to runs/NAME (about 1 s per second of video). Returns what it found.",
        {"type": "object", "properties": {"video": {"type": "string"},
                                          "name": {"type": "string", "description": "short run name, e.g. dance2"},
                                          "max_seconds": {"type": "number"}}, "required": ["video", "name"]}),
    "render_video": llm.tool_spec(
        "render_video", "Render a kept version (keep_attempt) in Blender and put it next to the dancer with FFmpeg. "
        "preview (default true) = fast untextured; false = the final textured render (several minutes).",
        {"type": "object", "properties": {"system": {"type": "string"},
                                          "preview": {"type": "boolean"},
                                          "engine": {"type": "string", "enum": ["eevee", "cycles"]}},
         "required": ["system"]}),
    "list_characters": llm.tool_spec(
        "list_characters", "The rigged characters ready to dance (prefixes for open_dance and render --char), with "
        "their front view.", {"type": "object", "properties": {}}),
    "import_character": llm.tool_spec(
        "import_character", "Import a GLB and build the soft rig: becomes assets/NAME_soft. NAME: letters and digits "
        "only. Returns the parts found (arms, hair, body vertices) and the views folder.",
        {"type": "object", "properties": {"model": {"type": "string"}, "name": {"type": "string"},
                                          "yaw": {"type": "number"}}, "required": ["model", "name"]}),
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
        "rendered and composed.", {"type": "object", "properties": {"attempt": {"type": "integer"},
                                                                    "system": {"type": "string"}}}),
}
ASK_SPEC = llm.tool_spec("ask_user", "Ask the user a question and wait for the answer.",
                         {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]})
TRIPO_TOOLS = ("create_character_from_text", "create_character_from_image", "segment_character")


def _video_seconds(p: Path) -> float | None:
    try:
        import cv2
        cap = cv2.VideoCapture(str(p))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return round(n / fps, 1) if fps else None
    except Exception:  # noqa: BLE001
        return None


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else "...\n" + text[-limit:]


class Director:
    def __init__(self, model: str | None = None, chat: bool = False, ask=None, log=print, max_turns: int = 60,
                 dance_only: bool = False):
        """dance_only: only the dance tools, on a dance set up by plan_dance (the evaluation setting)."""
        self.model, self.chat, self.ask, self.log, self.max_turns = model, chat, ask, log, max_turns
        self.dance_only = dance_only
        self.box: Toolbox | None = None
        self.run_dir: Path | None = None
        self.char: Path | None = None
        self.tripo = bool(os.environ.get("TRIPO_API_KEY"))
        self.makers: dict[str, CharacterTools] = {}
        self.cost, self.calls = 0.0, 0
        self.pending_images: list[str] = []
        self.commands: list[dict] = []
        system = (DANCE_SYSTEM if dance_only else
                  DIRECTOR.format(root=ROOT, chat_rule=CHAT_RULE if chat else GO_RULE, dance=DANCE_SYSTEM))
        self.messages: list[dict] = [{"role": "system", "content": system}]

    # ------------------------------------------------------------------ the loop
    def specs(self) -> list[dict]:
        if self.dance_only:
            return self.box.specs() if self.box is not None else []
        out = list(WORK_SPECS.values())
        if self.chat:
            out.append(ASK_SPEC)
        if self.tripo:
            for name in TRIPO_TOOLS:
                s = json.loads(json.dumps(CHARACTER_SPECS[name]))
                s["function"]["parameters"]["properties"]["name"] = {"type": "string",
                                                                     "description": "the character's name"}
                s["function"]["parameters"].setdefault("required", []).append("name")
                out.append(s)
        if self.box is not None:
            out += self.box.specs()
        return out

    def send(self, text: str) -> str:
        """One user request: the agent works until it replies. Returns the reply."""
        self.messages.append({"role": "user", "content": text})
        if self.box is not None and not self.dance_only:  # a fresh budget of plans and looks for each request
            self.box.max_attempts = len(self.box.attempts) + 6
            self.box.max_looks = len(self.box.looks) + 4
        for _ in range(self.max_turns):
            resp = llm.call(self.messages, tools=self.specs(), model=self.model)
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
                a = c["arguments"]
                result = (json.dumps({"error": "arguments were not valid JSON", "details": a["_parse_error"]})
                          if "_parse_error" in a else self.call(c["name"], a))
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            if self.pending_images:  # images go in a user message after the tool results
                content = [{"type": "text", "text": "The images you asked to view: "
                            + ", ".join(self.pending_images)}]
                content += [llm.image_block(p) for p in self.pending_images]
                self.messages.append({"role": "user", "content": content})
                self.pending_images = []
        return "(stopped: too many steps without a reply)"

    def call(self, name: str, args: dict) -> str:
        self.log(f"> {name} {json.dumps(args)[:200]}")
        try:
            fn = getattr(self, "t_" + name, None)
            if self.dance_only and name not in {x["function"]["name"] for x in self.specs()}:
                res = {"error": f"unknown tool {name}"}
            elif fn is not None:
                res = fn(args)
            elif name in TRIPO_TOOLS and self.tripo:
                res = json.loads(self._maker(args.pop("name")).call(name, args))
            elif self.box is not None and name in {x["function"]["name"] for x in self.box.specs()}:
                res = json.loads(self.box.call(name, args))
            else:
                res = {"error": f"{name} needs an open dance: run open_dance first" if name in DANCE_SPECS
                       else f"unknown tool {name}"}
        except SystemExit as e:
            res = {"error": str(e)}
        except Exception as e:  # noqa: BLE001  a failing tool is reported to the agent, not fatal
            res = {"error": f"{type(e).__name__}: {e}"}
        if isinstance(res, dict) and "error" in res:
            self.log(f"  ! {str(res['error'])[:300]}")
        return json.dumps(res, default=str)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"messages": self.messages, "commands": self.commands,
                                    "llm_calls": self.calls, "cost_usd": round(self.cost, 4),
                                    "model": self.model or llm.DEFAULT_MODEL}, indent=1, default=str))

    # ------------------------------------------------------------------ workspace
    def t_find_files(self, a):
        folder = Path(a.get("folder") or ROOT).expanduser()
        if not folder.is_absolute():
            folder = ROOT / folder
        if not folder.is_dir():
            return {"error": f"no folder {folder}"}
        kind = a.get("kind", "any")
        pattern = a.get("pattern") or "*"
        it = folder.rglob(pattern) if a.get("recursive", True) else folder.glob(pattern)
        files = []
        for p in it:
            if not p.is_file() or any(part.startswith(".") for part in p.relative_to(folder).parts):
                continue
            ext = p.suffix.lower()
            if (kind == "video" and ext not in VIDEO_EXT) or (kind == "model" and ext not in MODEL_EXT):
                continue
            if "/frames/" in str(p) or "/preview/" in str(p):
                continue
            files.append(p)
            if len(files) > 400:
                break
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in files[:60]:
            row = {"path": str(p), "mb": round(p.stat().st_size / 1e6, 2)}
            if p.suffix.lower() in VIDEO_EXT:
                row["seconds"] = _video_seconds(p)
            out.append(row)
        return {"folder": str(folder), "found": len(files), "files": out}

    def t_run_shell(self, a):
        cmd = a["command"].strip()
        for pat in BLOCKED:
            if re.search(pat, cmd):
                return {"error": f"refused: commands matching {pat!r} are not allowed here"}
        env = dict(os.environ)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        t0 = time.time()
        self.log(f"  $ {cmd}")
        try:
            p = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=int(a.get("timeout_s") or 1800), executable=shutil.which("bash") or None)
            code, out = p.returncode, (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired as e:
            code, out = -1, f"timed out after {e.timeout} s\n" + str(e.stdout or "")[-2000:]
        secs = round(time.time() - t0, 1)
        self.commands.append({"command": cmd, "exit": code, "seconds": secs})
        self.log(f"    exit {code} in {secs} s")
        return {"exit_code": code, "seconds": secs, "output": _tail(out)}

    def t_run_python(self, a):
        tmp = ROOT / ".cache" / "director"
        tmp.mkdir(parents=True, exist_ok=True)
        f = tmp / f"snippet_{len(self.commands)}.py"
        f.write_text(a["code"])
        res = self.t_run_shell({"command": f"python {shlex.quote(str(f))}", "timeout_s": 600})
        return res

    def t_view_image(self, a):
        p = Path(a["path"]).expanduser()
        p = p if p.is_absolute() else ROOT / p
        if not p.exists() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            return {"error": f"no png/jpg image at {p}"}
        self.pending_images.append(str(p))
        return {"ok": True, "note": "the image follows the tool results"}

    def t_ask_user(self, a):
        if not self.ask:
            return {"answer": "nobody can answer in this run; decide yourself"}
        return {"answer": self.ask(a["question"])}

    # ------------------------------------------------------------------ perception and rendering
    def t_perceive_dance(self, a):
        from dancebuddy.tools.perception.video import perceive
        video = Path(a["video"]).expanduser()
        video = video if video.is_absolute() else ROOT / video
        if not video.exists():
            return {"error": f"no video at {video}"}
        name = re.sub(r"[^A-Za-z0-9_-]", "", a["name"]) or "dance"
        run = ROOT / "runs" / name
        src = run / "source.txt"
        if (run / "dance_score.json").exists() and src.exists() and src.read_text().strip() != str(video.resolve()):
            return {"error": f"runs/{name} already holds another video ({src.read_text().strip()}); pick another name"}
        t0 = time.time()
        score = perceive(video, run, a.get("max_seconds"))
        src.write_text(str(video.resolve()))
        return {"run": f"runs/{name}", "seconds": round(time.time() - t0, 1), "duration_s": score["duration_s"],
                "beats": len(score["beats"]), "accents": len(score["accents"]), "units": len(score["units"])}

    def t_render_video(self, a):
        from dancebuddy.tools.render import compose, render_system
        if self.run_dir is None:
            return {"error": "open_dance and keep_attempt first"}
        system = re.sub(r"[^A-Za-z0-9_-]", "", a["system"])
        if not (self.run_dir / system / "curves.json").exists():
            return {"error": f"no kept version {system}: keep_attempt first"}
        preview = a.get("preview", True)
        t0 = time.time()
        frames = render_system(self.run_dir, system, self.char, a.get("engine", "eevee"), preview=preview)
        out = compose(self.run_dir, [system], preview)
        rel = lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        return {"video": rel(out), "frames": rel(frames), "seconds": round(time.time() - t0, 1),
                "first_frame": rel(frames / "frame_0001.png")}

    # ------------------------------------------------------------------ characters
    def t_list_characters(self, a):
        out = []
        for rig in sorted((ROOT / "assets").glob("*_rig.json")):
            prefix = Path(str(rig)[: -len("_rig.json")])
            if not Path(str(prefix) + ".blend").exists() or "_soft" not in prefix.name:
                continue
            name = prefix.name.split("_")[0]
            views = ROOT / "assets" / f"{name}_views" / "front.png"
            if not views.exists():
                views = ROOT / "assets" / f"{prefix.name}_views" / "front.png"
            parts = json.loads(rig.read_text()).get("parts")
            out.append({"character": str(prefix.relative_to(ROOT)), "parts": parts,
                        "front_view": str(views.relative_to(ROOT)) if views.exists() else None})
        return {"characters": out, "default": "assets/plush_soft"}

    def _maker(self, name: str) -> CharacterTools:
        name = re.sub(r"[^A-Za-z0-9]", "", name) or "character"
        if name not in self.makers:
            self.makers[name] = CharacterTools(name, ROOT / "runs" / "characters" / name)
        return self.makers[name]

    def t_import_character(self, a):
        name = re.sub(r"[^A-Za-z0-9]", "", a["name"])
        if not name:
            return {"error": "the name needs letters or digits"}
        model = Path(a["model"]).expanduser()
        model = model if model.is_absolute() else ROOT / model
        args = {"model": str(model)}
        if a.get("yaw"):
            args["yaw"] = a["yaw"]
        t0 = time.time()
        res = json.loads(self._maker(name).call("import_and_rig_soft", args))
        res["seconds"] = round(time.time() - t0, 1)
        if "views" in res:
            res["front_view"] = res["views"] + "/front.png"
        return res

    # ------------------------------------------------------------------ the dance
    def t_open_dance(self, a):
        from dancebuddy.tools.perception.video import load_keypoints
        run = Path(a["run"]).expanduser()
        run = run if run.is_absolute() else ROOT / run
        if not (run / "dance_score.json").exists():
            return {"error": f"{run} has no dance_score.json: run perception first"}
        char = Path(a["character"]).expanduser()
        char = char if char.is_absolute() else ROOT / char
        if not Path(str(char) + "_rig.json").exists():
            return {"error": f"no rigged character at {char} (see list_characters)"}
        score = json.loads((run / "dance_score.json").read_text())
        src = (run / "source.txt").read_text().strip() if (run / "source.txt").exists() else score.get("source")
        out = run / (a.get("workdir") or "agent_work")
        self.box = Toolbox(load_keypoints(run / "keypoints.npz"), score, character_kb.rig(char),
                           character_kb.spec(char), out, blend=character_kb.blend(char),
                           source=src if src and Path(src).exists() else None, model=self.model,
                           max_attempts=int(a.get("attempts") or 6), max_looks=4,
                           use_moves=bool(a.get("own_moves", False)))
        self.run_dir, self.char = run, char
        return {"run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
                "character": str(char.relative_to(ROOT)) if char.is_relative_to(ROOT) else str(char),
                "duration_s": score["duration_s"], "units": len(score["units"]), "accents": len(score["accents"]),
                "attempts_dir": str(out), "can_look": bool(self.box.blend and self.box.source),
                "note": "the dance tools are now available: read the dance and the character, then plan"}

    def t_keep_attempt(self, a):
        if self.box is None or not self.box.attempts:
            return {"error": "no attempts yet: open_dance and submit_plan first"}
        atts = self.box.attempts
        att = (next((x for x in atts if x["attempt"] == int(a["attempt"])), None) if a.get("attempt")
               else max(atts, key=lambda x: rank(x["result"])))
        if att is None:
            return {"error": f"no attempt {a['attempt']}; there are {len(atts)}"}
        system = re.sub(r"[^A-Za-z0-9_-]", "", a.get("system") or "agent") or "agent"
        dest = self.run_dir / system
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("plan.json", "curves.json", "checks.json"):
            shutil.copy(Path(att["dir"]) / f, dest / f)
        (dest / "summary.json").write_text(json.dumps({
            "attempt": att["attempt"], "passed": att["result"]["passed"], "character": str(self.char),
            "checks": {k: v.get("passed") for k, v in att["result"]["checks"].items()}}, indent=1))
        rel = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
        char = self.char.relative_to(ROOT) if self.char.is_relative_to(ROOT) else self.char
        run = self.run_dir.relative_to(ROOT) if self.run_dir.is_relative_to(ROOT) else self.run_dir
        return {"kept": att["attempt"], "to": str(rel), "passed": att["result"]["passed"],
                "next": f"render_video(system={system!r})"}


def plan_dance(kp: dict, score: dict, rig: dict, spec: dict, out_dir: str | Path, max_attempts: int = 4,
               max_turns: int = 24, model: str | None = None, blend: str | None = None, source: str | None = None,
               max_looks: int = 4, use_looks: bool = True, use_geometry: bool = True, use_moves: bool = True) -> dict:
    """The same agent with only the dance tools, on a dance that is already perceived and a character that is
    already rigged: the evaluation's setting, so every system starts from the same inputs. use_looks /
    use_geometry / use_moves switch a tool off (ablations). Writes trace.json, summary.json, every attempt and the
    best plan / curves / checks to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = Director(model=model, max_turns=max_turns, dance_only=True, log=lambda *_: None)
    d.box = Toolbox(kp, score, rig, spec, out, blend=blend, source=source, model=model, max_attempts=max_attempts,
                    max_looks=max_looks, use_looks=use_looks, use_geometry=use_geometry, use_moves=use_moves)
    missing = d.box.missing()
    note = f" In this run you do not have {', '.join(missing)}; skip those steps." if missing else ""
    reply = d.send("Make the character dance this dance. Start by reading the dance and the character." + note)
    (out / "trace.json").write_text(json.dumps(d.messages, indent=1))
    attempts, looks = d.box.attempts, d.box.looks
    if not attempts:
        raise RuntimeError("the agent never submitted a valid plan; see trace.json")
    best = max(attempts, key=lambda x: rank(x["result"]))
    for f in ("plan.json", "curves.json", "checks.json"):
        (out / f).write_text((Path(best["dir"]) / f).read_text())
    summary = {"attempts": len(attempts), "best_attempt": best["attempt"], "passed": best["result"]["passed"],
               "history": [{"attempt": x["attempt"], "passed": x["result"]["passed"],
                            "problems": sum(len(u["problems"]) for u in x["result"]["units"])} for x in attempts],
               "looks": [{"attempt": x["attempt"], "frames": x["frames"], "picture": x["picture"]} for x in looks],
               "final_message": reply, "model": model or llm.DEFAULT_MODEL, "capabilities": d.box.use,
               "llm_calls": d.calls + len(looks),
               "cost_usd": round(d.cost + sum(x.get("cost_usd", 0.0) for x in looks), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"agent: {len(attempts)} attempt(s), best = {best['attempt']}, passed = {best['result']['passed']}")
    return summary


def go(request: str, model: str | None = None, trace: Path | None = None) -> str:
    d = Director(model=model)
    reply = d.send(request)
    d.save(trace or ROOT / "runs" / "director" / f"go_{time.strftime('%Y%m%d_%H%M%S')}.json")
    print(f"\n{reply}\n\n({d.calls} LLM calls, ${d.cost:.3f})")
    return reply


def chat(model: str | None = None, first: str | None = None) -> None:
    def ask(q: str) -> str:
        print(f"\nDanceBuddy asks: {q}")
        return input("you> ").strip()

    d = Director(model=model, chat=True, ask=ask)
    trace = ROOT / "runs" / "director" / f"chat_{time.strftime('%Y%m%d_%H%M%S')}.json"
    print("DanceBuddy chat. Tell me which dance and which character, and how it should dance. "
          "Empty line or 'quit' to stop.")
    text = first
    while True:
        if text is None:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        reply = d.send(text)
        d.save(trace)
        print(f"\nDanceBuddy: {reply}\n({d.calls} LLM calls so far, ${d.cost:.3f})")
        text = None
    d.save(trace)
    print(f"conversation saved to {trace}")
