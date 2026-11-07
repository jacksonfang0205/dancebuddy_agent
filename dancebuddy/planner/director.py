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

from dancebuddy import llm
from dancebuddy.characters.make import CHARACTER_SPECS, CharacterTools
from dancebuddy.critics.numbers import rank
from dancebuddy.knowledge import character as character_kb
from dancebuddy.planner.loop import SYSTEM as DANCE_SYSTEM
from dancebuddy.tools import SPECS as DANCE_SPECS, Toolbox

ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
MODEL_EXT = {".glb", ".gltf", ".fbx", ".obj"}

# commands the agent may not run: they delete, publish, install or reach outside the project
BLOCKED = [r"\brm\s+-[a-z]*r", r"\brm\s+-[a-z]*f", r"\bsudo\b", r"\bgit\s+(push|reset|clean|checkout|rebase)",
           r"\b(curl|wget|scp|ssh)\b", r"\b(pip|uv)\s+(install|add|remove)\b", r"\bbrew\b", r">\s*/dev/sd",
           r"\bmkfs\b", r"\bchmod\s+-R\b", r"\bkill(all)?\b", r"\.env\b"]

DIRECTOR = """You are DanceBuddy's director. You take a request in plain words and carry out the whole pipeline
yourself, stage by stage, on this computer, until there is a video of the character dancing next to the dancer.

The project is at {root} (the working directory of every command). Python is `python` (the project's venv);
`python -m dancebuddy --help` lists the commands. The stages:
 1. Inputs. Find the dancer's video and the character. Use find_files (videos: .mp4/.mov; models: .glb). Paths the
    user gives may be relative to the project or absolute. Characters already rigged are listed by list_characters
    (prefix like assets/plush_soft); use one of those when the user names it or gives none (default plush_soft).
 2. Perception: `python -m dancebuddy perceive VIDEO runs/NAME` (MediaPipe; about 1 s per second of video).
    Pick a short NAME from the video. Skip it if runs/NAME/dance_score.json exists for the same video
    (runs/NAME/source.txt holds the video path).
 3. Character: a new GLB is imported and soft-rigged with import_character (about a minute). Then view_image its
    front view and check the arms were found; if not, try `yaw` (90 or -90) once. The soft rig suits plush-like
    bodies with blue arms; say so if the model is something else.
 4. Plan: open_dance(run, character), then work as the dance planner below (read, plan, submit, check, look,
    revise). The user's wishes come first: if they ask for something the dancer does not do (keep spinning, just
    bounce, freeze at the end, stay still and only wave), use plan.layers (see get_plan_format) and make sure
    checks.requests passes. Units the user took off the copy do not need to pass the copy checks.
    Do not chase the accent check with big moves: a hit is better as a squash (size up to 1.0) than as a hop, use
    a hop only where the dancer jumps, and keep arms at most 1.2. A smooth copy that misses an accent or two is
    better than a jerky one that passes.
 5. keep_attempt: write the attempt you choose to runs/NAME/<system>. Give each new version its own short system
    name that says what it is (agent_spin, agent_bounce, ...), so earlier results are kept; use the same name only
    when the user asks to replace one. The video is then runs/NAME/side_by_side_<system>[_preview].mp4.
 6. Render and compose with run_shell:
      python -m dancebuddy render runs/NAME --systems SYSTEM --char CHARACTER --preview      (fast, untextured)
      python -m dancebuddy compose runs/NAME --systems SYSTEM --preview
    or the final look without --preview (`--engine eevee` is quicker than the default cycles). Use the preview
    unless the user asks for the final render. Then view_image one or two rendered frames
    (runs/NAME/SYSTEM/preview/frame_0001.png ...) to confirm the character is there and moving.
 7. Reply with what you did, where the video is, what the checks say and anything still wrong.

Rules: use run_python for small calculations or for reading JSON; never delete or overwrite the user's files
outside runs/ and assets/; do not repeat a stage that already succeeded; if a command fails, read its error,
fix the cause and retry once, then report. {chat_rule}

----- the dance planner (stage 4) -----
{dance}"""

CHAT_RULE = ("This is a conversation: when a request is ambiguous in a way that changes the result (which video, "
             "which character), ask_user once; otherwise decide and say what you chose. After a reply the user may "
             "ask for changes: revise the same run (open_dance again only to switch run or character).")
GO_RULE = "Nobody will answer questions in this run: decide yourself and say what you chose."

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
    def __init__(self, model: str | None = None, chat: bool = False, ask=None, log=print, max_turns: int = 60):
        self.model, self.chat, self.ask, self.log, self.max_turns = model, chat, ask, log, max_turns
        self.box: Toolbox | None = None
        self.run_dir: Path | None = None
        self.char: Path | None = None
        self.tripo = bool(os.environ.get("TRIPO_API_KEY"))
        self.makers: dict[str, CharacterTools] = {}
        self.cost, self.calls = 0.0, 0
        self.pending_images: list[str] = []
        self.commands: list[dict] = []
        system = DIRECTOR.format(root=ROOT, chat_rule=CHAT_RULE if chat else GO_RULE, dance=DANCE_SYSTEM)
        self.messages: list[dict] = [{"role": "system", "content": system}]

    # ------------------------------------------------------------------ the loop
    def specs(self) -> list[dict]:
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
            out += [s for s in self.box.specs() if s["function"]["name"] not in CHARACTER_SPECS]
        return out

    def send(self, text: str) -> str:
        """One user request: the agent works until it replies. Returns the reply."""
        self.messages.append({"role": "user", "content": text})
        if self.box is not None:  # a fresh budget of plans and looks for each request
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
            if fn is not None:
                res = fn(args)
            elif name in TRIPO_TOOLS and self.tripo:
                res = json.loads(self._maker(args.pop("name")).call(name, args))
            elif self.box is not None:
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
        from dancebuddy.perception.video import load_keypoints
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
                "render": f"python -m dancebuddy render {run} --systems {system} --char {char} --preview",
                "compose": f"python -m dancebuddy compose {run} --systems {system} --preview",
                "video": f"{run}/side_by_side{'' if system == 'agent' else '_' + system}_preview.mp4"}


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
