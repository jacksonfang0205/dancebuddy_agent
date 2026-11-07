"""Workspace tools: find files, run a shell command (through the sandbox), run a Python snippet."""

from __future__ import annotations

import shlex
from pathlib import Path

from dancebuddy import llm
from dancebuddy.sandbox import ROOT, run_terminal

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
MODEL_EXT = {".glb", ".gltf", ".fbx", ".obj"}

SPECS = {
    "find_files": llm.tool_spec(
        "find_files", "List files under a folder whose names match a glob (e.g. '*.mp4'), newest first, with size "
        "and duration for videos. kind='video' or 'model' lists all videos / 3D models.",
        {"type": "object", "properties": {"folder": {"type": "string", "description": "default: the project"},
                                          "pattern": {"type": "string"},
                                          "kind": {"type": "string", "enum": ["video", "model", "any"]},
                                          "recursive": {"type": "boolean"}}}),
    "run_shell": llm.tool_spec(
        "run_shell", "Run a shell command in the project folder (bash, the project's python first on PATH). "
        "Returns the exit code and the end of the output. Destructive commands are refused.",
        {"type": "object", "properties": {"command": {"type": "string"},
                                          "timeout_s": {"type": "integer", "description": "default 1800"}},
         "required": ["command"]}),
    "run_python": llm.tool_spec(
        "run_python", "Run a Python snippet with the project's Python (numpy, json, the dancebuddy package). Print "
        "what you want back.", {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}),
}


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def video_seconds(p: Path) -> float | None:
    try:
        import cv2
        cap = cv2.VideoCapture(str(p))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return round(n / fps, 1) if fps else None
    except Exception:  # noqa: BLE001
        return None


def find_files(a: dict) -> dict:
    folder = resolve(a.get("folder") or str(ROOT))
    if not folder.is_dir():
        return {"error": f"no folder {folder}"}
    kind, pattern = a.get("kind", "any"), a.get("pattern") or "*"
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
            row["seconds"] = video_seconds(p)
        out.append(row)
    return {"folder": str(folder), "found": len(files), "files": out}


def run_shell(a: dict) -> dict:
    return run_terminal(a["command"], int(a.get("timeout_s") or 1800))


def run_python(a: dict, n: int = 0) -> dict:
    tmp = ROOT / ".cache" / "snippets"
    tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / f"snippet_{n}.py"
    f.write_text(a["code"])
    return run_terminal(f"python {shlex.quote(str(f))}", 600)
