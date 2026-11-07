"""The terminal sandbox: run one shell command the agent wrote, in the project folder.

check_command refuses commands that delete files, publish, install packages or reach outside the machine, before
anything runs. run_terminal runs the rest with the project's Python first on PATH, a timeout, and the end of the
output returned (long logs are cut from the front). This is a guard against mistakes, not a security boundary: the
command runs as the user.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED = [r"\brm\s+-[a-z]*r", r"\brm\s+-[a-z]*f", r"\bsudo\b", r"\bgit\s+(push|reset|clean|checkout|rebase)",
           r"\b(curl|wget|scp|ssh)\b", r"\b(pip|uv)\s+(install|add|remove)\b", r"\bbrew\b", r">\s*/dev/sd",
           r"\bmkfs\b", r"\bchmod\s+-R\b", r"\bkill(all)?\b", r"\.env\b"]


def check_command(cmd: str) -> str | None:
    """None if the command may run, else why not."""
    for pat in BLOCKED:
        if re.search(pat, cmd):
            return f"refused: commands matching {pat!r} are not allowed here"
    return None


def tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else "...\n" + text[-limit:]


def run_terminal(cmd: str, timeout_s: int = 1800, cwd: Path = ROOT) -> dict:
    """{"exit_code", "seconds", "output"} or {"error"} when the command is refused."""
    cmd = cmd.strip()
    why = check_command(cmd)
    if why:
        return {"error": why}
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s,
                           executable=shutil.which("bash") or None)
        code, out = p.returncode, (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
    except subprocess.TimeoutExpired as e:
        code, out = -1, f"timed out after {e.timeout} s\n" + str(e.stdout or "")[-2000:]
    return {"exit_code": code, "seconds": round(time.time() - t0, 1), "output": tail(out)}
