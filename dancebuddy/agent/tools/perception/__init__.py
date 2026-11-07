"""Perception tool, perceive_dance: a dance video becomes pose keypoints (MediaPipe, video.py), a dance score
(beats, accents, move units: beats.py) and continuous signals of the dancer (signals.py), in runs/NAME."""

from __future__ import annotations

import re
import time

from dancebuddy import llm
from dancebuddy.sandbox import ROOT

SPECS = {
    "perceive_dance": llm.tool_spec(
        "perceive_dance", "Run perception on a dance video: MediaPipe pose, then beats, accents, move units and the "
        "dancer's signals, written to runs/NAME (about 1 s per second of video). Returns what it found.",
        {"type": "object", "properties": {"video": {"type": "string"},
                                          "name": {"type": "string", "description": "short run name, e.g. dance2"},
                                          "max_seconds": {"type": "number"}}, "required": ["video", "name"]}),
}


def perceive_dance(a: dict) -> dict:
    from dancebuddy.agent.tools.perception.video import perceive
    from dancebuddy.agent.tools.workspace import resolve
    video = resolve(a["video"])
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
