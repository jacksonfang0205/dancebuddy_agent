"""Character tools: list the rigged characters, import a GLB and build the soft rig (make.py, with the Blender
scripts), make a new one with Tripo (tripo.py). spec.py reads a character's rig and motion spec; card.py describes
its body."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dancebuddy import llm
from dancebuddy.sandbox import ROOT

SPECS = {
    "list_characters": llm.tool_spec(
        "list_characters", "The rigged characters ready to dance (prefixes for open_dance), with their front view.",
        {"type": "object", "properties": {}}),
    "import_character": llm.tool_spec(
        "import_character", "Import a GLB and build the soft rig: becomes assets/NAME_soft. NAME: letters and digits "
        "only. Returns the parts found (arms, hair, body vertices) and the front view.",
        {"type": "object", "properties": {"model": {"type": "string"}, "name": {"type": "string"},
                                          "yaw": {"type": "number"}}, "required": ["model", "name"]}),
}
TRIPO_TOOLS = ("create_character_from_text", "create_character_from_image", "segment_character")


def tripo_specs() -> dict:
    """The Tripo tools, each with a `name` for the character it is for."""
    from dancebuddy.agent.tools.character.make import CHARACTER_SPECS
    out = {}
    for name in TRIPO_TOOLS:
        s = json.loads(json.dumps(CHARACTER_SPECS[name]))
        s["function"]["parameters"]["properties"]["name"] = {"type": "string", "description": "the character's name"}
        s["function"]["parameters"].setdefault("required", []).append("name")
        out[name] = s
    return out


class Characters:
    """One session's character work (Tripo jobs and imports), by character name."""

    def __init__(self):
        self.makers = {}

    def maker(self, name: str):
        from dancebuddy.agent.tools.character.make import CharacterTools
        name = re.sub(r"[^A-Za-z0-9]", "", name) or "character"
        if name not in self.makers:
            self.makers[name] = CharacterTools(name, ROOT / "runs" / "characters" / name)
        return self.makers[name]

    def list_characters(self, a: dict) -> dict:
        out = []
        for rig in sorted((ROOT / "assets").glob("*_rig.json")):
            prefix = Path(str(rig)[: -len("_rig.json")])
            if not Path(str(prefix) + ".blend").exists() or "_soft" not in prefix.name:
                continue
            views = ROOT / "assets" / f"{prefix.name.split('_')[0]}_views" / "front.png"
            if not views.exists():
                views = ROOT / "assets" / f"{prefix.name}_views" / "front.png"
            out.append({"character": str(prefix.relative_to(ROOT)), "parts": json.loads(rig.read_text()).get("parts"),
                        "front_view": str(views.relative_to(ROOT)) if views.exists() else None})
        return {"characters": out, "default": "assets/plush_soft"}

    def import_character(self, a: dict) -> dict:
        from dancebuddy.agent.tools.workspace import resolve
        name = re.sub(r"[^A-Za-z0-9]", "", a["name"])
        if not name:
            return {"error": "the name needs letters or digits"}
        args = {"model": str(resolve(a["model"]))}
        if a.get("yaw"):
            args["yaw"] = a["yaw"]
        t0 = time.time()
        res = json.loads(self.maker(name).call("import_and_rig_soft", args))
        res["seconds"] = round(time.time() - t0, 1)
        if "views" in res:
            res["front_view"] = res["views"] + "/front.png"
        return res

    def tripo(self, tool: str, a: dict) -> dict:
        a = dict(a)
        return json.loads(self.maker(a.pop("name")).call(tool, a))
