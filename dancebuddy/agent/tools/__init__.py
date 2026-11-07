"""The agent's tools. Each is a deterministic function; the agent decides which to call, with what, and when.

  workspace.py    find_files, run_shell (through dancebuddy/sandbox.py), run_python
  perception/     perceive_dance: MediaPipe pose -> dance score (beats, accents, move units, signals)
  character/      list_characters, import_character (soft rig), Tripo text/image to 3D (with TRIPO_API_KEY)
  dance/          open_dance; then get_dance_score, get_dance_signals, get_leg_events, get_character_spec,
                  get_plan_format, submit_plan (plan -> motion -> checks), check_arm_geometry, look_at_frames;
                  keep_attempt
  render.py       render_video: Blender render and side-by-side video with FFmpeg

build_tools(session) returns the tools offered on this turn: the dance tools only once a dance is open, ask_user
and view_image from the session itself.
"""

from __future__ import annotations

import os

from dancebuddy.agent.registry import Tool, ToolRegistry


def build_tools(session) -> ToolRegistry:
    from dancebuddy.agent.tools import character, dance, perception, render, workspace

    box = session.box
    dance_tools = ([Tool(s, (lambda n: lambda a: box.call_dict(n, a))(s["function"]["name"])) for s in box.specs()]
                   if box is not None else [])
    if session.dance_only:  # the evaluation's setting: only the dance tools, on a dance that is already open
        return ToolRegistry(dance_tools)

    def python_snippet(a):
        session.snippets += 1
        return workspace.run_python(a, session.snippets)

    def open_dance(a):
        new_box, run, char, res = dance.open_dance(a, session.model)
        if new_box is not None:
            session.box, session.run_dir, session.char = new_box, run, char
        return res

    tools = [Tool(workspace.SPECS["find_files"], workspace.find_files),
             Tool(workspace.SPECS["run_shell"], workspace.run_shell),
             Tool(workspace.SPECS["run_python"], python_snippet),
             Tool(session.VIEW_SPEC, session.view_image),
             Tool(perception.SPECS["perceive_dance"], perception.perceive_dance),
             Tool(character.SPECS["list_characters"], session.characters.list_characters),
             Tool(character.SPECS["import_character"], session.characters.import_character),
             Tool(dance.SESSION_SPECS["open_dance"], open_dance),
             Tool(dance.SESSION_SPECS["keep_attempt"],
                  lambda a: dance.keep_attempt(session.box, session.run_dir, session.char, a)),
             Tool(render.RENDER_SPEC, lambda a: render.render_video(session.run_dir, session.char, a))]
    if session.user.can_answer:
        tools.append(Tool(session.ASK_SPEC, session.ask_user))
    if os.environ.get("TRIPO_API_KEY"):
        tools += [Tool(s, (lambda n: lambda a: session.characters.tripo(n, a))(n))
                  for n, s in character.tripo_specs().items()]
    return ToolRegistry(tools + dance_tools)
