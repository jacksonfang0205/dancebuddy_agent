"""The agent's tools. Each tool is a deterministic function; the agent decides which to call and with what.

  workspace       find_files, run_shell, run_python, view_image, ask_user (in agent/director.py)
  perception/     perceive_dance: MediaPipe pose -> dance score (beats, accents, move units, signals)
  character/      list_characters, import_character (soft rig), Tripo text/image to 3D, the character's spec
  dance/          open_dance, then: read the dance and the body, submit_plan (plan -> motion -> checks),
                  check_arm_geometry, look_at_frames, keep_attempt
  render.py       render_video: Blender render + side-by-side video with FFmpeg
"""
