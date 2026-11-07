"""DanceBuddy: an agent that makes a plush character dance along with a human dancer.

  agent/     the LLM in a tool-calling loop, its instructions and model access
  tools/     what it can call: workspace, perception, character, dance planning and checks, rendering
  blender/   scripts the tools run inside Blender (rigging, posing, rendering, mesh checks)
"""
