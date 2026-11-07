"""The agent: one LLM in a tool-calling loop.

  session.py     AgentSession, the loop: each turn the model sees the conversation and the tools offered now, and
                 either calls tools or replies; plan_dance runs the same loop with only the dance tools (the
                 evaluation's setting)
  registry.py    Tool and ToolRegistry: the tools' schemas for the model, and the dispatch
  prompts.py     its instructions: how to run the whole job, and how to plan a dance
  tools/         the tools it can call
"""
