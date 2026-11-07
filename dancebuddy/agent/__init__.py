"""The agent: an LLM in a tool-calling loop.

  director.py    the agent: go (one request) and chat (a conversation); it decides which tool to call next
  prompts.py     its instructions: how to run the whole job, and how to plan a dance
  llm.py         OpenRouter client with a response cache and a cost ledger
  dance_only.py  the same agent with only the dance tools, for the evaluation and its ablations
"""
