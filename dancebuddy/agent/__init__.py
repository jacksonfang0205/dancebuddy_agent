"""The agent: one LLM in a tool-calling loop (director.py).

  director.py    the loop: go (one request), chat (a conversation), plan_dance (only the dance tools, on a dance
                 that is already perceived: how the evaluation runs it, and the ablations switch tools off)
  prompts.py     its instructions: how to run the whole job, and how to plan a dance
  llm.py         OpenRouter client with a response cache and a cost ledger
"""
