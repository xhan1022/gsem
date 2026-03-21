"""Agents package."""
import os

if os.getenv("GSEM_USE_MCQA_REACT", "0") == "1":
    from .react_agent_mcqa import ReActAgentMCQA as ReActAgent
else:
    from .react_agent import ReActAgent

__all__ = ["ReActAgent"]
