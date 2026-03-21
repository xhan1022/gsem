"""Prompt provider layer for switching prompt sets without changing pipeline logic.

Usage:
- Default prompt module: ``src.prompts``
- Switch via env var:
    GSEM_PROMPTS_MODULE=src.prompt_sets.mcqa_prompts
"""
import importlib
import os
from types import ModuleType


def _load_prompt_module() -> ModuleType:
    module_path = os.getenv("GSEM_PROMPTS_MODULE", "src.prompts")
    return importlib.import_module(module_path)


_PROMPTS = _load_prompt_module()


# Expose all UPPER_CASE constants to keep import interface stable.
for _name in dir(_PROMPTS):
    if _name.isupper():
        globals()[_name] = getattr(_PROMPTS, _name)


def __getattr__(name: str):
    return getattr(_PROMPTS, name)


__all__ = [name for name in dir(_PROMPTS) if name.isupper()]
