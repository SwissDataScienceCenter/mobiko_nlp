# entity_extraction/llm/prompts.py
"""
System/user prompts for the LLM extraction strategies.
Wraps or re-exports your existing prompt constants.
"""

from prompts import (
    DEFAULT_SYSTEM_PROMPT_NEW_2,
    NO_CHUNK_CANDIDATE_SYSTEM_PROMPT_2,
    SYSTEM_PROMPT_FEW_SHOT_2,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT_NEW_2",
    "NO_CHUNK_CANDIDATE_SYSTEM_PROMPT_2",
    "SYSTEM_PROMPT_FEW_SHOT_2",
]
