# entity_extraction/llm/prompts.py
"""
System/user prompts for the LLM extraction strategies.
Wraps or re-exports your existing prompt constants.
"""

from prompts_two_output import (
    DEFAULT_SYSTEM_PROMPT_NEW,
    NO_CHUNK_CANDIDATE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_FEW_SHOT,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT_NEW",
    "NO_CHUNK_CANDIDATE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT_FEW_SHOT",
]
