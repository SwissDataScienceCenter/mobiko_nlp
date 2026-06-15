# llm/client.py
from typing import Dict
from dataclasses import dataclass
import threading
from openai import OpenAI
import regex as re
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = os.getenv("MOBIKO_ENV_FILE") or _REPO_ROOT / ".env"
if load_dotenv is not None:
    load_dotenv(_ENV_FILE, override=False)

class EmptyLLMResponseError(RuntimeError):
    """Raised when the model returns empty/blank content for a call.

    We surface this instead of silently passing an empty string downstream so
    the affected sentence is left unprocessed and can be retried via --resume,
    rather than being baked into the output as a (false) "no entities" result.
    """


def remove_thinking_blocks(content: str | None) -> str:
    """Strip <think>...</think> and unwrap ```json fences."""
    if not content:
        return ""
    cleaned = _THINK_RE.sub("", content).strip()
    if cleaned.startswith("```json"):
        start = cleaned.find("```json") + 7
        end = cleaned.rfind("```")
        if end > start:
            cleaned = cleaned[start:end].strip()
    return cleaned


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    base_url: str
    api_key: str
    model_name: str


class ModelRegistry:
    """Registry of available model configurations."""

    CONFIGS: Dict[str, ModelConfig] = {
        "qwen3-4B": ModelConfig(
            base_url="https://qwen3-4b-instruct-runai-mobiko-anisia.inference.compute.datascience.ch/v1",
            api_key="EMPTY",
            model_name="Qwen/Qwen3-4B-Instruct-2507"
        ),
        "medgemma-4b": ModelConfig(
            base_url="http://medgemma-4b-it.runai-mobiko-anisia.inference.compute.datascience.ch",
            api_key="EMPTY",
            model_name="google/medgemma-4b-it"
        ),
        "biomistral-7b-awq": ModelConfig(
            base_url="https://mistral-7b-awq.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
            api_key="EMPTY",
            model_name="BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
        ),
        "gpt4o": ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key=None,  # Will use OPENAI_API_KEY env var
            model_name="gpt-4o"
        ),
        "qwen3-35B-vllm": ModelConfig(
            base_url="https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1",
            api_key=None,
            model_name="Qwen/Qwen3.6-35B-A3B-FP8",
        ),
        "gemma4-26B": ModelConfig(
            base_url="https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1",
            api_key=None,
            model_name="google/gemma-4-26B-A4B-it",
        )
    }

    @classmethod
    def get_config(cls, model_type: str) -> ModelConfig:
        """Get configuration for a model type."""
        if model_type not in cls.CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.CONFIGS.keys())}")
        return cls.CONFIGS[model_type]

    @classmethod
    def register(cls, model_type: str, config: ModelConfig) -> None:
        """Register a new model configuration."""
        cls.CONFIGS[model_type] = config


class LLMClient:
    """Thread-safe LLM client manager."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.config = ModelRegistry.get_config(model_type)
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._query_count = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0

    @property
    def client(self) -> OpenAI:
        """Get or create thread-local OpenAI client.

        Behaves like the old get_openai_client():
        - Resolve API key from config OR environment variables.
        - Throw clear error if no key is provided.
        """
        if not hasattr(self._local, "client"):
            # API key resolution (matches old get_openai_client)
            api_key = (
                self.config.api_key
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("OPEN_WEB_UI_API_KEY")
            )

            if not api_key:
                raise ValueError(
                    f"API key required for model '{self.model_type}'. "
                    "Set OPENAI_API_KEY or OPEN_WEB_UI_API_KEY."
                )

            self._local.client = OpenAI(
                base_url=self.config.base_url,
                api_key=api_key,
            )

        return self._local.client


    @property
    def model_name(self) -> str:
        """Get the model name for API calls."""
        return self.config.model_name

    def call(self, messages: list, temperature: float = 0.0, **kwargs) -> str:
        """Make a single LLM call."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        message = response.choices[0].message
        content = message.content
        if content is None:
            # Some vLLM/Qwen reasoning models return content=None and place the
            # text in reasoning_content, or reply with only a tool call. Fall back
            # to reasoning_content so we don't crash / lose the output.
            content = getattr(message, "reasoning_content", None)
        if self.model_type in ("qwen3-32B", "gpt4o", "qwen3-35B-vllm", "qwen3-32B-vllm"):
            content = remove_thinking_blocks(content)
        else:
            content = content or ""
        # Account for usage before any raise: the API call happened (and may be
        # billed) even when the returned content is empty.
        usage = response.usage
        if usage is not None:
            with self._stats_lock:
                self._query_count += 1
                self._prompt_tokens_total += usage.prompt_tokens or 0
                self._completion_tokens_total += usage.completion_tokens or 0
        if not content.strip():
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            raise EmptyLLMResponseError(
                f"Model '{self.model_name}' returned empty content "
                f"(finish_reason={finish_reason!r}). Sentence left unprocessed; "
                f"rerun with --resume to retry it."
            )
        return content

    def token_stats(self) -> dict:
        """Return token usage statistics accumulated across all calls."""
        with self._stats_lock:
            n = self._query_count
            return {
                "queries": n,
                "prompt_tokens_total": self._prompt_tokens_total,
                "completion_tokens_total": self._completion_tokens_total,
                "prompt_tokens_mean": self._prompt_tokens_total / n if n else 0.0,
                "completion_tokens_mean": self._completion_tokens_total / n if n else 0.0,
            }

    def call_batch(self, message_batches: list, temperature: float = 0.0, **kwargs) -> list:
        """Make multiple LLM calls (not true batching, sequential calls)."""
        results = []
        for messages in message_batches:
            try:
                result = self.call(messages, temperature=temperature, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(f"ERROR: {str(e)}")
        return results


class LLMClientFactory:
    """Factory for creating LLM clients."""

    _instances: Dict[str, LLMClient] = {}
    _lock = threading.Lock()

    @classmethod
    def create(cls, model_type: str) -> LLMClient:
        """Create or retrieve cached LLM client (singleton per model type)."""
        with cls._lock:
            if model_type not in cls._instances:
                cls._instances[model_type] = LLMClient(model_type)
            return cls._instances[model_type]

    @classmethod
    def reset(cls) -> None:
        """Clear all cached clients (useful for testing)."""
        cls._instances.clear()
