# llm/client.py
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading
from openai import OpenAI
import regex as re
import os

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def remove_thinking_blocks(content: str) -> str:
    """Strip <think>...</think> and unwrap ```json fences."""
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
            base_url="https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
            api_key="EMPTY",
            model_name="Qwen/Qwen3-4B-Instruct-2507"
        ),
        "qwen3-32B": ModelConfig(
            base_url="https://openwebui-runai-codev-llm.inference.compute.datascience.ch/api",
            api_key=None,  # Will use OPEN_WEB_UI_API_KEY env var
            model_name="Qwen/Qwen3-32B-AWQ"
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
        "qwen3-32B-vllm": ModelConfig(
            base_url="https://vllm-gateway-runai-codev-llm.inference.compute.datascience.ch/v1",
            api_key=None,  # read from env
            model_name="Qwen/Qwen3-32B-AWQ"  # use the exact id your gateway serves
        ),
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
        content = response.choices[0].message.content
        if self.model_type in ("qwen3-32B", "gpt4o", "qwen3-32B-vllm"):
            content = remove_thinking_blocks(content)
        return content

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
