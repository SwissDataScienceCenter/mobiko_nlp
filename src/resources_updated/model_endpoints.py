"""Model endpoints, shared by the multi-agent pipeline and entity_extraction.

Extracted from multi_agent_annotation_ag2 so both systems read ONE table. They
previously kept separate lists, which had already drifted: entity_extraction's
CLI offered two model names absent from its own registry, and lacked every RCP
model the multi-agent runs use, so the two pipelines could not be run on the same
model without editing code.

Deliberately dependency-free — importing this must not pull in autogen or any
model client, so entity_extraction can use it standalone.

Each entry: base_url, api_key_env (the environment variable holding the key), and
model (the provider's model id).
"""

MODEL_ENDPOINTS = {
    "qwen3-4B": {
        "base_url": "https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
    },
    "qwen3-32B": {
        "base_url": "https://openwebui-runai-codev-llm.inference.compute.datascience.ch/api",
        "api_key_env": "OPEN_WEB_UI_API_KEY",
        "model": "Qwen/Qwen3-32B-AWQ",
    },
    "qwen3-32B-vllm": {
        "base_url": "https://vllm-gateway-runai-codev-llm.inference.compute.datascience.ch/v1",
        "api_key_env": "OPEN_WEB_UI_API_KEY",
        "model": "Qwen/Qwen3-32B-AWQ",
    },
    "gpt4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o",
    },
    "qwen3-35B-vllm": {
        "base_url": "https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1",
        "api_key_env": "OPEN_WEB_UI_API_KEY",
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    },
    "gemma4-26B": {
        "base_url": "https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1",
        "api_key_env": "OPEN_WEB_UI_API_KEY",
        "model": "google/gemma-4-26B-A4B-it",
    },
    "swissai-qwen3.5-27B": {
        "base_url": "https://api.swissai.svc.cscs.ch/v1",
        "api_key_env": "SWISSAI_API_KEY",
        "model": "Qwen/Qwen3.5-27B",
    },
    "swissai-gemma-31B": {
        "base_url": "https://api.swissai.svc.cscs.ch/v1",
        "api_key_env": "SWISSAI_API_KEY",
        "model": "google/gemma-4-31B-it-bdoan",
    },
    "swissai-apertus-70B": {
        "base_url": "https://api.swissai.svc.cscs.ch/v1",
        "api_key_env": "SWISSAI_API_KEY",
        "model": "swiss-ai/Apertus-v1.5-70B",
    },
    "swissai-llama-3.3-70B": {
        "base_url": "https://api.swissai.svc.cscs.ch/v1",
        "api_key_env": "SWISSAI_API_KEY",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "rcp-qwen3.6-35B": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "reasoning": True,
    },
    "rcp-kimi-2.7": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "moonshotai/Kimi-K2.7-Code",
        "reasoning": True,
    },
    "rcp-gpt-oss": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "openai/gpt-oss-120b",
        "reasoning": True,
    },
    "rcp-deepseek": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "reasoning": False,
    }
}
