"""Model endpoints, shared by the multi-agent pipeline and entity_extraction.

Extracted from multi_agent_annotation_ag2 so both systems read ONE table. They
previously kept separate lists, which had already drifted: entity_extraction's
CLI offered two model names absent from its own registry, and lacked every RCP
model the multi-agent runs use, so the two pipelines could not be run on the same
model without editing code.

Deliberately dependency-free — importing this must not pull in autogen or any
model client, so entity_extraction can use it standalone.

Each entry: base_url, api_key_env (the environment variable holding the key), and
model (the provider's model id). Optional: `reasoning` (the endpoint understands
the enable_thinking chat-template toggle and emits reasoning_content) and
`withdrawn` (a note that the provider no longer serves it — the entry is kept so
archived runs that stamped this model id stay interpretable; calling it fails).
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
    # ─────────────────────────────────────────────────────────────────────
    # RCP (inference.rcp.epfl.ch)
    #
    # Every entry below was probed live on 2026-10-19 with a single streamed
    # call. All available models serve logprobs at top_logprobs=20 and report
    # usage when stream_options.include_usage is requested, so both RQ1 and RQ2
    # are measurable on any of them. `reasoning` records whether the stream
    # actually carried reasoning_content, not a guess from the model name.
    #
    # BUDGET WARNING for every reasoning model here: the chain-of-thought is
    # spent from the same completion budget as the answer, so a small max_tokens
    # truncates before any content (or any tool call) is emitted. A 64-token
    # probe of Qwen3.6 produced no tool call for exactly this reason and looked
    # like a missing capability; with room it calls tools normally.
    # ─────────────────────────────────────────────────────────────────────
    "rcp-qwen3.6-35B": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "reasoning": True,
    },
    # Large Qwen, same family as the 35B above — the pair isolates scale from
    # model family, which no single-model comparison can.
    "rcp-qwen3.5-397B": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "Qwen/Qwen3.5-397B-A17B",
        "reasoning": True,
    },
    # NOTE a behavioural change from the archived DeepSeek runs: V4-Flash-0731
    # was NOT a reasoning model, V4.1-Flash-1017 is. The DeepSeek arm therefore
    # will not be a drop-in continuation of those runs. Use the -1017 pin; the
    # unsuffixed deepseek-ai/DeepSeek-V4.1-Flash is flagged EOL upstream.
    "rcp-deepseek-4.1": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "reasoning": True,
    },
    "rcp-glm-5.3": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "zai-org/GLM-5.3-Flash",
        "reasoning": True,
    },
    # Apertus pair: the other within-family scale contrast, and the only
    # non-reasoning models in the set, which makes them the clean case for
    # separating "uncertainty is unreadable" from "the logprobs are dominated by
    # chain-of-thought tokens".
    "rcp-apertus-70B": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "swiss-ai/Apertus-v1.5-70B",
        "reasoning": False,
    },
    "rcp-apertus-8B": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "swiss-ai/Apertus-v1.5-8B",
        "reasoning": False,
    },
    # Despite carrying no "tools" label in the RCP model table, this one does
    # honour tool calls — the table's labels are not a reliable capability
    # guide, which is why the entries here are probed rather than inferred.
    "rcp-mistral-small-24B": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "reasoning": False,
    },
    # EOL upstream, and DROPPED from the rebuilt grid (see grid_config.MODELS).
    # Live as of the probe, and the only model null on every deliberation signal
    # measured — challenge->false-positive OR 0.82/1.18, detection disagreement
    # 0.84/1.08, error-type discrimination 0.78/0.94, all n.s. Its three archived
    # runs stay in the paper as that contrast; kept here so they stay readable.
    "rcp-gpt-oss": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "openai/gpt-oss-120b",
        "reasoning": True,
    },

    # ── PROBABLE ALIASES, not distinct models ──
    # epfl-aiaas/large and /small price identically to GLM-5.3-Flash and
    # DeepSeek-V4.1-Flash respectively, AND return byte-identical prompt token
    # counts for the same message (16 and 34), which is a tokeniser fingerprint.
    # Treat them as routes to those two models: including one alongside its
    # twin would fake a replicate and shrink the standard errors.
    "rcp-aiaas-large": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "epfl-aiaas/large",
        "reasoning": True,
    },
    "rcp-aiaas-small": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "epfl-aiaas/small",
        "reasoning": True,
    },

    # ── WITHDRAWN: 403 key_model_access_denied as of 2026-10-19 ──
    # Kept, not deleted. Every run in data/auto_annotated_clean and
    # data/auto_annotated_clean_multi_run stamps its resolved model id into
    # run_meta, and those records cannot be interpreted without the entry that
    # produced them. Invoking either now raises PermissionDeniedError.
    #
    # rcp-kimi-2.7 produced: multi_pass_rcp-kimi-2.7_{1,2},
    #   single_pass_kimi27_no_deliberation{,_1}, POSTFIX kimi_{1,2},
    #   and the Critic/Adjudicator half of multi_pass_deepseek_kimi_kimi_1.
    #   It was the CODE variant, which is worth remembering: the model carrying
    #   every positive deliberation signal was code-specialised.
    # rcp-deepseek produced: multi_pass_deepseek_{1,2,3}, both DeepSeek solo
    #   arms, POSTFIX deepseek_{1,2}, and the Annotator half of the mixed run.
    "rcp-kimi-2.7": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "moonshotai/Kimi-K2.7-Code",
        "reasoning": True,
        "withdrawn": "403 since 2026-10; archived runs only",
    },
    "rcp-deepseek": {
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "RCP_API_KEY",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "reasoning": False,
        "withdrawn": "403 since 2026-10; archived runs only",
    },
}
