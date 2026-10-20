"""
Tests for per-turn token accounting.

WHY THIS EXISTS
All 17 archived runs record prompt_tokens=0, completion_tokens=0 for DeepSeek,
Kimi and GPT-OSS — every model not served by the local vLLM. The cause: the
pipeline runs stream=True to capture logprobs, and an OpenAI-compatible API
sends no `usage` block at all on a streamed response unless
stream_options.include_usage is requested. AG2 had nothing to account and
reported zero. Cost per sentence was therefore measurable for one model out of
four, which is not enough to say anything about what deliberation costs — and
the zeros were indistinguishable from genuinely free calls.

The fix asks for usage, reads it off the response rather than trusting AG2's
price table, falls back to counting the streamed tokens we already capture, and
records WHICH of those happened. These tests lock in the part that matters most:
an unknown token count never silently becomes a zero.

    pytest src/multi_agent_annotation/tests/test_token_accounting.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
for _p in (_PKG, _PKG / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import multi_agent_annotation_ag2 as M  # noqa: E402

A = M.MultiAgentAnnotator


@pytest.fixture(autouse=True)
def _clean():
    M.reset_logprob_capture()
    yield
    M.reset_logprob_capture()


# ── tier selection ──────────────────────────────────────────────────────────

def test_api_usage_wins_when_present():
    M._usage_records().append({"prompt_tokens": 1200, "completion_tokens": 300})
    got = A._turn_usage({})
    assert got["token_source"] == "api"
    assert (got["prompt_tokens"], got["completion_tokens"]) == (1200, 300)
    assert got["total_tokens"] == 1500


def test_api_usage_sums_across_calls_in_a_turn():
    """A turn with a tool exchange makes several create() calls."""
    M._usage_records().extend([{"prompt_tokens": 100, "completion_tokens": 10},
                               {"prompt_tokens": 250, "completion_tokens": 40}])
    got = A._turn_usage({})
    assert (got["prompt_tokens"], got["completion_tokens"]) == (350, 50)


def test_falls_back_to_ag2_cost():
    cost = {"usage_including_cached_inference":
            {"some-model": {"prompt_tokens": 90, "completion_tokens": 7}}}
    got = A._turn_usage(cost)
    assert got["token_source"] == "ag2"
    assert got["total_tokens"] == 97


def test_falls_back_to_counting_streamed_tokens():
    """The last resort: every generated token passed through the logprob tee."""
    M._logprob_streams().append([M.TokenLP("a", -0.1, []),
                                 M.TokenLP("b", -0.2, []),
                                 M.TokenLP("c", -0.3, [])])
    got = A._turn_usage({})
    assert got["token_source"] == "streamed"
    assert got["completion_tokens"] == 3
    assert got["prompt_tokens"] is None, "a prompt we cannot count is not a free prompt"
    assert got["total_tokens"] is None


def test_all_zero_api_usage_is_treated_as_absent():
    """A gateway reporting 0/0 is reporting nothing, not a free call."""
    M._usage_records().append({"prompt_tokens": 0, "completion_tokens": 0})
    assert A._turn_usage({})["token_source"] != "api"


def test_nothing_available_is_unavailable_not_zero():
    """This is the regression: the archived runs' zeros were unknowns."""
    got = A._turn_usage({})
    assert got["token_source"] == "unavailable"
    assert got["prompt_tokens"] is None
    assert got["completion_tokens"] is None


# ── accumulation ────────────────────────────────────────────────────────────

def test_accumulates_known_values():
    acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    A._add_usage(acc, {"prompt_tokens": 10, "completion_tokens": 2,
                       "total_tokens": 12, "token_source": "api"})
    A._add_usage(acc, {"prompt_tokens": 20, "completion_tokens": 3,
                       "total_tokens": 23, "token_source": "api"})
    assert (acc["prompt_tokens"], acc["completion_tokens"]) == (30, 5)
    assert A._seal_usage(acc)["token_source"] == ["api"]


def test_one_unknown_turn_makes_the_role_total_unknown():
    """Adding zero for an unmeasured turn would produce a plausible-looking and
    wrong per-sentence cost, which is worse than reporting no figure."""
    acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    A._add_usage(acc, {"prompt_tokens": 10, "completion_tokens": 2,
                       "total_tokens": 12, "token_source": "api"})
    A._add_usage(acc, {"prompt_tokens": None, "completion_tokens": 5,
                       "total_tokens": None, "token_source": "streamed"})
    assert acc["prompt_tokens"] is None
    assert acc["completion_tokens"] == 7, "a field that stayed known must still sum"
    assert acc["total_tokens"] is None


def test_unknown_is_sticky():
    acc = {"prompt_tokens": 0}
    A._add_usage(acc, {"prompt_tokens": None, "token_source": "streamed"})
    A._add_usage(acc, {"prompt_tokens": 500, "token_source": "api"})
    assert acc["prompt_tokens"] is None


def test_mixed_sources_are_visible():
    acc = {}
    A._add_usage(acc, {"completion_tokens": 3, "token_source": "api"})
    A._add_usage(acc, {"completion_tokens": 4, "token_source": "streamed"})
    assert A._seal_usage(acc)["token_source"] == ["api", "streamed"]


def test_role_accumulators_fold_into_a_sentence_total():
    """token_source arrives as a string per turn and as a set per role; folding
    roles into the sentence total must accept the set form."""
    roles = []
    for src in ("api", "streamed"):
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        A._add_usage(acc, {"prompt_tokens": 10, "completion_tokens": 1,
                           "total_tokens": 11, "token_source": src})
        roles.append(acc)
    total = {}
    for r in roles:
        A._add_usage(total, r)
    assert total["completion_tokens"] == 2
    assert A._seal_usage(total)["token_source"] == ["api", "streamed"]


def test_seal_is_json_serialisable():
    import json
    acc = {}
    A._add_usage(acc, {"completion_tokens": 1, "token_source": "api"})
    json.dumps(A._seal_usage(acc))  # must not raise on a set


def test_empty_accumulator_seals_to_unavailable():
    assert A._seal_usage({})["token_source"] == ["unavailable"]


# ── the capture accessors ───────────────────────────────────────────────────

def test_captured_usage_is_none_before_anything_arrives():
    assert M.captured_usage() is None


def test_streamed_count_is_none_when_no_tokens_captured():
    assert M.streamed_completion_tokens() is None


def test_reset_clears_usage_as_well_as_logprobs():
    """Usage must reset per turn alongside logprobs, or one turn's tokens are
    attributed to the next."""
    M._usage_records().append({"prompt_tokens": 5, "completion_tokens": 1})
    M._logprob_streams().append([M.TokenLP("x", -0.1, [])])
    M.reset_logprob_capture()
    assert M.captured_usage() is None
    assert M.streamed_completion_tokens() is None


# ── registry guards ─────────────────────────────────────────────────────────

def test_withdrawn_models_fail_fast_with_a_reason():
    """Calling a withdrawn model used to surface as a gateway 403 partway into a
    batch. The registry keeps those entries so archived runs stay readable, so
    the guard has to be here rather than by deleting them."""
    with pytest.raises(ValueError, match="withdrawn"):
        M.build_llm_config("rcp-kimi-2.7")


def test_withdrawn_error_names_a_live_alternative():
    try:
        M.build_llm_config("rcp-deepseek")
    except ValueError as exc:
        assert "rcp-deepseek-4.1" in str(exc)
    else:
        pytest.fail("expected ValueError")


def test_live_rcp_entries_declare_reasoning_explicitly():
    """`reasoning` controls whether the enable_thinking chat-template toggle is
    sent. Leaving it unset on a live entry means guessing at run time, so every
    live RCP entry must state it — each value here was probed, not inferred."""
    from src.resources_updated.model_endpoints import MODEL_ENDPOINTS
    for key, ep in MODEL_ENDPOINTS.items():
        if "rcp.epfl" not in ep.get("base_url", "") or ep.get("withdrawn"):
            continue
        assert "reasoning" in ep, f"{key} does not declare `reasoning`"
        assert isinstance(ep["reasoning"], bool)
