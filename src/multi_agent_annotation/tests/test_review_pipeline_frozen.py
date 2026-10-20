"""
The review pipeline must stay runnable, unchanged.

WHY. Every archived run — all 17 in data/auto_annotated_clean and
data/auto_annotated_clean_multi_run — was produced by the review pipeline:
Annotator -> Critic reviews -> Adjudicator. Those runs are also on models the
provider has since withdrawn (Kimi-K2.7-Code and DeepSeek-V4-Flash-0731 both
403 now), so the only way to compare the review design against the blind-pair
design on equal footing is to RE-RUN the review design on current models. That
is impossible if adding the new design quietly altered the old one.

So the blind-pair work is additive by rule: new prompts are siblings, never
rewrites. In particular _adjudicator_system_msg is NOT modified for the
blind-pair input shape — that gets its own function — because the review
pipeline's Adjudicator prompt is written around a Critic review with
"disagreements" and "missing_annotations" lists, and editing it in place would
silently change what the archived design does.

These hashes were taken from the prompts as committed and verified identical to
`git show HEAD:...prompts.py` at the time the blind-pair work began.

IF A HASH FAILS HERE, a review-path prompt changed. That is a result-affecting
change, not a cleanup: it makes new review-mode runs incomparable to the
archived ones. Decide deliberately and re-pin with a note saying why — do not
update the constant to make the test pass.

    pytest src/multi_agent_annotation/tests/test_review_pipeline_frozen.py -v
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
for _p in (_PKG, _PKG.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import prompts as P  # noqa: E402

RT = ["HAS_PROPERTY", "HAS_ROLE", "LOCATED_IN"]
GL, ES, RS = "GUIDELINE", "ENTITY SCHEMA", {}

GOLDEN = {
    "annotator_pipeline_mandatory": "d4c04ce36b8fb6da",
    "annotator_pipeline_optional":  "482ad22696f41d5f",
    "critic":                       "a0d338232a5e6ea9",
    "critic_strict":                "a23dc50cfb7b1542",
    "adjudicator":                  "254619eb0f5e298a",
    "annotator_coldstart":          "b15e1adf21fe6e8a",
    "critic_coldstart":             "a68c28701cc59b34",
    "adjudicator_coldstart":        "3cd72f239a5cb2de",
}

RENDER = {
    "annotator_pipeline_mandatory": lambda: P._annotator_system_msg(
        GL, ES, RS, guideline_search_mandatory=True, relation_types=RT),
    "annotator_pipeline_optional": lambda: P._annotator_system_msg(
        GL, ES, RS, guideline_search_mandatory=False, relation_types=RT),
    "critic": lambda: P._critic_system_msg(GL, ES, RS, relation_types=RT),
    "critic_strict": lambda: P._critic_system_msg_strict(GL, ES, RS, relation_types=RT),
    "adjudicator": lambda: P._adjudicator_system_msg(GL, ES, RS, relation_types=RT),
    "annotator_coldstart": lambda: P._annotator_system_msg_coldstart(
        GL, ES, RS, relation_types=RT),
    "critic_coldstart": lambda: P._critic_system_msg_coldstart(
        GL, ES, RS, relation_types=RT),
    "adjudicator_coldstart": lambda: P._adjudicator_system_msg_coldstart(
        GL, ES, RS, relation_types=RT),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_review_path_prompt_is_unchanged(name):
    got = hashlib.sha256(RENDER[name]().encode()).hexdigest()[:16]
    assert got == GOLDEN[name], (
        f"{name} changed ({GOLDEN[name]} -> {got}). The review pipeline is the "
        f"design all 17 archived runs used and the baseline the blind-pair "
        f"design is compared against; changing its prompt makes new review-mode "
        f"runs incomparable to those. Re-pin deliberately, with a reason.")


def test_every_review_prompt_is_covered():
    """A prompt function added to the review path without a hash here would be
    unprotected, which is how the freeze quietly stops being a freeze."""
    assert set(GOLDEN) == set(RENDER)
    expected = {
        "_annotator_system_msg", "_critic_system_msg", "_critic_system_msg_strict",
        "_adjudicator_system_msg", "_annotator_system_msg_coldstart",
        "_critic_system_msg_coldstart", "_adjudicator_system_msg_coldstart",
        # BLIND-PAIR ARM — exercised by test_blind_pair_prompts and
        # test_blind_adjudicator, deliberately NOT frozen: these are the new
        # design and are expected to change while it is being developed. They
        # are listed so the coverage check stays meaningful; adding a
        # REVIEW-path prompt without a golden hash still fails.
        "_annotator_system_msg_solo",
        "_adjudicator_system_msg_blind",
    }
    actual = {n for n in dir(P)
              if n.startswith("_") and "system_msg" in n and callable(getattr(P, n))}
    assert actual == expected, (
        f"prompt functions changed: added {actual - expected}, "
        f"removed {expected - actual}. A new review-path prompt needs a golden "
        f"hash above.")


# ── the review pipeline's deference wording is part of what is frozen ───────

def test_review_revision_message_is_unchanged():
    """_build_annotator_revision_msg is what tells the Annotator the Critic "is
    usually right" and that conceding "should be your default". That wording is
    a property of the design being measured, so it is frozen too — the
    blind-pair contrast is meaningless if the baseline drifts."""
    import multi_agent_annotation_ag2 as M
    ann = ('{"entities": [], "relations": [], "uncertain_cases": [], '
           '"reasoning": "ok"}')
    msg = M.MultiAgentAnnotator._build_annotator_revision_msg(
        "A sentence.", ann, '{"disagreements": []}', None)
    got = hashlib.sha256(msg.encode()).hexdigest()[:16]
    assert got == "e96a86696282160c", f"review revision message changed -> {got}"
