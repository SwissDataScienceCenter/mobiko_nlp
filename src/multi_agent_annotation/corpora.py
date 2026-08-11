"""Corpus presets: every resource that must change together to switch corpus.

WHY A PRESET RATHER THAN INDIVIDUAL FLAGS
Each field here has a MoBiKo default that applies SILENTLY when omitted:
entity_types_list falls back to the 16 biodiversity types, guideline_path=None
resolves to the MoBiKo guideline rather than to no guideline, and the prompt module
defaults to the biodiversity templates. Set them one at a time and a single
forgotten flag yields a run that looks completely normal while annotating newswire
against a biodiversity schema with biodiversity prompts. Selecting a named corpus
sets all of them at once.

WHY IT LIVES HERE
Both entry points need it — demo_ag2 (full pipeline: annotator, critic,
adjudicator) and scripts/xrun/selfconsistency_sample.py (annotator only, K times).
Defining it in one of them and copying to the other is how driver.py and harvest.py
drifted apart until their run lists disagreed. One definition, two importers.
"""
from __future__ import annotations

from pathlib import Path

from src.resources_updated.entity_schema import (
    SCHEMA_BIODIV_SHORT, SCHEMA_BIODIV_LIST)
from src.resources_updated.entity_schema_conll import (
    SCHEMA_CONLL_SHORT, SCHEMA_CONLL_LIST)

_REPO = Path(__file__).resolve().parent.parent.parent

CORPORA = {
    "mobiko": {
        "entity_schema": SCHEMA_BIODIV_SHORT,
        "entity_types": SCHEMA_BIODIV_LIST,
        "guideline": None,            # None -> the pipeline's MoBiKo default
        "decision_support": True,
        # entity_schema.py holds the LABEL list; load_schema wants the RELATION
        # schema, which is relation_schema_new.py (7 relations — the vocabulary
        # the scored runs actually use).
        "schema": _REPO / "src/resources_updated/relation_schema_new.py",
        "seeds": _REPO / "src/resources_updated/manual_seeds_filled.py",
        "prompt_set": "mobiko",
    },
    "conll": {
        "entity_schema": SCHEMA_CONLL_SHORT,
        "entity_types": SCHEMA_CONLL_LIST,
        "guideline": _REPO / "src/multi_agent_annotation/CoNLL_label_guidance.md",
        # MoBiKo's Decision_support.csv is biodiversity-specific; there is no
        # CoNLL equivalent, and the annotator treats absence as an empty table.
        "decision_support": False,
        # CoNLL-2003 annotates no relations. None (not a path) so schema_lookup
        # reports nothing valid instead of offering biodiversity relations.
        "schema": None,
        "seeds": None,
        # Parallel prompt templates: newswire framing, no relations, CoNLL
        # boundary conventions. See prompts_conll.
        "prompt_set": "conll",
    },
}


def resolve(corpus: str, *, schema=None, seeds=None, guideline=None,
            no_guideline: bool = False, no_decision_support: bool = False) -> dict:
    """Preset for `corpus`, with optional per-flag overrides.

    Returns the kwargs both entry points pass to MultiAgentAnnotator. Overrides are
    applied only when given, so omitting a flag means "use the preset" — except
    no_guideline, which is the ONLY way to express "no guideline at all", since a
    None path means "fall back to the default".
    """
    try:
        preset = CORPORA[corpus]
    except KeyError:
        raise ValueError(
            f"unknown corpus {corpus!r}; expected one of {sorted(CORPORA)}") from None
    schema_path = schema or preset["schema"]
    seeds_path = seeds or preset["seeds"]
    guideline_path = guideline or preset["guideline"]
    return {
        "schema_path": Path(schema_path).resolve() if schema_path else None,
        "seeds_path": Path(seeds_path).resolve() if seeds_path else None,
        "guideline_path": Path(guideline_path).resolve() if guideline_path else None,
        "use_guideline": not no_guideline,
        "use_decision_support": preset["decision_support"] and not no_decision_support,
        "entity_schema_str": preset["entity_schema"],
        "entity_types_list": preset["entity_types"],
        "prompt_set": preset["prompt_set"],
    }


def add_corpus_args(parser) -> None:
    """Attach the corpus flags to an argparse parser, identically in both scripts."""
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="mobiko",
                        help="selects entity schema, guideline, decision support, "
                             "relation schema, seeds and prompt templates together. "
                             "Individual flags below override the preset.")
    parser.add_argument("--guideline", type=Path, default=None,
                        help="guideline .md/.docx; defaults to the corpus preset")
    parser.add_argument("--no-guideline", action="store_true",
                        help="run with NO guideline at all. Needed because omitting "
                             "--guideline means 'use the preset', not 'use none'.")
    parser.add_argument("--no-decision-support", action="store_true",
                        help="run with no decision-support table")
