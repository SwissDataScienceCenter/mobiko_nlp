"""
Demo / runner script for the AG2-based multi-agent annotation system.

Default resource paths are relative to this file so it works from any CWD.

Dry-run (no LLM needed):
    cd src/multi_agent_annotation
    python demo_ag2.py --dry-run

Live run (requires model endpoint):
    python demo_ag2.py \\
        --annotator-model qwen3-32B-vllm \\
        --critic-model qwen3-32B-vllm \\
        --adjudicator-model qwen3-32B-vllm \\
        --output demo_ag2_results.jsonl \\
        --num-sentences 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

# ── Ensure repo root is on sys.path so src.* imports work ────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print(f"Repo   root: {_REPO_ROOT}")

if load_dotenv is not None:
    load_dotenv(os.getenv("MOBIKO_ENV_FILE") or _REPO_ROOT / ".env", override=False)

from src.resources_updated.entity_schema import SCHEMA_BIODIV_SHORT, SCHEMA_BIODIV_LIST
from src.multi_agent_annotation.multi_agent_annotation_ag2 import (
    MultiAgentAnnotator,
    analyze_disagreements,
    load_schema,
    load_seeds,
    load_decision_support,
    load_guideline_from_docx,
    _init_tool_state,
    _DEFAULT_DECISION_SUPPORT,
    _DEFAULT_GUIDELINE,
    schema_lookup,
    guideline_search,
    consistency_check,
    list_entity_types,
)

# ── Demo sentences ─────────────────────────────
DEMO_SENTENCES = [
    "While our work confirms prior findings that predator presence drives strong reductions in insect emergence, we find that the effects of predation are significantly weaker in warmer lakes (2% reduction in warmest lakes studied vs. 75% reduction in coldest)."
    # "In high-income countries, food insecurity is more commonly characterised by chronic compromises in dietary quality and anxiety associated with accessing food.",
    # "Accordingly, the species might have niche segregation, as they are species specific, showing annual and inter-annual variability in total consumption of the different prey species.",
    # "The Hainan gibbon, Nomascus hainanus (Thomas), is the world’s rarest ape and one of world’s most endangered mammal species (Bryant et al. 2015; Geissmann and Bleisch 2008; Stone 2011; Zhou et al. 2005)",
]


def run_dry(schema_path: Path, seeds_path: Path) -> None:
    """Load all resources and run tool tests without making any LLM calls."""
    print("=" * 60)
    print("  DRY-RUN MODE")
    print("=" * 60)

    # Load schema
    schema = load_schema(schema_path)
    print(f"\n[OK] Schema loaded: {len(schema)} relations → {list(schema.keys())}")

    # Load seeds
    seeds = load_seeds(seeds_path)
    total_seeds = sum(len(v) for v in seeds.values())
    print(f"[OK] Seeds loaded: {total_seeds} examples across {len(seeds)} relations")

    # Load guideline documents
    decision_support_sections = (
        load_decision_support(_DEFAULT_DECISION_SUPPORT)
        if _DEFAULT_DECISION_SUPPORT.exists()
        else []
    )
    guidance_sections = (
        load_guideline_from_docx(_DEFAULT_GUIDELINE)
        if _DEFAULT_GUIDELINE.exists()
        else []
    )
    print(f"[OK] Decision support: {len(decision_support_sections)} sections "
          f"(from {'file' if _DEFAULT_DECISION_SUPPORT.exists() else 'fallback'})")
    print(f"[OK] MoBiKo v2 guidance: {len(guidance_sections)} sections "
          f"(from {'file' if _DEFAULT_GUIDELINE.exists() else 'fallback'})")

    all_sections = decision_support_sections + guidance_sections
    _init_tool_state(schema, all_sections, seeds, entity_types_list=SCHEMA_BIODIV_LIST)

    # list_entity_types tool
    types_json = list_entity_types()
    types = json.loads(types_json)
    print(f"\n[OK] list_entity_types() → {len(types)} types:")
    for t in types:
        print(f"       {t}")

    # schema_lookup tool
    result = json.loads(schema_lookup("BIOTIC ENTITY", "SPATIAL ENTITY"))
    print(f"\n[OK] schema_lookup(BIOTIC ENTITY, SPATIAL ENTITY):")
    print(f"     forward relations: {result['valid_relations_forward']}")

    # guideline_search tool
    search_result = json.loads(guideline_search("habitat spatial property"))
    hits = search_result["results"]
    print(
        f"\n[OK] guideline_search('habitat spatial property') → "
        f"{len(hits)} section(s) matched via {search_result['backend']}"
    )

    # consistency_check tool
    consistency_query = "species of vascular plants"
    matches = json.loads(consistency_check(consistency_query))
    print(f"\n[OK] consistency_check('{consistency_query}') → {len(matches)} match(es)")

    # Entity schema string
    print(f"\n[OK] SCHEMA_BIODIV_SHORT ({len(SCHEMA_BIODIV_SHORT)} chars) — first 200 chars:")
    print("     " + SCHEMA_BIODIV_SHORT[:200].replace("\n", "\n     "))

    print("\n[OK] All dry-run checks passed.")


def run_live(
    sentences: list[str],
    schema_path: Path,
    seeds_path: Path,
    annotator_model: str,
    critic_model: str,
    adjudicator_model: str,
    max_rounds: int,
    output_path: Path,
) -> None:
    annotator = MultiAgentAnnotator(
        annotator_model=annotator_model,
        critic_model=critic_model,
        adjudicator_model=adjudicator_model,
        schema_path=schema_path,
        # Defaults point to docs in src/multi_agent_annotation/ — no explicit paths needed
        seeds_path=seeds_path,
        max_rounds=max_rounds,
        entity_schema_str=SCHEMA_BIODIV_SHORT,
        entity_types_list=SCHEMA_BIODIV_LIST,
    )

    output_path.write_text("")  # clear output file
    records = annotator.annotate_batch(sentences, output_path=output_path)

    stats = analyze_disagreements(records)
    print(f"\n{'=' * 60}")
    print("  BATCH ANALYSIS")
    print(f"{'=' * 60}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nResults written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo runner for AG2-based multi-agent biodiversity annotation."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load resources and run tool tests without LLM calls.",
    )
    parser.add_argument(
        "--schema", type=Path,
        help="Path to relation schema .py or .json file.",
    )
    parser.add_argument(
        "--seeds", type=Path,
        help="Path to seed examples .py or .json file.",
    )
    parser.add_argument(
        "--annotator-model", type=str, default="qwen3-35B-vllm",
    )
    parser.add_argument(
        "--critic-model", type=str, default="qwen3-35B-vllm",
    )
    parser.add_argument(
        "--adjudicator-model", type=str,
        default=os.getenv("ADJUDICATOR_MODEL", "qwen3-35B-vllm"),
        help="Model key for the Adjudicator (recommend 'gpt4o' for stronger reasoning; "
             "override with ADJUDICATOR_MODEL env var).",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=1,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./data/auto_annotated/datademo_ag2_results.jsonl"),
        help="Output JSONL file for full run.",
    )
    parser.add_argument(
        "--num-sentences", type=int, default=len(DEMO_SENTENCES),
        help="Number of demo sentences to annotate (default: all 5).",
    )
    args = parser.parse_args()

    schema_path = args.schema.resolve()
    seeds_path  = args.seeds.resolve()

    if args.dry_run:
        run_dry(schema_path, seeds_path)
    else:
        sentences = DEMO_SENTENCES[: args.num_sentences]
        run_live(
            sentences=sentences,
            schema_path=schema_path,
            seeds_path=seeds_path,
            annotator_model=args.annotator_model,
            critic_model=args.critic_model,
            adjudicator_model=args.adjudicator_model,
            max_rounds=args.max_rounds,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
