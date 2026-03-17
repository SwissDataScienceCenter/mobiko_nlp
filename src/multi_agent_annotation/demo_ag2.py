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

# ── Ensure repo root is on sys.path so src.* imports work ────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.resources.entity_schema import SCHEMA_BIODIV_SHORT, SCHEMA_BIODIV_LIST
from src.multi_agent_annotation.multi_agent_annotation_ag2 import (
    MultiAgentAnnotator,
    analyze_disagreements,
    load_schema,
    load_seeds,
    _init_tool_state,
    _ALL_ENTITY_TYPES,
    schema_lookup,
    guideline_search,
    consistency_check,
    list_entity_types,
)





# ── Demo sentences ─────────────────────────────
DEMO_SENTENCES = [
    "In high-income countries, food insecurity is more commonly characterised by chronic compromises in dietary quality and anxiety associated with accessing food.",
    "Accordingly, the species might have niche segregation, as they are species specific, showing annual and inter-annual variability in total consumption of the different prey species.",
    "The Hainan gibbon, Nomascus hainanus (Thomas), is the world’s rarest ape and one of world’s most endangered mammal species (Bryant et al. 2015; Geissmann and Bleisch 2008; Stone 2011; Zhou et al. 2005)",
    "Snow leopards in the Himalayas have declined due to habitat loss and poaching.",
    "Climate warming is causing glacial retreat, reducing water availability for downstream ecosystems.",
    "Alpine plant communities show decreasing species richness at higher elevations.",
    "Overgrazing by livestock leads to soil erosion and loss of native vegetation cover.",
    "The wolf population in Yellowstone has recovered following reintroduction programs.",
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

    # Initialise tool state with canonical entity list
    from src.multi_agent_annotation.multi_agent_annotation_ag2 import _get_embedded_guideline
    sections = _get_embedded_guideline()
    _init_tool_state(schema, sections, seeds, entity_types_list=SCHEMA_BIODIV_LIST)

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
    hits = json.loads(guideline_search("habitat spatial property"))
    print(f"\n[OK] guideline_search('habitat spatial property') → {len(hits)} section(s) matched")

    # consistency_check tool
    matches = json.loads(consistency_check("species"))
    print(f"\n[OK] consistency_check('species') → {len(matches)} match(es)")

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
        guideline_path=None,  # no .docx in repo — falls back to embedded guideline
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
        "--annotator-model", type=str, default="qwen3-32B-vllm",
    )
    parser.add_argument(
        "--critic-model", type=str, default="qwen3-32B-vllm",
    )
    parser.add_argument(
        "--adjudicator-model", type=str, default="qwen3-32B-vllm",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=2,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("demo_ag2_results.jsonl"),
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
