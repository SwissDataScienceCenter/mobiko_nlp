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
import src.multi_agent_annotation.multi_agent_annotation_ag2 as _ann_mod
from src.multi_agent_annotation.multi_agent_annotation_ag2 import (
    MultiAgentAnnotator,
    DeliberationRecord,
    analyze_disagreements,
    load_schema,
    load_seeds,
    load_decision_support,
    load_guideline_from_docx,
    _init_tool_state,
    _DEFAULT_DECISION_SUPPORT,
    _DEFAULT_GUIDELINE,
    _fill_char_offsets,
    schema_lookup,
    guideline_search,
    consistency_check,
    list_entity_types,
)

# ── Demo sentences ─────────────────────────────
DEMO_SENTENCES = [
    "However, the limited information on the effects of overexploitation on the current status and community composition of wildlife hinders effective conservation efforts, including the implementation of targeted patrols to reduce snaring.",
    "Finally, we used a dissimilarity index to assess the level of defaunation, revealing 16% of the community had been lost, with higher levels of defaunation for threatened and larger-sized species.",
    "Our findings provide insights into the status, distribution, and occurrence of the ground-dwelling mammal and bird communities in the Langbian Plateau, and can help stakeholders design more effective conservation strategies to protect existing populations.",
    """[methods]
2.1 Study site
We surveyed four contiguous protected areas in the core forest area of the southern Annamites: Bidoup—Nui Ba National Park, Phuoc Binh National Park, Da Nhim Protection Forest, and Dran Protection Forest (Figure 1).""",
    "Historically, Bidoup—Nui Ba and Phuoc Binh National Park were a part of the Thuong Da Nhim Nature Reserve established in 1986 (Eames, 1995), but in 1992 the two areas were split into two forest units and managed separately (Southern Institute of Ecology, 2017).",
    "Da Nhim and Dran forests were also managed under a single administration authority from 1987 (Eames, 1995), but separated into two protection forests in the late 1990s, in which timber extraction and wildlife exploitation are not completely prohibited (Law on Forestry, 2017).",
    "However, precipitation is unevenly distributed within the study sites due to the rain shadow effect, and this, combined with differences in soil type, contribute to two major habitat types: broadleaf evergreen forest and coniferous forest (Nguyen, 1966; Rundel, 1999).",
    "The eastern and western slopes of Bidoup Nui Ba National Park, and the majority of Phuoc Binh National Park, receive high levels of rainfall and are dominated by evergreen broadleaf forests.",
    "In total, we set up 157 stations spanning all four protected areas and both main habitat types (Table 1, Figure 1).",
    "The vast majority (96.6%) of insects collected from emergence traps were Diptera (flies), while 0.8% were Trichoptera (caddisflies) and Ephemeroptera (mayflies).",
    "While our work confirms prior findings that predator presence drives strong reductions in insect emergence, we find that the effects of predation are significantly weaker in warmer lakes (2% reduction in warmest lakes studied vs. 75% reduction in coldest).",
    "In high-income countries, food insecurity is more commonly characterised by chronic compromises in dietary quality and anxiety associated with accessing food.",
    "The Hainan gibbon, Nomascus hainanus (Thomas), is the world’s rarest ape and one of world’s most endangered mammal species (Bryant et al. 2015; Geissmann and Bleisch 2008; Stone 2011; Zhou et al. 2005)",
]


def load_sentences(input_path: Path) -> list[str]:
    """Load one sentence per non-empty line from a .txt file or directory of .txt files."""
    if input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
        if not files:
            print(f"[ERROR] No .txt files found in directory: {input_path}")
            sys.exit(1)
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)

    sentences: list[str] = []
    for f in files:
        # Split ONLY on "\n" (one sentence per line). Do NOT use str.splitlines(),
        # which also breaks on Unicode line boundaries (\v \f \x85     …)
        # that occur inside PDF-extracted text and would over-count sentences.
        for line in f.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line:
                sentences.append(line)

    if not sentences:
        print(f"[ERROR] No sentences found in: {input_path}")
        sys.exit(1)

    print(f"[OK] Loaded {len(sentences)} sentence(s) from {input_path}")
    return sentences


def run_fix_offsets(output_path: Path) -> None:
    """Load an existing JSONL, fill any null start/end offsets, rewrite in place."""
    if not output_path.exists():
        print(f"[ERROR] File not found: {output_path}")
        sys.exit(1)

    records = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(DeliberationRecord.model_validate_json(line))
            except Exception as e:
                print(f"[WARN] Skipping malformed line: {e}")

    n_fixed = 0
    for rec in records:
        if _fill_char_offsets(rec.sentence, rec.final_entities, rec.final_relations):
            n_fixed += 1

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    print(f"Processed {len(records)} record(s); filled offsets in {n_fixed}.")
    print(f"Written to: {output_path}")


def run_dry(schema_path: Path, seeds_path: Path, guideline_search_backend: str = "embedding") -> None:
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
    _init_tool_state(schema, all_sections, seeds, entity_types_list=SCHEMA_BIODIV_LIST,
                     guideline_search_backend=guideline_search_backend)

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

    # guideline_search tool — verify embedding backend and result quality
    _EMBED_QUERIES = [
        "habitat spatial property",
        "conservation status threatened species",
    ]
    dry_run_ok = True
    for _q in _EMBED_QUERIES:
        _res = json.loads(guideline_search(_q))
        _backend = _res["backend"]
        _hits = _res["results"] or []

        # Check section embeddings were actually computed
        _emb_cache = _ann_mod._GUIDELINE_SECTION_EMBEDDINGS
        _emb_count = len(_emb_cache) if _emb_cache is not None else 0

        if guideline_search_backend == "embedding" and _backend != "embedding":
            print(f"\n[FAIL] guideline_search('{_q}') fell back to '{_backend}' "
                  f"(expected embedding). Suggestion: {_res.get('suggestion')}")
            dry_run_ok = False
        elif all_sections and not _hits:
            print(f"\n[WARN] guideline_search('{_q}') → 0 hits via {_backend} "
                  f"({len(all_sections)} sections loaded, {_emb_count} embedded)")
        else:
            print(f"\n[OK] guideline_search('{_q}') → {len(_hits)} hit(s) via {_backend} "
                  f"({_emb_count} section embeddings cached):")
            for _h in _hits:
                print(f"       • {_h.get('title', '(no title)')}")

    if not dry_run_ok:
        raise RuntimeError("Embedding guideline_search check failed — see [FAIL] lines above.")

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
    precedent_store_path: Path | None = None,
    guideline_search_backend: str = "embedding",
    use_precedent_memory: bool = False,
    resume: bool = False,
    max_retries: int = 2,
    request_timeout: int = 600,
    strict_critic: bool = False,
    guideline_search_mandatory: bool = True,
    tool_choice: str | None = None,
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
        precedent_store_path=precedent_store_path,
        guideline_search_backend=guideline_search_backend,
        use_precedent_memory=use_precedent_memory,
        request_timeout=request_timeout,
        strict_critic=strict_critic,
        guideline_search_mandatory=guideline_search_mandatory,
        tool_choice=tool_choice,
    )

    records = annotator.annotate_batch(sentences, output_path=output_path, resume=resume, max_retries=max_retries)

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
        "--fix-offsets", action="store_true",
        help="Fill null start/end positions in an existing output JSONL and exit. "
             "Requires --output to point to the file to fix.",
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
        "--max-rounds", type=int, default=2,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./data/auto_annotated/datademo_manually_labeled2.jsonl"),
        help="Output JSONL file for full run.",
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("./data/manually_labeled_last"),
        help="Path to a .txt file (one sentence per line) or a directory of such files. "
             "Defaults to data/manually_labeled_last. Pass --use-demo-sentences to ignore "
             "this and use the built-in DEMO_SENTENCES instead.",
    )
    parser.add_argument(
        "--use-demo-sentences", action="store_true",
        help="Use the built-in DEMO_SENTENCES instead of reading from --input.",
    )
    parser.add_argument(
        "--num-sentences", type=int, default=None,
        help="Limit the number of sentences to annotate (default: all loaded).",
    )
    parser.add_argument(
        "--precedent-store", type=Path, default=None,
        help="Path to precedent store JSON file (loaded and updated across sentences).",
    )
    parser.add_argument(
        "--guideline-search-backend", type=str, default="embedding",
        choices=["lexical", "embedding"],
        help="Backend for guideline_search tool (default: embedding).",
    )
    parser.add_argument(
        "--precedent-memory", action="store_true",
        help="Enable cross-sentence precedent memory (lookup_precedent tool). "
             "Default: disabled — not currently used.",
    )
    parser.add_argument(
        "--strict-critic", action="store_true",
        help="Use the strict Critic mode: more aggressive challenge posture, "
             "missing spans checked first, low-confidence items forced into disagreements, "
             "and temperature=0.5 (vs default 0.3). Default Critic mode is unchanged.",
    )
    parser.add_argument(
        "--guideline-search", choices=["mandatory", "optional"], default="optional",
        help="Whether Annotator/Critic MUST call guideline_search before assigning/judging "
             "each entity type ('mandatory') or only when a label is unclear "
             "('optional', default). The full guideline is already injected into the system "
             "prompt, so 'optional' avoids redundant tool calls/latency; use 'mandatory' only "
             "if you specifically want forced per-span guideline retrieval.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip sentences already present in the output file and append new results.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Number of times to retry a sentence that raises an exception before skipping it "
             "(default: 2). Skipped sentences are not written to the output so --resume will "
             "retry them on the next run.",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Per-request HTTP timeout in seconds for all model API calls (default: 600). "
             "Increase this if you see 504 Gateway Timeout errors.",
    )
    parser.add_argument(
        "--tool-choice", choices=["auto", "required", "none"], default="auto",
        help="Controls whether agents may/must emit tool calls. 'auto' (default) = tools "
             "optional, the model decides; 'required' = the model MUST emit a tool call every "
             "turn (diagnostic only — breaks final-JSON turns); 'none' = tools disabled.",
    )
    args = parser.parse_args()

    if args.fix_offsets:
        run_fix_offsets(args.output)
        return

    schema_path = args.schema.resolve()
    seeds_path  = args.seeds.resolve()

    if args.dry_run:
        run_dry(schema_path, seeds_path, guideline_search_backend=args.guideline_search_backend)
    else:
        if args.use_demo_sentences:
            sentences = DEMO_SENTENCES
        else:
            sentences = load_sentences(args.input.resolve())
        if args.num_sentences is not None:
            sentences = sentences[: args.num_sentences]
        run_live(
            sentences=sentences,
            schema_path=schema_path,
            seeds_path=seeds_path,
            annotator_model=args.annotator_model,
            critic_model=args.critic_model,
            adjudicator_model=args.adjudicator_model,
            max_rounds=args.max_rounds,
            output_path=args.output,
            precedent_store_path=args.precedent_store,
            guideline_search_backend=args.guideline_search_backend,
            use_precedent_memory=args.precedent_memory,
            resume=args.resume,
            max_retries=args.max_retries,
            request_timeout=args.timeout,
            strict_critic=args.strict_critic,
            guideline_search_mandatory=(args.guideline_search == "mandatory"),
            tool_choice=args.tool_choice,
        )


if __name__ == "__main__":
    main()
