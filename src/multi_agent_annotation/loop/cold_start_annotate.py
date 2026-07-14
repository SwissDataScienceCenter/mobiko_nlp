"""
cold_start_annotate.py — guideline-parameterised annotation runner (RQ-D).

A thin, standalone entry point around the existing ``MultiAgentAnnotator`` that
adds the one knob the cold-start reconstruction loop needs and the stock CLIs
lack: an explicit ``--guideline`` so each iteration can annotate the working set
with a *different* guideline G_i.

It mirrors the faithful configuration used by ``demo_ag2.run_live`` (the same
entity schema string / type list, embedding guideline search, etc.) so the
deliberation logs it writes are directly comparable to the rest of the pipeline.
Nothing in the existing pipeline is modified — this only *imports* it.

The reconstruction loop (``reconstruct_loop.py``) subprocesses this once per
iteration so the annotation pipeline's module-global tool state is reset between
guidelines.

Usage:
  python cold_start_annotate.py \
      --input-txt ./working_set.txt \
      --guideline ./run/guidelines/G0.md \
      --schema ../resources_updated/relation_schema.py \
      --seeds  ../resources_updated/manual_seeds_filled.py \
      --output ./run/iter_00/deliberations.jsonl \
      --annotator-model qwen3-35B-vllm \
      --critic-model    qwen3-35B-vllm \
      --adjudicator-model qwen3-35B-vllm \
      --max-rounds 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv optional — fall back to exported env vars
    load_dotenv = None

_THIS_DIR = Path(__file__).resolve().parent          # …/multi_agent_annotation/loop
_PKG_ROOT = _THIS_DIR.parent                         # …/multi_agent_annotation (shared core)
_SRC = _PKG_ROOT.parent                              # …/src (for resources_updated)
_REPO_ROOT = _SRC.parent                             # repo root
# The pipeline core lives at the package root and the entity schema under src/;
# make both importable before the flat imports below (the loop also sets these
# on PYTHONPATH when it subprocesses this script).
for _p in (_SRC, _PKG_ROOT, _PKG_ROOT / "loop", _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from multi_agent_annotation_ag2 import MultiAgentAnnotator, analyze_disagreements
from resources_updated.entity_schema import SCHEMA_BIODIV_SHORT, SCHEMA_BIODIV_LIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pick up API keys the same way as the rest of the pipeline.
if load_dotenv:
    load_dotenv()
    load_dotenv(_REPO_ROOT / ".env", override=False)  # repo-root .env


def load_sentences_from_jsonl(path: Path) -> List[str]:
    """One sentence per JSONL record (``sentence`` or ``text`` field), in order.

    Exact duplicates are dropped so a sentence that appears twice in the working
    set is not annotated twice (the deliberation logs key confusions by sentence).
    """
    sentences: List[str] = []
    seen: set = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            s = (obj.get("sentence") or obj.get("text") or "").strip()
            if s and s not in seen:
                seen.add(s)
                sentences.append(s)
    return sentences


def load_sentences_from_txt(path: Path) -> List[str]:
    """One non-empty sentence per line (split only on ``\\n``, never splitlines)."""
    sentences: List[str] = []
    seen: set = set()
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            sentences.append(line)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guideline-parameterised multi-agent annotation runner (RQ-D)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-jsonl", type=Path,
                     help="JSONL with a 'sentence' or 'text' field per record.")
    src.add_argument("--input-txt", type=Path,
                     help="Plain text, one sentence per line.")

    parser.add_argument("--guideline", type=Path, default=None,
                        help="Narrative guideline G_i (.md/.docx) for Critic & Adjudicator. "
                             "Omit to use the pipeline default.")
    parser.add_argument("--decision-support", type=Path, default=None,
                        help="Decision table D_i (.csv) for the Annotator. "
                             "Omit to use the pipeline default Decision_support.csv.")
    parser.add_argument("--schema", type=Path, required=True,
                        help="Relation schema (.py/.json).")
    parser.add_argument("--seeds", type=Path, default=None,
                        help="Seed examples (.py/.json).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output deliberation JSONL.")

    parser.add_argument("--annotator-model", type=str, default="qwen3-35B-vllm")
    parser.add_argument("--critic-model", type=str, default="qwen3-35B-vllm")
    parser.add_argument("--adjudicator-model", type=str, default="qwen3-35B-vllm")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--num-sentences", type=int, default=None,
                        help="Limit the number of sentences (default: all).")

    parser.add_argument("--precedent-store", type=Path, default=None,
                        help="Persistent precedent-store JSON (default: fresh in-memory store).")
    parser.add_argument("--precedent-memory", action="store_true",
                        help="Enable the lookup_precedent tool / precedent store "
                             "(default: disabled — not currently used).")
    parser.add_argument("--guideline-search-backend", type=str, default="embedding",
                        choices=["lexical", "embedding"])
    parser.add_argument("--guideline-search", choices=["mandatory", "optional"],
                        default="optional",
                        help="Whether agents MUST call guideline_search before deciding.")
    parser.add_argument("--strict-critic", action="store_true")
    parser.add_argument("--tool-choice", type=str, default=None,
                        choices=["auto", "required", "none"])
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true",
                        help="Skip sentences already present in --output.")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    if args.input_jsonl:
        sentences = load_sentences_from_jsonl(args.input_jsonl.resolve())
    else:
        sentences = load_sentences_from_txt(args.input_txt.resolve())
    if args.num_sentences is not None:
        sentences = sentences[: args.num_sentences]
    if not sentences:
        parser.error("No sentences loaded from the input.")
    logger.info("Loaded %d sentence(s).", len(sentences))

    guideline_path: Optional[Path] = args.guideline.resolve() if args.guideline else None
    decision_support_path: Optional[Path] = (
        args.decision_support.resolve() if args.decision_support else None
    )
    logger.info("Guideline:       %s", guideline_path or "(pipeline default)")
    logger.info("Decision table:  %s", decision_support_path or "(pipeline default)")

    annotator = MultiAgentAnnotator(
        annotator_model=args.annotator_model,
        critic_model=args.critic_model,
        adjudicator_model=args.adjudicator_model,
        schema_path=args.schema.resolve(),
        guideline_path=guideline_path,
        decision_support_path=decision_support_path,
        seeds_path=args.seeds.resolve() if args.seeds else None,
        max_rounds=args.max_rounds,
        # Faithful config (mirrors demo_ag2.run_live).
        entity_schema_str=SCHEMA_BIODIV_SHORT,
        entity_types_list=SCHEMA_BIODIV_LIST,
        precedent_store_path=args.precedent_store,
        guideline_search_backend=args.guideline_search_backend,
        use_precedent_memory=args.precedent_memory,
        request_timeout=args.timeout,
        strict_critic=args.strict_critic,
        guideline_search_mandatory=(args.guideline_search == "mandatory"),
        tool_choice=args.tool_choice,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = annotator.annotate_batch(
        sentences,
        output_path=args.output.resolve(),
        resume=args.resume,
        max_retries=args.max_retries,
    )

    stats = analyze_disagreements(records)
    logger.info("Annotated %d sentence(s) → %s", len(records), args.output)
    logger.info("avg_agreement=%.3f avg_rounds=%.2f",
                stats.get("avg_agreement", 0.0), stats.get("avg_rounds", 0.0))


if __name__ == "__main__":
    main()