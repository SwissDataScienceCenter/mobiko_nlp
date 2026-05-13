# entity_extraction/entity_pipeline.py
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import random
import numpy as np
import torch

import spacy  # just for validation of model existence
from nltk.tokenize import sent_tokenize

from llm.client import LLMClientFactory
from llm.strategies import (
    call_llm_batch_two_path,
    run_C1_vanilla,
    run_C2_diverse,
    run_C3_critique_revise,
    run_C4_self_consistency,
)
from candidates import process_sentences_batch
from candidates.bioc import (
    load_bioc_index_from_dir,
    load_bioc_sentence_index_from_dir,
    dedupe_bioc_index,
    _collect_allowed_content_ids_from_document,
    _BIOC_EXCLUDED_SENTENCE_FIELDS,
)
from candidates.ner import load_ner
from candidates.chunk import get_spacy_model
from candidates.gazetteer import load_gazetteer_matcher
from fusion import DEFAULT_SOURCE_WEIGHTS
from fact_filter import split_into_clauses_spacy, classify_clauses_llm
from llm.strategies import _ablation_accept
# from .metrics import MetricsLogger
from span_utils import dedupe_overlaps_longest, iou_tuple
try:
    from . import __all__  # silence unused warning (package import)
except Exception:
    # fallback when running the file directly
    try:
        from entity_extraction import __all__  # type: ignore
    except Exception:
        __all__ = []


def set_base_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_txt_files(indir: str):
    for name in os.listdir(indir):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(indir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            yield os.path.splitext(name)[0], f.read()


def _iter_with_split_flag(doc_iter):
    for doc_id, text in doc_iter:
        yield doc_id, text, False


def read_bioc_passages(indir: str):
    for name in os.listdir(indir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(indir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        articles = doc.get("sibils_article_set") or doc.get("articles") or []
        if not articles:
            articles = [doc]
        for a_idx, article in enumerate(articles):
            allowed_content_ids, allowed_section_ids, has_body_section_metadata = (
                _collect_allowed_content_ids_from_document(article)
            )

            def _is_allowed_field_content(field: str, content_id: str) -> bool:
                fld = str(field or "").strip().lower()
                if fld in _BIOC_EXCLUDED_SENTENCE_FIELDS:
                    return False
                if fld == "abstract":
                    return True
                if fld != "text":
                    return False
                if not has_body_section_metadata:
                    return True
                cid = str(content_id or "").strip()
                if not cid:
                    return False
                if cid in allowed_content_ids:
                    return True
                return any(cid == sec_id or cid.startswith(sec_id + ".") for sec_id in allowed_section_ids)

            passages = article.get("passages", []) or []
            if passages:
                passage_texts = []
                for passage in passages:
                    infons = passage.get("infons", {}) or {}
                    if not _is_allowed_field_content(infons.get("field"), infons.get("content_id")):
                        continue
                    text = passage.get("text") or ""
                    if text:
                        passage_texts.append(text)
                if passage_texts:
                    doc_id = f"{os.path.splitext(name)[0]}__a{a_idx}_passages"
                    yield doc_id, passage_texts, True
                    continue
            sentences = []
            for s in article.get("sentences", []) or []:
                if not _is_allowed_field_content(s.get("field"), s.get("content_id")):
                    continue
                text = s.get("sentence") or ""
                if text:
                    sentences.append(text)
            if sentences:
                doc_id = f"{os.path.splitext(name)[0]}__a{a_idx}_sentences"
                yield doc_id, sentences, True


def split_sentences(text: str) -> List[str]:
    try:
        sentences = sent_tokenize(text)
    except LookupError as exc:
        raise RuntimeError(
            "NLTK punkt tokenizer not found. Install it with: "
            "python -m nltk.downloader punkt"
        ) from exc
    return [s.strip() for s in sentences if s.strip()]


def load_checkpoint(path: str, total_sents: int) -> List[Any]:
    out: List[Any] = [None] * total_sents
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("idx")
            sent = rec.get("sentence")
            if isinstance(idx, int) and 0 <= idx < total_sents and sent is not None:
                if out[idx] is None:
                    out[idx] = sent
    return out


def _extract_types_from_llm(llm_result: Dict[str, Any]) -> List[str]:
    spans = None
    if isinstance(llm_result, dict):
        if isinstance(llm_result.get("final_spans"), list):
            spans = llm_result.get("final_spans")
        elif isinstance(llm_result.get("final_accepted"), list) or isinstance(llm_result.get("final_missing"), list):
            spans = (llm_result.get("final_accepted") or []) + (llm_result.get("final_missing") or [])
        elif isinstance(llm_result.get("accepted"), list) or isinstance(llm_result.get("missing"), list):
            spans = (llm_result.get("accepted") or []) + (llm_result.get("missing") or [])
    if not spans:
        return []
    types = []
    for s in spans:
        if isinstance(s, dict):
            t = s.get("type")
            if t:
                types.append(t)
    return types


def stats_from_out_sents(out_sents: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in out_sents:
        if not s or not isinstance(s, dict):
            continue
        llm_result = s.get("llm")
        for t in _extract_types_from_llm(llm_result):
            counts[t] = counts.get(t, 0) + 1
    return counts


def write_stats(path: str, doc_id: str, counts: Dict[str, int], processed: int, total: int) -> None:
    payload = {
        "doc_id": doc_id,
        "processed_sentences": processed,
        "total_sentences": total,
        "type_counts": counts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_global_stats(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"docs": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_global_stats(path: str, docs: Dict[str, Any]) -> None:
    totals: Dict[str, int] = {}
    for doc_stats in docs.values():
        for t, c in (doc_stats.get("type_counts") or {}).items():
            totals[t] = totals.get(t, 0) + int(c)
    payload = {
        "docs": docs,
        "type_counts": totals,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _candidate_sources(cand: Dict[str, Any]) -> List[str]:
    sources: List[str] = []
    raw_sources = cand.get("sources")
    if isinstance(raw_sources, list):
        for s in raw_sources:
            if isinstance(s, dict) and s.get("name"):
                sources.append(s["name"])
    if cand.get("source"):
        sources.append(cand["source"])
    return sources


def _tier_from_sources(sources: List[str]) -> int | None:
    if any(s in ("bioc", "gazetteer") for s in sources):
        return 1
    if any(s in ("ner", "chunks") for s in sources):
        return 2
    return None


def _tier_for_span(span: Dict[str, Any], candidates: List[Dict[str, Any]] | None, iou_thr: float) -> int | None:
    if not candidates:
        return None
    try:
        s = int(span.get("start_char"))
        e = int(span.get("end_char"))
    except Exception:
        return None
    if e <= s:
        return None
    found_tier2 = False
    for cand in candidates:
        try:
            cs = int(cand.get("start_char"))
            ce = int(cand.get("end_char"))
        except Exception:
            continue
        if ce <= cs:
            continue
        if iou_tuple((s, e), (cs, ce)) >= iou_thr or (s == cs and e == ce):
            tier = _tier_from_sources(_candidate_sources(cand))
            if tier == 1:
                return 1
            if tier == 2:
                found_tier2 = True
    return 2 if found_tier2 else None


def _assign_tiers_to_llm(llm_result: Dict[str, Any], candidates: List[Dict[str, Any]] | None, iou_thr: float) -> None:
    if not llm_result or not isinstance(llm_result, dict):
        return
    accepted = llm_result.get("accepted") or []
    missing = llm_result.get("missing") or []
    if isinstance(accepted, list):
        for a in accepted:
            if not isinstance(a, dict):
                continue
            a["tier"] = _tier_for_span(a, candidates, iou_thr) or 2
    if isinstance(missing, list):
        for m in missing:
            if not isinstance(m, dict):
                continue
            m["tier"] = 3
            
            
            
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=False, default=None, help="Folder with .txt documents")
    ap.add_argument("--in_bioc_dir", required=False, default=None,
                    help="Folder with BioC JSON files. Passage text is used as input.")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (one object per document)")
    ap.add_argument("--model_type", choices=["qwen3-4B", "qwen3-32B", "gpt4o", "biomistral-7b-awq", "qwen3-35B-vllm", "qwen3-32B-vllm"],
                    default="gpt4o", help="LLM model to use")
    ap.add_argument("--use_chunks", action="store_true", help="Use noun phrase chunks as candidates")
    ap.add_argument("--max_sents_per_doc", type=int, default=999999, help="Cap sentences per doc (debug)")
    ap.add_argument("--sample_every", type=int, default=1, help="Process every Nth sentence (e.g., 5 to sample)")
    ap.add_argument("--batch_size", type=int, default=15, help="Batch size for processing")
    ap.add_argument("--max_workers", type=int, default=4, help="Max worker threads")
    ap.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Directory for per-doc checkpoints (JSONL per doc). Defaults to <out_jsonl>.ckpt")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing checkpoints and skip processed sentences.")

    # ==== Candidate source switch ====
    ap.add_argument("--candidates_from", choices=["chunks", "ner", "bioc", "none", "all"], default="ner",
                    help="Generate LLM candidates from spaCy noun chunks, NER predictions, BioC, none (gazetteer-only if --gaz_dir set), or all (NER + chunks + gazetteer).")

    # ==== BioC ====
    ap.add_argument("--bioc_candidates_dir", type=str, default=None,
                    help="Directory with BioC JSON files. We will extract sentence->spans as candidates.")
    ap.add_argument("--use_bioc", action="store_true",
                    help="When using NER candidates, also add BioC spans if available.")
    ap.add_argument("--bioc_index_mode", choices=["sentence", "passage"], default="sentence",
                    help="How to index BioC for candidate lookup (sentence is recommended).")

    ap.add_argument("--schema_csv", type=str, default=None,
                    help="Optional schema CSV passed to the converter.")
    ap.add_argument("--envo_gazetteer_csv", type=str, default=None,
                    help="Optional ENVO gazetteer CSV passed to the converter.")
    ap.add_argument("--prefer_spacy", action="store_true",
                    help="Prefer spaCy PhraseMatcher inside the converter.")

    # ==== NER path params ====
    ap.add_argument("--ner_model_dir", type=str, default=None,
                    help="HF token classification model directory (must contain labels.txt).")
    ap.add_argument("--ner_batch_size", type=int, default=16)
    ap.add_argument("--ner_max_length", type=int, default=512)

    # ==== spaCy chunker params ===
    ap.add_argument("--spacy_model", default="en_core_web_trf", help="spaCy model (needs parser for noun_chunks)")

    ap.add_argument("--np_fallback", action="store_true",
                    help="If NER finds no entities in a sentence, fill candidates with NP chunks.")

    # ==== Gazetteer ====
    ap.add_argument("--gaz_dir", type=str, default=None,
                    help="Directory with CSV/TSV gazetteers; filename is used as entity type.")
    ap.add_argument("--general_table_dir", type=str, default=None,
                    help="Directory with general term table CSV.")
    ap.add_argument("--gaz_locked", action="store_true", help="Lock gazetteer candidates (no LLM changes).")

    # ==== Few-shot ====
    ap.add_argument("--few_shot", action="store_true", help="Use few-shot examples in system prompt.")

    # ==== Fusion ====
    ap.add_argument("--type_map_json", type=str, default=None,
                    help="JSON file mapping external labels to canonical schema (e.g., {'Biomes':'HABITAT'}).")
    ap.add_argument("--source_weights_json", type=str, default=None,
                    help="JSON file mapping source->weight (e.g., {'gazetteer':0.9,'ner':0.7}).")

    # ==== Fact checking ====
    ap.add_argument("--fact_filter", choices=["off", "llm"], default="off",
                    help="Filter candidates to FACT-only sentences. 'rule' uses cue-phrases, 'llm' uses the chat model, 'off' disables.")
    ap.add_argument("--fact_filter_policy", choices=["strict", "lenient"], default="strict",
                    help="If 'strict', only FACT passes. If 'lenient', FACT or UNSURE passes.")
    ap.add_argument("--fact_filter_scope", choices=["sentence", "clause"], default="clause",
                    help="Gate at sentence or clause level. Use 'clause' for mixed sentences.")

    # ==== LLM multi-pass strategy ====
    ap.add_argument("--llm_condition", choices=["C0", "C1", "C2", "C3", "C4"], default="C0",
                    help="C0=single pass (current two-path); C1=vanilla multipass; C2=diversity multipass; C3=critique-revise; C4=self-consistency.")
    ap.add_argument("--passes", type=int, default=3, help="Number of passes for C1/C2/C3.")
    ap.add_argument("--samples_k", type=int, default=5, help="Number of parallel samples for C4.")

    ap.add_argument("--base_seed", type=int, default=0, help="Random seed.")

    ap.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for span deduplication and tier assignment.")
    ap.add_argument("--lock_iou_thr", type=float, default=0.75, help="IoU threshold for locking high-confidence candidates before LLM.")

    ap.add_argument(
        "--ablation",
        choices=["off", "gaz_only", "ner_only", "gaz_ner"],
        default="off",
        help="Skip LLM and directly accept spans: gaz_only | ner_only | gaz_ner."
    )
    ap.add_argument("--metrics_csv", type=str, default=None, help="Optional CSV file to log metrics.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_base_seed(args.base_seed)

    type_map = {
          "biomes": "ABIOTIC ENTITY",
          "biota": "BIOTIC ENTITY",
          "mountains": "BIOTIC ENTITY",
          "mountainrange": "BIOTIC ENTITY",
          "geography": "SPATIAL ENTITY",
          "env_feature": "ABIOTIC PROPERTY",
          "population": "BIOTIC COLLECTIVE ENTITY",
          "taxon": "BIOTIC ENTITY",
          "location": "SPACIAL ENTITY",
          "habitat": "BIOTIC PROPERTY",
          "threat": "ANTHROPOGENIC PROCESS"
        }

    if args.type_map_json:
        with open(args.type_map_json, "r", encoding="utf-8") as f:
            type_map = json.load(f)

    source_weights = DEFAULT_SOURCE_WEIGHTS.copy()
    if args.source_weights_json:
        with open(args.source_weights_json, "r", encoding="utf-8") as f:
            source_weights.update(json.load(f))

    # Initialize LLM client
    try:
        llm = LLMClientFactory.create(args.model_type)
        print(f"Using {args.model_type} model: {llm.model_name}")
    except Exception as e:
        print(f"Error initializing {args.model_type} client: {e}", file=sys.stderr)
        sys.exit(1)

    gaz_matcher = load_gazetteer_matcher(args.gaz_dir, args.general_table_dir)

    # Validate spaCy model
    if args.candidates_from in ("chunks", "ner", "all") or args.np_fallback:
        try:
            test_nlp = spacy.load(args.spacy_model)
            if "parser" not in test_nlp.pipe_names:
                print("WARNING: spaCy parser not enabled; noun_chunks may be empty.", file=sys.stderr)
        except OSError:
            print(f"spaCy model '{args.spacy_model}' not found. Install with: python -m spacy download {args.spacy_model}",
                  file=sys.stderr)
            sys.exit(1)

    Path(os.path.dirname(args.out_jsonl) or ".").mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"{args.out_jsonl}.ckpt"
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing enabled at: {args.checkpoint_dir}")

    if args.in_bioc_dir and args.in_dir:
        print("Provide only one of --in_dir or --in_bioc_dir", file=sys.stderr)
        sys.exit(1)
    if not args.in_bioc_dir and not args.in_dir:
        print("Provide --in_dir or --in_bioc_dir", file=sys.stderr)
        sys.exit(1)

    if args.bioc_candidates_dir is None and args.in_bioc_dir:
        args.bioc_candidates_dir = args.in_bioc_dir

    # Load BioC candidates if requested
    bioc_index = None
    if args.candidates_from == "bioc" or args.use_bioc:
        if not args.bioc_candidates_dir:
            print("Provide --bioc_candidates_dir for candidates_from=bioc", file=sys.stderr)
            sys.exit(1)
        if args.bioc_index_mode == "passage":
            bioc_index = load_bioc_index_from_dir(args.bioc_candidates_dir)
        else:
            bioc_index = load_bioc_sentence_index_from_dir(args.bioc_candidates_dir)
        print(f"Loaded BioC candidates for {len(bioc_index)} sentences from {args.bioc_candidates_dir}")

        before = len(bioc_index)
        bioc_index = dedupe_bioc_index(bioc_index)
        after = len(bioc_index)
        print(f"[BioC] deduped unique sentences: {after} (from {before})")

    # NER runtime (loaded once)
    ner_runtime = None
    if args.candidates_from in ("ner", "all"):
        if not args.ner_model_dir:
            print("Provide --ner_model_dir for candidates_from=ner/all", file=sys.stderr)
            sys.exit(1)
        ner_runtime = load_ner(args.ner_model_dir)

    docs_written = 0
    global_stats_path = None
    global_docs: Dict[str, Any] = {}
    if args.checkpoint_dir:
        global_stats_path = os.path.join(args.checkpoint_dir, "_global.stats.json")
        if args.resume and os.path.exists(global_stats_path):
            global_docs = load_global_stats(global_stats_path).get("docs", {})

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        if args.in_bioc_dir:
            doc_iter = read_bioc_passages(args.in_bioc_dir)
        else:
            doc_iter = _iter_with_split_flag(read_txt_files(args.in_dir))


        for doc_id, text, is_pre_split in doc_iter:
            # metrics.start_doc()

            # no debug limit

            # Split text into sentences using NLTK unless already pre-split
            if is_pre_split:
                sentences = text
            else:
                sentences = split_sentences(text)

            out_sents: List[Any] = [None] * len(sentences)
            ckpt_path = None
            ckpt_fh = None
            stats_path = None
            type_counts: Dict[str, int] = {}
            if args.checkpoint_dir:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{doc_id}.jsonl")
                stats_path = os.path.join(args.checkpoint_dir, f"{doc_id}.stats.json")
                if args.resume and os.path.exists(ckpt_path):
                    out_sents = load_checkpoint(ckpt_path, len(sentences))
                elif os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                ckpt_fh = open(ckpt_path, "a", encoding="utf-8")

            if args.resume and out_sents:
                type_counts = stats_from_out_sents(out_sents)

            processed = sum(1 for s in out_sents if s is not None)
            print(f"Processing {len(sentences)} sentences from lines (cached: {processed})")
            if stats_path:
                write_stats(stats_path, doc_id, type_counts, processed, len(sentences))
            if global_stats_path:
                global_docs[doc_id] = {
                    "processed_sentences": processed,
                    "total_sentences": len(sentences),
                    "type_counts": type_counts,
                }
                write_global_stats(global_stats_path, global_docs)

            total_batches = (len(sentences) + args.batch_size - 1) // args.batch_size

            try:
                for bidx, i in enumerate(range(0, len(sentences), args.batch_size), start=1):
                    batch_indices = list(range(i, min(i + args.batch_size, len(sentences))))
                    pending_indices = [j for j in batch_indices if out_sents[j] is None]
                    if not pending_indices:
                        print(f"Skipping batch {bidx}/{total_batches} (all cached)")
                        continue
                    batch = [sentences[j] for j in pending_indices]

                    # === Candidate generation in-batch (spaCy tokenization for both modes)
                    candidate_results = process_sentences_batch(
                        batch,
                        args.spacy_model,
                        args.use_chunks,
                        candidates_from=args.candidates_from,
                        ner_runtime=ner_runtime,
                        ner_max_length=args.ner_max_length,
                        ner_runtime_batch_size=args.ner_batch_size,
                        np_fallback=args.np_fallback,
                        bioc_index=bioc_index,
                        gazetteer_matcher=gaz_matcher,
                        use_bioc=args.use_bioc,
                        type_map=type_map,
                        source_weights=source_weights
                    )

                    assert len(candidate_results) == len(batch), \
                        f"Lost sentences in candidate building: {len(batch)} -> {len(candidate_results)}"

                    if args.ablation != "off":
                        llm_results = []
                        for cr in candidate_results:
                            llm_results.append(
                                _ablation_accept(cr.get("candidates"), args.ablation, iou_thr=args.iou_thr)
                            )
                    else:

                        cond = args.llm_condition
                        if cond == "C0":
                            dec = dict(temperature=0.0, top_p=1.0, presence_penalty=0.0)
                            llm_results = call_llm_batch_two_path(llm, args.model_type, args.few_shot,
                                                                  candidate_results,
                                                                  lock_over_iou=args.lock_iou_thr,
                                                                  decoding=dec, gaz_lock=args.gaz_locked,
                                                                  max_workers=args.max_workers)
                        elif cond == "C1":
                            llm_results = run_C1_vanilla(llm, args.model_type, args.few_shot,
                                                         candidate_results, T=args.passes,
                                                         lock_over_iou=args.lock_iou_thr,
                                                         gaz_lock=args.gaz_locked,
                                                         max_workers=args.max_workers)
                        elif cond == "C2":
                            llm_results = run_C2_diverse(llm, args.model_type, args.few_shot,
                                                         candidate_results, T=args.passes,
                                                         lock_over_iou=args.lock_iou_thr,
                                                         gaz_lock=args.gaz_locked,
                                                         max_workers=args.max_workers)
                        elif cond == "C3":
                            llm_results = run_C3_critique_revise(llm, args.model_type, args.few_shot,
                                                                 candidate_results, T=args.passes,
                                                                 lock_over_iou=args.lock_iou_thr,
                                                                 gaz_lock=args.gaz_locked,
                                                                 max_workers=args.max_workers)
                        elif cond == "C4":
                            llm_results = run_C4_self_consistency(llm, args.model_type, args.few_shot,
                                                                  candidate_results, K=args.samples_k,
                                                                  lock_over_iou=args.lock_iou_thr,
                                                                  gaz_lock=args.gaz_locked,
                                                                  max_workers=args.max_workers)
                        else:
                            raise ValueError(f"Unknown --llm_condition {cond}")


                    if len(llm_results) != len(candidate_results):
                        raise RuntimeError(f"LLM results mismatch: {len(llm_results)} vs {len(candidate_results)}")

                    for llm_result in llm_results:
                        if not isinstance(llm_result, dict):
                            continue
                        if isinstance(llm_result.get("final_spans"), list):
                            continue
                        accepted = llm_result.get("accepted") or []
                        missing = llm_result.get("missing") or []
                        # Normalize outputs to always include final_spans.
                        llm_result["final_spans"] = dedupe_overlaps_longest(
                            accepted + missing,
                            iou_thr=args.iou_thr,
                        ) if (accepted or missing) else []

                    for idx_in_batch, (spacy_result, llm_result) in enumerate(zip(candidate_results, llm_results)):
                        _assign_tiers_to_llm(llm_result, spacy_result.get("candidates"), args.iou_thr)
                        sentence_data = {
                            "text": spacy_result["sentence"],
                            "llm": llm_result,
                        }
                        # if "notes" in spacy_result:
                        #     sentence_data["notes"] = spacy_result["notes"]
                        if "fact_clause_labels" in spacy_result:
                            sentence_data["fact_clause_labels"] = spacy_result["fact_clause_labels"]
                        if spacy_result["candidates"] is not None:
                            sentence_data["candidates"] = spacy_result["candidates"]
                        global_idx = pending_indices[idx_in_batch]
                        out_sents[global_idx] = sentence_data
                        if ckpt_fh:
                            ckpt_fh.write(json.dumps(
                                {"idx": global_idx, "sentence": sentence_data},
                                ensure_ascii=False
                            ) + "\n")
                        for t in _extract_types_from_llm(sentence_data.get("llm", {})):
                            type_counts[t] = type_counts.get(t, 0) + 1
                    if ckpt_fh:
                        ckpt_fh.flush()
                    if stats_path:
                        processed = sum(1 for s in out_sents if s is not None)
                        write_stats(stats_path, doc_id, type_counts, processed, len(sentences))
                    if global_stats_path:
                        global_docs[doc_id] = {
                            "processed_sentences": processed,
                            "total_sentences": len(sentences),
                            "type_counts": type_counts,
                        }
                        write_global_stats(global_stats_path, global_docs)

                    print(
                        f"Processed batch {bidx}/{total_batches} "
                        f"({len(pending_indices)}/{len(batch_indices)} new)"
                    )
            finally:
                if ckpt_fh:
                    ckpt_fh.close()

            # Write results
            missing = [i for i, s in enumerate(out_sents) if s is None]
            if missing:
                raise RuntimeError(
                    f"Missing {len(missing)} sentences for doc {doc_id}; "
                    f"rerun with --resume to fill from checkpoints."
                )
            if stats_path:
                write_stats(stats_path, doc_id, type_counts, len(out_sents), len(sentences))
            if global_stats_path:
                global_docs[doc_id] = {
                    "processed_sentences": len(out_sents),
                    "total_sentences": len(sentences),
                    "type_counts": type_counts,
                }
                write_global_stats(global_stats_path, global_docs)
            rec = {
                "doc_id": doc_id,
                "sentences": out_sents,
                "config": {
                    "model_type": args.model_type,
                    "model_name": llm.model_name,
                    "use_chunks": args.use_chunks
                }
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_written += 1
            print(f"Completed document {doc_id} with {len(out_sents)} sentences")
            # metrics.end_doc()

    mode_str = "with chunks" if args.use_chunks else "without chunks"
    print(f"Done. Processed {docs_written} documents using {args.model_type} {mode_str} and wrote to {args.out_jsonl}")
    stats = llm.token_stats()
    print(f"Token usage — queries: {stats['queries']}, "
          f"prompt: {stats['prompt_tokens_total']}, "
          f"completion: {stats['completion_tokens_total']}, "
          f"mean prompt/query: {stats['prompt_tokens_mean']:.1f}, "
          f"mean completion/query: {stats['completion_tokens_mean']:.1f}")


if __name__ == "__main__":
    main()
