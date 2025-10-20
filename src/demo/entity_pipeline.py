import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any
import threading
from pathlib import Path
import re, unicodedata
from collections import defaultdict, Counter

from statistics import median

import spacy
from openai import OpenAI

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ner.labels import EntityLabel, build_bio_labels
from ner.ner_infer import NerInferencer
from preprocess.gazetteer_matcher import Rule as GazetteerRule, load_gaz_rules_from_dir, GazetteerMatcher
import random, numpy as np, torch

from prompts import *
import glob



def set_base_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


IOU_THR = 0.7  # span merge tolerance
DEFAULT_SOURCE_WEIGHTS = {"gazetteer": 0.9, "ner": 0.7, "chunks": 0.6, "bioc": 0.8, "unknown": 0.6}


# Model configurations
MODEL_CONFIGS = {
    "qwen3-4B": {
        "base_url": "https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507"
    },
    "qwen3-32B": {
        "base_url": "https://openwebui.runai-codev-llm.inference.compute.datascience.ch/api",
        "api_key": None,  # Will use OPEN_WEB_UI_API_KEY env var
        "model_name": "Qwen/Qwen3-32B-AWQ"
    },
    "medgemma-4b": {
        "base_url": "http://medgemma-4b-it.runai-mobiko-anisia.inference.compute.datascience.ch",
        "api_key": "EMPTY",
        "model_name": "google/medgemma-4b-it"
    },
    "biomistral-7b-awq": {
        "base_url": "https://mistral-7b-awq.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model_name": "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
    },
    "gpt4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": None,  # Will use OPENAI_API_KEY env var
        "model_name": "gpt-4o"
    },
}



# Thread-local storage for spaCy models
thread_local = threading.local()


_ws = re.compile(r"\s+")


def _ws_tokenize_with_offsets(text: str):
    """
    Whitespace tokenization with character offsets.
    Returns (tokens, token_spans) where token_spans[i] = (start_char, end_char).
    """
    tokens, spans = [], []
    pos = 0
    while True:
        m = _ws.search(text, pos)
        end = m.start() if m else len(text)
        if end > pos:  # non-empty
            tokens.append(text[pos:end])
            spans.append((pos, end))
        if not m:
            break
        pos = m.end()
    return tokens, spans


def _bio_from_ids_to_spans(tags: List[str], token_spans: List[tuple], text: str):
    """
    Map BIO word tags → sentence-level char spans using precomputed token spans.
    """
    spans, active_type, start_i = [], None, None
    for i, tag in enumerate(tags):
        if not tag or tag == "O":
            if active_type is not None:
                s, _ = token_spans[start_i]; _, e = token_spans[i-1]
                spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
                active_type, start_i = None, None
            continue
        if "-" in tag:
            pref, typ = tag.split("-", 1)
        else:
            pref, typ = "B", tag
        if pref == "B" or (active_type and typ != active_type):
            if active_type is not None:
                s, _ = token_spans[start_i]; _, e = token_spans[i-1]
                spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
            active_type, start_i = typ, i
    if active_type is not None:
        s, _ = token_spans[start_i]; _, e = token_spans[len(token_spans)-1]
        spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
    return spans



def _load_bioc_index_from_dir(bioc_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read all *.json in a directory of BioC files.
    Extract per-sentence spans.
    Return {sentence_text -> [ {text, start_char, end_char, type}, ... ]}.
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    for path in glob.glob(os.path.join(bioc_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
            articles = (doc.get("sibils_article_set") or doc.get("articles"))
            for article in articles:

                if article.get("passages") and isinstance(article.get("passages"), list):
                    # Build sentence index: (field, sentence_number) -> sentence text

                    for passage in article["passages"]:
                        text = passage.get("text") or ""
                        if not text:
                            continue
                        spans = []
                        for annotation in passage.get("annotations", []):
                            infons = annotation.get("infons", {}) or {}

                            for location in annotation.get("locations", []):
                                start = int(location.get("offset", 0))
                                length = int(location.get("length", 0))
                                end = start + length

                                # Validate span boundaries
                                if end <= start or start < 0 or end > len(text):
                                    continue

                                spans.append({
                                    "start": start,
                                    "end": end,
                                    "text": text[start:end],
                                    "source": infons.get("concept_source"),
                                    "concept_id": infons.get("concept_id"),
                                    "preferred_term": infons.get("preferred_term")
                                })
                        # Sort spans (stable) for predictability
                        spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
                        index[text] = spans
    return index


# === Robust JSON parsing helpers ===
_JSON_OBJ_RE = re.compile(r'\{.*\}', flags=re.DOTALL)

def safe_json_from_llm(raw: str, kind: str = "extract") -> dict:
    """
    Best-effort parse. Returns a dict with accepted/rejected/missing; never raises.
    - Strips code fences and preambles
    - Normalizes curly quotes
    - Extracts the first {...} block if there's extra text
    - Removes trailing commas
    - Tries a single-quote -> double-quote coercion path
    - Falls back to empty structure with a note
    """
    # ... (helper uses _extract_first_json, _remove_trailing_commas, etc.)
    return {"accepted": [], "rejected": [], "missing": [], "notes": "json_repair_failed"}



#---- Strategies ------------

# === Decoding profiles & consensus merge ===

def _decoding_profile(pass_idx: int, condition: str) -> dict:
    """
    Return decoding params per pass/condition.
    - C1: fixed low-temp (vanilla)
    - C2: diversity-forced grid across passes
    - C3: like C1 for first pass; revision uses its own temp
    - C4: handled separately (multiple samples)
    """
    if condition in ("C1", "C3"):
        return dict(temperature=0.7, top_p=0.95, presence_penalty=0.0)
    if condition == "C2":
        grid = [
            dict(temperature=0.3, top_p=0.90, presence_penalty=0.0),
            dict(temperature=0.6, top_p=0.95, presence_penalty=0.7),
            dict(temperature=0.9, top_p=0.98, presence_penalty=1.0),
        ]
        return grid[(pass_idx - 1) % len(grid)]
    # fallback
    return dict(temperature=0.7, top_p=0.95, presence_penalty=0.0)


def _span_iou_char(a, b) -> float:
    s = max(int(a["start_char"]), int(b["start_char"]))
    e = min(int(a["end_char"]), int(b["end_char"]))
    inter = max(0, e - s)
    if inter <= 0:
        return 0.0
    ua = int(a["end_char"]) - int(a["start_char"])
    ub = int(b["end_char"]) - int(b["start_char"])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _consensus_merge_by_type(spans_list: List[List[Dict[str,Any]]],
                             iou_thr: float = 0.5) -> List[Dict[str,Any]]:
    """
    Merge many accepted-span lists (from passes or parallel samples) into one.
    Group by TYPE, then greedy IoU clustering. Boundaries = median of members.
    """
    pool = []
    for spans in spans_list:
        for s in spans or []:
            if "type" in s and s.get("type"):
                pool.append({**s, "confidence": float(s.get("confidence", 1.0))})

    # simple greedy clustering per type
    out = []
    used = [False]*len(pool)
    for i, si in enumerate(pool):
        if used[i]:
            continue
        group = [si]; used[i] = True
        for j, sj in enumerate(pool):
            if used[j] or sj.get("type") != si.get("type"):
                continue
            if _span_iou_char(si, sj) >= iou_thr:
                used[j] = True
                group.append(sj)
        starts = sorted(int(g["start_char"]) for g in group)
        ends   = sorted(int(g["end_char"]) for g in group)
        merged = {
            "text": group[0]["text"],  # optional: can slice original sentence later if needed
            "type": group[0]["type"],
            "start_char": int(median(starts)),
            "end_char":   int(median(ends)),
            "confidence": sum(g.get("confidence", 1.0) for g in group)/len(group),
            "votes": len(group)
        }
        out.append(merged)
    return out


def _is_gazetteer(fused_cand: dict) -> bool:
    """True if any provenance source is 'gazetteer'."""
    for s in fused_cand.get("sources", []):
        if s.get("name") == "gazetteer":
            return True
    return fused_cand.get("source") == "gazetteer"


def _choose_gaz_type(fused_cand: dict, fallback_argmax: bool = True) -> Optional[str]:
    """Pick a definitive type for a locked gazetteer span."""
    gaz_types = [s.get("type") for s in fused_cand.get("sources", []) if s.get("name") == "gazetteer" and s.get("type")]
    gaz_types = [t for t in gaz_types if t]
    if gaz_types:
        t, _ = Counter(gaz_types).most_common(1)[0]
        return t
    if fallback_argmax:
        tv = fused_cand.get("type_votes") or {}
        if tv:
            return max(tv.items(), key=lambda kv: kv[1])[0]
    return None


def _overlaps_any(span: tuple[int,int], spans: list[tuple[int,int]], iou_thr: float = 0.5) -> bool:
    s, e = span
    for (s2, e2) in spans:
        if _iou_tuple((s, e), (s2, e2)) >= iou_thr:
            return True
    return False


def get_openai_client(model_type: str):
    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"Unknown model type: {model_type}. Use: {list(MODEL_CONFIGS.keys())}")

    api_key = config["api_key"] or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_WEB_UI_API_KEY")
    if not api_key:
        raise ValueError(f"API key required for {model_type}. Set OPENAI_API_KEY or OPEN_WEB_UI_API_KEY environment variable.")

    return OpenAI(
                base_url=config["base_url"],
                api_key=api_key
                ), config["model_name"]


def get_spacy_model(model_name: str):
    """Get thread-local spaCy model for parallel processing."""
    if not hasattr(thread_local, 'nlp'):
        thread_local.nlp = spacy.load(model_name)
    return thread_local.nlp


def read_txt_files(indir: str):
    for name in os.listdir(indir):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(indir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            yield os.path.splitext(name)[0], f.read()


def find_span_positions(text: str, span_text: str):
    """
    Find all positions of span_text in text using regex.
    Returns list of (start, end) tuples for all matches.
    """
    # Escape special regex characters in the span text
    escaped_span = re.escape(span_text.strip())

    # Find all matches (case-insensitive, word boundaries optional)
    matches = []
    for match in re.finditer(escaped_span, text, re.IGNORECASE):
        start, end = match.span()
        matches.append((start, end))

    return matches


def fix_span_indices(spans: list, sentence_text: str) -> List[Dict]:
    """
    Fix span indices using regex matching.
    Returns updated spans with correct start_char and end_char.
    """

    fixed_spans = []
    for span in spans:
        span_text = span.get("text", "").strip()
        if not span_text:
            continue

        positions = find_span_positions(sentence_text, span_text)
        if positions:
            # Use the first available position
            start, end = positions[0]
            fixed_span = dict(span)
            fixed_span["start_char"] = start
            fixed_span["end_char"] = end
            fixed_span["text"] = sentence_text[start:end]  # Use actual text from sentence
            fixed_spans.append(fixed_span)
        else:
            # Span text not found in sentence - log warning but keep original
            print(f"WARNING: Could not find span '{span_text}' in sentence: {sentence_text}")
            fixed_span = dict(span)
            fixed_span["start_char"] = 0
            fixed_span["end_char"] = 0
            fixed_span["text"] = span_text
            fixed_spans.append(fixed_span)

    return fixed_spans


def remove_thinking_blocks(content: str) -> str:
    # Remove <think>...</think> blocks (including nested content)
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL)

    # Clean up extra whitespace
    cleaned = cleaned.strip()

    # If content starts with ```json, extract just the JSON part
    if cleaned.startswith('```json'):
        # Find the JSON block
        start = cleaned.find('```json') + 7
        end = cleaned.rfind('```')
        if end > start:
            cleaned = cleaned[start:end].strip()

    return cleaned


# --------- Multi-pass code ---------


def _iou(a, b):
    s = max(a["start_char"], b["start_char"])
    e = min(a["end_char"], b["end_char"])
    inter = max(0, e - s)
    union = (a["end_char"] - a["start_char"]) + (b["end_char"] - b["start_char"]) - inter
    return inter / union if union else 0.0


def merge_spans(primary, additions, iou_thr=0.5):
    merged = primary[:]
    for cand in additions:
        if not any(_iou(cand, m) >= iou_thr for m in merged):
            merged.append(cand)
    return merged



def call_llm_batch_revision(client, model_name, model_type, requests, temperature=0.3):
    results = []
    for req in requests:
        system_prompt = """You are revising an earlier extraction.
        Return STRICT JSON:
        {"missing": [...], "notes": "short rationale ≤160 chars"}
            Rules:
            - Consider the previous output AND the running notes.
            - DO NOT DELETE earlier accepted spans.
            - Only propose NEW or CORRECTED spans (no duplicates).
            - Prefer precise boundaries; don't re-state identical spans.    
            - Use the same TYPE schema as before.
            """
        payload = {"sentence": req["sentence"], "previous": req["prev_json"],
                   "prev_notes": req.get("prev_notes", "")  # short running notes
}
        if model_type == "qwen3-32B":
            system_prompt = f"<no_think/>\n\n{system_prompt}"

        content = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"{system_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
            temperature=temperature,
            max_tokens=512,
            timeout=360,
        ).choices[0].message.content

        if model_type in ("qwen3-32B", "gpt4o"):
            content = remove_thinking_blocks(content)
        obj = safe_json_from_llm(content, kind="extract")  # or kind="revision"
        obj["missing"] = fix_span_indices(obj.get("missing", []), req["sentence"])
        results.append(obj)
    return results


def call_llm_batch_consolidate(client, model_name, model_type, requests):
    results = []
    for req in requests:
        system_prompt = """Consolidate proposed spans.
            Return STRICT JSON:
            {"accepted":[{"text":"...", "type":"...", "start_char":int, "end_char":int}], "rejected":[...], "missing":[], "notes":"optional"}
            Rules:
            - Merge overlapping duplicates; keep one with best boundaries
            """
        payload = {"sentence": req["sentence"], "proposals": req["proposals"]}
        if model_type == "qwen3-32B":
            system_prompt = f"<no_think/>\n\n{system_prompt}"

        content = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"{system_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
            temperature=0.0,
            max_tokens=768,
            timeout=360
        ).choices[0].message.content

        if model_type in ("qwen3-32B", "gpt4o"):
            content = remove_thinking_blocks(content)
        obj = json.loads(content)

        for k in ("accepted", "rejected"):
            obj[k] = fix_span_indices(obj.get(k, []), req["sentence"])
        obj["missing"] = []
        results.append(obj)
    return results


def call_llm_multipass(client, model_name, model_type, requests, passes=3, temp1=0.3):
    # Pass 1
    p1 = call_llm_batch(client, model_name, model_type, requests)

    # Build revision prompts (Pass 2): same requests, but include previous JSON and ask for "missing" only
    rev_requests = []
    for req, out in zip(requests, p1):
        rev_requests.append({
            "sentence": req["sentence"],
            "candidates": req.get("candidates"),
            "prev_json": out  # include accepted/rejected/missing
        })
    p2 = call_llm_batch_revision(client, model_name, model_type, rev_requests, temperature=temp1)

    # Merge P1 accepted + P2 missing
    merged_after_p2 = []
    for o1, o2, req in zip(p1, p2, requests):
        accepted = o1.get("accepted", [])
        missing = o2.get("missing", [])
        merged = merge_spans(accepted, missing, iou_thr=0.5)
        merged_after_p2.append({"accepted": merged, "notes": ""})

    # Pass 3: consolidation/type-check using unioned spans as "proposal"
    cons_requests = []
    for req, uni in zip(requests, merged_after_p2):
        cons_requests.append({
            "sentence": req["sentence"],
            "proposals": uni["accepted"]
        })
    p3 = call_llm_batch_consolidate(client, model_name, model_type, cons_requests)

    return p3


def call_llm_batch_two_path(
    client,
    model_name: str,
    model_type: str,
    few_shot: bool,
    requests: List[Dict],
    lock_over_iou: float = 0.5,
    decoding: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Two-path inference:
      - Gazetteer-backed candidates are auto-accepted (locked).
      - LLM evaluates only non-gazetteer candidates.
      - On overlap, gazetteer wins.
    """
    # Split requests into (locked gaz, to_llm) per sentence
    filtered_reqs = []
    per_req_locked = []   # keep locked spans and their geometry

    for req in requests:
        sent = req["sentence"]
        cands = req.get("candidates")

        if not cands:
            # Nothing to lock/ask; let base function handle None path
            filtered_reqs.append(req)
            per_req_locked.append({"locked": [], "locked_spans": []})
            continue

        # Identify fused gazetteer candidates
        locked = [c for c in cands if _is_gazetteer(c)]
        to_llm = [c for c in cands if not _is_gazetteer(c)]

        # Materialize locked accepteds now
        lock_accepteds = []
        locked_spans = []
        for c in locked:
            t = _choose_gaz_type(c, fallback_argmax=True)
            if not t:
                # If no type could be chosen, skip locking
                continue
            lock_accepteds.append({
                "text": c["text"],
                "type": t,
                "start_char": int(c["start_char"]),
                "end_char": int(c["end_char"]),
                "source": "gazetteer"
            })
            locked_spans.append((int(c["start_char"]), int(c["end_char"])))


        print(f"Locked spans for sent: {len(lock_accepteds)}")
        print(f"To LLM spans for sent: {len(to_llm)}")
        # Build a request for LLM with only non-gaz candidates
        if to_llm:
            filtered_reqs.append({"sentence": sent, "candidates": to_llm})
        else:
            filtered_reqs.append({"sentence": sent, "candidates": []})  # keeps alignment

        per_req_locked.append({"locked": lock_accepteds, "locked_spans": locked_spans})

    # Call your existing batch function
    llm_outs = call_llm_batch(client, model_name, model_type, few_shot, filtered_reqs, decoding)

    # Merge locked + llm_out, gaz wins on overlaps
    merged_outs: List[Dict] = []
    for info, llm_res in zip(per_req_locked, llm_outs):
        locked_acc = info["locked"]
        locked_spans = info["locked_spans"]

        if not llm_res:
            merged_outs.append({"accepted": locked_acc, "rejected": [], "missing": [], "notes": "llm_empty"})
            continue

        merged_accepted = list(locked_acc)
        for a in llm_res.get("accepted", []):
            if not _overlaps_any((int(a["start_char"]), int(a["end_char"])), locked_spans, iou_thr=lock_over_iou):
                merged_accepted.append(a)

        merged_rejected = []
        for r in llm_res.get("rejected", []):
            sc = int(r.get("start_char", -1)); ec = int(r.get("end_char", -1))
            if sc >= 0 and ec >= 0 and _overlaps_any((sc, ec), locked_spans, iou_thr=lock_over_iou):
                continue  # ignore rejection that conflicts with a lock
            merged_rejected.append(r)

        merged_missing = []
        for m in llm_res.get("missing", []):
            if not _overlaps_any((int(m["start_char"]), int(m["end_char"])), locked_spans, iou_thr=lock_over_iou):
                merged_missing.append(m)

        merged_outs.append({
            "accepted": merged_accepted,
            "rejected": merged_rejected,
            "missing": merged_missing,
            "notes": llm_res.get("notes", "")
        })

    return merged_outs


def call_llm_batch(
        client,
        model_name: str,
        model_type: str,
        few_shot: bool,
        requests: List[Dict],
        decoding: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """Process multiple LLM requests efficiently."""
    results = []

    decoding = decoding or {}
    temperature = float(decoding.get("temperature", 0.7))
    top_p = float(decoding.get("top_p", 0.95))
    presence_penalty = float(decoding.get("presence_penalty", 0.0))


    # Configure model-specific parameters
    if model_type in ["qwen3-32B", "gpt4o"]:
        max_tokens = 1024
    else:
        max_tokens = 500


    for req in requests:
        sentence = req["sentence"]
        candidates = req.get("candidates")

        if candidates is None:
            system_prompt = NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
            user_payload = {"sentence": sentence}
        else:
            # Determine whether candidates include NER-proposed types
            has_types = any("type" in c and c["type"] for c in candidates)
            if has_types:
                # NER-aware path: include proposed_type
                system_prompt = NER_AWARE_SYSTEM_PROMPT
                cand_objs = []
                for c in candidates:
                    obj = {
                        "text": c["text"].strip(),
                        "start_char": c["start_char"],
                        "end_char": c["end_char"],
                        "proposed_type": c.get("type", None)
                    }
                    cand_objs.append(obj)
                user_payload = {"sentence": sentence, "candidates": cand_objs}
            else:
                if few_shot:
                    system_prompt = SYSTEM_PROMPT_FEW_SHOT_NEW
                else:
                    # Legacy chunks path: no types proposed
                    system_prompt = DEFAULT_SYSTEM_PROMPT_NEW
                cand_objs = [
                    {"text": c["text"].strip(), "start_char": c["start_char"], "end_char": c["end_char"]}
                    for c in candidates
                ]
                user_payload = {"sentence": sentence, "candidates": cand_objs}

        # Modify system prompt for Qwen 32B
        if model_type == "qwen3-32B":
            system_prompt = f"<no_think/>\n\n{system_prompt}"

        full_prompt = f"{system_prompt}\n\nUser input: {json.dumps(user_payload, ensure_ascii=False)}"

        try:
            print(f'Sending LLM request for sentence: {sentence[:50]}... with {len(candidates) if candidates else 0} candidates')

            # if candidates and len(candidates) >= 10:
            #     print(full_prompt)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                timeout=800

            )
            print('LLM response received.')
            content = response.choices[0].message.content

            # Postprocess content for Qwen 32B to remove thinking blocks
            if model_type == "qwen3-32B" or model_type == "gpt4o":
                content = remove_thinking_blocks(content)

            llm_result = json.loads(content)

            # Fix indices for all span categories
            for category in ["accepted", "missing", "rejected"]:
                if category in llm_result:
                    llm_result[category] = fix_span_indices(
                        llm_result[category], sentence)

            results.append(llm_result)

        except Exception as e:
            print(f"Error calling LLM for sentence: {sentence[:50]}...: {e}")
            results.append({
                "accepted": [], "rejected": [], "missing": [],
                "notes": f"llm_error: {repr(e)}"
            })

    return results


# === ADD: Multi-pass runners ===

def run_C1_vanilla(
    client, model_name, model_type, few_shot, candidate_results, T: int = 3, lock_over_iou: float = 0.5) -> List[Dict]:
    """
    Vanilla multi-pass:
      P1: standard two-path call
      P2..T: 'revision' (missing-only) then consolidate
    """
    # P1
    dec = _decoding_profile(1, "C1")
    p1 = call_llm_batch_two_path(client, model_name, model_type, few_shot,
                                 candidate_results, lock_over_iou=lock_over_iou, decoding=dec)

    if T == 1:
        return p1

    # Build revision requests from P1
    rev_reqs = []
    for req, out in zip(candidate_results, p1):
        rev_reqs.append({"sentence": req["sentence"], "prev_json": out})

    p2 = call_llm_batch_revision(client, model_name, model_type, rev_reqs, temperature=0.7)

    # Merge P1 accepted + P2 missing
    merged_after_p2 = []
    for o1, o2, req in zip(p1, p2, candidate_results):
        merged = merge_spans(o1.get("accepted", []), o2.get("missing", []), iou_thr=0.5)
        merged_after_p2.append({"sentence": req["sentence"], "accepted": merged})
        print(merged_after_p2)

    # Consolidate (optional final pass if T>=3)
    if T >= 3:
        cons_requests = [{"sentence": m["sentence"], "proposals": m["accepted"]} for m in merged_after_p2]
        p3 = call_llm_batch_consolidate(client, model_name, model_type, cons_requests)
        return p3
    else:
        # fabricate consolidate-like structure
        return [{"accepted": m["accepted"], "rejected": [], "missing": [], "notes": "C1-P2"} for m in merged_after_p2]


def run_C2_diverse(
    client, model_name, model_type, few_shot, candidate_results, T: int = 3, lock_over_iou=0.5) -> List[Dict]:
    """
    Diversity-forced multi-pass: each pass uses a different decoding profile,
    aggregate with NMS-style consensus by type.
    """
    pass_accepteds_per_sent = [[] for _ in candidate_results]

    for t in range(1, T+1):
        dec = _decoding_profile(t, "C2")
        out = call_llm_batch_two_path(client, model_name, model_type, few_shot,
                                      candidate_results, lock_over_iou=lock_over_iou, decoding=dec)
        for i, o in enumerate(out):
            pass_accepteds_per_sent[i].append(o.get("accepted", []))

    # consensus merge per sentence
    merged = []
    for i, req in enumerate(candidate_results):
        merged_acc = _consensus_merge_by_type(pass_accepteds_per_sent[i], iou_thr=0.5)
        merged.append({"accepted": merged_acc, "rejected": [], "missing": [], "notes": f"C2-T{T}"})
    return merged


# def run_C3_critique_revise(
#     client, model_name, model_type, few_shot, candidate_results, T: int = 3
# ) -> List[Dict]:
#     """
#     Critique-and-revise: P1 as usual, then iterative 'missing-only' revisions,
#     finally consolidate.
#     """
#     # P1 (fixed profile)
#     dec = _decoding_profile(1, "C3")
#     p = call_llm_batch_two_path(client, model_name, model_type, few_shot,
#                                 candidate_results, lock_over_iou=0.5, decoding=dec)
#
#     # P2..T-1: iterative revise -> union
#     current = []
#     for i, o in enumerate(p):
#         current.append({"sentence": candidate_results[i]["sentence"], "accepted": o.get("accepted", [])})
#
#     for t in range(2, max(2, T)):
#         rev_reqs = []
#         for i, cur in enumerate(current):
#             prev_json = {"accepted": cur["accepted"], "rejected": [], "missing": []}
#             rev_reqs.append({"sentence": cur["sentence"], "prev_json": prev_json, "prev_notes": cur.get("notes", "")})
#         rev = call_llm_batch_revision(client, model_name, model_type, rev_reqs, temperature=0.9)
#         # union with IoU
#         for i, r in enumerate(rev):
#             acc = current[i]["accepted"]
#             add = r.get("missing", [])
#             current[i]["accepted"] = merge_spans(acc, add, iou_thr=0.5)
#
#     # Final consolidate
#     cons_requests = [{"sentence": cur["sentence"], "proposals": cur["accepted"]} for cur in current]
#     final = call_llm_batch_consolidate(client, model_name, model_type, cons_requests)
#     return final

def run_C3_critique_revise(client, model_name, model_type, few_shot, candidate_results, T: int = 3,
                           lock_over_iou: float = 0.5) -> List[Dict]:
    # P1: normal two-path extraction (already returns {"accepted": [...], "notes": "..."} via safe_json_from_llm)
    dec = _decoding_profile(1, "C3")
    p = call_llm_batch_two_path(client, model_name, model_type, few_shot,
                                candidate_results, lock_over_iou=lock_over_iou, decoding=dec)

    # Keep rolling state per sentence
    state = []
    for i, base in enumerate(p):
        state.append({
            "sentence": candidate_results[i]["sentence"],
            "accepted": base.get("accepted", []),
            "notes": (base.get("notes") or "").strip()
        })

    # P2..T-1: critique→revise with notes
    for t in range(2, max(2, T)):
        rev_reqs = []
        for s in state:
            prev_json = {"accepted": s["accepted"], "rejected": [], "missing": []}
            rev_reqs.append({
                "sentence": s["sentence"],
                "prev_json": prev_json,
                "prev_notes": s["notes"]  # <<<<<< pass the rolling notes
            })

        rev = call_llm_batch_revision(client, model_name, model_type, rev_reqs, temperature=0.9)

        # union spans and append new notes
        for i, r in enumerate(rev):
            new_missing = r.get("missing", [])
            state[i]["accepted"] = merge_spans(state[i]["accepted"], new_missing, iou_thr=lock_over_iou)
            note_add = (r.get("notes") or "").strip()
            if note_add:
                # keep notes compact but cumulative
                if state[i]["notes"]:
                    state[i]["notes"] = (state[i]["notes"] + " | " + note_add)[:240]
                else:
                    state[i]["notes"] = note_add[:240]

    # Final consolidate pass can also see notes if you want (optional)
    cons_requests = []
    for s in state:
        cons_requests.append({
            "sentence": s["sentence"],
            "proposals": s["accepted"],
            # Optional: include notes so the consolidator can resolve conflicts
            "prev_notes": s["notes"]
        })
    final = call_llm_batch_consolidate(client, model_name, model_type, cons_requests)

    # Preserve rolled notes into the final objects (for logging/analysis)
    for i, obj in enumerate(final):
        if state[i]["notes"]:
            obj["notes"] = (obj.get("notes") or "")
            if obj["notes"]:
                obj["notes"] = (obj["notes"] + " | " + state[i]["notes"])[:300]
            else:
                obj["notes"] = state[i]["notes"][:300]

    return final


def run_C4_self_consistency(
    client, model_name, model_type, few_shot, candidate_results, K: int = 5
) -> List[Dict]:
    """
    Self-consistency in one pass: sample K independent outputs, then consensus merge.
    """
    samples_per_sent = [[] for _ in candidate_results]
    # Use a small diverse grid across K
    grid = [
        dict(temperature=0.3, top_p=0.90, presence_penalty=0.0),
        dict(temperature=0.6, top_p=0.95, presence_penalty=0.7),
        dict(temperature=0.9, top_p=0.98, presence_penalty=1.0),
    ]
    for k in range(K):
        dec = grid[k % len(grid)]
        out = call_llm_batch_two_path(client, model_name, model_type, few_shot,
                                      candidate_results, lock_over_iou=0.5, decoding=dec)
        for i, o in enumerate(out):
            samples_per_sent[i].append(o.get("accepted", []))

    merged = []
    for i, req in enumerate(candidate_results):
        merged_acc = _consensus_merge_by_type(samples_per_sent[i], iou_thr=0.5)
        merged.append({"accepted": merged_acc, "rejected": [], "missing": [], "notes": f"C4-K{K}"})
    return merged



def process_with_chunks(sent):
    """ Extract noun phrase candidates from sentence """
    cands = []
    for np in sent.noun_chunks:
        if np.root.pos_ not in ("NOUN", "PROPN"):
            continue
        np_text = np.text.strip()
        if not np_text:
            continue

        cands.append({
            "start_char": np.start_char,
            "end_char": np.end_char,
            "text": np_text,
            "source": "chunks"
        })
    return cands


def _load_labels_from_model(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "id2label" in cfg:
            # ensure sorted by id
            id2label = {int(k): v for k, v in cfg["id2label"].items()}
            return [id2label[i] for i in sorted(id2label.keys())]
        if "label2id" in cfg:
            label2id = {k: int(v) for k, v in cfg["label2id"].items()}
            return [lab for lab, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
    # Fallback to your provided set if config lacks labels
    provided = build_bio_labels()
    return sorted(provided)


def _load_ner(model_dir: str):
    # Backwards-compat shim if other code relies on this name
    infer = NerInferencer(model_dir, dtype="auto")
    return infer


def _normalize_bioc_spans(spans: List[Dict[str, Any]], sent_text: str) -> List[Dict[str, Any]]:
    """Map BioC spans to the schema your LLM path expects, then fix indices."""
    norm_spans = []
    for s in spans:
        if "text" in s and "start" in s and "end" in s:
            norm_spans.append({
                "text": s["text"].strip(),
                "start_char": int(s["start"]),
                "end_char": int(s["end"]),
                "type": s.get("type"),  # may be None; prompt handles both
                "source": "bioc"
            })
    return fix_span_indices(norm_spans, sent_text)


def _build_np_fallback(
    sentence_batch: List[str],
    spacy_model: str,
    empty_idx: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    """Compute chunk candidates only for sentences where NER found nothing."""

    fallback_maps: Dict[int, List[Dict[str, Any]]] = {}
    if not empty_idx:
        return fallback_maps

    nlp = get_spacy_model(spacy_model)
    empty_sents = [sentence_batch[i] for i in empty_idx]
    empty_docs = list(nlp.pipe(empty_sents))

    for j, doc in enumerate(empty_docs):
        i = empty_idx[j]
        # keep chunk candidates untyped; LLM prompt will use DEFAULT path
        fallback_maps[i] = process_with_chunks(doc)
    return fallback_maps


def process_sentences_batch(
    sentence_batch: List[str],
    spacy_model: str,
    use_chunks: bool,
    candidates_from: str = "ner",
    ner_runtime: Optional["NerInferencer"] = None,
    ner_max_length: int = 512,
    ner_runtime_batch_size: int = 64,
    np_fallback: bool = False,
    bioc_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    gazetteer_matcher: GazetteerMatcher | None = None,
    use_bioc: bool = False,
    type_map: Optional[Dict[str, str]] = None,
    source_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Build candidates for a batch of sentences.

    When candidates_from == "ner", this uses a shared NerInferencer to produce
    BIO→char spans (with types) directly from the NER model, so the same runtime
    is used across pipeline and eval.

    Returns:
        List[{"sentence": <str>, "candidates": List[{"start_char": int,
                                                     "end_char": int,
                                                     "text": str,
                                                     "type": str}]}]
        for NER mode. For other modes, keep your previous structure.
    """
    batch_results: List[Dict[str, Any]] = []

    if candidates_from == "bioc":
        if bioc_index is None:
            raise ValueError("candidates_from='bioc' requires bioc_index (load with _load_bioc_index).")
        for sent_text in sentence_batch:
            spans = bioc_index.get(sent_text, [])
            norm_spans = _normalize_bioc_spans(spans, sent_text)
            batch_results.append({"sentence": sent_text, "candidates": norm_spans if norm_spans else None})
        print(f'Processed {len(batch_results)} sentences with bioc candidates')
        return batch_results


    # All-chunks mode
    if candidates_from == "chunks" and use_chunks:
        nlp = get_spacy_model(spacy_model)
        docs = list(nlp.pipe(sentence_batch))
        for sent_text, sent_doc in zip(sentence_batch, docs):
            cands = process_with_chunks(sent_doc)

            if gazetteer_matcher is not None:
                gz = gazetteer_candidates(sent_text, gazetteer_matcher)
                if len(gz):
                    print(gz)
                    cands.extend(gz)

            if not cands:
                continue

            fused = fuse_candidates(
                cands,
                type_map=type_map,
                source_weights=source_weights
            )
            batch_results.append({"sentence": sent_text, "candidates": fused})
        print(f'Processed {len(batch_results)} sentences with spaCy chunks')
        return batch_results


    # NER mode (+ optional NP fallback)
    if candidates_from == "ner":
        if ner_runtime is None:
            raise ValueError(
                "NER candidates requested but ner_runtime is None. "
                "Initialize with NerInferencer(model_dir) and pass it in."
            )

        spans_lists = ner_runtime.predict_spans_for_sentences(
            sentences=sentence_batch,
            batch_size=ner_runtime_batch_size,
            max_length=ner_max_length,
            entity_threshold=0.25,
            entity_bias=0.25
        )

        # Collect indices with no NER spans (eligible for fallback)
        empty_idx = [i for i, spans in enumerate(spans_lists) if not spans]

        fallback_maps: Dict[int, List[Dict[str, Any]]] = {}
        if np_fallback and empty_idx:
            fallback_maps = _build_np_fallback(sentence_batch, spacy_model, empty_idx)

        # build unified results
        for i, sent_text in enumerate(sentence_batch):
            spans = spans_lists[i]

            # Set source for NER spans
            for c in spans or []:
                c.setdefault("source", "ner")

            gz = gazetteer_candidates(sent_text, gazetteer_matcher) if gazetteer_matcher is not None else []

            if spans:
                # Combine NER + gazetteer candidates
                if len(gz):
                    spans = spans + gz
                # Also add BioC spans if available
                if use_bioc and bioc_index:
                    spans = bioc_index.get(sent_text, [])
                    norm_spans = _normalize_bioc_spans(spans, sent_text)
                    if norm_spans:
                        spans = spans + norm_spans

                fused = fuse_candidates(
                    spans,
                    type_map=type_map,
                    source_weights=source_weights
                )
                batch_results.append({"sentence": sent_text, "candidates": fused})
            else:
                # Fallback (if any), else leave candidates=None
                if np_fallback:
                    cands = fallback_maps.get(i, [])
                    if gz:
                        cands = cands + gz
                    if cands:
                        fused = fuse_candidates(
                            cands,
                            type_map=type_map,
                            source_weights=source_weights
                        )
                        batch_results.append({"sentence": sent_text, "candidates": fused})
                    else:
                        batch_results.append({"sentence": sent_text, "candidates": None})
                else:
                    if gz:
                        batch_results.append({"sentence": sent_text, "candidates": gz})
                    else:
                        batch_results.append({"sentence": sent_text, "candidates": None})
        return batch_results

    # candidates_from == "none"
    for sent_text in sentence_batch:
        batch_results.append({"sentence": sent_text, "candidates": None})
        print(f'Processed {len(batch_results)} sentences without candidates')
    return batch_results


_WS = re.compile(r"\s+")
_PUNCT = str.maketrans({
    "“":"\"", "”":"\"", "‘":"'", "’":"'",
    "–":"-", "—":"-", "−":"-", "…":"...",
    "\u00A0":" ", "\u2009":" ", "\u200A":" ", "\u200B":" ",
})


def _canon(s: str) -> str:
    # same canonicalization as for gold: normalize, collapse space, lowercase
    s = unicodedata.normalize("NFKC", s).translate(_PUNCT)
    s = _WS.sub(" ", s).strip().lower()
    return s


def _dedupe_bioc_index(
    bioc_index: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Merge entries for identical sentences (by canonical form), prefer the first
    exact text seen as the key, union spans, remove duplicate spans.
    """
    # map canon -> (original_text_key, merged_spans_list)
    tmp: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}

    def _span_key(sp):
        # dedupe by geometry + text + mapped type
        return (int(sp["start"]), int(sp["end"]),
                sp.get("text",""), sp.get("type"))

    for sent_text, spans in bioc_index.items():
        c = _canon(sent_text)
        if c not in tmp:
            tmp[c] = (sent_text, [])
        key_text, merged = tmp[c]
        # extend
        merged.extend(spans)

    deduped: Dict[str, List[Dict[str, Any]]] = {}
    for c, (orig_text, merged) in tmp.items():
        # remove duplicate spans
        seen = set()
        uniq = []
        for sp in merged:
            k = _span_key(sp)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(sp)
        # sort deterministic
        uniq.sort(key=lambda s: (int(s["start"]), -(int(s["end"]) - int(s["start"]))))
        deduped[orig_text] = uniq
    return deduped



def gazetteer_candidates(sentence: str, gazetteer: GazetteerMatcher) -> list[dict]:
    if gazetteer is None:
        return []
    hits = gazetteer.match(sentence)  # already returns start/end/text/type/source/rule_id
    # normalize + tag provenance
    out = []
    for h in hits:
        out.append({
            "start_char": int(h["start_char"]),
            "end_char": int(h["end_char"]),
            "text": h["text"],
            "type": h.get("type"),              # filename-as-type (e.g., 'Mountain', 'Biome', ...)
            "source": "gazetteer",
            "meta": {"rule_id": h.get("rule_id"), "backend": h.get("source")},
        })
    return out


# Helpers for candidate fusion

def _normalize_text_min(s: str) -> str:
    # light canonicalization for dedupe
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return " ".join(s.split())


def _iou_tuple(a: tuple[int,int], b: tuple[int,int]) -> float:
    (s1,e1),(s2,e2) = a,b
    inter = max(0, min(e1,e2) - max(s1,s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def _noisy_or(ps: list[float]) -> float:
    prod = 1.0
    for p in ps:
        p = max(0.0, min(1.0, p))
        prod *= (1.0 - p)
    return 1.0 - prod


def _map_type(raw_type: Optional[str], tmap: Dict[str,str]) -> Optional[str]:
    if not raw_type:
        return None
    return tmap.get(raw_type, raw_type)


def fuse_candidates(
    cands: List[Dict[str,Any]],
    type_map: Dict[str,str] | None = None,
    source_weights: Dict[str,float] | None = None,
    iou_thr: float = IOU_THR,
) -> List[Dict[str,Any]]:
    """
    Merge duplicate spans coming from different sources into a single candidate.
    Keeps provenance and aggregates type hypotheses via noisy-OR.
    """
    if not cands:
        return []

    tmap = type_map or {}
    sweights = source_weights or DEFAULT_SOURCE_WEIGHTS

    # normalize + copy
    items = []
    for c in cands:
        s = int(c["start_char"]); e = int(c["end_char"])
        if e <= s:  # guard
            continue
        txt = c["text"]
        src = c.get("source", "unknown")
        score = float(c.get("score", 1.0))
        mapped = _map_type(c.get("type"), tmap)
        items.append({
            "text": txt,
            "text_norm": _normalize_text_min(txt),
            "start_char": s, "end_char": e,
            "type": mapped,
            "source": src,
            "score": score,
            "meta": c.get("meta", {})
        })

    # cluster by (text_norm) and IoU
    items.sort(key=lambda x: (x["text_norm"], x["start_char"], -(x["end_char"]-x["start_char"])))
    clusters: list[list[dict]] = []
    for it in items:
        placed = False
        for cl in clusters:
            if cl[0]["text_norm"] != it["text_norm"]:
                continue
            # IoU with cluster leader OR exact match with any member
            if _iou_tuple((it["start_char"], it["end_char"]),
                    (cl[0]["start_char"], cl[0]["end_char"])) >= iou_thr or \
               any(m["start_char"]==it["start_char"] and m["end_char"]==it["end_char"] for m in cl):
                cl.append(it); placed = True; break
        if not placed:
            clusters.append([it])

    fused: list[dict] = []

    for cl in clusters:
        # choose canonical span: prefer longest; tiebreak by highest score
        can = max(cl, key=lambda x: (x["end_char"] - x["start_char"], x["score"]))
        s,e = can["start_char"], can["end_char"]
        text = can["text"]

        # evidence per type
        type_scores: dict[str, list[float]] = defaultdict(list)
        sources = []

        for m in cl:
            t = m["type"]
            w = sweights.get(m["source"], sweights.get("unknown", 0.6))
            if t:
                type_scores[t].append(w * m["score"])
            sources.append({
                "name": m["source"], "type": t,
                "score": m["score"], "meta": m.get("meta", {})
            })

        type_votes = {t: _noisy_or(vs) for t,vs in type_scores.items()}

        aliases = []
        for m in cl:
            if (m["start_char"], m["end_char"]) != (s,e):
                aliases.append({
                    "text": m["text"],
                    "start_char": m["start_char"],
                    "end_char": m["end_char"],
                    "source": m["source"],
                    "type": m["type"]
                })

        fused.append({
            "text": text,
            "start_char": s, "end_char": e,
            "span_policy": "longest",
            "type_votes": type_votes,      # multiple hypotheses preserved
            "sources": sources,            # provenance
            "aliases": aliases             # optional / informative
        })

    return fused


#--------- Fact classification ---------


def split_into_clauses_spacy(sentence: str, nlp) -> list[dict]:
    """
    Return list of clauses as dicts: {"start": int, "end": int, "text": str}
    Uses dependency-based subtrees around clausal heads; falls back to punctuation splits.
    """
    doc = nlp(sentence)
    if len(doc) == 0:
        return [{"start":0, "end":len(sentence), "text":sentence}]

    # collect clause roots (finite verbs, conj roots, clausal dependents)
    clause_heads = []
    for t in doc:
        if t.dep_ in ("ROOT","conj","parataxis","ccomp","xcomp","advcl","acl","relcl"):
            clause_heads.append(t)

    spans = []
    for h in clause_heads or [doc[0]]:
        subtree_tokens = list(h.subtree)
        s = min(tok.idx for tok in subtree_tokens)
        e = max(tok.idx + len(tok) for tok in subtree_tokens)
        spans.append((s,e))

    # also add punctuation-based splits for ';' ':' '—'
    for m in re.finditer(r"[;:]", sentence):
        cut = m.start()
        spans.extend([(0, cut), (cut+1, len(sentence))])

    # normalize/merge overlapping spans
    spans = sorted(set(spans), key=lambda x: (x[0], x[1]))
    merged = []
    for s,e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # de-overlap hard: clip to [0, len]
    out = []
    for s,e in merged:
        s = max(0, s); e = min(len(sentence), e)
        if e > s:
            out.append({"start": s, "end": e, "text": sentence[s:e]})

    # fallback: whole sentence if weird
    if not out:
        out = [{"start":0, "end":len(sentence), "text":sentence}]
    return out


def classify_clauses_llm(client, model_name, model_type, clauses: list[dict]) -> list[str]:
    sys_prompt = (
        "Label each clause as FACT, SPECULATIVE, or UNSURE.\n"
        "- FACT: reports observed findings/results.\n"
        "- SPECULATIVE: hypotheses, assumptions, theories, plans, hedged claims (may/might/could), open questions.\n"
        "Return STRICT JSON list of labels, same order as input."
    )
    if model_type == "qwen3-32B":
        sys_prompt = "<no_think/>\n\n" + sys_prompt
    payload = {"clauses": [c["text"] for c in clauses]}

    content = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content": f"{sys_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
            temperature=0.0, max_tokens=256
        ).choices[0].message.content
    if model_type in ("qwen3-32B","gpt4o"):
        content = remove_thinking_blocks(content)
    return json.loads(content)


def classify_sentences_fact_llm(client, model_name, model_type, sents: List[str]) -> List[str]:
    """
    Return one of {'FACT','SPECULATIVE','UNSURE'} per sentence using the chat model.
    Batched naively (small lists) to keep it simple.
    """
    out = []
    sys_prompt = (
        "Label each sentence as FACT, SPECULATIVE, or UNSURE.\n"
        "- FACT: reports findings, measurements, observed results.\n"
        "- SPECULATIVE: hypotheses, assumptions, theories, plans, hedged claims (may/might/could), open questions.\n"
        "Return STRICT JSON list of labels, same order as input."
    )
    if model_type == "qwen3-32B":
        sys_prompt = "<no_think/>\n\n" + sys_prompt

    for chunk_start in range(0, len(sents), 20):
        chunk = sents[chunk_start:chunk_start+20]
        payload = {"sentences": chunk}
        content = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content": f"{sys_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
            temperature=0.0,
            max_tokens=256
        ).choices[0].message.content
        if model_type in ("qwen3-32B","gpt4o"):
            content = remove_thinking_blocks(content)
        labels = json.loads(content)
        out.extend(labels)
    return out




def main():
    global client, model_name


    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with .txt documents")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (one object per document)")
    ap.add_argument("--model_type", choices=["qwen3-4B", "qwen3-32B", "gpt4o", "biomistral-7b-awq"], default="gpt4o", help="LLM model to use")
    ap.add_argument("--use_chunks", action="store_true", help="Use noun phrase chunks as candidates")
    ap.add_argument("--max_sents_per_doc", type=int, default=999999, help="Cap sentences per doc (debug)")
    ap.add_argument("--sample_every", type=int, default=1, help="Process every Nth sentence (e.g., 5 to sample)")
    ap.add_argument("--batch_size", type=int, default=15, help="Batch size for processing")
    ap.add_argument("--max_workers", type=int, default=4, help="Max worker threads")

    # ==== Candidate source switch ====
    ap.add_argument("--candidates_from", choices=["chunks", "ner", "bioc"], default="ner",
                    help="Generate LLM candidates from spaCy noun chunks or NER predictions.")

    # ==== BioC ====
    ap.add_argument("--bioc_candidates_dir", type=str, default=None,
                    help = "Directory with BioC JSON files. We will extract sentence->spans as candidates.")
    ap.add_argument("--use_bioc", action="store_true", help="When using NER candidates, also add BioC spans if available.")

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

    ap.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for span merging.")

    args = ap.parse_args()

    set_base_seed(args.base_seed)
    type_map = {
          "Biomes": "HABITAT",
          "Biota": "TAXON",
          "Mountains": "HABITAT",
          "MountainRange": "HABITAT",
          "geography": "HABITAT",
          "ENV_FEATURE": "ENV_FEATURE",
          "POPULATION": "POPULATION",
          "TAXON": "TAXON",
          "LOCATION": "LOCATION",
          "HABITAT": "HABITAT",
          "THREAT": "THREAT"
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
        client, model_name = get_openai_client(args.model_type)
        print(f"Using {args.model_type} model: {model_name}")
    except Exception as e:
        print(f"Error initializing {args.model_type} client: {e}", file=sys.stderr)
        sys.exit(1)

    if args.gaz_dir:
        gaz_rules = load_gaz_rules_from_dir(
            dir_path=args.gaz_dir,
        )
        gaz_matcher = GazetteerMatcher(gaz_rules)
        print(f"[gazetteer] Loaded {len(gaz_rules)} rules from {args.gaz_dir} "
              f"(phrase={len(gaz_matcher.phrase_rules)}, regex={len(gaz_matcher.regex_rules)})")
    else:
        gaz_matcher = None

    # Validate spaCy model
    if args.use_chunks:
        try:
            test_nlp = spacy.load(args.spacy_model)
            if "parser" not in test_nlp.pipe_names:
                print("WARNING: spaCy parser not enabled; noun_chunks may be empty.", file=sys.stderr)
        except OSError:
            print(f"spaCy model '{args.spacy_model}' not found. Install with: python -m spacy download {args.spacy_model}",
                  file=sys.stderr)
            sys.exit(1)

    Path(os.path.dirname(args.out_jsonl) or ".").mkdir(parents=True, exist_ok=True)

    # Load BioC candidates if requested
    bioc_index = None
    if args.candidates_from == "bioc":
        if not args.bioc_candidates_dir:
            print("Provide --bioc_candidates_dir for candidates_from=bioc", file=sys.stderr)
            sys.exit(1)
        bioc_index = _load_bioc_index_from_dir(args.bioc_candidates_dir)
        print(f"Loaded BioC candidates for {len(bioc_index)} sentences from {args.bioc_candidates_dir}")

        before = len(bioc_index)
        bioc_index = _dedupe_bioc_index(bioc_index)
        after = len(bioc_index)
        print(f"[BioC] deduped unique sentences: {after} (from {before})")

    # NER runtime (loaded once)
    ner_runtime = None
    if args.candidates_from == "ner":
        if not args.ner_model_dir:
            print("Provide --ner_model_dir for candidates_from=ner", file=sys.stderr)
            sys.exit(1)
        ner_runtime = _load_ner(args.ner_model_dir)

    docs_written = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for doc_id, text in read_txt_files(args.in_dir):
            if docs_written > 1:  # Debug limit
                break

            # Split text into lines (one sentence per line)
            lines = text.strip().split('\n')
            sentences = [line.strip() for line in lines if line.strip()]

            out_sents = []
            print(f"Processing {len(sentences)} sentences from lines")

            total_batches = (len(sentences) + args.batch_size - 1) // args.batch_size

            for bidx, i in enumerate(range(0, len(sentences), args.batch_size), start=1):
                batch = sentences[i:i + args.batch_size]

                # # ---- FACT FILTER: classify sentences in this batch
                # if args.fact_filter == "llm":
                #     fact_labels = classify_sentences_fact_llm(client, model_name, args.model_type, batch)
                # else:
                #     fact_labels = ["FACT"] * len(batch)  # no filtering
                #
                # def _passes(label: str) -> bool:
                #     if args.fact_filter == "off":
                #         return True
                #     if args.fact_filter_policy == "strict":
                #         return label == "FACT"
                #     # lenient: FACT or UNSURE
                #     return label in ("FACT", "UNSURE")

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

                # === FACT-GATE at clause level ===
                if args.fact_filter != "off":
                    nlp = get_spacy_model(args.spacy_model)  # already used elsewhere
                    for idx, cr in enumerate(candidate_results):
                        sent = cr["sentence"]
                        cands = cr.get("candidates")
                        if not cands:
                            cr["notes"] = (cr.get("notes", "") + "|fact_gate:no_cands").strip("|")
                            continue

                        # derive sentence/clause labels
                        if args.fact_filter_scope == "sentence":
                            def _passes_label(lbl: str) -> bool:
                                return (lbl == "FACT") if args.fact_filter_policy == "strict" else (
                                            lbl in ("FACT", "UNSURE"))

                            lbl = classify_clauses_llm(client, model_name, args.model_type,
                                                             [{"text": sent, "start": 0, "end": len(sent)}])[0]
                            if not _passes_label(lbl):
                                cr["candidates"] = None
                                cr["notes"] = (cr.get("notes", "") + f"|fact_gate_sentence:{lbl}").strip("|")
                                continue
                            else:
                                cr["notes"] = (cr.get("notes", "") + f"|fact_gate_sentence:{lbl}").strip("|")
                                continue  # sentence-level all pass

                        # clause scope
                        clauses = split_into_clauses_spacy(sent, nlp)
                        labels = classify_clauses_llm(client, model_name, args.model_type, clauses)

                        def _passes(lbl: str) -> bool:
                            return (lbl == "FACT") if args.fact_filter_policy == "strict" else (
                                        lbl in ("FACT", "UNSURE"))

                        # build filtered candidate list
                        filtered = []
                        for c in cands:
                            # take span midpoint to assign to a clause
                            mid = int((int(c["start_char"]) + int(c["end_char"])) / 2)
                            # find clause containing midpoint
                            cl_idx = next((i for i, cl in enumerate(clauses) if cl["start"] <= mid < cl["end"]), None)
                            lbl = labels[cl_idx] if cl_idx is not None else "UNSURE"
                            if _passes(lbl):
                                filtered.append(c)
                        if filtered:
                            cr["candidates"] = filtered
                            # keep audit: per-clause labels
                            cr["notes"] = (cr.get("notes", "") + f"|fact_gate_clauses:{labels}").strip("|")
                        else:
                            cr["candidates"] = None
                            cr["notes"] = (cr.get("notes", "") + f"|fact_gate_all_filtered:{labels}").strip("|")

                # # wipe ALL candidates (NER/NP/gaz/gold) for sentences that do not pass the fact gate
                # for j, cr in enumerate(candidate_results):
                #     label = fact_labels[j]
                #     if not _passes(label):
                #         # Remove all candidates; also stash a note for transparency
                #         cr["candidates"] = None
                #         cr["notes"] = f"filtered_out_by_fact_gate:{label}"
                #     else:
                #         cr["notes"] = f"fact_gate:{label}"

                assert len(candidate_results) == len(batch), \
                    f"Lost sentences in candidate building: {len(batch)} -> {len(candidate_results)}"

                # llm_results = call_llm_batch_two_path(client, model_name, args.model_type, args.few_shot,
                #                                       candidate_results, lock_over_iou=0.5)
                cond = args.llm_condition
                if cond == "C0":
                    dec = dict(temperature=0.0, top_p=1.0, presence_penalty=0.0)
                    llm_results = call_llm_batch_two_path(client, model_name, args.model_type, args.few_shot,
                                                          candidate_results, lock_over_iou=args.iou_thr, #todo: T = 0.4
                                                          decoding=dec)
                elif cond == "C1":
                    llm_results = run_C1_vanilla(client, model_name, args.model_type, args.few_shot,
                                                 candidate_results, T=args.passes, lock_over_iou=args.iou_thr)
                elif cond == "C2":
                    llm_results = run_C2_diverse(client, model_name, args.model_type, args.few_shot,
                                                 candidate_results, T=args.passes, lock_over_iou=args.iou_thr)
                elif cond == "C3":
                    llm_results = run_C3_critique_revise(client, model_name, args.model_type, args.few_shot,
                                                         candidate_results, T=args.passes, lock_over_iou=args.iou_thr)
                elif cond == "C4":
                    llm_results = run_C4_self_consistency(client, model_name, args.model_type, args.few_shot,
                                                          candidate_results, K=args.samples_k, lock_over_iou=args.iou_thr)
                else:
                    raise ValueError(f"Unknown --llm_condition {cond}")




                if len(llm_results) != len(candidate_results):
                    raise RuntimeError(f"LLM results mismatch: {len(llm_results)} vs {len(candidate_results)}")

                for idx_in_batch, (spacy_result, llm_result) in enumerate(zip(candidate_results, llm_results)):
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
                    out_sents.append(sentence_data)

                print(f"Processed batch {bidx}/{total_batches}")

                # Write results
            rec = {
                "doc_id": doc_id,
                "sentences": out_sents,
                "config": {
                    "model_type": args.model_type,
                    "model_name": model_name,
                    "use_chunks": args.use_chunks
                }
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_written += 1
            print(f"Completed document {doc_id} with {len(out_sents)} sentences")


    mode_str = "with chunks" if args.use_chunks else "without chunks"
    print(f"Done. Processed {docs_written} documents using {args.model_type} {mode_str} and wrote to {args.out_jsonl}")


if __name__ == "__main__":
    main()