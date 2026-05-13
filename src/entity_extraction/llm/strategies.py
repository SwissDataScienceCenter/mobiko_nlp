# entity_extraction/llm/strategies.py
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
import regex as re
from collections import Counter

from .client import remove_thinking_blocks
from src.entity_extraction.llm.prompts_two_output import (
    DEFAULT_SYSTEM_PROMPT_NEW,
    NO_CHUNK_CANDIDATE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_FEW_SHOT,
)
from span_utils import (
    fix_span_indices,
    merge_spans,
    consensus_merge_by_type,
    iou_tuple,
    overlaps_any,
    dedupe_overlaps_longest
)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def safe_json_from_llm(raw: str, kind: str = "extract") -> Dict[str, Any]:
    """
    Best-effort JSON parsing for LLM outputs.

    Returns a dict with keys: accepted, rejected, missing, notes.
    Never raises; on total failure, returns empty structure with a note.
    """
    cleaned = (raw or "").strip()

    # Strip ```... fences if present
    if cleaned.startswith("```"):
        nl = cleaned.find("\n")
        if nl != -1:
            cleaned = cleaned[nl + 1 :]
        if "```" in cleaned:
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

    # Try direct parse
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        # Try to extract first {...} block
        m = _JSON_OBJECT_RE.search(cleaned)
        if m:
            frag = m.group(0)
            try:
                obj = json.loads(frag)
            except Exception:
                obj = None

    if obj is None or not isinstance(obj, dict):
        return {
            "accepted": [],
            "rejected": [],
            "missing": [],
            "notes": f"{kind}_json_parse_failed",
        }

    accepted = obj.get("accepted", []) or []
    rejected = obj.get("rejected", []) or []
    missing = obj.get("missing", []) or []
    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "missing": missing,
        "notes": notes,
    }


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


def _is_ner(cand: dict) -> bool:
    return any([s.get("name") == "ner" for s in cand.get("sources", [])])


def _as_accepted(c: dict, forced_type: Optional[str] = None) -> dict:
    return {
        "text": c.get("text") if c.get("text") else c.get("mention_text"),
        "type": forced_type if forced_type else c.get("type"),
        "start_char": int(c["start_char"]),
        "end_char": int(c["end_char"]),
        "source": c.get("source", "unknown"),
    }


def call_llm_batch(
        client,
        model_type: str,
        few_shot: bool,
        requests: List[Dict],
        decoding: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
) -> List[Dict]:
    """Process multiple LLM requests in parallel (I/O-bound, thread-safe)."""
    decoding = decoding or {}
    temperature = float(decoding.get("temperature", 0.7))
    top_p = float(decoding.get("top_p", 0.95))
    presence_penalty = float(decoding.get("presence_penalty", 0.0))

    if model_type in ["qwen3-32B", "gpt4o", "qwen3-32B-vllm", "qwen3-35B-vllm"]:
        max_tokens = 1024
    else:
        max_tokens = 500

    def _process_one(req: Dict) -> Dict:
        sentence = req["sentence"]
        candidates = req.get("candidates")

        if candidates is None:
            system_prompt = NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
            user_payload = {"sentence": sentence}
        else:
            if few_shot:
                system_prompt = SYSTEM_PROMPT_FEW_SHOT
            else:
                system_prompt = DEFAULT_SYSTEM_PROMPT_NEW
            cand_objs = [
                {"text": c["text"].strip(), "start_char": c["start_char"], "end_char": c["end_char"]}
                for c in candidates
            ]
            user_payload = {"sentence": sentence, "candidates": cand_objs}

        if "qwen3-35B" in model_type or "qwen3-32B" in model_type:
            system_prompt = f"<no_think/>\n\n{system_prompt}"

        full_prompt = f"{system_prompt}\n\nUser input: {json.dumps(user_payload, ensure_ascii=False)}"

        try:
            print(f'Sending LLM request for sentence: {sentence[:50]}... with {len(candidates) if candidates else 0} candidates')
            content = client.call(
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                timeout=800,
            )
            print('LLM response received.')
            llm_result = safe_json_from_llm(content, kind="extract")
            for category in ["accepted", "missing", "rejected"]:
                if category in llm_result:
                    llm_result[category] = fix_span_indices(
                        llm_result[category], sentence, candidates)
            return llm_result
        except Exception as e:
            print(f"Error calling LLM for sentence: {sentence[:50]}...: {e}")
            return {
                "accepted": [], "rejected": [], "missing": [],
                "notes": f"llm_error: {repr(e)}"
            }

    if max_workers <= 1 or len(requests) <= 1:
        return [_process_one(req) for req in requests]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(requests))) as executor:
        return list(executor.map(_process_one, requests))


def call_llm_batch_two_path(
    client,
    model_type: str,
    few_shot: bool,
    requests: List[Dict],
    lock_over_iou: float = 0.5,
    decoding: Optional[Dict[str, Any]] = None,
    gaz_lock: bool = False,
    max_workers: int = 4,
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

        if gaz_lock is True:
            # Identify fused gazetteer candidates
            locked = [c for c in cands if _is_gazetteer(c)]
            to_llm = [c for c in cands if not _is_gazetteer(c)]
        else:
            locked = []
            to_llm  = cands[:]

        # Materialize locked accepteds now
        lock_accepteds = []
        locked_spans = []
        for c in locked:
            t = _choose_gaz_type(c, fallback_argmax=True)
            if not t:
                # If no type could be chosen, skip locking
                continue
            lock_accepteds.append({
                "text": c.get("text") if c.get("text") else c.get("mention_text"),
                "type": t,
                "start_char": int(c["start_char"]),
                "end_char": int(c["end_char"]),
                "source": "gazetteer"
            })
            locked_spans.append((int(c["start_char"]), int(c["end_char"])))


        # print(f"Locked spans for sent: {len(lock_accepteds)}")
        # print(f"To LLM spans for sent: {len(to_llm)}")
        # Build a request for LLM with only non-gaz candidates

        if to_llm:
            filtered_reqs.append({"sentence": sent, "candidates": to_llm})
        else:
            filtered_reqs.append({"sentence": sent, "candidates": []})  # keeps alignment

        per_req_locked.append({"locked": lock_accepteds, "locked_spans": locked_spans})

    # Call your existing batch function
    llm_outs = call_llm_batch(client, model_type, few_shot, filtered_reqs, decoding, max_workers=max_workers)

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
            if not overlaps_any((int(a["start_char"]), int(a["end_char"])), locked_spans, iou_thr=lock_over_iou):
                merged_accepted.append(a)

        merged_accepted = dedupe_overlaps_longest(
            merged_accepted,
            iou_thr=lock_over_iou,
        )

        merged_rejected = []
        for r in llm_res.get("rejected", []):
            sc = int(r.get("start_char", -1)); ec = int(r.get("end_char", -1))
            if sc >= 0 and ec >= 0 and overlaps_any((sc, ec), locked_spans, iou_thr=lock_over_iou):
                continue  # ignore rejection that conflicts with a lock
            merged_rejected.append(r)

        merged_missing = []
        for m in llm_res.get("missing", []):
            if not overlaps_any((int(m["start_char"]), int(m["end_char"])), locked_spans, iou_thr=lock_over_iou):
                merged_missing.append(m)

        # final_spans: longest over accepted ∪ missing
        final_spans = dedupe_overlaps_longest(
            merged_accepted + merged_missing,
            iou_thr=lock_over_iou,
        )

        merged_outs.append({
            "accepted": merged_accepted,
            "rejected": merged_rejected,
            "missing": merged_missing,
            "final_spans": final_spans,
            "notes": llm_res.get("notes", ""),
        })

    return merged_outs


def call_llm_batch_revision(client, model_type, requests, temperature=0.3, max_workers: int = 4):
    _system_prompt_base = """You are revising an earlier extraction.
        Return STRICT JSON:
        {"missing": ["type": ...,  "concept_text": ..., "uncertain": ..., "text": ...], "notes": "short rationale ≤160 chars"}
            Rules:
            - Consider the previous output AND the running notes.
            - DO NOT DELETE earlier accepted spans.
            - Only propose NEW or CORRECTED spans (no duplicates).
            - Prefer precise boundaries; don't re-state identical spans.
            - Use the same TYPE schema as before.
            - If uncertain about a span, set "uncertain": true and explain briefly in notes.
            - "concept_text" is a canonical form of the entity text, it does not have to be in the sentence verbatim but should be clearly linked to the surface text.
            """

    def _process_one(req: Dict) -> Dict:
        system_prompt = _system_prompt_base
        payload = {"sentence": req["sentence"], "previous": req["prev_json"],
                   "prev_notes": req.get("prev_notes", "")}
        if "qwen3-35B" in model_type or "qwen3-32B" in model_type:
            system_prompt = f"<no_think/>\n\n{system_prompt}"
        content = client.call(
            messages=[{"role": "user", "content": f"{system_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
            temperature=temperature,
            max_tokens=512,
            timeout=360,
        )
        obj = safe_json_from_llm(content, kind="extract")
        obj["missing"] = fix_span_indices(obj.get("missing", []), req["sentence"])
        return obj

    if max_workers <= 1 or len(requests) <= 1:
        return [_process_one(req) for req in requests]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(requests))) as executor:
        return list(executor.map(_process_one, requests))


def call_llm_batch_consolidate(client, model_type, requests, max_workers: int = 4):
    _system_prompt_base = """Consolidate proposed spans.
            Return STRICT JSON:
            {"accepted":[{"text":"...", "type":"...", "start_char":int, "end_char":int, "concept_text: ..., "uncertain": ...}], "rejected":[...], "missing":[], "notes":"optional"}
            Rules:
            - Merge overlapping duplicates; keep one with best boundaries.
            """

    def _process_one(req: Dict) -> Dict:
        system_prompt = _system_prompt_base
        payload = {"sentence": req["sentence"], "proposals": req["proposals"]}
        if "qwen3-35B" in model_type or "qwen3-32B" in model_type:
            system_prompt = f"<no_think/>\n\n{system_prompt}"
        content = client.call(
            messages=[{"role": "user", "content": f"{system_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
            temperature=0.0,
            max_tokens=768,
            timeout=360,
        )
        obj = json.loads(content)
        for k in ("accepted", "rejected"):
            obj[k] = fix_span_indices(obj.get(k, []), req["sentence"])
        obj["missing"] = []
        return obj

    if max_workers <= 1 or len(requests) <= 1:
        return [_process_one(req) for req in requests]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(requests))) as executor:
        return list(executor.map(_process_one, requests))


def _ablation_accept(cands: Optional[List[Dict[str,Any]]],
                     mode: str,
                     iou_thr: float = 0.5) -> Dict[str, Any]:
    """
    Return {"accepted": [...], "rejected": [], "missing": [], "notes": "..."}.
    - gaz_only: keep only gazetteer; type chosen via _choose_gaz_type()
    - ner_only: keep only NER spans; keep their 'type' as predicted
    - gaz_ner: union; when IoU>=thr with any gaz span, gazetteer wins (type+geometry)
    """
    if not cands:
        return {"accepted": [], "rejected": [], "missing": [], "notes": f"ablation:{mode}|no_cands"}

    if mode == "gaz_only":
        gaz = [c for c in cands if _is_gazetteer(c)]
        acc = []
        for g in gaz:
            t = _choose_gaz_type(g, fallback_argmax=True)
            if t:
                acc.append(_as_accepted(g, forced_type=t))
        return {"accepted": acc, "rejected": [], "missing": [], "notes": f"ablation:{mode}"}

    if mode == "ner_only":
        ner = [c for c in cands if _is_ner(c)]
        acc = [_as_accepted(n) for n in ner]
        return {"accepted": acc, "rejected": [], "missing": [], "notes": f"ablation:{mode}"}

    if mode == "gaz_ner":
        gaz = []
        gaz_spans = []
        for c in cands:
            if _is_gazetteer(c):
                t = _choose_gaz_type(c, fallback_argmax=True)
                if not t:
                    continue
                a = _as_accepted(c, forced_type=t)
                gaz.append(a)
                gaz_spans.append((a["start_char"], a["end_char"]))

        acc = list(gaz)
        for c in cands:
            if _is_gazetteer(c):
                continue
            s, e = int(c["start_char"]), int(c["end_char"])
            if any(iou_tuple((s, e), gs) >= iou_thr for gs in gaz_spans): # take both
                # conflict with gazetteer → gaz wins; skip this NER span
                continue
            if _is_ner(c):
                acc.append(_as_accepted(c))
        return {"accepted": acc, "rejected": [], "missing": [], "notes": f"ablation:{mode}"}

    # Fallback (shouldn't happen)
    return {"accepted": [], "rejected": [], "missing": [], "notes": f"ablation:{mode}|unhandled"}


# === ADD: Multi-pass runners ===

def run_C1_vanilla(client, model_type, few_shot, candidate_results, T: int = 3,
        lock_over_iou: float = 0.5, gaz_lock: bool = False, max_workers: int = 4) -> List[Dict]:
    """
    Vanilla multi-pass:
      P1: standard two-path call
      P2..T: 'revision' (missing-only) then consolidate
    """
    dec = _decoding_profile(1, "C1")
    p1 = call_llm_batch_two_path(client, model_type, few_shot,
                                 candidate_results, lock_over_iou=lock_over_iou, decoding=dec,
                                 gaz_lock=gaz_lock, max_workers=max_workers)

    if T == 1:
        return p1

    rev_reqs = [{"sentence": req["sentence"], "prev_json": out}
                for req, out in zip(candidate_results, p1)]
    p2 = call_llm_batch_revision(client, model_type, rev_reqs, temperature=0.7, max_workers=max_workers)

    merged_after_p2 = []
    for o1, o2, req in zip(p1, p2, candidate_results):
        merged = merge_spans(o1.get("accepted", []), o2.get("missing", []), iou_thr=0.5)
        merged_after_p2.append({"sentence": req["sentence"], "accepted": merged})

    if T >= 3:
        cons_requests = [{"sentence": m["sentence"], "proposals": m["accepted"]} for m in merged_after_p2]
        p3 = call_llm_batch_consolidate(client, model_type, cons_requests, max_workers=max_workers)
        return p3
    else:
        return [{"accepted": m["accepted"], "rejected": [], "missing": [], "notes": "C1-P2"} for m in merged_after_p2]


def run_C2_diverse(
    client, model_type, few_shot, candidate_results, T: int = 3,
        lock_over_iou=0.5, gaz_lock: bool = True, max_workers: int = 4) -> List[Dict]:
    """
    Diversity-forced multi-pass: each pass uses a different decoding profile,
    aggregate with NMS-style consensus by type.
    """
    pass_accepteds_per_sent = [[] for _ in candidate_results]

    for t in range(1, T+1):
        dec = _decoding_profile(t, "C2")
        out = call_llm_batch_two_path(client, model_type, few_shot,
                                      candidate_results, lock_over_iou=lock_over_iou,
                                      decoding=dec, gaz_lock=gaz_lock, max_workers=max_workers)
        for i, o in enumerate(out):
            pass_accepteds_per_sent[i].append(o.get("accepted", []))

    # consensus merge per sentence
    merged = []
    for i, req in enumerate(candidate_results):
        merged_acc = consensus_merge_by_type(pass_accepteds_per_sent[i], iou_thr=0.5)
        merged.append({"accepted": merged_acc, "rejected": [], "missing": [], "notes": f"C2-T{T}"})
    return merged


def run_C3_critique_revise(client, model_type, few_shot, candidate_results, T: int = 3,
                           lock_over_iou: float = 0.5, gaz_lock: bool = True, max_workers: int = 4) -> List[Dict]:
    dec = _decoding_profile(1, "C3")
    p = call_llm_batch_two_path(client, model_type, few_shot,
                                candidate_results, lock_over_iou=lock_over_iou, decoding=dec,
                                gaz_lock=gaz_lock, max_workers=max_workers)

    state = []
    for i, base in enumerate(p):
        state.append({
            "sentence": candidate_results[i]["sentence"],
            "accepted": base.get("accepted", []),
            "notes": (base.get("notes") or "").strip()
        })

    for t in range(2, max(2, T)):
        rev_reqs = []
        for s in state:
            prev_json = {"accepted": s["accepted"], "rejected": [], "missing": []}
            rev_reqs.append({
                "sentence": s["sentence"],
                "prev_json": prev_json,
                "prev_notes": s["notes"],
            })

        rev = call_llm_batch_revision(client, model_type, rev_reqs, temperature=0.9, max_workers=max_workers)

        for i, r in enumerate(rev):
            new_missing = r.get("missing", [])
            state[i]["accepted"] = merge_spans(state[i]["accepted"], new_missing, iou_thr=lock_over_iou)
            note_add = (r.get("notes") or "").strip()
            if note_add:
                if state[i]["notes"]:
                    state[i]["notes"] = (state[i]["notes"] + " | " + note_add)[:240]
                else:
                    state[i]["notes"] = note_add[:240]

    cons_requests = [{"sentence": s["sentence"], "proposals": s["accepted"], "prev_notes": s["notes"]}
                     for s in state]
    final = call_llm_batch_consolidate(client, model_type, cons_requests, max_workers=max_workers)

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
    client, model_type, few_shot, candidate_results, K: int = 5, lock_over_iou = 0.5,
    gaz_lock: bool = True, max_workers: int = 4,
) -> List[Dict]:
    """
    Self-consistency in one pass: sample K independent outputs, then consensus merge.
    """
    samples_per_sent = [[] for _ in candidate_results]
    grid = [
        dict(temperature=0.3, top_p=0.90, presence_penalty=0.0),
        dict(temperature=0.6, top_p=0.95, presence_penalty=0.7),
        dict(temperature=0.9, top_p=0.98, presence_penalty=1.0),
    ]
    for k in range(K):
        dec = grid[k % len(grid)]
        out = call_llm_batch_two_path(client, model_type, few_shot,
                                      candidate_results, lock_over_iou=0.5, decoding=dec,
                                      gaz_lock=gaz_lock, max_workers=max_workers)
        for i, o in enumerate(out):
            samples_per_sent[i].append(o.get("accepted", []))

    merged = []
    for i, req in enumerate(candidate_results):
        merged_acc = consensus_merge_by_type(samples_per_sent[i], iou_thr=0.5)
        merged.append({"accepted": merged_acc, "rejected": [], "missing": [], "notes": f"C4-K{K}"})
    return merged



__all__ = [
    "call_llm_batch",
    "call_llm_batch_two_path",
    "call_llm_batch_revision",
    "call_llm_batch_consolidate",
    "_ablation_accept",
    "run_C1_vanilla",
    "run_C2_diverse",
    "run_C3_critique_revise",
    "run_C4_self_consistency",
]
