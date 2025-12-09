# entity_extraction/fact_filter.py
from __future__ import annotations
from typing import List, Dict, Any
import json
import re

from llm.client import remove_thinking_blocks
from candidates.chunk import get_spacy_model


def split_into_clauses_spacy(sentence: str, nlp) -> List[Dict[str, Any]]:
    doc = nlp(sentence)
    if len(doc) == 0:
        return [{"start": 0, "end": len(sentence), "text": sentence}]

    clause_heads = [
        t
        for t in doc
        if t.dep_ in ("ROOT", "conj", "parataxis", "ccomp", "xcomp", "advcl", "acl", "relcl")
    ]
    spans = []
    for h in clause_heads or [doc[0]]:
        subtree_tokens = list(h.subtree)
        s = min(tok.idx for tok in subtree_tokens)
        e = max(tok.idx + len(tok) for tok in subtree_tokens)
        spans.append((s, e))

    for m in re.finditer(r"[;:]", sentence):
        cut = m.start()
        spans.extend([(0, cut), (cut + 1, len(sentence))])

    spans = sorted(set(spans), key=lambda x: (x[0], x[1]))
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    out = []
    for s, e in merged:
        s = max(0, s)
        e = min(len(sentence), e)
        if e > s:
            out.append({"start": s, "end": e, "text": sentence[s:e]})
    if not out:
        out = [{"start": 0, "end": len(sentence), "text": sentence}]
    return out


def classify_clauses_llm(client, model_type, clauses: List[Dict[str, Any]]) -> List[str]:
    sys_prompt = (
        "Label each clause as FACT, SPECULATIVE, or UNSURE.\n"
        "- FACT: reports observed findings/results.\n"
        "- SPECULATIVE: hypotheses, assumptions, theories, plans, hedged claims, open questions.\n"
        "Return STRICT JSON list of labels, same order as input."
    )
    if "qwen3-32B" in model_type:
        sys_prompt = "<no_think/>\n\n" + sys_prompt
    payload = {"clauses": [c["text"] for c in clauses]}
    content = client.call(
        messages=[{"role": "user", "content": f"{sys_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
        temperature=0.0,
        max_tokens=256,
    )
    return json.loads(content)


def classify_sentences_fact_llm(client, model_type, sents: List[str]) -> List[str]:
    out: List[str] = []
    sys_prompt = (
        "Label each sentence as FACT, SPECULATIVE, or UNSURE.\n"
        "- FACT: reports findings, measurements, observed results.\n"
        "- SPECULATIVE: hypotheses, assumptions, theories, plans, hedged claims, open questions.\n"
        "Return STRICT JSON list of labels, same order as input."
    )
    if "qwen3-32B" in model_type:
        sys_prompt = "<no_think/>\n\n" + sys_prompt

    for chunk_start in range(0, len(sents), 20):
        chunk = sents[chunk_start : chunk_start + 20]
        payload = {"sentences": chunk}
        content = client.call(
            messages=[{"role": "user", "content": f"{sys_prompt}\n\nUser input: {json.dumps(payload, ensure_ascii=False)}"}],
            temperature=0.0,
            max_tokens=256,
        )
        labels = json.loads(content)
        out.extend(labels)
    return out
