# entity_extraction/candidates/chunks.py
from __future__ import annotations
from typing import List, Dict, Any
import threading
import spacy

thread_local = threading.local()


def get_spacy_model(model_name: str):
    if not hasattr(thread_local, "nlp"):
        thread_local.nlp = spacy.load(model_name)
    return thread_local.nlp


def build_np_fallback(
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


def process_with_chunks(doc) -> List[Dict[str, Any]]:
    """Extract noun phrase candidates from a spaCy Doc."""
    cands: List[Dict[str, Any]] = []
    for np in doc.noun_chunks:
        if np.root.pos_ not in ("NOUN", "PROPN"):
            continue
        np_text = np.text.strip()
        if not np_text:
            continue
        cands.append(
            {
                "start_char": np.start_char,
                "end_char": np.end_char,
                "text": np_text,
                "source": "chunks",
            }
        )
    return cands



