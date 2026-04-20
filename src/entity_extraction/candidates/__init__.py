# entity_extraction/candidates/__init__.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

from .ner import load_ner, predict_spans
from .chunk import get_spacy_model, process_with_chunks, build_np_fallback
from .gazetteer import gazetteer_candidates
from .bioc import normalize_bioc_spans
from fusion import fuse_candidates
from src.preprocess.gazetteer_matcher import GazetteerMatcher


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
            norm_spans = normalize_bioc_spans(spans, sent_text)
            batch_results.append({"sentence": sent_text, "candidates": norm_spans if norm_spans else None})
        print(f'Processed {len(batch_results)} sentences with bioc candidates')
        return batch_results


    # All-chunks mode
    if candidates_from == "chunks":
        nlp = get_spacy_model(spacy_model)
        docs = list(nlp.pipe(sentence_batch))
        for sent_text, sent_doc in zip(sentence_batch, docs):
            cands = process_with_chunks(sent_doc)

            if gazetteer_matcher is not None:
                gz = gazetteer_candidates(sent_text, gazetteer_matcher)
                if len(gz):
                    print(gz)
                    cands.extend(gz)

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
            fallback_maps = build_np_fallback(sentence_batch, spacy_model, empty_idx)

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
                    bioc_spans = bioc_index.get(sent_text, [])
                    norm_spans = normalize_bioc_spans(bioc_spans, sent_text)
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

    # NER + chunks + gazetteer combined mode
    if candidates_from == "all":
        if ner_runtime is None:
            raise ValueError(
                "candidates_from='all' requires ner_runtime. "
                "Initialize with NerInferencer(model_dir) and pass it in."
            )

        spans_lists = ner_runtime.predict_spans_for_sentences(
            sentences=sentence_batch,
            batch_size=ner_runtime_batch_size,
            max_length=ner_max_length,
            entity_threshold=0.25,
            entity_bias=0.25
        )

        nlp = get_spacy_model(spacy_model)
        docs = list(nlp.pipe(sentence_batch))

        for i, (sent_text, sent_doc) in enumerate(zip(sentence_batch, docs)):
            ner_spans = spans_lists[i] or []
            for c in ner_spans:
                c.setdefault("source", "ner")

            chunk_spans = process_with_chunks(sent_doc)
            gz = gazetteer_candidates(sent_text, gazetteer_matcher) if gazetteer_matcher is not None else []

            cands = ner_spans + chunk_spans + gz
            if cands:
                fused = fuse_candidates(cands, type_map=type_map, source_weights=source_weights)
                batch_results.append({"sentence": sent_text, "candidates": fused})
            else:
                batch_results.append({"sentence": sent_text, "candidates": None})
        print(f'Processed {len(batch_results)} sentences with NER + chunks + gazetteer')
        return batch_results

    # candidates_from == "none"
    for sent_text in sentence_batch:
        gz = gazetteer_candidates(sent_text, gazetteer_matcher) if gazetteer_matcher is not None else []
        if gz:
            fused = fuse_candidates(gz, type_map=type_map, source_weights=source_weights)
            batch_results.append({"sentence": sent_text, "candidates": fused})
        else:
            batch_results.append({"sentence": sent_text, "candidates": None})
    print(f'Processed {len(batch_results)} sentences without candidates (gazetteer only)')
    return batch_results


__all__ = [
    "process_sentences_batch",
    "load_ner",
]
