# entity_extraction/candidates/ner.py

from __future__ import annotations

import json
import os
from typing import List, Dict, Any

from src.ner.labels import build_bio_labels
from src.ner.ner_infer import NerInferencer


def load_labels_from_model(model_dir: str) -> List[str]:
    """
    Try to infer label set from HF config.json, fallback to build_bio_labels().
    """
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "id2label" in cfg:
            id2label = {int(k): v for k, v in cfg["id2label"].items()}
            return [id2label[i] for i in sorted(id2label.keys())]
        if "label2id" in cfg:
            label2id = {k: int(v) for k, v in cfg["label2id"].items()}
            return [lab for lab, _ in sorted(label2id.items(), key=lambda kv: kv[1])]

    provided = build_bio_labels()
    return sorted(provided)


def load_ner(model_dir: str) -> NerInferencer:
    """
    Backwards-compat shim to create the NerInferencer.
    """
    infer = NerInferencer(model_dir, dtype="auto")
    return infer


def predict_spans(
    ner_runtime: NerInferencer,
    sentences: List[str],
    batch_size: int,
    max_length: int,
    entity_threshold: float = 0.25,
    entity_bias: float = 0.25,
) -> List[List[Dict[str, Any]]]:
    """
    Thin wrapper so the rest of the code doesn't depend on NerInferencer details.
    """
    return ner_runtime.predict_spans_for_sentences(
        sentences=sentences,
        batch_size=batch_size,
        max_length=max_length,
        entity_threshold=entity_threshold,
        entity_bias=entity_bias,
    )
