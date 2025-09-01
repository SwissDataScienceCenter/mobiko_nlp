#!/usr/bin/env python3
"""
NER utilities: create train/dev/test splits and evaluate a trained model on a test set.

Subcommands
-----------
1) split  – make randomized splits from a JSONL of records
2) test   – run inference on a JSONL and report seqeval metrics
3) predict – dump predictions (tags) to JSONL/CoNLL

Input JSONL format (one record per line) as produced by your converter:
{
  "doc_id": str,
  "sentence_id": int | null,
  "text": str,
  "tokens": [str, ...],
  "tags": ["B-…"|"I-…"|"O", ...]
}

"""


import argparse, json, os, random, sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification

from train_ner import NerDataset, read_jsonl, create_bio_labels, build_label_list

# seqeval metrics
try:
    from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
except Exception:
    classification_report = None
    precision_score = recall_score = f1_score = None


# ---------------------
# IO helpers
# ---------------------

def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------
# Split
# ---------------------

def split_dataset(records: List[Dict[str, Any]], train_ratio=0.8, dev_ratio=0.1, seed=13) -> Tuple[List, List, List]:
    assert 0 < train_ratio < 1 and 0 <= dev_ratio < 1 and train_ratio + dev_ratio < 1
    test_ratio = 1.0 - train_ratio - dev_ratio

    # Optional: group by doc_id to avoid leaking sentences across splits
    by_doc = {}
    for r in records:
        did = r.get("doc_id") or "_"
        by_doc.setdefault(did, []).append(r)

    rng = random.Random(seed)
    doc_ids = list(by_doc.keys())
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(round(n * train_ratio))
    n_dev = int(round(n * dev_ratio))
    train_ids = set(doc_ids[:n_train])
    dev_ids = set(doc_ids[n_train:n_train+n_dev])
    test_ids = set(doc_ids[n_train+n_dev:])

    train = [r for did in train_ids for r in by_doc[did]]
    dev   = [r for did in dev_ids   for r in by_doc[did]]
    test  = [r for did in test_ids  for r in by_doc[did]]

    return train, dev, test


def percentage_no_entities(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    n_none = 0
    for r in records:
        tags = r.get("tags", [])
        if all(t == "O" for t in tags):
            n_none += 1
    return 100.0 * n_none / len(records)

# ---------------------
# Dataset for inference
# ---------------------

# class NerDataset(Dataset):
#     def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length=256):
#         self.examples = examples
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.examples)
#
#     def __getitem__(self, idx):
#         ex = self.examples[idx]
#         tokens = ex["tokens"]
#         enc = self.tokenizer(
#             tokens,
#             is_split_into_words=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_offsets_mapping=False,
#         )
#         return {k: torch.tensor(v) for k,v in enc.items()}


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, id2label: Dict[int, str]):
    preds = np.argmax(predictions, axis=2)
    batch_preds = []
    for pred in preds:
        batch_preds.append([id2label[int(p)] for p in pred])
    return batch_preds


def strip_ignored(pred_tags: List[str], example: Dict[str, Any], tokenizer) -> List[str]:
    """Reduce subword predictions back to word level by keeping the first subword's tag."""
    enc = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, return_offsets_mapping=False)
    word_ids = enc.word_ids()
    out = []
    seen = set()
    for tag, wi in zip(pred_tags, word_ids):
        if wi is None:
            continue
        if wi not in seen:
            out.append(tag)
            seen.add(wi)
    # ensure same length as tokens
    if len(out) != len(example["tokens"]):
        # pad/truncate conservatively
        out = (out + ["O"]*len(example["tokens"]))[:len(example["tokens"])]
    return out


# ---------------------
# Commands
# ---------------------

def cmd_split(args):
    recs = read_jsonl(args.input_jsonl)
    print(args.out_dir)
    train, dev, test = split_dataset(recs, args.train_ratio, args.dev_ratio, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "dev.jsonl"), dev)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)
    print(f"Wrote: train={len(train)}, dev={len(dev)}, test={len(test)} → {args.out_dir}")
    print(f"No-entity sentences: train={percentage_no_entities(train):.1f}%, dev={percentage_no_entities(dev):.1f}%, test={percentage_no_entities(test):.1f}%")


def load_labels(model_dir: str) -> List[str]:
    labels_path = os.path.join(model_dir, "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    # fallback: try from config
    cfg = AutoConfig.from_pretrained(model_dir)
    id2label = getattr(cfg, "id2label", None) or {}
    if id2label:
        return [id2label[i] for i in range(len(id2label))]
    raise RuntimeError("labels.txt not found and id2label missing in config")


def load_model(model_dir: str, labels: List[str]=None):
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    config = AutoConfig.from_pretrained(model_dir, num_labels=len(labels), id2label=id2label, label2id=label2id)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, config=config)
    model.eval()
    return tokenizer, model, label2id, id2label


def run_inference(model_dir: str, records: List[Dict[str, Any]], labels: List,
                  batch_size=16, max_length=512):

    tokenizer, model, label2id, id2label = load_model(model_dir, labels)

    # Sanity: classifier dims vs labels
    num_labels_model = model.config.num_labels
    if len(labels) != num_labels_model or len(id2label) != num_labels_model:
        raise RuntimeError(
            f"Label size mismatch: model has {num_labels_model} labels, "
            f"but labels list has {len(labels)} and id2label has {len(id2label)}."
        )

    # id2label should cover 0..num_labels-1
    for i in range(num_labels_model):
        if i not in id2label:
            raise RuntimeError(f"id2label missing index {i}")

    ds = NerDataset(records, tokenizer, label2id, max_length=max_length)

    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    all_preds = []

    # debug_limit = 5  # Process only 5 batches for debugging
    # with torch.no_grad():
    #     for i, batch in enumerate(loader):
    #         if i >= debug_limit:
    #             break
    #         for k in batch:
    #             batch[k] = batch[k].to(model.device)
    #         logits = model(**batch).logits.cpu().numpy()
    #         all_preds.extend(list(np.argmax(logits, axis=2)))

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(model.device)
                print(batch[k].shape)
            print(len(batch))
            logits = model(**batch).logits.cpu().numpy()
            all_preds.extend(list(np.argmax(logits, axis=2)))

    # Map back to tags per example (first subword per word)
    out_tags = []
    for pred_ids, ex in zip(all_preds, records):
        pred_strs = [id2label[int(i)] for i in pred_ids]
        word_tags = strip_ignored(pred_strs, ex, tokenizer)
        out_tags.append(word_tags)
    return out_tags


def entity_level_metrics(y_true: List[List[str]], y_pred: List[List[str]]):
    if f1_score is None:
        return {"precision": None, "recall": None, "f1": None, "report": "seqeval not installed"}
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=3)
    }


def cmd_test(args):
    records = read_jsonl(args.test_jsonl)
    y_true = [r["tags"] for r in records]

    provided = create_bio_labels()
    labels = build_label_list(records, provided_labels=provided)
    if "O" not in labels:
        labels = ["O"] + labels

    y_pred = run_inference(args.model_dir, records, labels, batch_size=args.batch_size, max_length=args.max_length)

    metrics = entity_level_metrics(y_true, y_pred)
    print({k:v for k,v in metrics.items() if k != "report"})
    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            f.write(metrics["report"]) 
    else:
        print(metrics["report"])

    # Optionally write JSONL with predictions
    if args.out_jsonl:
        preds = []
        for r,tags in zip(records, y_pred):
            rr = dict(r)
            rr["pred_tags"] = tags
            preds.append(rr)
        write_jsonl(args.out_jsonl, preds)


def to_conll(tokens: List[str], tags: List[str]) -> str:
    lines = []
    for tok, tag in zip(tokens, tags):
        lines.append(f"{tok} {tag}")
    return "\n".join(lines) + "\n\n"


def cmd_predict(args):
    records = read_jsonl(args.input_jsonl)
    y_pred = run_inference(args.model_dir, records, batch_size=args.batch_size, max_length=args.max_length)
    # Write JSONL
    if args.out_jsonl:
        preds = []
        for r,tags in zip(records, y_pred):
            rr = dict(r)
            rr["pred_tags"] = tags
            preds.append(rr)
        write_jsonl(args.out_jsonl, preds)
    # Write CoNLL
    if args.out_conll:
        with open(args.out_conll, "w", encoding="utf-8") as f:
            for r,tags in zip(records, y_pred):
                f.write(to_conll(r["tokens"], tags))


# ---------------------
# CLI
# ---------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split")
    sp.add_argument("--input_jsonl", required=True)
    sp.add_argument("--out_dir", required=True)
    sp.add_argument("--train_ratio", type=float, default=0.8)
    sp.add_argument("--dev_ratio", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=13)
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser("test")
    sp.add_argument("--model_dir", required=True)
    sp.add_argument("--test_jsonl", required=True)
    sp.add_argument("--batch_size", type=int, default=16)
    sp.add_argument("--max_length", type=int, default=512)
    sp.add_argument("--report_path", default=None)
    sp.add_argument("--out_jsonl", default=None)
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("predict")
    sp.add_argument("--model_dir", required=True)
    sp.add_argument("--input_jsonl", required=True)
    sp.add_argument("--out_jsonl", default=None)
    sp.add_argument("--out_conll", default=None)
    sp.add_argument("--batch_size", type=int, default=16)
    sp.add_argument("--max_length", type=int, default=512)
    sp.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
