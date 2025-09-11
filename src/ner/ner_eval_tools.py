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

from collections import defaultdict

import io
from typing import Optional


import numpy as np
import torch
from tqdm import tqdm


from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification

from train_ner import NerDataset, read_jsonl, build_label_list
from labels import EntityLabel, build_bio_labels
from ner_infer import NerInferencer


# seqeval metrics
try:
    from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
except Exception:
    classification_report = None
    precision_score = recall_score = f1_score = None

from itertools import chain
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


ENTITY_PREFIXES = ("B-", "I-")


# ---------------------
# Helpers
# ---------------------


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_conll(tokens: List[str], tags: List[str]) -> str:
    lines = []
    for tok, tag in zip(tokens, tags):
        lines.append(f"{tok} {tag}")
    return "\n".join(lines) + "\n\n"


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


def run_inference(model, tokenizer, label2id, id2label, records: List[Dict[str, Any]], labels: List,
                  batch_size=16, max_length=512):

    ds = NerDataset(records, tokenizer, label2id, max_length=max_length)

    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(model.device)
                # print(batch[k].shape)
            # print(len(batch))
            logits = model(**batch).logits.cpu().numpy()
            all_preds.extend(list(np.argmax(logits, axis=2)))

    # Map back to tags per example (first subword per word)
    out_tags = []
    for pred_ids, ex in zip(all_preds, records):
        pred_strs = [id2label[int(i)] for i in pred_ids]
        word_tags = strip_ignored(pred_strs, ex, tokenizer)
        out_tags.append(word_tags)
    return out_tags



# --------------- FAST JSON READER ---------------
try:
    import orjson as _json  # ~5-10x faster than json
    def _json_loads(s): return _json.loads(s)
    def _json_dumps(o): return _json.dumps(o).decode("utf-8")
except Exception:
    import json as _json
    def _json_loads(s): return _json.loads(s)
    def _json_dumps(o): return _json.dumps(o)


# ----------------------
# Stream the split
# ----------------------

def stratify_groups(group_masks: Dict[str, np.uint32], ratios: List[float], seed: int = 13):
    """Stratify *groups* (e.g., docs). Returns group->split_id map."""
    group_ids = list(group_masks.keys())
    masks = np.array([group_masks[g] for g in group_ids], dtype=np.uint32)
    _, assign = iterative_stratify_bitmask(masks, ratios, seed)
    return {g: int(assign[i]) for i, g in enumerate(group_ids)}


# Map EntityLabel values -> stable index [0..K-1]
def _label_index_map():
    # Preserve your canonical order by .value
    types = sorted([e for e in EntityLabel], key=lambda e: e.value)
    return {t: i for i, t in enumerate(types)}, types


def _tags_to_bitmask(tags, lab2idx):
    # Convert BIO tags to bitmask of present label types
    mask = 0
    for t in tags or []:
        if t and t != "O":
            _, typ = t.split("-", 1)
            try:
                idx = lab2idx[EntityLabel[typ]]
                mask |= (1 << idx)
            except KeyError:
                # unknown label type - ignore
                pass
    return np.uint32(mask)


def _scan_jsonl_bitmasks_and_offsets(path: str, tags_field: str, strict_types: Optional[List[str]] = None):
    """ Gather per-line offsets and bitmasks, optionally doc_id grouping."""

    offsets: List[int] = []
    masks: List[np.uint32] = []
    types_set = set(strict_types or [])

    # To allow optional grouping by document later, capture lightweight info
    doc_ids: List[Optional[str]] = []


    with open(path, "rb") as f:  # binary for exact offsets
        off = f.tell()
        line = f.readline()
        while line:
            # record offset of the start of this line
            offsets.append(off)

            # lightweight parse just for "tags"
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                # Fallback: if any line fails to parse, treat as empty
                rec = {}

            # collect types
            tags = rec.get(tags_field, []) or []
            present_types = set()
            for t in tags:
                if not t or t == "O":
                    continue
                if "-" in t:
                    _, typ = t.split("-", 1)
                else:
                    typ = t  # tolerate raw types
                present_types.add(typ)
                if strict_types is None:
                    types_set.add(typ)
            doc_ids.append(rec.get("doc_id"))

            # placeholder; we fill mask bits after we finalize the type ordering
            masks.append(present_types)
            off = f.tell()
            line = f.readline()

    types = sorted(types_set)
    lab2idx = {t: i for i, t in enumerate(types)}

    # convert sets -> bitmasks
    masks_u32 = np.zeros(len(masks), dtype=np.uint32)
    for i, s in enumerate(masks):
        if isinstance(s, set):
            m = 0
            for typ in s:
                idx = lab2idx.get(typ)
                if idx is not None:
                    m |= (1 << idx)
            masks_u32[i] = np.uint32(m)
        else:
            masks_u32[i] = np.uint32(0)
    return np.asarray(offsets, dtype=np.int64), masks_u32, types, doc_ids


def _bit_counts(masks, K):
    """Per-label counts from bitmasks.
    Example:
    masks = [3, 5, 0]

    3 = 0000011 → {TAXON, HABITAT}
    5 = 0000101 → {TAXON, ENV_FEATURE}
    0 = 0000000 → no entities
    K: number of possible entity types

    """
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        counts[k] = int(np.count_nonzero((masks >> k) & 1))
    return counts


def iterative_stratify_bitmask(masks: np.ndarray, ratios: List[float], seed: int = 13):
    """
    Iterative stratification working on uint32 masks (multi-label per row).
    Returns list of index arrays for each split. """

    rnd = np.random.RandomState(seed)
    N = masks.shape[0]
    K = max(1, int(int(masks.max()).bit_length()))

    # Target sizes
    sizes = np.rint(np.array(ratios, dtype=float) * N).astype(int)
    while sizes.sum() < N:
        sizes[np.argmin(sizes)] += 1
    while sizes.sum() > N:
        sizes[np.argmax(sizes)] -= 1

    # Desired per-label counts per split
    label_tot = _bit_counts(masks, K)
    needs = np.rint(np.outer(ratios, label_tot)).astype(int)  # [S, K]
    S = len(ratios)
    remaining = np.arange(N)
    assign = -np.ones(N, dtype=np.int8)
    quota = sizes.copy()
    per_split = [list() for _ in range(S)]

    # Rarest-first labels
    order = np.argsort(np.where(label_tot > 0, label_tot, 10**12))

    # Fast helper to pick candidates
    def has_label(arr, bit):
        return ((masks[arr] >> bit) & 1).astype(bool)

    for lbl in order:
        if label_tot[lbl] == 0:
            continue

        # candidates with this label and unassigned
        cand = remaining[has_label(remaining, lbl)]
        if cand.size == 0:
            continue
        rnd.shuffle(cand)
        for idx in cand:
            # pick split maximizing (need on this label, remaining quota)
            best, best_key = None, (-1, -1)
            for s in range(S):
                key = (needs[s, lbl], quota[s])
                if key > best_key and quota[s] > 0:
                    best_key, best = key, s
            if best is None:
                continue
            assign[idx] = best
            per_split[best].append(idx)
            quota[best] -= 1

            # decrement needs for *all* labels present in this sample
            m = masks[idx]
            k, mm = 0, m
            while mm:
                if (mm & 1) and needs[best, k] > 0:
                    needs[best, k] -= 1
                mm >>= 1; k += 1
        # update remaining
        remaining = np.where(assign < 0)[0]
        if remaining.size == 0:
            break

    # If anything left unassigned, fill by remaining quotas
    if remaining.size:
        rnd.shuffle(remaining)
        for idx in remaining:
            s = int(np.argmax(quota))
            if quota[s] == 0:
                s = int(np.argmin([len(x) for x in per_split]))
            assign[idx] = s
            per_split[s].append(idx)
            if quota[s] > 0:
                quota[s] -= 1

    return [np.array(x, dtype=np.int64) for x in per_split], assign


def write_streamed_splits(input_path: str, offsets: np.ndarray, assign: np.ndarray, out_dir: str,
                          tags_field: str = "tags"):
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        os.path.join(out_dir, "train.jsonl"),
        os.path.join(out_dir, "dev.jsonl"),
        os.path.join(out_dir, "test.jsonl")
    ]
    fhs = [open(p, "wb") for p in paths]

    counts_split = [defaultdict(int) for _ in range(3)]

    try:
        with open(input_path, "rb") as f:
            for i, off in enumerate(offsets):
                f.seek(off, io.SEEK_SET)
                line = f.readline()
                s = int(assign[i])
                fhs[s].write(line)

                try:
                    rec = json.loads(line.decode("utf-8"))
                    tags = rec.get(tags_field) or []
                except Exception:
                    tags = []

                # Accumulate BIO tag counts
                for t in tags:
                    counts_split[s][t] += 1
    finally:
        for fh in fhs:
            fh.close()

    # Save counts dicts as JSON (sorted by key for stability)
    for s, name in enumerate(["train", "dev", "test"]):
        out_counts = os.path.join(out_dir, f"{name}_label_counts.json")
        with open(out_counts, "w", encoding="utf-8") as g:
            json.dump(dict(sorted(counts_split[s].items())),
                      g, indent=2, ensure_ascii=False)

# ---------------------
# Commands
# ---------------------


def cmd_split(args):

    ratios = [args.train_ratio, args.dev_ratio, args.test_ratio]

    # if getattr(args, "stream", False):
    # 1) first pass: offsets + bitmasks
    offsets, masks, types, doc_ids = _scan_jsonl_bitmasks_and_offsets(args.input_jsonl,
                                                                      args.tags_field,
                                                                      args.strict_types)


    K = len(types)
    print(f"Discovered {K} entity types: {', '.join(types) if K else '—'}")
    print(f"Records: {len(offsets)}")

    if args.group_by:
        # Build group (e.g., doc) masks by OR over members
        group_key = args.group_by
        # We already captured doc_ids from "doc_id"; if group_by != doc_id re-read minimally
        if group_key != "doc_id":
            # Light re-scan just for the group field
            gids = []
            with open(args.input_jsonl, "rb") as f:
                line = f.readline()
                while line:
                    try:
                        rec = json.loads(line.decode("utf-8"))
                        gids.append(rec.get(group_key))
                    except Exception:
                        gids.append(None)
                    line = f.readline()
        else:
            gids = doc_ids

        group_masks = defaultdict(lambda: np.uint32(0))
        for i, g in enumerate(gids):
            if g is None:
                g = f"__ungrouped_{i}"  # isolate singletons
            group_masks[g] |= masks[i]

        g2split = stratify_groups(group_masks, ratios, args.seed)

        # Expand to per-line assignment
        assign = np.empty(len(offsets), dtype=np.int8)
        for i, g in enumerate(gids):
            if g is None: g = f"__ungrouped_{i}"
            assign[i] = g2split[g]
    else:
        # Sentence-level stratification
        _, assign = iterative_stratify_bitmask(masks, ratios, args.seed)

    write_streamed_splits(args.input_jsonl, offsets, assign, args.out_dir, "tags")

    # Quick report
    def pct_no_ent(idx):
        return 100.0 * float(np.count_nonzero(masks[idx] == 0)) / max(1, len(idx))

    splits = [np.where(assign == s)[0] for s in (0, 1, 2)]
    names = ["train", "dev", "test"]
    print(f"Wrote: train={len(splits[0])}, dev={len(splits[1])}, test={len(splits[2])} → {args.out_dir}")

    for s, name in enumerate(names):
        present = [t for k, t in enumerate(types) if np.any(((masks[splits[s]] >> k) & 1))]
        print(f"{name:>5} no-entity %: {pct_no_ent(splits[s]):5.1f} | classes: {', '.join(present) if present else '—'}")


def cmd_test_infer(args):

    infer = NerInferencer(args.model_dir, dtype="auto")
    # Stream the test file in chunks, but call a single shared predictor
    y_true, y_pred = [], []
    writer = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    # count lines for tqdm (optional; reuse your helper if you prefer)

    try:
        with open(args.test_jsonl, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
    except Exception:
        total = None

    def _iter_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def _batched(it, n):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf: yield buf

    with tqdm(total=total, unit="rec", desc="Inference") as pbar:
        for chunk in _batched(_iter_jsonl(args.test_jsonl), getattr(args, "chunk_size", 10000)):
            y_true.extend([rec.get("tags", []) for rec in chunk])
            # shared runtime on already-tokenized records
            preds = infer.predict_word_tags_for_tokenized(
                records = chunk,
                batch_size = args.batch_size,
                max_length = args.max_length
            )
            y_pred.extend(preds)
            if writer:
                for rec, tags_hat in zip(chunk, preds):
                    rr = dict(rec)
                    rr["pred_tags"] = tags_hat
                    writer.write(json.dumps(rr, ensure_ascii=False) + "\n")
            pbar.update(len(chunk))
    if writer: writer.close()
    print(classification_report(y_true, y_pred, digits=3))

    y_true_flat = list(chain.from_iterable(y_true))
    y_pred_flat = list(chain.from_iterable(y_pred))
    labels = sorted(set(y_true_flat) | set(y_pred_flat))  # or your fixed label order
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    print(labels)
    print(cm)


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
    sp.add_argument("--dev_ratio", type=float, default=0.1) # minimal dev because the dataset is huge
    sp.add_argument("--test_ratio", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=13)
    sp.add_argument("--stratify", action="store_true", help="Use iterative stratification to"
                                                            " balance classes across splits")
    sp.add_argument("--group-by", default=None, help="Optional field name to group by (e.g., doc_id)")
    sp.add_argument("--stream", action="store_true",
                    help="Memory-light stratified split: 2-pass streaming with bitmasks/offsets")
    sp.add_argument("--tags-field", default="tags", help="Field containing BIO tags (list[str])")
    sp.add_argument("--strict-types", nargs="*",
                   help="Optional fixed list of allowed entity types (e.g., TAXON HABITAT ...)")
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser("test")
    sp.add_argument("--model_dir", required=True)
    sp.add_argument("--test_jsonl", required=True)
    sp.add_argument("--batch_size", type=int, default=16)
    sp.add_argument("--max_length", type=int, default=512)
    sp.add_argument("--report_path", default=None)
    sp.add_argument("--out_jsonl", default=None)
    sp.add_argument("--chunk_size", type=int, default=5000, help="Records per streaming chunk")

    sp.set_defaults(func=cmd_test_infer)

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
