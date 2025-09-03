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

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification

from train_ner import NerDataset, read_jsonl, build_label_list
from labels import EntityLabel, build_bio_labels

# seqeval metrics
try:
    from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
except Exception:
    classification_report = None
    precision_score = recall_score = f1_score = None

ENTITY_PREFIXES = ("B-", "I-")

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


def split_dataset_stratified(records: List[Dict[str, Any]], train_ratio=0.8, dev_ratio=0.1, seed=13) -> Tuple[List, List, List]:
    # Build label list and map to indices (ignore 'O')

    labels = build_label_space(records)
    print(labels)
    lab2id = {l:i for i,l in enumerate(labels)}
    # multi-label presence per sentence
    X_bin = [
        { lab2id[l] for l in extract_present_classes(r) if l in lab2id }
        for r in records
    ]

    # run iterative stratification into 3 parts
    splits = iterative_stratify(X_bin, [train_ratio, dev_ratio, 1.0-train_ratio-dev_ratio], seed=seed)

    # enforce presence (best-effort)
    enforce_class_presence(splits, X_bin, num_labels=len(labels))
    train = [records[i] for i in splits[0]]
    dev   = [records[i] for i in splits[1]]
    test  = [records[i] for i in splits[2]]
    return train, dev, test, labels


def percentage_no_entities(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    n_none = 0
    for r in records:
        tags = r.get("tags", [])
        if all(t == "O" for t in tags):
            n_none += 1
    return 100.0 * n_none / len(records)


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

# ----------------------
# Stream the split
# ----------------------\


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



def write_streamed_splits(input_path: str, offsets: np.ndarray, assign: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        os.path.join(out_dir, "train.jsonl"),
        os.path.join(out_dir, "dev.jsonl"),
        os.path.join(out_dir, "test.jsonl")
    ]
    fhs = [open(p, "wb") for p in paths]
    try:
        with open(input_path, "rb") as f:
            for i, off in enumerate(offsets):
                f.seek(off, io.SEEK_SET)
                line = f.readline()
                s = int(assign[i])
                fhs[s].write(line)
    finally:
        for fh in fhs: fh.close()


# ---------------------
# Iterative stratification
# ---------------------

def extract_present_classes(rec):
    """Return a set of EntityLabel values present in this sentence."""
    present = set()
    for t in rec.get("tags", []):
        if t and t != "O":
            _, typ = t.split("-", 1)
            try:
                present.add(EntityLabel[typ])
            except KeyError:
                # skip unknowns
                print(f'Warning: unknown label type "{typ}" in record {rec.get("doc_id")}', file=sys.stderr)
                continue
    return present


def iterative_stratify(X_bin, ratios, seed=13):
    """
    Iterative stratification (Sechidis et al. 2011) for multi-label data.

    Args:
        X_bin: List of sets, where each set contains label indices present in that sample
        ratios: List of floats summing to 1.0 (e.g., [0.8, 0.1, 0.1])
        seed: Random seed for reproducibility

    Returns:
        List of lists containing sample indices for each split
    """

    rnd = random.Random(seed)
    n_samples = len(X_bin)
    n_splits = len(ratios)

    # Calculate target sizes for each split
    target_sizes = [int(round(n_samples * ratio)) for ratio in ratios]
    target_sizes = _adjust_target_sizes(target_sizes, n_samples)

    # Initialize tracking variables
    remaining_quota = target_sizes[:]
    remaining_samples = list(range(n_samples))
    splits = [[] for _ in range(n_splits)]

    # Calculate label statistics
    n_labels = max((max(sample_labels) if sample_labels else -1) for sample_labels in X_bin) + 1
    label_needs = _calculate_label_needs(X_bin, ratios, n_labels, n_splits)
    labels_by_rarity = _get_labels_by_rarity(X_bin, n_labels)


    while remaining_samples:
        # pick next label to place: rarest label with unmet need
        target_label = _find_next_target_label(label_needs, labels_by_rarity)

        # if all label needs are met, just fill by remaining quota
        if target_label is None:
            # All label needs satisfied, distribute remaining samples by quota
            _distribute_remaining_samples(remaining_samples, splits, remaining_quota, rnd)
            break

        _assign_samples_for_label(
            target_label, remaining_samples, X_bin, splits,
            remaining_quota, label_needs, rnd
        )

    return splits


def _adjust_target_sizes(target_sizes, n_samples):
    """Adjust target sizes to ensure they sum to n_samples."""
    while sum(target_sizes) > n_samples:
        max_idx = target_sizes.index(max(target_sizes))
        target_sizes[max_idx] -= 1
    while sum(target_sizes) < n_samples:
        min_idx = target_sizes.index(min(target_sizes))
        target_sizes[min_idx] += 1
    return target_sizes


def _calculate_label_needs(X_bin, ratios, n_labels, n_splits):
    """Calculate desired label distribution across splits."""
    total_label_counts = [0] * n_labels
    for sample_labels in X_bin:
        for label in sample_labels:
            total_label_counts[label] += 1

    label_needs = [[0] * n_labels for _ in range(n_splits)]
    for split_idx in range(n_splits):
        for label_idx in range(n_labels):
            label_needs[split_idx][label_idx] = int(
                round(total_label_counts[label_idx] * ratios[split_idx])
            )

    return label_needs


def _get_labels_by_rarity(X_bin, n_labels):
    """Sort labels by rarity (rarest first)."""
    label_counts = [0] * n_labels
    for sample_labels in X_bin:
        for label in sample_labels:
            label_counts[label] += 1

    return sorted(range(n_labels), key=lambda label: label_counts[label] if label_counts[label] > 0 else 10 ** 9)


def _find_next_target_label(label_needs, labels_by_rarity):
    """Find the rarest label that still has unmet needs."""
    for label in labels_by_rarity:
        if sum(need_row[label] for need_row in label_needs) > 0:
            return label
    return None


def _distribute_remaining_samples(remaining_samples, splits, remaining_quota, rnd):
    """Distribute remaining samples based on remaining quota."""
    rnd.shuffle(remaining_samples)
    for sample_idx in remaining_samples:
        split_idx = max(range(len(remaining_quota)), key=lambda k: remaining_quota[k])
        splits[split_idx].append(sample_idx)
        remaining_quota[split_idx] -= 1


def _assign_samples_for_label(target_label, remaining_samples, X_bin, splits,
                              remaining_quota, label_needs, rnd):
    """Assign samples containing the target label to appropriate splits."""
    candidates = [idx for idx in remaining_samples if target_label in X_bin[idx]]

    if not candidates:
        # No samples with this label, mark needs as satisfied
        for split_needs in label_needs:
            split_needs[target_label] = 0
        return

    rnd.shuffle(candidates)

    for sample_idx in candidates[:]:  # Use slice copy to modify during iteration
        if sample_idx not in remaining_samples:
            continue

        # Choose split with highest need for this label, then by remaining quota
        best_split = max(
            range(len(splits)),
            key=lambda k: (label_needs[k][target_label], remaining_quota[k])
        )

        if remaining_quota[best_split] <= 0:
            continue

        # Assign sample to split
        splits[best_split].append(sample_idx)
        remaining_quota[best_split] -= 1
        remaining_samples.remove(sample_idx)

        # Update label needs for all labels in this sample
        for label in X_bin[sample_idx]:
            if label_needs[best_split][label] > 0:
                label_needs[best_split][label] -= 1


def enforce_class_presence(splits, X_bin, num_labels):
    """
    If any class missing from a split, move one sample with that class from another split.
    """
    K = len(splits)

    for label_idx in range(num_labels):
        # check which splits contrain the label
        splits_with_label = [
            any(label_idx in X_bin[sample_idx] for sample_idx in splits[split_idx])
            for split_idx in range(K)
        ]
        for target_split in range(K):
            if splits_with_label[target_split]:
                continue

            # Find a donor split that can safely give away a sample with this label
            donor_split, sample_to_move = _find_donor_sample(
                splits, X_bin, label_idx, target_split, K
            )

            if donor_split is not None:
                splits[donor_split].remove(sample_to_move)
                splits[target_split].append(sample_to_move)
            # else: Label is extremely rare (single instance), cannot redistribute


def _find_donor_sample(splits, X_bin, label_idx, target_split, K):
    """
    Find a donor split and sample that can be moved without losing label coverage.

    Returns:
        Tuple of (donor_split_idx, sample_idx) or (None, None) if no donor found
    """
    for potential_donor in range(K):
        if potential_donor == target_split:
            continue

        # Find a candidate sample with the required label
        candidate_sample = next(
            (sample_idx for sample_idx in splits[potential_donor]
             if label_idx in X_bin[sample_idx]),
            None
        )

        if candidate_sample is None:
            continue

        # Check if donor would still have this label after removal
        would_retain_label = any(
            label_idx in X_bin[sample_idx]
            for sample_idx in splits[potential_donor]
            if sample_idx != candidate_sample
        )

        if would_retain_label:
            return potential_donor, candidate_sample

    return None, None


def build_label_space(records):
    labs = set()
    for r in records:
        labs |= extract_present_classes(r)
    return sorted(labs, key=lambda l: l.value)


# ---------------------
# Commands
# ---------------------


def cmd_split(args):

    ratios = [args.train_ratio, args.dev_ratio, 1.0 - args.train_ratio - args.dev_ratio]
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        print("Ratios must be non-negative and sum to 1.", file=sys.stderr); sys.exit(2)


    # if getattr(args, "stream", False):
    # 1) first pass: offsets + bitmasks
    offsets, masks, types, doc_ids = _scan_jsonl_bitmasks_and_offsets(args.input_jsonl,
                                                                             args.tags_field, args.strict_types)


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
            with open(args.input, "rb") as f:
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

    write_streamed_splits(args.input_jsonl, offsets, assign, args.out_dir)

    # Quick report
    def pct_no_ent(idx):
        return 100.0 * float(np.count_nonzero(masks[idx] == 0)) / max(1, len(idx))

    splits = [np.where(assign == s)[0] for s in (0, 1, 2)]
    names = ["train", "dev", "test"]
    print(f"Wrote: train={len(splits[0])}, dev={len(splits[1])}, test={len(splits[2])} → {args.out_dir}")

    for s, name in enumerate(names):
        present = [t for k, t in enumerate(types) if np.any(((masks[splits[s]] >> k) & 1))]
        print(f"{name:>5} no-entity %: {pct_no_ent(splits[s]):5.1f} | classes: {', '.join(present) if present else '—'}")



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

    provided = build_bio_labels()
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
