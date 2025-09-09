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


def _load_labels_from_model(model_dir):
    # Prefer model's id2label/label2id so we don't scan test
    import json, os
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

def _best_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_inference_stack(model_dir, dtype="auto"):
    device = _best_device()
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # dtype selection
    if dtype == "auto":
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                model.to(device=device, dtype=torch.bfloat16)
            else:
                model.to(device=device, dtype=torch.float16)
        else:
            model.to(device)  # CPU or MPS: leave fp32
    elif dtype == "bf16":
        model.to(device=device, dtype=torch.bfloat16)
    elif dtype == "fp16":
        model.to(device=device, dtype=torch.float16)
    else:
        model.to(device=device)

    # perf knobs
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True  # speeds up varying shapes (still pad to max_length)
    return tok, model, device



def _iter_jsonl(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                yield json.loads(line)
            except Exception:
                continue


def _batched_iter(it, batch_size):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------- FAST JSON READER ---------------
try:
    import orjson as _json  # ~5-10x faster than json
    def _json_loads(s): return _json.loads(s)
    def _json_dumps(o): return _json.dumps(o).decode("utf-8")
except Exception:
    import json as _json
    def _json_loads(s): return _json.loads(s)
    def _json_dumps(o): return _json.dumps(o)



# --------------- FAST ALIGNMENT (first-subtoken) ---------------
def align_ids_to_words(batch_records, encodings, pred_ids, id2label):
    """
    Map subtoken predictions to word-level tags by taking the first subtoken’s label.
    Avoid per-subtoken Python loops as much as possible.
    """
    out = []
    word_id_batches = encodings.word_ids  # requires fast tokenizer; returns list[list[int|None]]
    # word_ids can be a callable in new HF; handle both
    if callable(word_id_batches): word_id_batches = [encodings.word_ids(i) for i in range(len(batch_records))]

    for rec, wp_ids, pred in zip(batch_records, word_id_batches, pred_ids):
        tags = []
        last_w = None
        for sub_id, w in zip(pred, wp_ids):
            if w is None or w == last_w:  # skip special tokens and continuation subtokens
                continue
            tags.append(id2label[int(sub_id)])
            last_w = w
        # Truncate/pad to number of input words (safety)
        n_words = len(rec.get("tokens", []))
        if len(tags) != n_words:
            tags = tags[:n_words] + ["O"] * max(0, n_words - len(tags))
        out.append(tags)
    return out


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


@torch.inference_mode()
def run_inference_stream(model_dir, test_jsonl, labels, out_jsonl=None,
                         batch_size=128, chunk_size=5000, max_length=256,
                         subset=None, dtype="auto"):

    tok, model, device = load_inference_stack(model_dir, dtype=dtype)

    # label mapping from model config if possible
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        # fallback to labels list
        id2label = {i: lab for i, lab in enumerate(labels)}

    # progress total
    try:
        with open(test_jsonl, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        if subset: total = min(total, subset)
    except Exception:
        total = None

    writer = None
    if out_jsonl:
        os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
        writer = open(out_jsonl, "w", encoding="utf-8")

    y_true, y_pred = [], []

    with tqdm(total=total, unit="rec", desc="Inference") as pbar:
        for chunk in _batched_iter(_iter_jsonl(test_jsonl, limit=subset), chunk_size):
            # prepare truths
            y_true.extend([rec.get("tags", []) for rec in chunk])

            # tokenize & predict in BATCHES
            chunk_preds = []
            for i in range(0, len(chunk), batch_size):
                batch = chunk[i:i+batch_size]
                tokens = [r["tokens"] for r in batch]

                enc = tok(tokens,
                          is_split_into_words=True,
                          return_tensors="pt",
                          padding="max_length",      # fixed shape speeds kernels
                          truncation=True,
                          max_length=max_length)

                for k in enc: enc[k] = enc[k].to(device, non_blocking=True)
                logits = model(**enc).logits    # [B, T, C]
                # argmax in-place-ish
                pred_ids = logits.argmax(-1).to("cpu").tolist()

                # align to word-level tags (fast)
                aligned = align_ids_to_words(batch, enc, pred_ids, id2label)
                chunk_preds.extend(aligned)

            y_pred.extend(chunk_preds)

            # stream write
            if writer:
                for rec, tags_hat in zip(chunk, chunk_preds):
                    rec_out = dict(rec)
                    rec_out["pred_tags"] = tags_hat
                    writer.write(_json_dumps(rec_out) + "\n")

            pbar.update(len(chunk))

    if writer: writer.close()
    return y_true, y_pred


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
    # get labels from model (don’t scan test)
    labels = _load_labels_from_model(args.model_dir)
    if "O" not in labels: labels = ["O"] + labels

    y_true, y_pred = run_inference_stream(
        model_dir=args.model_dir,
        test_jsonl=args.test_jsonl,
        labels=labels,
        out_jsonl=args.out_jsonl,
        batch_size=args.batch_size,          # try 128–256 on 16–24GB GPUs
        chunk_size=getattr(args, "chunk_size", 10000),
        max_length=args.max_length,          # keep modest, e.g., 256/320
        subset=getattr(args, "subset", None),
        dtype="auto",                        # bf16 on Ampere+, else fp16 on CUDA
    )

    metrics = entity_level_metrics(y_true, y_pred)
    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            f.write(metrics["report"])
    else:
        print(metrics["report"])
    print(classification_report(y_true, y_pred, digits=3))
    y_true_flat = list(chain.from_iterable(y_true))
    y_pred_flat = list(chain.from_iterable(y_pred))
    labels = sorted(set(y_true_flat) | set(y_pred_flat))  # or your fixed label order
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    print(labels)
    print(cm)


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
