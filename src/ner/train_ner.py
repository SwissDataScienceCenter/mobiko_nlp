#!/usr/bin/env python3
"""
Train a token-classification (NER) model on your JSONL exported from create_conll_from_bioc_integrated.py.

Input JSONL format (one record per line):
{
  "doc_id": str,
  "sentence_id": int | null,
  "text": str,
  "tokens": [str, ...],
  "spans": [ {"start": int, "end": int, "text": str, "label": str, ...}, ... ],
  "tags": ["B-LABEL" | "I-LABEL" | "O", ...]   # length == len(tokens)
}

This script:
  - builds the label set from training data (or from a file);
  - tokenizes with a HF tokenizer (is_split_into_words=True);
  - aligns word-level BIO tags to subword tokens (first subword keeps B-, rest become I-);
  - trains a Transformer (BERT/RoBERTa/etc.) with a class-weighted loss (overrides compute_loss);
  - evaluates with seqeval (precision/recall/F1 per entity + overall);
  - saves model, tokenizer, and label mapping to output_dir.

Usage example:
python train_ner.py \
  --train_jsonl /mnt/data/biodiv_train.jsonl \
  --valid_jsonl /mnt/data/biodiv_dev.jsonl \
  --model_name roberta-base \
  --output_dir /mnt/data/ner_roberta \
  --epochs 10 --lr 3e-5 --batch_size 16 --grad_accum 2 --fp16 \
  --class_weight_power 0.5 --o_weight 0.2

"""

import argparse, json, math, os, random, sys, time, mlflow
from transformers.integrations import MLflowCallback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from preprocess.create_conll_from_bioc_integrated import EntityLabel

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting NER training script - importing dependencies")


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
)

# seqeval for NER metrics
try:
    from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
except Exception:
    classification_report = None
    precision_score = recall_score = f1_score = None


# --------------------------
# Data utilities
# --------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def create_bio_labels():
    """Create BIO tagging scheme labels from entity types."""
    labels = ["O"]  # Outside tag

    # Add B- and I- prefixes for each entity type
    for entity in EntityLabel:
        labels.append(f"B-{entity.value}")
        labels.append(f"I-{entity.value}")

    return labels


def build_label_list(train_examples: List[Dict[str, Any]], provided_labels: List[str] = None) -> List[str]:
    if provided_labels:
        labels = list(provided_labels)
    else:
        uniq = set()
        for ex in train_examples:
            for t in ex.get("tags", []):
                if t is not None:
                    uniq.add(t)
        if "O" not in uniq:
            uniq.add("O")
        # stable order: O first, then B-*, then I-* alphabetically
        b_tags = sorted([x for x in uniq if x.startswith("B-")])
        i_tags = sorted([x for x in uniq if x.startswith("I-")])
        other  = sorted([x for x in uniq if x not in b_tags and x not in i_tags and x != "O"])
        labels = ["O"] + b_tags + i_tags + other
    return labels


@dataclass
class EncodedBatch:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[List[int]]


class NerDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, label2id: Dict[str, int], max_length: int = 256, label_all_subwords: bool = True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_subwords = label_all_subwords

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens: List[str] = ex["tokens"]
        tags: List[str] = ex["tags"]
        assert len(tokens) == len(tags), f"tokens/tags length mismatch at idx={idx}: {len(tokens)} vs {len(tags)}"

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=False,
        )
        # align labels to wordpieces
        word_ids = enc.word_ids()
        labels = []
        prev_word_id = None
        for wi in word_ids:
            if wi is None:
                labels.append(-100)
                continue
            tag = tags[wi]
            if not tag or tag == "":
                tag = "O"
            if wi != prev_word_id:  # first subword of this token
                labels.append(self.label2id[tag])
            else:
                if self.label_all_subwords and tag != "O":
                    # turn continuation subwords into I-<TYPE>
                    if tag.startswith("B-"):
                        cont = "I-" + tag[2:]
                    elif tag.startswith("I-"):
                        cont = tag
                    else:
                        cont = "O"
                    labels.append(self.label2id.get(cont, self.label2id["O"]))
                else:
                    labels.append(-100)
            prev_word_id = wi

        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels)
        return item


# --------------------------
# Weighted loss Trainer
# --------------------------

# class WeightedTokenTrainer(Trainer):
#     def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
#         logits = outputs.get("logits")  # (bsz, seq, num_labels)
#         # flatten
#         bsz, seqlen, nlab = logits.shape
#         loss = F.cross_entropy(
#             logits.view(-1, nlab),
#             labels.view(-1),
#             weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
#             ignore_index=-100,
#             reduction="mean",
#         )
#         return (loss, outputs) if return_outputs else loss


class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        # Handle both tokenizer and processing_class for compatibility
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')

        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")  # (bsz, seq, num_labels)
        # flatten
        bsz, seqlen, nlab = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, nlab),
            labels.view(-1),
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=-100,
            reduction="mean",
        )
        return (loss, outputs) if return_outputs else loss

# --------------------------
# Metrics
# --------------------------

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, id2label: Dict[int, str]):
    preds = np.argmax(predictions, axis=2)
    batch_preds, batch_labels = [], []
    for pred, lab in zip(preds, label_ids):
        true_pred = []
        true_lab = []
        for p, l in zip(pred, lab):
            if l == -100:
                continue
            true_pred.append(id2label[int(p)])
            true_lab.append(id2label[int(l)])
        batch_preds.append(true_pred)
        batch_labels.append(true_lab)
    return batch_preds, batch_labels


def build_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(p):
        if f1_score is None:
            return {}
        preds, labels = align_predictions(p.predictions, p.label_ids, id2label)
        return {
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "f1": f1_score(labels, preds),
        }
    return compute_metrics


# --------------------------
# Helpers
# --------------------------

def compute_class_weights(train_ds: NerDataset, num_labels: int, o_id: int, power: float = 0.5, o_weight: float = None) -> torch.Tensor:
    """Compute inverse-frequency class weights over visible tokens (labels != -100).
    power in [0..1], lower = more uniform. Optionally override 'O' weight.
    """
    counts = torch.zeros(num_labels, dtype=torch.float)
    for i in range(len(train_ds)):
        y = train_ds[i]["labels"]
        for v in y.tolist():
            if v != -100:
                counts[v] += 1
    counts = torch.clamp(counts, min=1.0)
    inv = (1.0 / counts) ** power
    weights = inv / inv.mean()
    if o_weight is not None:
        weights[o_id] = o_weight
    return weights


def save_labels(output_dir: str, labels: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for l in labels:
            f.write(l + "\n")


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--valid_jsonl", required=True)
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fp16", action="store_true")

    # class weighting
    ap.add_argument("--class_weight_power", type=float, default=0.5)
    ap.add_argument("--o_weight", type=float, default=None, help="Override weight for 'O' label (e.g., 0.2)")
    # mlflow
    ap.add_argument("--mlflow_experiment", type=str, default="ner", help="MLflow experiment name")
    ap.add_argument("--mlflow_tracking_uri", type=str, default="http://localhost:5000",
                    help="MLflow tracking URI; if omitted uses env or local ./mlruns")
    ap.add_argument("--mlflow_tags", type=str, default="",
                    help='Comma-separated tags like: project=mobiko,task=BioDiv-NER')

    args = ap.parse_args()
    set_seed(args.seed)

    # --- MLflow setup ---
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    # If not set here, MLflow will use $MLFLOW_TRACKING_URI or default to local ./mlruns
    # export MLFLOW_TRACKING_URI=http://localhost:5000
    mlflow.set_experiment(args.mlflow_experiment)

    # Parse tags "k=v,k2=v2"
    run_tags = {}
    if args.mlflow_tags.strip():
        for kv in args.mlflow_tags.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                run_tags[k.strip()] = v.strip()

    # Create a human-friendly run name
    _run_name = f"{args.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    # NOTE: We’ll rely on HF's built-in MLflow reporting; we’ll still log extra params/artifacts manually.

    # Load data
    train = read_jsonl(args.train_jsonl)
    valid = read_jsonl(args.valid_jsonl)
    logger.info("Loaded %d train and %d valid examples", len(train), len(valid))




    # Label list
    provided = create_bio_labels()
    labels = build_label_list(train, provided_labels=provided)
    print(labels)
    if "O" not in labels:
        labels = ["O"] + labels
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}

    logger.info("Using %d labels: %s", len(labels), labels)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, add_prefix_space=True)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)

    logger.info("Model loaded: %s", args.model_name)

    # Datasets
    train_ds = NerDataset(train, tokenizer, label2id, max_length=args.max_length)
    valid_ds = NerDataset(valid, tokenizer, label2id, max_length=args.max_length)

    # Class weights
    weights = compute_class_weights(train_ds, num_labels=len(labels), o_id=label2id["O"], power=args.class_weight_power, o_weight=args.o_weight)
    label_info = {labels[i]: round(float(w),3) for i,w in enumerate(weights)}
    logger.info(f"Class weights: {label_info}")

    # Training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        logging_steps=50,
        seed=args.seed,
        dataloader_drop_last=False,
        report_to=["mlflow"],
        run_name=_run_name,
    )



    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = WeightedTokenTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(id2label),
        class_weights=weights,
    )

    trainer.add_callback(MLflowCallback())

    logger.info("Starting training")

    for i, ex in enumerate(train[:3]):
        logger.info("Sample %d: tokens=%d, tags=%d", i, len(ex.get("tokens", [])), len(ex.get("tags", [])))

    # --- One-time parameter logging ---
    with mlflow.start_run(run_name=_run_name, nested=False) as run:
        # Core configs
        mlflow.set_tags(run_tags)
        mlflow.log_params({
            "model_name": args.model_name,
            "train_examples": len(train),
            "valid_examples": len(valid),
            "seed": targs.seed,
            "epochs": targs.num_train_epochs,
            "per_device_train_batch_size": targs.per_device_train_batch_size,
            "per_device_eval_batch_size": targs.per_device_eval_batch_size,
            "learning_rate": targs.learning_rate,
            "weight_decay": targs.weight_decay,
            "warmup_ratio": targs.warmup_ratio if hasattr(targs, "warmup_ratio") else None,
            "gradient_accumulation_steps": targs.gradient_accumulation_steps,
            "fp16": targs.fp16 if hasattr(targs, "fp16") else False,
            "bf16": targs.bf16 if hasattr(targs, "bf16") else False,
            "max_seq_length": args.max_length if hasattr(args, "max_length") else None,
            "label_all_tokens": getattr(args, "label_all_tokens", None),
        })

        # Log label map and class weights (JSON)
        try:
            mlflow.log_text(json.dumps(id2label, indent=2), "artifacts/id2label.json")
            mlflow.log_text(json.dumps(label2id, indent=2), "artifacts/label2id.json")
        except Exception:
            pass
        try:
            mlflow.log_text(json.dumps(class_weights_dict, indent=2), "artifacts/class_weights.json")
        except Exception:
            pass

        # Record data paths
        try:
            mlflow.log_params({
                "train_jsonl": os.path.abspath(args.train_jsonl),
                "valid_jsonl": os.path.abspath(args.valid_jsonl),
                "output_dir": os.path.abspath(args.output_dir),
            })
        except Exception:
            pass


        # --- Train as usual (HF will stream loss/metrics to MLflow) ---
        train_result = trainer.train()

        # Log final eval (HF will already log epoch metrics; we also dump a final JSON for convenience)
        try:
            metrics = trainer.evaluate()
            mlflow.log_metrics({f"final_{k}": float(v) for k, v in metrics.items()})
            mlflow.log_text(json.dumps(metrics, indent=2), "artifacts/final_eval_metrics.json")
        except Exception:
            pass

        # Save and log model/tokenizer artifacts (best or final)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        save_labels(args.output_dir, labels)

        try:
            # Log minimal important files
            for fname in ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json"]:
                fpath = os.path.join(args.output_dir, fname)
                if os.path.exists(fpath):
                    mlflow.log_artifact(fpath, artifact_path="model")
        except Exception:
            pass

    logger.info("Model, tokenizer, and labels saved to %s", args.output_dir)

    # Pretty report (optional)
    if classification_report is not None:
        preds = trainer.predict(valid_ds)
        y_pred, y_true = align_predictions(preds.predictions, preds.label_ids, id2label)
        print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    main()
