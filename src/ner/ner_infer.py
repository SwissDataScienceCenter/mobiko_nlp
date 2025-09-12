# ner_infer.py
from typing import List, Dict, Any, Optional, Tuple
import os, json
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification
)
import torch.nn.functional as F


def _best_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _load_labels_from_model_dir(model_dir: str) -> List[str]:
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "id2label" in cfg:
            m = {int(k): v for k, v in cfg["id2label"].items()}
            return [m[i] for i in sorted(m.keys())]
        if "label2id" in cfg:
            m = {k: int(v) for k, v in cfg["label2id"].items()}
            return [k for k,_ in sorted(m.items(), key=lambda kv: kv[1])]
    # last resort
    return ["O"]


def _align_first_subtoken(batch_records: List[Dict[str, Any]],
                          encodings, pred_ids: List[List[int]], id2label: Dict[int, str]) -> List[List[str]]:
    """
    Keep first subtoken tag per word and pad/truncate to word count.
    """
    word_id_batches = encodings.word_ids if hasattr(encodings, "word_ids") else None
    if callable(word_id_batches):
        word_id_batches = [encodings.word_ids(i) for i in range(len(batch_records))]

    out = []
    for rec, wp_ids, pred in zip(batch_records, word_id_batches, pred_ids):
        tags, last_w = [], None
        for pid, w in zip(pred, wp_ids):
            if w is None or w == last_w:
                continue
            tags.append(id2label[int(pid)])
            last_w = w
        n_words = len(rec.get("tokens", []))
        if len(tags) != n_words:
            tags = tags[:n_words] + ["O"] * max(0, n_words - len(tags))
        out.append(tags)
    return out


def _bio_to_spans(tags: List[str], token_spans: List[Tuple[int,int]], text: str) -> List[Dict[str, Any]]:
    spans, active_type, start_i = [], None, None
    for i, tag in enumerate(tags):
        if not tag or tag == "O":
            if active_type is not None:
                s,_ = token_spans[start_i]
                _,e = token_spans[i-1]
                spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
                active_type, start_i = None, None
            continue
        pref, typ = (tag.split("-",1)+[""])[:2] if "-" in tag else ("B", tag)
        if pref == "B" or (active_type and typ != active_type):
            if active_type is not None:
                s,_ = token_spans[start_i]
                _,e = token_spans[i-1]
                spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
            active_type, start_i = typ, i
    if active_type is not None:
        s,_ = token_spans[start_i]
        _,e = token_spans[len(token_spans)-1]
        spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
    return spans


def _ws_tokenize_with_offsets(text: str):
    tokens, spans, i, n = [], [], 0, len(text)
    while i < n:
        # skip spaces
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        spans.append((i,j))
        i = j
    return tokens, spans



class NerInferencer:
    def __init__(self, model_dir: str, dtype: str = "auto", pad_to_max_length: bool = True):
        self.labels = _load_labels_from_model_dir(model_dir)
        if "O" not in self.labels:
            self.labels = ["O"] + self.labels
        self.label2id = {l:i for i,l in enumerate(self.labels)}
        self.id2label = {i:l for l,i in self.label2id.items()}
        self.o_id = self.label2id["O"]
        self.num_labels = len(self.labels)
        self._entity_mask = torch.ones(self.num_labels, dtype=torch.bool)
        self._entity_mask[self.o_id] = False

        self.device = _best_device()
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        cfg = AutoConfig.from_pretrained(
            model_dir,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            return_dict=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, config=cfg).eval()

        # dtype move
        if dtype == "auto":
            if self.device == "cuda":
                if torch.cuda.is_bf16_supported():
                    self.model.to(self.device, dtype=torch.bfloat16)
                else:
                    self.model.to(self.device, dtype=torch.float16)
            else:
                self.model.to(self.device)
        elif dtype == "bf16":
            self.model.to(self.device, dtype=torch.bfloat16)
        elif dtype == "fp16":
            self.model.to(self.device, dtype=torch.float16)
        else:
            self.model.to(self.device)

        self.pad_to_max_length = pad_to_max_length
        self.collator = DataCollatorForTokenClassification(self.tok, padding=True)
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True


    def _predict_ids_with_threshold(
            self,
            logits: torch.Tensor,
            entity_threshold: float | None = None,
            entity_bias: float | None = None,
    ) -> list[list[int]]:
        """
        logits: [B, T, C] (same dtype/device as model output)
        entity_threshold:
          - None  -> plain argmax over all labels
          - float -> choose max over entity labels; if max_entity_prob <= thr => 'O'
        """

        if entity_bias is not None:
            mask = self._entity_mask.to(logits.device)
            logits = logits.clone()  # avoid modifying in-place
            logits[..., mask] += entity_bias

        if entity_threshold is None:
            return logits.argmax(dim=-1).to("cpu").tolist()

        # clone mask to logits device once
        entity_mask = self._entity_mask.to(logits.device)

        # probs in float32 for numerical stability (keeps bf16 speed for matmuls; softmax cast is cheap)
        probs = F.softmax(logits.to(torch.float32), dim=-1)  # [B,T,C]
        entity_probs = probs[..., entity_mask]  # [B,T,C-1]
        max_entity_prob, max_entity_idx = entity_probs.max(dim=-1)  # [B,T]

        # map entity argmax indices back to original label ids
        entity_ids = torch.arange(self.num_labels, device=logits.device)[entity_mask]  # [C-1]
        chosen_entity_ids = entity_ids[max_entity_idx]  # [B,T]

        # threshold: if max_entity_prob <= thr => O
        o_tensor = torch.full_like(chosen_entity_ids, fill_value=self.o_id)
        pred_ids = torch.where(max_entity_prob > entity_threshold, chosen_entity_ids, o_tensor)
        return pred_ids.to("cpu").tolist()


    def predict_word_tags_for_tokenized(
        self,
        records: List[Dict[str, Any]],
        batch_size: int = 128,
        max_length: int = 256,
        entity_threshold: float | None = None,
    ) -> List[List[str]]:
        """
        records: [{"tokens":[..]}, ...]  →  BIO word-level tags per record
        """
        # HF encodes list[list[str]] when is_split_into_words=True
        enc = self.tok(
            [r["tokens"] for r in records],
            is_split_into_words=True,
            return_tensors=None,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=False
        )

        # Build per-sample dicts so we can collate
        sample_dicts = [{k: (v[i] if hasattr(v, "__getitem__") else v) for k,v in enc.items()}
                        for i in range(len(records))]

        preds = []
        for i in range(0, len(sample_dicts), batch_size):
            batch = self.collator(sample_dicts[i:i+batch_size])
            batch = {k: v.to(self.model.device, non_blocking=True) for k,v in batch.items()}
            out = self.model(**batch)
            pred_ids = self._predict_ids_with_threshold(out.logits, entity_threshold)
            preds.extend(pred_ids)

        # Need word_ids to collapse subtokens → words
        # Re-run to get word_ids in the same order (cheap)
        enc2 = self.tok(
            [r["tokens"] for r in records],
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            max_length=max_length
        )
        word_tags = _align_first_subtoken(records, enc2, preds, self.id2label)
        return word_tags


    def predict_spans_for_sentences(
        self,
        sentences: List[str],
        batch_size: int = 128,
        max_length: int = 256,
        entity_threshold: float | None = None,
        entity_bias: float | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        sentences: [str]  →  list of span dicts per sentence (char indices)
        """
        tokens_offsets = [_ws_tokenize_with_offsets(s) for s in sentences]
        words_list = [w for (w, _) in [t for t in [t[0:2] for t in tokens_offsets]]]

        enc = self.tok(
            words_list,
            is_split_into_words=True,
            return_tensors=None,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=False
        )

        sample_dicts = [{k: (v[i] if hasattr(v, "__getitem__") else v) for k,v in enc.items()}
                        for i in range(len(sentences))]

        preds = []
        for i in range(0, len(sample_dicts), batch_size):
            batch = self.collator(sample_dicts[i:i+batch_size])
            batch = {k: v.to(self.model.device, non_blocking=True) for k,v in batch.items()}
            out = self.model(**batch)
            pred_ids = self._predict_ids_with_threshold(out.logits, entity_threshold, entity_bias)
            preds.extend(pred_ids)

        # collapse to word-level tags, then BIO→spans with the original offsets
        enc2 = self.tok(
            words_list,
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            max_length=max_length
        )
        pseudo_records = [{"tokens": w} for w in words_list]
        word_tags = _align_first_subtoken(pseudo_records, enc2, preds, self.id2label)

        spans_per_sent = []
        for (sent, (words, offsets)), tags in zip(zip(sentences, tokens_offsets), word_tags):
            if len(tags) < len(words):
                tags = tags + ["O"] * (len(words) - len(tags))
            elif len(tags) > len(words):
                tags = tags[:len(words)]
            spans_per_sent.append(_bio_to_spans(tags, offsets, sent))
        return spans_per_sent
