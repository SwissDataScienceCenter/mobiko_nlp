import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# Loading PLM documents
def load_plm_docs(path: Path) -> List[Dict[str, Any]]:
    """
    Load SciER PLM split.
    SciER PLM files are JSONL (one doc per line).
    """
    docs = []
    with path.open() as f:
        first = f.read(1)
        if not first:
            return []
        f.seek(0)
        if first == "{":
            # Try JSONL first
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    docs.append(json.loads(line))
            except json.JSONDecodeError:
                f.seek(0)
                obj = json.load(f)
                if isinstance(obj, list):
                    docs = obj
                else:
                    docs = [obj]
        else:
            # Assume standard JSON
            f.seek(0)
            obj = json.load(f)
            if isinstance(obj, list):
                docs = obj
            else:
                docs = [obj]
    return docs


# 2. Document-level token indexing helpers
def flatten_sentences(doc: Dict[str, Any]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Flatten doc["sentences"] into a single list of tokens
    and return sentence-level doc index spans.

    Returns:
        flat_tokens: [t0, t1, ..., tN-1]
        sent_offsets: list of (start_doc_idx, end_doc_idx) per sentence
    """
    flat_tokens = []
    sent_offsets = []
    doc_idx = 0
    for sent in doc["sentences"]:
        start = doc_idx
        flat_tokens.extend(sent)
        doc_idx += len(sent)
        end = doc_idx - 1
        sent_offsets.append((start, end))
    return flat_tokens, sent_offsets


def doc_index_to_sent_pos(doc_idx: int,
                          sent_offsets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Map doc-level token index to (sent_id, pos_in_sent).

    Returns (-1, -1) if not found.
    """
    for sid, (start, end) in enumerate(sent_offsets):
        if start <= doc_idx <= end:
            return sid, doc_idx - start
    return -1, -1


# 3. Build sentence-level entity-pair examples
def build_examples_from_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    For one PLM document, build sentence-level entity-pair examples.

    Each example:
        {
            "sent_tokens": List[str],
            "entities": List[(start_in_sent, end_in_sent, ent_type, ent_global_id)],
            "subj_idx": int,
            "obj_idx": int,
            "label": str,
        }
    """
    flat_tokens, sent_offsets = flatten_sentences(doc)

    # sentence -> entities
    sent_entities = defaultdict(list)  # sid -> list of (s,e,type,ent_id)
    # sentence -> list of relations in local span coords
    sent_relations = defaultdict(list)  # sid -> list of (ps1,pe1,ps2,pe2,rel_type)

    ent_global_id = 0

    # 3.1 entities
    # doc["ner"] is a list per sentence, with spans indexed at doc-level
    for sid, ent_list in enumerate(doc.get("ner", [])):
        for span in ent_list:
            if len(span) != 3:
                # Some variants could carry extra info, be defensive
                s, e, ent_type = span[:3]
            else:
                s, e, ent_type = span
            sid_mapped, pos_start = doc_index_to_sent_pos(s, sent_offsets)
            sid2, pos_end = doc_index_to_sent_pos(e, sent_offsets)
            if sid_mapped == -1 or sid2 == -1 or sid_mapped != sid2:
                # Skip weird / cross-sentence spans
                continue
            sent_entities[sid_mapped].append((pos_start, pos_end, ent_type, ent_global_id))
            ent_global_id += 1

    # 3.2 relations
    # doc["relations"] is a list per sentence, spans doc-level
    for sid, rel_list in enumerate(doc.get("relations", [])):
        for rel in rel_list:
            if len(rel) < 5:
                # expected [s1,e1,s2,e2,rel_type]
                continue
            s1, e1, s2, e2, rel_type = rel[:5]
            sid1, ps1 = doc_index_to_sent_pos(s1, sent_offsets)
            sid1b, pe1 = doc_index_to_sent_pos(e1, sent_offsets)
            sid2, ps2 = doc_index_to_sent_pos(s2, sent_offsets)
            sid2b, pe2 = doc_index_to_sent_pos(e2, sent_offsets)

            if -1 in (sid1, sid1b, sid2, sid2b):
                continue
            if sid1 != sid1b or sid2 != sid2b:
                continue
            if sid1 != sid2:
                # relation crossing sentences: ignore for sentence-level baseline
                continue

            sent_relations[sid1].append((ps1, pe1, ps2, pe2, rel_type))

    # 3.3 build examples per sentence
    examples = []
    for sid, entities in sent_entities.items():
        if not entities:
            continue

        sent_tokens = doc["sentences"][sid]

        # map (span_start, span_end) -> index within this sentence
        span2idx = {(s, e): i for i, (s, e, t, gid) in enumerate(entities)}

        # gold relation lookup
        pair2label = {}
        for (ps1, pe1, ps2, pe2, rel_type) in sent_relations.get(sid, []):
            i = span2idx.get((ps1, pe1))
            j = span2idx.get((ps2, pe2))
            if i is None or j is None:
                continue
            pair2label[(i, j)] = rel_type

        n = len(entities)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                label = pair2label.get((i, j), "NO_RELATION")
                examples.append(
                    {
                        "sent_tokens": sent_tokens,
                        "entities": entities,
                        "subj_idx": i,
                        "obj_idx": j,
                        "label": label,
                    }
                )

    return examples


def build_examples_from_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_ex = []
    for d in docs:
        all_ex.extend(build_examples_from_doc(d))
    return all_ex


# 4. Entity marking
def insert_markers(
    sent_tokens: List[str],
    entities: List[Tuple[int, int, str, int]],
    subj_idx: int,
    obj_idx: int,
) -> List[str]:
    """
    Wrap subject and object spans in [E1]...[/E1], [E2]...[/E2].

    entities: list of (start, end, ent_type, ent_global_id) in *sentence* indices.
    """

    def _insert(tokens, start, end, open_tok, close_tok):
        return (
            tokens[:start]
            + [open_tok]
            + tokens[start : end + 1]
            + [close_tok]
            + tokens[end + 1 :]
        )

    s1, e1, _, _ = entities[subj_idx]
    s2, e2, _, _ = entities[obj_idx]

    # We need to insert from right to left so indices don’t shift incorrectly
    tokens = list(sent_tokens)

    if s1 < s2:
        # object is to the right
        tokens = _insert(tokens, s2, e2, "[E2]", "[/E2]")
        tokens = _insert(tokens, s1, e1, "[E1]", "[/E1]")
    else:
        tokens = _insert(tokens, s1, e1, "[E1]", "[/E1]")
        tokens = _insert(tokens, s2, e2, "[E2]", "[/E2]")

    return tokens


# 5. Dataset + Dataloader
class SciERRelDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        marked_tokens = insert_markers(
            ex["sent_tokens"],
            ex["entities"],
            ex["subj_idx"],
            ex["obj_idx"],
        )

        enc = self.tokenizer(
            marked_tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        label_id = self.label2id[ex["label"]]

        # squeeze batch dim
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label_id, dtype=torch.long)
        return enc


# 6. Stats: class distribution, etc.
def compute_label_stats(examples: List[Dict[str, Any]]) -> None:
    labels = [ex["label"] for ex in examples]
    counter = Counter(labels)
    total = sum(counter.values())

    print("=== Label distribution (including NO_RELATION) ===")
    for label, count in counter.most_common():
        print(f"{label:20s}  {count:8d}  {count/total:7.4f}")

    if "NO_RELATION" in counter:
        pos_total = total - counter["NO_RELATION"]
        print("\nTotal examples:", total)
        print("Positive examples:", pos_total)
        print("NO_RELATION    :", counter['NO_RELATION'])
        if pos_total > 0:
            print(f"Positive ratio: {pos_total/total:.4f}")

        print("\n=== Positive label distribution (no NO_RELATION) ===")
        for label, count in counter.items():
            if label == "NO_RELATION":
                continue
            print(f"{label:20s}  {count:8d}  {count/pos_total:7.4f}")
    else:
        print("\n(No NO_RELATION in this split?)")


# 7. Glue: build everything for a split
def prepare_split(
    plm_path: Path,
    pretrained_model_name: str = "allenai/scibert_scivocab_uncased",
    max_length: int = 256,
    batch_size: int = 16,
):
    print(f"Loading PLM docs from {plm_path} ...")
    docs = load_plm_docs(plm_path)
    print(f"Loaded {len(docs)} documents")

    print("Building entity-pair examples ...")
    examples = build_examples_from_docs(docs)
    print(f"Built {len(examples)} sentence-level entity pairs")

    # Collect labels (including NO_RELATION)
    labels = sorted({ex["label"] for ex in examples})
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    print("\nLabel mapping:")
    for lbl, i in label2id.items():
        print(f"{i:2d}: {lbl}")

    compute_label_stats(examples)

    # Tokenizer with entity markers
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    special = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special)

    dataset = SciERRelDataset(
        examples=examples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\nDataset size: {len(dataset)} examples")
    print(f"Batch size:   {batch_size}")
    print(f"Steps/epoch:  {len(dataloader)}")

    return {
        "docs": docs,
        "examples": examples,
        "label2id": label2id,
        "id2label": id2label,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "dataloader": dataloader,
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plm-path",
        type=str,
        required=True,
        help="Path to SciER PLM split, e.g. SciER/PLM/train.json",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="HF model name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    result = prepare_split(
        plm_path=Path(args.plm_path),
        pretrained_model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

