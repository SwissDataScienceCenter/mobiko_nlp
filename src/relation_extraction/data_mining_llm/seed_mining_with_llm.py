import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import sys
import regex as re

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None



# Model configurations
MODEL_CONFIGS = {
    "qwen3-4B": {
        "base_url": "https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507"
    },
    "qwen3-32B": {
        "base_url": "https://openwebui-runai-codev-llm.inference.compute.datascience.ch/api",
        "api_key": None,  # Will use OPEN_WEB_UI_API_KEY env var
        "model_name": "Qwen/Qwen3-32B-AWQ"
    },
    "medgemma-4b": {
        "base_url": "http://medgemma-4b-it.runai-mobiko-anisia.inference.compute.datascience.ch",
        "api_key": "EMPTY",
        "model_name": "google/medgemma-4b-it"
    },
    "biomistral-7b-awq": {
        "base_url": "https://mistral-7b-awq.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model_name": "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
    },
    "gpt4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": None,  # Will use OPENAI_API_KEY env var
        "model_name": "gpt-4o"
    },
    "qwen3-32B-vllm": {
        "base_url": "https://vllm-gateway-runai-codev-llm.inference.compute.datascience.ch/v1",
        "api_key": None,  # read from env
        "model_name": "Qwen/Qwen3-32B-AWQ"  # use the exact id your gateway serves
    }

}

def get_openai_client(model_type: str):
    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"Unknown model type: {model_type}. Use: {list(MODEL_CONFIGS.keys())}")

    api_key = config["api_key"] or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_WEB_UI_API_KEY")
    if not api_key:
        raise ValueError(f"API key required for {model_type}. Set OPENAI_API_KEY or OPEN_WEB_UI_API_KEY environment variable.")

    return OpenAI(
                base_url=config["base_url"],
                api_key=api_key
                ), config["model_name"]


# ---------- Data structures ----------

@dataclass
class Entity:
    text: str
    type: str
    start: int
    end: int


@dataclass
class SeedExample:
    relation: str
    sentence: str
    e1_text: str
    e1_type: str
    e2_text: str
    e2_type: str
    marked: str


@dataclass
class CandidatePair:
    paper_id: str
    sent_id: int
    relation_candidates: List[str]  # possible relations by type schema
    sentence: str
    e1: Entity
    e2: Entity
    marked: str


# ---------- Utility functions ----------

def load_json(path: Path):
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def mark_two_entities(text: str,
                      span1: Tuple[int, int],
                      span2: Tuple[int, int],
                      e1_tag: str = "E1",
                      e2_tag: str = "E2") -> str:
    """
    Insert <E1>...</E1> and <E2>...</E2> tags around the two spans.
    Insert from right to left so indices don't shift.
    """
    (s1, e1), (s2, e2) = span1, span2
    # sort so we insert later span first
    if s1 > s2:
        first_span, first_tag = (s2, e2), e2_tag
        second_span, second_tag = (s1, e1), e1_tag
    else:
        first_span, first_tag = (s1, e1), e1_tag
        second_span, second_tag = (s2, e2), e2_tag

    def insert_tags(txt: str, span: Tuple[int, int], tag: str) -> str:
        s, e = span
        return txt[:s] + f"<{tag}>" + txt[s:e] + f"</{tag}>" + txt[e:]

    # insert second span first (higher index), then first
    txt = insert_tags(text, second_span, second_tag)
    txt = insert_tags(txt, first_span, first_tag)
    return txt


def find_first_span(sentence: str, phrase: str) -> Optional[Tuple[int, int]]:
    idx = sentence.find(phrase)
    if idx == -1:
        return None
    return idx, idx + len(phrase)


# ---------- Embedding model ----------

class SentenceEmbedder(nn.Module):
    """
    Simple wrapper around a HF encoder with mean pooling.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.encoder.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.encoder(**encoded)
            attn_mask = encoded["attention_mask"].unsqueeze(-1)
            sum_embs = (outputs.last_hidden_state * attn_mask).sum(dim=1)
            lengths = attn_mask.sum(dim=1).clamp(min=1)
            embs = sum_embs / lengths
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)



# ---------- LLM validator ----------


def remove_thinking_blocks(content: str) -> str:
    # Remove <think>...</think> blocks (including nested content)
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL)

    # Clean up extra whitespace
    cleaned = cleaned.strip()

    # If content starts with ```json, extract just the JSON part
    if cleaned.startswith('```json'):
        # Find the JSON block
        start = cleaned.find('```json') + 7
        end = cleaned.rfind('```')
        if end > start:
            cleaned = cleaned[start:end].strip()

    return cleaned

class LLMValidator:
    """
    Abstract interface: given a mined example, say if relation holds.
    """

    def validate(self, example: dict) -> dict:
        """
        Input: mined example dict
        Must return dict with keys:
          - "llm_label": "YES" or "NO"
          - "llm_reason": str
        """
        raise NotImplementedError


class OpenAILLMValidator(LLMValidator):
    """
    Concrete validator using OpenAI-compatible chat API.

    Requires:
      - pip install openai
      - OPENAI_API_KEY env var set
    """

    def __init__(self, model_name: str):
        self.client, self.model_name = get_openai_client(model_name)

    def _build_prompt(self, example: dict) -> List[dict]:
        sentence = example["sentence"]
        e1 = example["e1"]
        e2 = example["e2"]
        relation = example["relation"]

        # Use the marked version for clarity
        marked = example.get("marked", sentence)

        sys_msg = (
            "You are an expert information extraction assistant for biodiversity. "
            "Your task is to decide if a specific semantic relation holds between two entities "
            "in a sentence. Be conservative: answer YES only if the relation is clearly stated "
            "or directly implied by the sentence."
        )

        user_msg = {
            "role": "user",
            "content": (
                "Decide whether the given relation holds between the two marked entities.\n\n"
                f"Sentence:\n{sentence}\n\n"
                f"Marked sentence (entities highlighted):\n{marked}\n\n"
                f"Entity 1: text='{e1['text']}', type='{e1['type']}'\n"
                f"Entity 2: text='{e2['text']}', type='{e2['type']}'\n"
                f"Candidate relation: {relation}\n\n"
                "Respond in strict JSON with keys 'label' and 'reason', where:\n"
                "  - 'label' is either 'YES' if the relation holds, or 'NO' if it does not.\n"
                "  - 'reason' is a short natural language explanation.\n\n"
                "Example response:\n"
                "{\"label\": \"YES\", \"reason\": \"The sentence explicitly states that the species occurs in this habitat.\"}"
            )
        }

        return [
            {"role": "system", "content": sys_msg},
            user_msg,
        ]

    def validate(self, example: dict) -> dict:
        messages = self._build_prompt(example)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            print('content thinking:', content)
            content = remove_thinking_blocks(content)
            print('content:', content)
        except Exception as e:
            # Fail closed: label as NO if LLM call fails
            return {
                "llm_label": "NO",
                "llm_reason": f"LLM call failed: {e}",
                "llm_raw": None,
            }

        # Parse JSON-ish response
        label = "NO"
        reason = ""
        try:
            obj = json.loads(content)
            label = str(obj.get("label", "")).strip().upper()
            if label not in ("YES", "NO"):
                label = "NO"
            reason = str(obj.get("reason", "")).strip()
        except Exception:
            # crude fallback: look for YES/NO in text
            upper = content.upper()
            if "YES" in upper and "NO" not in upper:
                label = "YES"
            elif "NO" in upper:
                label = "NO"
            reason = content
        print('label:', label)
        return {
            "llm_label": label,
            "llm_reason": reason,
            "llm_raw": content,
        }


# ---------- Loading seeds ----------

def load_seeds(seeds_path: Path,
               schema: Dict[str, List[List[str]]]) -> List[SeedExample]:
    data = load_json(seeds_path)
    seeds: List[SeedExample] = []

    for rel, examples in data.items():
        if rel not in schema:
            print(f"Warning: relation {rel} in seeds.json not in schema.json, skipping.")
            continue
        for ex in examples:
            sent = ex["sentence"]
            e1 = ex["e1"]
            e2 = ex["e2"]
            span1 = find_first_span(sent, e1["text"])
            span2 = find_first_span(sent, e2["text"])
            if span1 is None or span2 is None:
                print(f"Warning: could not find entity spans in seed: {ex}")
                continue
            marked = mark_two_entities(
                sent,
                span1,
                span2,
                e1_tag="E1",
                e2_tag="E2"
            )
            seeds.append(
                SeedExample(
                    relation=rel,
                    sentence=sent,
                    e1_text=e1["text"],
                    e1_type=e1["type"],
                    e2_text=e2["text"],
                    e2_type=e2["type"],
                    marked=marked,
                )
            )
    return seeds


# ---------- Building candidate pairs from corpus ----------

def build_relation_candidate_pairs(
    corpus_path: Path,
    schema: Dict[str, List[List[str]]],
) -> List[CandidatePair]:
    """
    For each sentence, generate all entity pairs whose types match at least
    one relation schema. We don't assign a single relation yet; we keep
    all possible relation labels this pair could satisfy.
    """
    pairs: List[CandidatePair] = []

    # precompute mapping from (type1,type2) -> possible relations
    type_pair_to_relations: Dict[Tuple[str, str], List[str]] = {}
    for rel, pairs_list in schema.items():
        for t1, t2 in pairs_list:
            type_pair_to_relations.setdefault((t1, t2), []).append(rel)

    m = 0
    for record in tqdm(list(load_jsonl(corpus_path)), desc="Scanning corpus"):
        if m > 100:
            break
        m += 1
        paper_id = record.get("paper_id", "")
        sent_id = record.get("sent_id", -1)
        text = record["text"]
        ents_raw = record.get("entities", [])

        entities = [
            Entity(
                text=e["text"],
                type=e["type"],
                start=e["start"],
                end=e["end"],
            )
            for e in ents_raw
        ]

        n = len(entities)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                e1 = entities[i]
                e2 = entities[j]
                possible_rels = type_pair_to_relations.get((e1.type, e2.type), [])
                if not possible_rels:
                    continue
                marked = mark_two_entities(
                    text,
                    (e1.start, e1.end),
                    (e2.start, e2.end),
                    e1_tag="E1",
                    e2_tag="E2",
                )
                pairs.append(
                    CandidatePair(
                        paper_id=paper_id,
                        sent_id=sent_id,
                        relation_candidates=possible_rels,
                        sentence=text,
                        e1=e1,
                        e2=e2,
                        marked=marked,
                    )
                )
    return pairs


# ---------- Similarity + mining ----------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a_norm @ b_norm.T


def mine_candidates(
    embedder: SentenceEmbedder,
    seeds: List[SeedExample],
    candidates: List[CandidatePair],
    similarity_threshold: float,
    batch_size: int = 32,
) -> List[dict]:
    """
    For each candidate pair, compute similarity to all seeds for each
    possible relation, keep the best relation above threshold.
    Returns a list of JSON-serializable mined examples.
    """
    # group seeds by relation
    seeds_by_relation: Dict[str, List[SeedExample]] = {}
    for s in seeds:
        seeds_by_relation.setdefault(s.relation, []).append(s)

    # pre-embed seeds
    seed_relation_texts = {
        rel: [s.marked for s in rel_seeds]
        for rel, rel_seeds in seeds_by_relation.items()
    }
    seed_relation_embs = {
        rel: embedder.encode(texts)
        for rel, texts in seed_relation_texts.items()
    }

    mined: List[dict] = []
    candidates = candidates[:100]

    # process candidates in batches
    for i in tqdm(range(0, len(candidates), batch_size), desc="Mining candidates"):
        batch = candidates[i:i + batch_size]
        marked_texts = [c.marked for c in batch]
        cand_embs = embedder.encode(marked_texts, batch_size=batch_size)

        for idx, cand in enumerate(batch):
            emb = cand_embs[idx].unsqueeze(0)  # shape (1, dim)

            best_rel = None
            best_score = -1.0

            for rel in cand.relation_candidates:
                if rel not in seed_relation_embs:
                    continue
                seed_embs = seed_relation_embs[rel]  # (n_seeds, dim)
                sims = cosine_sim(emb, seed_embs)  # (1, n_seeds)
                score = sims.max().item()
                if score > best_score:
                    best_score = score
                    best_rel = rel

            if best_rel is None or best_score < similarity_threshold:
                continue

            mined.append(
                {
                    "paper_id": cand.paper_id,
                    "sent_id": cand.sent_id,
                    "relation": best_rel,
                    "similarity": best_score,
                    "sentence": cand.sentence,
                    "e1": {
                        "text": cand.e1.text,
                        "type": cand.e1.type,
                        "start": cand.e1.start,
                        "end": cand.e1.end,
                    },
                    "e2": {
                        "text": cand.e2.text,
                        "type": cand.e2.type,
                        "start": cand.e2.start,
                        "end": cand.e2.end,
                    },
                    "marked": cand.marked,
                }
            )

    return mined


# ---------- LLM filtering step ----------

def llm_filter_candidates(
    candidates: List[dict],
    validator: LLMValidator,
    max_calls: Optional[int] = None,
) -> List[dict]:
    """
    For each mined candidate, call LLM to validate relation.
    Keep all candidates, but add llm_label/llm_reason/llm_raw.
    If you want to drop NOs, filter after calling this.
    """
    filtered: List[dict] = []
    to_process = candidates
    if max_calls is not None:
        to_process = candidates[:max_calls]

    for ex in tqdm(to_process, desc="LLM validating"):
        verdict = validator.validate(ex)
        ex_llm = dict(ex)
        ex_llm.update(verdict)
        filtered.append(ex_llm)

    # append untouched remainder (without LLM fields) if we truncated
    if max_calls is not None and max_calls < len(candidates):
        filtered.extend(candidates[max_calls:])

    return filtered


# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Seed-based mining of relation candidates from biodiversity corpus with optional LLM validation"
    )
    parser.add_argument("--corpus-jsonl", type=Path, required=True,
                        help="Path to corpus sentences JSONL with entities.")
    parser.add_argument("--schema-json", type=Path, required=True,
                        help="Path to schema.json describing allowed type pairs per relation.")
    parser.add_argument("--seeds-json", type=Path, required=True,
                        help="Path to seeds.json with seed examples per relation.")
    parser.add_argument("--model-name", type=str, default="allenai/scibert_scivocab_uncased",
                        help="HF model name for embeddings.")
    parser.add_argument("--similarity-threshold", type=float, default=0.75,
                        help="Cosine similarity threshold to accept a candidate.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding.")
    parser.add_argument("--output-jsonl", type=Path, required=True,
                        help="Where to write mined (and possibly LLM-filtered) candidates as JSONL.")

    # LLM options
    parser.add_argument("--use-llm", action="store_true",
                        help="If set, call an LLM to validate each mined candidate.")
    parser.add_argument("--llm-model", type=str, help="LLM model name (for OpenAI-style client).")
    parser.add_argument("--llm-base-url", type=str, default=None,
                        help="Optional base URL for self-hosted OpenAI-compatible endpoint.")
    parser.add_argument("--llm-max-calls", type=int, default=None,
                        help="Optional cap on number of LLM calls (for debugging / cost control).")

    args = parser.parse_args()

    schema = load_json(args.schema_json)
    seeds = load_seeds(args.seeds_json, schema)
    print(f"Loaded {len(seeds)} seed examples.")

    candidates = build_relation_candidate_pairs(args.corpus_jsonl, schema)
    print(f"Built {len(candidates)} candidate pairs from corpus.")
    print(candidates[0])
    embedder = SentenceEmbedder(args.model_name)

    mined = mine_candidates(
        embedder=embedder,
        seeds=seeds,
        candidates=candidates,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size,
    )
    print(f"Mined {len(mined)} candidates above similarity {args.similarity_threshold}.")
    print(mined[0])

    # Optional LLM validation
    if args.use_llm:
        if OpenAI is None and args.llm_base_url is None:
            raise RuntimeError("openai package missing or misconfigured. Install it or plug your own LLM client.")
        print(f"Running LLM validation with model={args.llm_model} ...")
        validator = OpenAILLMValidator(model_name=args.llm_model)
        mined = llm_filter_candidates(
            candidates=mined,
            validator=validator,
            max_calls=args.llm_max_calls,
        )
        mined = [ex for ex in mined if ex.get("llm_label") == "YES"]

    with args.output_jsonl.open("w", encoding="utf8") as f:
        for rec in mined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(mined)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
