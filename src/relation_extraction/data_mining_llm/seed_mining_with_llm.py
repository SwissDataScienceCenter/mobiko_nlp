import argparse
import ast
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




os.environ["OPEN_WEB_UI_API_KEY"] = ""


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
    tier: Optional[str] = None
    uncertainty: Optional[float] = None
    concept_text: Optional[str] = None


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

def entity_to_output_dict(entity: Entity) -> Dict[str, object]:
    out = {
        "text": entity.text,
        "type": entity.type,
        "start": entity.start,
        "end": entity.end,
    }
    if entity.tier is not None:
        out["tier"] = entity.tier
    if entity.uncertainty is not None:
        out["uncertainty"] = entity.uncertainty
    if entity.concept_text is not None:
        out["concept_text"] = entity.concept_text
    return out

def normalize_entity_type_to_tag(entity_type: str) -> str:
    """
    Convert entity type text into a safe tag name.
    Example: "BIOTIC ENTITY" -> "BIOTIC_ENTITY".
    """
    tag = re.sub(r"[^A-Za-z0-9]+", "_", entity_type.strip().upper()).strip("_")
    if not tag:
        tag = "ENTITY"
    if tag[0].isdigit():
        tag = f"TYPE_{tag}"
    return tag


def resolve_marker_tag(entity_type: str, fallback_tag: str, marker_style: str) -> str:
    if marker_style == "generic":
        return fallback_tag
    return normalize_entity_type_to_tag(entity_type)


def collect_schema_entity_types(schema: Dict[str, List[List[str]]]) -> List[str]:
    types = set()
    for pairs_list in schema.values():
        for t1, t2 in pairs_list:
            types.add(t1)
            types.add(t2)
    return sorted(types)


def load_json(path: Path):
    with path.open("r", encoding="utf8") as f:
        print(f"Loading JSON from {path}...")
        return json.load(f)


def load_schema(path: Path) -> Dict[str, List[List[str]]]:
    """
    Load relation schema from JSON or Python dict literal.
    Supported Python forms:
    - bare dict literal file
    - assignment to `schema` or `SCHEMA`
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)

    if suffix != ".py":
        raise ValueError(f"Unsupported schema format: {path}. Use .json or .py")

    text = path.read_text(encoding="utf8")
    print(f"Loading Python schema from {path}...")
    tree = ast.parse(text, filename=str(path), mode="exec")

    schema_obj = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"schema", "SCHEMA"}:
                    schema_obj = ast.literal_eval(node.value)
                    break
        if schema_obj is not None:
            break

    if schema_obj is None and len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        schema_obj = ast.literal_eval(tree.body[0].value)

    if not isinstance(schema_obj, dict):
        raise ValueError(
            f"Could not read schema dict from {path}. "
            "Expected dict literal or assignment to `schema`/`SCHEMA`."
        )

    return schema_obj


def load_seeds_data(path: Path) -> Dict[str, List[dict]]:
    """
    Load seeds from JSON or Python dict literal.
    Supported Python forms:
    - bare dict literal file
    - assignment to `seeds` or `SEEDS`
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)

    if suffix != ".py":
        raise ValueError(f"Unsupported seeds format: {path}. Use .json or .py")

    text = path.read_text(encoding="utf8")
    print(f"Loading Python seeds from {path}...")
    tree = ast.parse(text, filename=str(path), mode="exec")

    seeds_obj = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"seeds", "SEEDS"}:
                    seeds_obj = ast.literal_eval(node.value)
                    break
        if seeds_obj is not None:
            break

    if seeds_obj is None and len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        seeds_obj = ast.literal_eval(tree.body[0].value)

    if not isinstance(seeds_obj, dict):
        raise ValueError(
            f"Could not read seeds dict from {path}. "
            "Expected dict literal or assignment to `seeds`/`SEEDS`."
        )

    return seeds_obj


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
    Wrapper around a HF encoder using entity-marker pooling.
    Returns [h_<E1>; h_<E2>] for each input text.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        marker_style: str = "generic",
        marker_entity_types: Optional[List[str]] = None,
    ):
        super().__init__()
        self.marker_style = marker_style
        self.tokenizer = self._load_tokenizer(model_name, tokenizer_name=tokenizer_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if marker_style == "generic":
            self.marker_open_tokens = ["<E1>", "<E2>"]
            marker_tokens = ["<E1>", "</E1>", "<E2>", "</E2>"]
        else:
            marker_entity_types = marker_entity_types or []
            marker_tags = sorted({normalize_entity_type_to_tag(t) for t in marker_entity_types})
            self.marker_open_tokens = [f"<{t}>" for t in marker_tags]
            marker_tokens = self.marker_open_tokens + [f"</{t}>" for t in marker_tags]
        vocab = self.tokenizer.get_vocab()
        missing = [tok for tok in marker_tokens if tok not in vocab]
        if missing:
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing})
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        if marker_style == "generic":
            self.e1_token_id = self.tokenizer.convert_tokens_to_ids("<E1>")
            self.e2_token_id = self.tokenizer.convert_tokens_to_ids("<E2>")
            self.marker_open_token_ids = set()
        else:
            self.e1_token_id = None
            self.e2_token_id = None
            self.marker_open_token_ids = {
                self.tokenizer.convert_tokens_to_ids(tok)
                for tok in self.marker_open_tokens
            }
            self.marker_open_token_ids.discard(self.tokenizer.unk_token_id)
        self.encoder.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.encoder.to(self.device)

    @staticmethod
    def _load_tokenizer(model_name: str, tokenizer_name: Optional[str] = None):
        """
        Load tokenizer robustly across model families:
        try fast tokenizer first, then fallback to slow tokenizer.
        """
        candidates: List[str] = []
        if tokenizer_name:
            candidates.append(tokenizer_name)
        candidates.append(model_name)
        # SPECTER2 commonly uses SciBERT tokenizer assets.
        if "specter2" in model_name.lower():
            candidates.append("allenai/scibert_scivocab_uncased")

        tried_errors = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                return AutoTokenizer.from_pretrained(candidate, use_fast=True)
            except Exception as fast_err:
                try:
                    return AutoTokenizer.from_pretrained(candidate, use_fast=False)
                except Exception as slow_err:
                    tried_errors.append(
                        f"{candidate} -> fast_err={fast_err} | slow_err={slow_err}"
                    )

        errors_joined = "\n".join(tried_errors)
        raise RuntimeError(
            "Failed to load tokenizer.\n"
            f"model_name='{model_name}', tokenizer_name='{tokenizer_name}'.\n"
            "Tried explicit tokenizer (if provided), model name, and known fallbacks.\n"
            "Install dependencies: `pip install sentencepiece tiktoken`.\n"
            f"Attempts:\n{errors_joined}"
        )

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
            hidden = outputs.last_hidden_state
            input_ids = encoded["input_ids"]
            attn_mask = encoded["attention_mask"].unsqueeze(-1)
            mean_pooled = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1)

            # Entity-marker pooling: concatenate hidden states at marker starts.
            # Fallback to mean pooled state when markers are absent.
            embs = []
            for row_idx in range(hidden.size(0)):
                if self.marker_style == "generic":
                    e1_positions = (input_ids[row_idx] == self.e1_token_id).nonzero(as_tuple=False)
                    e2_positions = (input_ids[row_idx] == self.e2_token_id).nonzero(as_tuple=False)
                    e1_state = hidden[row_idx, e1_positions[0].item()] if e1_positions.numel() > 0 else mean_pooled[row_idx]
                    e2_state = hidden[row_idx, e2_positions[0].item()] if e2_positions.numel() > 0 else mean_pooled[row_idx]
                else:
                    marker_positions = [
                        pos for pos, tok_id in enumerate(input_ids[row_idx].tolist())
                        if tok_id in self.marker_open_token_ids
                    ]
                    e1_state = hidden[row_idx, marker_positions[0]] if len(marker_positions) > 0 else mean_pooled[row_idx]
                    e2_state = hidden[row_idx, marker_positions[1]] if len(marker_positions) > 1 else mean_pooled[row_idx]
                embs.append(torch.cat([e1_state, e2_state], dim=-1))
            embs = torch.stack(embs, dim=0)
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
            #Fail closed: label as NO if LLM call fails
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

def load_seeds(
    seeds_path: Path,
    schema: Dict[str, List[List[str]]],
    marker_style: str = "generic",
) -> List[SeedExample]:
    data = load_seeds_data(seeds_path)
    seeds: List[SeedExample] = []

    for rel, examples in data.items():
        if rel not in schema:
            print(f"Warning: relation {rel} in seeds file not in schema, skipping.")
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
                e1_tag=resolve_marker_tag(e1["type"], "E1", marker_style),
                e2_tag=resolve_marker_tag(e2["type"], "E2", marker_style),
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
    marker_style: str = "generic",
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

    for record in tqdm(list(load_jsonl(corpus_path)), desc="Scanning corpus"):
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
                tier=e.get("tier"),
                uncertainty=e.get("uncertainty"),
                concept_text=e.get("concept_text"),
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
                    e1_tag=resolve_marker_tag(e1.type, "E1", marker_style),
                    e2_tag=resolve_marker_tag(e2.type, "E2", marker_style),
                )
                candidate = CandidatePair(
                        paper_id=paper_id,
                        sent_id=sent_id,
                        relation_candidates=possible_rels,
                        sentence=text,
                        e1=e1,
                        e2=e2,
                        marked=marked,
                    )
                if candidate not in pairs:
                    pairs.append(candidate)
    return pairs


# ---------- Similarity + mining ----------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a_norm @ b_norm.T


def build_triplet_text(e1_text: str, e1_type: str, relation: str,
                       e2_text: str, e2_type: str) -> str:
    """Build a structural triplet string for embedding.

    Example: "alpine plant species [BIOTIC ENTITY] HAS_PROPERTY perennial [TEMPORAL PROPERTY]"
    """
    return f"{e1_text} [{e1_type}] {relation} {e2_text} [{e2_type}]"


def extract_relational_context(sentence: str,
                               e1_text: str, e1_type: str,
                               e2_text: str, e2_type: str,
                               e1_start: Optional[int] = None,
                               e1_end: Optional[int] = None,
                               e2_start: Optional[int] = None,
                               e2_end: Optional[int] = None) -> str:
    """Extract text between two entities, replacing entity text with type markers.

    Returns e.g. "[BIOTIC ENTITY] are found in [SPATIAL ENTITY]"
    Falls back to "[e1_type] <full sentence> [e2_type]" if spans can't be located.
    """
    # Determine spans: use offsets if provided, otherwise string search
    if e1_start is not None and e1_end is not None:
        span1 = (e1_start, e1_end)
    else:
        span1 = find_first_span(sentence, e1_text)

    if e2_start is not None and e2_end is not None:
        span2 = (e2_start, e2_end)
    else:
        span2 = find_first_span(sentence, e2_text)

    if span1 is None or span2 is None:
        return f"[{e1_type}] {sentence} [{e2_type}]"

    # Order by position in sentence
    if span1[0] <= span2[0]:
        first_type, second_type = e1_type, e2_type
        between = sentence[span1[1]:span2[0]]
    else:
        first_type, second_type = e2_type, e1_type
        between = sentence[span2[1]:span1[0]]

    between = between.strip()
    return f"[{first_type}] {between} [{second_type}]"


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
                        **entity_to_output_dict(cand.e1),
                    },
                    "e2": {
                        **entity_to_output_dict(cand.e2),
                    },
                    "marked": cand.marked,
                }
            )

    return mined


def mine_candidates_multiview(
    embedder: SentenceEmbedder,
    seeds: List[SeedExample],
    candidates: List[CandidatePair],
    similarity_threshold: float,
    batch_size: int = 32,
    w_structural: float = 0.3,
    w_relational: float = 0.4,
    w_sentence: float = 0.3,
) -> List[dict]:
    """
    Multi-view hybrid similarity mining.

    Combines 3 complementary signals:
      1. Structural triplet: embeds "{e1} [TYPE] RELATION {e2} [TYPE]"
      2. Relational context: embeds text between entities with type markers
      3. Full sentence: embeds the whole marked sentence (same as original)

    Final score = w_structural*sim1 + w_relational*sim2 + w_sentence*sim3
    """
    # --- group seeds by relation ---
    seeds_by_relation: Dict[str, List[SeedExample]] = {}
    for s in seeds:
        seeds_by_relation.setdefault(s.relation, []).append(s)

    # --- pre-embed seeds (3 views, per relation) ---
    seed_sentence_embs: Dict[str, torch.Tensor] = {}
    seed_structural_embs: Dict[str, torch.Tensor] = {}
    seed_relational_embs: Dict[str, torch.Tensor] = {}

    for rel, rel_seeds in seeds_by_relation.items():
        # Sentence view
        seed_sentence_embs[rel] = embedder.encode(
            [s.marked for s in rel_seeds]
        )
        # Structural triplet view
        seed_structural_embs[rel] = embedder.encode(
            [build_triplet_text(s.e1_text, s.e1_type, rel, s.e2_text, s.e2_type)
             for s in rel_seeds]
        )
        # Relational context view
        seed_relational_embs[rel] = embedder.encode(
            [extract_relational_context(s.sentence, s.e1_text, s.e1_type,
                                        s.e2_text, s.e2_type)
             for s in rel_seeds]
        )

    mined: List[dict] = []

    # --- process candidates in batches ---
    for i in tqdm(range(0, len(candidates), batch_size), desc="Mining candidates (multiview)"):
        batch = candidates[i:i + batch_size]

        # -- Sentence view embeddings --
        cand_sentence_embs = embedder.encode(
            [c.marked for c in batch], batch_size=batch_size
        )

        # -- Relational context view embeddings --
        cand_context_texts = [
            extract_relational_context(
                c.sentence, c.e1.text, c.e1.type, c.e2.text, c.e2.type,
                c.e1.start, c.e1.end, c.e2.start, c.e2.end,
            )
            for c in batch
        ]
        cand_context_embs = embedder.encode(cand_context_texts, batch_size=batch_size)

        # -- Structural triplet view embeddings --
        # Build two triplets per (candidate, relation_candidate): forward and reverse
        triplet_texts: List[str] = []
        # (batch_idx, rel) -> (fwd_idx, rev_idx) in triplet_texts
        triplet_map: Dict[Tuple[int, str], Tuple[int, int]] = {}
        for idx, cand in enumerate(batch):
            for rel in cand.relation_candidates:
                fwd_idx = len(triplet_texts)
                triplet_texts.append(
                    build_triplet_text(cand.e1.text, cand.e1.type, rel,
                                       cand.e2.text, cand.e2.type)
                )
                rev_idx = len(triplet_texts)
                triplet_texts.append(
                    build_triplet_text(cand.e2.text, cand.e2.type, rel,
                                       cand.e1.text, cand.e1.type)
                )
                triplet_map[(idx, rel)] = (fwd_idx, rev_idx)
        cand_triplet_embs = embedder.encode(triplet_texts, batch_size=batch_size) if triplet_texts else None

        # -- Score each candidate --
        for idx, cand in enumerate(batch):
            best_rel = None
            best_score = -1.0
            best_sub_scores = (0.0, 0.0, 0.0)

            for rel in cand.relation_candidates:
                if rel not in seed_sentence_embs:
                    continue

                # Structural similarity (max of forward and reverse triplet)
                if cand_triplet_embs is not None and (idx, rel) in triplet_map:
                    fwd_i, rev_i = triplet_map[(idx, rel)]
                    fwd_emb = cand_triplet_embs[fwd_i].unsqueeze(0)
                    rev_emb = cand_triplet_embs[rev_i].unsqueeze(0)
                    sim_fwd = cosine_sim(fwd_emb, seed_structural_embs[rel]).max().item()
                    sim_rev = cosine_sim(rev_emb, seed_structural_embs[rel]).max().item()
                    sim_structural = max(sim_fwd, sim_rev)
                else:
                    sim_structural = 0.0

                # Relational context similarity
                ctx_emb = cand_context_embs[idx].unsqueeze(0)
                sim_relational = cosine_sim(ctx_emb, seed_relational_embs[rel]).max().item()

                # Sentence similarity
                sent_emb = cand_sentence_embs[idx].unsqueeze(0)
                sim_sentence = cosine_sim(sent_emb, seed_sentence_embs[rel]).max().item()

                # Combined score
                final_score = (w_structural * sim_structural
                               + w_relational * sim_relational
                               + w_sentence * sim_sentence)

                if final_score > best_score:
                    best_score = final_score
                    best_rel = rel
                    best_sub_scores = (sim_structural, sim_relational, sim_sentence)

            if best_rel is None or best_score < similarity_threshold:
                continue

            if sim_relational < 0.8:
                print(f"Low relational similarity for candidate {cand.paper_id}:{cand.sent_id} rel={best_rel} score={best_score:.4f} (struct={best_sub_scores[0]:.4f}, rel={best_sub_scores[1]:.4f}, sent={best_sub_scores[2]:.4f})")
                continue

            mined.append(
                {
                    "paper_id": cand.paper_id,
                    "sent_id": cand.sent_id,
                    "relation": best_rel,
                    "similarity": best_score,
                    "sim_structural": best_sub_scores[0],
                    "sim_relational": best_sub_scores[1],
                    "sim_sentence": best_sub_scores[2],
                    "sentence": cand.sentence,
                    "e1": {
                        **entity_to_output_dict(cand.e1),
                    },
                    "e2": {
                        **entity_to_output_dict(cand.e2),
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


def _save_checkpoint(path: Path, payload: dict) -> None:
    tmp_path = Path(str(path) + ".tmp")
    with tmp_path.open("w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _load_checkpoint(path: Path) -> dict:
    with path.open("r", encoding="utf8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} is not a JSON object.")
    return payload


def llm_filter_and_stream_yes(
    candidates: List[dict],
    validator: LLMValidator,
    output_path: Path,
    max_calls: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
    checkpoint_every: int = 1,
) -> int:
    """
    Validate candidates with LLM and stream YES-labeled examples to disk immediately.
    Returns number of written YES records.
    """
    to_process = candidates if max_calls is None else candidates[:max_calls]
    total = len(to_process)

    start_idx = 0
    written = 0
    file_mode = "w"
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        state = _load_checkpoint(checkpoint_path)
        start_idx = int(state.get("next_index", 0))
        written = int(state.get("written_yes", 0))
        start_idx = max(0, min(start_idx, total))
        file_mode = "a"
        print(f"Resuming LLM validation from index {start_idx}/{total} "
              f"(already wrote {written} YES examples).")

    if start_idx >= total:
        print("Checkpoint indicates all selected candidates are already processed.")
        return written

    with output_path.open(file_mode, encoding="utf8") as f:
        for idx in tqdm(range(start_idx, total), desc="LLM validating"):
            ex = to_process[idx]
            verdict = validator.validate(ex)
            ex_llm = dict(ex)
            ex_llm.update(verdict)
            if ex_llm.get("llm_label") == "YES":
                f.write(json.dumps(ex_llm, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
            if checkpoint_path is not None:
                processed = idx + 1
                if checkpoint_every <= 1 or (processed % checkpoint_every == 0) or processed == total:
                    _save_checkpoint(
                        checkpoint_path,
                        {
                            "next_index": processed,
                            "written_yes": written,
                            "total": total,
                        },
                    )
    return written


# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Seed-based mining of relation candidates from biodiversity corpus with optional LLM validation"
    )
    parser.add_argument("--corpus-jsonl", type=Path, required=True,
                        help="Path to corpus sentences JSONL with entities.")
    parser.add_argument("--schema-json", type=Path, required=True,
                        help="Path to schema file (.json or .py) describing allowed type pairs per relation.")
    parser.add_argument("--seeds-json", type=Path, required=True,
                        help="Path to seeds file (.json or .py) with seed examples per relation.")
    parser.add_argument("--model-name", type=str, default="allenai/scibert_scivocab_uncased",
                        help="HF model name for embeddings.")
    parser.add_argument("--tokenizer-name", type=str, default=None,
                        help="Optional tokenizer model id/path when model repo lacks tokenizer assets.")
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
    parser.add_argument("--checkpoint-path", type=Path, default=None,
                        help="Path to JSON checkpoint file for resumable LLM validation.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume LLM validation from checkpoint if available.")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save checkpoint every N processed candidates (default: 1).")

    # Multi-view similarity weights
    parser.add_argument("--w-structural", type=float, default=0.3,
                        help="Weight for structural triplet similarity (default: 0.3).")
    parser.add_argument("--w-relational", type=float, default=0.4,
                        help="Weight for relational context similarity (default: 0.4).")
    parser.add_argument("--w-sentence", type=float, default=0.3,
                        help="Weight for full sentence similarity (default: 0.3).")
    parser.add_argument(
        "--marker-style",
        choices=["generic", "type-specific"],
        default="generic",
        help="Marker format: generic (<E1>/<E2>) or type-specific (<BIOTIC_ENTITY>/...).",
    )

    args = parser.parse_args()

    # Validate weights sum to 1.0
    weight_sum = args.w_structural + args.w_relational + args.w_sentence
    if abs(weight_sum - 1.0) > 1e-6:
        parser.error(f"Weights must sum to 1.0, got {weight_sum:.4f} "
                     f"(structural={args.w_structural}, relational={args.w_relational}, "
                     f"sentence={args.w_sentence})")

    schema = load_schema(args.schema_json)
    seeds = load_seeds(args.seeds_json, schema, marker_style=args.marker_style)
    print(f"Loaded {len(seeds)} seed examples.")

    candidates = build_relation_candidate_pairs(
        args.corpus_jsonl,
        schema,
        marker_style=args.marker_style,
    )
    print(f"Built {len(candidates)} candidate pairs from corpus.")
    print(candidates[0])
    schema_entity_types = collect_schema_entity_types(schema)
    embedder = SentenceEmbedder(
        args.model_name,
        tokenizer_name=args.tokenizer_name,
        marker_style=args.marker_style,
        marker_entity_types=schema_entity_types,
    )

    print(f"Using multiview weights: structural={args.w_structural}, "
          f"relational={args.w_relational}, sentence={args.w_sentence}")
    mined = mine_candidates_multiview(
        embedder=embedder,
        seeds=seeds,
        candidates=candidates,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size,
        w_structural=args.w_structural,
        w_relational=args.w_relational,
        w_sentence=args.w_sentence,
    )
    print(f"Mined {len(mined)} candidates above similarity {args.similarity_threshold}.")
    if mined:
        print(mined[0])

    # Optional LLM validation
    if args.use_llm:
        if OpenAI is None and args.llm_base_url is None:
            raise RuntimeError("openai package missing or misconfigured. Install it or plug your own LLM client.")
        if args.checkpoint_every < 1:
            parser.error(f"--checkpoint-every must be >= 1, got {args.checkpoint_every}")
        checkpoint_path = args.checkpoint_path or Path(str(args.output_jsonl) + ".ckpt.json")
        print(f"Running LLM validation with model={args.llm_model} ...")
        print(f"Checkpoint file: {checkpoint_path} (resume={args.resume}, every={args.checkpoint_every})")
        validator = OpenAILLMValidator(model_name=args.llm_model)
        written = llm_filter_and_stream_yes(
            candidates=mined,
            validator=validator,
            output_path=args.output_jsonl,
            max_calls=args.llm_max_calls,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"Wrote {written} records to {args.output_jsonl}")
        return

    with args.output_jsonl.open("w", encoding="utf8") as f:
        for rec in mined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(mined)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
