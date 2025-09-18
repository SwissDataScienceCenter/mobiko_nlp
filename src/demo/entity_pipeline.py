import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any
import threading
from pathlib import Path
import re, unicodedata


import spacy
from openai import OpenAI

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ner.labels import EntityLabel, build_bio_labels
from ner.ner_infer import NerInferencer
from preprocess.gazetteer_matcher import Rule as GazetteerRule, load_gaz_rules_from_dir, GazetteerMatcher


from prompts import DEFAULT_SYSTEM_PROMPT, NO_CHUNK_CANDIDATE_SYSTEM_PROMPT, NER_AWARE_SYSTEM_PROMPT
import glob




# Model configurations
MODEL_CONFIGS = {
    "qwen3-4B": {
        "base_url": "https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507"
    },
    "qwen3-32B": {
        "base_url": "https://openwebui.runai-codev-llm.inference.compute.datascience.ch/api",
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
}



# Thread-local storage for spaCy models
thread_local = threading.local()


_ws = re.compile(r"\s+")


def _ws_tokenize_with_offsets(text: str):
    """
    Whitespace tokenization with character offsets.
    Returns (tokens, token_spans) where token_spans[i] = (start_char, end_char).
    """
    tokens, spans = [], []
    pos = 0
    while True:
        m = _ws.search(text, pos)
        end = m.start() if m else len(text)
        if end > pos:  # non-empty
            tokens.append(text[pos:end])
            spans.append((pos, end))
        if not m:
            break
        pos = m.end()
    return tokens, spans


def _bio_from_ids_to_spans(tags: List[str], token_spans: List[tuple], text: str):
    """
    Map BIO word tags → sentence-level char spans using precomputed token spans.
    """
    spans, active_type, start_i = [], None, None
    for i, tag in enumerate(tags):
        if not tag or tag == "O":
            if active_type is not None:
                s, _ = token_spans[start_i]; _, e = token_spans[i-1]
                spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
                active_type, start_i = None, None
            continue
        if "-" in tag:
            pref, typ = tag.split("-", 1)
        else:
            pref, typ = "B", tag
        if pref == "B" or (active_type and typ != active_type):
            if active_type is not None:
                s, _ = token_spans[start_i]; _, e = token_spans[i-1]
                spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
            active_type, start_i = typ, i
    if active_type is not None:
        s, _ = token_spans[start_i]; _, e = token_spans[len(token_spans)-1]
        spans.append({"start_char": s, "end_char": e, "text": text[s:e], "type": active_type})
    return spans



def _load_bioc_index_from_dir(bioc_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read all *.json in a directory of BioC files.
    Extract per-sentence spans.
    Return {sentence_text -> [ {text, start_char, end_char, type}, ... ]}.
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    for path in glob.glob(os.path.join(bioc_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
            articles = (doc.get("sibils_article_set") or doc.get("articles"))
            for article in articles:

                if article.get("passages") and isinstance(article.get("passages"), list):
                    # Build sentence index: (field, sentence_number) -> sentence text

                    for passage in article["passages"]:
                        text = passage.get("text") or ""
                        if not text:
                            continue
                        spans = []
                        for annotation in passage.get("annotations", []):
                            infons = annotation.get("infons", {}) or {}

                            for location in annotation.get("locations", []):
                                start = int(location.get("offset", 0))
                                length = int(location.get("length", 0))
                                end = start + length

                                # Validate span boundaries
                                if end <= start or start < 0 or end > len(text):
                                    continue

                                spans.append({
                                    "start": start,
                                    "end": end,
                                    "text": text[start:end],
                                    "source": infons.get("concept_source"),
                                    "concept_id": infons.get("concept_id"),
                                    "preferred_term": infons.get("preferred_term")
                                })
                        # Sort spans (stable) for predictability
                        spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
                        index[text] = spans
    return index



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


def get_spacy_model(model_name: str):
    """Get thread-local spaCy model for parallel processing."""
    if not hasattr(thread_local, 'nlp'):
        thread_local.nlp = spacy.load(model_name)
    return thread_local.nlp


def read_txt_files(indir: str):
    for name in os.listdir(indir):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(indir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            yield os.path.splitext(name)[0], f.read()


def find_span_positions(text: str, span_text: str):
    """
    Find all positions of span_text in text using regex.
    Returns list of (start, end) tuples for all matches.
    """
    # Escape special regex characters in the span text
    escaped_span = re.escape(span_text.strip())

    # Find all matches (case-insensitive, word boundaries optional)
    matches = []
    for match in re.finditer(escaped_span, text, re.IGNORECASE):
        start, end = match.span()
        matches.append((start, end))

    return matches


def fix_span_indices(spans: list, sentence_text: str) -> List[Dict]:
    """
    Fix span indices using regex matching.
    Returns updated spans with correct start_char and end_char.
    """

    fixed_spans = []
    for span in spans:
        span_text = span.get("text", "").strip()
        if not span_text:
            continue

        positions = find_span_positions(sentence_text, span_text)
        if positions:
            # Use the first available position
            start, end = positions[0]
            fixed_span = dict(span)
            fixed_span["start_char"] = start
            fixed_span["end_char"] = end
            fixed_span["text"] = sentence_text[start:end]  # Use actual text from sentence
            fixed_spans.append(fixed_span)
        else:
            # Span text not found in sentence - log warning but keep original
            print(f"WARNING: Could not find span '{span_text}' in sentence: {sentence_text}")
            fixed_span = dict(span)
            fixed_span["start_char"] = 0
            fixed_span["end_char"] = 0
            fixed_span["text"] = span_text
            fixed_spans.append(fixed_span)

    return fixed_spans


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


def call_llm_batch(client, model_name: str, model_type: str, requests: List[Dict]) -> List[Dict]:
    """Process multiple LLM requests efficiently."""
    results = []

    # Configure model-specific parameters
    if model_type in ["qwen3-32B", "gpt4o"]:
        max_tokens = 1024
    else:
        max_tokens = 500


    for req in requests:
        sentence = req["sentence"]
        candidates = req.get("candidates")

        if candidates is None:
            system_prompt = NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
            user_payload = {"sentence": sentence}
        else:
            # Determine whether candidates include NER-proposed types
            has_types = any("type" in c and c["type"] for c in candidates)
            if has_types:
                # NER-aware path: include proposed_type
                system_prompt = NER_AWARE_SYSTEM_PROMPT
                cand_objs = []
                for c in candidates:
                    obj = {
                        "text": c["text"].strip(),
                        "start_char": c["start_char"],
                        "end_char": c["end_char"],
                        "proposed_type": c.get("type", None)
                    }
                    cand_objs.append(obj)
                user_payload = {"sentence": sentence, "candidates": cand_objs}
            else:
                # Legacy chunks path: no types proposed
                system_prompt = DEFAULT_SYSTEM_PROMPT
                cand_objs = [
                    {"text": c["text"].strip(), "start_char": c["start_char"], "end_char": c["end_char"]}
                    for c in candidates
                ]
                user_payload = {"sentence": sentence, "candidates": cand_objs}


        # Modify system prompt for Qwen 32B
        if model_type == "qwen3-32B":
            system_prompt = f"<no_think/>\n\n{system_prompt}"

        full_prompt = f"{system_prompt}\n\nUser input: {json.dumps(user_payload, ensure_ascii=False)}"

        try:
            print(f'Sending LLM request for sentence: {sentence[:50]}... with {len(candidates) if candidates else 0} candidates')

            # if candidates and len(candidates) >= 10:
            #     print(full_prompt)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=800

            )
            print('LLM response received.')
            content = response.choices[0].message.content

            # Postprocess content for Qwen 32B to remove thinking blocks
            if model_type == "qwen3-32B" or model_type == "gpt4o":
                content = remove_thinking_blocks(content)

            llm_result = json.loads(content)

            # Fix indices for all span categories
            for category in ["accepted", "missing", "rejected"]:
                if category in llm_result:
                    llm_result[category] = fix_span_indices(
                        llm_result[category], sentence)

            results.append(llm_result)

        except Exception as e:
            print(f"Error calling LLM for sentence: {sentence[:50]}...: {e}")
            results.append({
                "accepted": [], "rejected": [], "missing": [],
                "notes": f"llm_error: {repr(e)}"
            })

    return results


def process_with_chunks(sent):
    """ Extract noun phrase candidates from sentence """
    cands = []
    for np in sent.noun_chunks:
        if np.root.pos_ not in ("NOUN", "PROPN"):
            continue
        np_text = np.text.strip()
        if not np_text:
            continue

        cands.append({
            "start_char": np.start_char,
            "end_char": np.end_char,
            "text": np_text
        })
    return cands


def _load_labels_from_model(model_dir):
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


def _load_ner(model_dir: str):
    # Backwards-compat shim if other code relies on this name
    infer = NerInferencer(model_dir, dtype="auto")
    return infer


def _normalize_bioc_spans(spans: List[Dict[str, Any]], sent_text: str) -> List[Dict[str, Any]]:
    """Map BioC spans to the schema your LLM path expects, then fix indices."""
    norm_spans = []
    for s in spans:
        if "text" in s and "start" in s and "end" in s:
            norm_spans.append({
                "text": s["text"].strip(),
                "start_char": int(s["start"]),
                "end_char": int(s["end"]),
                "type": s.get("type"),  # may be None; prompt handles both
            })
    return fix_span_indices(norm_spans, sent_text)


def _build_np_fallback(
    sentence_batch: List[str],
    spacy_model: str,
    empty_idx: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    """Compute chunk candidates only for sentences where NER found nothing."""

    fallback_maps: Dict[int, List[Dict[str, Any]]] = {}
    if not empty_idx:
        return fallback_maps

    nlp = get_spacy_model(spacy_model)
    empty_sents = [sentence_batch[i] for i in empty_idx]
    empty_docs = list(nlp.pipe(empty_sents))

    for j, doc in enumerate(empty_docs):
        i = empty_idx[j]
        # keep chunk candidates untyped; LLM prompt will use DEFAULT path
        fallback_maps[i] = process_with_chunks(doc)
    return fallback_maps


def process_sentences_batch(
    sentence_batch: List[str],
    spacy_model: str,
    use_chunks: bool,
    candidates_from: str = "ner",
    ner_runtime: Optional["NerInferencer"] = None,
    ner_max_length: int = 512,
    ner_runtime_batch_size: int = 64,
    np_fallback: bool = False,
    bioc_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    gazetteer_matcher: GazetteerMatcher | None = None,
    use_bioc: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build candidates for a batch of sentences.

    When candidates_from == "ner", this uses a shared NerInferencer to produce
    BIO→char spans (with types) directly from the NER model, so the same runtime
    is used across pipeline and eval.

    Returns:
        List[{"sentence": <str>, "candidates": List[{"start_char": int,
                                                     "end_char": int,
                                                     "text": str,
                                                     "type": str}]}]
        for NER mode. For other modes, keep your previous structure.
    """
    batch_results: List[Dict[str, Any]] = []

    if candidates_from == "bioc":
        if bioc_index is None:
            raise ValueError("candidates_from='bioc' requires bioc_index (load with _load_bioc_index).")
        for sent_text in sentence_batch:
            spans = bioc_index.get(sent_text, [])
            norm_spans = _normalize_bioc_spans(spans, sent_text)
            batch_results.append({"sentence": sent_text, "candidates": norm_spans if norm_spans else None})
        print(f'Processed {len(batch_results)} sentences with bioc candidates')
        return batch_results


    # All-chunks mode
    if candidates_from == "chunks" and use_chunks:
        nlp = get_spacy_model(spacy_model)
        docs = list(nlp.pipe(sentence_batch))
        for sent_text, sent_doc in zip(sentence_batch, docs):
            cands = process_with_chunks(sent_doc)

            if gazetteer_matcher is not None:
                gz = gazetteer_candidates(sent_text, gazetteer_matcher)
                if len(gz):
                    print(gz)
                    cands.extend(gz)

            if not cands:
                continue
            batch_results.append({"sentence": sent_text, "candidates": cands})
        print(f'Processed {len(batch_results)} sentences with spaCy chunks')
        return batch_results


    # NER mode (+ optional NP fallback)
    if candidates_from == "ner":
        if ner_runtime is None:
            raise ValueError(
                "NER candidates requested but ner_runtime is None. "
                "Initialize with NerInferencer(model_dir) and pass it in."
            )

        spans_lists = ner_runtime.predict_spans_for_sentences(
            sentences=sentence_batch,
            batch_size=ner_runtime_batch_size,
            max_length=ner_max_length,
            entity_threshold=0.25,
            entity_bias=0.25
        )

        # Collect indices with no NER spans (eligible for fallback)
        empty_idx = [i for i, spans in enumerate(spans_lists) if not spans]

        fallback_maps: Dict[int, List[Dict[str, Any]]] = {}
        if np_fallback and empty_idx:
            fallback_maps = _build_np_fallback(sentence_batch, spacy_model, empty_idx)

        # build unified results
        for i, sent_text in enumerate(sentence_batch):
            spans = spans_lists[i]

            gz = gazetteer_candidates(sent_text, gazetteer_matcher) if gazetteer_matcher is not None else []

            if spans:
                # Combine NER + gazetteer candidates
                if len(gz):
                    spans = spans + gz
                # Also add BioC spans if available
                if use_bioc and bioc_index:
                    spans = bioc_index.get(sent_text, [])
                    norm_spans = _normalize_bioc_spans(spans, sent_text)
                    if norm_spans:
                        spans = spans + norm_spans
                batch_results.append({"sentence": sent_text, "candidates": spans})
            else:
                # Fallback (if any), else leave candidates=None
                if np_fallback:
                    cands = fallback_maps.get(i, [])
                    if gz:
                        cands = cands + gz
                    if cands:
                        batch_results.append({"sentence": sent_text, "candidates": cands})
                    else:
                        batch_results.append({"sentence": sent_text, "candidates": None})
                else:
                    if gz:
                        batch_results.append({"sentence": sent_text, "candidates": gz})
                    else:
                        batch_results.append({"sentence": sent_text, "candidates": None})
        return batch_results

    # candidates_from == "none"
    for sent_text in sentence_batch:
        batch_results.append({"sentence": sent_text, "candidates": None})
        print(f'Processed {len(batch_results)} sentences without candidates')
    return batch_results



_WS = re.compile(r"\s+")
_PUNCT = str.maketrans({
    "“":"\"", "”":"\"", "‘":"'", "’":"'",
    "–":"-", "—":"-", "−":"-", "…":"...",
    "\u00A0":" ", "\u2009":" ", "\u200A":" ", "\u200B":" ",
})


def _canon(s: str) -> str:
    # same canonicalization as for gold: normalize, collapse space, lowercase
    s = unicodedata.normalize("NFKC", s).translate(_PUNCT)
    s = _WS.sub(" ", s).strip().lower()
    return s



def _dedupe_bioc_index(
    bioc_index: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Merge entries for identical sentences (by canonical form), prefer the first
    exact text seen as the key, union spans, remove duplicate spans.
    """
    # map canon -> (original_text_key, merged_spans_list)
    tmp: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}

    def _span_key(sp):
        # dedupe by geometry + text + mapped type
        return (int(sp["start"]), int(sp["end"]),
                sp.get("text",""), sp.get("type"))

    for sent_text, spans in bioc_index.items():
        c = _canon(sent_text)
        if c not in tmp:
            tmp[c] = (sent_text, [])
        key_text, merged = tmp[c]
        # extend
        merged.extend(spans)

    deduped: Dict[str, List[Dict[str, Any]]] = {}
    for c, (orig_text, merged) in tmp.items():
        # remove duplicate spans
        seen = set()
        uniq = []
        for sp in merged:
            k = _span_key(sp)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(sp)
        # sort deterministic
        uniq.sort(key=lambda s: (int(s["start"]), -(int(s["end"]) - int(s["start"]))))
        deduped[orig_text] = uniq
    return deduped



def gazetteer_candidates(sentence: str, gazetteer: GazetteerMatcher) -> list[dict]:
    if gazetteer is None:
        return []
    hits = gazetteer.match(sentence)  # already returns start/end/text/type/source/rule_id
    # normalize + tag provenance
    out = []
    for h in hits:
        out.append({
            "start_char": int(h["start_char"]),
            "end_char": int(h["end_char"]),
            "text": h["text"],
            "type": h.get("type"),              # filename-as-type (e.g., 'Mountain', 'Biome', ...)
            "source": "gazetteer",
            "meta": {"rule_id": h.get("rule_id"), "backend": h.get("source")},
        })
    return out



def main():
    global client, model_name

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with .txt documents")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (one object per document)")
    ap.add_argument("--model_type", choices=["qwen3-4B", "qwen3-32B", "gpt4o", "biomistral-7b-awq"], default="gpt4o", help="LLM model to use")
    ap.add_argument("--use_chunks", action="store_true", help="Use noun phrase chunks as candidates")
    ap.add_argument("--max_sents_per_doc", type=int, default=999999, help="Cap sentences per doc (debug)")
    ap.add_argument("--sample_every", type=int, default=1, help="Process every Nth sentence (e.g., 5 to sample)")
    ap.add_argument("--batch_size", type=int, default=15, help="Batch size for processing")
    ap.add_argument("--max_workers", type=int, default=4, help="Max worker threads")

    # ==== Candidate source switch ====
    ap.add_argument("--candidates_from", choices=["chunks", "ner", "bioc"], default="ner",
                    help="Generate LLM candidates from spaCy noun chunks or NER predictions.")

    # ==== BioC ====
    ap.add_argument("--bioc_candidates_dir", type=str, default=None,
                    help = "Directory with BioC JSON files. We will extract sentence->spans as candidates.")
    ap.add_argument("--use_bioc", action="store_true", help="When using NER candidates, also add BioC spans if available.")

    ap.add_argument("--schema_csv", type=str, default=None,
                    help="Optional schema CSV passed to the converter.")
    ap.add_argument("--envo_gazetteer_csv", type=str, default=None,
                    help="Optional ENVO gazetteer CSV passed to the converter.")
    ap.add_argument("--prefer_spacy", action="store_true",
                    help="Prefer spaCy PhraseMatcher inside the converter.")

    # ==== NER path params ====
    ap.add_argument("--ner_model_dir", type=str, default=None,
                    help="HF token classification model directory (must contain labels.txt).")
    ap.add_argument("--ner_batch_size", type=int, default=16)
    ap.add_argument("--ner_max_length", type=int, default=512)


    # ==== spaCy chunker params ===
    ap.add_argument("--spacy_model", default="en_core_web_trf", help="spaCy model (needs parser for noun_chunks)")

    ap.add_argument("--np_fallback", action="store_true",
                    help="If NER finds no entities in a sentence, fill candidates with NP chunks.")

    # ==== Gazetteer ====
    ap.add_argument("--gaz_dir", type=str, default=None,
                    help="Directory with CSV/TSV gazetteers; filename is used as entity type.")

    args = ap.parse_args()


    # Initialize LLM client
    try:
        client, model_name = get_openai_client(args.model_type)
        print(f"Using {args.model_type} model: {model_name}")
    except Exception as e:
        print(f"Error initializing {args.model_type} client: {e}", file=sys.stderr)
        sys.exit(1)

    if args.gaz_dir:
        gaz_rules = load_gaz_rules_from_dir(
            dir_path=args.gaz_dir,
        )
        gaz_matcher = GazetteerMatcher(gaz_rules)
        print(f"[gazetteer] Loaded {len(gaz_rules)} rules from {args.gaz_dir} "
              f"(phrase={len(gaz_matcher.phrase_rules)}, regex={len(gaz_matcher.regex_rules)})")
    else:
        gaz_matcher = None

    # Validate spaCy model
    if args.use_chunks:
        try:
            test_nlp = spacy.load(args.spacy_model)
            if "parser" not in test_nlp.pipe_names:
                print("WARNING: spaCy parser not enabled; noun_chunks may be empty.", file=sys.stderr)
        except OSError:
            print(f"spaCy model '{args.spacy_model}' not found. Install with: python -m spacy download {args.spacy_model}",
                  file=sys.stderr)
            sys.exit(1)

    Path(os.path.dirname(args.out_jsonl) or ".").mkdir(parents=True, exist_ok=True)

    # Load BioC candidates if requested
    bioc_index = None
    if args.candidates_from == "bioc":
        if not args.bioc_candidates_dir:
            print("Provide --bioc_candidates_dir for candidates_from=bioc", file=sys.stderr)
            sys.exit(1)
        bioc_index = _load_bioc_index_from_dir(args.bioc_candidates_dir)
        print(f"Loaded BioC candidates for {len(bioc_index)} sentences from {args.bioc_candidates_dir}")

        before = len(bioc_index)
        bioc_index = _dedupe_bioc_index(bioc_index)
        after = len(bioc_index)
        print(f"[BioC] deduped unique sentences: {after} (from {before})")

    # NER runtime (loaded once)
    ner_runtime = None
    if args.candidates_from == "ner":
        if not args.ner_model_dir:
            print("Provide --ner_model_dir for candidates_from=ner", file=sys.stderr)
            sys.exit(1)
        ner_runtime = _load_ner(args.ner_model_dir)

    docs_written = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for doc_id, text in read_txt_files(args.in_dir):
            if docs_written > 1:  # Debug limit
                break

            # Split text into lines (one sentence per line)
            lines = text.strip().split('\n')
            sentences = [line.strip() for line in lines if line.strip()]

            out_sents = []
            print(f"Processing {len(sentences)} sentences from lines")

            total_batches = (len(sentences) + args.batch_size - 1) // args.batch_size

            for bidx, i in enumerate(range(0, len(sentences), args.batch_size), start=1):
                batch = sentences[i:i + args.batch_size]

                # === Candidate generation in-batch (spaCy tokenization for both modes)
                candidate_results = process_sentences_batch(
                    batch,
                    args.spacy_model,
                    args.use_chunks,
                    candidates_from=args.candidates_from,
                    ner_runtime=ner_runtime,
                    ner_max_length=args.ner_max_length,
                    ner_runtime_batch_size=args.ner_batch_size,
                    np_fallback=args.np_fallback,
                    bioc_index=bioc_index,
                    gazetteer_matcher=gaz_matcher,
                    use_bioc=args.use_bioc
                )

                assert len(candidate_results) == len(batch), \
                    f"Lost sentences in candidate building: {len(batch)} -> {len(candidate_results)}"

                llm_results = call_llm_batch(client, model_name, args.model_type, candidate_results)
                if len(llm_results) != len(candidate_results):
                    raise RuntimeError(f"LLM results mismatch: {len(llm_results)} vs {len(candidate_results)}")

                for spacy_result, llm_result in zip(candidate_results, llm_results):
                    sentence_data = {
                        "text": spacy_result["sentence"],
                        "llm": llm_result
                    }
                    if spacy_result["candidates"] is not None:
                        sentence_data["candidates"] = spacy_result["candidates"]
                    out_sents.append(sentence_data)

                print(f"Processed batch {bidx}/{total_batches}")

                # Write results
            rec = {
                "doc_id": doc_id,
                "sentences": out_sents,
                "config": {
                    "model_type": args.model_type,
                    "model_name": model_name,
                    "use_chunks": args.use_chunks
                }
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_written += 1
            print(f"Completed document {doc_id} with {len(out_sents)} sentences")


    mode_str = "with chunks" if args.use_chunks else "without chunks"
    print(f"Done. Processed {docs_written} documents using {args.model_type} {mode_str} and wrote to {args.out_jsonl}")


if __name__ == "__main__":
    main()