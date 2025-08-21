import os
import sys
import json
import time
import argparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import threading

import spacy
from openai import OpenAI


from promps import DEFAULT_SYSTEM_PROMPT, NO_CHUNK_CANDIDATE_SYSTEM_PROMPT


# Model configurations
MODEL_CONFIGS = {
    "qwen3-4B": {
        "base_url": "https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507"
    },
    "qwen3-32B": {
        "base_url": "curl -X POST https://openwebui.runai-codev-llm.inference.compute.datascience.ch/v1",
        "api_key": None,
        "model_name": "Qwen/Qwen3-32B-AWQ"
    },
    "gpt4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": None,  # Will use OPENAI_API_KEY env var
        "model_name": "gpt-4o"
    },

}

# Thread-local storage for spaCy models
thread_local = threading.local()


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


def find_span_positions(text: str, span_text: str, used_positions: set = None):
    """
    Find all positions of span_text in text using regex.
    Returns list of (start, end) tuples for all matches not in used_positions.
    """
    if used_positions is None:
        used_positions = set()

    # Escape special regex characters in the span text
    escaped_span = re.escape(span_text.strip())

    # print(used_positions)
    # Find all matches (case-insensitive, word boundaries optional)
    matches = []
    for match in re.finditer(escaped_span, text, re.IGNORECASE):
        start, end = match.span()
        # if (start, end) not in used_positions:
        matches.append((start, end))

    return matches


def fix_span_indices(spans: list, sentence_text: str, used_positions: set = None) -> List[Dict]:
    """
    Fix span indices using regex matching.
    Returns updated spans with correct start_char and end_char.
    """
    if used_positions is None:
        used_positions = set()

    fixed_spans = []
    for span in spans:
        span_text = span.get("text", "").strip()
        if not span_text:
            continue

        positions = find_span_positions(sentence_text, span_text, used_positions)
        if positions:
            # Use the first available position
            start, end = positions[0]
            used_positions.add((start, end))
            fixed_span = dict(span)
            fixed_span["start_char"] = start
            fixed_span["end_char"] = end
            fixed_span["text"] = sentence_text[start:end]  # Use actual text from sentence
            fixed_spans.append(fixed_span)
        else:
            # Span text not found in sentence - log warning but keep original
            print(f"WARNING: Could not find span '{span_text}' in sentence: {sentence_text}...")

            fixed_span = dict(span)
            fixed_span["start_char"] = 0
            fixed_span["end_char"] = 0
            fixed_span["text"] = span_text
            fixed_spans.append(fixed_span)

    return fixed_spans


def call_llm(client, model_name: str, sentence: str, candidates: list = None, retries: int = 3, sleep_s: float = 3.0) -> dict:
    if candidates is None:
        # No-chunk mode: use NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
        system_prompt = NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
        user_payload = {"sentence": sentence}
    else:
        # Chunk mode: use DEFAULT_SYSTEM_PROMPT with candidates
        system_prompt = DEFAULT_SYSTEM_PROMPT
        user_payload = {
            "sentence": sentence,
            "candidates": [{"text": c["text"].strip(), "start_char": c["start_char"], "end_char": c["end_char"]} for c in candidates],
        }

    full_prompt = f"{system_prompt}\n\nUser input: {json.dumps(user_payload, ensure_ascii=False)}"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            print(f'RESPONSE: {response}')

            content = response.choices[0].message.content
            llm_result = json.loads(content)

            # Fix indices for all span categories using regex
            used_positions = set()

            for category in ["accepted", "missing", "rejected"]:
                if category in llm_result:
                    llm_result[category] = fix_span_indices(
                        llm_result[category], sentence, used_positions
                    )

            return llm_result

        except Exception as e:
            if attempt == retries - 1:
                return {"accepted": [], "rejected": [], "missing": [], "notes": f"llm_error: {repr(e)}"}
            time.sleep(sleep_s * (attempt + 1))



def call_llm_batch(client, model_name: str, requests: List[Dict]) -> List[Dict]:
    """Process multiple LLM requests efficiently."""
    results = []

    for req in requests:
        sentence = req["sentence"]
        candidates = req.get("candidates")

        if candidates is None:
            system_prompt = NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
            user_payload = {"sentence": sentence}
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT
            user_payload = {
                "sentence": sentence,
                "candidates": [{"text": c["text"].strip(), "start_char": c["start_char"], "end_char": c["end_char"]} for
                               c in candidates],
            }

        full_prompt = f"{system_prompt}\n\nUser input: {json.dumps(user_payload, ensure_ascii=False)}"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                max_tokens=2048,
            )

            content = response.choices[0].message.content
            llm_result = json.loads(content)

            # Fix indices for all span categories
            used_positions = set()
            for category in ["accepted", "missing", "rejected"]:
                if category in llm_result:
                    llm_result[category] = fix_span_indices(
                        llm_result[category], sentence, used_positions
                    )

            results.append(llm_result)

        except Exception as e:
            results.append({
                "accepted": [], "rejected": [], "missing": [],
                "notes": f"llm_error: {repr(e)}"
            })

    return results


def process_with_chunks(sent):
    """Extract noun phrase candidates from sentence"""
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



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with .txt documents")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (one object per document)")
    ap.add_argument("--model_type", choices= ['qwen3-4B', 'qwen3-32B', 'gpt4o'], default="qwen3-4B", help="LLM model to use")
    ap.add_argument("--use_chunks", action="store_true", help="Use noun phrase chunks as candidates")
    ap.add_argument("--spacy_model", default="en_core_web_trf", help="spaCy model (needs parser for noun_chunks)")
    ap.add_argument("--max_sents_per_doc", type=int, default=999999, help="Cap sentences per doc (debug)")
    ap.add_argument("--sample_every", type=int, default=1, help="Process every Nth sentence (e.g., 5 to sample)")
    args = ap.parse_args()


    # Initialize LLM client
    try:
        client, model_name = get_openai_client(args.model_type)
        print(f"Using {args.model_type} model: {model_name}")
    except Exception as e:
        print(f"Error initializing {args.model_type} client: {e}", file=sys.stderr)
        sys.exit(1)

    # Load spaCy model
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError:
        print(f"spaCy model '{args.spacy_model}' not found. Install with: python -m spacy download {args.spacy_model}",
              file=sys.stderr)
        sys.exit(1)

    # Check parser if using chunks
    if args.use_chunks and "parser" not in nlp.pipe_names:
        print("WARNING: spaCy parser not enabled; noun_chunks may be empty. Use a model with parser.", file=sys.stderr)

    Path(os.path.dirname(args.out_jsonl) or ".").mkdir(parents=True, exist_ok=True)

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


            for i, sent_text in enumerate(sentences):
                if i >= args.max_sents_per_doc:
                    break
                if (i % args.sample_every) != 0:
                    continue

                # Process each sentence with spaCy to get chunks
                sent_doc = nlp(sent_text)

                if args.use_chunks:
                    # Chunk-based processing with DEFAULT_SYSTEM_PROMPT
                    cands = process_with_chunks(sent_doc)
                    if not cands:
                        continue  # Skip sentences without NP candidates

                    verdict = call_llm(client, model_name, sent_text, cands)
                    print(f'VERDICT: {verdict}')


                    out_sents.append({
                        "text": sent_text,
                        "candidates": cands,
                        "llm": verdict
                    })
                else:
                    # No-chunk processing with NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
                    verdict = call_llm(client, model_name, sent_text, None)
                    print(f'VERDICT: {verdict}')


                    out_sents.append({
                        "text": sent_text,
                        "llm": verdict
                    })

            # Write one JSON object per document
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

    mode_str = "with chunks" if args.use_chunks else "without chunks"
    print(f"Done. Processed {docs_written} documents using {args.model_type} {mode_str} and wrote to {args.out_jsonl}")


if __name__ == "__main__":
    main()