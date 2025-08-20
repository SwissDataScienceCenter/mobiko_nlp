import os
import sys
import json
import time
import argparse
from pathlib import Path

import spacy
from openai import OpenAI



from promps import DEFAULT_SYSTEM_PROMPT, NO_CHUNK_CANDIDATE_SYSTEM_PROMPT

# Model configurations
MODEL_CONFIGS = {
    "qwen": {
        "base_url": "https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507"
    },
    "gpt4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": None,  # Will use OPENAI_API_KEY env var
        "model_name": "gpt-4o"
    }
}


def get_openai_client(model_type: str):
    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"Unknown model type: {model_type}. Use: {list(MODEL_CONFIGS.keys())}")

    api_key = config["api_key"] or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"API key required for {model_type}. Set OPENAI_API_KEY environment variable.")

    return OpenAI(
                base_url=config["base_url"],
                api_key=api_key
                ), config["model_name"]


def read_txt_files(indir: str):
    for name in os.listdir(indir):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(indir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            yield os.path.splitext(name)[0], f.read()


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
            "candidates": [{"text": c["text"], "start_char": c["start_char"], "end_char": c["end_char"]} for c in candidates],
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
            return json.loads(content)

        except Exception as e:
            if attempt == retries - 1:
                return {"accepted": [], "rejected": [], "missing": [], "notes": f"llm_error: {repr(e)}"}
            time.sleep(sleep_s * (attempt + 1))


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
            "start_char": np.start_char - sent.start_char,
            "end_char": np.end_char - sent.start_char,
            "text": np_text
        })
    return cands


def clamp_span(x, sent_text):
    """Clamp spans to sentence bounds"""
    s = max(0, int(x.get("start_char", 0)))
    e = min(len(sent_text), int(x.get("end_char", s)))
    y = dict(x)
    y["start_char"] = s
    y["end_char"] = e
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with .txt documents")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (one object per document)")
    ap.add_argument("--model_type", choices=["qwen", "gpt4o"], default="qwen", help="LLM model to use")
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

            doc = nlp(text)
            out_sents = []

            for i, sent in enumerate(doc.sents):
                if i >= args.max_sents_per_doc:
                    break
                if (i % args.sample_every) != 0:
                    continue

                if args.use_chunks:
                    # Chunk-based processing with DEFAULT_SYSTEM_PROMPT
                    cands = process_with_chunks(sent)
                    if not cands:
                        continue  # Skip sentences without NP candidates

                    verdict = call_llm(client, model_name, sent.text, cands)
                    print(f'VERDICT: {verdict}')

                    # Clamp spans to sentence bounds
                    verdict["accepted"] = [clamp_span(x, sent.text) for x in verdict.get("accepted", []) if x.get("text")]
                    verdict["missing"] = [clamp_span(x, sent.text) for x in verdict.get("missing", []) if x.get("text")]

                    out_sents.append({
                        "text": sent.text,
                        "candidates": cands,
                        "llm": verdict
                    })
                else:
                    # No-chunk processing with NO_CHUNK_CANDIDATE_SYSTEM_PROMPT
                    verdict = call_llm(client, model_name, sent.text, None)
                    print(f'VERDICT: {verdict}')

                    # For no-chunk mode, clamp accepted entities only
                    verdict["accepted"] = [clamp_span(x, sent.text) for x in verdict.get("accepted", []) if x.get("text")]

                    out_sents.append({
                        "text": sent.text,
                        "llm": verdict
                    })

            # Write one JSON object per document
            rec = {
                "id": doc_id,
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