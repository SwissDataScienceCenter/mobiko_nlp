# src/ner/bert_ner_baseline.py
import os
import re
import json
import argparse
from pathlib import Path
import nltk
nltk.download('punkt_tab')

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

try:
    nltk.download('punkt', quiet=True)
except Exception:
    print("Warning: Could not download NLTK punkt tokenizer")


def nltk_sentences(text: str):
    return nltk.sent_tokenize(text)


def read_txts(indir: str):
    for name in os.listdir(indir):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(indir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield os.path.splitext(name)[0], f.read()
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with .txt documents")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (one JSON per document)")
    ap.add_argument("--batch_size", type=int, default=16, help="Pipeline batch size (sentences)")
    ap.add_argument("--device", type=int, default=-1, help="GPU id (e.g., 0) or -1 for CPU")
    args = ap.parse_args()

    try:
        # WARNING: I discovered that this model was not trained for NER, looks like a base LM. Need to replace with other model
        tokenizer = AutoTokenizer.from_pretrained("NoYo25/BiodivBERT")
        model = AutoModelForTokenClassification.from_pretrained("NoYo25/BiodivBERT")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # merges subword tokens into entities
        device=args.device
    )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs_written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for doc_id, text in read_txts(args.in_dir):
            sentences = nltk_sentences(text)
            out_sents = []
            for i in range(0, len(sentences), args.batch_size):
                batch = sentences[i:i+args.batch_size]
                results = ner(batch)  # returns list[list[ent]] aligned to batch
                for sent_text, ents in zip(batch, results):
                    # Convert HF entities to span records (relative to sentence)
                    spans = []
                    for e in ents:
                        # HF returns absolute indices within the given string (the sentence here)
                        spans.append({
                            "start_char": int(e["start"]),
                            "end_char": int(e["end"]),
                            "text": sent_text[e["start"]:e["end"]],
                            "label": e.get("entity_group") or e.get("entity"),
                            "score": float(e["score"])
                        })
                    if spans:
                        out_sents.append({"text": sent_text, "spans": spans})

            rec = {"id": doc_id, "sentences": out_sents}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_written += 1

    print(f"Done. Wrote {docs_written} documents → {args.out_jsonl}")


if __name__ == "__main__":
    main()
