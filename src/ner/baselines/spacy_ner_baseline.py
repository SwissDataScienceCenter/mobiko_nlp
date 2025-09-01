import os
import csv
import json
import argparse
from collections import defaultdict, Counter

"""
Note: Just general NER model won't serve us well for the task. It extracts too many irrelevant entities and will skip
the ones we are really after, e.g., common animal names. Usually generic NER models are pretrained on PERSON/LOCATION/etc
data format, which doesn't serve our needs.
"""


import spacy


def norm(s: str) -> str:
    return " ".join((s or "").lower().split()).strip()


def read_txt_files(indir: str):
    for filename in os.listdir(indir):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(indir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            yield os.path.splitext(filename)[0], f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with .txt documents")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (doc-per-line)")
    ap.add_argument("--model", default="en_core_web_sm", help="spaCy model (e.g., en_core_web_sm or en_core_web_trf)")
    args = ap.parse_args()

    # Load spaCy model
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"spaCy model '{args.model}' not found. Install with: python -m spacy download {args.model}")
        return

    if "ner" not in nlp.pipe_names:
        print("WARNING: selected spaCy model has no NER component; doc.ents will be empty.")

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    unk_counter = Counter()

    docs_written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for doc_id, text in read_txt_files(args.in_dir):
            doc = nlp(text)
            out_sents = []

            for sent in doc.sents:
                # Collect spaCy entity spans that fall inside this sentence
                spans = []
                for ent in doc.ents:
                    if ent.start_char < sent.start_char or ent.end_char > sent.end_char:
                        continue
                    ent_text = ent.text.strip()
                    if not ent_text:
                        continue

                    spans.append({
                        "start_char": ent.start_char - sent.start_char,
                        "end_char":   ent.end_char   - sent.start_char,
                        "text": ent_text,
                        "spacy_label": ent.label_  # keep for auditing
                    })

                out_sents.append({
                    "text": sent.text,
                    "spans": spans
                })

            rec = {"id": doc_id, "sentences": out_sents}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_written += 1

    print(f"Done. Wrote {docs_written} documents to {args.out_jsonl}")


if __name__ == "__main__":
    main()
