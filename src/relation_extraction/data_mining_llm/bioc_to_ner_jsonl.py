import argparse
import json
import gzip
import os
import re
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path
import nltk


src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

print(sys.path)

from ner.ner_infer import NerInferencer


def _open_maybe_gz(path: str, mode: str = "rt", encoding: str = "utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return open(path, mode, encoding=encoding)



def split_into_sentences(text: str) -> List[str]:
    sent_text = nltk.sent_tokenize(text)
    return sent_text


def get_paper_id(article: Dict[str, Any]) -> str:
    doc = article.get("document", {})
    # Prefer PMC-style IDs, then pmid/doi, then _id as fallback
    for key in ("pmcid", "pmid", "doi", "_id"):
        val = doc.get(key) or article.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return article.get("_id", "UNKNOWN")


def collect_sentences_from_article(article: Dict[str, Any]) -> List[str]:
    """
    Collect sentence texts from:
      - document title
      - document abstract
      - body sections (excluding title/abstract duplicates)
    Returns a flat list of sentences in reading order.
    """
    doc = article.get("document", {})
    sentences: List[str] = []

    # Title
    title = doc.get("title")
    if isinstance(title, str) and title.strip():
        sentences.extend(split_into_sentences(title))

    # Abstract (document-level string)
    abstract = doc.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        sentences.extend(split_into_sentences(abstract))

    # Body sections – skip sections that are just "Title" or "Abstract"
    body_sections = doc.get("body_sections", [])
    for sec in body_sections:
        sec_title = (sec.get("title") or "").strip()
        sec_tag = (sec.get("tag") or "").strip().lower()

        # avoid duplicating title/abstract text
        if sec_title.lower() in {"title", "abstract"} or sec_tag == "abstract":
            continue

        contents = sec.get("contents", [])
        for content in contents:
            text = content.get("text")
            if not isinstance(text, str):
                continue
            for sent in split_into_sentences(text):
                sentences.append(sent)

    return sentences


def iter_input_files(input_path: str) -> List[str]:
    """
    If input_path is a file → [file].
    If directory → all .json / .json.gz under it.
    """
    if os.path.isfile(input_path):
        return [input_path]
    paths: List[str] = []
    print(input_path)
    for root, _, files in os.walk(input_path):
        for name in files:
            if name.endswith(".json") or name.endswith(".json.gz"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def annotate_to_jsonl(
    input_path: str,
    output_jsonl: str,
    model_dir: str,
    batch_size: int = 128,
    max_length: int = 256,
    entity_threshold: Optional[float] = None,
    entity_bias: Optional[float] = None,
) -> None:
    """
    Main driver:
      - read BIOC files
      - extract sentences
      - run NER
      - write JSONL with desired schema
    """
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    inferencer = NerInferencer(model_dir=model_dir)

    input_files = iter_input_files(input_path)
    if not input_files:
        raise SystemExit(f"No BIOC JSON(.gz) files found under {input_path}")

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        global_paper_count = 0

        for path_idx, in_path in enumerate(input_files, 1):
            print(f"[{path_idx}/{len(input_files)}] Processing {in_path}")
            try:
                with _open_maybe_gz(in_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)

                articles = data.get("sibils_article_set", [])
                if not isinstance(articles, list):
                    continue

                for art_idx, article in enumerate(articles, 1):
                    paper_id = get_paper_id(article)
                    sents = collect_sentences_from_article(article)
                    if not sents:
                        continue

                    spans_per_sent = inferencer.predict_spans_for_sentences(
                        sents,
                        batch_size=batch_size,
                        max_length=max_length,
                        entity_threshold=entity_threshold,
                        entity_bias=entity_bias,
                    )

                    # sentence IDs are local within each paper
                    for sent_id, (sent_text, spans) in enumerate(
                        zip(sents, spans_per_sent)
                    ):
                        entities = []
                        for ent_idx, span in enumerate(spans, 1):
                            entities.append(
                                {
                                    "id": f"T{ent_idx}",
                                    "text": span.get("text", ""),
                                    "type": span.get("type", ""),
                                    "start": span.get("start_char"),
                                    "end": span.get("end_char"),
                                }
                            )
                        record = {
                            "paper_id": paper_id,
                            "sent_id": sent_id,
                            "text": sent_text,
                            "entities": entities,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    global_paper_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to process {in_path}: {e}", file=sys.stderr)

        print(f"Done. Wrote JSONL for {global_paper_count} papers to {output_jsonl}")


def main():
    ap = argparse.ArgumentParser(
        description="Run NER on SIBiLS BIOC files and dump sentence-level JSONL."
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Path to fine-tuned NER model directory",
    )
    ap.add_argument(
        "--input-path",
        required=True,
        help="BIOC file or directory with BIOC files (.json/.json.gz)",
    )
    ap.add_argument(
        "--output-jsonl",
        required=True,
        help="Path to output JSONL file",
    )

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument(
        "--entity-threshold",
        type=float,
        default=None,
        help="If set, probability threshold for entity labels; below this becomes 'O'.",
    )
    ap.add_argument(
        "--entity-bias",
        type=float,
        default=None,
        help="Optional logit bias added to entity labels before prediction.",
    )

    args = ap.parse_args()

    annotate_to_jsonl(
        input_path=args.input_path,
        output_jsonl=args.output_jsonl,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        entity_threshold=args.entity_threshold,
        entity_bias=args.entity_bias,
    )


if __name__ == "__main__":
    main()
