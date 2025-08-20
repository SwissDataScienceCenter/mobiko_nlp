import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict


def _merge_source_lists(existing: List[Dict[str, str]], incoming: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge and deduplicate sources by (source, source_id) preserving order."""
    seen = {(s.get("source"), s.get("source_id")) for s in existing}
    for s in incoming:
        key = (s.get("source"), s.get("source_id"))
        if key not in seen:
            existing.append({"source": s.get("source"), "source_id": s.get("source_id")})
            seen.add(key)
    return existing


def build_json_from_csv(
    csv_path: Path,
    doc_id: Optional[str] = None,
    case_insensitive: bool = True,
    aggregate_repeating_texts: bool = True,
) -> Dict[str, Any]:

    if doc_id is None:
        doc_id = csv_path.stem

    # We'll aggregate by sentence text using an OrderedDict to preserve first-appearance order.
    # Each value keeps a sentence dict and a span index for fast merging.
    sentences_by_text: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    flags = re.IGNORECASE if case_insensitive else 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"line", "keyword", "source", "source_id"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            line = (row.get("line") or "").strip()
            keyword = (row.get("keyword") or "").strip()
            source = (row.get("source") or "").strip()
            source_id = (row.get("source_id") or "").strip()

            if not line and not keyword:
                continue

            # Choose or create the target sentence bucket
            if line:
                text_key = line if aggregate_repeating_texts else None
            else:
                continue

            if aggregate_repeating_texts:
                # Create or get the aggregated sentence entry
                if text_key not in sentences_by_text:
                    sentences_by_text[text_key] = {
                        "sentence": {"text": line, "spans": []},
                        "span_index": {}  # (start,end,keyword_lower,text) -> span dict
                    }
                target = sentences_by_text[text_key]
            else:
                # Fallback (not typically used): create a fresh sentence per appearance
                # with a unique key to keep order (text plus running index).
                unique_key = f"{line}\n__offset__{len([k for k in sentences_by_text if k.startswith(line)])}"
                sentences_by_text[unique_key] = {
                    "sentence": {"text": line, "spans": []},
                    "span_index": {}
                }
                target = sentences_by_text[unique_key]

            if not keyword:
                continue

            # Find all occurrences of keyword in the sentence text
            pattern = re.compile(re.escape(keyword), flags=flags)
            for m in pattern.finditer(target["sentence"]["text"]):
                start, end = m.start(), m.end()
                key = (
                    start,
                    end,
                    keyword.lower() if case_insensitive else keyword,
                    target["sentence"]["text"][start:end],
                )
                if key in target["span_index"]:
                    span_obj = target["span_index"][key]
                    _merge_source_lists(span_obj["sources"], [{"source": source, "source_id": source_id}])
                else:
                    span_obj = {
                        "text": target["sentence"]["text"][start:end],
                        "start_char": start,
                        "end_char": end,
                        "keyword": keyword,
                        "sources": [{"source": source, "source_id": source_id}],
                    }
                    target["sentence"]["spans"].append(span_obj)
                    target["span_index"][key] = span_obj

    # Finalize output
    out_sentences = []
    for entry in sentences_by_text.values():
        sent = entry["sentence"]
        # Sort spans for determinism
        sent["spans"].sort(key=lambda s: (s.get("start_char", -1), s.get("end_char", -1), s.get("keyword") or "", s.get("text") or ""))
        out_sentences.append(sent)

    data: Dict[str, Any] = {"doc_id": doc_id, "sentences": out_sentences}
    return data


def main():
    ap = argparse.ArgumentParser(description="Convert CSV to JSON with merged sources per span and aggregation across repeating sentence texts.")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV path.")
    ap.add_argument("-o", "--output", type=Path, help="Output JSON path (default: <csv_stem>.json)")
    ap.add_argument("--doc_id", type=str, default=None, help="doc_id for the JSON (default: CSV filename stem)")
    ap.add_argument("--case-sensitive", action="store_true", help="Case-sensitive keyword matching.")
    ap.add_argument("--no-aggregate", action="store_true", help="Do not aggregate same sentence texts (keeps duplicates)." )
    args = ap.parse_args()

    data = build_json_from_csv(
        args.csv,
        doc_id=args.doc_id,
        case_insensitive=not args.case_sensitive,
        aggregate_repeating_texts=not args.no_aggregate,
    )

    out_path = args.output or args.csv.with_suffix(".json")
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
