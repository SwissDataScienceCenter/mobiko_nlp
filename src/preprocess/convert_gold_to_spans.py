import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional


def build_json_from_csv(
    csv_path: Path,
    doc_id: Optional[str] = None,
    case_insensitive: bool = True
) -> Dict[str, Any]:

    if doc_id is None:
        doc_id = csv_path.stem

    data: Dict[str, Any] = {"doc_id": doc_id, "sentences": []}

    current_sentence: Optional[Dict[str, Any]] = None

    flags = re.IGNORECASE if case_insensitive else 0

    # For tracking span index within a sentence: {(start, end, keyword_lower) : span_index}
    span_index_map = {}

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

            # Start a new sentence
            if line:
                if current_sentence is not None:
                    data["sentences"].append(current_sentence)
                current_sentence = {"text": line, "spans": []}
                span_index_map.clear()

            if current_sentence is None or not keyword:
                continue

            pattern = re.compile(re.escape(keyword), flags=flags)
            for m in pattern.finditer(current_sentence["text"]):
                start, end = m.start(), m.end()
                key = (start, end, keyword.lower() if case_insensitive else keyword)

                if key in span_index_map:
                    # Already have this occurrence → append source
                    idx = span_index_map[key]
                    current_sentence["spans"][idx]["sources"].append(
                        {"source": source, "source_id": source_id}
                    )
                else:
                    # New span occurrence
                    span_data = {
                        "text": current_sentence["text"][start:end],
                        "span_start": start,
                        "span_end": end,
                        "keyword": keyword,
                        "sources": [{"source": source, "source_id": source_id}],
                    }
                    current_sentence["spans"].append(span_data)
                    span_index_map[key] = len(current_sentence["spans"]) - 1

    if current_sentence is not None:
        data["sentences"].append(current_sentence)

    return data


def main():
    ap = argparse.ArgumentParser(description="Convert CSV to JSON with merged sources for each span occurrence.")
    ap.add_argument("--csv", type=Path, help="Input CSV path.")
    ap.add_argument("-o", "--output", type=Path, help="Output JSON path (default: <csv_stem>.json)")
    ap.add_argument("--doc_id", type=str, default=None, help="doc_id for the JSON (default: CSV filename stem)")
    ap.add_argument("--case-sensitive", action="store_true", help="Case-sensitive keyword matching.")
    args = ap.parse_args()

    data = build_json_from_csv(
        args.csv,
        doc_id=args.doc_id,
        case_insensitive=not args.case_sensitive,
    )

    out_path = args.output or args.csv.with_suffix(".json")
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
