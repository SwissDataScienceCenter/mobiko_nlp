# biodivner_prepare.py
import csv, json, sys
from pathlib import Path

PUNCT_NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "’", "”", "»"}
PUNCT_NO_SPACE_AFTER  = {"(", "[", "{", "‘", "“", "«"}

def detok(tokens):
    out = []
    for i, tok in enumerate(tokens):
        if not out:
            out.append(tok)
            continue
        prev = out[-1]
        if tok in PUNCT_NO_SPACE_BEFORE:
            out[-1] = prev + tok
        elif prev and prev[-1] == "-":
            out[-1] = prev + tok
        elif tok in PUNCT_NO_SPACE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out)

def csv_to_sentences_and_gold(csv_path, out_sentences, out_gold_jsonl, out_index_jsonl):
    doc_id = "biodivner_test"
    sentences = []
    gold_jsonl = []
    index_map = []

    current_tokens = []   # [(word, tag)]

    def flush():
        if not current_tokens:
            return
        tokens = [w for w, _ in current_tokens]
        tags   = [t for _, t in current_tokens]
        text = detok(tokens)

        # token char offsets in detok text
        spans = []
        offsets = []
        pos = 0
        for w in tokens:
            start = text.find(w, pos)
            if start < 0:
                start = pos
            end = start + len(w)
            offsets.append((start, end))
            pos = end

        cur = None
        for (start, end), tag in zip(offsets, tags):
            if tag.startswith("B-"):
                if cur:
                    spans.append(cur)
                span = "".join(text[start:end])
                cur = {"start_char": start, "end_char": end, "label": tag[2:], "text": span}
            elif tag.startswith("I-") and cur and cur["label"] == tag[2:]:
                cur["end_char"] = end
            else:
                if cur:
                    spans.append(cur); cur = None
        if cur:
            spans.append(cur)

        sentences.append(text)
        gold_jsonl.append({
            "doc_id": doc_id,
            "sentences": [{"text": text, "spans": spans}]
        })
        current_tokens.clear()

    def is_header(row):
        if len(row) >= 3:
            return (row[0].strip().lower().replace(" ", "") in {"sentence#", "sentence#,"} and
                    row[1].strip().lower() == "word" and
                    row[2].strip().lower() == "tag")
        return False

    with open(csv_path, newline='', encoding="utf-8") as f:
        r = csv.reader(f)
        sent_id = -1
        for row in r:
            if not row or all((c is None or c.strip() == "") for c in row):
                continue
            if is_header(row):
                continue

            first = (row[0] or "").strip()

            # New sentence marker that also contains the first token and tag
            if first.startswith("Sentence:"):
                # close previous sentence
                flush()
                sent_id += 1
                # consume the token on the same row if present
                word = row[1].strip() if len(row) >= 2 else ""
                tag  = row[2].strip() if len(row) >= 3 else ""
                if word:
                    current_tokens.append((word, tag))
                continue

            # Regular token rows:
            # format 1: ["", word, tag]
            if len(row) >= 3 and row[0] == "":
                word, tag = row[1].strip(), row[2].strip()
                if word:
                    current_tokens.append((word, tag))
                continue
            # format 2: [word, tag]
            if len(row) >= 2:
                word, tag = row[0].strip(), row[1].strip()
                if word:
                    current_tokens.append((word, tag))
                continue

        # flush last sentence
        flush()

    Path(out_sentences).write_text("\n".join(sentences) + "\n", encoding="utf-8")
    with open(out_gold_jsonl, "w", encoding="utf-8") as f:
        for obj in gold_jsonl:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(out_index_jsonl, "w", encoding="utf-8") as f:
        for i in range(len(sentences)):
            f.write(json.dumps({"line_idx": i, "doc_id": doc_id, "sent_id": i}) + "\n")

    print(f"Wrote {len(sentences)} sentences to {out_sentences}")
    print(f"Wrote gold JSONL to {out_gold_jsonl}")
    print(f"Wrote index map to {out_index_jsonl}")

if __name__ == "__main__":
    # Usage:
    # python biodivner_prepare.py BiodivNER_test.csv sentences.txt biodivner_test_gold.jsonl index_map.jsonl
    csv_path = sys.argv[1]
    out_sentences = sys.argv[2]
    out_gold = sys.argv[3]
    out_index = sys.argv[4]
    csv_to_sentences_and_gold(csv_path, out_sentences, out_gold, out_index)
