import os
import json

indir = "./data/gold/bioc_annotated_json"
outdir = "./data/gold/bioc_extracted_txts"
os.makedirs(outdir, exist_ok=True)

for filename in os.listdir(indir):
    if not filename.endswith(".json"):
        continue

    doc_id = filename.replace('.json', '')
    with open(os.path.join(indir, filename), encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for article in data.get("sibils_article_set", []):
        for passage in article.get("passages", []):
            text = passage.get("text", "").strip()
            if text:
                lines.append(text)

    with open(os.path.join(outdir, f"{doc_id}.txt"), "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(lines))
