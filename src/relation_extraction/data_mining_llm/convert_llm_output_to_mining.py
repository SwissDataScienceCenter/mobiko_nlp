#!/usr/bin/env python
import json
import sys
from pathlib import Path
from typing import Dict

def convert_file(input_path: str, output_path: str) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            paper_id = doc.get("doc_id", "")

            sentences = doc.get("sentences", [])
            for sent_id, sent_obj in enumerate(sentences):
                text = sent_obj.get("text", "")
                llm_block = sent_obj.get("llm", {})
                accepted = llm_block.get("final_spans", []) or []

                entities_out = []
                for idx, ent in enumerate(accepted, start=1):
                    try:
                        ent_text = ent["text"]
                        ent_type = ent["type"]
                        start = ent["start_char"]
                        end = ent["end_char"]
                        tier = ent["tier"]
                        uncertain = ent["uncertain"]
                        concept_text = ent["concept_text"]

                        entities_out.append(
                            {
                                "id": f"T{idx}",
                                "text": ent_text,
                                "type": ent_type,
                                "start": start,
                                "end": end,
                                "tier": tier,
                                "uncertain": uncertain,
                                "concept_text": concept_text,
                            }
                        )
                    except Exception as e:
                        print(ent)
                        print(
                            f"[WARNING] Skipping entity in paper {paper_id}, "
                            f"sentence {sent_id} due to error: {e}",
                            file=sys.stderr,
                        )
                        continue

                if not len(entities_out) or len(entities_out) == 1:
                    continue
                out_record = {
                    "paper_id": paper_id,
                    "sent_id": sent_id,
                    "text": text,
                    "entities": entities_out,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python convert_entity_pipeline_to_mining.py "
            "input_entity_pipeline.jsonl output_mining_ready.jsonl",
            file=sys.stderr,
        )
        sys.exit(1)

    convert_file(sys.argv[1], sys.argv[2])
