import os
import sys
import json
import time
import argparse
from pathlib import Path

import spacy

# pip install openai>=1.0.0
from openai import OpenAI

# Get API key from environment variable for security
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

if not OPENAI_API_KEY:
    print("ERROR: Please set OPENAI_API_KEY environment variable", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)


SHEMA_BIODIV = """

Biodiversity 	HAS PROPERTY	Species						
                HAS PROPERTY	Ecosystem						
                HAS PROPERTY	Gene						

Species	HAS PROPERTY	Characteristics	HAS PROPERTY	Habitat				
                                        HAS PROPERTY	Feeding regime				
                                        HAS PROPERTY	Life history				

	    HAS PROPERTY	Diversity			HAS PROPERTY	status		
					                        HAS PROPERTY	trend		
	    HAS PROPERTY	Population	HAS PROPERTY	Density	HAS PROPERTY	status		
					                                        HAS PROPERTY	trend		
			                        HAS PROPERTY	Distribution	HAS PROPERTY	status		
					                                                HAS PROPERTY	trend		
			                        HAS PROPERTY	Size	HAS PROPERTY	status		
					                                        HAS PROPERTY	trend		
	    HAS PROPERTY	Distribution	HAS PROPERTY	Space	HAS PROPERTY	status		
					                                            HAS PROPERTY	trend		
			                            HAS PROPERTY	Elevation	HAS PROPERTY	status		
					                                                HAS PROPERTY	trend		
				                        HAS PROPERTY    Time	HAS PROPERTY	trend		
	    HAS PROPERTY	Conservation Status			            HAS PROPERTY	status		
					                                            HAS PROPERTY	trend		

	    IS AFFECTED BY	Driver						

	    IS DETERMINING 	Functions						
		                Ecosystem services						


Drivers	HAS TYPE	Climate			                        HAS PROPERTY	trend		
			                        HAS TYPE	Temperature	HAS PROPERTY	status		
					                                        HAS EFFECT		HAS PROPERTY	trend
			HAS TYPE	Precipitation	HAS PROPERTY	status		
					                    HAS PROPERTY	trend		
					                    HAS EFFECT		HAS PROPERTY	trend
			HAS TYPE	Wind	        HAS PROPERTY	status		
					                    HAS PROPERTY	trend		
					                    HAS EFFECT		HAS PROPERTY	trend
			HAS TYPE	Drought	        HAS PROPERTY	status		
					                    HAS PROPERTY	trend		
					                    HAS EFFECT		HAS PROPERTY	trend
			HAS TYPE	Extreme events	HAS PROPERTY	status		
					                    HAS PROPERTY	trend		
					                    HAS EFFECT		HAS PROPERTY	trend
"""


SYSTEM_PROMPT = f"""You are a careful biodiversity information extractor.

Given ONE sentence, decide which words are biodiversity entities (and assign a TYPE from the provided schema),

Provided schema: {SCHEMA_BIODIV}

Return STRICT JSON only, matching this schema:
{{
  "accepted": [{{"text":"...", "type":"CLIMATE Temperature trend", "start_char":int, "end_char":int}}],
  "notes": "optional short string"
}}
Do not include explanations outside JSON. If unsure about spans, estimate conservatively.
"""


def read_txt_files(indir: str):
    """Generator to read all .txt files from input directory."""
    try:
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
                print(f"Warning: Could not read {path}: {e}", file=sys.stderr)
                continue
    except Exception as e:
        print(f"Error reading directory {indir}: {e}", file=sys.stderr)



def call_llm(sentence: str, retries: int = 3, sleep_s: float = 3.0) -> dict:
    user_payload = {
        "sentence": sentence,
    }

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ]
            )
            print(f'RESPONSE: {resp}')
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == retries - 1:
                return {"accepted": [], "notes": f"llm_error: {repr(e)}"}
            time.sleep(sleep_s * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with .txt documents")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL (one object per document)")
    ap.add_argument("--model", default="en_core_web_trf", help="spaCy model (needs parser for noun_chunks)")
    ap.add_argument("--max_sents_per_doc", type=int, default=999999, help="Cap sentences per doc (debug)")
    ap.add_argument("--sample_every", type=int, default=1, help="Process every Nth sentence (e.g., 5 to sample)")
    args = ap.parse_args()


    # Load spaCy with parser (noun_chunks needs it)
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"spaCy model '{args.model}' not found. Install with: python -m spacy download {args.model}",
              file=sys.stderr)
        sys.exit(1)

    # Quick check parser present
    if "parser" not in nlp.pipe_names:
        print("WARNING: spaCy parser not enabled; noun_chunks may be empty. Use a model with parser.", file=sys.stderr)

    Path(os.path.dirname(args.out_jsonl) or ".").mkdir(parents=True, exist_ok=True)

    docs_written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for doc_id, text in read_txt_files(args.in_dir):
            if docs_written > 1:
                break
            doc = nlp(text)
            out_sents = []
            for i, sent in enumerate(doc.sents):
                if i >= args.max_sents_per_doc:
                    break
                if (i % args.sample_every) != 0:
                    continue

                verdict = call_llm(sent.text)
                print(verdict)

                # clamp spans to sentence bounds (defensive)
                def clamp_span(x):
                    s = max(0, int(x.get("start_char", 0)))
                    e = min(len(sent.text), int(x.get("end_char", s)))
                    y = dict(x);
                    y["start_char"] = s;
                    y["end_char"] = e
                    return y

                verdict["accepted"] = [clamp_span(x) for x in verdict.get("accepted", []) if x.get("text")]

                out_sents.append({
                    "text": sent.text,
                    "llm": verdict
                })

            # Write one JSON object per document
            rec = {"id": doc_id, "sentences": out_sents}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_written += 1

    print(f"Done. Wrote {docs_written} documents to {args.out_jsonl}")


if __name__ == "__main__":
    main()
