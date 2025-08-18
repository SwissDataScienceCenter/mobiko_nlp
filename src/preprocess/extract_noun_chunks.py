import argparse, glob, json, sys
import spacy
from spacy.tokens import Doc


import os
import argparse


def iter_docs(indir):
    for filename in os.listdir(indir):
        path = os.path.join(indir, filename)
        if os.path.isfile(path) and filename.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                yield filename.replace('.txt', ''), f.read()


def main():
    """
    Run spacy/en_core_web_sm on CPU, and spacy/en_core_web_trf on GPU
    For biomedical texts use https://allenai.github.io/scispacy/:
    1. en_core_sci_md or
    2. en_core_sci_scibert
    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out', required=True, help='Save to jsonl file.')
    ap.add_argument('--model', default='en_core_web_trf')
    args = ap.parse_args()

    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Please install spaCy model {args.model}: python -m spacy download {args.model}", file=sys.stderr)
        sys.exit(1)

    #nlp.add_pipe("sentencizer", first=True) #
    total_docs = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for doc_id, text in iter_docs(args.in_dir):

            doc = nlp(text)
            sentences = []

            for sent in doc.sents:
                # collect noun-chunks in the sentence
                nps = []

                for np in sent.noun_chunks:
                    # skip pronouns/determiners
                    if np.root.pos_ not in ("NOUN", "PROPN"):
                        continue
                    if np.text.strip():
                        nps.append({
                            "start_char": np.start_char - sent.start_char,
                            "end_char": np.end_char - sent.start_char,
                            "text": np.text
                        })

                sentences.append({
                    "text": sent.text,
                    "nps": nps
                })

            rec = {
                "doc_id": doc_id,
                "sentences": sentences
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_docs += 1
    print(f'Processed {total_docs} documents!')


if __name__ == '__main__':
    main()
