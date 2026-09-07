# Mobiko NLP

A biodiversity information extraction pipeline using NLP techniques.

## Overview

This project provides tools for extracting and classifying biodiversity-related entities from text documents using:
- BERT-based Named Entity Recognition (NER). Work in progress!
- LLM-based extraction for biodiversity entity classification with structured schemas (Demo version).
- spaCy for text processing and noun phrase extraction


## Installation

### Docker Compose Usage

```bash
# Start the development container
docker-compose up -d

# Run commands inside the container
docker-compose exec biodiv python src/ner/bert_ner_baseline.py --in_dir data --out_jsonl output/ner_results.jsonl
docker-compose exec biodiv python src/demo/demo.py --in_dir data --out_jsonl output/demo_results.jsonl

# Run one-off tasks without starting the persistent container
docker-compose run --rm biodiv python src/ner/bert_ner_baseline.py --help

# Stop the container
docker-compose down
```

### Local Installation

If installing locally, refer to `Dockerfile` for exact dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download spaCy model:
```bash
python -m spacy download en_core_web_trf
```

### Entity linking (`src/linking`)

Entity de-duplication and ontology canonicalisation run in three tiers. Only
Tier A is required, and it needs no GPU and no API key, so a collaborator can
run the pipeline on a laptop:

| Tier | Extra dependencies | Adds |
|---|---|---|
| **A** — required | `sqlite3` (stdlib), `rapidfuzz`, `nltk` + WordNet data, `spacy` + `en_core_web_sm` | normalisation, type routing, exact/alias/fuzzy matching, WordNet must-not-link, clustering of unmatched mentions |
| **B** — optional | `sentence-transformers`, `torch`, `en_core_web_trf` | span unrolling (needs `trf` — see below), dense candidate generation over the unmatched set |
| **C** — optional | an LLM API key or a local GPU | adjudication of the hard residue, every proposed ID verified by lookup |

Tier A one-time setup:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
```

Tier B additionally needs `python -m spacy download en_core_web_trf`.

The authority dictionary is a single versioned SQLite file built once on a
machine with network access (`src/linking/authorities/`). Collaborators
download that file; nothing in the pipeline calls a live API, so results do not
move when GBIF or ENVO publish a new release.

For OpenAI integration, set your API key in the environment on the machine that runs the code:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For the remote interpreter workflow, keep secrets out of tracked files:
```bash
cp .env.example .env
# fill in OPENAI_API_KEY and/or OPEN_WEB_UI_API_KEY on the remote machine
```

The code now auto-loads `.env` from the repo root when present. If you prefer to keep the secrets file elsewhere on the remote host, set:
```bash
export MOBIKO_ENV_FILE="/absolute/path/to/remote-secrets.env"
```
