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
