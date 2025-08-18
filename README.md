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

For OpenAI integration, set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

