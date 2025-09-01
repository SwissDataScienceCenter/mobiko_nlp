#!/bin/bash
# Evaluate NER model

set -e  # Exit on error

# Default values
MODEL_DIR="models/ner_roberta"
TEST_JSONL="data/ner_data/splits/test.jsonl"
BATCH_SIZE="16"
REPORT_PATH="results/test_report.txt"

# Create results directory
mkdir -p "$(dirname "$REPORT_PATH")"

python3 src/ner/ner_eval_tools.py test \
  --model_dir "$MODEL_DIR" \
  --test_jsonl "$TEST_JSONL" \
  --batch_size "$BATCH_SIZE" \
  --report_path "$REPORT_PATH"

echo "Model evaluation completed → $REPORT_PATH"