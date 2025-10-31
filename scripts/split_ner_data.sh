#!/bin/bash
# Split dataset into train/dev/test

set -e  # Exit on error

# Default values
INPUT_JSONL="${INPUT_JSONL:data/ner_data/bioc_conll.jsonl}"
OUT_DIR="${OUT_DIR:data/ner_data/splits}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
DEV_RATIO="${DEV_RATIO:-0.1}"
SEED="${SEED:-13}"

echo "Splitting dataset: $INPUT_JSONL"

python3 src/ner/ner_eval_tools.py split \
  --input_jsonl "$INPUT_JSONL" \
  --out_dir "$OUT_DIR" \
  --train_ratio "$TRAIN_RATIO" \
  --dev_ratio "$DEV_RATIO" \
  --seed "$SEED"

echo "Dataset split completed: train/dev/test → $OUT_DIR"