#!/bin/bash
# Train NER model

set -e  # Exit on error

# Default values
TRAIN_JSONL="${TRAIN_JSONL:data/ner_data/splits/train.jsonl}"
VALID_JSONL="${VALID_JSONL:data/ner_data/splits/dev.jsonl}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-models/ner_roberta}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
O_WEIGHT="${O_WEIGHT:-0.2}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

python3 src/ner/train_ner.py \
  --train_jsonl "$TRAIN_JSONL" \
  --valid_jsonl "$VALID_JSONL" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --grad_accum "$GRAD_ACCUM" \
  --fp16 \
  --class_weight_power "$CLASS_WEIGHT_POWER" \
  --o_weight "$O_WEIGHT"

echo "Model training completed → $OUTPUT_DIR"