# syntax=docker/dockerfile:1

FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf \
    HF_CACHE=/app/.cache/hf \
    HF_DATASETS_CACHE=/app/.cache/hf_datasets \
    SPACY_HOME=/app/.cache/spacy \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# System dependencies (faiss + torch often rely on libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Combine pip operations and model downloads to reduce layers
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    python -m nltk.downloader punkt_tab && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_trf && \
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

COPY . /app

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# More appropriate default command - can be overridden when running
CMD ["python", "-c", "print('Container ready. Run with: docker run <image> python your_script.py')"]