#!/usr/bin/env bash
# =============================================================================
# eval_full.sh  —  Full reproduction script (EACL 2026)
#
# Reproduces all results from:
#   "Which Works Best for Vietnamese? A Practical Study of Information
#   Retrieval Methods across Domains"
#
# Part 1: Lexical / sparse baselines  (TF-IDF, BM25, SPLADE, ColBERT)
# Part 2: Dense models × 5 methods   (dense, +tfidf-α, +tfidf-rrf,
#                                      +bm25-α, +bm25-rrf)
#
# Usage:
#   bash scripts/eval_full.sh
#
# After completion, aggregate results with:
#   python scripts/summarize_report.py --outputs-root outputs \
#     --datasets CSConDa EduCoQA ALQAC \
#     --metrics P@1,R@10,MRR@10,nDCG@10 --percent --save VIRE_Report.md
#
# Requirements:
#   pip install -e ".[all]"
#   pip install "colbert-ai==0.2.22" --no-deps   # ColBERT support
#   export OPENAI_API_KEY=...                     # OpenAI backend only
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASETS=(
  "ALQAC"
  "CSConDa"
  "EduCoQA"
  "UIT-ViQuAD2"
  "ViMedAQA_v2"
  "ViNewsQA"
  "ViRe4MRC_v2"
  "ViRHE4QA_v2"
  "VlogQA_2"
  "ZaloLegalQA"
)

# Format: "backend:model_id"
#   sbert  — encoder-only models (sentence-transformers compatible)
#   llm    — decoder-only / LLM-based models (flash-attn, left-padding)
#   openai — OpenAI Embeddings API (no --batch-size, uses --embed-model)
DENSE_MODELS=(
  "sbert:BAAI/bge-m3"
  "sbert:intfloat/multilingual-e5-large"
  "sbert:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  "sbert:AITeamVN/Vietnamese_Embedding_v2"
  "sbert:bkai-foundation-models/vietnamese-bi-encoder"
  "sbert:dangvantuan/vietnamese-document-embedding"
  "sbert:Snowflake/snowflake-arctic-embed-l-v2.0"
  "sbert:Alibaba-NLP/gte-multilingual-base"
  "llm:jinaai/jina-embeddings-v3"
  "llm:google/embeddinggemma-300m"
  "llm:Qwen/Qwen3-Embedding-0.6B"
  "llm:Alibaba-NLP/gte-Qwen2-1.5B-instruct"
  "llm:BAAI/bge-multilingual-gemma2"
  "openai:text-embedding-3-large"
)

SPLADE_MODEL="naver/splade-v3"
COLBERT_MODEL="colbert-ir/colbertv2.0"

BATCH=256    # embedding batch size (not used for openai backend)
MAX_SAMPLES=1000  # cap queries per dataset (0 = no cap)
K_REF=20     # reference k for fail@K error analysis
OUTDIR_ROOT="outputs"

# Per-dataset token-level truncation for OOD long-document datasets.
# ZaloLegalQA articles are very long; truncating to 256 tokens avoids OOM
# and keeps contexts within most models' effective range.
ZALO_MAX_LEN=256

BM25_K1=1.5  # Okapi BM25 k1
BM25_B=0.75  # Okapi BM25 b
ALPHA=0.7    # score-fusion alpha weight
RRF_K=60     # RRF constant k

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Populates globals $CSV, $QRELS, and $MAXLEN_ARGS for the given dataset.
# ZaloLegalQA ships with external qrels instead of inline gold contexts.
# ALQAC has very long documents and requires token-level truncation.
setup_dataset() {
  local NAME="$1"
  if [[ "$NAME" == "ZaloLegalQA" ]]; then
    CSV="data/ZaloLegalQA/dataset.csv"
    QRELS=(
      --qrels        "data/ZaloLegalQA/all.jsonl"
      --csv-qid-col  qid
      --csv-docid-col doc_id
      --qid-col      query-id
      --docid-col    corpus-id
      --rel-col      score
    )
  else
    CSV="data/${NAME}.csv"
    QRELS=()
  fi

  # Apply per-dataset max-len truncation where needed
  if [[ "$NAME" == "ZaloLegalQA" ]]; then
    MAXLEN_ARGS=(--max-len "$ZALO_MAX_LEN")
  else
    MAXLEN_ARGS=()
  fi
}

# Thin wrapper: injects dataset ($CSV, $QRELS, $MAXLEN_ARGS) and all shared
# flags so each call site only needs to specify method-specific arguments.
# Globals required: $CSV, $QRELS, $MAXLEN_ARGS, $OUTDIR
vire() {
  vi-retrieval-eval \
    --csv        "$CSV" \
    "${QRELS[@]}" \
    "${MAXLEN_ARGS[@]}" \
    --max-samples "$MAX_SAMPLES" \
    --prefer-unique \
    --dedup \
    --show-size \
    --progress \
    --error-k    "$K_REF" \
    --force \
    --output-dir "$OUTDIR" \
    "$@"
}

# =============================================================================
# Part 1 — Lexical / sparse (one pass per dataset, no dense model)
# =============================================================================

echo "================================================================"
echo " Part 1 — Lexical / sparse standalone"
echo "================================================================"

for NAME in "${DATASETS[@]}"; do
  setup_dataset "$NAME"
  OUTDIR="${OUTDIR_ROOT}/${NAME}"
  mkdir -p "$OUTDIR"

  echo
  echo "  Dataset: ${NAME}"
  echo "  --------------------------------------------------------------"

  vire --method tfidf

  vire --method bm25 \
       --bm25-k1 "$BM25_K1" \
       --bm25-b  "$BM25_B"

  vire --method splade \
       --splade-model "$SPLADE_MODEL"

  vire --method colbert \
       --dense-backend colbert \
       --colbert-model "$COLBERT_MODEL"

done

# =============================================================================
# Part 2 — Dense models (dense + 4 hybrid variants × 10 datasets)
# =============================================================================

echo
echo "================================================================"
echo " Part 2 — Dense models"
echo "================================================================"

for MODEL_ENTRY in "${DENSE_MODELS[@]}"; do
  BACKEND="${MODEL_ENTRY%%:*}"
  MODEL_ID="${MODEL_ENTRY#*:}"
  MODEL_SAFE=$(echo "$MODEL_ID" | tr '/.' '_')

  echo
  echo "================================================================"
  echo " Model:   ${MODEL_ID}"
  echo " Backend: ${BACKEND}"
  echo "================================================================"

  # openai → --embed-model, no --batch-size
  # sbert / llm → --sbert-model + --batch-size
  if [[ "$BACKEND" == "openai" ]]; then
    MODEL_FLAGS=(--dense-backend openai --embed-model "$MODEL_ID")
  else
    MODEL_FLAGS=(--dense-backend "$BACKEND" --sbert-model "$MODEL_ID" --batch-size "$BATCH")
  fi

  for NAME in "${DATASETS[@]}"; do
    setup_dataset "$NAME"
    OUTDIR="${OUTDIR_ROOT}/${MODEL_SAFE}/${NAME}"
    mkdir -p "$OUTDIR"

    echo
    echo "  Dataset: ${NAME}"
    echo "  --------------------------------------------------------------"

    vire "${MODEL_FLAGS[@]}" \
         --method dense

    vire "${MODEL_FLAGS[@]}" \
         --method dense+tfidf --fusion alpha --alpha "$ALPHA" \
         --save-intersection

    vire "${MODEL_FLAGS[@]}" \
         --method dense+tfidf --fusion rrf --rrf-k "$RRF_K" \
         --save-intersection

    vire "${MODEL_FLAGS[@]}" \
         --method dense+bm25 --fusion alpha --alpha "$ALPHA" \
         --bm25-k1 "$BM25_K1" --bm25-b "$BM25_B" \
         --save-intersection

    vire "${MODEL_FLAGS[@]}" \
         --method dense+bm25 --fusion rrf --rrf-k "$RRF_K" \
         --bm25-k1 "$BM25_K1" --bm25-b "$BM25_B" \
         --save-intersection

  done

  echo
  echo "  Finished: ${MODEL_ID}"

done
