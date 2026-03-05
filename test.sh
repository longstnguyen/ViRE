#!/usr/bin/env bash
set -euo pipefail

# Smoke/integration test for vi-retrieval-eval (src pipeline)
# - Always runs core offline methods (tfidf, bm25)
# - Runs advanced methods only when dependencies/API keys are available

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-CSConDa}"
CSV="${CSV:-data/${DATASET}.csv}"
OUTROOT="${OUTROOT:-outputs_test_smoke}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
KS="${KS:-1,3,5,10}"
BATCH="${BATCH:-16}"

SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
SPLADE_MODEL="${SPLADE_MODEL:-naver/splade-v3}"
COLBERT_MODEL="${COLBERT_MODEL:-colbert-ir/colbertv2.0}"

RUN_LLM="${RUN_LLM:-0}" # set RUN_LLM=1 to include heavy LLM embedder test

PASS=0
FAIL=0
SKIP=0
FAIL_CASES=()
SKIP_CASES=()

if [[ ! -f "$CSV" ]]; then
  echo "[FATAL] Dataset file not found: $CSV"
  exit 2
fi

if command -v vi-retrieval-eval >/dev/null 2>&1; then
  EVAL_BIN=(vi-retrieval-eval)
else
  EVAL_BIN=(python3 -m vi_retrieval_eval.cli)
fi

has_module() {
  python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)"
}

has_sparse_encoder() {
  python3 - <<'PY'
import sys
try:
    from sentence_transformers import SparseEncoder  # noqa: F401
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

run_case() {
  local name="$1"
  shift

  local out_dir="${OUTROOT}/${DATASET}/${name}"
  mkdir -p "$out_dir"

  echo "\n[RUN ] $name"
  if "$@" \
      --csv "$CSV" \
      --method "${name%%__*}" \
      --max-samples "$MAX_SAMPLES" \
      --sample-seed 42 \
      --prefer-unique \
      --dedup \
      --ks "$KS" \
      --error-k 10 \
      --force \
      --output-dir "$out_dir"; then
    echo "[PASS] $name"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] $name"
    FAIL=$((FAIL + 1))
    FAIL_CASES+=("$name")
  fi
}

run_custom_case() {
  local name="$1"
  shift

  echo "\n[RUN ] $name"
  if "$@"; then
    echo "[PASS] $name"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] $name"
    FAIL=$((FAIL + 1))
    FAIL_CASES+=("$name")
  fi
}

skip_case() {
  local name="$1"
  local reason="$2"
  echo "[SKIP] $name -> $reason"
  SKIP=$((SKIP + 1))
  SKIP_CASES+=("$name ($reason)")
}

echo "==============================================="
echo "Smoke test: vi-retrieval-eval"
echo "Dataset: $CSV"
echo "Output : $OUTROOT"
echo "CLI    : ${EVAL_BIN[*]}"
echo "==============================================="

# 1) Core lexical (always)
run_case "tfidf" "${EVAL_BIN[@]}"
run_case "bm25" "${EVAL_BIN[@]}"

# 2) Dense + hybrid via SBERT (optional)
if has_module sentence_transformers; then
  run_custom_case "dense__sbert" \
    "${EVAL_BIN[@]}" \
    --csv "$CSV" \
    --method dense \
    --dense-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --batch-size "$BATCH" \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed 42 \
    --prefer-unique \
    --dedup \
    --ks "$KS" \
    --error-k 10 \
    --force \
    --output-dir "${OUTROOT}/${DATASET}/dense_sbert"

  run_custom_case "dense+tfidf__sbert_alpha" \
    "${EVAL_BIN[@]}" \
    --csv "$CSV" \
    --method dense+tfidf \
    --fusion alpha \
    --alpha 0.7 \
    --dense-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --batch-size "$BATCH" \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed 42 \
    --prefer-unique \
    --dedup \
    --ks "$KS" \
    --error-k 10 \
    --force \
    --output-dir "${OUTROOT}/${DATASET}/dense_tfidf_alpha"

  run_custom_case "dense+bm25__sbert_rrf" \
    "${EVAL_BIN[@]}" \
    --csv "$CSV" \
    --method dense+bm25 \
    --fusion rrf \
    --rrf-k 60 \
    --dense-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --batch-size "$BATCH" \
    --bm25-k1 1.5 \
    --bm25-b 0.75 \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed 42 \
    --prefer-unique \
    --dedup \
    --ks "$KS" \
    --error-k 10 \
    --force \
    --output-dir "${OUTROOT}/${DATASET}/dense_bm25_rrf"
else
  skip_case "dense__sbert / dense+hybrid" "sentence_transformers not installed"
fi

# 3) SPLADE (optional)
if has_sparse_encoder; then
  run_custom_case "splade" \
    "${EVAL_BIN[@]}" \
    --csv "$CSV" \
    --method splade \
    --splade-model "$SPLADE_MODEL" \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed 42 \
    --prefer-unique \
    --dedup \
    --ks "$KS" \
    --error-k 10 \
    --force \
    --output-dir "${OUTROOT}/${DATASET}/splade"
else
  skip_case "splade" "SparseEncoder (sentence_transformers) unavailable"
fi

# 4) ColBERT (optional)
if has_module colbert; then
  run_custom_case "colbert" \
    "${EVAL_BIN[@]}" \
    --csv "$CSV" \
    --method colbert \
    --colbert-model "$COLBERT_MODEL" \
    --max-samples "${MAX_SAMPLES}" \
    --sample-seed 42 \
    --prefer-unique \
    --dedup \
    --ks "$KS" \
    --error-k 10 \
    --force \
    --output-dir "${OUTROOT}/${DATASET}/colbert"
else
  skip_case "colbert" "colbert-ai not installed"
fi

# 5) API-based dense backends (optional)
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  run_custom_case "dense__openai" \
    "${EVAL_BIN[@]}" \
    --csv "$CSV" \
    --method dense \
    --dense-backend openai \
    --embed-model text-embedding-3-large \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed 42 \
    --prefer-unique \
    --dedup \
    --ks "$KS" \
    --error-k 10 \
    --force \
    --output-dir "${OUTROOT}/${DATASET}/dense_openai"
else
  skip_case "dense__openai" "OPENAI_API_KEY not set"
fi

if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  run_custom_case "dense__gemini" \
    "${EVAL_BIN[@]}" \
    --csv "$CSV" \
    --method dense \
    --dense-backend gemini \
    --gemini-model text-embedding-004 \
    --batch-size "$BATCH" \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed 42 \
    --prefer-unique \
    --dedup \
    --ks "$KS" \
    --error-k 10 \
    --force \
    --output-dir "${OUTROOT}/${DATASET}/dense_gemini"
else
  skip_case "dense__gemini" "GEMINI_API_KEY not set"
fi

# 6) Heavy LLM embedder (optional and opt-in)
if [[ "$RUN_LLM" == "1" ]]; then
  if has_module sentence_transformers; then
    run_custom_case "dense__llm" \
      "${EVAL_BIN[@]}" \
      --csv "$CSV" \
      --method dense \
      --dense-backend llm \
      --sbert-model "$LLM_MODEL" \
      --batch-size 4 \
      --max-len 512 \
      --max-samples "$MAX_SAMPLES" \
      --sample-seed 42 \
      --prefer-unique \
      --dedup \
      --ks "$KS" \
      --error-k 10 \
      --force \
      --output-dir "${OUTROOT}/${DATASET}/dense_llm"
  else
    skip_case "dense__llm" "sentence_transformers not installed"
  fi
else
  skip_case "dense__llm" "disabled by default (set RUN_LLM=1)"
fi

echo "\n==============================================="
echo "Summary: PASS=$PASS | FAIL=$FAIL | SKIP=$SKIP"
if (( ${#FAIL_CASES[@]} > 0 )); then
  echo "Failed cases:"
  for c in "${FAIL_CASES[@]}"; do
    echo "  - $c"
  done
fi
if (( ${#SKIP_CASES[@]} > 0 )); then
  echo "Skipped cases:"
  for c in "${SKIP_CASES[@]}"; do
    echo "  - $c"
  done
fi
echo "==============================================="

if (( FAIL > 0 )); then
  exit 1
fi

exit 0
