# VIRE: Vietnamese Information Retrieval Evaluation Toolkit

A lightweight, extensible toolkit for benchmarking lexical, dense, sparse, late-interaction, and hybrid retrieval on Vietnamese datasets.

> This repository accompanies the paper:  
> **Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains**  
> Long S. T. Nguyen, Tho T. Quan. Accepted at _EACL 2026_.

---

## Why VIRE?

Benchmarking retrieval fairly is harder than it looks — different tools use different preprocessing, sampling, and evaluation conventions, making comparisons unreliable. VIRE removes that friction by providing a single, reproducible CLI that covers the full spectrum from classical lexical methods to dense neural and hybrid approaches, all under the same data pipeline.

- Unified CLI to benchmark BM25, TF-IDF, Dense, SPLADE, ColBERT, and hybrid retrieval with fair, reproducible settings
- Dataset-agnostic schema — plug in any CSV/qrels following the standard specification
- Full pipeline: normalization $\rightarrow$ indexing $\rightarrow$ evaluation $\rightarrow$ error analysis $\rightarrow$ reporting

---

## Core Features

VIRE covers the full retrieval benchmarking lifecycle — from corpus deduplication and normalization through indexing, scoring, and multi-metric evaluation — with support for a wide range of retrieval paradigms and embedding backends.

**Diverse Retrieval Methods**

- Lexical: TF-IDF, BM25
- Dense (single-vector): OpenAI API, Gemini API, local SBERT (`--dense-backend sbert`), local LLM embedders (`--dense-backend llm`)
- Sparse neural: SPLADE (`--method splade`)
- Late interaction: ColBERT / ColBERTv2 (`--method colbert`)
- Hybrid: dense/SPLADE + TF-IDF or BM25 with $\alpha$-fusion or Reciprocal Rank Fusion (RRF)

**Vietnamese-aware Preprocessing**

- NFKC normalization, invisible-character removal, emoji stripping, lowercase, space collapsing (`--normalize-all`)
- Random and unique-context sampling with corpus deduplication

**Evaluation and Reporting**

- Precision@_k_, Recall@_k_, HitRate@_k_, MRR@_k_, MAP@_k_, nDCG@_k_, _R_-Precision, First Relevant Rank (mean, median, found rate)
- Outputs: `metrics.json`, `ranks.json`, `errors/fail@K.csv`, cached embeddings & FAISS index
- Markdown aggregation via `scripts/summarize_report.py`

**Error Analysis**

- Exports `errors/fail@K.csv` per run
- Cross-method failure intersection with `--save-intersection`
- Per-query diagnostics: query, gold docs, retrieved docs, scores

**Embedding Backends**

- OpenAI and Gemini APIs (cloud, no GPU needed)
- Local SBERT (`sbert`): any `sentence-transformers`-compatible model
- Local LLM embedder (`llm`): large instruction-following models (Qwen3-Embedding, BGE-Gemma2, Jina v3, etc.) with flash-attention support and OOM-safe batching
- SPLADE sparse encoder (via `sentence-transformers` SparseEncoder)
- ColBERT / ColBERTv2 (via `colbert-ai`)

**Reusability and Extensibility**

- Embedding and FAISS caching; `--force` to rebuild
- Multi-gold qrels support
- Plugin architecture: add a new backend with one `@register` decorator

---

## Command-Line Interface (CLI)

All functionality is exposed through a single command, `vi-retrieval-eval`. The complete reference for every flag is shown below; you can also run `vi-retrieval-eval --help` at any time to see the same output.

```
Evaluate lexical, dense, sparse, late-interaction, and hybrid retrieval for Vietnamese QA.

options:
  -h, --help            show this help message and exit
  --csv CSV             Input dataset file (CSV/JSONL) with columns or keys:
                        question, context (default: None)
  --qrels QRELS         Optional qrels file (CSV/TSV/JSONL) with columns:
                        qid/doc_id/rel (default: None)
  --qid-col QID_COL     Qrels column for query id (default: qid)
  --docid-col DOCID_COL
                        Qrels column for doc id (default: doc_id)
  --rel-col REL_COL     Qrels column for relevance (>0) (default: rel)
  --csv-qid-col CSV_QID_COL
                        Column name in main CSV/JSONL for qid (optional)
                        (default: None)
  --csv-docid-col CSV_DOCID_COL
                        Column name in main CSV/JSONL for doc_id (optional)
                        (default: None)
  --output-dir OUTPUT_DIR
                        Root folder to save results (default: outputs)
  --method {tfidf,bm25,dense,dense+tfidf,dense+bm25,splade,splade+tfidf,splade+bm25,splade+dense,colbert}
                        Retrieval method (default: None)
  --fusion {none,alpha,rrf}
                        Fusion for hybrid methods (dense+tfidf / dense+bm25 /
                        splade+*) (default: none)
  --alpha ALPHA         Alpha for score fusion (0..1) (default: 0.5)
  --rrf-k RRF_K         RRF constant k (>=1) (default: 60)
  --max-samples MAX_SAMPLES
                        Random subset size, e.g. 1000 (default: None)
  --sample-frac SAMPLE_FRAC
                        Random subset fraction, e.g. 0.1 (10%) (default: None)
  --sample-seed SAMPLE_SEED
                        Random seed for sampling (default: 42)
  --prefer-unique       Prefer samples with unique contexts when sampling
                        (normalized) (default: False)
  --unique-col UNIQUE_COL
                        Column name used as uniqueness key for --prefer-unique
                        (default: context)
  --dedup-lower         Lowercase in normalization key for unique/dedup
                        (default: False)
  --dedup-remove-emoji  Strip emoji-like symbols in normalization key for
                        unique/dedup (default: False)
  --max-len MAX_LEN     Optional max token length for embedding (tokenizer-
                        based). If set, texts are truncated right before
                        embedding; if None, keep original. (default: None)
  --normalize-all       Normalize ALL questions/contexts with NFKC, remove
                        invisibles/controls, strip emoji, collapse spaces,
                        lowercase BEFORE snapshot/dedup/eval (default: False)
  --bm25-k1 BM25_K1     BM25 Okapi k1 (default: 1.5)
  --bm25-b BM25_B       BM25 Okapi b (default: 0.75)
  --dense-backend {openai,gemini,sbert,llm,colbert}
                        Dense embedding backend (for dense / splade+dense)
                        (default: openai)
  --embed-model EMBED_MODEL
                        OpenAI embedding model (default: text-embedding-3-large)
  --gemini-model GEMINI_MODEL
                        Gemini embedding model (default: text-embedding-004)
  --sbert-model SBERT_MODEL
                        Sentence-Transformers / HF local model (for sbert/llm
                        backends) (default: sentence-transformers/all-MiniLM-L6-v2)
  --batch-size BATCH_SIZE
                        Embedding batch size (default: 128)
  --index-metric {ip,l2}
                        FAISS metric (recommend 'ip' with normalized vectors)
                        (default: ip)
  --splade-model SPLADE_MODEL
                        HuggingFace checkpoint for SPLADE (sparse encoder)
                        (default: naver/splade-v3)
  --colbert-model COLBERT_MODEL
                        Checkpoint ColBERT / ColBERTv2 (default: colbert-ir/colbertv2.0)
  --force               Force rebuild embeddings and index (default: False)
  --lower               Lowercase text before processing (kept for backward-
                        compat; ignored if --normalize-all) (default: False)
  --ks KS               Comma-separated k values (default: 1,3,5,10,20,50,100)
  --show-size           Print dataset head + sizes (default: False)
  --progress            Show progress bars (default: False)
  --log-level {debug,info,warning,error}
                        Logger level (default: info)
  --log-file LOG_FILE   Optional log file path (default: None)
  --list-backends       List available dense embedding backends and exit
                        (default: False)
  --dedup               Deduplicate identical contexts in the corpus before
                        indexing (queries remain unchanged) (default: False)
  --error-k ERROR_K     Reference K used to mark a query as FAIL
                        (default: max(ks)). (default: None)
  --save-intersection   After run, save the intersection of fail@K across ALL
                        methods for this dataset. (default: False)
  --max-errors MAX_ERRORS
                        When printing previews, max rows to show. (default: 30)
```

---

## Installation

VIRE requires Python 3.9+ and can be installed directly from the repository. The `[all]` extra pulls in all optional backends; you can also install only what you need.

```bash
git clone https://github.com/longstnguyen/ViRE.git
cd vi-retrieval-eval
pip install -e ".[all]"
```

For ColBERT support (optional, requires separate install):

```bash
pip install "colbert-ai==0.2.22" --no-deps
```

Set environment variables for API-based backends:

```bash
export OPENAI_API_KEY=...     # for --dense-backend openai
export GEMINI_API_KEY=...     # for --dense-backend gemini
```

---

## Quickstart

### Single Runs

The examples below cover all supported methods. The flags `--prefer-unique --dedup` are generally recommended to ensure each context appears only once in the corpus, which gives cleaner evaluation numbers.

```bash
# BM25 baseline
vi-retrieval-eval --csv data/CSConDa.csv --method bm25 \
  --max-samples 1000 --prefer-unique --dedup --output-dir outputs

# Dense – local SBERT
vi-retrieval-eval --csv data/CSConDa.csv --method dense \
  --dense-backend sbert --sbert-model AITeamVN/Vietnamese_Embedding_v2 \
  --max-samples 1000 --prefer-unique --dedup \
  --batch-size 64 --progress --output-dir outputs

# Dense – large LLM embedder (e.g. Qwen3-Embedding)
vi-retrieval-eval --csv data/CSConDa.csv --method dense \
  --dense-backend llm --sbert-model Qwen/Qwen3-Embedding-0.6B \
  --max-samples 1000 --prefer-unique --dedup \
  --batch-size 16 --max-len 512 --progress --output-dir outputs

# Dense – OpenAI
vi-retrieval-eval --csv data/CSConDa.csv --method dense \
  --dense-backend openai --embed-model text-embedding-3-large \
  --max-samples 1000 --prefer-unique --dedup --output-dir outputs

# Hybrid: Dense + BM25 (Alpha fusion)
vi-retrieval-eval --csv data/CSConDa.csv \
  --method dense+bm25 --fusion alpha --alpha 0.7 \
  --dense-backend sbert --sbert-model AITeamVN/Vietnamese_Embedding_v2 \
  --max-samples 1000 --prefer-unique --dedup --output-dir outputs

# Hybrid: Dense + BM25 (RRF)
vi-retrieval-eval --csv data/CSConDa.csv \
  --method dense+bm25 --fusion rrf --rrf-k 60 \
  --dense-backend sbert --sbert-model AITeamVN/Vietnamese_Embedding_v2 \
  --max-samples 1000 --prefer-unique --dedup --output-dir outputs

# SPLADE (sparse neural)
vi-retrieval-eval --csv data/CSConDa.csv --method splade \
  --splade-model naver/splade-v3 \
  --max-samples 1000 --prefer-unique --dedup --output-dir outputs

# SPLADE + BM25 hybrid (alpha)
vi-retrieval-eval --csv data/CSConDa.csv \
  --method splade+bm25 --fusion alpha --alpha 0.7 \
  --splade-model naver/splade-v3 \
  --max-samples 1000 --prefer-unique --dedup --output-dir outputs

# ColBERT (late interaction)
vi-retrieval-eval --csv data/CSConDa.csv --method colbert \
  --colbert-model colbert-ir/colbertv2.0 \
  --max-samples 1000 --prefer-unique --dedup --output-dir outputs
```

### Method × Backend Reference

Not every method needs the same set of flags. The table below shows exactly what is required for each `--method` value, alongside the supported fusion strategies.

| `--method`     | Required flags                                              | Notes                                        |
| -------------- | ----------------------------------------------------------- | -------------------------------------------- |
| `tfidf`        | _(none)_                                                    | Pure lexical                                 |
| `bm25`         | _(none)_                                                    | `--bm25-k1`, `--bm25-b` to tune              |
| `dense`        | `--dense-backend`, model flag                               | Backends: `openai`, `gemini`, `sbert`, `llm` |
| `dense+tfidf`  | `--dense-backend`, model flag, `--fusion`                   | $\alpha$ or RRF                              |
| `dense+bm25`   | `--dense-backend`, model flag, `--fusion`                   | $\alpha$ or RRF                              |
| `splade`       | `--splade-model`                                            | Sparse neural via SparseEncoder              |
| `splade+tfidf` | `--splade-model`, `--fusion`                                | $\alpha$ or RRF                              |
| `splade+bm25`  | `--splade-model`, `--fusion`                                | $\alpha$ or RRF                              |
| `splade+dense` | `--splade-model`, `--dense-backend`, model flag, `--fusion` | Sparse + dense hybrid                        |
| `colbert`      | `--colbert-model`                                           | Requires `colbert-ai` installed              |

Each dense backend uses a different flag to specify the model checkpoint:

- `--dense-backend openai` $\rightarrow$ `--embed-model`
- `--dense-backend gemini` $\rightarrow$ `--gemini-model`
- `--dense-backend sbert` or `llm` $\rightarrow$ `--sbert-model`

### Batch Runs for Multiple Datasets

For large-scale evaluations across multiple datasets, the pattern below is recommended. It handles the special case of ZaloLegalQA, which ships with external qrels rather than inline gold contexts.

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASETS=("CSConDa" "EduCoQA" "VlogQA_2" "ViRe4MRC_v2")
BACKEND="sbert"
SBERT_MODEL="AITeamVN/Vietnamese_Embedding_v2"
BATCH=64
MAX_SAMPLES=1000
K_REF=20
OUTDIR="outputs"

for NAME in "${DATASETS[@]}"; do
  echo "=== $NAME ==="

  if [[ "$NAME" == "ZaloLegalQA" ]]; then
    CSV="data/ZaloLegalQA/dataset.csv"
    QRELS=(--qrels "data/ZaloLegalQA/all.jsonl"
           --csv-qid-col qid --csv-docid-col doc_id
           --qid-col query-id --docid-col corpus-id --rel-col score)
  else
    CSV="data/${NAME}.csv"
    QRELS=()
  fi

  BASE_FLAGS=(--csv "$CSV" "${QRELS[@]}"
    --dense-backend "$BACKEND" --sbert-model "$SBERT_MODEL"
    --batch-size "$BATCH" --max-samples "$MAX_SAMPLES"
    --prefer-unique --dedup --progress --error-k "$K_REF"
    --output-dir "$OUTDIR")

  # 1. Baselines
  vi-retrieval-eval "${BASE_FLAGS[@]}" --method bm25
  vi-retrieval-eval "${BASE_FLAGS[@]}" --method dense

  # 2. Hybrid Alpha
  vi-retrieval-eval "${BASE_FLAGS[@]}" --method dense+bm25 --fusion alpha --alpha 0.7

  # 3. Hybrid RRF (with cross-method intersection)
  vi-retrieval-eval "${BASE_FLAGS[@]}" --method dense+bm25 --fusion rrf --rrf-k 60 \
    --save-intersection

  echo "Done: $NAME"
done
```

---

## Reporting

After running experiments, use [`scripts/summarize_report.py`](scripts/summarize_report.py) to aggregate metrics from multiple output directories into a single Markdown table. It automatically groups runs by dense model, bolds the best value per column, and underlines the second-best.

```bash
python scripts/summarize_report.py \
  --outputs-root outputs \
  --datasets CSConDa EduCoQA ALQAC \
  --metrics P@1,R@10,MRR@10,nDCG@10,R@20 \
  --percent \
  --save VIRE_Report.md
```

**Options:**

- `--outputs-root` — root directory with experimental results
- `--datasets` — space-separated dataset names
- `--metrics` — metric names (aliases supported: `p@1`, `r@10`, `ndcg@10`, etc.)
- `--percent` — display as percentages
- `--ndigits N` — decimal places (default: 4)
- `--include REGEX` — filter method folder names by regex

The script groups results by dense model, bolds the best value per column, and underlines the second-best.

---

## Sample Results

The table below shows benchmark results on the **CSConDa** dataset (customer support, 1,000-sample standardised subset, `--prefer-unique --dedup`, seed 42), comparing all dense models under both $\alpha$-fusion and RRF strategies. Full results across all 10 datasets and all tested models are in [`VIRE_Report.md`](VIRE_Report.md).

### CSConDa (Customer Support)

| Method                                                                       | P@1           | R@10          | MRR@10        | nDCG@10       | R@20          |
| ---------------------------------------------------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| tfidf                                                                        | 15.70%        | 38.50%        | 22.49%        | 26.30%        | 47.20%        |
| bm25                                                                         | 17.40%        | 36.80%        | 22.99%        | 26.27%        | 45.90%        |
| colbert                                                                      | 11.20%        | 28.30%        | 15.79%        | 18.72%        | 35.00%        |
| splade                                                                       | 9.10%         | 22.20%        | 12.65%        | 14.89%        | 27.70%        |
| **Dense model: openai / text-embedding-3-large**                             |               |               |               |               |               |
| dense                                                                        | 33.70%        | 56.80%        | 41.06%        | 44.84%        | 63.80%        |
| dense + tfidf ($\alpha$)                                                     | <u>34.90%</u> | <u>60.40%</u> | 42.45%        | 46.73%        | 66.50%        |
| dense + bm25 ($\alpha$)                                                      | **36.40%**    | 60.20%        | **43.60%**    | <u>47.55%</u> | 66.20%        |
| dense + tfidf (RRF)                                                          | 28.80%        | 55.00%        | 36.45%        | 40.85%        | 63.80%        |
| dense + bm25 (RRF)                                                           | 29.60%        | 54.40%        | 37.05%        | 41.18%        | 64.00%        |
| **Dense model: AITeamVN/Vietnamese_Embedding_v2**                            |               |               |               |               |               |
| dense                                                                        | 31.40%        | 54.00%        | 38.40%        | 42.14%        | 61.40%        |
| dense + tfidf ($\alpha$)                                                     | 32.70%        | 57.50%        | 40.51%        | 44.59%        | 65.30%        |
| dense + bm25 ($\alpha$)                                                      | 33.70%        | 57.90%        | 41.22%        | 45.20%        | 64.30%        |
| dense + tfidf (RRF)                                                          | 28.10%        | 54.60%        | 35.84%        | 40.29%        | 63.90%        |
| dense + bm25 (RRF)                                                           | 28.80%        | 54.70%        | 36.70%        | 40.99%        | 62.40%        |
| **Dense model: bkai-foundation-models/vietnamese-bi-encoder**                |               |               |               |               |               |
| dense                                                                        | 15.70%        | 34.90%        | 21.09%        | 24.34%        | 41.70%        |
| dense + tfidf ($\alpha$)                                                     | 22.20%        | 46.00%        | 28.95%        | 32.96%        | 52.40%        |
| dense + bm25 ($\alpha$)                                                      | 22.70%        | 45.70%        | 28.93%        | 32.87%        | 52.80%        |
| dense + tfidf (RRF)                                                          | 19.10%        | 44.20%        | 26.34%        | 30.56%        | 55.00%        |
| dense + bm25 (RRF)                                                           | 20.00%        | 44.90%        | 26.98%        | 31.19%        | 54.30%        |
| **Dense model: dangvantuan/vietnamese-document-embedding**                   |               |               |               |               |               |
| dense                                                                        | 28.40%        | 53.00%        | 36.12%        | 40.18%        | 59.90%        |
| dense + tfidf ($\alpha$)                                                     | 31.10%        | 57.90%        | 39.46%        | 43.87%        | 65.50%        |
| dense + bm25 ($\alpha$)                                                      | 32.40%        | 57.80%        | 40.05%        | 44.28%        | 64.60%        |
| dense + tfidf (RRF)                                                          | 26.00%        | 51.80%        | 33.88%        | 38.16%        | 64.00%        |
| dense + bm25 (RRF)                                                           | 26.70%        | 52.60%        | 34.65%        | 38.92%        | 63.70%        |
| **Dense model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** |               |               |               |               |               |
| dense                                                                        | 11.80%        | 30.00%        | 16.71%        | 19.82%        | 39.10%        |
| dense + tfidf ($\alpha$)                                                     | 19.60%        | 45.30%        | 27.05%        | 31.38%        | 51.40%        |
| dense + bm25 ($\alpha$)                                                      | 18.90%        | 45.40%        | 26.18%        | 30.71%        | 51.60%        |
| dense + tfidf (RRF)                                                          | 17.70%        | 43.80%        | 25.03%        | 29.47%        | 54.30%        |
| dense + bm25 (RRF)                                                           | 17.50%        | 43.80%        | 24.63%        | 29.15%        | 53.30%        |
| **Dense model: intfloat/multilingual-e5-large**                              |               |               |               |               |               |
| dense                                                                        | 27.20%        | 48.10%        | 33.78%        | 37.21%        | 55.90%        |
| dense + tfidf ($\alpha$)                                                     | 31.90%        | 53.60%        | 38.76%        | 42.32%        | 61.70%        |
| dense + bm25 ($\alpha$)                                                      | 32.60%        | 54.90%        | 39.46%        | 43.16%        | 60.70%        |
| dense + tfidf (RRF)                                                          | 28.70%        | 53.40%        | 36.11%        | 40.22%        | 61.40%        |
| dense + bm25 (RRF)                                                           | 29.00%        | 53.50%        | 36.30%        | 40.39%        | 61.30%        |
| **Dense model: jinaai/jina-embeddings-v3**                                   |               |               |               |               |               |
| dense                                                                        | 32.10%        | 57.40%        | 40.03%        | 44.19%        | 64.30%        |
| dense + tfidf ($\alpha$)                                                     | 34.90%        | **61.20%**    | 42.70%        | 47.12%        | <u>66.90%</u> |
| dense + bm25 ($\alpha$)                                                      | 35.40%        | **61.20%**    | 43.42%        | **47.68%**    | **67.60%**    |
| dense + tfidf (RRF)                                                          | 29.30%        | 57.30%        | 37.96%        | 42.58%        | 66.10%        |
| dense + bm25 (RRF)                                                           | 31.40%        | 56.60%        | 39.01%        | 43.20%        | 65.90%        |
| **Dense model: BAAI/bge-m3**                                                 |               |               |               |               |               |
| dense                                                                        | 30.80%        | 53.90%        | 37.98%        | 41.80%        | 61.00%        |
| dense + tfidf ($\alpha$)                                                     | 33.10%        | 57.00%        | 40.67%        | 44.59%        | 63.80%        |
| dense + bm25 ($\alpha$)                                                      | 33.90%        | 56.90%        | 40.97%        | 44.78%        | 63.90%        |
| dense + tfidf (RRF)                                                          | 28.40%        | 54.20%        | 35.82%        | 40.17%        | 63.90%        |
| dense + bm25 (RRF)                                                           | 28.60%        | 53.70%        | 36.24%        | 40.40%        | 62.80%        |
| **Dense model: Snowflake/snowflake-arctic-embed-l-v2.0**                     |               |               |               |               |               |
| dense                                                                        | 32.80%        | 56.70%        | 40.69%        | 44.56%        | 63.20%        |
| dense + tfidf ($\alpha$)                                                     | 34.30%        | 58.80%        | 42.15%        | 46.15%        | 66.20%        |
| dense + bm25 ($\alpha$)                                                      | <u>35.60%</u> | 59.80%        | <u>43.56%</u> | 47.46%        | 66.30%        |
| dense + tfidf (RRF)                                                          | 30.10%        | 55.30%        | 37.85%        | 42.03%        | 65.00%        |
| dense + bm25 (RRF)                                                           | 29.80%        | 56.40%        | 38.16%        | 42.54%        | 65.20%        |
| **Dense model: Alibaba-NLP/gte-multilingual-base**                           |               |               |               |               |               |
| dense                                                                        | 28.10%        | 51.60%        | 35.23%        | 39.13%        | 57.70%        |
| dense + tfidf ($\alpha$)                                                     | 29.30%        | 55.50%        | 37.45%        | 41.78%        | 63.50%        |
| dense + bm25 ($\alpha$)                                                      | 30.80%        | 56.80%        | 38.82%        | 43.12%        | 63.20%        |
| dense + tfidf (RRF)                                                          | 26.70%        | 52.10%        | 34.42%        | 38.64%        | 62.10%        |
| dense + bm25 (RRF)                                                           | 27.20%        | 52.00%        | 34.88%        | 38.98%        | 62.20%        |
| **Dense model: BAAI/bge-multilingual-gemma2**                                |               |               |               |               |               |
| dense                                                                        | 14.30%        | 30.50%        | 18.65%        | 21.43%        | 37.00%        |
| dense + tfidf ($\alpha$)                                                     | 23.50%        | 43.00%        | 29.72%        | 32.91%        | 49.60%        |
| dense + bm25 ($\alpha$)                                                      | 23.30%        | 43.30%        | 29.50%        | 32.81%        | 49.90%        |
| dense + tfidf (RRF)                                                          | 19.10%        | 43.30%        | 25.92%        | 30.01%        | 52.20%        |
| dense + bm25 (RRF)                                                           | 20.30%        | 43.10%        | 27.02%        | 30.84%        | 51.80%        |
| **Dense model: google/EmbeddingGemma-300m**                                  |               |               |               |               |               |
| dense                                                                        | 29.90%        | 54.70%        | 37.34%        | 41.49%        | 60.80%        |
| dense + tfidf ($\alpha$)                                                     | 32.50%        | 57.80%        | 40.50%        | 44.65%        | 65.10%        |
| dense + bm25 ($\alpha$)                                                      | 33.60%        | 58.80%        | 41.28%        | 45.45%        | 64.90%        |
| dense + tfidf (RRF)                                                          | 28.30%        | 54.60%        | 36.23%        | 40.61%        | 63.20%        |
| dense + bm25 (RRF)                                                           | 28.70%        | 54.50%        | 36.35%        | 40.67%        | 63.30%        |
| **Dense model: Alibaba-NLP/gte-Qwen2-1.5B-instruct**                         |               |               |               |               |               |
| dense                                                                        | 6.70%         | 16.50%        | 9.33%         | 11.01%        | 21.20%        |
| dense + tfidf ($\alpha$)                                                     | 17.40%        | 35.80%        | 22.82%        | 25.90%        | 42.20%        |
| dense + bm25 ($\alpha$)                                                      | 16.40%        | 34.70%        | 21.68%        | 24.76%        | 41.00%        |
| dense + tfidf (RRF)                                                          | 11.90%        | 32.10%        | 17.29%        | 20.76%        | 44.50%        |
| dense + bm25 (RRF)                                                           | 12.00%        | 32.00%        | 17.29%        | 20.72%        | 43.80%        |
| **Dense model: Qwen/Qwen3-Embedding-0.6B**                                   |               |               |               |               |               |
| dense                                                                        | 24.80%        | 49.40%        | 32.27%        | 36.34%        | 56.60%        |
| dense + tfidf ($\alpha$)                                                     | 29.10%        | 56.40%        | 37.31%        | 41.85%        | 63.10%        |
| dense + bm25 ($\alpha$)                                                      | 30.10%        | 55.70%        | 38.11%        | 42.33%        | 62.40%        |
| dense + tfidf (RRF)                                                          | 24.60%        | 53.20%        | 33.14%        | 37.91%        | 62.20%        |
| dense + bm25 (RRF)                                                           | 26.70%        | 53.10%        | 34.41%        | 38.85%        | 62.30%        |

---

## Datasets

VIRE covers six Vietnamese domains, spanning both formal and informal language, technical and general content. Two datasets — EduCoQA and CSConDa — are newly proposed in this work; the rest are established public benchmarks.

**Education** — EduCoQA (proposed), ViRHE4QA [1]

**Customer Support** — CSConDa (proposed)

**Legal** — [ALQAC](https://alqac.github.io), [Zalo Legal Text Retrieval](https://challenge.zalo.ai/portal/legal-text-retrieval)

**Healthcare** — ViNewsQA [2], ViMedAQA [3]

**Lifestyle & Reviews** — VlogQA [4], ViRe4MRC [5]

**Cross-domain Open Knowledge** — UIT-ViQuAD 2.0 [6]

> 1,000-sample standardized subsets are in `data/`.

---

## Data Format

Each dataset is expected as a CSV file with at least `question` and `context` columns. For multi-gold scenarios — where a single query has more than one relevant document — an optional qrels file can be provided separately.

```
qid,doc_id,question,context
q1,d12,"How to apply for admission?","Application procedures and requirements..."
q2,d15,"What are the graduation requirements?","Students must complete all required courses..."
```

Optional multi-gold qrels:

```
qid,doc_id,rel
q1,d12,1
q1,d87,1
```

---

## Outputs

Each run writes its results to a self-describing directory under `--output-dir`. Results are organised by model, then dataset, then a method tag that encodes all run parameters — so different experimental configurations never overwrite each other.

```
outputs/<model>/<dataset>/<method-tag>/
├── metrics.json
├── ranks.json
├── errors/
│   └── fail@K.csv
├── index.faiss
├── doc_embeddings.npy
└── query_embeddings.npy
```

Example path:
`outputs/Alibaba-NLP_gte-multilingual-base/CSConDa/dense-s1000-uniq-dedup-sbert-gte-multilingual-base`

---

## Code Structure

The source code lives entirely under `src/vi_retrieval_eval/`. Each module has a single, well-defined responsibility, and embedding backends are isolated so they can be added or swapped without touching the core evaluation logic.

```
src/vi_retrieval_eval/
├── cli.py              # CLI and argument parsing
├── runner.py           # Evaluation orchestrator
├── metrics.py          # P@k, R@k, MRR, nDCG, MAP, etc.
├── lexical.py          # TF-IDF and BM25
├── dense_index.py      # FAISS indexing and dense retrieval
├── fusion.py           # Alpha and RRF fusion
├── embeddings/
│   ├── base.py         #   Registry (@register decorator, get_embedder)
│   ├── openai_embed.py #   OpenAI API
│   ├── gemini_embed.py #   Gemini API
│   ├── sbert_embed.py  #   Local SBERT models
│   ├── llm_embed.py    #   Large LLM embedders (Qwen3, BGE-Gemma2, Jina v3…)
│   ├── splade_backend.py  # SPLADE sparse encoder
│   └── colbert_backend.py # ColBERT late-interaction
├── qrels.py            # Relevance judgment handling
├── io_utils.py         # CSV / JSON / JSONL I/O
├── sampling.py         # Dataset sampling
├── dedup.py            # Corpus deduplication
├── textnorm.py         # Vietnamese text normalization
├── tokenization.py     # Tokenization utilities
├── stats.py            # Dataset statistics
├── progress.py         # Progress bar utilities
└── logging_utils.py    # Logging configuration
```

---

## Reproducibility

All results reported in the paper can be reproduced exactly using [`scripts/eval_full.sh`](scripts/eval_full.sh). The toolkit is designed to minimise sources of non-determinism at every stage of the pipeline.

- Fixed random seeds for sampling and tie-breaking
- Stable mergesort for consistent ranking
- Dependency pinning via `pyproject.toml`
- Cached embeddings and FAISS indices for efficient reruns
- Deterministic text normalization

> **Note on long-document datasets:** for datasets with very long contexts (e.g. ALQAC), pass `--max-len 256` to truncate inputs at the token level before embedding, avoiding OOD inputs and OOM errors. This is applied automatically in `eval_full.sh`.

---

## Extending VIRE

VIRE uses a lightweight plugin architecture throughout, so extending any part of the system requires minimal boilerplate. The most common extension points are listed below.

- **New retriever:** add under `embeddings/`, decorate with `@register("your_name")`
- **New dataset:** convert to CSV/qrels schema or add a loader in `io_utils.py`
- **New fusion:** implement in `fusion.py`, expose via `--fusion your_method`
- **New metrics:** add to `metrics.py` following the existing pattern

---

## License

VIRE is released as open-source software. The two newly contributed datasets are available for non-commercial research use.

- **Code:** MIT
- **New datasets (EduCoQA, CSConDa):** CC BY-NC 4.0
- **Third-party datasets:** original licenses apply

---

## Citation

```bibtex
@inproceedings{nguyen2026vire,
  title  = {Which Works Best for Vietnamese? A Practical Study of
            Information Retrieval Methods across Domains},
  author = {Nguyen, Long S. T. and Quan, Tho T.},
  booktitle = {Proceedings of the 19th Conference of the European Chapter
               of the Association for Computational Linguistics (EACL 2026)},
  year   = {2026},
  url    = {https://github.com/longstnguyen/ViRE}
}
```

## References

[1] T. P. P. Do et al., "R2GQA: retriever-reader-generator question answering system to support students understanding legal regulations in higher education", _Artificial Intelligence and Law_, May 2025.

[2] K. Van Nguyen et al., "New Vietnamese Corpus for Machine Reading Comprehension of Health News Articles", _ACM TALIP_, vol. 21, no. 5, Sep. 2022.

[3] M.-N. Tran et al., "ViMedAQA: A Vietnamese Medical Abstractive Question-Answering Dataset", in _ACL 2024 Student Research Workshop_, Bangkok, 2024.

[4] T. Ngo et al., "VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension", in _EACL 2024_, Malta, 2024.

[5] T. P. P. Do et al., "Machine Reading Comprehension for Vietnamese Customer Reviews", in _PACLIC 37_, Hong Kong, 2023.

[6] K. Van Nguyen et al., "A Vietnamese Dataset for Evaluating Machine Reading Comprehension", in _COLING 2020_, Barcelona, 2020.
