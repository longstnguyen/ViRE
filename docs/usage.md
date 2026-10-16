<p align="center"><img src="assets/usage-header.svg" width="100%" alt="ViRE usage guide: models, data, evaluation, and reusable recipes."></p>

# Using ViRE

[← Toolkit overview](../README.md) · [Quickstart](../README.md#quickstart) · [Results](../VIRE_Report.md)

Run commands from the repository root with the virtual environment activated. The [quickstart](../README.md#quickstart) installs the CPU lexical pipeline. Install `python -m pip install -e ".[all]"` for the optional neural and API backends.

## On this page

Follow the guide from model selection to report export, or jump directly to a task.

[Embedding backends](#choose-an-embedding-backend) · [Retrieval methods](#sparse-late-interaction-and-hybrid-retrieval) · [Data preparation](#prepare-your-data) · [Multiple relevant documents](#multiple-relevant-documents) · [Metrics & failures](#read-metrics-and-failures) · [Batch evaluation](#batch-evaluation) · [CLI reference](#cli-reference)

## Choose an embedding backend

Select a local model or an API provider independently of the retrieval method. The model flags below apply to standalone dense retrieval and hybrids that include a dense branch.

| Backend | Flag | Model selection | Additional setup |
| :-- | :-- | :-- | :-- |
| Local sentence encoder | `--dense-backend sbert` | `--sbert-model` | Optional dependencies; model downloaded on first use |
| Local LLM embedder | `--dense-backend llm` | `--sbert-model` | Optional dependencies; device memory sized for the model |
| OpenAI | `--dense-backend openai` | `--embed-model` | `OPENAI_API_KEY` |
| Gemini | `--dense-backend gemini` | `--gemini-model` | `GEMINI_API_KEY` |

Use `vi-retrieval-eval --list-backends` to inspect registered backends. Set API credentials in your shell environment; API runs send the selected texts to the chosen provider.

### Local dense retrieval

Encode queries and passages with a Vietnamese sentence encoder and evaluate their similarity on a small, seeded sample.

```bash
vi-retrieval-eval --csv data/CSConDa.csv --method dense \
  --dense-backend sbert --sbert-model AITeamVN/Vietnamese_Embedding_v2 \
  --max-samples 100 --sample-seed 42 --prefer-unique --dedup \
  --batch-size 16 --progress --output-dir outputs/vietnamese-v2
```

### Local LLM embeddings

Use the LLM embedding backend for decoder-based models, with an explicit input-length limit and batch size.

```bash
vi-retrieval-eval --csv data/CSConDa.csv --method dense \
  --dense-backend llm --sbert-model Qwen/Qwen3-Embedding-0.6B \
  --max-samples 100 --sample-seed 42 --prefer-unique --dedup \
  --batch-size 16 --max-len 512 --progress --output-dir outputs/qwen3-0.6b
```

`--max-len` controls token truncation before embedding. Reduce `--batch-size` to fit the available device memory.

### API embeddings

After configuring the relevant API key:

```bash
vi-retrieval-eval --csv data/CSConDa.csv --method dense \
  --dense-backend openai --embed-model text-embedding-3-large \
  --max-samples 100 --sample-seed 42 --prefer-unique --dedup \
  --output-dir outputs/openai-3-large
```

The Gemini backend uses the same workflow with `--dense-backend gemini --gemini-model MODEL_ID`. Select a model available to your account.

## Sparse, late-interaction, and hybrid retrieval

These recipes use the same 100-pair sample as the dense examples, making it easy to compare retrieval families while keeping the input data fixed.

### SPLADE

Evaluate learned sparse representations with the selected SPLADE checkpoint.

```bash
vi-retrieval-eval --csv data/CSConDa.csv --method splade \
  --splade-model naver/splade-v3 --batch-size 16 \
  --max-samples 100 --sample-seed 42 --prefer-unique --dedup \
  --output-dir outputs/splade-v3
```

### ColBERT

Install the separate ColBERT package in the neural environment:

```bash
python -m pip install "colbert-ai==0.2.22" --no-deps

vi-retrieval-eval --csv data/CSConDa.csv --method colbert \
  --colbert-model colbert-ir/colbertv2.0 --batch-size 16 \
  --max-samples 100 --sample-seed 42 --prefer-unique --dedup \
  --output-dir outputs/colbert-v2
```

### Reciprocal rank fusion

Combine dense and BM25 rankings with RRF, using the same local encoder as the dense example.

```bash
vi-retrieval-eval --csv data/CSConDa.csv --method dense+bm25 \
  --dense-backend sbert --sbert-model AITeamVN/Vietnamese_Embedding_v2 \
  --fusion rrf --rrf-k 60 --batch-size 16 \
  --max-samples 100 --sample-seed 42 --prefer-unique --dedup \
  --output-dir outputs/vietnamese-v2
```

For score fusion, replace `--fusion rrf --rrf-k 60` with `--fusion alpha --alpha 0.7`. Scores are min–max normalized per query before interpolation. The dense branch receives weight α in dense–lexical hybrids and `splade+dense`. For `splade+tfidf` and `splade+bm25`, SPLADE receives weight α.

The same fusion choices apply to `dense+tfidf`, `splade+tfidf`, `splade+bm25`, and `splade+dense`. For `splade+dense`, supply both `--splade-model` and the dense-backend model flags.

## Prepare your data

CSV and JSONL inputs use `question` and `context` fields. Without external qrels, each row's context is relevant to that row's question. All retained contexts are retrieval candidates.

- `--normalize-all` applies NFKC normalization, removes invisible/control characters and emoji-like symbols, collapses whitespace, and lowercases questions and contexts.
- `--max-samples N` or `--sample-frac F` selects paired rows using `--sample-seed` (default: 42).
- `--prefer-unique` prioritizes distinct normalized contexts during sampling.
- `--dedup` merges identical candidate contexts and remaps the relevant-document indices; questions remain separate.

Use consistent preparation and sampling flags when comparing retrieval methods.

## Multiple relevant documents

Supply stable IDs in the input and list relevant query–document pairs in a separate qrels file. Every referenced document must be present in the evaluated corpus.

`your_dataset.csv`:

```csv
qid,doc_id,question,context
q1,d1,"Hồ sơ nhập học gồm những gì?","Hồ sơ cần có giấy báo trúng tuyển."
q2,d2,"Có cần mang giấy tờ cá nhân không?","Sinh viên cần mang giấy tờ cá nhân khi nhập học."
```

`your_qrels.csv`:

```csv
qid,doc_id,rel
q1,d1,1
q1,d2,1
q2,d2,1
```

```bash
vi-retrieval-eval --csv your_dataset.csv --qrels your_qrels.csv \
  --csv-qid-col qid --csv-docid-col doc_id \
  --method bm25 --ks 1,2 --output-dir outputs/custom
```

Qrels accept CSV, TSV, or JSONL. Use `--qid-col`, `--docid-col`, and `--rel-col` when the qrels columns have different names. Positive relevance values identify relevant documents.

## Read metrics and failures

`--ks 1,10,20` selects ranking cutoffs. ViRE reports Precision, Recall, HitRate, MRR, MAP, nDCG, R-Precision, and first-relevant-rank statistics. Normalized metrics in `metrics.json` use a 0–1 scale; report export with `--percent` displays percentages.

```text
outputs/
└── lexical/                         # chosen model/group output root
    └── CSConDa/
        ├── statistics/              # sampled input and metadata
        └── bm25-s100-uniq-dedup/
            ├── metrics.json
            ├── ranks.json
            └── errors/
                └── fail@20.csv
```

Dense runs also save embeddings and a FAISS index. Use `--force` to rebuild caches. Choose separate output roots for experiments that vary data, preprocessing, or model configuration.

`--error-k 10` exports failures at cutoff 10. `--save-intersection` compares failure sets among method runs under the same dataset output directory. Keep the sampled data and query identities consistent for this comparison.

## Batch evaluation

[`scripts/eval_full.sh`](../scripts/eval_full.sh) defines the dataset paths, model list, batch size, sampling cap, fusion parameters, and long-document truncation settings. It includes neural downloads and OpenAI API runs; edit its arrays and configuration to select the experiments you intend to run.

```bash
bash scripts/eval_full.sh

python scripts/summarize_report.py --outputs-root outputs \
  --datasets CSConDa EduCoQA ALQAC \
  --metrics P@1,R@10,MRR@10,nDCG@10,R@20 \
  --percent --save outputs/report.md
```

The report scanner expects `outputs/<model-or-group>/<dataset>/<method-tag>/metrics.json`. The CLI writes `<dataset>/<method-tag>/` beneath the root passed to `--output-dir`, so use a root such as `outputs/lexical` or `outputs/vietnamese-v2` for individual runs.

The report groups runs by dataset, model, and method. Use separate top-level output directories for independent sweeps; inspect the script's duplicate-run options when combining runs with several parameter settings.

## CLI reference

Use these option groups to configure a run. The command help lists accepted values and defaults for every flag.

```bash
vi-retrieval-eval --help
python scripts/summarize_report.py --help
```

| Option group | Flags | Purpose |
| :-- | :-- | :-- |
| Input | `--csv`, `--qrels` | Load question–context pairs and optional relevance judgments |
| Document and query IDs | `--csv-qid-col`, `--csv-docid-col`, `--qid-col`, `--docid-col`, `--rel-col` | Map input and qrels fields |
| Sampling | `--max-samples`, `--sample-frac`, `--sample-seed`, `--prefer-unique`, `--unique-col` | Select the evaluation sample |
| Text preparation | `--normalize-all`, `--dedup`, `--dedup-lower`, `--dedup-remove-emoji` | Normalize text and merge duplicate contexts |
| Retrieval | `--method`, `--bm25-k1`, `--bm25-b`, `--index-metric` | Select and configure the retriever |
| Embedding backend | `--dense-backend`, `--sbert-model`, `--embed-model`, `--gemini-model`, `--list-backends` | Choose a local encoder or API provider |
| Sparse and late interaction | `--splade-model`, `--colbert-model` | Select the model checkpoint |
| Embedding inputs | `--batch-size`, `--max-len` | Control batching and token truncation |
| Fusion | `--fusion`, `--alpha`, `--rrf-k` | Combine retrieval scores or ranks |
| Evaluation | `--ks`, `--error-k`, `--save-intersection`, `--max-errors` | Set ranking cutoffs and inspect failures |
| Output and cache | `--output-dir`, `--force` | Choose the result directory and rebuild cached artifacts |
| Logging | `--progress`, `--show-size`, `--log-level`, `--log-file` | Inspect run progress and input statistics |

[← Back to the toolkit](../README.md) · [Explore benchmark results →](../VIRE_Report.md)
