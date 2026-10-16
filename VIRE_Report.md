<p align="center"><img src="docs/assets/results-header.svg" width="100%" alt="ViRE benchmark results across ten Vietnamese datasets."></p>

# ViRE Retrieval Report

[← Toolkit overview](README.md) · [Key findings](README.md#findings) · [Run an evaluation](docs/usage.md)

Compare lexical, learned-sparse, late-interaction, dense, and hybrid retrieval across the benchmark. Choose a dataset below and expand its table to inspect the individual configurations.

## Reading the results

All scores are percentages on a **0–100 scale**, with higher values indicating better retrieval. **Bold** marks the best score per metric within each dataset; <ins>underlining</ins> marks the second distinct score. Ties share the same highlighting.

| Metric | What it measures |
| :-- | :-- |
| P@1 | Precision of the top-ranked result |
| R@10 | Fraction of relevant documents found in the first ten results |
| MRR@10 | Reciprocal rank of the first relevant result, up to rank ten |
| nDCG@10 | Ranking quality with greater weight on relevant results near the top |
| R@20 | Fraction of relevant documents found in the first twenty results |

Each table starts with the standalone baselines, followed by dense backbones and their hybrid variants. **α** denotes linear score fusion; **RRF** denotes reciprocal rank fusion. See the [method overview](README.md#methods) for the scoring conventions.

## Datasets

Browse results by domain. The [README summary](README.md#results) provides a compact comparison of BM25, the best dense configuration, and the best hybrid configuration on each dataset.

| Domain | Datasets |
| :-- | :-- |
| Education | [EduCoQA](#educoqa) · [ViRHE4QA](#virhe4qa_v2) |
| Customer support | [CSConDa](#csconda) |
| Legal | [ALQAC](#alqac) · [ZaloLegalQA](#zalolegalqa) |
| Healthcare | [ViNewsQA](#vinewsqa) · [ViMedAQA](#vimedaqa_v2) |
| Lifestyle & reviews | [VlogQA](#vlogqa_2) · [ViRe4MRC](#vire4mrc_v2) |
| Open knowledge | [UIT-ViQuAD 2.0](#uit-viquad2) |

## ALQAC

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">82.83%</td><td align="right">96.23%</td><td align="right">88.34%</td><td align="right">90.31%</td><td align="right">98.11%</td></tr>
<tr><td align="left">bm25</td><td align="right">89.25%</td><td align="right">97.92%</td><td align="right">92.20%</td><td align="right">93.59%</td><td align="right">99.25%</td></tr>
<tr><td align="left">splade</td><td align="right">68.68%</td><td align="right">91.32%</td><td align="right">76.33%</td><td align="right">79.97%</td><td align="right">95.66%</td></tr>
<tr><td align="left">colbert</td><td align="right">69.43%</td><td align="right">89.62%</td><td align="right">76.42%</td><td align="right">79.63%</td><td align="right">93.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">88.30%</td><td align="right">98.30%</td><td align="right">92.04%</td><td align="right">93.59%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">90.00%</td><td align="right">98.87%</td><td align="right">93.46%</td><td align="right">94.81%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">88.11%</td><td align="right">98.68%</td><td align="right">92.06%</td><td align="right">93.70%</td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">91.89%</td><td align="right">99.06%</td><td align="right">94.90%</td><td align="right">95.94%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">90.38%</td><td align="right">99.25%</td><td align="right">93.84%</td><td align="right">95.18%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">65.47%</td><td align="right">89.62%</td><td align="right">73.38%</td><td align="right">77.29%</td><td align="right">94.34%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">82.26%</td><td align="right">96.79%</td><td align="right">88.03%</td><td align="right">90.21%</td><td align="right">98.11%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.23%</td><td align="right">95.66%</td><td align="right">83.16%</td><td align="right">86.22%</td><td align="right">98.11%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">87.92%</td><td align="right">97.55%</td><td align="right">91.67%</td><td align="right">93.13%</td><td align="right">98.11%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">79.06%</td><td align="right">96.60%</td><td align="right">85.19%</td><td align="right">87.97%</td><td align="right">98.49%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">85.28%</td><td align="right">99.06%</td><td align="right">90.37%</td><td align="right">92.51%</td><td align="right">99.43%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">88.87%</td><td align="right">99.06%</td><td align="right">93.20%</td><td align="right">94.67%</td><td align="right">99.43%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">88.11%</td><td align="right">98.11%</td><td align="right">92.31%</td><td align="right">93.76%</td><td align="right">99.06%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">92.26%</td><td align="right">99.62%</td><td align="right">95.22%</td><td align="right">96.32%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">92.64%</td><td align="right">98.87%</td><td align="right">95.01%</td><td align="right">95.97%</td><td align="right">99.43%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">87.74%</td><td align="right">99.06%</td><td align="right">92.07%</td><td align="right">93.80%</td><td align="right">99.43%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">89.25%</td><td align="right">99.43%</td><td align="right">93.20%</td><td align="right">94.74%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">87.17%</td><td align="right">99.06%</td><td align="right">91.70%</td><td align="right">93.52%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">91.32%</td><td align="right"><strong>100.00%</strong></td><td align="right">94.61%</td><td align="right">95.93%</td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">90.38%</td><td align="right">99.62%</td><td align="right">94.00%</td><td align="right">95.40%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">89.06%</td><td align="right">99.25%</td><td align="right">92.91%</td><td align="right">94.48%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">89.81%</td><td align="right">98.87%</td><td align="right">93.70%</td><td align="right">95.01%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">86.98%</td><td align="right">98.87%</td><td align="right">91.84%</td><td align="right">93.60%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">92.64%</td><td align="right">99.43%</td><td align="right">95.47%</td><td align="right">96.47%</td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">91.32%</td><td align="right">99.25%</td><td align="right">94.73%</td><td align="right">95.87%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">84.34%</td><td align="right">98.49%</td><td align="right">89.64%</td><td align="right">91.83%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">87.92%</td><td align="right">99.06%</td><td align="right">92.65%</td><td align="right">94.27%</td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">85.28%</td><td align="right">99.06%</td><td align="right">90.99%</td><td align="right">93.01%</td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">90.38%</td><td align="right">99.62%</td><td align="right">94.12%</td><td align="right">95.49%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">89.62%</td><td align="right">99.25%</td><td align="right">93.95%</td><td align="right">95.29%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">84.53%</td><td align="right">98.68%</td><td align="right">90.13%</td><td align="right">92.26%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">88.87%</td><td align="right">99.25%</td><td align="right">93.22%</td><td align="right">94.74%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">87.36%</td><td align="right">98.30%</td><td align="right">91.83%</td><td align="right">93.45%</td><td align="right">99.06%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">93.02%</td><td align="right">99.43%</td><td align="right">95.79%</td><td align="right">96.72%</td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">90.57%</td><td align="right">98.87%</td><td align="right">94.06%</td><td align="right">95.28%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">85.66%</td><td align="right">99.06%</td><td align="right">90.44%</td><td align="right">92.54%</td><td align="right">99.43%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">89.62%</td><td align="right">98.68%</td><td align="right">93.05%</td><td align="right">94.44%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">86.42%</td><td align="right">98.68%</td><td align="right">91.25%</td><td align="right">93.09%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">91.51%</td><td align="right"><ins>99.81%</ins></td><td align="right">94.60%</td><td align="right">95.88%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">89.81%</td><td align="right">99.43%</td><td align="right">93.71%</td><td align="right">95.14%</td><td align="right">99.62%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">90.38%</td><td align="right">99.43%</td><td align="right">94.14%</td><td align="right">95.47%</td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">92.08%</td><td align="right">99.43%</td><td align="right">95.02%</td><td align="right">96.12%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">88.49%</td><td align="right">98.49%</td><td align="right">92.37%</td><td align="right">93.88%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>94.72%</strong></td><td align="right">99.62%</td><td align="right"><strong>96.66%</strong></td><td align="right"><strong>97.40%</strong></td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">91.13%</td><td align="right">99.06%</td><td align="right">94.40%</td><td align="right">95.56%</td><td align="right">99.62%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">65.85%</td><td align="right">92.45%</td><td align="right">74.37%</td><td align="right">78.73%</td><td align="right">95.09%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">83.96%</td><td align="right">97.36%</td><td align="right">89.27%</td><td align="right">91.29%</td><td align="right">99.06%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.55%</td><td align="right">96.79%</td><td align="right">84.64%</td><td align="right">87.62%</td><td align="right">98.68%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">88.30%</td><td align="right">97.36%</td><td align="right">91.80%</td><td align="right">93.18%</td><td align="right">99.25%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">80.19%</td><td align="right">97.55%</td><td align="right">86.93%</td><td align="right">89.55%</td><td align="right">98.68%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">80.75%</td><td align="right">95.66%</td><td align="right">86.21%</td><td align="right">88.52%</td><td align="right">98.11%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">88.68%</td><td align="right">98.49%</td><td align="right">92.60%</td><td align="right">94.07%</td><td align="right">99.25%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">84.53%</td><td align="right">97.74%</td><td align="right">89.43%</td><td align="right">91.48%</td><td align="right">99.25%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">89.62%</td><td align="right">98.68%</td><td align="right">93.33%</td><td align="right">94.67%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">88.30%</td><td align="right">98.30%</td><td align="right">92.13%</td><td align="right">93.65%</td><td align="right">99.62%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">85.85%</td><td align="right">98.49%</td><td align="right">90.51%</td><td align="right">92.48%</td><td align="right">99.25%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">89.06%</td><td align="right">98.87%</td><td align="right">92.96%</td><td align="right">94.44%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">87.92%</td><td align="right">97.92%</td><td align="right">91.80%</td><td align="right">93.32%</td><td align="right">99.43%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">92.64%</td><td align="right">99.25%</td><td align="right">95.21%</td><td align="right">96.21%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">89.43%</td><td align="right">98.68%</td><td align="right">93.28%</td><td align="right">94.63%</td><td align="right">99.43%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">90.38%</td><td align="right">99.06%</td><td align="right">93.96%</td><td align="right">95.24%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">92.64%</td><td align="right">99.62%</td><td align="right">95.26%</td><td align="right">96.33%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">90.19%</td><td align="right">98.49%</td><td align="right">93.15%</td><td align="right">94.44%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>93.77%</ins></td><td align="right">99.25%</td><td align="right"><ins>95.95%</ins></td><td align="right"><ins>96.78%</ins></td><td align="right"><strong>100.00%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">92.08%</td><td align="right">98.68%</td><td align="right">94.78%</td><td align="right">95.75%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">88.30%</td><td align="right">99.43%</td><td align="right">92.68%</td><td align="right">94.36%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">89.81%</td><td align="right">99.25%</td><td align="right">93.51%</td><td align="right">94.94%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">87.17%</td><td align="right">99.06%</td><td align="right">91.76%</td><td align="right">93.57%</td><td align="right">99.62%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">92.83%</td><td align="right">99.62%</td><td align="right">95.62%</td><td align="right">96.62%</td><td align="right"><ins>99.81%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">91.70%</td><td align="right">99.25%</td><td align="right">94.89%</td><td align="right">95.99%</td><td align="right">99.62%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## CSConDa

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">15.70%</td><td align="right">38.50%</td><td align="right">22.49%</td><td align="right">26.30%</td><td align="right">47.20%</td></tr>
<tr><td align="left">bm25</td><td align="right">17.40%</td><td align="right">36.80%</td><td align="right">22.99%</td><td align="right">26.27%</td><td align="right">45.90%</td></tr>
<tr><td align="left">splade</td><td align="right">9.10%</td><td align="right">22.20%</td><td align="right">12.65%</td><td align="right">14.89%</td><td align="right">27.70%</td></tr>
<tr><td align="left">colbert</td><td align="right">11.20%</td><td align="right">28.30%</td><td align="right">15.79%</td><td align="right">18.72%</td><td align="right">35.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">28.10%</td><td align="right">51.60%</td><td align="right">35.23%</td><td align="right">39.13%</td><td align="right">57.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">29.30%</td><td align="right">55.50%</td><td align="right">37.45%</td><td align="right">41.78%</td><td align="right">63.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">26.70%</td><td align="right">52.10%</td><td align="right">34.42%</td><td align="right">38.64%</td><td align="right">62.10%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">30.80%</td><td align="right">56.80%</td><td align="right">38.82%</td><td align="right">43.12%</td><td align="right">63.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">27.20%</td><td align="right">52.00%</td><td align="right">34.88%</td><td align="right">38.98%</td><td align="right">62.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">6.70%</td><td align="right">16.50%</td><td align="right">9.33%</td><td align="right">11.01%</td><td align="right">21.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">17.40%</td><td align="right">35.80%</td><td align="right">22.82%</td><td align="right">25.90%</td><td align="right">42.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">11.90%</td><td align="right">32.10%</td><td align="right">17.29%</td><td align="right">20.76%</td><td align="right">44.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">16.40%</td><td align="right">34.70%</td><td align="right">21.68%</td><td align="right">24.76%</td><td align="right">41.00%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">12.00%</td><td align="right">32.00%</td><td align="right">17.29%</td><td align="right">20.72%</td><td align="right">43.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">14.30%</td><td align="right">30.50%</td><td align="right">18.65%</td><td align="right">21.43%</td><td align="right">37.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">23.50%</td><td align="right">43.00%</td><td align="right">29.72%</td><td align="right">32.91%</td><td align="right">49.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">19.10%</td><td align="right">43.30%</td><td align="right">25.92%</td><td align="right">30.01%</td><td align="right">52.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">23.30%</td><td align="right">43.30%</td><td align="right">29.50%</td><td align="right">32.81%</td><td align="right">49.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">20.30%</td><td align="right">43.10%</td><td align="right">27.02%</td><td align="right">30.84%</td><td align="right">51.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">29.90%</td><td align="right">54.70%</td><td align="right">37.34%</td><td align="right">41.49%</td><td align="right">60.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">32.50%</td><td align="right">57.80%</td><td align="right">40.50%</td><td align="right">44.65%</td><td align="right">65.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">28.30%</td><td align="right">54.60%</td><td align="right">36.23%</td><td align="right">40.61%</td><td align="right">63.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">33.60%</td><td align="right">58.80%</td><td align="right">41.28%</td><td align="right">45.45%</td><td align="right">64.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">28.70%</td><td align="right">54.50%</td><td align="right">36.35%</td><td align="right">40.67%</td><td align="right">63.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">27.20%</td><td align="right">48.10%</td><td align="right">33.78%</td><td align="right">37.21%</td><td align="right">55.90%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">31.90%</td><td align="right">53.60%</td><td align="right">38.76%</td><td align="right">42.32%</td><td align="right">61.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">28.70%</td><td align="right">53.40%</td><td align="right">36.11%</td><td align="right">40.22%</td><td align="right">61.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">32.60%</td><td align="right">54.90%</td><td align="right">39.46%</td><td align="right">43.16%</td><td align="right">60.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">29.00%</td><td align="right">53.50%</td><td align="right">36.30%</td><td align="right">40.39%</td><td align="right">61.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">32.10%</td><td align="right">57.40%</td><td align="right">40.03%</td><td align="right">44.19%</td><td align="right">64.30%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">34.90%</td><td align="right"><strong>61.20%</strong></td><td align="right">42.70%</td><td align="right">47.12%</td><td align="right"><ins>66.90%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">29.30%</td><td align="right">57.30%</td><td align="right">37.96%</td><td align="right">42.58%</td><td align="right">66.10%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">35.40%</td><td align="right"><strong>61.20%</strong></td><td align="right">43.42%</td><td align="right"><strong>47.68%</strong></td><td align="right"><strong>67.60%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">31.40%</td><td align="right">56.60%</td><td align="right">39.01%</td><td align="right">43.20%</td><td align="right">65.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">33.70%</td><td align="right">56.80%</td><td align="right">41.06%</td><td align="right">44.84%</td><td align="right">63.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">34.90%</td><td align="right"><ins>60.40%</ins></td><td align="right">42.45%</td><td align="right">46.73%</td><td align="right">66.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">28.80%</td><td align="right">55.00%</td><td align="right">36.45%</td><td align="right">40.85%</td><td align="right">63.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>36.40%</strong></td><td align="right">60.20%</td><td align="right"><strong>43.60%</strong></td><td align="right"><ins>47.55%</ins></td><td align="right">66.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">29.60%</td><td align="right">54.40%</td><td align="right">37.05%</td><td align="right">41.18%</td><td align="right">64.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">24.80%</td><td align="right">49.40%</td><td align="right">32.27%</td><td align="right">36.34%</td><td align="right">56.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">29.10%</td><td align="right">56.40%</td><td align="right">37.31%</td><td align="right">41.85%</td><td align="right">63.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">24.60%</td><td align="right">53.20%</td><td align="right">33.14%</td><td align="right">37.91%</td><td align="right">62.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">30.10%</td><td align="right">55.70%</td><td align="right">38.11%</td><td align="right">42.33%</td><td align="right">62.40%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">26.70%</td><td align="right">53.10%</td><td align="right">34.41%</td><td align="right">38.85%</td><td align="right">62.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">30.80%</td><td align="right">53.90%</td><td align="right">37.98%</td><td align="right">41.80%</td><td align="right">61.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">33.10%</td><td align="right">57.00%</td><td align="right">40.67%</td><td align="right">44.59%</td><td align="right">63.80%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">28.40%</td><td align="right">54.20%</td><td align="right">35.82%</td><td align="right">40.17%</td><td align="right">63.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">33.90%</td><td align="right">56.90%</td><td align="right">40.97%</td><td align="right">44.78%</td><td align="right">63.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">28.60%</td><td align="right">53.70%</td><td align="right">36.24%</td><td align="right">40.40%</td><td align="right">62.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">11.80%</td><td align="right">30.00%</td><td align="right">16.71%</td><td align="right">19.82%</td><td align="right">39.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">19.60%</td><td align="right">45.30%</td><td align="right">27.05%</td><td align="right">31.38%</td><td align="right">51.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">17.70%</td><td align="right">43.80%</td><td align="right">25.03%</td><td align="right">29.47%</td><td align="right">54.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">18.90%</td><td align="right">45.40%</td><td align="right">26.18%</td><td align="right">30.71%</td><td align="right">51.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">17.50%</td><td align="right">43.80%</td><td align="right">24.63%</td><td align="right">29.15%</td><td align="right">53.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">15.70%</td><td align="right">34.90%</td><td align="right">21.09%</td><td align="right">24.34%</td><td align="right">41.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">22.20%</td><td align="right">46.00%</td><td align="right">28.95%</td><td align="right">32.96%</td><td align="right">52.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">19.10%</td><td align="right">44.20%</td><td align="right">26.34%</td><td align="right">30.56%</td><td align="right">55.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">22.70%</td><td align="right">45.70%</td><td align="right">28.93%</td><td align="right">32.87%</td><td align="right">52.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">20.00%</td><td align="right">44.90%</td><td align="right">26.98%</td><td align="right">31.19%</td><td align="right">54.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">28.40%</td><td align="right">53.00%</td><td align="right">36.12%</td><td align="right">40.18%</td><td align="right">59.90%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">31.10%</td><td align="right">57.90%</td><td align="right">39.46%</td><td align="right">43.87%</td><td align="right">65.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">26.00%</td><td align="right">51.80%</td><td align="right">33.88%</td><td align="right">38.16%</td><td align="right">64.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">32.40%</td><td align="right">57.80%</td><td align="right">40.05%</td><td align="right">44.28%</td><td align="right">64.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">26.70%</td><td align="right">52.60%</td><td align="right">34.65%</td><td align="right">38.92%</td><td align="right">63.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">31.40%</td><td align="right">54.00%</td><td align="right">38.40%</td><td align="right">42.14%</td><td align="right">61.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">32.70%</td><td align="right">57.50%</td><td align="right">40.51%</td><td align="right">44.59%</td><td align="right">65.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">28.10%</td><td align="right">54.60%</td><td align="right">35.84%</td><td align="right">40.29%</td><td align="right">63.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">33.70%</td><td align="right">57.90%</td><td align="right">41.22%</td><td align="right">45.20%</td><td align="right">64.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">28.80%</td><td align="right">54.70%</td><td align="right">36.70%</td><td align="right">40.99%</td><td align="right">62.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">32.80%</td><td align="right">56.70%</td><td align="right">40.69%</td><td align="right">44.56%</td><td align="right">63.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">34.30%</td><td align="right">58.80%</td><td align="right">42.15%</td><td align="right">46.15%</td><td align="right">66.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">30.10%</td><td align="right">55.30%</td><td align="right">37.85%</td><td align="right">42.03%</td><td align="right">65.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>35.60%</ins></td><td align="right">59.80%</td><td align="right"><ins>43.56%</ins></td><td align="right">47.46%</td><td align="right">66.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">29.80%</td><td align="right">56.40%</td><td align="right">38.16%</td><td align="right">42.54%</td><td align="right">65.20%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## EduCoQA

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">14.68%</td><td align="right">42.47%</td><td align="right">22.23%</td><td align="right">26.99%</td><td align="right">53.82%</td></tr>
<tr><td align="left">bm25</td><td align="right">14.68%</td><td align="right">43.44%</td><td align="right">23.09%</td><td align="right">27.93%</td><td align="right">53.42%</td></tr>
<tr><td align="left">splade</td><td align="right">11.35%</td><td align="right">30.53%</td><td align="right">16.62%</td><td align="right">19.91%</td><td align="right">39.92%</td></tr>
<tr><td align="left">colbert</td><td align="right">14.48%</td><td align="right">39.33%</td><td align="right">21.54%</td><td align="right">25.75%</td><td align="right">48.73%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">18.00%</td><td align="right">52.45%</td><td align="right">28.72%</td><td align="right">34.42%</td><td align="right">62.82%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">21.53%</td><td align="right">59.10%</td><td align="right">31.79%</td><td align="right">38.23%</td><td align="right">68.49%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">18.98%</td><td align="right">54.99%</td><td align="right">29.02%</td><td align="right">35.17%</td><td align="right">69.47%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">21.53%</td><td align="right">59.69%</td><td align="right">32.41%</td><td align="right">38.88%</td><td align="right">69.28%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">19.37%</td><td align="right">57.34%</td><td align="right">29.98%</td><td align="right">36.43%</td><td align="right">68.69%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">6.26%</td><td align="right">21.92%</td><td align="right">9.98%</td><td align="right">12.75%</td><td align="right">29.55%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">11.94%</td><td align="right">38.75%</td><td align="right">19.55%</td><td align="right">24.08%</td><td align="right">46.77%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">9.39%</td><td align="right">30.72%</td><td align="right">15.07%</td><td align="right">18.72%</td><td align="right">44.03%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">13.50%</td><td align="right">38.16%</td><td align="right">20.55%</td><td align="right">24.71%</td><td align="right">47.55%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">10.18%</td><td align="right">28.96%</td><td align="right">15.67%</td><td align="right">18.80%</td><td align="right">42.07%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">12.13%</td><td align="right">40.90%</td><td align="right">20.03%</td><td align="right">24.95%</td><td align="right">51.08%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">19.18%</td><td align="right">46.18%</td><td align="right">26.40%</td><td align="right">31.06%</td><td align="right">57.53%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">17.81%</td><td align="right">46.58%</td><td align="right">25.81%</td><td align="right">30.71%</td><td align="right">56.75%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">19.18%</td><td align="right">47.16%</td><td align="right">27.13%</td><td align="right">31.88%</td><td align="right">57.93%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">18.40%</td><td align="right">46.38%</td><td align="right">26.21%</td><td align="right">30.97%</td><td align="right">59.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">21.14%</td><td align="right">53.82%</td><td align="right">30.70%</td><td align="right">36.19%</td><td align="right">62.23%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">22.31%</td><td align="right">58.12%</td><td align="right">32.23%</td><td align="right">38.35%</td><td align="right">66.14%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">20.94%</td><td align="right">56.75%</td><td align="right">31.12%</td><td align="right">37.17%</td><td align="right">66.93%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">22.50%</td><td align="right">57.73%</td><td align="right">32.81%</td><td align="right">38.73%</td><td align="right">65.75%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">20.16%</td><td align="right">57.53%</td><td align="right">30.89%</td><td align="right">37.20%</td><td align="right">65.95%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">23.87%</td><td align="right">53.82%</td><td align="right">32.70%</td><td align="right">37.72%</td><td align="right">65.56%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">23.29%</td><td align="right"><strong>61.06%</strong></td><td align="right">33.62%</td><td align="right">40.08%</td><td align="right">68.88%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">23.48%</td><td align="right">59.10%</td><td align="right">32.62%</td><td align="right">38.81%</td><td align="right">68.69%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>25.24%</strong></td><td align="right"><ins>60.47%</ins></td><td align="right"><strong>35.24%</strong></td><td align="right"><strong>41.19%</strong></td><td align="right"><strong>70.06%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">21.92%</td><td align="right">57.73%</td><td align="right">32.41%</td><td align="right">38.42%</td><td align="right">69.08%</td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">21.72%</td><td align="right">52.05%</td><td align="right">31.01%</td><td align="right">36.02%</td><td align="right">62.43%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">22.11%</td><td align="right">58.32%</td><td align="right">32.34%</td><td align="right">38.49%</td><td align="right">68.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">21.33%</td><td align="right">56.36%</td><td align="right">30.61%</td><td align="right">36.66%</td><td align="right">69.28%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">23.48%</td><td align="right">59.30%</td><td align="right">34.13%</td><td align="right"><ins>40.12%</ins></td><td align="right">67.91%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">20.74%</td><td align="right">56.75%</td><td align="right">30.81%</td><td align="right">36.93%</td><td align="right">69.28%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">20.16%</td><td align="right">51.86%</td><td align="right">30.30%</td><td align="right">35.47%</td><td align="right">63.01%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">22.70%</td><td align="right">57.34%</td><td align="right">32.40%</td><td align="right">38.30%</td><td align="right">66.54%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">20.55%</td><td align="right">55.19%</td><td align="right">29.80%</td><td align="right">35.75%</td><td align="right">67.12%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">22.11%</td><td align="right">57.34%</td><td align="right">32.54%</td><td align="right">38.43%</td><td align="right">65.95%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">22.90%</td><td align="right">54.99%</td><td align="right">31.36%</td><td align="right">36.89%</td><td align="right">66.14%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">21.53%</td><td align="right">54.99%</td><td align="right">31.50%</td><td align="right">37.10%</td><td align="right">65.95%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">21.72%</td><td align="right">59.69%</td><td align="right">32.47%</td><td align="right">38.92%</td><td align="right"><ins>69.67%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">19.77%</td><td align="right">58.32%</td><td align="right">30.32%</td><td align="right">36.92%</td><td align="right">68.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">22.70%</td><td align="right">60.08%</td><td align="right">33.58%</td><td align="right">39.88%</td><td align="right">69.28%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">20.55%</td><td align="right">59.30%</td><td align="right">31.33%</td><td align="right">37.92%</td><td align="right">67.71%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right"><ins>24.66%</ins></td><td align="right">55.77%</td><td align="right"><ins>34.22%</ins></td><td align="right">39.37%</td><td align="right">64.77%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">23.68%</td><td align="right">57.93%</td><td align="right">33.78%</td><td align="right">39.52%</td><td align="right">67.51%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">20.55%</td><td align="right">53.82%</td><td align="right">29.72%</td><td align="right">35.40%</td><td align="right">67.32%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">22.90%</td><td align="right">58.71%</td><td align="right">33.68%</td><td align="right">39.63%</td><td align="right">67.51%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">21.92%</td><td align="right">55.38%</td><td align="right">31.13%</td><td align="right">36.83%</td><td align="right">67.12%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">14.48%</td><td align="right">40.70%</td><td align="right">20.90%</td><td align="right">25.49%</td><td align="right">52.05%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">18.79%</td><td align="right">50.68%</td><td align="right">27.07%</td><td align="right">32.59%</td><td align="right">61.25%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">17.81%</td><td align="right">51.47%</td><td align="right">25.98%</td><td align="right">31.88%</td><td align="right">61.84%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">18.59%</td><td align="right">52.84%</td><td align="right">27.68%</td><td align="right">33.57%</td><td align="right">61.45%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">18.59%</td><td align="right">52.84%</td><td align="right">27.06%</td><td align="right">33.03%</td><td align="right">61.64%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">18.79%</td><td align="right">48.53%</td><td align="right">27.01%</td><td align="right">32.07%</td><td align="right">58.51%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">19.18%</td><td align="right">52.25%</td><td align="right">28.58%</td><td align="right">34.19%</td><td align="right">64.97%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">20.16%</td><td align="right">52.05%</td><td align="right">28.53%</td><td align="right">34.05%</td><td align="right">64.77%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">20.74%</td><td align="right">52.84%</td><td align="right">29.95%</td><td align="right">35.38%</td><td align="right">64.19%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">20.94%</td><td align="right">52.64%</td><td align="right">29.52%</td><td align="right">34.97%</td><td align="right">64.97%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">20.55%</td><td align="right">53.42%</td><td align="right">30.76%</td><td align="right">36.19%</td><td align="right">63.21%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">20.94%</td><td align="right">56.75%</td><td align="right">31.15%</td><td align="right">37.21%</td><td align="right">66.14%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">20.16%</td><td align="right">54.99%</td><td align="right">28.89%</td><td align="right">34.97%</td><td align="right">64.97%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">21.92%</td><td align="right">56.36%</td><td align="right">31.91%</td><td align="right">37.74%</td><td align="right">65.56%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">19.96%</td><td align="right">54.79%</td><td align="right">29.33%</td><td align="right">35.31%</td><td align="right">64.77%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">19.77%</td><td align="right">50.29%</td><td align="right">28.99%</td><td align="right">34.08%</td><td align="right">59.88%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">18.98%</td><td align="right">56.36%</td><td align="right">29.33%</td><td align="right">35.70%</td><td align="right">65.36%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">19.37%</td><td align="right">53.82%</td><td align="right">28.80%</td><td align="right">34.70%</td><td align="right">65.36%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">19.57%</td><td align="right">56.16%</td><td align="right">29.93%</td><td align="right">36.14%</td><td align="right">65.17%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">21.14%</td><td align="right">53.23%</td><td align="right">29.90%</td><td align="right">35.39%</td><td align="right">64.58%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">20.74%</td><td align="right">54.79%</td><td align="right">31.05%</td><td align="right">36.73%</td><td align="right">64.58%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">22.50%</td><td align="right">58.90%</td><td align="right">32.47%</td><td align="right">38.69%</td><td align="right">68.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">19.96%</td><td align="right">54.99%</td><td align="right">29.70%</td><td align="right">35.66%</td><td align="right">67.91%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">23.48%</td><td align="right">58.51%</td><td align="right">33.33%</td><td align="right">39.28%</td><td align="right">68.49%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">20.35%</td><td align="right">55.77%</td><td align="right">30.49%</td><td align="right">36.48%</td><td align="right">66.54%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## UIT-ViQuAD2

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">50.00%</td><td align="right">91.00%</td><td align="right">64.57%</td><td align="right">71.05%</td><td align="right">94.00%</td></tr>
<tr><td align="left">bm25</td><td align="right">70.80%</td><td align="right">91.60%</td><td align="right">78.09%</td><td align="right">81.38%</td><td align="right">93.90%</td></tr>
<tr><td align="left">splade</td><td align="right">44.30%</td><td align="right">73.10%</td><td align="right">53.57%</td><td align="right">58.27%</td><td align="right">79.00%</td></tr>
<tr><td align="left">colbert</td><td align="right">50.90%</td><td align="right">75.10%</td><td align="right">58.26%</td><td align="right">62.29%</td><td align="right">81.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">75.00%</td><td align="right">95.20%</td><td align="right">82.34%</td><td align="right">85.49%</td><td align="right">97.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">80.40%</td><td align="right">98.20%</td><td align="right">86.92%</td><td align="right">89.69%</td><td align="right">98.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.30%</td><td align="right">97.50%</td><td align="right">84.12%</td><td align="right">87.42%</td><td align="right">98.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">85.00%</td><td align="right">98.50%</td><td align="right">90.26%</td><td align="right">92.31%</td><td align="right">99.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">82.30%</td><td align="right">98.10%</td><td align="right">88.43%</td><td align="right">90.83%</td><td align="right">99.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">56.70%</td><td align="right">84.10%</td><td align="right">65.64%</td><td align="right">70.09%</td><td align="right">88.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">72.30%</td><td align="right">94.90%</td><td align="right">80.73%</td><td align="right">84.22%</td><td align="right">97.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">67.10%</td><td align="right">92.20%</td><td align="right">75.85%</td><td align="right">79.83%</td><td align="right">96.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">79.20%</td><td align="right">95.50%</td><td align="right">85.30%</td><td align="right">87.82%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">69.30%</td><td align="right">93.10%</td><td align="right">77.90%</td><td align="right">81.61%</td><td align="right">96.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">70.90%</td><td align="right">93.00%</td><td align="right">78.74%</td><td align="right">82.23%</td><td align="right">96.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">78.70%</td><td align="right">96.80%</td><td align="right">85.60%</td><td align="right">88.37%</td><td align="right">98.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">75.50%</td><td align="right">95.60%</td><td align="right">82.95%</td><td align="right">86.07%</td><td align="right">98.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">83.10%</td><td align="right">97.50%</td><td align="right">88.42%</td><td align="right">90.66%</td><td align="right">98.50%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">77.90%</td><td align="right">96.30%</td><td align="right">84.82%</td><td align="right">87.65%</td><td align="right">98.60%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">76.90%</td><td align="right">94.90%</td><td align="right">83.45%</td><td align="right">86.26%</td><td align="right">97.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">81.80%</td><td align="right">98.50%</td><td align="right">88.43%</td><td align="right">90.94%</td><td align="right">98.80%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.40%</td><td align="right">97.80%</td><td align="right">85.00%</td><td align="right">88.17%</td><td align="right">98.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">86.20%</td><td align="right">98.50%</td><td align="right">91.24%</td><td align="right">93.06%</td><td align="right">99.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">83.50%</td><td align="right">98.20%</td><td align="right">89.30%</td><td align="right">91.52%</td><td align="right">99.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">83.60%</td><td align="right">97.90%</td><td align="right">89.29%</td><td align="right">91.43%</td><td align="right">98.90%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">85.70%</td><td align="right">99.00%</td><td align="right">90.83%</td><td align="right">92.86%</td><td align="right">99.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">78.60%</td><td align="right">98.40%</td><td align="right">86.52%</td><td align="right">89.49%</td><td align="right">99.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>89.70%</strong></td><td align="right">98.90%</td><td align="right"><strong>93.62%</strong></td><td align="right"><strong>94.96%</strong></td><td align="right"><strong>99.50%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">86.50%</td><td align="right">98.70%</td><td align="right">91.27%</td><td align="right">93.12%</td><td align="right"><ins>99.40%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">72.20%</td><td align="right">93.20%</td><td align="right">79.54%</td><td align="right">82.87%</td><td align="right">95.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.70%</td><td align="right">97.80%</td><td align="right">86.36%</td><td align="right">89.18%</td><td align="right">98.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">74.50%</td><td align="right">97.10%</td><td align="right">83.04%</td><td align="right">86.53%</td><td align="right">98.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">84.60%</td><td align="right">98.00%</td><td align="right">89.77%</td><td align="right">91.81%</td><td align="right">98.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">80.40%</td><td align="right">97.60%</td><td align="right">87.26%</td><td align="right">89.84%</td><td align="right">98.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">71.20%</td><td align="right">92.40%</td><td align="right">78.75%</td><td align="right">82.09%</td><td align="right">96.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.30%</td><td align="right">97.20%</td><td align="right">86.24%</td><td align="right">88.96%</td><td align="right">98.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">74.10%</td><td align="right">96.10%</td><td align="right">82.18%</td><td align="right">85.60%</td><td align="right">98.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">82.90%</td><td align="right">97.00%</td><td align="right">88.46%</td><td align="right">90.58%</td><td align="right">98.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">77.80%</td><td align="right">95.80%</td><td align="right">84.59%</td><td align="right">87.37%</td><td align="right">98.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">73.70%</td><td align="right">94.10%</td><td align="right">80.76%</td><td align="right">84.01%</td><td align="right">97.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">78.20%</td><td align="right">97.80%</td><td align="right">85.81%</td><td align="right">88.79%</td><td align="right">99.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">74.10%</td><td align="right">97.60%</td><td align="right">82.96%</td><td align="right">86.58%</td><td align="right">98.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">85.30%</td><td align="right">98.30%</td><td align="right">90.41%</td><td align="right">92.38%</td><td align="right">99.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">81.60%</td><td align="right">98.00%</td><td align="right">88.06%</td><td align="right">90.53%</td><td align="right">99.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">80.60%</td><td align="right">96.40%</td><td align="right">86.62%</td><td align="right">89.04%</td><td align="right">98.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">83.20%</td><td align="right">98.70%</td><td align="right">89.48%</td><td align="right">91.78%</td><td align="right">99.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.20%</td><td align="right">97.60%</td><td align="right">84.95%</td><td align="right">88.07%</td><td align="right">99.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">88.30%</td><td align="right">99.00%</td><td align="right">92.41%</td><td align="right">94.04%</td><td align="right">99.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">80.80%</td><td align="right">97.30%</td><td align="right">87.15%</td><td align="right">89.67%</td><td align="right">99.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">55.90%</td><td align="right">81.30%</td><td align="right">64.15%</td><td align="right">68.28%</td><td align="right">87.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">70.20%</td><td align="right">93.90%</td><td align="right">78.74%</td><td align="right">82.46%</td><td align="right">96.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">66.30%</td><td align="right">93.20%</td><td align="right">75.45%</td><td align="right">79.76%</td><td align="right">96.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">75.80%</td><td align="right">95.10%</td><td align="right">82.76%</td><td align="right">85.78%</td><td align="right">97.50%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">69.00%</td><td align="right">93.60%</td><td align="right">77.59%</td><td align="right">81.49%</td><td align="right">97.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">68.00%</td><td align="right">88.40%</td><td align="right">74.62%</td><td align="right">77.94%</td><td align="right">92.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">77.40%</td><td align="right">95.00%</td><td align="right">83.93%</td><td align="right">86.66%</td><td align="right">97.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">72.20%</td><td align="right">94.70%</td><td align="right">80.16%</td><td align="right">83.71%</td><td align="right">97.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">81.20%</td><td align="right">96.10%</td><td align="right">86.71%</td><td align="right">89.02%</td><td align="right">97.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">75.70%</td><td align="right">94.90%</td><td align="right">82.50%</td><td align="right">85.54%</td><td align="right">97.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">75.70%</td><td align="right">95.60%</td><td align="right">83.00%</td><td align="right">86.10%</td><td align="right">97.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">81.40%</td><td align="right">97.50%</td><td align="right">87.45%</td><td align="right">89.95%</td><td align="right">99.00%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.40%</td><td align="right">97.10%</td><td align="right">83.82%</td><td align="right">87.07%</td><td align="right">98.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">85.50%</td><td align="right">98.70%</td><td align="right">90.34%</td><td align="right">92.40%</td><td align="right">99.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">79.60%</td><td align="right">97.10%</td><td align="right">86.07%</td><td align="right">88.79%</td><td align="right">99.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">82.20%</td><td align="right">98.10%</td><td align="right">88.24%</td><td align="right">90.67%</td><td align="right">99.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">84.40%</td><td align="right"><strong>99.30%</strong></td><td align="right">90.18%</td><td align="right">92.45%</td><td align="right">99.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">79.00%</td><td align="right">97.60%</td><td align="right">86.02%</td><td align="right">88.87%</td><td align="right">99.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>89.10%</ins></td><td align="right"><ins>99.20%</ins></td><td align="right"><ins>93.00%</ins></td><td align="right"><ins>94.53%</ins></td><td align="right">99.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">82.10%</td><td align="right">97.50%</td><td align="right">87.95%</td><td align="right">90.32%</td><td align="right">99.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">75.40%</td><td align="right">94.70%</td><td align="right">82.31%</td><td align="right">85.33%</td><td align="right">96.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.80%</td><td align="right">97.20%</td><td align="right">86.41%</td><td align="right">89.08%</td><td align="right">98.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">75.30%</td><td align="right">97.10%</td><td align="right">83.51%</td><td align="right">86.87%</td><td align="right">98.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">84.50%</td><td align="right">97.70%</td><td align="right">89.71%</td><td align="right">91.70%</td><td align="right">98.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">80.70%</td><td align="right">97.00%</td><td align="right">87.10%</td><td align="right">89.57%</td><td align="right">98.60%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## ViMedAQA_v2

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">61.50%</td><td align="right">84.80%</td><td align="right">69.46%</td><td align="right">73.18%</td><td align="right">88.20%</td></tr>
<tr><td align="left">bm25</td><td align="right">65.40%</td><td align="right">84.50%</td><td align="right">71.36%</td><td align="right">74.51%</td><td align="right">87.30%</td></tr>
<tr><td align="left">splade</td><td align="right">47.40%</td><td align="right">68.80%</td><td align="right">54.08%</td><td align="right">57.60%</td><td align="right">73.60%</td></tr>
<tr><td align="left">colbert</td><td align="right">49.60%</td><td align="right">71.40%</td><td align="right">56.42%</td><td align="right">60.01%</td><td align="right">75.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">73.90%</td><td align="right">89.90%</td><td align="right">79.53%</td><td align="right">82.05%</td><td align="right">92.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.70%</td><td align="right">93.30%</td><td align="right">84.58%</td><td align="right">86.71%</td><td align="right">95.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">74.60%</td><td align="right">91.60%</td><td align="right">80.99%</td><td align="right">83.61%</td><td align="right">94.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">80.30%</td><td align="right">93.20%</td><td align="right">84.95%</td><td align="right">86.96%</td><td align="right">95.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">76.70%</td><td align="right">92.30%</td><td align="right">82.10%</td><td align="right">84.58%</td><td align="right">94.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">59.90%</td><td align="right">81.20%</td><td align="right">66.93%</td><td align="right">70.37%</td><td align="right">84.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">66.70%</td><td align="right">89.00%</td><td align="right">74.62%</td><td align="right">78.14%</td><td align="right">91.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">64.60%</td><td align="right">84.10%</td><td align="right">71.55%</td><td align="right">74.62%</td><td align="right">89.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">71.50%</td><td align="right">88.50%</td><td align="right">76.95%</td><td align="right">79.73%</td><td align="right">90.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">66.70%</td><td align="right">84.00%</td><td align="right">72.53%</td><td align="right">75.31%</td><td align="right">89.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">58.40%</td><td align="right">80.20%</td><td align="right">65.39%</td><td align="right">68.94%</td><td align="right">86.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">73.70%</td><td align="right">92.30%</td><td align="right">80.22%</td><td align="right">83.18%</td><td align="right">94.00%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">69.50%</td><td align="right">90.60%</td><td align="right">76.46%</td><td align="right">79.87%</td><td align="right">93.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">76.20%</td><td align="right">92.60%</td><td align="right">81.99%</td><td align="right">84.59%</td><td align="right">94.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">71.30%</td><td align="right">91.10%</td><td align="right">78.11%</td><td align="right">81.26%</td><td align="right">94.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">77.80%</td><td align="right">95.30%</td><td align="right">83.62%</td><td align="right">86.45%</td><td align="right">97.30%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">81.20%</td><td align="right">96.10%</td><td align="right">86.50%</td><td align="right">88.85%</td><td align="right"><strong>98.20%</strong></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.80%</td><td align="right">93.90%</td><td align="right">83.90%</td><td align="right">86.38%</td><td align="right">95.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>84.10%</ins></td><td align="right">96.10%</td><td align="right">88.27%</td><td align="right">90.18%</td><td align="right"><strong>98.20%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">80.50%</td><td align="right">93.60%</td><td align="right">85.35%</td><td align="right">87.38%</td><td align="right">96.60%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">83.30%</td><td align="right"><ins>96.40%</ins></td><td align="right">87.94%</td><td align="right">90.01%</td><td align="right">97.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">82.80%</td><td align="right">95.90%</td><td align="right">87.46%</td><td align="right">89.51%</td><td align="right">97.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">78.40%</td><td align="right">94.20%</td><td align="right">84.38%</td><td align="right">86.80%</td><td align="right">96.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>84.70%</strong></td><td align="right"><strong>96.50%</strong></td><td align="right"><strong>88.82%</strong></td><td align="right"><strong>90.69%</strong></td><td align="right">97.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">81.20%</td><td align="right">94.60%</td><td align="right">85.92%</td><td align="right">88.03%</td><td align="right">96.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">79.90%</td><td align="right">95.40%</td><td align="right">85.29%</td><td align="right">87.75%</td><td align="right">97.30%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">82.50%</td><td align="right">95.80%</td><td align="right">87.39%</td><td align="right">89.45%</td><td align="right">97.80%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.10%</td><td align="right">94.10%</td><td align="right">83.83%</td><td align="right">86.38%</td><td align="right">95.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>84.70%</strong></td><td align="right">96.00%</td><td align="right"><ins>88.66%</ins></td><td align="right"><ins>90.45%</ins></td><td align="right">97.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">79.60%</td><td align="right">94.30%</td><td align="right">85.04%</td><td align="right">87.30%</td><td align="right">96.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">80.40%</td><td align="right">95.40%</td><td align="right">85.44%</td><td align="right">87.84%</td><td align="right">97.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">80.90%</td><td align="right">96.30%</td><td align="right">86.22%</td><td align="right">88.66%</td><td align="right">97.80%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">74.70%</td><td align="right">91.70%</td><td align="right">80.63%</td><td align="right">83.33%</td><td align="right">95.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">83.00%</td><td align="right">96.10%</td><td align="right">87.37%</td><td align="right">89.48%</td><td align="right"><ins>97.90%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">77.70%</td><td align="right">92.10%</td><td align="right">82.16%</td><td align="right">84.53%</td><td align="right">95.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">77.50%</td><td align="right">93.00%</td><td align="right">82.75%</td><td align="right">85.23%</td><td align="right">95.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">82.10%</td><td align="right">94.40%</td><td align="right">86.56%</td><td align="right">88.48%</td><td align="right">96.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.30%</td><td align="right">93.50%</td><td align="right">83.26%</td><td align="right">85.77%</td><td align="right">95.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">82.80%</td><td align="right">94.80%</td><td align="right">87.12%</td><td align="right">89.00%</td><td align="right">96.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">79.30%</td><td align="right">93.30%</td><td align="right">84.28%</td><td align="right">86.49%</td><td align="right">95.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">81.20%</td><td align="right">94.10%</td><td align="right">85.60%</td><td align="right">87.67%</td><td align="right">96.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">81.30%</td><td align="right">94.40%</td><td align="right">85.95%</td><td align="right">88.01%</td><td align="right">96.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">74.70%</td><td align="right">91.40%</td><td align="right">80.76%</td><td align="right">83.36%</td><td align="right">94.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">82.50%</td><td align="right">94.60%</td><td align="right">86.58%</td><td align="right">88.52%</td><td align="right">96.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">76.80%</td><td align="right">91.30%</td><td align="right">81.59%</td><td align="right">83.93%</td><td align="right">94.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">48.00%</td><td align="right">73.50%</td><td align="right">56.19%</td><td align="right">60.34%</td><td align="right">80.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">67.00%</td><td align="right">87.10%</td><td align="right">74.05%</td><td align="right">77.22%</td><td align="right">90.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">61.40%</td><td align="right">84.20%</td><td align="right">69.16%</td><td align="right">72.80%</td><td align="right">90.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">69.70%</td><td align="right">87.00%</td><td align="right">75.24%</td><td align="right">78.06%</td><td align="right">90.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">63.90%</td><td align="right">83.80%</td><td align="right">70.27%</td><td align="right">73.51%</td><td align="right">89.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">70.10%</td><td align="right">87.20%</td><td align="right">75.68%</td><td align="right">78.46%</td><td align="right">90.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">75.70%</td><td align="right">90.90%</td><td align="right">81.10%</td><td align="right">83.50%</td><td align="right">93.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">69.90%</td><td align="right">89.30%</td><td align="right">76.88%</td><td align="right">79.92%</td><td align="right">92.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">77.10%</td><td align="right">90.80%</td><td align="right">82.00%</td><td align="right">84.15%</td><td align="right">92.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">72.80%</td><td align="right">89.30%</td><td align="right">78.34%</td><td align="right">80.99%</td><td align="right">92.60%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">75.50%</td><td align="right">90.60%</td><td align="right">80.86%</td><td align="right">83.24%</td><td align="right">93.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">78.30%</td><td align="right">93.10%</td><td align="right">83.73%</td><td align="right">86.03%</td><td align="right">94.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">73.00%</td><td align="right">91.10%</td><td align="right">79.39%</td><td align="right">82.23%</td><td align="right">94.10%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">79.80%</td><td align="right">92.60%</td><td align="right">84.37%</td><td align="right">86.39%</td><td align="right">94.50%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">74.70%</td><td align="right">90.40%</td><td align="right">80.03%</td><td align="right">82.54%</td><td align="right">93.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">76.90%</td><td align="right">93.50%</td><td align="right">82.40%</td><td align="right">85.06%</td><td align="right">96.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.80%</td><td align="right">94.10%</td><td align="right">84.61%</td><td align="right">86.91%</td><td align="right">96.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">75.00%</td><td align="right">91.80%</td><td align="right">80.92%</td><td align="right">83.57%</td><td align="right">94.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">81.40%</td><td align="right">93.80%</td><td align="right">85.38%</td><td align="right">87.40%</td><td align="right">95.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">76.60%</td><td align="right">90.80%</td><td align="right">81.43%</td><td align="right">83.71%</td><td align="right">94.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">79.40%</td><td align="right">94.30%</td><td align="right">84.56%</td><td align="right">86.93%</td><td align="right">96.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">81.20%</td><td align="right">94.70%</td><td align="right">86.03%</td><td align="right">88.15%</td><td align="right">96.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.60%</td><td align="right">93.50%</td><td align="right">82.97%</td><td align="right">85.56%</td><td align="right">95.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">83.80%</td><td align="right">95.10%</td><td align="right">87.75%</td><td align="right">89.53%</td><td align="right">96.50%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">80.30%</td><td align="right">93.60%</td><td align="right">84.89%</td><td align="right">87.00%</td><td align="right">95.60%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## ViNewsQA

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">52.20%</td><td align="right">79.10%</td><td align="right">60.93%</td><td align="right">65.31%</td><td align="right">84.70%</td></tr>
<tr><td align="left">bm25</td><td align="right">59.00%</td><td align="right">80.20%</td><td align="right">66.13%</td><td align="right">69.53%</td><td align="right">84.30%</td></tr>
<tr><td align="left">splade</td><td align="right">26.70%</td><td align="right">50.20%</td><td align="right">33.44%</td><td align="right">37.40%</td><td align="right">58.00%</td></tr>
<tr><td align="left">colbert</td><td align="right">29.70%</td><td align="right">51.00%</td><td align="right">35.77%</td><td align="right">39.37%</td><td align="right">57.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">53.00%</td><td align="right">76.60%</td><td align="right">60.79%</td><td align="right">64.60%</td><td align="right">80.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">61.20%</td><td align="right">83.00%</td><td align="right">68.51%</td><td align="right">72.03%</td><td align="right">87.80%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">58.30%</td><td align="right">82.40%</td><td align="right">65.97%</td><td align="right">69.92%</td><td align="right">88.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">63.30%</td><td align="right">85.30%</td><td align="right">70.70%</td><td align="right">74.23%</td><td align="right">89.50%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">62.10%</td><td align="right">84.80%</td><td align="right">69.39%</td><td align="right">73.10%</td><td align="right">89.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">54.00%</td><td align="right">75.30%</td><td align="right">61.21%</td><td align="right">64.63%</td><td align="right">80.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">61.60%</td><td align="right">82.80%</td><td align="right">68.73%</td><td align="right">72.13%</td><td align="right">86.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">57.20%</td><td align="right">82.20%</td><td align="right">65.52%</td><td align="right">69.54%</td><td align="right">87.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">65.60%</td><td align="right">84.30%</td><td align="right">71.77%</td><td align="right">74.80%</td><td align="right">88.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">63.30%</td><td align="right">84.10%</td><td align="right">70.18%</td><td align="right">73.53%</td><td align="right">89.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">44.00%</td><td align="right">70.40%</td><td align="right">52.36%</td><td align="right">56.68%</td><td align="right">75.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">58.30%</td><td align="right">82.00%</td><td align="right">66.30%</td><td align="right">70.11%</td><td align="right">86.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">53.50%</td><td align="right">80.00%</td><td align="right">62.29%</td><td align="right">66.57%</td><td align="right">87.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">61.80%</td><td align="right">84.20%</td><td align="right">68.99%</td><td align="right">72.65%</td><td align="right">88.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">58.70%</td><td align="right">83.40%</td><td align="right">66.65%</td><td align="right">70.68%</td><td align="right">89.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">54.30%</td><td align="right">77.90%</td><td align="right">62.21%</td><td align="right">65.99%</td><td align="right">82.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">62.40%</td><td align="right">83.80%</td><td align="right">69.50%</td><td align="right">72.95%</td><td align="right">88.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">58.30%</td><td align="right">83.10%</td><td align="right">66.66%</td><td align="right">70.64%</td><td align="right">88.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">65.50%</td><td align="right">85.80%</td><td align="right">72.45%</td><td align="right">75.69%</td><td align="right">89.50%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">62.70%</td><td align="right">85.80%</td><td align="right">70.33%</td><td align="right">74.06%</td><td align="right">91.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">57.40%</td><td align="right">79.70%</td><td align="right">64.91%</td><td align="right">68.48%</td><td align="right">84.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">64.30%</td><td align="right">85.90%</td><td align="right">71.35%</td><td align="right">74.85%</td><td align="right">89.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">61.40%</td><td align="right">84.80%</td><td align="right">69.07%</td><td align="right">72.87%</td><td align="right">90.10%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>66.10%</ins></td><td align="right"><strong>88.50%</strong></td><td align="right"><ins>73.59%</ins></td><td align="right"><ins>77.19%</ins></td><td align="right"><strong>91.60%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">65.00%</td><td align="right">87.00%</td><td align="right">72.10%</td><td align="right">75.68%</td><td align="right"><ins>91.40%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">55.00%</td><td align="right">76.40%</td><td align="right">61.95%</td><td align="right">65.44%</td><td align="right">81.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">62.40%</td><td align="right">83.50%</td><td align="right">69.46%</td><td align="right">72.86%</td><td align="right">88.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">58.30%</td><td align="right">83.10%</td><td align="right">66.57%</td><td align="right">70.57%</td><td align="right">88.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">64.80%</td><td align="right">84.90%</td><td align="right">71.60%</td><td align="right">74.83%</td><td align="right">90.00%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">63.80%</td><td align="right">85.00%</td><td align="right">70.66%</td><td align="right">74.11%</td><td align="right">90.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">49.20%</td><td align="right">76.40%</td><td align="right">58.59%</td><td align="right">62.90%</td><td align="right">81.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">62.40%</td><td align="right">84.50%</td><td align="right">69.96%</td><td align="right">73.49%</td><td align="right">89.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">56.20%</td><td align="right">83.30%</td><td align="right">65.25%</td><td align="right">69.63%</td><td align="right">89.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">64.70%</td><td align="right">85.90%</td><td align="right">71.83%</td><td align="right">75.23%</td><td align="right">90.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">61.00%</td><td align="right">85.30%</td><td align="right">68.74%</td><td align="right">72.72%</td><td align="right">90.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">55.00%</td><td align="right">76.30%</td><td align="right">61.97%</td><td align="right">65.43%</td><td align="right">81.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">60.60%</td><td align="right">83.70%</td><td align="right">68.34%</td><td align="right">72.04%</td><td align="right">87.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">57.50%</td><td align="right">82.50%</td><td align="right">65.92%</td><td align="right">69.92%</td><td align="right">88.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">64.60%</td><td align="right">85.20%</td><td align="right">71.44%</td><td align="right">74.77%</td><td align="right">89.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">62.70%</td><td align="right">85.00%</td><td align="right">70.16%</td><td align="right">73.75%</td><td align="right">90.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">57.60%</td><td align="right">79.00%</td><td align="right">64.72%</td><td align="right">68.17%</td><td align="right">83.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">63.90%</td><td align="right">86.40%</td><td align="right">71.41%</td><td align="right">75.04%</td><td align="right">89.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">59.90%</td><td align="right">84.60%</td><td align="right">68.10%</td><td align="right">72.08%</td><td align="right">89.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">65.50%</td><td align="right">87.10%</td><td align="right">72.87%</td><td align="right">76.33%</td><td align="right">90.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">63.70%</td><td align="right">85.20%</td><td align="right">70.63%</td><td align="right">74.12%</td><td align="right">90.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">25.50%</td><td align="right">52.20%</td><td align="right">33.54%</td><td align="right">37.98%</td><td align="right">60.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">50.70%</td><td align="right">76.80%</td><td align="right">58.94%</td><td align="right">63.23%</td><td align="right">83.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">41.70%</td><td align="right">73.40%</td><td align="right">51.25%</td><td align="right">56.53%</td><td align="right">84.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">49.70%</td><td align="right">77.30%</td><td align="right">58.61%</td><td align="right">63.10%</td><td align="right">82.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">44.40%</td><td align="right">76.10%</td><td align="right">53.95%</td><td align="right">59.23%</td><td align="right">85.60%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">45.50%</td><td align="right">67.50%</td><td align="right">52.39%</td><td align="right">56.01%</td><td align="right">73.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">58.80%</td><td align="right">80.70%</td><td align="right">66.41%</td><td align="right">69.88%</td><td align="right">85.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">52.70%</td><td align="right">78.80%</td><td align="right">61.72%</td><td align="right">65.86%</td><td align="right">86.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">59.90%</td><td align="right">80.40%</td><td align="right">66.83%</td><td align="right">70.11%</td><td align="right">85.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">56.90%</td><td align="right">79.60%</td><td align="right">64.12%</td><td align="right">67.83%</td><td align="right">86.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">54.90%</td><td align="right">76.80%</td><td align="right">61.77%</td><td align="right">65.38%</td><td align="right">80.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">61.70%</td><td align="right">84.30%</td><td align="right">69.22%</td><td align="right">72.86%</td><td align="right">88.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">57.90%</td><td align="right">82.30%</td><td align="right">66.03%</td><td align="right">69.97%</td><td align="right">88.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">64.70%</td><td align="right">85.20%</td><td align="right">71.51%</td><td align="right">74.82%</td><td align="right">88.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">62.70%</td><td align="right">83.80%</td><td align="right">69.44%</td><td align="right">72.90%</td><td align="right">89.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">55.40%</td><td align="right">78.20%</td><td align="right">63.01%</td><td align="right">66.69%</td><td align="right">82.90%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">65.30%</td><td align="right">86.10%</td><td align="right">72.35%</td><td align="right">75.68%</td><td align="right">90.00%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">62.10%</td><td align="right">85.40%</td><td align="right">69.71%</td><td align="right">73.48%</td><td align="right">89.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>68.60%</strong></td><td align="right"><ins>87.30%</ins></td><td align="right"><strong>74.96%</strong></td><td align="right"><strong>77.95%</strong></td><td align="right">90.00%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">65.10%</td><td align="right">86.30%</td><td align="right">72.00%</td><td align="right">75.44%</td><td align="right">89.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">52.70%</td><td align="right">75.00%</td><td align="right">60.12%</td><td align="right">63.71%</td><td align="right">79.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">60.20%</td><td align="right">82.30%</td><td align="right">67.62%</td><td align="right">71.18%</td><td align="right">87.00%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">56.10%</td><td align="right">81.80%</td><td align="right">64.69%</td><td align="right">68.83%</td><td align="right">87.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">63.20%</td><td align="right">84.50%</td><td align="right">70.37%</td><td align="right">73.78%</td><td align="right">88.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">61.30%</td><td align="right">83.70%</td><td align="right">68.72%</td><td align="right">72.34%</td><td align="right">89.20%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## ViRe4MRC_v2

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">3.70%</td><td align="right">17.50%</td><td align="right">7.24%</td><td align="right">9.63%</td><td align="right">23.70%</td></tr>
<tr><td align="left">bm25</td><td align="right">6.60%</td><td align="right">20.40%</td><td align="right">10.25%</td><td align="right">12.62%</td><td align="right">26.70%</td></tr>
<tr><td align="left">splade</td><td align="right">5.20%</td><td align="right">13.00%</td><td align="right">7.40%</td><td align="right">8.72%</td><td align="right">19.20%</td></tr>
<tr><td align="left">colbert</td><td align="right">4.80%</td><td align="right">15.90%</td><td align="right">7.66%</td><td align="right">9.59%</td><td align="right">21.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">9.00%</td><td align="right">27.00%</td><td align="right">13.74%</td><td align="right">16.83%</td><td align="right">33.90%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">10.80%</td><td align="right">29.40%</td><td align="right">16.11%</td><td align="right">19.24%</td><td align="right">37.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">9.70%</td><td align="right">28.40%</td><td align="right">14.92%</td><td align="right">18.09%</td><td align="right">36.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">11.50%</td><td align="right">29.90%</td><td align="right">16.64%</td><td align="right">19.76%</td><td align="right">38.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">10.30%</td><td align="right">29.20%</td><td align="right">15.46%</td><td align="right">18.69%</td><td align="right">37.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">2.10%</td><td align="right">9.80%</td><td align="right">4.07%</td><td align="right">5.40%</td><td align="right">13.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">4.00%</td><td align="right">16.10%</td><td align="right">7.32%</td><td align="right">9.39%</td><td align="right">22.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">4.90%</td><td align="right">13.00%</td><td align="right">7.07%</td><td align="right">8.47%</td><td align="right">20.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">5.50%</td><td align="right">17.30%</td><td align="right">8.83%</td><td align="right">10.84%</td><td align="right">22.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">5.60%</td><td align="right">14.10%</td><td align="right">7.80%</td><td align="right">9.27%</td><td align="right">21.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">9.20%</td><td align="right">22.10%</td><td align="right">12.64%</td><td align="right">14.86%</td><td align="right">27.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">8.80%</td><td align="right">24.00%</td><td align="right">13.06%</td><td align="right">15.64%</td><td align="right">31.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">8.70%</td><td align="right">23.30%</td><td align="right">12.51%</td><td align="right">15.04%</td><td align="right">31.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">10.20%</td><td align="right">24.40%</td><td align="right">14.25%</td><td align="right">16.65%</td><td align="right">32.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">9.30%</td><td align="right">23.70%</td><td align="right">13.23%</td><td align="right">15.70%</td><td align="right">33.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">10.60%</td><td align="right">27.50%</td><td align="right">15.19%</td><td align="right">18.09%</td><td align="right">34.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">11.70%</td><td align="right">29.80%</td><td align="right">16.72%</td><td align="right">19.79%</td><td align="right">38.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">10.70%</td><td align="right">28.00%</td><td align="right">15.56%</td><td align="right">18.50%</td><td align="right">37.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">12.50%</td><td align="right"><ins>31.10%</ins></td><td align="right">17.35%</td><td align="right">20.57%</td><td align="right">39.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">11.20%</td><td align="right">30.50%</td><td align="right">16.26%</td><td align="right">19.60%</td><td align="right">37.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">11.80%</td><td align="right">29.00%</td><td align="right">16.77%</td><td align="right">19.66%</td><td align="right">37.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">12.00%</td><td align="right">30.40%</td><td align="right">17.02%</td><td align="right">20.17%</td><td align="right">37.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">11.20%</td><td align="right">29.20%</td><td align="right">16.19%</td><td align="right">19.26%</td><td align="right">37.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">12.40%</td><td align="right"><strong>31.60%</strong></td><td align="right">17.84%</td><td align="right">21.10%</td><td align="right">39.40%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">11.90%</td><td align="right">30.20%</td><td align="right">16.94%</td><td align="right">20.07%</td><td align="right">38.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">9.90%</td><td align="right">27.50%</td><td align="right">14.73%</td><td align="right">17.74%</td><td align="right">36.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">12.10%</td><td align="right">29.70%</td><td align="right">16.90%</td><td align="right">19.90%</td><td align="right">39.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">12.10%</td><td align="right">28.90%</td><td align="right">16.50%</td><td align="right">19.40%</td><td align="right">37.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>13.00%</strong></td><td align="right">30.70%</td><td align="right">17.82%</td><td align="right">20.84%</td><td align="right"><ins>40.10%</ins></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">12.70%</td><td align="right">30.50%</td><td align="right">17.10%</td><td align="right">20.20%</td><td align="right">39.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">9.70%</td><td align="right">30.00%</td><td align="right">15.03%</td><td align="right">18.54%</td><td align="right">38.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">10.80%</td><td align="right">29.60%</td><td align="right">16.20%</td><td align="right">19.37%</td><td align="right">39.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">10.70%</td><td align="right">26.90%</td><td align="right">15.32%</td><td align="right">18.05%</td><td align="right">37.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">12.10%</td><td align="right">30.00%</td><td align="right">16.88%</td><td align="right">19.95%</td><td align="right">39.00%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">10.90%</td><td align="right">28.50%</td><td align="right">15.69%</td><td align="right">18.70%</td><td align="right">38.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">11.50%</td><td align="right">28.10%</td><td align="right">15.97%</td><td align="right">18.81%</td><td align="right">35.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">11.20%</td><td align="right">31.00%</td><td align="right">16.62%</td><td align="right">19.99%</td><td align="right">39.00%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">10.10%</td><td align="right">29.00%</td><td align="right">15.22%</td><td align="right">18.46%</td><td align="right">37.10%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>12.90%</ins></td><td align="right">30.70%</td><td align="right">17.84%</td><td align="right">20.87%</td><td align="right">38.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">11.00%</td><td align="right">30.00%</td><td align="right">16.16%</td><td align="right">19.42%</td><td align="right">36.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">12.40%</td><td align="right">30.80%</td><td align="right">17.25%</td><td align="right">20.42%</td><td align="right">40.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">12.00%</td><td align="right">30.90%</td><td align="right">17.64%</td><td align="right">20.79%</td><td align="right">39.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">10.30%</td><td align="right">28.70%</td><td align="right">15.86%</td><td align="right">18.91%</td><td align="right">38.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">12.40%</td><td align="right"><ins>31.10%</ins></td><td align="right"><ins>18.02%</ins></td><td align="right"><ins>21.12%</ins></td><td align="right"><strong>40.80%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">11.50%</td><td align="right">29.30%</td><td align="right">16.88%</td><td align="right">19.82%</td><td align="right">39.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">4.70%</td><td align="right">16.10%</td><td align="right">7.30%</td><td align="right">9.33%</td><td align="right">22.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">8.40%</td><td align="right">22.20%</td><td align="right">11.89%</td><td align="right">14.30%</td><td align="right">29.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">6.90%</td><td align="right">21.00%</td><td align="right">10.60%</td><td align="right">13.03%</td><td align="right">29.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">8.50%</td><td align="right">23.20%</td><td align="right">12.48%</td><td align="right">15.00%</td><td align="right">31.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">7.30%</td><td align="right">21.90%</td><td align="right">11.26%</td><td align="right">13.75%</td><td align="right">31.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">8.60%</td><td align="right">21.10%</td><td align="right">11.97%</td><td align="right">14.11%</td><td align="right">29.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">9.80%</td><td align="right">25.80%</td><td align="right">14.48%</td><td align="right">17.16%</td><td align="right">33.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">9.40%</td><td align="right">25.40%</td><td align="right">14.14%</td><td align="right">16.82%</td><td align="right">32.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">10.10%</td><td align="right">27.30%</td><td align="right">15.16%</td><td align="right">18.04%</td><td align="right">33.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">9.80%</td><td align="right">26.30%</td><td align="right">14.47%</td><td align="right">17.26%</td><td align="right">33.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">10.70%</td><td align="right">27.40%</td><td align="right">15.02%</td><td align="right">17.90%</td><td align="right">35.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">10.80%</td><td align="right">28.90%</td><td align="right">15.83%</td><td align="right">18.90%</td><td align="right">37.80%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">10.60%</td><td align="right">28.10%</td><td align="right">15.15%</td><td align="right">18.18%</td><td align="right">35.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">11.60%</td><td align="right">29.70%</td><td align="right">16.55%</td><td align="right">19.63%</td><td align="right">38.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">11.00%</td><td align="right">28.20%</td><td align="right">15.55%</td><td align="right">18.50%</td><td align="right">36.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">10.60%</td><td align="right">28.60%</td><td align="right">15.48%</td><td align="right">18.56%</td><td align="right">38.70%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">12.20%</td><td align="right">30.60%</td><td align="right">17.40%</td><td align="right">20.51%</td><td align="right">40.00%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">12.20%</td><td align="right">29.40%</td><td align="right">16.98%</td><td align="right">19.91%</td><td align="right">37.10%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>13.00%</strong></td><td align="right">30.80%</td><td align="right"><strong>18.21%</strong></td><td align="right"><strong>21.19%</strong></td><td align="right">39.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right"><strong>13.00%</strong></td><td align="right">29.40%</td><td align="right">17.56%</td><td align="right">20.36%</td><td align="right">38.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">9.20%</td><td align="right">24.80%</td><td align="right">13.66%</td><td align="right">16.29%</td><td align="right">33.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">10.60%</td><td align="right">28.20%</td><td align="right">15.66%</td><td align="right">18.63%</td><td align="right">35.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">10.30%</td><td align="right">27.20%</td><td align="right">15.03%</td><td align="right">17.90%</td><td align="right">34.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">11.50%</td><td align="right">28.30%</td><td align="right">16.37%</td><td align="right">19.21%</td><td align="right">36.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">11.30%</td><td align="right">27.40%</td><td align="right">15.71%</td><td align="right">18.47%</td><td align="right">36.40%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## ViRHE4QA_v2

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">55.60%</td><td align="right">92.00%</td><td align="right">67.70%</td><td align="right">73.60%</td><td align="right">95.90%</td></tr>
<tr><td align="left">bm25</td><td align="right">65.80%</td><td align="right">93.50%</td><td align="right">76.05%</td><td align="right">80.34%</td><td align="right">96.90%</td></tr>
<tr><td align="left">splade</td><td align="right">35.40%</td><td align="right">72.90%</td><td align="right">46.57%</td><td align="right">52.83%</td><td align="right">82.20%</td></tr>
<tr><td align="left">colbert</td><td align="right">42.90%</td><td align="right">73.40%</td><td align="right">52.49%</td><td align="right">57.50%</td><td align="right">81.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">54.40%</td><td align="right">87.80%</td><td align="right">65.52%</td><td align="right">70.91%</td><td align="right">93.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">61.10%</td><td align="right">93.80%</td><td align="right">72.66%</td><td align="right">77.81%</td><td align="right">96.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">59.10%</td><td align="right">91.80%</td><td align="right">70.59%</td><td align="right">75.77%</td><td align="right">95.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">65.50%</td><td align="right">94.50%</td><td align="right">75.77%</td><td align="right">80.33%</td><td align="right">97.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">62.70%</td><td align="right">93.50%</td><td align="right">73.71%</td><td align="right">78.54%</td><td align="right">97.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">31.90%</td><td align="right">68.60%</td><td align="right">42.71%</td><td align="right">48.87%</td><td align="right">77.30%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">55.90%</td><td align="right">90.80%</td><td align="right">67.73%</td><td align="right">73.34%</td><td align="right">94.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">46.70%</td><td align="right">84.10%</td><td align="right">58.92%</td><td align="right">64.99%</td><td align="right">91.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">64.40%</td><td align="right">92.20%</td><td align="right">74.30%</td><td align="right">78.69%</td><td align="right">95.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">51.90%</td><td align="right">86.20%</td><td align="right">62.79%</td><td align="right">68.41%</td><td align="right">92.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">50.80%</td><td align="right">82.70%</td><td align="right">61.73%</td><td align="right">66.84%</td><td align="right">89.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">60.90%</td><td align="right">93.10%</td><td align="right">72.27%</td><td align="right">77.35%</td><td align="right">96.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">57.40%</td><td align="right">91.80%</td><td align="right">69.41%</td><td align="right">74.86%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">66.20%</td><td align="right">93.70%</td><td align="right">75.89%</td><td align="right">80.23%</td><td align="right">97.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">62.90%</td><td align="right">93.60%</td><td align="right">73.27%</td><td align="right">78.20%</td><td align="right">97.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">55.20%</td><td align="right">88.50%</td><td align="right">66.37%</td><td align="right">71.73%</td><td align="right">92.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">62.60%</td><td align="right">92.80%</td><td align="right">73.34%</td><td align="right">78.11%</td><td align="right">96.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">59.40%</td><td align="right">91.80%</td><td align="right">70.81%</td><td align="right">75.93%</td><td align="right">96.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">65.80%</td><td align="right">94.40%</td><td align="right">76.20%</td><td align="right">80.67%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">63.30%</td><td align="right">94.20%</td><td align="right">74.34%</td><td align="right">79.20%</td><td align="right">96.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">58.50%</td><td align="right">91.90%</td><td align="right">69.85%</td><td align="right">75.21%</td><td align="right">95.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">65.40%</td><td align="right">94.50%</td><td align="right">75.96%</td><td align="right">80.52%</td><td align="right">97.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">61.80%</td><td align="right">93.50%</td><td align="right">73.21%</td><td align="right">78.18%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">69.60%</td><td align="right">95.50%</td><td align="right">79.19%</td><td align="right">83.21%</td><td align="right">97.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">66.70%</td><td align="right">94.90%</td><td align="right">77.06%</td><td align="right">81.45%</td><td align="right">97.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">49.50%</td><td align="right">86.10%</td><td align="right">61.08%</td><td align="right">67.09%</td><td align="right">91.30%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">59.60%</td><td align="right">92.30%</td><td align="right">70.88%</td><td align="right">76.10%</td><td align="right">96.50%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">57.00%</td><td align="right">90.50%</td><td align="right">68.39%</td><td align="right">73.76%</td><td align="right">96.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">64.10%</td><td align="right">93.70%</td><td align="right">74.08%</td><td align="right">78.83%</td><td align="right">97.00%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">60.40%</td><td align="right">93.40%</td><td align="right">71.73%</td><td align="right">77.00%</td><td align="right">96.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">52.60%</td><td align="right">88.80%</td><td align="right">64.94%</td><td align="right">70.74%</td><td align="right">93.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">62.90%</td><td align="right">94.00%</td><td align="right">73.97%</td><td align="right">78.88%</td><td align="right">97.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">60.10%</td><td align="right">93.80%</td><td align="right">71.95%</td><td align="right">77.28%</td><td align="right">97.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">66.70%</td><td align="right">95.40%</td><td align="right">76.74%</td><td align="right">81.30%</td><td align="right">97.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">66.40%</td><td align="right">94.40%</td><td align="right">76.09%</td><td align="right">80.56%</td><td align="right">98.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">55.40%</td><td align="right">87.80%</td><td align="right">66.28%</td><td align="right">71.48%</td><td align="right">93.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">60.80%</td><td align="right">93.00%</td><td align="right">72.12%</td><td align="right">77.22%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">58.70%</td><td align="right">92.00%</td><td align="right">70.37%</td><td align="right">75.64%</td><td align="right">96.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">65.90%</td><td align="right">93.90%</td><td align="right">75.93%</td><td align="right">80.33%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">63.90%</td><td align="right">93.40%</td><td align="right">74.44%</td><td align="right">79.08%</td><td align="right">97.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">59.20%</td><td align="right">91.10%</td><td align="right">70.40%</td><td align="right">75.45%</td><td align="right">95.10%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">65.90%</td><td align="right">95.30%</td><td align="right">76.55%</td><td align="right">81.15%</td><td align="right">97.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">61.80%</td><td align="right">94.60%</td><td align="right">73.72%</td><td align="right">78.85%</td><td align="right">97.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>71.10%</ins></td><td align="right">95.90%</td><td align="right"><ins>79.94%</ins></td><td align="right"><ins>83.86%</ins></td><td align="right">98.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">67.60%</td><td align="right">95.60%</td><td align="right">77.43%</td><td align="right">81.86%</td><td align="right">97.60%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">30.80%</td><td align="right">63.00%</td><td align="right">40.43%</td><td align="right">45.81%</td><td align="right">71.90%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">52.10%</td><td align="right">84.80%</td><td align="right">62.76%</td><td align="right">68.07%</td><td align="right">90.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">44.20%</td><td align="right">80.90%</td><td align="right">56.02%</td><td align="right">62.01%</td><td align="right">90.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">55.10%</td><td align="right">85.70%</td><td align="right">65.18%</td><td align="right">70.13%</td><td align="right">92.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">47.00%</td><td align="right">82.50%</td><td align="right">58.78%</td><td align="right">64.52%</td><td align="right">92.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">46.80%</td><td align="right">77.80%</td><td align="right">56.58%</td><td align="right">61.67%</td><td align="right">85.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">59.00%</td><td align="right">91.30%</td><td align="right">69.89%</td><td align="right">75.08%</td><td align="right">95.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">55.30%</td><td align="right">90.00%</td><td align="right">66.77%</td><td align="right">72.38%</td><td align="right">95.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">62.10%</td><td align="right">91.90%</td><td align="right">72.34%</td><td align="right">77.09%</td><td align="right">96.00%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">59.50%</td><td align="right">90.70%</td><td align="right">70.00%</td><td align="right">75.01%</td><td align="right">96.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">50.80%</td><td align="right">86.70%</td><td align="right">63.01%</td><td align="right">68.75%</td><td align="right">92.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">62.50%</td><td align="right">93.40%</td><td align="right">73.35%</td><td align="right">78.24%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">58.30%</td><td align="right">92.80%</td><td align="right">70.36%</td><td align="right">75.83%</td><td align="right">96.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">66.70%</td><td align="right">94.90%</td><td align="right">76.34%</td><td align="right">80.85%</td><td align="right">97.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">64.00%</td><td align="right">94.20%</td><td align="right">74.50%</td><td align="right">79.29%</td><td align="right">97.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">61.40%</td><td align="right">92.20%</td><td align="right">72.04%</td><td align="right">76.96%</td><td align="right">95.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">68.90%</td><td align="right">96.00%</td><td align="right">78.93%</td><td align="right">83.14%</td><td align="right">97.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">63.80%</td><td align="right">95.80%</td><td align="right">75.41%</td><td align="right">80.43%</td><td align="right">97.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>72.50%</strong></td><td align="right"><strong>96.90%</strong></td><td align="right"><strong>81.42%</strong></td><td align="right"><strong>85.23%</strong></td><td align="right"><strong>98.80%</strong></td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">68.60%</td><td align="right"><ins>96.20%</ins></td><td align="right">78.72%</td><td align="right">83.03%</td><td align="right"><ins>98.30%</ins></td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">60.00%</td><td align="right">91.50%</td><td align="right">70.41%</td><td align="right">75.50%</td><td align="right">95.30%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">63.40%</td><td align="right">93.90%</td><td align="right">74.50%</td><td align="right">79.26%</td><td align="right">97.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">60.80%</td><td align="right">92.90%</td><td align="right">72.12%</td><td align="right">77.20%</td><td align="right">96.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">68.20%</td><td align="right">95.70%</td><td align="right">78.02%</td><td align="right">82.35%</td><td align="right">97.70%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">65.80%</td><td align="right">94.90%</td><td align="right">76.31%</td><td align="right">80.87%</td><td align="right">97.50%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## VlogQA_2

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">13.40%</td><td align="right">34.60%</td><td align="right">19.55%</td><td align="right">23.10%</td><td align="right">47.00%</td></tr>
<tr><td align="left">bm25</td><td align="right">18.00%</td><td align="right">39.50%</td><td align="right">23.90%</td><td align="right">27.57%</td><td align="right">45.50%</td></tr>
<tr><td align="left">splade</td><td align="right">2.90%</td><td align="right">10.00%</td><td align="right">4.66%</td><td align="right">5.90%</td><td align="right">15.90%</td></tr>
<tr><td align="left">colbert</td><td align="right">5.30%</td><td align="right">15.40%</td><td align="right">8.01%</td><td align="right">9.74%</td><td align="right">19.40%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">20.20%</td><td align="right">46.70%</td><td align="right">28.12%</td><td align="right">32.53%</td><td align="right">55.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">27.50%</td><td align="right">56.70%</td><td align="right">36.01%</td><td align="right">40.91%</td><td align="right">66.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">24.40%</td><td align="right">56.00%</td><td align="right">33.65%</td><td align="right">38.94%</td><td align="right">67.80%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">28.40%</td><td align="right">57.20%</td><td align="right">36.82%</td><td align="right">41.65%</td><td align="right">65.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">27.10%</td><td align="right">56.40%</td><td align="right">35.81%</td><td align="right">40.72%</td><td align="right">67.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">3.60%</td><td align="right">11.20%</td><td align="right">5.40%</td><td align="right">6.75%</td><td align="right">16.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">13.80%</td><td align="right">34.50%</td><td align="right">19.62%</td><td align="right">23.12%</td><td align="right">44.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">10.70%</td><td align="right">28.50%</td><td align="right">15.22%</td><td align="right">18.31%</td><td align="right">38.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">15.20%</td><td align="right">31.70%</td><td align="right">19.83%</td><td align="right">22.64%</td><td align="right">40.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">10.30%</td><td align="right">26.80%</td><td align="right">14.79%</td><td align="right">17.61%</td><td align="right">38.60%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">18.80%</td><td align="right">41.20%</td><td align="right">25.26%</td><td align="right">29.03%</td><td align="right">49.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">24.00%</td><td align="right">53.40%</td><td align="right">32.99%</td><td align="right">37.85%</td><td align="right">62.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">22.70%</td><td align="right">51.80%</td><td align="right">31.12%</td><td align="right">36.00%</td><td align="right">63.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">25.80%</td><td align="right">52.20%</td><td align="right">33.64%</td><td align="right">38.05%</td><td align="right">62.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">24.30%</td><td align="right">52.10%</td><td align="right">32.66%</td><td align="right">37.28%</td><td align="right">65.60%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">21.10%</td><td align="right">47.40%</td><td align="right">28.72%</td><td align="right">33.14%</td><td align="right">55.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">26.60%</td><td align="right">57.10%</td><td align="right">36.16%</td><td align="right">41.19%</td><td align="right">67.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">25.10%</td><td align="right">56.70%</td><td align="right">34.22%</td><td align="right">39.55%</td><td align="right">69.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">28.20%</td><td align="right">57.90%</td><td align="right">37.03%</td><td align="right">41.98%</td><td align="right">67.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">28.40%</td><td align="right">59.00%</td><td align="right">36.74%</td><td align="right">41.98%</td><td align="right">68.50%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">23.90%</td><td align="right">49.20%</td><td align="right">31.09%</td><td align="right">35.38%</td><td align="right">57.00%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">29.00%</td><td align="right">59.90%</td><td align="right">37.85%</td><td align="right">43.06%</td><td align="right">67.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">26.80%</td><td align="right">58.40%</td><td align="right">35.63%</td><td align="right">41.00%</td><td align="right">69.00%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">29.00%</td><td align="right">57.90%</td><td align="right">37.79%</td><td align="right">42.56%</td><td align="right">66.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">28.40%</td><td align="right">56.90%</td><td align="right">36.69%</td><td align="right">41.48%</td><td align="right">67.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">24.00%</td><td align="right">51.90%</td><td align="right">32.03%</td><td align="right">36.74%</td><td align="right">59.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">28.20%</td><td align="right">60.60%</td><td align="right">37.73%</td><td align="right">43.16%</td><td align="right">68.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">27.20%</td><td align="right">60.50%</td><td align="right">36.62%</td><td align="right">42.24%</td><td align="right"><ins>69.80%</ins></td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">30.00%</td><td align="right"><ins>61.40%</ins></td><td align="right"><strong>39.53%</strong></td><td align="right"><strong>44.73%</strong></td><td align="right">68.30%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right"><strong>30.70%</strong></td><td align="right">60.60%</td><td align="right"><ins>39.41%</ins></td><td align="right"><ins>44.43%</ins></td><td align="right"><strong>69.90%</strong></td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">13.50%</td><td align="right">37.00%</td><td align="right">20.45%</td><td align="right">24.37%</td><td align="right">45.60%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">20.90%</td><td align="right">48.20%</td><td align="right">28.95%</td><td align="right">33.53%</td><td align="right">56.70%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">20.10%</td><td align="right">48.40%</td><td align="right">28.72%</td><td align="right">33.42%</td><td align="right">58.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">21.50%</td><td align="right">48.70%</td><td align="right">29.33%</td><td align="right">33.92%</td><td align="right">59.20%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">22.10%</td><td align="right">50.10%</td><td align="right">30.59%</td><td align="right">35.25%</td><td align="right">61.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">23.50%</td><td align="right">48.10%</td><td align="right">30.71%</td><td align="right">34.82%</td><td align="right">55.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">27.20%</td><td align="right">56.60%</td><td align="right">36.11%</td><td align="right">40.99%</td><td align="right">67.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">24.70%</td><td align="right">57.40%</td><td align="right">34.36%</td><td align="right">39.81%</td><td align="right">67.40%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">28.10%</td><td align="right">57.40%</td><td align="right">37.07%</td><td align="right">41.91%</td><td align="right">65.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">26.70%</td><td align="right">55.90%</td><td align="right">35.51%</td><td align="right">40.36%</td><td align="right">66.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">24.20%</td><td align="right">51.90%</td><td align="right">32.49%</td><td align="right">37.10%</td><td align="right">59.90%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">27.20%</td><td align="right">57.70%</td><td align="right">36.19%</td><td align="right">41.30%</td><td align="right">64.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">25.10%</td><td align="right">56.90%</td><td align="right">34.62%</td><td align="right">39.92%</td><td align="right">64.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><ins>30.50%</ins></td><td align="right">59.20%</td><td align="right">39.20%</td><td align="right">43.97%</td><td align="right">66.10%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">27.50%</td><td align="right">57.00%</td><td align="right">36.31%</td><td align="right">41.24%</td><td align="right">65.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">4.50%</td><td align="right">14.80%</td><td align="right">7.16%</td><td align="right">8.95%</td><td align="right">20.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">14.40%</td><td align="right">33.20%</td><td align="right">19.67%</td><td align="right">22.86%</td><td align="right">41.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">12.30%</td><td align="right">34.00%</td><td align="right">18.02%</td><td align="right">21.76%</td><td align="right">46.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">12.00%</td><td align="right">32.30%</td><td align="right">17.65%</td><td align="right">21.11%</td><td align="right">39.80%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">13.10%</td><td align="right">37.90%</td><td align="right">19.66%</td><td align="right">23.93%</td><td align="right">49.10%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">13.90%</td><td align="right">34.30%</td><td align="right">19.46%</td><td align="right">22.95%</td><td align="right">42.80%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">22.50%</td><td align="right">47.40%</td><td align="right">29.60%</td><td align="right">33.79%</td><td align="right">54.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">19.90%</td><td align="right">48.90%</td><td align="right">28.26%</td><td align="right">33.15%</td><td align="right">58.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">21.40%</td><td align="right">46.70%</td><td align="right">28.71%</td><td align="right">32.98%</td><td align="right">54.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">21.10%</td><td align="right">49.40%</td><td align="right">29.60%</td><td align="right">34.32%</td><td align="right">60.30%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">22.70%</td><td align="right">46.90%</td><td align="right">29.62%</td><td align="right">33.71%</td><td align="right">56.20%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">25.50%</td><td align="right">54.90%</td><td align="right">34.33%</td><td align="right">39.24%</td><td align="right">65.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">22.80%</td><td align="right">55.80%</td><td align="right">32.32%</td><td align="right">37.88%</td><td align="right">65.30%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">28.80%</td><td align="right">57.50%</td><td align="right">36.96%</td><td align="right">41.81%</td><td align="right">66.90%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">25.60%</td><td align="right">57.20%</td><td align="right">35.03%</td><td align="right">40.32%</td><td align="right">66.00%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right">22.30%</td><td align="right">49.00%</td><td align="right">29.92%</td><td align="right">34.44%</td><td align="right">57.50%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">25.80%</td><td align="right">55.70%</td><td align="right">35.04%</td><td align="right">39.98%</td><td align="right">63.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">24.10%</td><td align="right">55.20%</td><td align="right">33.67%</td><td align="right">38.82%</td><td align="right">63.20%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">29.20%</td><td align="right">57.80%</td><td align="right">37.80%</td><td align="right">42.55%</td><td align="right">65.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">27.10%</td><td align="right">56.40%</td><td align="right">35.80%</td><td align="right">40.69%</td><td align="right">65.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">20.70%</td><td align="right">49.60%</td><td align="right">28.91%</td><td align="right">33.81%</td><td align="right">59.30%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">26.70%</td><td align="right">59.00%</td><td align="right">36.63%</td><td align="right">41.96%</td><td align="right">67.30%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">25.20%</td><td align="right">58.90%</td><td align="right">35.24%</td><td align="right">40.86%</td><td align="right">67.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">29.40%</td><td align="right"><strong>61.80%</strong></td><td align="right">38.91%</td><td align="right">44.34%</td><td align="right">69.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">27.60%</td><td align="right">58.30%</td><td align="right">36.75%</td><td align="right">41.88%</td><td align="right">68.80%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)

## ZaloLegalQA

<details>
<summary><strong>View all retrieval configurations</strong></summary>

<table>
<thead><tr><th align="left">Method</th><th align="right">P@1</th><th align="right">R@10</th><th align="right">MRR@10</th><th align="right">nDCG@10</th><th align="right">R@20</th></tr></thead>
<tbody>
<tr><td align="left">tfidf</td><td align="right">64.70%</td><td align="right">92.47%</td><td align="right">75.07%</td><td align="right">79.28%</td><td align="right">96.45%</td></tr>
<tr><td align="left">bm25</td><td align="right">71.40%</td><td align="right">92.18%</td><td align="right">79.39%</td><td align="right">82.49%</td><td align="right">94.42%</td></tr>
<tr><td align="left">splade</td><td align="right">40.30%</td><td align="right">65.53%</td><td align="right">48.42%</td><td align="right">52.35%</td><td align="right">72.58%</td></tr>
<tr><td align="left">colbert</td><td align="right">54.40%</td><td align="right">75.98%</td><td align="right">61.78%</td><td align="right">65.07%</td><td align="right">80.73%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-multilingual-base</th></tr>
<tr><td align="left">dense</td><td align="right">79.00%</td><td align="right">96.22%</td><td align="right">85.56%</td><td align="right">88.13%</td><td align="right">98.32%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.10%</td><td align="right">97.82%</td><td align="right">86.12%</td><td align="right">88.94%</td><td align="right">98.40%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">74.20%</td><td align="right">97.05%</td><td align="right">82.97%</td><td align="right">86.40%</td><td align="right">98.25%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">80.90%</td><td align="right">97.17%</td><td align="right">87.09%</td><td align="right">89.52%</td><td align="right">97.97%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">76.60%</td><td align="right">95.27%</td><td align="right">83.68%</td><td align="right">86.48%</td><td align="right">97.07%</td></tr>
<tr><th colspan="6" align="left">Dense model: Alibaba-NLP_gte-Qwen2-1_5B-instruct</th></tr>
<tr><td align="left">dense</td><td align="right">84.10%</td><td align="right">98.23%</td><td align="right">89.76%</td><td align="right">91.77%</td><td align="right">99.05%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">82.50%</td><td align="right">98.60%</td><td align="right">88.79%</td><td align="right">91.19%</td><td align="right">99.05%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">75.90%</td><td align="right">97.25%</td><td align="right">84.42%</td><td align="right">87.58%</td><td align="right">98.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">86.00%</td><td align="right">98.47%</td><td align="right">90.99%</td><td align="right"><ins>92.76%</ins></td><td align="right">98.95%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">78.20%</td><td align="right">95.67%</td><td align="right">85.03%</td><td align="right">87.62%</td><td align="right">97.90%</td></tr>
<tr><th colspan="6" align="left">Dense model: BAAI_bge-multilingual-gemma2</th></tr>
<tr><td align="left">dense</td><td align="right">75.60%</td><td align="right">95.68%</td><td align="right">83.21%</td><td align="right">86.18%</td><td align="right">97.15%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.10%</td><td align="right">97.75%</td><td align="right">86.53%</td><td align="right">89.26%</td><td align="right">98.90%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">75.60%</td><td align="right">96.50%</td><td align="right">83.77%</td><td align="right">86.87%</td><td align="right">98.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">82.60%</td><td align="right">97.72%</td><td align="right">88.60%</td><td align="right">90.78%</td><td align="right">98.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">76.50%</td><td align="right">96.10%</td><td align="right">84.47%</td><td align="right">87.28%</td><td align="right">97.85%</td></tr>
<tr><th colspan="6" align="left">Dense model: google_embeddinggemma-300m</th></tr>
<tr><td align="left">dense</td><td align="right">82.50%</td><td align="right">96.67%</td><td align="right">88.11%</td><td align="right">90.12%</td><td align="right">97.87%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">80.60%</td><td align="right">97.87%</td><td align="right">87.42%</td><td align="right">89.94%</td><td align="right">98.57%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.00%</td><td align="right">96.72%</td><td align="right">84.19%</td><td align="right">87.23%</td><td align="right">98.27%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">83.00%</td><td align="right">97.37%</td><td align="right">88.65%</td><td align="right">90.71%</td><td align="right">98.22%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">77.90%</td><td align="right">94.97%</td><td align="right">84.39%</td><td align="right">86.94%</td><td align="right">96.97%</td></tr>
<tr><th colspan="6" align="left">Dense model: intfloat_multilingual-e5-large</th></tr>
<tr><td align="left">dense</td><td align="right">83.80%</td><td align="right">97.98%</td><td align="right">89.69%</td><td align="right">91.64%</td><td align="right">98.88%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">83.80%</td><td align="right">98.48%</td><td align="right">89.56%</td><td align="right">91.71%</td><td align="right">98.77%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.40%</td><td align="right">97.58%</td><td align="right">84.82%</td><td align="right">87.92%</td><td align="right">98.42%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">84.80%</td><td align="right">98.08%</td><td align="right">90.15%</td><td align="right">92.05%</td><td align="right">98.53%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">78.20%</td><td align="right">95.33%</td><td align="right">84.86%</td><td align="right">87.41%</td><td align="right">97.57%</td></tr>
<tr><th colspan="6" align="left">Dense model: jinaai_jina-embeddings-v3</th></tr>
<tr><td align="left">dense</td><td align="right">83.10%</td><td align="right"><ins>98.85%</ins></td><td align="right">89.39%</td><td align="right">91.67%</td><td align="right"><strong>99.40%</strong></td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">81.40%</td><td align="right"><strong>98.90%</strong></td><td align="right">88.36%</td><td align="right">90.94%</td><td align="right">99.25%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.40%</td><td align="right">97.75%</td><td align="right">84.77%</td><td align="right">87.95%</td><td align="right">98.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">85.00%</td><td align="right">98.55%</td><td align="right">90.43%</td><td align="right">92.40%</td><td align="right">99.15%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">78.80%</td><td align="right">96.17%</td><td align="right">85.27%</td><td align="right">87.92%</td><td align="right">97.65%</td></tr>
<tr><th colspan="6" align="left">Dense model: openai-text-embedding-3-large</th></tr>
<tr><td align="left">dense</td><td align="right">80.50%</td><td align="right">96.83%</td><td align="right">87.10%</td><td align="right">89.39%</td><td align="right">98.53%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">82.20%</td><td align="right">98.07%</td><td align="right">88.52%</td><td align="right">90.80%</td><td align="right"><ins>99.27%</ins></td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.10%</td><td align="right">97.72%</td><td align="right">85.08%</td><td align="right">88.15%</td><td align="right">98.90%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">84.00%</td><td align="right">97.93%</td><td align="right">89.83%</td><td align="right">91.77%</td><td align="right">98.92%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">78.10%</td><td align="right">96.28%</td><td align="right">85.50%</td><td align="right">88.13%</td><td align="right">98.12%</td></tr>
<tr><th colspan="6" align="left">Dense model: Qwen_Qwen3-Embedding-0_6B</th></tr>
<tr><td align="left">dense</td><td align="right">80.80%</td><td align="right">97.68%</td><td align="right">87.48%</td><td align="right">89.88%</td><td align="right">98.40%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">80.10%</td><td align="right">98.20%</td><td align="right">87.15%</td><td align="right">89.82%</td><td align="right">98.65%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">77.10%</td><td align="right">97.25%</td><td align="right">84.84%</td><td align="right">87.86%</td><td align="right">98.35%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">83.20%</td><td align="right">97.63%</td><td align="right">88.90%</td><td align="right">90.96%</td><td align="right">98.45%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">77.60%</td><td align="right">95.27%</td><td align="right">84.47%</td><td align="right">87.09%</td><td align="right">97.45%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-bge-m3</th></tr>
<tr><td align="left">dense</td><td align="right">82.30%</td><td align="right">97.83%</td><td align="right">88.52%</td><td align="right">90.75%</td><td align="right">98.77%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">80.50%</td><td align="right">98.12%</td><td align="right">87.69%</td><td align="right">90.26%</td><td align="right">99.05%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.00%</td><td align="right">97.22%</td><td align="right">84.43%</td><td align="right">87.58%</td><td align="right">98.70%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">82.10%</td><td align="right">98.02%</td><td align="right">88.62%</td><td align="right">90.93%</td><td align="right">98.62%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">79.20%</td><td align="right">96.17%</td><td align="right">86.05%</td><td align="right">88.52%</td><td align="right">97.67%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2</th></tr>
<tr><td align="left">dense</td><td align="right">51.10%</td><td align="right">80.23%</td><td align="right">60.75%</td><td align="right">65.30%</td><td align="right">85.87%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">69.70%</td><td align="right">93.92%</td><td align="right">78.50%</td><td align="right">82.18%</td><td align="right">96.60%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">64.10%</td><td align="right">91.60%</td><td align="right">73.74%</td><td align="right">78.00%</td><td align="right">96.25%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">70.60%</td><td align="right">94.32%</td><td align="right">79.43%</td><td align="right">82.98%</td><td align="right">95.95%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">64.60%</td><td align="right">91.62%</td><td align="right">74.46%</td><td align="right">78.56%</td><td align="right">96.70%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-bi-encoder</th></tr>
<tr><td align="left">dense</td><td align="right">71.00%</td><td align="right">92.72%</td><td align="right">79.20%</td><td align="right">82.46%</td><td align="right">94.72%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">77.40%</td><td align="right">96.72%</td><td align="right">84.88%</td><td align="right">87.76%</td><td align="right">98.20%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">72.90%</td><td align="right">95.52%</td><td align="right">81.48%</td><td align="right">84.88%</td><td align="right">97.85%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">80.30%</td><td align="right">96.67%</td><td align="right">86.67%</td><td align="right">89.08%</td><td align="right">97.97%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">75.30%</td><td align="right">95.62%</td><td align="right">82.96%</td><td align="right">85.99%</td><td align="right">97.20%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-vietnamese-document-embedding</th></tr>
<tr><td align="left">dense</td><td align="right">77.70%</td><td align="right">96.27%</td><td align="right">85.08%</td><td align="right">87.81%</td><td align="right">97.95%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">79.80%</td><td align="right">98.35%</td><td align="right">86.80%</td><td align="right">89.60%</td><td align="right">99.00%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.20%</td><td align="right">97.10%</td><td align="right">84.15%</td><td align="right">87.31%</td><td align="right">98.50%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">82.40%</td><td align="right">97.42%</td><td align="right">88.33%</td><td align="right">90.54%</td><td align="right">98.60%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">77.60%</td><td align="right">96.30%</td><td align="right">84.92%</td><td align="right">87.68%</td><td align="right">97.80%</td></tr>
<tr><th colspan="6" align="left">Dense model: sbert-Vietnamese_Embedding_V2</th></tr>
<tr><td align="left">dense</td><td align="right"><ins>86.20%</ins></td><td align="right">98.32%</td><td align="right"><ins>91.04%</ins></td><td align="right">92.74%</td><td align="right">98.97%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">85.40%</td><td align="right">98.42%</td><td align="right">90.91%</td><td align="right">92.71%</td><td align="right">99.10%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">79.10%</td><td align="right">97.65%</td><td align="right">86.65%</td><td align="right">89.35%</td><td align="right">98.60%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right"><strong>87.40%</strong></td><td align="right">98.17%</td><td align="right"><strong>91.67%</strong></td><td align="right"><strong>93.18%</strong></td><td align="right">98.67%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">81.00%</td><td align="right">96.27%</td><td align="right">87.12%</td><td align="right">89.31%</td><td align="right">97.75%</td></tr>
<tr><th colspan="6" align="left">Dense model: Snowflake_snowflake-arctic-embed-l-v2_0</th></tr>
<tr><td align="left">dense</td><td align="right">83.20%</td><td align="right">98.03%</td><td align="right">89.30%</td><td align="right">91.42%</td><td align="right">98.72%</td></tr>
<tr><td align="left">dense + tfidf (α)</td><td align="right">82.00%</td><td align="right">98.12%</td><td align="right">88.28%</td><td align="right">90.68%</td><td align="right">98.47%</td></tr>
<tr><td align="left">dense + tfidf (RRF)</td><td align="right">76.80%</td><td align="right">97.02%</td><td align="right">84.59%</td><td align="right">87.63%</td><td align="right">98.42%</td></tr>
<tr><td align="left">dense + bm25 (α)</td><td align="right">84.90%</td><td align="right">98.13%</td><td align="right">90.07%</td><td align="right">91.99%</td><td align="right">98.42%</td></tr>
<tr><td align="left">dense + bm25 (RRF)</td><td align="right">77.70%</td><td align="right">95.33%</td><td align="right">84.63%</td><td align="right">87.23%</td><td align="right">97.32%</td></tr>
</tbody>
</table>

</details>

[↑ Back to datasets](#datasets)
