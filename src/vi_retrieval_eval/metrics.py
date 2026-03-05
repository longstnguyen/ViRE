# src/vi_retrieval_eval/metrics.py
from typing import Dict, List, Iterable
import numpy as np
from .progress import iter_progress


def _argsort_desc(row: np.ndarray) -> np.ndarray:
    # stable sort for consistent tie-breaking
    return np.argsort(-row, kind="mergesort")


def _dcg_at_k(gains: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    g = gains[:k]
    # binary relevance: DCG = sum(1/log2(rank+1)), rank starts at 1
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(g)))
    return float((g * discounts).sum())


def _ideal_dcg_at_k(num_rels: int, k: int) -> float:
    if num_rels <= 0 or k <= 0:
        return 0.0
    ones = np.ones(min(num_rels, k), dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(ones)))
    return float((ones * discounts).sum())


def _ap_at_k(gains: np.ndarray, k: int, R: int) -> float:
    """Average Precision @k (chia cho min(R, k))."""
    if R <= 0 or k <= 0:
        return 0.0
    denom = float(min(R, k))
    running_hits = 0.0
    ap_sum = 0.0
    upto = min(k, len(gains))
    for i in range(upto):
        if gains[i] > 0:
            running_hits += 1.0
            ap_sum += running_hits / float(i + 1)
    return float(ap_sum / denom)


def _first_rel_rank(gains: np.ndarray) -> float:
    hits = np.where(gains > 0)[0]
    return float(hits[0] + 1) if hits.size > 0 else float("inf")


def evaluate_all(
    scores: np.ndarray,
    gold_lists: List[List[int]],
    ks: Iterable[int] = (1, 3, 5, 10, 20, 50, 100),
    show_progress: bool = False,
) -> Dict[str, float]:
    """
    scores: (Q, N) score matrix, higher is better.
    gold_lists: list of relevant doc indices per query (multi-gold OK).
                Indices may be 0-based or 1-based; handled automatically.
    ks: metric cutoffs.
    show_progress: display per-query progress.

    Returns a dictionary of metrics averaged across all queries.
    """
    Q, N = scores.shape
    ks = sorted({int(k) for k in ks if int(k) > 0})

    prec_at = {k: [] for k in ks}
    rec_at = {k: [] for k in ks}
    mrr_at = {k: [] for k in ks}
    hit_at = {k: [] for k in ks}
    ndcg_at = {k: [] for k in ks}
    map_at = {k: [] for k in ks}
    rprec_list: List[float] = []  # R-precision
    first_ranks: List[float] = []

    it_q = iter_progress(
        range(Q), enable=show_progress, tqdm_desc="Evaluating", total=Q
    )
    for qi in it_q:
        order = _argsort_desc(scores[qi])

        # gold may contain 0-based, 1-based, or out-of-bounds indices
        # (mismatch between corpus IDs and positions in scores).
        raw_gold = gold_lists[qi]

        # unique + cast sang numpy
        gold = (
            np.fromiter(set(raw_gold), dtype=int)
            if raw_gold
            else np.empty(0, dtype=int)
        )

        # Auto-detect 1-based: if max == N and min >= 1 (common case)
        if gold.size > 0:
            max_idx = int(gold.max())
            min_idx = int(gold.min())
            if max_idx == N and min_idx >= 1:
                gold = gold - 1  # convert to 0-based

        # Keep only valid indices in [0, N-1]
        if gold.size > 0:
            valid_mask = (gold >= 0) & (gold < N)
            gold = gold[valid_mask]
        # R is the number of relevant docs actually within the N-doc pool
        R = int(gold.size)

        # binary relevance gains
        gains = np.zeros(N, dtype=np.float32)
        if R > 0:
            gains[gold] = 1.0

        # reorder by rank (scores descending)
        gains = gains[order]

        # rank of the first relevant element
        fr = _first_rel_rank(gains)
        first_ranks.append(fr)

        # R-precision = Precision@R (returns 0 if R=0)
        if R > 0:
            topR = int(min(R, N))
            rprec = float(gains[:topR].sum()) / float(topR)
        else:
            rprec = 0.0
        rprec_list.append(rprec)

        for k in ks:
            kk = min(k, N)
            topk_hits = float(gains[:kk].sum())
            hit = 1.0 if topk_hits > 0 else 0.0
            prec = topk_hits / float(kk)
            rec = (topk_hits / float(R)) if R > 0 else 0.0

            # MRR@k based on rank of the first relevant element within top-k
            mrrk = (1.0 / fr) if np.isfinite(fr) and fr <= kk else 0.0

            # nDCG@k
            dcg = _dcg_at_k(gains, kk)
            idcg = _ideal_dcg_at_k(R, kk)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            # MAP@k
            apk = _ap_at_k(gains, kk, R)

            hit_at[k].append(hit)
            prec_at[k].append(prec)
            rec_at[k].append(rec)
            mrr_at[k].append(mrrk)
            ndcg_at[k].append(ndcg)
            map_at[k].append(apk)

    out: Dict[str, float] = {}
    for k in ks:
        out[f"HitRate@{k}"] = float(np.mean(hit_at[k])) if hit_at[k] else 0.0
        out[f"Precision@{k}"] = float(np.mean(prec_at[k])) if prec_at[k] else 0.0
        out[f"Recall@{k}"] = float(np.mean(rec_at[k])) if rec_at[k] else 0.0
        out[f"MRR@{k}"] = float(np.mean(mrr_at[k])) if mrr_at[k] else 0.0
        out[f"nDCG@{k}"] = float(np.mean(ndcg_at[k])) if ndcg_at[k] else 0.0
        out[f"MAP@{k}"] = float(np.mean(map_at[k])) if map_at[k] else 0.0

    # R-Precision (mean theo query)
    out["R-Precision"] = float(np.mean(rprec_list)) if rprec_list else 0.0

    # first-relevant-rank statistics
    finite = [r for r in first_ranks if np.isfinite(r)]
    out["MeanFirstRelRank"] = float(np.mean(finite)) if finite else float("inf")
    out["MedianFirstRelRank"] = float(np.median(finite)) if finite else float("inf")
    out["FirstRelFoundRate"] = (
        float(len(finite) / len(first_ranks)) if first_ranks else 0.0
    )

    return out
