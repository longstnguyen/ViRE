from typing import List
import numpy as np


def minmax_rowwise(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Apply min-max normalization independently per row.

    Args:
        x: Input score matrix.
        eps: Numerical floor for denominator stability.

    Returns:
        np.ndarray: Row-wise normalized matrix in [0, 1].
    """
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    denom = np.maximum(xmax - xmin, eps)
    return (x - xmin) / denom


def ranks_from_scores(scores_row: np.ndarray) -> np.ndarray:
    """Convert one score row into 1-based ranks.

    Args:
        scores_row: Scores for one query across documents.

    Returns:
        np.ndarray: Rank array where 1 is the best rank.
    """
    order = np.argsort(-scores_row, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return ranks


def rrf_fuse_ranks(ranks_list: List[np.ndarray], k: int = 60) -> np.ndarray:
    """Fuse rank arrays using Reciprocal Rank Fusion (RRF).

    Args:
        ranks_list: List of rank arrays to combine.
        k: RRF constant.

    Returns:
        np.ndarray: Fused RRF score array.
    """
    fused = np.zeros_like(ranks_list[0], dtype=np.float32)
    for r in ranks_list:
        fused += 1.0 / (k + r.astype(np.float32))
    return fused
