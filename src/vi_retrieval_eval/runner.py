from typing import List, Dict, Optional, Any
import os
import csv
import numpy as np

from .io_utils import save_json
from .lexical import build_tfidf, tfidf_scores, build_bm25, bm25_scores
from .fusion import minmax_rowwise, ranks_from_scores, rrf_fuse_ranks
from .qrels import rank_of_first_gold
from .dense_index import DenseFAISS
from .metrics import evaluate_all

# Trigger backend registry via side-effect imports
from .embeddings import *  # noqa: F401
from .embeddings.base import get_embedder

# optional logging
try:
    from .logging_utils import setup_logger
except Exception:

    def setup_logger(level: str = "info"):
        """Return a no-op logger when logging utilities are unavailable.

        Args:
            level: Requested log level (ignored by the fallback logger).

        Returns:
            _Dummy: Logger-compatible object with no-op logging methods.
        """

        class _Dummy:
            """Minimal logger-compatible object with no-op methods."""

            def info(self, *a, **k):
                """Ignore info-level log calls."""
                pass

            def debug(self, *a, **k):
                """Ignore debug-level log calls."""
                pass

            def warning(self, *a, **k):
                """Ignore warning-level log calls."""
                pass

            def error(self, *a, **k):
                """Ignore error-level log calls."""
                pass

        return _Dummy()


def safe_model_name(name: str) -> str:
    """Sanitize model names for filesystem-safe paths.

    Args:
        name: Raw model identifier.

    Returns:
        str: Sanitized model identifier safe for directories/files.
    """
    return name.replace("/", "__").replace(":", "_").replace(" ", "_")


# =====================================================================
# Helpers
# =====================================================================


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save_csv(path: str, rows: List[Dict[str, Any]], field_order: List[str]):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            rr = {}
            for k, v in r.items():
                if isinstance(v, (list, tuple, dict)):
                    import json as _json

                    rr[k] = _json.dumps(v, ensure_ascii=False)
                else:
                    rr[k] = v
            w.writerow(rr)


def _topk_indices_rowwise(score_mat: np.ndarray, k: int) -> np.ndarray:
    Q, N = score_mat.shape
    k = min(max(k, 1), N)
    part = np.argpartition(-score_mat, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(Q)[:, None]
    ord_in_k = np.argsort(-score_mat[row_idx, part], axis=1)
    return part[row_idx, ord_in_k]


def _normalize_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip()
    return x or None


def _validate_method_and_models(
    method: str,
    dense_backend: Optional[str],
    dense_model: Optional[str],
    splade_model: Optional[str],
    colbert_model: Optional[str],
):
    """
    Enforce method/model constraints:

    - PURE:
        tfidf, bm25
        dense           → dense_backend + dense_model
        splade          → splade_model
        colbert         → colbert_model

    - HYBRID:
        dense+tfidf     → dense_backend + dense_model
        dense+bm25      → dense_backend + dense_model
        splade+tfidf    → splade_model
        splade+bm25     → splade_model
        splade+dense    → splade_model + dense_backend + dense_model
    """
    dense_backend = _normalize_str(dense_backend)
    dense_model = _normalize_str(dense_model)
    splade_model = _normalize_str(splade_model)
    colbert_model = _normalize_str(colbert_model)

    dense_backends_valid = {"sbert", "openai", "gemini", "llm"}

    if method in ("tfidf", "bm25"):
        # No model required
        return

    if method == "dense":
        if dense_backend not in dense_backends_valid:
            raise ValueError(
                f"[dense] require --dense-backend in {dense_backends_valid}, got: {dense_backend}"
            )
        if not dense_model:
            raise ValueError("[dense] require --dense-model.")
        return

    # ✅ Legacy hybrid dense: dense+tfidf / dense+bm25
    if method in ("dense+tfidf", "dense+bm25"):
        if dense_backend not in dense_backends_valid:
            raise ValueError(
                f"[{method}] require --dense-backend in {dense_backends_valid}, got: {dense_backend}"
            )
        if not dense_model:
            raise ValueError(f"[{method}] require --dense-model.")
        return

    if method == "splade":
        if not splade_model:
            raise ValueError("[splade] require --splade-model.")
        return

    if method == "colbert":
        if not colbert_model:
            raise ValueError("[colbert] require --colbert-model.")
        # dense_backend / dense_model / splade_model are ignored for colbert
        return

    if method in ("splade+tfidf", "splade+bm25"):
        if not splade_model:
            raise ValueError(f"[{method}] require --splade-model.")
        return

    if method == "splade+dense":
        if not splade_model:
            raise ValueError("[splade+dense] require --splade-model.")
        if dense_backend not in dense_backends_valid:
            raise ValueError(
                f"[splade+dense] require --dense-backend in {dense_backends_valid}, got: {dense_backend}"
            )
        if not dense_model:
            raise ValueError("[splade+dense] require --dense-model.")
        return

    raise ValueError(f"Unknown method: {method}")


# =====================================================================
# MAIN RUN
# =====================================================================


def run(
    method: str,
    fusion: str,
    questions: List[str],
    contexts: List[str],
    gold_lists: List[List[int]],
    out_dir: str,
    *,
    dense_backend: Optional[str],
    dense_model: Optional[str],
    splade_model: Optional[str],
    colbert_model: Optional[str],
    batch_size: int,
    max_len: Optional[int] = None,
    index_metric: str,
    alpha: float,
    rrf_k: int,
    force: bool,
    ks: List[int],
    show_progress: bool = False,
    log_level: str = "info",
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    qids: Optional[List[str]] = None,
    error_k: Optional[int] = None,
) -> Dict[str, float]:
    """Execute one retrieval evaluation run and export artifacts.

    Args:
        method: Retrieval method name.
        fusion: Fusion strategy for hybrid methods.
        questions: Query texts.
        contexts: Candidate document/passages.
        gold_lists: Gold document indices per query.
        out_dir: Output directory for run artifacts.
        dense_backend: Dense backend name, if applicable.
        dense_model: Dense model name, if applicable.
        splade_model: SPLADE model name, if applicable.
        colbert_model: ColBERT model name, if applicable.
        batch_size: Embedding batch size hint.
        max_len: Optional token-length truncation for embedding backends.
        index_metric: Vector index metric (`ip` or `l2`).
        alpha: Alpha for score-based fusion.
        rrf_k: Constant for Reciprocal Rank Fusion.
        force: Whether to force rebuilding caches/indexes.
        ks: Evaluation cutoffs.
        show_progress: Whether to display progress bars.
        log_level: Logging level.
        bm25_k1: BM25 k1 parameter.
        bm25_b: BM25 b parameter.
        qids: Optional query IDs.
        error_k: Optional cutoff used for fail@K artifacts.

    Returns:
        Dict[str, float]: Evaluation metrics for the run.
    """

    # Normalize inputs then validate method/model constraints
    dense_backend = _normalize_str(dense_backend)
    dense_model = _normalize_str(dense_model)
    splade_model = _normalize_str(splade_model)
    colbert_model = _normalize_str(colbert_model)

    _validate_method_and_models(
        method=method,
        dense_backend=dense_backend,
        dense_model=dense_model,
        splade_model=splade_model,
        colbert_model=colbert_model,
    )

    logger = setup_logger(log_level)
    os.makedirs(out_dir, exist_ok=True)
    ranks_path = os.path.join(out_dir, "ranks.json")
    metrics_path = os.path.join(out_dir, "metrics.json")

    logger.info(f"Method={method} | Fusion={fusion} | OutDir={out_dir}")
    logger.debug(f"Questions={len(questions)} | Contexts={len(contexts)} | Ks={ks}")

    scores: Optional[np.ndarray] = None

    # =====================================================================
    # PURE LEXICAL: TF-IDF / BM25
    # =====================================================================
    if method == "tfidf":
        logger.info("Building TF-IDF...")
        vect, X_docs = build_tfidf(contexts)
        logger.info("Scoring TF-IDF...")
        scores = tfidf_scores(vect, X_docs, questions)

    elif method == "bm25":
        logger.info(f"Building BM25 (k1={bm25_k1}, b={bm25_b})...")
        bm25 = build_bm25(contexts, k1=bm25_k1, b=bm25_b)
        logger.info("Scoring BM25...")
        scores = bm25_scores(bm25, questions, show_progress=show_progress)

    # =====================================================================
    # COLBERT: late-interaction, does not use FAISS / dense backend
    # =====================================================================
    elif method == "colbert":
        assert colbert_model is not None
        # Store ColBERT index under the dataset name
        dataset_name = os.path.normpath(out_dir).split(os.sep)[-2]
        colbert_root = os.path.join("cache", "colbert_indexes", dataset_name)
        _ensure_dir(colbert_root)

        logger.info(f"[ColBERT] Using root directory: {colbert_root}")

        embedder = get_embedder(
            "colbert",
            model_name=colbert_model,
            root=colbert_root,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        logger.info("[ColBERT] Building index over contexts...")
        embedder.build_docs(contexts)

        logger.info("[ColBERT] Scoring queries via MaxSim...")
        scores = embedder.score(questions)

    # =====================================================================
    # SPLADE & HYBRID SPLADE
    # =====================================================================
    elif method.startswith("splade"):
        assert splade_model is not None

        logger.info(f"Preparing SPLADE backend with model: {splade_model}")
        splade_backend = get_embedder(
            "splade",
            model_name=splade_model,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        logger.info("[SPLADE] Scoring SPLADE similarities...")
        splade_scores = splade_backend.score(questions, contexts)  # (Q, N)

        # ---- PURE SPLADE ----
        if method == "splade":
            scores = splade_scores

        # ---- SPLADE + TF-IDF ----
        elif method == "splade+tfidf":
            logger.info("Building TF-IDF for hybrid SPLADE+TFIDF...")
            vect, X_docs = build_tfidf(contexts)
            logger.info("Scoring TF-IDF...")
            t_scores = tfidf_scores(vect, X_docs, questions)

            if fusion == "alpha":
                scores = alpha * minmax_rowwise(splade_scores) + (
                    1 - alpha
                ) * minmax_rowwise(t_scores)
            elif fusion == "rrf":
                Q, N = len(questions), len(contexts)
                ranks_splade = np.zeros((Q, N), dtype=np.int32)
                ranks_tfidf = np.zeros((Q, N), dtype=np.int32)
                for i in range(Q):
                    ranks_splade[i] = ranks_from_scores(splade_scores[i])
                    ranks_tfidf[i] = ranks_from_scores(t_scores[i])
                scores = rrf_fuse_ranks([ranks_splade, ranks_tfidf], k=rrf_k)
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion}")

        # ---- SPLADE + BM25 ----
        elif method == "splade+bm25":
            logger.info(
                f"Building BM25 (k1={bm25_k1}, b={bm25_b}) for hybrid SPLADE+BM25..."
            )
            bm25 = build_bm25(contexts, k1=bm25_k1, b=bm25_b)
            logger.info("Scoring BM25...")
            b_scores = bm25_scores(bm25, questions, show_progress=show_progress)

            if fusion == "alpha":
                scores = alpha * minmax_rowwise(splade_scores) + (
                    1 - alpha
                ) * minmax_rowwise(b_scores)
            elif fusion == "rrf":
                Q, N = len(questions), len(contexts)
                ranks_splade = np.zeros((Q, N), dtype=np.int32)
                ranks_bm25 = np.zeros((Q, N), dtype=np.int32)
                for i in range(Q):
                    ranks_splade[i] = ranks_from_scores(splade_scores[i])
                    ranks_bm25[i] = ranks_from_scores(b_scores[i])
                scores = rrf_fuse_ranks([ranks_splade, ranks_bm25], k=rrf_k)
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion}")

        # ---- SPLADE + DENSE ----
        elif method == "splade+dense":
            assert dense_backend is not None and dense_model is not None
            logger.info(
                f"Preparing dense backend for SPLADE+DENSE: {dense_backend} | model={dense_model}"
            )

            # load dense embedder
            if dense_backend == "openai":
                dense_embedder = get_embedder(
                    "openai",
                    model=dense_model,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
                dense_model_name = dense_model

            elif dense_backend == "gemini":
                dense_embedder = get_embedder(
                    "gemini",
                    model=dense_model,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
                dense_model_name = dense_model

            elif dense_backend in ("sbert", "llm"):
                dense_embedder = get_embedder(
                    dense_backend,
                    model_name=dense_model,
                    batch_size=batch_size,
                    show_progress=show_progress,
                    max_len=max_len,
                )
                dense_model_name = dense_model

            else:
                raise ValueError(
                    f"Unknown dense_backend for splade+dense: {dense_backend}"
                )

            # FAISS index cho dense
            dataset_name = os.path.normpath(out_dir).split(os.sep)[-2]
            model_tag = f"{dense_backend}_{safe_model_name(dense_model_name)}"
            cache_path = os.path.join("cache", dataset_name, model_tag)

            logger.info(f"[SPLADE+DENSE] Using FAISS DB path: {cache_path}")
            dense = DenseFAISS(base_dir=cache_path, index_metric=index_metric)

            dense.build_or_load_docs(
                contexts,
                embed_fn=dense_embedder.embed,
                force=force,
                show_progress=show_progress,
                batch_note=model_tag,
            )

            logger.info("[SPLADE+DENSE] Embedding queries for dense backend...")
            q_embs = dense.embed_queries(
                questions,
                embed_fn=dense_embedder.embed,
                force=force,
                show_progress=show_progress,
            )

            logger.info("[SPLADE+DENSE] Dense scoring (q @ d.T)...")
            d_scores = dense.dense_scores_for_queries(
                q_embs, show_progress=show_progress
            )

            # Fusion SPLADE (splade_scores) + dense (d_scores)
            if fusion == "alpha":
                scores = alpha * minmax_rowwise(d_scores) + (
                    1 - alpha
                ) * minmax_rowwise(splade_scores)
            elif fusion == "rrf":
                Q, N = len(questions), len(contexts)
                ranks_dense = np.zeros((Q, N), dtype=np.int32)
                ranks_splade = np.zeros((Q, N), dtype=np.int32)
                for i in range(Q):
                    ranks_dense[i] = ranks_from_scores(d_scores[i])
                    ranks_splade[i] = ranks_from_scores(splade_scores[i])
                scores = rrf_fuse_ranks([ranks_dense, ranks_splade], k=rrf_k)
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion}")

        else:
            raise ValueError(f"Unknown SPLADE-based method: {method}")

    # =====================================================================
    # DENSE-ONLY & LEGACY HYBRID DENSE
    # =====================================================================
    else:
        # Legacy branch for dense, dense+tfidf, dense+bm25 methods
        # now using dense_backend + dense_model.
        assert dense_backend is not None and dense_model is not None

        logger.info(f"Preparing dense backend: {dense_backend}")

        # ------------ load embedder -----------------
        if dense_backend == "openai":
            embedder = get_embedder(
                "openai",
                model=dense_model,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            model_name = dense_model

        elif dense_backend == "gemini":
            embedder = get_embedder(
                "gemini",
                model=dense_model,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            model_name = dense_model

        elif dense_backend in ("sbert", "llm"):
            embedder = get_embedder(
                dense_backend,
                model_name=dense_model,
                batch_size=batch_size,
                show_progress=show_progress,
                max_len=max_len,
            )
            model_name = dense_model

        else:
            raise ValueError(f"Unknown dense_backend: {dense_backend}")

        # =================================================================
        # FAISS dense index
        # =================================================================
        dataset_name = os.path.normpath(out_dir).split(os.sep)[-2]
        model_tag = f"{dense_backend}_{safe_model_name(model_name)}"
        cache_path = os.path.join("cache", dataset_name, model_tag)

        logger.info(f"Using FAISS DB path: {cache_path}")
        dense = DenseFAISS(base_dir=cache_path, index_metric=index_metric)

        dense.build_or_load_docs(
            contexts,
            embed_fn=embedder.embed,
            force=force,
            show_progress=show_progress,
            batch_note=model_tag,
        )

        logger.info("Embedding queries...")
        q_embs = dense.embed_queries(
            questions,
            embed_fn=embedder.embed,
            force=force,
            show_progress=show_progress,
        )

        logger.info("Dense scoring (q @ d.T)...")
        d_scores = dense.dense_scores_for_queries(q_embs, show_progress=show_progress)

        # ------------------ DENSE ONLY ------------------
        if method == "dense":
            scores = d_scores

        # ------------------ HYBRID TF-IDF (legacy) ------------------
        elif method == "dense+tfidf":
            vect, X_docs = build_tfidf(contexts)
            t_scores = tfidf_scores(vect, X_docs, questions)

            if fusion == "alpha":
                scores = alpha * minmax_rowwise(d_scores) + (
                    1 - alpha
                ) * minmax_rowwise(t_scores)

            elif fusion == "rrf":
                Q, N = len(questions), len(contexts)
                ranks_dense = np.zeros((Q, N), dtype=np.int32)
                ranks_tfidf = np.zeros((Q, N), dtype=np.int32)
                for i in range(Q):
                    ranks_dense[i] = ranks_from_scores(d_scores[i])
                    ranks_tfidf[i] = ranks_from_scores(t_scores[i])
                scores = rrf_fuse_ranks([ranks_dense, ranks_tfidf], k=rrf_k)

        # ------------------ HYBRID BM25 (legacy) ------------------
        elif method == "dense+bm25":
            bm25 = build_bm25(contexts, k1=bm25_k1, b=bm25_b)
            b_scores = bm25_scores(bm25, questions, show_progress=show_progress)

            if fusion == "alpha":
                scores = alpha * minmax_rowwise(d_scores) + (
                    1 - alpha
                ) * minmax_rowwise(b_scores)

            elif fusion == "rrf":
                Q, N = len(questions), len(contexts)
                ranks_dense = np.zeros((Q, N), dtype=np.int32)
                ranks_bm25 = np.zeros((Q, N), dtype=np.int32)
                for i in range(Q):
                    ranks_dense[i] = ranks_from_scores(d_scores[i])
                    ranks_bm25[i] = ranks_from_scores(b_scores[i])
                scores = rrf_fuse_ranks([ranks_dense, ranks_bm25], k=rrf_k)

        else:
            raise ValueError(f"Unknown method in dense branch: {method}")

    # =====================================================================
    # EXPORT RESULTS
    # =====================================================================
    assert scores is not None, "Internal error: scores not computed"

    logger.info("Saving ranks.json...")
    ranks_first = rank_of_first_gold(scores, gold_lists)
    save_json(ranks_first, ranks_path)

    logger.info("Evaluating retrieval metrics...")
    metrics = evaluate_all(scores, gold_lists, ks=ks, show_progress=show_progress)
    save_json(metrics, metrics_path)

    # =====================================================================
    # FAIL@K EXPORT
    # =====================================================================
    k_ref = error_k if error_k is not None else (max(ks) if ks else 10)
    errors_dir = os.path.join(out_dir, "errors")
    _ensure_dir(errors_dir)

    topk_ids = _topk_indices_rowwise(scores, k_ref)
    row_idx = np.arange(scores.shape[0])[:, None]
    topk_scores = scores[row_idx, topk_ids]

    rows = []
    Q = len(questions)

    for qi in range(Q):
        gold_id_set = set(gold_lists[qi]) if gold_lists else set()
        ret_ids = topk_ids[qi].tolist()

        if any(g in ret_ids for g in gold_id_set):
            continue

        gold_ids = list(gold_id_set)
        gold_texts = [contexts[g] for g in gold_ids if 0 <= g < len(contexts)]

        ret_texts = [contexts[d] for d in ret_ids]
        ret_scores_list = topk_scores[qi].tolist()

        rows.append(
            {
                "k_ref": k_ref,
                "qid": (qids[qi] if (qids and qi < len(qids)) else str(qi)),
                "question": questions[qi],
                "gold_doc_ids": gold_ids,
                "gold_texts": gold_texts,
                "retrieved_doc_ids": ret_ids,
                "retrieved_texts": ret_texts,
                "retrieved_scores": ret_scores_list,
            }
        )

    fail_csv = os.path.join(errors_dir, f"fail@{k_ref}.csv")
    _save_csv(
        fail_csv,
        rows,
        field_order=[
            "k_ref",
            "qid",
            "question",
            "gold_doc_ids",
            "gold_texts",
            "retrieved_doc_ids",
            "retrieved_texts",
            "retrieved_scores",
        ],
    )

    logger.info(f"Done. Artifacts written to: {out_dir}")
    return metrics
