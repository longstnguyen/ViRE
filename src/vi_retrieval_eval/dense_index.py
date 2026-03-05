# src/vi_retrieval_eval/dense_index.py
from typing import List, Tuple, Callable, Optional
import os
import numpy as np

from .progress import iter_progress

HAVE_FAISS = False
try:
    import faiss  # type: ignore

    HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore


def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize an array along a target axis.

    Args:
        mat: Input matrix.
        axis: Axis to normalize.
        eps: Numerical floor to avoid division by zero.

    Returns:
        np.ndarray: L2-normalized float32 matrix.
    """
    norm = np.linalg.norm(mat, ord=2, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return (mat / norm).astype(np.float32)


class DenseFAISS:
    """
    FAISS index for dense retrieval. Accepts embed_fn: List[str] -> np.ndarray (B, D).
    - embed_fn may already normalize; set normalize=True to force L2-normalize again.
    - Using IP (dot) with normalized vectors is equivalent to cosine similarity.
    """

    def __init__(self, base_dir: str, index_metric: str = "ip"):
        if not HAVE_FAISS:
            raise RuntimeError("FAISS not available. Please `pip install faiss-cpu`.")
        self.base_dir = base_dir
        self.index_metric = index_metric.lower().strip()  # 'ip' | 'l2'
        os.makedirs(self.base_dir, exist_ok=True)

        self.doc_emb_path = os.path.join(base_dir, "doc_embeddings.npy")
        self.query_emb_path = os.path.join(base_dir, "query_embeddings.npy")
        self.index_path = os.path.join(base_dir, "index.faiss")

        self.index: Optional["faiss.Index"] = None
        self.doc_embs: Optional[np.ndarray] = None

    # -------------------------- Docs --------------------------
    def build_or_load_docs(
        self,
        docs: List[str],
        embed_fn: Callable[[List[str]], np.ndarray],
        *,
        force: bool = False,
        normalize: bool = False,
        show_progress: bool = False,
        batch_note: str = "",
        batch_size_hint: Optional[int] = None,  # only used for progress display
    ) -> Tuple[np.ndarray, "faiss.Index"]:
        """Build or load document embeddings and FAISS index.

        Args:
            docs: Document texts.
            embed_fn: Embedding function that maps texts to vectors.
            force: Whether to force rebuilding cached artifacts.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Whether to show progress bars.
            batch_note: Optional note appended to progress description.
            batch_size_hint: Optional batch-size hint for progress display only.

        Returns:
            Tuple[np.ndarray, faiss.Index]: Document embeddings and FAISS index.
        """
        if (
            (not force)
            and os.path.exists(self.doc_emb_path)
            and os.path.exists(self.index_path)
        ):
            self.doc_embs = np.load(self.doc_emb_path)
            self.index = faiss.read_index(self.index_path)  # type: ignore
            return self.doc_embs, self.index  # type: ignore

        # Embed (embed_fn may handle batching + progress internally; we show one outer bar)
        desc = "Embedding docs"
        if batch_note:
            desc = f"{desc} ({batch_note})"
        # If embed_fn has no internal progress bar, this is a single-step placeholder
        outer = iter_progress([0], enable=show_progress, tqdm_desc=desc, total=1)
        for _ in outer:
            embs = embed_fn(docs)

        if embs.dtype != np.float32:
            embs = embs.astype(np.float32)

        if normalize:
            embs = l2_normalize(embs, axis=1)

        self.doc_embs = embs
        np.save(self.doc_emb_path, embs)

        d = int(embs.shape[1])
        if self.index_metric == "ip":
            index = faiss.IndexFlatIP(d)  # type: ignore
        elif self.index_metric == "l2":
            index = faiss.IndexFlatL2(d)  # type: ignore
        else:
            raise ValueError(
                f"Unknown index_metric: {self.index_metric}. Use 'ip' or 'l2'."
            )

        index.add(embs)  # type: ignore
        faiss.write_index(index, self.index_path)  # type: ignore
        self.index = index  # type: ignore
        return embs, index  # type: ignore

    # -------------------------- Queries --------------------------
    def embed_queries(
        self,
        queries: List[str],
        embed_fn: Callable[[List[str]], np.ndarray],
        *,
        force: bool = False,
        normalize: bool = False,
        show_progress: bool = False,
        batch_size_hint: Optional[int] = None,  # only used for progress display
    ) -> np.ndarray:
        """Build or load query embeddings.

        Args:
            queries: Query texts.
            embed_fn: Embedding function that maps texts to vectors.
            force: Whether to force rebuilding cached embeddings.
            normalize: Whether to L2-normalize query vectors.
            show_progress: Whether to show progress bars.
            batch_size_hint: Optional batch-size hint for progress display only.

        Returns:
            np.ndarray: Query embedding matrix.
        """
        if (not force) and os.path.exists(self.query_emb_path):
            try:
                arr = np.load(self.query_emb_path)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                return arr
            except Exception:
                pass

        desc = "Embedding queries"
        outer = iter_progress([0], enable=show_progress, tqdm_desc=desc, total=1)
        for _ in outer:
            q_embs = embed_fn(queries)

        if q_embs.dtype != np.float32:
            q_embs = q_embs.astype(np.float32)

        if normalize:
            q_embs = l2_normalize(q_embs, axis=1)

        np.save(self.query_emb_path, q_embs)
        return q_embs

    # -------------------------- Scoring --------------------------
    def dense_scores_for_queries(
        self,
        q_embs: np.ndarray,
        *,
        show_progress: bool = False,
        score_batch_size: int = 128,
    ) -> np.ndarray:
        """Compute dense query-document scores.

        Uses matrix multiplication: (Q, D) @ (D, N) -> (Q, N).

        Args:
            q_embs: Query embedding matrix with shape (Q, D).
            show_progress: Whether to show a progress bar.
            score_batch_size: Query batch size for block-wise scoring.

        Returns:
            np.ndarray: Score matrix with shape (Q, N).
        """
        assert self.doc_embs is not None, "doc_embs not built yet"
        Q = q_embs.shape[0]
        rows = []
        total = (Q + score_batch_size - 1) // score_batch_size

        it = iter_progress(
            range(0, Q, score_batch_size),
            enable=show_progress,
            tqdm_desc="Dense scoring",
            total=total,
        )
        for i in it:
            qq = q_embs[i : i + score_batch_size]  # (B, D)
            # dot product (IP); even for L2 index, dot product suffices for ranking
            block = qq @ self.doc_embs.T  # (B, N)
            rows.append(block.astype(np.float32))

        return np.vstack(rows)
