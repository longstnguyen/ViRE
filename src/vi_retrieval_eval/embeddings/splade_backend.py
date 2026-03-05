# src/vi_retrieval_eval/embeddings/splade_backend.py
# -*- coding: utf-8 -*-

from typing import List, Optional
import logging

import numpy as np

from .base import register
from ..progress import iter_progress


logger = logging.getLogger("vi-retrieval-eval")


@register("splade")
class SpladeBackend:
    """
    SPLADE backend using `sentence-transformers` SparseEncoder.

    - Default model: `naver/splade-v3`
    - Returns sparse similarity (dot product) between query and document.
    - In ViRE:
        * method = "splade"          → SPLADE-only (sparse retriever)
        * method = "splade+tfidf"    → hybrid SPLADE + TF-IDF
        * method = "splade+bm25"     → hybrid SPLADE + BM25
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 16,
        show_progress: bool = False,
        **_: object,
    ) -> None:
        """
        Args:
            model_name: HF SPLADE checkpoint (default: naver/splade-v3).
            batch_size: currently unused; present for get_embedder compatibility.
            show_progress: enable tqdm if True.
        """
        try:
            from sentence_transformers import SparseEncoder  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "To use backend='splade', install sentence-transformers:\n"
                "    pip install -U sentence-transformers"
            ) from e

        self.model_name = model_name or "naver/splade-v3"
        self.batch_size = batch_size
        self.show_progress = show_progress

        self.SparseEncoder = SparseEncoder
        self.encoder = self.SparseEncoder(self.model_name)

        logger.info("[SPLADE] Loaded SparseEncoder model: %s", self.model_name)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Main API: compute score(Q, N) between a list of queries and contexts
    # ------------------------------------------------------------------
    def score(self, questions: List[str], contexts: List[str]) -> np.ndarray:
        """
        Compute sparse similarity between queries and contexts.

        Returns:
            scores: np.ndarray shape (Q, N), dot product from SPLADE.
        """
        import torch

        Q = len(questions)
        N = len(contexts)

        if Q == 0 or N == 0:
            return np.zeros((Q, N), dtype=np.float32)

        logger.info("[SPLADE] Encoding %d documents...", N)
        doc_embs = self.encoder.encode_document(
            contexts,
            convert_to_tensor=True,
            show_progress_bar=self.show_progress,
        )  # shape: [N, D] (sparse or dense tensor)

        logger.info("[SPLADE] Encoding %d queries...", Q)
        query_embs = self.encoder.encode_query(
            questions,
            convert_to_tensor=True,
            show_progress_bar=self.show_progress,
        )  # shape: [Q, D]

        logger.info("[SPLADE] Computing sparse similarity (dot product)...")
        with torch.no_grad():
            sim = self.encoder.similarity(query_embs, doc_embs)  # [Q, N]

        scores = sim.detach().cpu().numpy().astype(np.float32)
        return scores
