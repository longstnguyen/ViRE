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
    SPLADE backend dùng `sentence-transformers` SparseEncoder.

    - Model default: `naver/splade-v3`
    - Trả về sparse similarity (dot product) giữa query và document.
    - Trong ViRE:
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
            model_name: tên checkpoint HF SPLADE (default: naver/splade-v3).
            batch_size: hiện tại không dùng, chỉ để đồng bộ với get_embedder.
            show_progress: nếu True thì bật tqdm.
        """
        try:
            from sentence_transformers import SparseEncoder  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "To use backend='splade', cần cài sentence-transformers:\n"
                "    pip install -U sentence-transformers"
            ) from e

        self.model_name = model_name or "naver/splade-v3"
        self.batch_size = batch_size
        self.show_progress = show_progress

        self.SparseEncoder = SparseEncoder
        self.encoder = self.SparseEncoder(self.model_name)

        logger.info("[SPLADE] Loaded SparseEncoder model: %s", self.model_name)

    # ------------------------------------------------------------------
    # API chính: tính score(Q, N) giữa list queries và list contexts
    # ------------------------------------------------------------------
    def score(self, questions: List[str], contexts: List[str]) -> np.ndarray:
        """
        Tính sparse similarity giữa queries và contexts.

        Returns:
            scores: np.ndarray shape (Q, N), là dot product SPLADE.
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
