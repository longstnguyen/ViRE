#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/vi_retrieval_eval/embeddings/colbert_backend.py

from typing import List, Optional
import os
import logging
import threading
import time

import numpy as np

from .base import register
from ..progress import iter_progress


@register("colbert")
class ColBERTBackend:
    """
    ColBERT backend (using the official `colbert-ai` library) for vi-retrieval-eval.

    Features:
      - Uses the full ColBERTv2 pipeline:
          * Indexer: encode + build index with late interaction.
          * Searcher: MaxSim search over the corpus.
      - Returns the full score matrix (Q, N) for use in the complete pipeline:
          * rank_of_first_gold, evaluate_all, fail@K, etc.

    Note:
      - NOT a single-vector dense embedding — do not use with DenseFAISS.
      - Integrated via get_embedder("colbert", model_name=..., batch_size=..., show_progress=...).
      - `runner.py` calls:
            embedder.build_docs(contexts)
            scores = embedder.score(questions)

    Install (per the current environment):
        pip install "colbert-ai==0.2.22" --no-deps
        pip install faiss-cpu ujson gitpython transformers
    PyTorch (2.x) + FAISS required.
    """

    def __init__(
        self,
        # alias for compatibility with get_embedder(...)
        model_name: Optional[str] = None,
        # allow passing checkpoint directly
        checkpoint: Optional[str] = None,
        # default root: cache/colbert_indexes (runner should override per dataset)
        root: str = os.path.join("cache", "colbert_indexes"),
        index_name: str = "colbert_index",
        nbits: int = 2,
        doc_maxlen: int = 256,
        k_search: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 16,  # passed by get_embedder but ColBERT-ai manages its own batching; unused
        **_: object,  # absorb extra keyword args for future compatibility
    ) -> None:
        """
        Args:
            model_name: HF model name (mapped to checkpoint, e.g. 'colbert-ir/colbertv2.0').
            checkpoint: if provided, takes priority over model_name.
            root: root directory for ColBERT (experiments/, indexes/, etc.).
                  Should be set by runner to cache/colbert_indexes/<dataset_name>.
            index_name: name of the ColBERT index.
            nbits: residual compression bits (ColBERTConfig).
            doc_maxlen: max tokens per passage.
            k_search: top-k docs per query; inferred in _search_all if None.
            show_progress: enable/disable tqdm.
            batch_size: not used directly; present for get_embedder compatibility.
        """
        self.logger = logging.getLogger("vi-retrieval-eval")

        # Lazy import to avoid crashing if the user doesn't install ColBERT
        try:
            from colbert.infra import Run, RunConfig, ColBERTConfig  # type: ignore
            from colbert import Indexer, Searcher  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "To use backend='colbert' you must install the official library:\n"
                '    pip install "colbert-ai==0.2.22" --no-deps\n'
                "and ensure PyTorch + FAISS are available."
            ) from e

        # Store class references to avoid re-importing on each call
        self.Run = Run
        self.RunConfig = RunConfig
        self.ColBERTConfig = ColBERTConfig
        self.Indexer = Indexer
        self.Searcher = Searcher

        # Priority: explicit checkpoint > model_name > default fallback
        if checkpoint is not None:
            self.checkpoint = checkpoint
        elif model_name is not None:
            self.checkpoint = model_name
        else:
            self.checkpoint = "colbert-ir/colbertv2.0"

        self.root = root
        self.index_name = index_name
        self.nbits = int(nbits)
        self.doc_maxlen = int(doc_maxlen)
        self.k_search_default = k_search  # may be None
        self.show_progress = bool(show_progress)

        # heartbeat interval (seconds) during ColBERT encode/index when progress is not visible
        self._heartbeat_sec = 10

        os.makedirs(self.root, exist_ok=True)

        self._num_docs: Optional[int] = None
        self._contexts: Optional[List[str]] = (
            None  # retained so Searcher knows the collection
        )

        self.logger.info(
            "[ColBERT] Initialized backend with checkpoint=%s, root=%s, index_name=%s",
            self.checkpoint,
            self.root,
            self.index_name,
        )

    # ------------------------------------------------------------------
    # Internal: build index from contexts (passages)
    # ------------------------------------------------------------------
    def _build_index(self, contexts: List[str]) -> None:
        """
        Build a ColBERT index from `contexts` (list[str]).

        Called once per corpus (e.g., after sampling/dedup).
        """
        if not contexts:
            self.logger.warning("[ColBERT] _build_index called with empty contexts.")
            self._num_docs = 0
            self._contexts = []
            return

        self._num_docs = len(contexts)
        self._contexts = list(contexts)  # retained so Searcher knows the collection

        self.logger.info(
            "[ColBERT] Building index for %d documents (doc_maxlen=%d, nbits=%d) ...",
            self._num_docs,
            self.doc_maxlen,
            self.nbits,
        )

        # ---- heartbeat thread: ensures we know the process isn't frozen ----
        stop_flag = {"stop": False}

        def _heartbeat() -> None:
            t0 = time.time()
            while not stop_flag["stop"]:
                self.logger.info(
                    "[ColBERT] still indexing/encoding... elapsed=%.0fs (docs=%d)",
                    time.time() - t0,
                    self._num_docs or 0,
                )
                time.sleep(self._heartbeat_sec)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()

        try:
            # nranks=1: use one GPU / process
            with self.Run().context(
                self.RunConfig(
                    nranks=1,
                    experiment="vi-retrieval-eval",
                    # IMPORTANT: root here → all experiments, plans, logs live under self.root
                    root=self.root,
                    # reduce deadlock risk from fork/tokenizers
                    avoid_fork_if_possible=True,
                )
            ):
                config = self.ColBERTConfig(
                    root=self.root,
                    nbits=self.nbits,
                    doc_maxlen=self.doc_maxlen,
                )

                indexer = self.Indexer(checkpoint=self.checkpoint, config=config)

                # collection may be list[str]; ColBERT assigns pid = position index
                indexer.index(
                    name=self.index_name,
                    collection=self._contexts,
                    overwrite=True,  # benchmark corpus is small; rebuilding each run is fine
                )
        finally:
            stop_flag["stop"] = True
            try:
                hb.join(timeout=1.0)
            except Exception:
                pass

        self.logger.info("[ColBERT] Index build completed for %d docs.", self._num_docs)

    # ------------------------------------------------------------------
    # Internal: search all queries → scores(Q, N)
    # ------------------------------------------------------------------
    def _search_all(
        self,
        questions: List[str],
        ks: List[int],
    ) -> np.ndarray:
        """
        Run ColBERT search for all queries and return scores(Q, N).

        Approach:
          - For each query q:
              pids, ranks, _ = searcher.search(q, k=k_search)
          - Set score(q, pid) = k_search + 1 - rank  (rank starts at 1).
            Docs outside top-k get score = 0 (treated as very low).
          - Score-based ranking therefore matches ColBERT ranking.
        """
        if self._num_docs is None or self._contexts is None:
            raise RuntimeError("[ColBERT] Index not built. Call _build_index() first.")

        Q = len(questions)
        N = self._num_docs
        scores = np.zeros((Q, N), dtype=np.float32)

        if Q == 0 or N == 0:
            return scores

        # k_search: max(max(ks), 10) but capped at N
        k_search = self.k_search_default
        if k_search is None:
            k_search = max(max(ks) if ks else 0, 10)
        k_search = min(k_search, N)

        self.logger.info("[ColBERT] Searching %d queries with k=%d ...", Q, k_search)

        with self.Run().context(
            self.RunConfig(
                nranks=1,
                experiment="vi-retrieval-eval",
                root=self.root,
                # synchronize with index, reduce deadlock risk
                avoid_fork_if_possible=True,
            )
        ):
            config = self.ColBERTConfig(
                root=self.root,
                nbits=self.nbits,
                doc_maxlen=self.doc_maxlen,
            )

            searcher = self.Searcher(
                index=self.index_name,
                collection=self._contexts,
                config=config,
            )

            it = iter_progress(
                range(Q),
                enable=self.show_progress,
                tqdm_desc="ColBERT search",
                total=Q,
            )

            for qi in it:
                q = questions[qi]
                # ColBERT returns pid, rank, colbert_score
                pids, ranks, _colbert_scores = searcher.search(q, k=k_search)

                for pid, r in zip(pids, ranks):
                    pid = int(pid)
                    r = int(r)
                    if 0 <= pid < N and r >= 1:
                        scores[qi, pid] = float(k_search + 1 - r)

        return scores

    # ------------------------------------------------------------------
    # Public: pipeline API – build_docs + score
    # ------------------------------------------------------------------
    def build_docs(self, contexts: List[str]) -> None:
        """Build a ColBERT index from contexts.

        Args:
            contexts: Corpus passages/documents to index.
        """
        self._build_index(contexts)

    def score(self, questions: List[str]) -> np.ndarray:
        """
        Called by `runner.run()` after build_docs().
        Returns the score matrix (Q, N).

        IMPORTANT:
        - Do not pass ks=[N]; this would cause ColBERT to search k=N (very slow).
        - Default ks=[1,10,20] suits P@1, MRR@10, nDCG@10, R@10, R@20.
        """
        if self._num_docs is None:
            raise RuntimeError(
                "[ColBERT] score() called before build_docs(). "
                "Call embedder.build_docs(contexts) first."
            )

        ks = [1, 10, 20]
        return self._search_all(questions, ks)

    # ------------------------------------------------------------------
    # Public: standalone API – for direct use outside runner
    # ------------------------------------------------------------------
    def run(
        self,
        questions: List[str],
        contexts: List[str],
        ks: List[int],
        out_dir: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convenience API for using ColBERTBackend independently:

            backend = ColBERTBackend(...)
            scores = backend.run(questions, contexts, ks, out_dir)

        Args:
            questions: list of query strings (Q).
            contexts: list of passages/docs (N).
            ks: list of k values (P@k, R@k, MRR@k ...) used to determine k_search.
            out_dir: output directory; if set, can be used to set a custom ColBERT root.

        Returns:
            scores: np.ndarray shape (Q, N), score(q, d) per ColBERT ranking.
        """
        if out_dir is not None:
            # Map 1-to-1 according to out_dir if desired:
            base_name = os.path.basename(os.path.normpath(out_dir))
            self.root = os.path.join("cache", "colbert_indexes", base_name)
            os.makedirs(self.root, exist_ok=True)

        self._build_index(contexts)
        scores = self._search_all(questions, ks)
        return scores
