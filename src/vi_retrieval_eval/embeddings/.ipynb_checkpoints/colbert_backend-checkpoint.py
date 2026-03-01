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
    ColBERT backend (dùng thư viện chính thức `colbert-ai`) cho vi-retrieval-eval.

    ✅ Đặc điểm:
      - Sử dụng đúng pipeline ColBERTv2:
          * Indexer: encode + build index với late interaction.
          * Searcher: MaxSim search trên corpus.
      - Trả về full score matrix (Q, N) để tái dùng toàn bộ pipeline:
          * rank_of_first_gold, evaluate_all, fail@K, v.v.

    ⚠️ Lưu ý:
      - KHÔNG phải dense-embedding kiểu single-vector → không dùng với DenseFAISS.
      - Được tích hợp qua get_embedder("colbert", model_name=..., batch_size=..., show_progress=...).
      - `runner.py` hiện đang gọi:
            embedder.build_docs(contexts)
            scores = embedder.score(questions)

    Cần cài (theo env của bạn hiện tại):
        pip install "colbert-ai==0.2.22" --no-deps
        pip install faiss-cpu ujson gitpython transformers
    và PyTorch (2.x) + FAISS đã ok.
    """

    def __init__(
        self,
        # alias để hợp với get_embedder(...)
        model_name: Optional[str] = None,
        # cho phép truyền thẳng checkpoint nếu muốn
        checkpoint: Optional[str] = None,
        # root mặc định: cache/colbert_indexes (runner nên override theo dataset)
        root: str = os.path.join("cache", "colbert_indexes"),
        index_name: str = "colbert_index",
        nbits: int = 2,
        doc_maxlen: int = 256,
        k_search: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 16,  # được get_embedder truyền vào, nhưng ColBERT-ai tự lo, nên tạm không dùng
        **_: object,          # nuốt bớt keyword thừa nếu sau này có thêm
    ) -> None:
        """
        Args:
            model_name: tên HF model (sẽ map sang checkpoint, ví dụ 'colbert-ir/colbertv2.0').
            checkpoint: nếu truyền trực tiếp checkpoint thì ưu tiên cái này.
            root: thư mục root cho ColBERT (experiments/, indexes/…).
                  Thực tế nên được runner set thành cache/colbert_indexes/<dataset_name>.
            index_name: tên index trong ColBERT.
            nbits: số bit residual compression (ColBERTConfig).
            doc_maxlen: max tokens cho passage.
            k_search: số doc top-k mỗi query; nếu None sẽ được suy trong _search_all.
            show_progress: bật/tắt tqdm.
            batch_size: không dùng trực tiếp, chỉ để tương thích với get_embedder.
        """
        self.logger = logging.getLogger("vi-retrieval-eval")

        # Import LAZY để không crash nếu user không dùng ColBERT
        try:
            from colbert.infra import Run, RunConfig, ColBERTConfig  # type: ignore
            from colbert import Indexer, Searcher  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "To use backend='colbert' you must install the official library:\n"
                '    pip install "colbert-ai==0.2.22" --no-deps\n'
                "and ensure PyTorch + FAISS are available."
            ) from e

        # Lưu reference class (để dùng sau, tránh re-import mỗi lần)
        self.Run = Run
        self.RunConfig = RunConfig
        self.ColBERTConfig = ColBERTConfig
        self.Indexer = Indexer
        self.Searcher = Searcher

        # Ưu tiên checkpoint nếu truyền trực tiếp, ngược lại lấy từ model_name,
        # cuối cùng fallback về colbert-ir/colbertv2.0
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
        self.k_search_default = k_search  # có thể None
        self.show_progress = bool(show_progress)

        # heartbeat interval (seconds) khi ColBERT encode/index mà không in progress
        self._heartbeat_sec = 10

        os.makedirs(self.root, exist_ok=True)

        self._num_docs: Optional[int] = None
        self._contexts: Optional[List[str]] = None  # giữ lại để Searcher biết collection

        self.logger.info(
            "[ColBERT] Initialized backend with checkpoint=%s, root=%s, index_name=%s",
            self.checkpoint,
            self.root,
            self.index_name,
        )

    # ------------------------------------------------------------------
    # Internal: build index từ contexts (passages)
    # ------------------------------------------------------------------
    def _build_index(self, contexts: List[str]) -> None:
        """
        Xây index ColBERT trên `contexts` (list[str]).

        Gọi mỗi lần cho một corpus (ví dụ, sau khi sampling/dedup).
        """
        if not contexts:
            self.logger.warning("[ColBERT] _build_index called with empty contexts.")
            self._num_docs = 0
            self._contexts = []
            return

        self._num_docs = len(contexts)
        self._contexts = list(contexts)  # giữ lại cho Searcher

        self.logger.info(
            "[ColBERT] Building index for %d documents (doc_maxlen=%d, nbits=%d) ...",
            self._num_docs,
            self.doc_maxlen,
            self.nbits,
        )

        # ---- heartbeat thread: để biết chắc là không treo ----
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
            # nranks=1: dùng 1 GPU / tiến trình
            with self.Run().context(
                self.RunConfig(
                    nranks=1,
                    experiment="vi-retrieval-eval",
                    # QUAN TRỌNG: root ở đây → mọi experiments, plans, logs đều nằm trong self.root
                    root=self.root,
                    # ✅ giảm nguy cơ deadlock do fork/tokenizers
                    avoid_fork_if_possible=True,
                )
            ):
                config = self.ColBERTConfig(
                    root=self.root,
                    nbits=self.nbits,
                    doc_maxlen=self.doc_maxlen,
                )

                indexer = self.Indexer(checkpoint=self.checkpoint, config=config)

                # collection: có thể là list[str]; ColBERT sẽ tự gán pid = index
                indexer.index(
                    name=self.index_name,
                    collection=self._contexts,
                    overwrite=True,  # benchmark corpus nhỏ, rebuild mỗi lần OK
                )
        finally:
            stop_flag["stop"] = True
            try:
                hb.join(timeout=1.0)
            except Exception:
                pass

        self.logger.info("[ColBERT] Index build completed for %d docs.", self._num_docs)

    # ------------------------------------------------------------------
    # Internal: search tất cả queries → scores(Q, N)
    # ------------------------------------------------------------------
    def _search_all(
        self,
        questions: List[str],
        ks: List[int],
    ) -> np.ndarray:
        """
        Chạy search ColBERT cho toàn bộ queries và trả về scores(Q, N).

        Ý tưởng:
          - Với mỗi query q:
              pids, ranks, _ = searcher.search(q, k=k_search)
          - Đặt score(q, pid) = k_search + 1 - rank  (rank bắt đầu từ 1).
            Các doc không trong top-k → score = 0 (coi như rất thấp).
          - Như vậy thứ hạng theo score trùng với thứ hạng ColBERT.
        """
        if self._num_docs is None or self._contexts is None:
            raise RuntimeError("[ColBERT] Index not built. Call _build_index() first.")

        Q = len(questions)
        N = self._num_docs
        scores = np.zeros((Q, N), dtype=np.float32)

        if Q == 0 or N == 0:
            return scores

        # k_search: max(max(ks), 10) nhưng không vượt quá N
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
                # ✅ đồng bộ với index, giảm deadlock
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
                # ColBERT trả về pid, rank, colbert_score
                pids, ranks, _colbert_scores = searcher.search(q, k=k_search)

                for pid, r in zip(pids, ranks):
                    pid = int(pid)
                    r = int(r)
                    if 0 <= pid < N and r >= 1:
                        scores[qi, pid] = float(k_search + 1 - r)

        return scores

    # ------------------------------------------------------------------
    # Public: API cho pipeline – build_docs + score
    # ------------------------------------------------------------------
    def build_docs(self, contexts: List[str]) -> None:
        """
        Được `runner.run()` gọi để build index từ list `contexts`.
        """
        self._build_index(contexts)

    def score(self, questions: List[str]) -> np.ndarray:
        """
        Được `runner.run()` gọi sau khi build_docs().
        Trả về ma trận score (Q, N).

        IMPORTANT:
        - Không được dùng ks=[N], vì sẽ khiến ColBERT search k=N (rất chậm).
        - Mặc định dùng ks=[1,10,20], phù hợp P@1, MRR@10, nDCG@10, R@10, R@20.
        """
        if self._num_docs is None:
            raise RuntimeError(
                "[ColBERT] score() được gọi trước khi build_docs(). "
                "Hãy gọi embedder.build_docs(contexts) trước."
            )

        ks = [1, 10, 20]
        return self._search_all(questions, ks)

    # ------------------------------------------------------------------
    # Public: API "real" – nếu muốn gọi trực tiếp ngoài runner
    # ------------------------------------------------------------------
    def run(
        self,
        questions: List[str],
        contexts: List[str],
        ks: List[int],
        out_dir: Optional[str] = None,
    ) -> np.ndarray:
        """
        API tiện nếu bạn muốn dùng ColBERTBackend độc lập:

            backend = ColBERTBackend(...)
            scores = backend.run(questions, contexts, ks, out_dir)

        Args:
            questions: list câu hỏi (Q).
            contexts: list passages/docs (N).
            ks: danh sách k (P@k, R@k, MRR@k...), dùng để suy ra k_search.
            out_dir: thư mục output; nếu set, có thể dùng để đặt root riêng cho ColBERT.

        Returns:
            scores: np.ndarray shape (Q, N), score(q, d) theo thứ hạng ColBERT.
        """
        if out_dir is not None:
            # Nếu muốn map 1-1 theo out_dir:
            base_name = os.path.basename(os.path.normpath(out_dir))
            self.root = os.path.join("cache", "colbert_indexes", base_name)
            os.makedirs(self.root, exist_ok=True)

        self._build_index(contexts)
        scores = self._search_all(questions, ks)
        return scores