# src/vi_retrieval_eval/embeddings/sbert_embed.py
from typing import List, Optional
import logging
import numpy as np

from .base import register
from ..progress import iter_progress


@register("sbert")
class SBERTEmbedder:
    """
    Sentence-Transformers (local).

    Yêu cầu:
      - pip install sentence-transformers
      - GPU (tuỳ chọn): tự dùng 'cuda' nếu có, hoặc bạn có thể chỉ định device.

    NEW:
      - max_len: nếu set, sẽ truncate input theo token length để giảm OOM do sequence dài.
        Nếu max_len=None: giữ nguyên (không cắt).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
        device: Optional[str] = None,
        show_progress: bool = False,
        normalize: bool = True,
        min_batch_size: int = 8,  # chống OOM: không giảm dưới mức này
        max_len: Optional[int] = None,  # <<< NEW (CLI: --max-len)
    ):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("Please `pip install sentence-transformers` to use SBERT") from e

        # auto-select device nếu không chỉ định
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.show_progress = bool(show_progress)
        self.normalize = bool(normalize)
        self.min_batch_size = int(min_batch_size)
        self.max_len = None if max_len is None else int(max_len)

        self._logger = logging.getLogger("vi-retrieval-eval")

        # load model
        self._logger.debug(f"[SBERT] loading model `{model_name}` on `{device}` ...")
        self._st = SentenceTransformer(model_name, device=device, trust_remote_code=True)

        # NEW: global truncation config (only if provided)
        if self.max_len is not None:
            # SentenceTransformer may respect this
            try:
                self._st.max_seq_length = int(self.max_len)
                self._logger.warning(f"[SBERT] Set SentenceTransformer.max_seq_length={self.max_len}")
            except Exception:
                pass

            # Tokenizer safety
            try:
                tok = getattr(self._st, "tokenizer", None)
                if tok is not None:
                    tok.model_max_length = int(self.max_len)
                    tok.truncation_side = "right"
                    self._logger.warning(f"[SBERT] Set tokenizer.model_max_length={self.max_len}")
            except Exception:
                pass

    # ----------------- truncation helpers -----------------

    def _truncate_texts_if_needed(self, texts: List[str]) -> List[str]:
        """
        Truncate theo TOKEN bằng tokenizer nếu có.
        Không truyền truncation/max_length vào SentenceTransformer.encode().
        """
        if self.max_len is None or not texts:
            return texts

        tok = getattr(self._st, "tokenizer", None)
        if tok is None:
            # fallback char-level
            return [t[: self.max_len] for t in texts]

        try:
            enc = tok(
                texts,
                truncation=True,
                max_length=int(self.max_len),
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            input_ids = enc.get("input_ids", None)
            if input_ids is None:
                return [t[: self.max_len] for t in texts]

            out = tok.batch_decode(input_ids, skip_special_tokens=True)
            return out
        except Exception as e:
            self._logger.warning(f"[SBERT] Tokenizer truncate failed ({e}). Falling back to char-truncate.")
            return [t[: self.max_len] for t in texts]

    # ----------------- encode -----------------

    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Encode một batch → np.ndarray (B, D), có normalize nếu bật.
        """
        arr = self._st.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalize thủ công để thống nhất
            show_progress_bar=False,
        ).astype(np.float32)

        if self.normalize and arr.size > 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr = (arr / norms).astype(np.float32)

        return arr

    # ----------------- public API -----------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Trả về embeddings cho toàn bộ `texts` với progress bar theo batch và chống OOM.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out_chunks: List[np.ndarray] = []
        bs = int(self.batch_size)
        total_batches = (len(texts) + bs - 1) // bs

        it = iter_progress(
            range(0, len(texts), bs),
            enable=self.show_progress,
            tqdm_desc=f"SBERT {self.model_name}",
            total=total_batches,
        )

        for start in it:
            while True:
                try:
                    batch = texts[start: start + bs]
                    batch = self._truncate_texts_if_needed(batch)  # <<< truncate here
                    arr = self._encode_batch(batch, batch_size=bs)
                    out_chunks.append(arr)
                    break
                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg or "cuda" in msg) and bs > self.min_batch_size:
                        new_bs = max(self.min_batch_size, bs // 2)
                        self._logger.warning(
                            f"[SBERT] OOM detected with batch_size={bs}. Retrying with batch_size={new_bs}."
                        )
                        bs = new_bs
                        continue
                    raise

        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    @property
    def dim(self) -> int:
        """Chiều embedding (encode 1 câu để đo)."""
        arr = self._encode_batch(["dimension probe"], batch_size=1)
        return int(arr.shape[1])