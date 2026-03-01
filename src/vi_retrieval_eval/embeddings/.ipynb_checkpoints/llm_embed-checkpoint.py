# src/vi_retrieval_eval/embeddings/llm_embed.py
from typing import List, Optional
import logging
import os
import numpy as np

from .base import register
from ..progress import iter_progress


@register("llm")
class LLMEmbedder:
    """
    LLM-based embedding backend using SentenceTransformer.

    Dùng cho các model như:
      - Qwen/Qwen3-Embedding-0.6B
      - Alibaba-NLP/gte-Qwen2-1.5B-instruct
      - BAAI/bge-multilingual-gemma2
      - jinaai/jina-embeddings-v3
      - v.v. (các HF embedding model có trust_remote_code)

    NEW:
      - max_len: nếu set, sẽ truncate input theo token length để tránh OOM do sequence quá dài.
        Nếu max_len=None: giữ nguyên (không cắt).

    NOTE (important for jina):
      - jinaai/jina-embeddings-v3 đôi khi kéo remote module "xlm-roberta-flash-implementation"
        và dễ bị cache lỗi thiếu file (rotary.py). Vì vậy: **tắt flash_attention_2** riêng cho jina.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 128,
        device: Optional[str] = None,
        show_progress: bool = False,
        normalize: bool = True,
        min_batch_size: int = 8,
        max_len: Optional[int] = None,  # CLI: --max-len
    ):
        try:
            import torch  # type: ignore
            from sentence_transformers import SentenceTransformer  # noqa: F401  # type: ignore
        except Exception as e:
            raise RuntimeError("Please `pip install sentence-transformers` to use LLM backend") from e

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
        self._logger.debug(f"[LLM] loading model `{model_name}` on `{device}` ...")

        # ---------------------------------------------------------------------
        # HOTFIX: transformers cache API mismatch (seen with Qwen2 remote code)
        # ---------------------------------------------------------------------
        try:
            from transformers.cache_utils import DynamicCache  # type: ignore
            if (not hasattr(DynamicCache, "get_usable_length")) and hasattr(DynamicCache, "get_seq_length"):

                def get_usable_length(self, *args, **kwargs):
                    return self.get_seq_length()

                DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]
                self._logger.warning(
                    "[LLM] Patched DynamicCache.get_usable_length(*args)->get_seq_length() "
                    "(compat for Qwen2 remote code)."
                )
        except Exception:
            pass

        # ----------------- model init policy -----------------
        from sentence_transformers import SentenceTransformer as ST  # type: ignore

        name_l = model_name.lower()

        # Disable flash for jina to avoid remote module import/cache issues
        disable_flash_env = os.getenv("VIRE_DISABLE_FLASH", "").strip() in {"1", "true", "yes"}
        is_jina = ("jinaai/" in name_l) or ("jina-embeddings" in name_l) or ("jina" in name_l)

        use_flash = (not disable_flash_env) and (not is_jina)

        # Qwen recommend left padding; for others keep default padding_side
        tokenizer_kwargs = {}
        if "qwen" in name_l:
            tokenizer_kwargs["padding_side"] = "left"

        st = None
        if use_flash:
            try:
                st = ST(
                    model_name,
                    device=device,
                    trust_remote_code=True,
                    model_kwargs={"attn_implementation": "flash_attention_2"},
                    tokenizer_kwargs=tokenizer_kwargs or None,
                )
                self._logger.debug("[LLM] Loaded with flash_attention_2 successfully.")
            except Exception as e:
                self._logger.warning(
                    f"[LLM] Could not enable flash_attention_2: {e}. "
                    "Falling back to default SentenceTransformer init."
                )
                st = ST(
                    model_name,
                    device=device,
                    trust_remote_code=True,
                    tokenizer_kwargs=tokenizer_kwargs or None,
                )
        else:
            # jina or env disabled flash -> default init (no flash)
            if is_jina:
                self._logger.warning("[LLM] Detected jina model -> disable flash_attention_2 (stability fix).")
            if disable_flash_env:
                self._logger.warning("[LLM] VIRE_DISABLE_FLASH=1 -> disable flash_attention_2.")
            st = ST(
                model_name,
                device=device,
                trust_remote_code=True,
                tokenizer_kwargs=tokenizer_kwargs or None,
            )

        self._st = st

        # Truncation config (best-effort); real truncate is done via tokenizer in _truncate_texts_if_needed
        if self.max_len is not None:
            try:
                self._st.max_seq_length = int(self.max_len)
                self._logger.warning(f"[LLM] Set SentenceTransformer.max_seq_length={self.max_len}")
            except Exception:
                pass

            try:
                tok = getattr(self._st, "tokenizer", None)
                if tok is not None:
                    tok.model_max_length = int(self.max_len)
                    tok.truncation_side = "right"
                    self._logger.warning(f"[LLM] Set tokenizer.model_max_length={self.max_len}")
            except Exception:
                pass

        # Detect query prompt support
        self._has_query_prompt = False
        try:
            prompts = getattr(self._st, "prompts", None)
            if isinstance(prompts, dict) and "query" in prompts:
                self._has_query_prompt = True
                self._logger.debug("[LLM] Detected `prompt_name='query'` support.")
        except Exception:
            self._has_query_prompt = False

    # ----------------- truncation helpers -----------------

    def _truncate_texts_if_needed(self, texts: List[str]) -> List[str]:
        """
        Truncate theo TOKEN bằng tokenizer nếu có.
        KHÔNG truyền (truncation/max_length) vào SentenceTransformer.encode()
        vì nhiều model sẽ ném ValueError "additional keyword arguments...".
        """
        if self.max_len is None or not texts:
            return texts

        # Guard: treat non-positive max_len as "no truncation"
        if self.max_len <= 0:
            return texts

        tok = getattr(self._st, "tokenizer", None)
        if tok is None:
            # fallback char-level (an toàn)
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
            self._logger.warning(f"[LLM] Tokenizer truncate failed ({e}). Falling back to char-truncate.")
            return [t[: self.max_len] for t in texts]

    # ----------------- core encode helpers -----------------

    def _encode_batch(
        self,
        texts: List[str],
        batch_size: int,
        *,
        is_query: bool,
    ) -> np.ndarray:
        """
        Encode 1 batch. Nếu is_query=True và model có prompt 'query'
        thì dùng prompt_name='query' như Qwen khuyến nghị.

        IMPORTANT:
          - Không truyền truncation/max_length vào encode().
          - Truncation đã xử lý ở _truncate_texts_if_needed().
        """
        encode_kwargs = dict(
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        if is_query and self._has_query_prompt:
            encode_kwargs["prompt_name"] = "query"

        arr = self._st.encode(texts, **encode_kwargs).astype(np.float32)

        if self.normalize and arr.size > 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr = (arr / norms).astype(np.float32)

        return arr

    # ----------------- public APIs -----------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed (doc-mode)."""
        return self._embed_generic(texts, is_query=False)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """Embed (query-mode, dùng prompt_name='query' nếu có)."""
        return self._embed_generic(texts, is_query=True)

    # ----------------- internal generic loop -----------------

    def _embed_generic(self, texts: List[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out_chunks: List[np.ndarray] = []
        bs = int(self.batch_size)
        total_batches = (len(texts) + bs - 1) // bs

        desc = f"LLM {self.model_name} ({'query' if is_query else 'doc'})"
        it = iter_progress(
            range(0, len(texts), bs),
            enable=self.show_progress,
            tqdm_desc=desc,
            total=total_batches,
        )

        for start in it:
            while True:
                try:
                    batch = texts[start: start + bs]
                    batch = self._truncate_texts_if_needed(batch)
                    arr = self._encode_batch(batch, batch_size=bs, is_query=is_query)
                    out_chunks.append(arr)
                    break
                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg or "cuda" in msg) and bs > self.min_batch_size:
                        new_bs = max(self.min_batch_size, bs // 2)
                        self._logger.warning(f"[LLM] OOM with batch_size={bs}. Retrying with batch_size={new_bs}.")
                        bs = new_bs
                        continue
                    raise

        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    @property
    def dim(self) -> int:
        arr = self._encode_batch(["dimension probe"], batch_size=1, is_query=False)
        return int(arr.shape[1])