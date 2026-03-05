# src/vi_retrieval_eval/embeddings/__init__.py

# Import backends to trigger @register(...) side-effects.
# Import order does not matter; only the registration side-effect is needed.
from . import openai_embed  # noqa: F401
from . import gemini_embed  # noqa: F401
from . import sbert_embed  # noqa: F401
from . import llm_embed  # noqa: F401
from . import colbert_backend  # noqa: F401
from . import splade_backend  # noqa: F401
