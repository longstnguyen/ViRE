# registry + factory
from typing import Protocol, List, Dict, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_REGISTRY: Dict[str, Callable[..., "BaseEmbedder"]] = {}


class BaseEmbedder(Protocol):
    """Protocol for embedding backends.

    Implementations convert a list of input texts into a 2D vector array.
    """

    def embed(self, texts: List[str]) -> "np.ndarray":
        """Embed input texts into vectors.

        Args:
            texts: Input texts to embed.

        Returns:
            np.ndarray: Embedding matrix with shape (len(texts), dim).
        """
        ...  # pragma: no cover


def register(name: str):
    """Create a decorator that registers an embedder class.

    Args:
        name: Backend key used in the embedder registry.

    Returns:
        Callable: Class decorator that registers the target class.
    """

    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls

    return _wrap


def available_backends() -> str:
    """Return registered embedding backends.

    Returns:
        str: Comma-separated backend names.
    """
    return ", ".join(sorted(_REGISTRY.keys()))


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Instantiate an embedder from the registry.

    Args:
        name: Backend key.
        **kwargs: Keyword arguments forwarded to the embedder constructor.

    Returns:
        BaseEmbedder: Instantiated embedder backend.

    Raises:
        ValueError: If the backend name is not registered.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dense backend: {name}. Available: {available_backends()}"
        )
    return cls(**kwargs)  # type: ignore


# Convenience function for quick backend inspection / debugging.
def _debug_registry():
    return dict(_REGISTRY)
