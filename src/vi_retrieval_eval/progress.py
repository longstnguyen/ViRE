# src/vi_retrieval_eval/progress.py
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    enable: bool,
    tqdm_desc: Optional[str] = None,
    total: Optional[int] = None,
) -> Iterable[T]:
    """
    Wrap iterable with tqdm if enable=True, else return original iterable.
    - Does not require tqdm to be installed (silently falls back if unavailable).
    """
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm  # lazy import

        return tqdm(iterable, desc=tqdm_desc, total=total)
    except Exception:
        return iterable  # no tqdm installed → just no bar
