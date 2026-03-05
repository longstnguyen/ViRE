# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from typing import Optional
from .textnorm import normalize_for_dedup


def _add_norm_key(
    df: pd.DataFrame, *, col: str, lower: bool, rm_emoji: bool
) -> pd.DataFrame:
    out = df.copy()
    out["_norm_key"] = (
        out[col]
        .astype(str)
        .apply(lambda s: normalize_for_dedup(s, do_lower=lower, remove_emoji=rm_emoji))
    )
    return out


def prefer_unique_context_sampling(
    df: pd.DataFrame,
    n: int,
    *,
    unique_col: str = "context",
    seed: int = 42,
    norm_lower: bool = False,
    norm_remove_emoji: bool = False,
) -> pd.DataFrame:
    """
    Try to return exactly n samples:
      1) Prefer rows with unique contexts by NORMALIZED KEY (_norm_key).
      2) If fewer unique rows than n, pad with random rows from the remainder (may include duplicates).
    The returned DataFrame preserves original columns; _norm_key is used only for selection then dropped.
    """
    if n <= 0:
        return df.iloc[0:0].copy()

    # Build a normalized key for deduplication matching
    df_keyed = _add_norm_key(
        df, col=unique_col, lower=norm_lower, rm_emoji=norm_remove_emoji
    )

    # Shuffle for random ordering
    shuffled = df_keyed.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Keep one row per normalized key, in shuffled order
    deduped_unique = shuffled.drop_duplicates(subset=["_norm_key"], keep="first")

    if len(deduped_unique) >= n:
        res = deduped_unique.head(n).drop(columns=["_norm_key"]).reset_index(drop=True)
        return res

    # Not enough unique rows; pad with remaining rows to reach n
    need = n - len(deduped_unique)
    remaining = shuffled.drop(deduped_unique.index)
    filler = remaining.head(min(need, len(remaining)))
    result = pd.concat([deduped_unique, filler], ignore_index=True)
    result = (
        result.head(min(n, len(result)))
        .drop(columns=["_norm_key"])
        .reset_index(drop=True)
    )
    return result


def sample_with_flags(
    df: pd.DataFrame,
    *,
    max_samples: Optional[int],
    sample_frac: Optional[float],
    seed: int,
    prefer_unique: bool,
    unique_col: str = "context",
    norm_lower: bool = False,
    norm_remove_emoji: bool = False,
) -> pd.DataFrame:
    """
    Unified sampling:
    - Compute target n from sample_frac / max_samples.
    - If prefer_unique=True: prefer rows with unique normalized keys, pad to n if needed.
    - If prefer_unique=False: standard random sampling.
    - If n >= len(df): return df; if prefer_unique=True, still deduplicate by key then pad.
    """
    # determine target n
    n_target: Optional[int] = None
    if sample_frac is not None and 0 < sample_frac < 1.0:
        n_target = max(1, round(sample_frac * len(df)))
    if max_samples is not None and max_samples > 0:
        n_target = max_samples if n_target is None else min(max_samples, n_target)

    if not n_target or n_target >= len(df):
        if prefer_unique:
            return prefer_unique_context_sampling(
                df,
                len(df),
                unique_col=unique_col,
                seed=seed,
                norm_lower=norm_lower,
                norm_remove_emoji=norm_remove_emoji,
            )
        return df

    if prefer_unique:
        return prefer_unique_context_sampling(
            df,
            n_target,
            unique_col=unique_col,
            seed=seed,
            norm_lower=norm_lower,
            norm_remove_emoji=norm_remove_emoji,
        )
    return df.sample(n=n_target, random_state=seed).reset_index(drop=True)
