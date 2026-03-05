# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict
from .textnorm import normalize_for_dedup


def dedup_by_content(
    contexts: List[str],
    *,
    do_lower: bool = False,
    remove_emoji: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Deduplicate corpus by normalized context string.
    Returns:
      - unique_contexts: list of deduplicated contexts.
      - doc_map: length == len(original contexts);
                 doc_map[i] = index of the unique context corresponding to contexts[i].
    """
    seen: Dict[str, int] = {}
    unique: List[str] = []
    doc_map: List[int] = []

    for c in contexts:
        key = normalize_for_dedup(c, do_lower=do_lower, remove_emoji=remove_emoji)
        idx = seen.get(key)
        if idx is None:
            idx = len(unique)
            seen[key] = idx
            unique.append(c)  # keep the original text (not the normalized key)
        doc_map.append(idx)
    return unique, doc_map


def remap_gold(gold_lists: List[List[int]], doc_map: List[int]) -> List[List[int]]:
    """
    Map gold doc indices from original space to deduplicated space using doc_map.
    Also deduplicates the gold indices (via set) for each query.
    """
    out: List[List[int]] = []
    for gold in gold_lists:
        mapped = sorted({doc_map[g] for g in gold})
        out.append(mapped)
    return out
