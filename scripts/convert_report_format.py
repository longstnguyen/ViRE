#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert VIRE_Report.md from old format (Method | Model | P@1 | ...) to new format
(model-as-group-header, bold/underline best/second-best).

The new format matches the output of summarize_report.py:
- No Model column
- Lexical methods (tfidf, bm25, splade, colbert) listed first as flat rows
- Dense methods grouped under "**Dense model: X**" header rows
- Best value per metric column shown in **bold**, second-best in <u>underline</u>
"""

import math
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants (must match summarize_report.py)
# ---------------------------------------------------------------------------
METHOD_DISPLAY = {
    "tfidf": "tfidf",
    "bm25": "bm25",
    "splade": "splade",
    "colbert": "colbert",
    "dense": "dense",
    "dense+tfidf-alpha": "dense + tfidf ($\\alpha$)",
    "dense+tfidf-rrf": "dense + tfidf (RRF)",
    "dense+bm25-alpha": "dense + bm25 ($\\alpha$)",
    "dense+bm25-rrf": "dense + bm25 (RRF)",
    "unknown": "unknown",
}

LEXICAL_METHODS: List[str] = ["tfidf", "bm25", "splade", "colbert"]
DENSE_METHODS_ORDER: List[str] = [
    "dense",
    "dense+tfidf-alpha",
    "dense+tfidf-rrf",
    "dense+bm25-alpha",
    "dense+bm25-rrf",
]
METRICS_LABELS: List[str] = ["P@1", "R@10", "MRR@10", "nDCG@10", "R@20"]


# ---------------------------------------------------------------------------
# Method canonicalization (must match summarize_report.py)
# ---------------------------------------------------------------------------
_dense_tfidf_alpha = re.compile(r"dense\+tfidf-alpha", re.IGNORECASE)
_dense_tfidf_rrf = re.compile(r"dense\+tfidf-rrf", re.IGNORECASE)
_dense_bm25_alpha = re.compile(r"dense\+bm25-alpha", re.IGNORECASE)
_dense_bm25_rrf = re.compile(r"dense\+bm25-rrf", re.IGNORECASE)


def canonicalize_method(method_raw: str) -> str:
    """Canonicalize method name from old format to standard key."""
    name = method_raw.strip().lower()
    if name == "tfidf":
        return "tfidf"
    if name == "bm25":
        return "bm25"
    if name == "splade":
        return "splade"
    if name == "colbert":
        return "colbert"
    if _dense_tfidf_alpha.search(name):
        return "dense+tfidf-alpha"
    if _dense_tfidf_rrf.search(name):
        return "dense+tfidf-rrf"
    if _dense_bm25_alpha.search(name):
        return "dense+bm25-alpha"
    if _dense_bm25_rrf.search(name):
        return "dense+bm25-rrf"
    if name.startswith("dense") or "dense" in name:
        return "dense"
    return "unknown"


# ---------------------------------------------------------------------------
# Parse old VIRE_Report.md
# ---------------------------------------------------------------------------
def parse_pct(s: str) -> Optional[float]:
    """Parse '89.25%' → 0.8925 (or None on failure)."""
    s = s.strip().rstrip("%")
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def parse_old_report(
    text: str,
) -> Tuple[List[str], Dict[str, Dict[Tuple[str, str], Dict[str, float]]]]:
    """
    Returns:
        dataset_order: list of dataset names in document order
        data: {dataset: {(canon_method, model_canon): {metric_label: float_0_to_1}}}

    model_canon normalises `sbert-Vietnamese_Embedding_v2` →
    `sbert-Vietnamese_Embedding_V2` so duplicates are merged.
    """
    dataset_order: List[str] = []
    # ds -> OrderedDict[(method, model)] -> {metric: value}
    data: Dict[str, "OrderedDict[Tuple[str, str], Dict[str, float]]"] = {}
    cur_ds: Optional[str] = None

    for line in text.splitlines():
        # Dataset header
        h2 = re.match(r"^##\s+(\S.*?)\s*$", line)
        if h2:
            name = h2.group(1).strip()
            if name not in ("Datasets",):
                cur_ds = name
                if cur_ds not in data:
                    dataset_order.append(cur_ds)
                    data[cur_ds] = OrderedDict()
            continue

        # Table row (skip separator rows and header row)
        if cur_ds is None:
            continue
        if not line.startswith("|"):
            continue
        if re.match(r"^\|\s*[-:]+", line):
            continue

        cells = [c.strip() for c in line.split("|")]
        # remove empty leading/trailing from split
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if len(cells) < 7:
            continue

        method_raw, model_raw = cells[0], cells[1]
        # skip header row
        if method_raw.lower() in ("method",):
            continue

        metric_vals = cells[2:7]  # P@1, R@10, MRR@10, nDCG@10, R@20

        parsed_vals = [parse_pct(v) for v in metric_vals]
        if all(v is None for v in parsed_vals):
            continue  # skip non-data rows

        canon = canonicalize_method(method_raw)
        # Normalise model name: treat v2/V2 suffix variants as identical
        model = model_raw.strip()
        model_norm = model.replace("_v2", "_V2")  # unify Vietnamese_Embedding_v2 → V2

        key = (canon, model_norm)
        # Deduplicate: keep first occurrence
        if key not in data[cur_ds]:
            row: Dict[str, float] = {}
            for label, val in zip(METRICS_LABELS, parsed_vals):
                if val is not None:
                    row[label] = val
            data[cur_ds][key] = row

    return dataset_order, data


# ---------------------------------------------------------------------------
# Build new-format Markdown (matches build_markdown_report logic)
# ---------------------------------------------------------------------------
def is_number(x) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except Exception:
        return False


def fmt_val(v: float, ndigits: int = 2) -> str:
    return f"{v * 100:.{ndigits}f}%"


def build_new_report(
    dataset_order: List[str],
    data: Dict[str, "OrderedDict[Tuple[str, str], Dict[str, float]]"],
) -> str:
    parts: List[str] = []
    parts.append("# ViRE Retrieval Report\n")
    parts.append(f"Metrics: `{', '.join(METRICS_LABELS)}` (shown as %)\n")
    parts.append("## Datasets\n")
    for ds in dataset_order:
        anchor = ds.lower()
        parts.append(f"- [{ds}](#{anchor})")
    parts.append("")

    for ds in dataset_order:
        parts.append(f"## {ds}\n")

        ds_data = data.get(ds, OrderedDict())
        if not ds_data:
            parts.append(f"*(No results found for {ds})*\n")
            continue

        # Collect all models for dense methods
        dense_models_set: Set[str] = set()
        for (method, model), _ in ds_data.items():
            if method in DENSE_METHODS_ORDER:
                dense_models_set.add(model)
        dense_models = sorted(dense_models_set, key=lambda s: s.lower())

        # Build row list: (display_str, values_dict or None)
        rows: List[Tuple[str, Optional[Dict[str, float]]]] = []

        # 1. Lexical methods (one row each, model not shown)
        lexical_seen: Set[str] = set()
        for method in LEXICAL_METHODS:
            for (m, model), vdict in ds_data.items():
                if m == method and method not in lexical_seen:
                    rows.append((METHOD_DISPLAY.get(method, method), vdict))
                    lexical_seen.add(method)
                    break

        # 2. Dense methods grouped by model
        for model in dense_models:
            # Check if this model has any dense methods
            model_methods = {
                m
                for (m, mo), _ in ds_data.items()
                if mo == model and m in DENSE_METHODS_ORDER
            }
            if not model_methods:
                continue
            rows.append((f"**Dense model: {model}**", None))
            for method in DENSE_METHODS_ORDER:
                if method in model_methods:
                    vdict = ds_data.get((method, model), {})
                    rows.append((f"  {METHOD_DISPLAY.get(method, method)}", vdict))

        # 3. Unknown (skip – not expected in practice)

        # Best / second-best per metric column
        from collections import defaultdict

        col_vals: "defaultdict[str, List[Tuple[float, int]]]" = defaultdict(list)
        for i, (_, vdict) in enumerate(rows):
            if vdict is None:
                continue
            for label in METRICS_LABELS:
                v = vdict.get(label)
                if is_number(v):
                    col_vals[label].append((float(v), i))

        best_mark: Dict[Tuple[str, int], str] = {}
        for label, pairs in col_vals.items():
            pairs.sort(reverse=True)
            if pairs:
                best_mark[(label, pairs[0][1])] = "best"
            if len(pairs) > 1:
                best_mark[(label, pairs[1][1])] = "second"

        # Render table
        header = ["Method"] + METRICS_LABELS
        table_lines = [
            " | ".join(header),
            " | ".join(["---"] * len(header)),
        ]
        for i, (display, vdict) in enumerate(rows):
            if vdict is None:
                cells = [display] + [""] * len(METRICS_LABELS)
            else:
                cells = [display]
                for label in METRICS_LABELS:
                    v = vdict.get(label)
                    if not is_number(v):
                        cells.append("—")
                        continue
                    raw = fmt_val(float(v))
                    mark = best_mark.get((label, i))
                    if mark == "best":
                        raw = f"**{raw}**"
                    elif mark == "second":
                        raw = f"<u>{raw}</u>"
                    cells.append(raw)
            table_lines.append(" | ".join(cells))

        parts.append("\n".join(table_lines))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    input_path = "VIRE_Report.md"
    output_path = "VIRE_Report.md"

    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    dataset_order, data = parse_old_report(text)

    print(f"Parsed {len(dataset_order)} datasets: {', '.join(dataset_order)}")
    for ds in dataset_order:
        n = len(data[ds])
        print(f"  {ds}: {n} (method, model) rows")

    new_text = build_new_report(dataset_order, data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"\nWrote converted report to: {output_path}")


if __name__ == "__main__":
    main()
