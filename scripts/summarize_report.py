#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan ViRE outputs/ and summarize datasets/models/methods (canonicalized),
optionally generate a Markdown report.

Expected layout:
outputs_root/
  MODEL_NAME/
    DATASET_NAME/
      RUN_FOLDER/
        metrics.json

Example:
outputs/Alibaba-NLP_gte-Qwen2-1_5B-instruct/ALQAC/dense+tfidf-alpha0.70-s1000-.../metrics.json

We canonicalize method from RUN_FOLDER name ONLY, ignoring configs (alpha value, s*, uniq, dedup, llm, etc.).

Canonical methods (current):
- tfidf
- bm25
- splade
- colbert
- dense
- dense+tfidf-alpha
- dense+tfidf-rrf
- dense+bm25-alpha
- dense+bm25-rrf
- unknown

NEW in this version:
- Skip non-run folders like statistics/ and errors_summary/ (and other dot/garbage dirs).
- Detect duplicates per (dataset, model, canonical_method).
- Print unknown folders.
- Robust parsing for patterns like dense+tfidf-alpha0.70-... (alpha glued with number).
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set


# -----------------------------
# Skip non-run folders
# -----------------------------
SKIP_RUN_FOLDERS = {
    "errors_summary",
    "statistics",
    ".ipynb_checkpoints",
    "__pycache__",
}
SKIP_PREFIXES = (".",)  # .DS_Store, .ipynb_checkpoints, ...


# -----------------------------
# Canonicalize metric keys
# -----------------------------
_re_at = re.compile(r"^([a-z\-]+)@(\d+)$", re.IGNORECASE)


def canonicalize_metric_key(name: str) -> str:
    s = name.strip()
    low = s.lower()
    m = _re_at.match(low)
    if m:
        head, k = m.group(1), m.group(2)
        if head in ("p", "precision"):
            return f"Precision@{k}"
        if head in ("r", "recall"):
            return f"Recall@{k}"
        if head == "mrr":
            return f"MRR@{k}"
        if head == "ndcg":
            return f"nDCG@{k}"
        if head == "map":
            return f"MAP@{k}"
        if head in ("hitrate", "hit", "hit-rate"):
            return f"HitRate@{k}"
    if low in ("r-precision", "rprecision"):
        return "R-Precision"
    return s


# -----------------------------
# Helpers
# -----------------------------
def is_number(x) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except Exception:
        return False


def load_metrics_json(run_dir: str) -> Optional[Dict[str, float]]:
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        for k, v in data.items():
            if is_number(v):
                out[str(k)] = float(v)
        return out
    except Exception:
        return None


def fmt_val(v: Optional[float], ndigits: int, as_pct: bool) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "NA"
    return f"{(v * 100 if as_pct else v):.{ndigits}f}{'%' if as_pct else ''}"


# -----------------------------
# Canonicalize method from RUN_FOLDER
# -----------------------------
# Robust regex for "alpha0.70" / "alpha-0.70" / "alpha_0.70" / end-string
_dense_tfidf_alpha = re.compile(r"dense\+tfidf\-alpha(?=[0-9\.\-_]|$)", re.IGNORECASE)
_dense_tfidf_rrf = re.compile(r"dense\+tfidf\-rrf(?=[0-9\.\-_]|$)", re.IGNORECASE)
_dense_bm25_alpha = re.compile(r"dense\+bm25\-alpha(?=[0-9\.\-_]|$)", re.IGNORECASE)
_dense_bm25_rrf = re.compile(r"dense\+bm25\-rrf(?=[0-9\.\-_]|$)", re.IGNORECASE)


def canonicalize_method(run_folder: str) -> str:
    """
    Parse canonical method from a run folder name, ignoring configs.

    Rules (case-insensitive):
    - startswith tfidf => tfidf
    - startswith bm25  => bm25
    - startswith splade => splade
    - startswith colbert => colbert
    - match dense+tfidf-(alpha|rrf) robustly even if "alpha0.70" glued
    - match dense+bm25-(alpha|rrf) robustly even if glued
    - else if contains 'dense' or startswith 'dense' => dense
    - fallback => unknown
    """
    name = run_folder.lower()

    if name.startswith("tfidf"):
        return "tfidf"
    if name.startswith("bm25"):
        return "bm25"
    if name.startswith("splade"):
        return "splade"
    if name.startswith("colbert"):
        return "colbert"

    # dense hybrids (robust)
    if _dense_tfidf_alpha.search(name):
        return "dense+tfidf-alpha"
    if _dense_tfidf_rrf.search(name):
        return "dense+tfidf-rrf"
    if _dense_bm25_alpha.search(name):
        return "dense+bm25-alpha"
    if _dense_bm25_rrf.search(name):
        return "dense+bm25-rrf"

    # dense pure
    if name.startswith("dense") or ("dense" in name):
        return "dense"

    return "unknown"


CANON_METHOD_ORDER = [
    "tfidf",
    "bm25",
    "splade",
    "colbert",
    "dense",
    "dense+tfidf-alpha",
    "dense+tfidf-rrf",
    "dense+bm25-alpha",
    "dense+bm25-rrf",
    "unknown",
]

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

LEXICAL_METHODS = ["tfidf", "bm25", "splade", "colbert"]
DENSE_METHODS_ORDER = [
    "dense",
    "dense+tfidf-alpha",
    "dense+tfidf-rrf",
    "dense+bm25-alpha",
    "dense+bm25-rrf",
]


# -----------------------------
# Scan outputs
# -----------------------------
def scan_outputs(
    outputs_root: str, allowed_datasets: Optional[Set[str]] = None
) -> Tuple[
    Set[str],  # models
    Set[str],  # datasets
    Set[str],  # methods
    Dict[str, Dict[str, Set[str]]],  # ds -> model -> methods (present)
    Dict[Tuple[str, str, str], List[str]],  # (ds, model, method) -> run_dirs
    List[Tuple[str, str, str, str]],  # unknown items: (model, ds, run_folder, run_dir)
]:
    models: Set[str] = set()
    datasets: Set[str] = set()
    methods: Set[str] = set()

    ds_model_methods: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    triple_runs: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    unknown_items: List[Tuple[str, str, str, str]] = []

    if not os.path.isdir(outputs_root):
        return models, datasets, methods, ds_model_methods, triple_runs, unknown_items

    for model_name in sorted(os.listdir(outputs_root)):
        if model_name.startswith("."):
            # skip .ipynb_checkpoints, .DS_Store, ...
            continue

        model_dir = os.path.join(outputs_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        models.add(model_name)

        for dataset_name in sorted(os.listdir(model_dir)):
            if allowed_datasets is not None and dataset_name not in allowed_datasets:
                continue

            ds_dir = os.path.join(model_dir, dataset_name)
            if not os.path.isdir(ds_dir):
                continue
            datasets.add(dataset_name)

            for run_folder in sorted(os.listdir(ds_dir)):
                # skip non-run folders
                if run_folder in SKIP_RUN_FOLDERS or run_folder.startswith(
                    SKIP_PREFIXES
                ):
                    continue

                run_dir = os.path.join(ds_dir, run_folder)
                if not os.path.isdir(run_dir):
                    continue

                # only treat as a run if metrics.json exists
                if not os.path.isfile(os.path.join(run_dir, "metrics.json")):
                    continue

                m = canonicalize_method(run_folder)
                methods.add(m)
                ds_model_methods[dataset_name][model_name].add(m)
                triple_runs[(dataset_name, model_name, m)].append(run_dir)

                if m == "unknown":
                    unknown_items.append(
                        (model_name, dataset_name, run_folder, run_dir)
                    )

    return models, datasets, methods, ds_model_methods, triple_runs, unknown_items


# -----------------------------
# Pick best run per (dataset, model, method)
# -----------------------------
def pick_best_run(
    triple_runs: Dict[Tuple[str, str, str], List[str]],
    metrics_keys: List[str],
    primary_metric: str,
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """
    For each (ds, model, method), there may be many run_dirs.
    We select the run that maximizes primary_metric (if available).
    Ties broken by next metrics in metrics_keys order.
    """
    best: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    order = [primary_metric] + [k for k in metrics_keys if k != primary_metric]

    for key, run_dirs in triple_runs.items():
        candidates = []
        for rd in run_dirs:
            data = load_metrics_json(rd)
            if not data:
                continue
            row: Dict[str, float] = {}
            for mk in metrics_keys:
                v = data.get(mk, None)
                if is_number(v):
                    row[mk] = float(v)
            candidates.append((rd, row))

        if not candidates:
            continue

        def score_tuple(row: Dict[str, float]) -> Tuple:
            return tuple(row.get(mk, float("-inf")) for mk in order)

        best_rd, best_row = max(candidates, key=lambda x: score_tuple(x[1]))
        # store special field for debugging (not used in report)
        best_row["__best_run_dir__"] = best_rd  # type: ignore
        best[key] = best_row

    return best


# -----------------------------
# DUP / UNKNOWN reporting
# -----------------------------
def collect_duplicates(
    triple_runs: Dict[Tuple[str, str, str], List[str]],
) -> List[Tuple[str, str, str, List[str]]]:
    dups = []
    for (ds, model, method), run_dirs in triple_runs.items():
        if method == "unknown":
            continue
        if len(run_dirs) >= 2:
            dups.append((ds, model, method, sorted(run_dirs)))
    dups.sort(key=lambda x: (x[2], x[0], x[1]))
    return dups


def dump_duplicate_details(dups, primary_metric: str, limit_runs_per_dup: int = 200):
    """
    Print each dup cell with all configs and their primary metric values (NA if missing).
    """
    for ds, model, method, run_dirs in dups:
        print(f"[DUP] {model} / {ds} / {method} : {len(run_dirs)} configs")
        rows = []
        for rd in run_dirs[:limit_runs_per_dup]:
            met = load_metrics_json(rd) or {}
            sc = met.get(primary_metric, None)
            sc_str = f"{float(sc):.6f}" if is_number(sc) else "NA"
            rows.append((sc_str, os.path.basename(rd)))

        # sort by score desc, NA last
        def _key(x):
            return (-float(x[0]) if x[0] != "NA" else float("inf"), x[1])

        rows.sort(key=_key)

        for sc_str, folder in rows:
            print(f"      - {sc_str:>10s}  {folder}")
        if len(run_dirs) > limit_runs_per_dup:
            print(f"      ... ({len(run_dirs)-limit_runs_per_dup} more)")
        print("")


# -----------------------------
# Markdown report
# -----------------------------
def build_markdown_report(
    datasets: List[str],
    metrics_labels: List[str],
    metrics_keys: List[str],
    ds_model_methods: Dict[str, Dict[str, Set[str]]],
    best_rows: Dict[Tuple[str, str, str], Dict[str, float]],
    ndigits: int,
    as_pct: bool,
) -> str:
    parts: List[str] = []
    parts.append("# ViRE Retrieval Report\n")
    parts.append(
        f"Metrics: `{', '.join(metrics_labels)}`{' (shown as %)' if as_pct else ''}\n"
    )
    parts.append("## Datasets\n")
    for d in datasets:
        parts.append(f"- [{d}](#{d.lower()})")
    parts.append("")

    for ds in datasets:
        parts.append(f"## {ds}\n")
        sorted_models = sorted(
            ds_model_methods.get(ds, {}).keys(), key=lambda s: s.lower()
        )
        if not sorted_models:
            parts.append(f"*(No results found for {ds})*\n")
            continue

        # ------------------------------------------------------------------
        # Build rows: (display_str, values_dict | None)
        #   values_dict = None  →  group-header row, no metric cells
        # ------------------------------------------------------------------
        rows: List[Tuple[str, Optional[Dict[str, float]]]] = []

        # 1. Lexical (tfidf, bm25, splade, colbert) – first model that has them
        for method in LEXICAL_METHODS:
            for model in sorted_models:
                if method in ds_model_methods[ds].get(model, set()):
                    vdict = best_rows.get((ds, model, method), {})
                    rows.append((METHOD_DISPLAY.get(method, method), vdict))
                    break

        # 2. Dense methods grouped by model
        dense_models = [
            m
            for m in sorted_models
            if any(
                dm in ds_model_methods[ds].get(m, set()) for dm in DENSE_METHODS_ORDER
            )
        ]
        for model in dense_models:
            rows.append((f"**Dense model: {model}**", None))
            for method in DENSE_METHODS_ORDER:
                if method in ds_model_methods[ds].get(model, set()):
                    vdict = best_rows.get((ds, model, method), {})
                    rows.append((f"  {METHOD_DISPLAY.get(method, method)}", vdict))

        # 3. Unknown
        for model in sorted_models:
            if "unknown" in ds_model_methods[ds].get(model, set()):
                vdict = best_rows.get((ds, model, "unknown"), {})
                rows.append(("unknown", vdict))
                break

        # ------------------------------------------------------------------
        # Best / 2nd-best per metric column (skip group-header rows)
        # ------------------------------------------------------------------
        col_pairs: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        for i, (_, vdict) in enumerate(rows):
            if vdict is None:
                continue
            for mk in metrics_keys:
                v = vdict.get(mk)
                if is_number(v):
                    col_pairs[mk].append((float(v), i))

        best_mark: Dict[Tuple[str, int], str] = {}
        for mk, pairs in col_pairs.items():
            pairs.sort(reverse=True)
            if pairs:
                best_mark[(mk, pairs[0][1])] = "best"
            if len(pairs) > 1:
                best_mark[(mk, pairs[1][1])] = "second"

        # ------------------------------------------------------------------
        # Render table
        # ------------------------------------------------------------------
        header = ["Method"] + metrics_labels
        table_lines = [
            " | ".join(header),
            " | ".join(["---"] * len(header)),
        ]

        for i, (display, vdict) in enumerate(rows):
            if vdict is None:
                cells = [display] + [""] * len(metrics_keys)
            else:
                cells = [display]
                for mk in metrics_keys:
                    v = vdict.get(mk)
                    if not is_number(v):
                        cells.append("—")
                        continue
                    raw = fmt_val(float(v), ndigits, as_pct)
                    mark = best_mark.get((mk, i))
                    if mark == "best":
                        raw = f"**{raw}**"
                    elif mark == "second":
                        raw = f"<u>{raw}</u>"
                    cells.append(raw)
            table_lines.append(" | ".join(cells))

        parts.append("\n".join(table_lines))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Summarize ViRE outputs and optionally generate Markdown tables (canonicalized methods)."
    )
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument(
        "--datasets", required=True, help="Comma-separated dataset names (allowlist)"
    )
    ap.add_argument(
        "--metrics",
        default="P@1,R@10,MRR@10,nDCG@10,R@20",
        help="Comma-separated metric names; aliases OK (P@k, R@k, MRR@k, nDCG@k).",
    )
    ap.add_argument(
        "--primary-metric",
        default="nDCG@10",
        help="Primary metric to choose best run among configs within same (dataset, model, method).",
    )
    ap.add_argument("--ndigits", type=int, default=4)
    ap.add_argument("--percent", action="store_true")
    ap.add_argument(
        "--save", default=None, help="If set, write Markdown report to this path."
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, do not write report.",
    )

    # flags
    ap.add_argument(
        "--show-duplicates", action="store_true", help="Print where duplicates exist."
    )
    ap.add_argument(
        "--dump-dup-configs",
        action="store_true",
        help="For each duplicate, dump all config folders + primary-metric for tracing.",
    )
    ap.add_argument(
        "--show-unknown", action="store_true", help="Print unknown run folders."
    )
    ap.add_argument(
        "--unknown-limit",
        type=int,
        default=300,
        help="Max number of unknown folders to print.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    allowed = {d.strip() for d in args.datasets.split(",") if d.strip()}
    metrics_in = [m.strip() for m in args.metrics.split(",") if m.strip()]

    metrics_keys: List[str] = []
    metrics_labels: List[str] = []
    seen = set()
    for m in metrics_in:
        canon = canonicalize_metric_key(m)
        if canon not in seen:
            metrics_keys.append(canon)
            metrics_labels.append(m)
            seen.add(canon)

    primary_metric = canonicalize_metric_key(args.primary_metric)

    (
        models,
        datasets_found,
        methods_found,
        ds_model_methods,
        triple_runs,
        unknown_items,
    ) = scan_outputs(args.outputs_root, allowed_datasets=allowed)

    # -------- SUMMARY --------
    print("=== SUMMARY ===")
    print(f"outputs_root : {args.outputs_root}")
    print(f"Allowed datasets ({len(allowed)}): {sorted(allowed)}")
    print(f"Found models   ({len(models)}): {sorted(models)}")
    print(f"Found datasets ({len(datasets_found)}): {sorted(datasets_found)}")

    # presence counts over dataset×model
    method_counts = {m: 0 for m in CANON_METHOD_ORDER}
    for ds in ds_model_methods:
        for model in ds_model_methods[ds]:
            for m in ds_model_methods[ds][model]:
                method_counts[m] = method_counts.get(m, 0) + 1

    extra = sorted([m for m in methods_found if m not in method_counts])
    if extra:
        for m in extra:
            method_counts[m] = method_counts.get(m, 0)

    print(f"Found canonical methods ({len(methods_found)}): {sorted(methods_found)}")
    print("Method presence counts (over dataset×model):")
    for m in CANON_METHOD_ORDER + extra:
        if m in method_counts:
            print(f"  - {m:18s}: {method_counts[m]}")

    print("\nPer-dataset breakdown (models, methods):")
    for ds in sorted(datasets_found):
        models_here = sorted(ds_model_methods[ds].keys(), key=lambda s: s.lower())
        methods_here = sorted({m for mo in ds_model_methods[ds].values() for m in mo})
        print(f"  * {ds}: models={len(models_here)}, methods={methods_here}")

    # -------- DUPLICATES --------
    dups = collect_duplicates(triple_runs)
    if args.show_duplicates:
        print("\n=== DUPLICATES (same dataset+model+method has >=2 run folders) ===")
        if not dups:
            print("(none)")
        else:
            for ds, model, method, run_dirs in dups[:300]:
                print(f"[DUP] {model} / {ds} / {method} : {len(run_dirs)} configs")
            if len(dups) > 300:
                print(f"... ({len(dups)-300} more)")
        if args.dump_dup_configs and dups:
            print("\n=== DUPLICATE DETAILS (configs + primary metric) ===")
            dump_duplicate_details(dups, primary_metric)

    # -------- UNKNOWN --------
    if args.show_unknown:
        print("\n=== UNKNOWN RUN FOLDERS ===")
        if not unknown_items:
            print("(none)")
        else:
            for model, ds, run_folder, _ in unknown_items[: args.unknown_limit]:
                print(f"[UNKNOWN] {model} / {ds} / {run_folder}")
            if len(unknown_items) > args.unknown_limit:
                print(f"... ({len(unknown_items)-args.unknown_limit} more)")

    # If only summary requested, stop here
    if args.summary_only or not args.save:
        if not args.save:
            print("\n(No --save provided, so report is not written.)")
        return

    # -------- REPORT --------
    best_rows = pick_best_run(triple_runs, metrics_keys, primary_metric)

    md = build_markdown_report(
        datasets=sorted(datasets_found),
        metrics_labels=metrics_labels,
        metrics_keys=metrics_keys,
        ds_model_methods=ds_model_methods,
        best_rows=best_rows,
        ndigits=args.ndigits,
        as_pct=args.percent,
    )

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nSaved Markdown report to: {args.save}")


if __name__ == "__main__":
    main()
