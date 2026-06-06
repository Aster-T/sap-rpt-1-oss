"""Diagnose & tune the target-routing and wide-table rules against real t4 data.

CPU-only. Samples a few parquet files per chunk, reads each file's first row
group, and reports — under the OLD rule (numeric->regression always, no class
cap, no width subsample) vs the NEW rule (rules._route_column with the given
thresholds) — how the classification/regression split and the wide-table
handling change. Use it to pick:

  * numeric_as_classification_max_unique -> numeric-cardinality histogram
  * classification_max_classes           -> how many categorical candidates exceed it
  * max_num_columns                       -> column-count distribution + per-cap rates

Usage (env python; CPU only, no GPU):
  /home/amax/.conda/envs/rpt/bin/python scripts/analysis/diagnose_table_rules.py \
      --input-root datasets/t4/datas --limit-chunks 25 --sample-per-chunk 200
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sap_rpt_oss.data.rules import (  # noqa: E402
    _is_date_like_column,
    _is_numeric_column,
    classify_target_column,
)

CARD_BUCKETS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 62, 100, 200, 500, 1000]
_BUCKET_LABELS = [f"<={b}" for b in CARD_BUCKETS] + [f">{CARD_BUCKETS[-1]}"]


def _route_old(column, num_rows, *, numeric_nan, cat_ratio):
    """The original rule: numeric (NaN gate) -> regression; non-numeric with
    unique-ratio <= threshold -> classification. No cardinality awareness."""
    if column.notna().sum() == 0:
        return None
    if _is_date_like_column(column):
        return None
    if _is_numeric_column(column):
        if column.isna().mean() <= numeric_nan:
            return "regression"
        return None
    ratio = column.nunique(dropna=True) / max(num_rows, 1)
    if ratio <= cat_ratio:
        return "classification"
    return None


def _bucket(k):
    for b in CARD_BUCKETS:
        if k <= b:
            return f"<={b}"
    return f">{CARD_BUCKETS[-1]}"


def _hist_str(counter):
    total = sum(counter.values())
    parts = []
    for label in _BUCKET_LABELS:
        c = counter.get(label, 0)
        if c:
            parts.append(f"{label}:{c}({100 * c / max(total, 1):.1f}%)")
    return "  ".join(parts) if parts else "(none)"


def iter_sample_tables(input_root, limit_chunks, sample_per_chunk):
    chunks = sorted(p for p in Path(input_root).glob("chunk-*") if p.is_dir())
    if limit_chunks and len(chunks) > limit_chunks:
        step = max(1, len(chunks) // limit_chunks)
        chunks = chunks[::step][:limit_chunks]
    for cd in chunks:
        n = 0
        try:
            scan = os.scandir(cd)
        except FileNotFoundError:
            continue
        with scan as it:
            for e in it:
                if not e.name.endswith(".parquet"):
                    continue
                try:
                    pf = pq.ParquetFile(e.path)
                    tbl = pf.read_row_group(0) if pf.num_row_groups else pf.read()
                    yield e.path, tbl.to_pandas()
                except Exception:
                    pass
                n += 1
                if n >= sample_per_chunk:
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="datasets/t4/datas")
    ap.add_argument("--limit-chunks", type=int, default=25)
    ap.add_argument("--sample-per-chunk", type=int, default=200)
    ap.add_argument("--numeric-as-classification-max-unique", type=int, default=20)
    ap.add_argument("--classification-max-classes", type=int, default=62)
    ap.add_argument("--numeric-nan-ratio-threshold", type=float, default=0.5)
    ap.add_argument("--categorical-unique-ratio-threshold", type=float, default=0.2)
    ap.add_argument("--caps", default="50,100,128,200,300,500")
    args = ap.parse_args()

    caps = [int(x) for x in args.caps.split(",") if x.strip()]

    n_tables = n_cols = 0
    ncols_list = []
    dtype_kinds = Counter()

    old_reg_tables = old_cls_tables = 0
    new_reg_tables = new_cls_tables = 0
    recovered_lowcard_numeric = 0  # numeric: old regression -> new classification
    cleaned_degenerate_numeric = 0  # numeric: old regression -> new None (k<2)
    garbage_cls_removed = 0  # categorical: old classification -> new None (>cap / single)
    numeric_card_hist = Counter()  # cardinality of every numeric column
    cat_cand_card_hist = Counter()  # cardinality of OLD classification candidates

    for _path, df in iter_sample_tables(
        args.input_root, args.limit_chunks, args.sample_per_chunk
    ):
        if df is None or df.shape[1] == 0 or len(df) == 0:
            continue
        n_tables += 1
        ncols_list.append(df.shape[1])
        num_rows = len(df)
        old_has_reg = old_has_cls = new_has_reg = new_has_cls = False

        for c in df.columns:
            col = df[c]
            n_cols += 1
            dtype_kinds[col.dtype.kind] += 1

            old = _route_old(
                col,
                num_rows,
                numeric_nan=args.numeric_nan_ratio_threshold,
                cat_ratio=args.categorical_unique_ratio_threshold,
            )
            new = classify_target_column(
                col,
                num_rows=num_rows,
                numeric_nan_ratio_threshold=args.numeric_nan_ratio_threshold,
                categorical_unique_ratio_threshold=args.categorical_unique_ratio_threshold,
                numeric_as_classification_max_unique=args.numeric_as_classification_max_unique,
                classification_max_classes=args.classification_max_classes,
            )

            if _is_numeric_column(col) and not _is_date_like_column(col):
                numeric_card_hist[_bucket(int(col.nunique(dropna=True)))] += 1
            if old == "classification":
                cat_cand_card_hist[_bucket(int(col.nunique(dropna=True)))] += 1

            if old == "regression" and new == "classification":
                recovered_lowcard_numeric += 1
            elif old == "regression" and new is None:
                cleaned_degenerate_numeric += 1
            elif old == "classification" and new is None:
                garbage_cls_removed += 1

            old_has_reg |= old == "regression"
            old_has_cls |= old == "classification"
            new_has_reg |= new == "regression"
            new_has_cls |= new == "classification"

        old_reg_tables += old_has_reg
        old_cls_tables += old_has_cls
        new_reg_tables += new_has_reg
        new_cls_tables += new_has_cls

    pct = lambda x: 100 * x / max(n_tables, 1)  # noqa: E731

    print("=" * 78)
    print(
        f"scanned {n_tables} tables, {n_cols} columns  "
        f"(input={args.input_root}, chunks<={args.limit_chunks}, "
        f"per_chunk={args.sample_per_chunk})"
    )
    print(
        f"thresholds: numeric_as_cls<={args.numeric_as_classification_max_unique}  "
        f"cls_max_classes={args.classification_max_classes}  "
        f"numeric_nan<={args.numeric_nan_ratio_threshold}  "
        f"cat_ratio<={args.categorical_unique_ratio_threshold}"
    )
    print(f"dtype kinds (i=int f=float O=object b=bool M=datetime): {dict(dtype_kinds)}")
    print()
    print("--- A. task SUPPLY: tables offering >=1 candidate of each type ---")
    print(
        f"  OLD: reg-tables={old_reg_tables} ({pct(old_reg_tables):.1f}%)   "
        f"cls-tables={old_cls_tables} ({pct(old_cls_tables):.1f}%)"
    )
    print(
        f"  NEW: reg-tables={new_reg_tables} ({pct(new_reg_tables):.1f}%)   "
        f"cls-tables={new_cls_tables} ({pct(new_cls_tables):.1f}%)"
    )
    print()
    print("--- A. column re-routing (OLD -> NEW) ---")
    print(f"  low-card numeric recovered to classification (was regression): {recovered_lowcard_numeric}")
    print(f"  degenerate numeric (k<2) dropped (was regression):             {cleaned_degenerate_numeric}")
    print(f"  categorical dropped: >cap class count or single-value (was cls): {garbage_cls_removed}")
    print()
    print("--- A. numeric-column cardinality (choose numeric_as_classification_max_unique) ---")
    print("  " + _hist_str(numeric_card_hist))
    print("--- A. OLD classification-candidate cardinality (how many exceed the cap) ---")
    print("  " + _hist_str(cat_cand_card_hist))
    print()

    a = np.array(ncols_list)
    print("--- B. column-count distribution (incl. target) ---")
    if len(a):
        pcts = [50, 75, 90, 95, 99, 99.5, 100]
        print(
            "  "
            + "  ".join(f"p{p}={np.percentile(a, p):.0f}" for p in pcts)
            + f"  mean={a.mean():.1f}  max={a.max()}"
        )
    print("--- B. wide-table handling per cap (OLD skipped these; NEW subsamples) ---")
    for cap in caps:
        over = a[a > cap] if len(a) else np.array([])
        rate = 100 * len(over) / max(len(a), 1)
        avg_dropped = (over - cap).mean() if len(over) else 0.0
        print(
            f"  cap={cap:>4}: affected {rate:5.2f}%  "
            f"(NEW subsamples them; avg {avg_dropped:.0f} cols dropped per affected table)"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
