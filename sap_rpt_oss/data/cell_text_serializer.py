# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""Serialize number / date cells to text for the per-cell text injection channel.

Implements the S0 -> S1 -> S2 verbalization ladder (the ablation itself):

    S0  raw            "42"
    S1  colname+value  "Age: 42"
    S2  quantile       "Age: 38-45 (quantile 50-60%)"   (TabSTAR-style)

Pure functions, no torch — unit-testable without a GPU/checkpoint. Only `number`
and `date` cells are serialized here; text/categorical/bool already go through the
existing content channel and must NOT be injected (see tokenizer_hook).

Missing / non-finite values map to the EMPTY sentinel; the data hook skips encoding
them so their cell keeps a zero vector. For S2, quantile edges are fit on the
**context (train) rows only** and then applied to context+query — no leakage.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Sentinel for "do not encode this cell" — the hook leaves a zero vector.
EMPTY = ""

MODES = ("s0", "s1", "s2")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_missing(value) -> bool:
    """True for None / NaN / inf / pandas-NA / NaT."""
    if value is None:
        return True
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return True
    except (TypeError, ValueError):
        pass
    try:
        result = pd.isna(value)
        # pd.isna on an array-like returns an array; treat only scalars here.
        if isinstance(result, bool):
            return result
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _fmt_num(x: float, sig: int = 4) -> str:
    """Format a number with `sig` significant figures, trimming noise like
    42.0000001 -> "42" while keeping enough precision. Returns EMPTY if non-finite."""
    if x is None or not math.isfinite(x):
        return EMPTY
    if x == 0:
        return "0"
    formatted = f"{x:.{sig}g}"
    # normalize "-0" and exponent casing
    if formatted in ("-0", "-0.0"):
        return "0"
    return formatted


# --------------------------------------------------------------------------- #
# numeric
# --------------------------------------------------------------------------- #
def fit_numeric(context_values: Sequence, num_quantile_bins: int = 10) -> dict:
    """Fit S2 quantile edges on the CONTEXT (train) finite values only.

    Returns a state dict consumed by `serialize_number(..., quantile_state=state)`.
    Fitting on context-only and reusing for query is what prevents leakage.
    """
    finite = []
    for v in context_values:
        if _is_missing(v):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            finite.append(fv)
    if not finite:
        edges = None
    else:
        arr = np.asarray(finite, dtype=float)
        edges = np.quantile(arr, np.linspace(0.0, 1.0, num_quantile_bins + 1))
        edges = np.asarray(edges, dtype=float)
    return {"edges": edges, "bins": int(num_quantile_bins)}


def _quantile_bucket(value: float, state: dict):
    """Return (lo, hi, p0, p1) — the value's quantile bin from a fitted state."""
    edges = state.get("edges")
    bins = int(state.get("bins", 10))
    if edges is None or len(edges) < 2:
        return value, value, 0, 100
    # bin index in [0, bins-1]; values outside the fitted range clamp to the ends.
    idx = int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, bins - 1))
    lo = float(edges[idx])
    hi = float(edges[idx + 1])
    p0 = int(round(100.0 * idx / bins))
    p1 = int(round(100.0 * (idx + 1) / bins))
    return lo, hi, p0, p1


def serialize_number(
    col: str, value, mode: str, quantile_state: Optional[dict] = None
) -> str:
    if _is_missing(value):
        return EMPTY
    try:
        v = float(value)
    except (TypeError, ValueError):
        return EMPTY
    if not math.isfinite(v):
        return EMPTY

    if mode == "s0":
        return _fmt_num(v)
    if mode == "s1":
        return f"{col}: {_fmt_num(v)}"
    if mode == "s2":
        if quantile_state is None:
            raise ValueError("mode 's2' requires quantile_state from fit_numeric()")
        lo, hi, p0, p1 = _quantile_bucket(v, quantile_state)
        return f"{col}: {_fmt_num(lo)}-{_fmt_num(hi)} (quantile {p0}-{p1}%)"
    raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")


# --------------------------------------------------------------------------- #
# date
# --------------------------------------------------------------------------- #
def serialize_date(col: str, value, mode: str) -> str:
    if _is_missing(value):
        return EMPTY
    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return EMPTY

    iso = ts.strftime("%Y-%m-%d")
    if mode == "s0":
        return iso
    if mode == "s1":
        return f"{col}: {iso}"
    if mode == "s2":
        quarter = (int(ts.month) - 1) // 3 + 1
        # %B = full month name, %A = full weekday name (locale C / English)
        return (
            f"{col}: year {int(ts.year)}, {ts.strftime('%B')}, "
            f"{ts.strftime('%A')}, Q{quarter}"
        )
    raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")
