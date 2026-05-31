# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""Gate: cardinality-aware target routing (rules._route_column, surfaced via
get_target_candidates / classify_target_column) and wide-table subsampling
(rules.filter_table_frames). No GPU / model / checkpoint needed.

Runs under pytest OR standalone:  python tests/test_target_routing.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sap_rpt_oss.configs import TableRulesConfig  # noqa: E402
from sap_rpt_oss.data.rules import (  # noqa: E402
    classify_target_column,
    filter_table_frames,
    get_target_candidates,
)

# Mirror TableRulesConfig defaults.
NUMERIC_NAN = 0.5
CAT_RATIO = 0.2
NUM_AS_CLS = 20
CLS_MAX = 62


def _route(col, n=None):
    s = pd.Series(col)
    n = len(s) if n is None else n
    return classify_target_column(
        s,
        num_rows=n,
        numeric_nan_ratio_threshold=NUMERIC_NAN,
        categorical_unique_ratio_threshold=CAT_RATIO,
        numeric_as_classification_max_unique=NUM_AS_CLS,
        classification_max_classes=CLS_MAX,
    )


# --- numeric routing (the central fix: low-cardinality numeric -> classification) ---
def test_binary_numeric_is_classification():
    assert _route([0, 1] * 50) == "classification"


def test_small_int_rating_is_classification():
    assert _route([1, 2, 3, 4, 5] * 40) == "classification"


def test_numeric_cutoff_boundary():
    # exactly numeric_as_classification_max_unique distinct -> classification;
    # one more -> regression.
    assert _route(list(range(NUM_AS_CLS)) * 5) == "classification"
    assert _route(list(range(NUM_AS_CLS + 1)) * 5) == "regression"


def test_high_cardinality_numeric_is_regression():
    rng = np.random.default_rng(0)
    assert _route(rng.normal(size=500)) == "regression"


def test_constant_numeric_is_none():
    assert _route([7.0] * 100) is None


def test_high_nan_numeric_is_none():
    # 60% NaN numeric column -> discarded (fails NaN gate).
    assert _route([1.0, 2.0, 3.0, 4.0] + [np.nan] * 6) is None


# --- categorical routing (the new absolute class cap) ---
def test_high_cardinality_categorical_capped_to_none():
    # ratio passes (70/1000 = 0.07 <= 0.2) but absolute class count 70 > 62 ->
    # rejected, so the fixed-width head never sees an overflow-bucket task.
    vals = [f"c{i}" for i in range(70)]
    col = (vals * 15)[:1000]
    assert _route(col, n=1000) is None


def test_categorical_within_cap_is_classification():
    vals = [f"c{i}" for i in range(10)]
    assert _route(vals * 100, n=1000) == "classification"


def test_high_ratio_categorical_is_none():
    # 100 distinct in 100 rows -> ratio 1.0 > 0.2 -> None.
    assert _route([f"u{i}" for i in range(100)], n=100) is None


def test_date_is_none():
    col = pd.to_datetime([f"2020-01-{i % 28 + 1:02d}" for i in range(60)])
    assert _route(col) is None


def test_bool_is_classification():
    assert _route([True, False] * 50) == "classification"


def test_two_value_string_is_classification():
    assert _route(["yes", "no"] * 50) == "classification"


# --- the two public functions must never disagree ---
def test_get_candidates_agrees_with_classify():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {
            "binary_int": rng.integers(0, 2, n),               # cls
            "rating": rng.integers(1, 6, n),                   # cls
            "continuous": rng.normal(size=n),                  # reg
            "many_cat": [f"c{i % 80}" for i in range(n)],      # None (ratio & cap)
            "few_cat": rng.choice(list("abc"), n),             # cls
            "dt": pd.to_datetime("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 100, n), unit="D"),  # None
            "const": np.ones(n),                               # None (k<2)
        }
    )
    reg, cls = get_target_candidates(
        df,
        numeric_nan_ratio_threshold=NUMERIC_NAN,
        categorical_unique_ratio_threshold=CAT_RATIO,
        numeric_as_classification_max_unique=NUM_AS_CLS,
        classification_max_classes=CLS_MAX,
    )
    for c in df.columns:
        t = classify_target_column(
            df[c],
            num_rows=n,
            numeric_nan_ratio_threshold=NUMERIC_NAN,
            categorical_unique_ratio_threshold=CAT_RATIO,
            numeric_as_classification_max_unique=NUM_AS_CLS,
            classification_max_classes=CLS_MAX,
        )
        assert (c in reg) == (t == "regression"), (c, t, "reg-mismatch")
        assert (c in cls) == (t == "classification"), (c, t, "cls-mismatch")
    assert reg == ["continuous"], reg
    assert set(cls) == {"binary_int", "rating", "few_cat"}, cls


# --- wide-table: subsample to cap instead of skipping ---
def test_wide_table_subsampled_to_cap():
    n = 60
    cols = {f"f{i}": np.arange(n) + i for i in range(200)}
    cols["target"] = np.arange(n) % 3
    df = pd.DataFrame(cols)
    out_fit, out_pred = filter_table_frames(
        df.iloc[:40],
        df.iloc[40:],
        "target",
        drop_constant_columns=True,
        max_num_columns=128,
        random_seed=0,
    )
    assert out_fit.shape[1] == 128, out_fit.shape
    assert out_pred.shape[1] == 128, out_pred.shape
    assert "target" in out_fit.columns and "target" in out_pred.columns


def test_narrow_table_not_subsampled():
    n = 60
    df = pd.DataFrame({f"f{i}": np.arange(n) for i in range(10)})
    df["target"] = np.arange(n) % 2
    out_fit, _ = filter_table_frames(
        df.iloc[:40],
        df.iloc[40:],
        "target",
        drop_constant_columns=False,
        max_num_columns=128,
        random_seed=0,
    )
    assert out_fit.shape[1] == 11  # 10 features + target, unchanged
    assert "target" in out_fit.columns


def test_default_config_subsamples_not_skips():
    tr = TableRulesConfig()
    # max_hard_columns None => wide tables are never skipped, only subsampled.
    assert tr.max_hard_columns is None
    assert tr.max_num_columns == 128
    assert tr.classification_max_classes == 62
    assert tr.numeric_as_classification_max_unique == 20


if __name__ == "__main__":
    fns = [
        v
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"target-routing: {passed}/{len(fns)} passed")
