# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""Gate 1: cell_text_serializer S0/S1/S2 correctness, S2 fit-on-context (no leakage),
missing-value handling, date semantic fields. No GPU / checkpoint needed.

Runs under pytest OR standalone: `python tests/test_cell_text_serializer.py`."""
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sap_rpt_oss.data.cell_text_serializer import (  # noqa: E402
    EMPTY,
    fit_numeric,
    serialize_date,
    serialize_number,
)


@contextlib.contextmanager
def raises(exc):
    """Minimal pytest.raises replacement so the test runs without pytest."""
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} but none was raised")


def test_number_s0_s1():
    assert serialize_number("Age", 42, "s0") == "42"
    assert serialize_number("Age", 42, "s1") == "Age: 42"


def test_number_significant_figures():
    assert serialize_number("x", 42.0000001, "s0") == "42"
    assert serialize_number("x", 3.14159265, "s0") == "3.142"
    assert serialize_number("x", 0.0, "s0") == "0"


def test_s2_fit_on_context_only_no_leakage():
    context = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    state = fit_numeric(context, num_quantile_bins=10)
    assert state["edges"][-1] == 100.0
    assert state["edges"][0] == 10.0

    out = serialize_number("Age", 1000, "s2", quantile_state=state)
    assert out.startswith("Age: ")
    assert "quantile 90-100%" in out
    assert "100" in out and "1000" not in out  # hi clamped to context max

    mid = serialize_number("Age", 55, "s2", quantile_state=state)
    assert mid.startswith("Age: ") and "quantile" in mid


def test_s2_requires_state():
    with raises(ValueError):
        serialize_number("x", 5, "s2", quantile_state=None)


def test_missing_values_map_to_empty():
    state = fit_numeric([1, 2, 3], 3)
    for bad in (None, float("nan"), float("inf"), float("-inf"), np.nan, pd.NA):
        assert serialize_number("x", bad, "s0") == EMPTY
        assert serialize_number("x", bad, "s2", quantile_state=state) == EMPTY
    for bad in (None, pd.NaT, float("nan")):
        assert serialize_date("d", bad, "s0") == EMPTY


def test_fit_numeric_all_missing():
    state = fit_numeric([None, float("nan")], 5)
    assert state["edges"] is None
    assert serialize_number("x", 5, "s2", quantile_state=state).startswith("x: ")


def test_date_s0_s1_s2():
    assert serialize_date("d", "2024-01-15", "s0") == "2024-01-15"
    assert serialize_date("d", "2024-01-15", "s1") == "d: 2024-01-15"
    s2 = serialize_date("d", "2024-01-15", "s2")
    assert "year 2024" in s2
    assert "January" in s2
    assert "Monday" in s2
    assert "Q1" in s2
    assert serialize_date("d", pd.Timestamp("2024-07-04"), "s2").endswith("Q3")


def test_unknown_mode_raises():
    with raises(ValueError):
        serialize_number("x", 1, "s9")
    with raises(ValueError):
        serialize_date("d", "2024-01-01", "s9")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"gate 1: {passed}/{len(fns)} passed")
