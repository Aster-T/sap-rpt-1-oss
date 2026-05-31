import datetime
from typing import Literal, Optional

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def filter_table_frames(
    fit_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    target_column: str,
    *,
    drop_constant_columns: bool,
    max_num_columns: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([fit_df, predict_df], ignore_index=True)

    if drop_constant_columns:
        feature_columns = combined.drop(columns=[target_column])
        constant_columns = list(
            feature_columns.columns[feature_columns.nunique() == 1]
        )
        if constant_columns:
            combined = combined.drop(columns=constant_columns)

    if combined.shape[1] > max_num_columns:
        feature_columns = combined.drop(columns=[target_column])
        sampled_columns = feature_columns.sample(
            n=max_num_columns - 1,
            axis=1,
            replace=False,
            random_state=random_seed,
        )
        combined = pd.concat([sampled_columns, combined[[target_column]]], axis=1)

    fit_rows = len(fit_df)
    return combined.iloc[:fit_rows].copy(), combined.iloc[fit_rows:].copy()


def exceeds_feature_limit(
    table: pd.DataFrame, max_num_features: Optional[int]
) -> bool:
    if max_num_features is None:
        return False
    return table.shape[1] - 1 > max_num_features


def _route_column(
    column: pd.Series,
    num_rows: int,
    *,
    numeric_nan_ratio_threshold: float,
    categorical_unique_ratio_threshold: float,
    numeric_as_classification_max_unique: int = 20,
    classification_max_classes: int = 62,
) -> Optional[Literal["regression", "classification"]]:
    """Decide a single column's task type, or ``None`` to discard it.

    Cardinality-aware routing — the single source of truth shared by
    ``get_target_candidates`` and ``classify_target_column`` so the two can
    never disagree:

      * all-null / date-like -> None
      * numeric (non-bool): must pass the NaN gate, then route by distinct
        count ``k``:
          - ``k < 2``                                     -> None (degenerate)
          - ``k <= numeric_as_classification_max_unique`` -> "classification"
            Low-cardinality numeric is almost always a categorical target
            encoded as integers (0/1 labels, 1-5 ratings, status codes). The
            old rule sent ALL numeric columns to regression, which both
            polluted the regression pool with discrete targets and starved the
            classification head of these natural samples.
          - else                                          -> "regression"
            (``k`` is necessarily ``> cutoff`` here, so a degenerate 2-value
            numeric is never used as a regression target)
      * non-numeric: classification iff
        ``2 <= k <= classification_max_classes`` AND
        ``k / num_rows <= categorical_unique_ratio_threshold``. The absolute
        cap (default = ``QUANTILE_DIMENSION - 2``) excludes columns whose class
        count overflows the fixed-width classification head, where the tokenizer
        would otherwise collapse everything past the top-62 into one bucket.
    """
    if column.notna().sum() == 0:
        return None
    if _is_date_like_column(column):
        return None

    if _is_numeric_column(column):
        if column.isna().mean() > numeric_nan_ratio_threshold:
            return None
        k = int(column.nunique(dropna=True))
        if k < 2:
            return None
        if k <= numeric_as_classification_max_unique:
            return "classification"
        return "regression"

    k = int(column.nunique(dropna=True))
    if k < 2 or k > classification_max_classes:
        return None
    if k / max(num_rows, 1) <= categorical_unique_ratio_threshold:
        return "classification"
    return None


def get_target_candidates(
    table: pd.DataFrame,
    *,
    numeric_nan_ratio_threshold: float,
    categorical_unique_ratio_threshold: float,
    numeric_as_classification_max_unique: int = 20,
    classification_max_classes: int = 62,
) -> tuple[list[str], list[str]]:
    """Enumerate ``(regression_candidates, classification_candidates)``.

    Each column is routed by :func:`_route_column`; it lands in at most one list.
    """
    regression_candidates = []
    classification_candidates = []
    num_rows = max(len(table), 1)

    for column_name in table.columns:
        task = _route_column(
            table[column_name],
            num_rows,
            numeric_nan_ratio_threshold=numeric_nan_ratio_threshold,
            categorical_unique_ratio_threshold=categorical_unique_ratio_threshold,
            numeric_as_classification_max_unique=numeric_as_classification_max_unique,
            classification_max_classes=classification_max_classes,
        )
        if task == "regression":
            regression_candidates.append(str(column_name))
        elif task == "classification":
            classification_candidates.append(str(column_name))

    return regression_candidates, classification_candidates


def classify_target_column(
    column: pd.Series,
    num_rows: int,
    *,
    numeric_nan_ratio_threshold: float,
    categorical_unique_ratio_threshold: float,
    numeric_as_classification_max_unique: int = 20,
    classification_max_classes: int = 62,
) -> Optional[Literal["regression", "classification"]]:
    """O(1)-cost variant of `get_target_candidates` that inspects a single column.

    Returns "regression", "classification", or None (column is to be discarded).
    Delegates to :func:`_route_column` so it always agrees with
    `get_target_candidates`.
    """
    return _route_column(
        column,
        num_rows,
        numeric_nan_ratio_threshold=numeric_nan_ratio_threshold,
        categorical_unique_ratio_threshold=categorical_unique_ratio_threshold,
        numeric_as_classification_max_unique=numeric_as_classification_max_unique,
        classification_max_classes=classification_max_classes,
    )


def _first_non_null_value(series: pd.Series):
    non_null = series[series.notna()]
    if non_null.empty:
        return None
    return non_null.iloc[0]


def _is_date_like_column(series: pd.Series) -> bool:
    dtype_str = str(series.dtype).lower()
    if any(token in dtype_str for token in ("date", "time", "timestamp")):
        return True

    value = _first_non_null_value(series)
    return isinstance(
        value,
        (datetime.date, datetime.time, datetime.datetime, pd.Timestamp),
    )


def _is_numeric_column(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)
