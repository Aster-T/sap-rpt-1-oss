import datetime
from typing import Optional

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


def get_target_candidates(
    table: pd.DataFrame,
    *,
    numeric_nan_ratio_threshold: float,
    categorical_unique_ratio_threshold: float,
) -> tuple[list[str], list[str]]:
    regression_candidates = []
    classification_candidates = []
    num_rows = max(len(table), 1)

    for column_name in table.columns:
        column = table[column_name]
        if column.notna().sum() == 0:
            continue
        if _is_date_like_column(column):
            continue

        if _is_numeric_column(column):
            if column.isna().mean() <= numeric_nan_ratio_threshold:
                regression_candidates.append(str(column_name))
            continue

        unique_ratio = column.nunique(dropna=True) / num_rows
        if unique_ratio <= categorical_unique_ratio_threshold:
            classification_candidates.append(str(column_name))

    return regression_candidates, classification_candidates


def is_eligible_target_column(
    table: pd.DataFrame,
    target_column: str,
    is_regression: bool,
    *,
    numeric_nan_ratio_threshold: float,
    categorical_unique_ratio_threshold: float,
) -> bool:
    regression_candidates, classification_candidates = get_target_candidates(
        table,
        numeric_nan_ratio_threshold=numeric_nan_ratio_threshold,
        categorical_unique_ratio_threshold=categorical_unique_ratio_threshold,
    )
    if is_regression:
        return target_column in regression_candidates
    return target_column in classification_candidates


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
