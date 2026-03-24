"""
python -m sap_rpt_oss.utils.data_div ./datas/chunk-0000 ./datas/chunk-0001 --output ./datas_classified

python -m sap_rpt_oss.utils.data_div ./datas/chunk-0000 --output ./datas_classified --summary-json ./datas_classified/summary.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - parquet support normally comes from pyarrow
    pq = None


TableKind = Literal["classification", "regression", "unknown"]

DEFAULT_MISSING_RATIO_THRESHOLD = 0.5
DEFAULT_DISCRETE_UNIQUE_RATIO_THRESHOLD = 0.2


@dataclass(slots=True)
class SourceItem:
    source_path: Path
    group_name: str
    relative_path: Path


@dataclass(slots=True)
class ClassificationResult:
    source_path: str
    target_column: str | None
    label: TableKind
    reason: str
    destination_path: str


def first_non_null_value(series: pd.Series):
    non_null = series[series.notna()]
    if non_null.empty:
        return None
    return non_null.iloc[0]


def is_date_like_column(series: pd.Series) -> bool:
    dtype_str = str(series.dtype).lower()
    if any(token in dtype_str for token in ("date", "time", "timestamp")):
        return True

    value = first_non_null_value(series)
    return isinstance(
        value,
        (datetime.date, datetime.time, datetime.datetime, pd.Timestamp),
    )


def coerce_numeric_series(series: pd.Series) -> pd.Series | None:
    if is_bool_dtype(series):
        return None
    if is_numeric_dtype(series):
        return series

    non_null = series.dropna()
    if non_null.empty:
        return None

    converted_non_null = pd.to_numeric(non_null, errors="coerce")
    if converted_non_null.isna().any():
        return None

    return pd.to_numeric(series, errors="coerce")


def is_integer_like_numeric_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool((((non_null % 1).abs()) < 1e-9).all())


def read_last_column(parquet_path: Path) -> tuple[str, pd.Series]:
    if pq is not None:
        schema = pq.read_schema(parquet_path)
        if not schema.names:
            raise ValueError(f"parquet file has no columns: {parquet_path}")
        last_column = str(schema.names[-1])
        frame = pd.read_parquet(parquet_path, columns=[last_column])
        return last_column, frame[last_column]

    frame = pd.read_parquet(parquet_path)
    if frame.shape[1] == 0:
        raise ValueError(f"parquet file has no columns: {parquet_path}")
    last_column = str(frame.columns[-1])
    return last_column, frame.iloc[:, -1]


def classify_last_column(
    series: pd.Series,
    *,
    missing_ratio_threshold: float = DEFAULT_MISSING_RATIO_THRESHOLD,
    discrete_unique_ratio_threshold: float = DEFAULT_DISCRETE_UNIQUE_RATIO_THRESHOLD,
) -> tuple[TableKind, str]:
    num_rows = len(series)
    if num_rows == 0:
        return "unknown", "empty_table"

    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return "unknown", "all_null_target"

    if is_date_like_column(series):
        return "unknown", "date_like_target"

    missing_ratio = float(series.isna().mean())
    if missing_ratio > missing_ratio_threshold:
        return "unknown", "missing_ratio_gt_50_percent"

    unique_count = int(series.nunique(dropna=True))
    unique_ratio = float(unique_count / max(num_rows, 1))
    numeric_series = coerce_numeric_series(series)
    if numeric_series is not None:
        if unique_ratio <= discrete_unique_ratio_threshold:
            return "classification", "numeric_discrete_target"
        if is_integer_like_numeric_series(numeric_series) and unique_count <= 20:
            return "classification", "numeric_integer_low_cardinality_target"
        return "regression", "numeric_continuous_target"

    if unique_ratio <= discrete_unique_ratio_threshold:
        return "classification", "categorical_discrete_target"
    if unique_count < non_null_count and unique_count <= 20:
        return "classification", "categorical_low_cardinality_target"

    return "unknown", "non_numeric_high_cardinality_target"


def iter_source_items(input_paths: Iterable[Path]) -> list[SourceItem]:
    source_items: list[SourceItem] = []
    for raw_path in input_paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {path}")

        if path.is_file():
            if path.suffix.lower() != ".parquet":
                raise ValueError(f"input file must be a parquet file: {path}")
            source_items.append(
                SourceItem(
                    source_path=path,
                    group_name=path.stem,
                    relative_path=Path(path.name),
                )
            )
            continue

        parquet_files = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() == ".parquet"
        )
        if not parquet_files:
            raise ValueError(f"input directory does not contain parquet files: {path}")

        for parquet_path in parquet_files:
            relative_path = parquet_path.relative_to(path)
            if len(relative_path.parts) > 1:
                group_name = relative_path.parts[0]
                destination_relative_path = Path(*relative_path.parts[1:])
            else:
                group_name = path.name
                destination_relative_path = relative_path
            source_items.append(
                SourceItem(
                    source_path=parquet_path,
                    group_name=group_name,
                    relative_path=destination_relative_path,
                )
            )

    return source_items


def build_destination_path(
    output_root: Path,
    item: SourceItem,
    label: TableKind,
) -> Path:
    if label == "classification":
        base_dir = output_root / f"{item.group_name}-classification"
    elif label == "regression":
        base_dir = output_root / f"{item.group_name}-regression"
    else:
        base_dir = output_root / "unknown" / item.group_name
    return base_dir / item.relative_path


def copy_to_destination(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def classify_parquet_file(
    parquet_path: Path,
    *,
    missing_ratio_threshold: float = DEFAULT_MISSING_RATIO_THRESHOLD,
    discrete_unique_ratio_threshold: float = DEFAULT_DISCRETE_UNIQUE_RATIO_THRESHOLD,
) -> tuple[str | None, TableKind, str]:
    try:
        target_column, target_series = read_last_column(parquet_path)
    except Exception as exc:
        return None, "unknown", f"read_error: {exc}"

    label, reason = classify_last_column(
        target_series,
        missing_ratio_threshold=missing_ratio_threshold,
        discrete_unique_ratio_threshold=discrete_unique_ratio_threshold,
    )
    return target_column, label, reason


def classify_inputs(
    input_paths: Iterable[str | Path],
    output_root: str | Path,
    *,
    missing_ratio_threshold: float = DEFAULT_MISSING_RATIO_THRESHOLD,
    discrete_unique_ratio_threshold: float = DEFAULT_DISCRETE_UNIQUE_RATIO_THRESHOLD,
) -> list[ClassificationResult]:
    output_root_path = Path(output_root).expanduser().resolve()
    source_items = iter_source_items(Path(path) for path in input_paths)

    results: list[ClassificationResult] = []
    for item in source_items:
        target_column, label, reason = classify_parquet_file(
            item.source_path,
            missing_ratio_threshold=missing_ratio_threshold,
            discrete_unique_ratio_threshold=discrete_unique_ratio_threshold,
        )
        destination_path = build_destination_path(output_root_path, item, label)
        copy_to_destination(item.source_path, destination_path)
        results.append(
            ClassificationResult(
                source_path=str(item.source_path),
                target_column=target_column,
                label=label,
                reason=reason,
                destination_path=str(destination_path),
            )
        )

    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify parquet tables by the last column and copy them into grouped folders."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input parquet files or directories that contain parquet files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root directory for classified parquet files.",
    )
    parser.add_argument(
        "--missing-ratio-threshold",
        type=float,
        default=DEFAULT_MISSING_RATIO_THRESHOLD,
        help="Columns above this missing ratio are sent to unknown. Default: 0.5",
    )
    parser.add_argument(
        "--discrete-unique-ratio-threshold",
        type=float,
        default=DEFAULT_DISCRETE_UNIQUE_RATIO_THRESHOLD,
        help=(
            "Unique ratio threshold used to distinguish discrete targets from continuous ones. "
            "Default: 0.2"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary of the classification results.",
    )
    return parser


def summarize_results(results: list[ClassificationResult]) -> dict[str, int]:
    summary = {"classification": 0, "regression": 0, "unknown": 0}
    for result in results:
        summary[result.label] += 1
    return summary


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    results = classify_inputs(
        input_paths=args.inputs,
        output_root=args.output,
        missing_ratio_threshold=args.missing_ratio_threshold,
        discrete_unique_ratio_threshold=args.discrete_unique_ratio_threshold,
    )
    summary = summarize_results(results)

    if args.summary_json is not None:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
