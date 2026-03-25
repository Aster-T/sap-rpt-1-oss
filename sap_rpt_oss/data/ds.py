import datetime
from math import ceil
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from sap_rpt_oss.data.tokenizer import Tokenizer


class TableSkippedError(ValueError):
    pass


class RPTTableDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        fit_size: Optional[Union[int, float]],
        is_regression: bool,
        tokenizer: Tokenizer,
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        drop_constant_columns: bool = True,
        max_num_columns: int = 500,
        max_num_features: Optional[int] = None,
        min_num_rows: int = 2,
        max_num_rows: Optional[int] = None,
        query_size_range: Optional[tuple[int, int]] = None,
        random_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.drop_constant_columns = drop_constant_columns
        self.max_num_columns = max_num_columns
        self.max_num_features = max_num_features
        self.min_num_rows = min_num_rows
        self.max_num_rows = max_num_rows
        self.query_size_range = query_size_range
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        (
            self.fit_df,
            self.predict_df,
            self.target_column,
            self.is_regression,
            self.predict_chunk_size,
        ) = self._prepare_table(
            table=table,
            fit_size=fit_size,
            is_regression=is_regression,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
        )

    def __len__(self):
        return ceil(len(self.predict_df) / self.predict_chunk_size)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.predict_chunk_size
        end = min(len(self.predict_df), start + self.predict_chunk_size)
        predict_chunk = self.predict_df.iloc[start:end]

        x_fit = self.fit_df.drop(columns=[self.target_column])
        y_fit = self.fit_df[[self.target_column]]
        x_predict = predict_chunk.drop(columns=[self.target_column])
        y_predict = predict_chunk[[self.target_column]]

        task = "regression" if self.is_regression else "classification"
        data, labels, _ = self.tokenizer(
            x_fit,
            y_fit,
            x_predict,
            y_predict,
            task,
        )
        return {
            "data": data,
            "labels": labels,
            "is_regression": self.is_regression,
        }

    def _next_random_state(self) -> int:
        return int(self.rng.integers(0, 2**32 - 1))

    def _resolve_fit_rows(
        self, num_rows: int, fit_size: Optional[Union[int, float]]
    ) -> int:
        if self.query_size_range is not None:
            min_query_rows, max_query_rows = sorted(
                (int(self.query_size_range[0]), int(self.query_size_range[1]))
            )
            max_query_rows = min(max_query_rows, num_rows - 1)
            min_query_rows = max(1, min(min_query_rows, max_query_rows))
            if min_query_rows > max_query_rows:
                raise ValueError("query_size_range does not fit the current table size")
            query_rows = int(self.rng.integers(min_query_rows, max_query_rows + 1))
            fit_rows = num_rows - query_rows
        else:
            if fit_size is None:
                raise ValueError(
                    "fit_size must be provided when query_size_range is not set"
                )
            if isinstance(fit_size, bool):
                raise ValueError(
                    "fit_size must be an integer row count or a float in (0, 1)"
                )

            if isinstance(fit_size, float):
                if not 0 < fit_size < 1:
                    raise ValueError(
                        "fit_size as a float must be in the interval (0, 1)"
                    )
                fit_rows = int(num_rows * fit_size)
            else:
                fit_rows = int(fit_size)

        if not 1 <= fit_rows < num_rows:
            raise ValueError(
                "fit_size must leave at least one row for both fit and predict"
            )
        return fit_rows

    def _prepare_frames(
        self,
        fit_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        target_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        combined = pd.concat([fit_df, predict_df], ignore_index=True)

        if self.drop_constant_columns:
            features = combined.drop(columns=[target_column])
            constant_columns = list(features.columns[features.nunique() == 1])
            if constant_columns:
                combined = combined.drop(columns=constant_columns)

        if combined.shape[1] > self.max_num_columns:
            features = combined.drop(columns=[target_column])
            sampled_columns = features.sample(
                n=self.max_num_columns - 1,
                axis=1,
                replace=False,
                random_state=self.random_seed,
            )
            combined = pd.concat([sampled_columns, combined[[target_column]]], axis=1)

        fit_rows = len(fit_df)
        return combined.iloc[:fit_rows].copy(), combined.iloc[fit_rows:].copy()

    def _prepare_table(
        self,
        table: pd.DataFrame,
        target_column: Optional[str],
        fit_size: Optional[Union[int, float]],
        is_regression: bool,
        predict_chunk_size: Optional[int],
        shuffle_table: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str, bool, int]:
        if not isinstance(table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")
        if table.shape[1] == 0:
            raise ValueError("table must contain at least one column")
        if len(table) < self.min_num_rows:
            raise ValueError(
                f"table must contain at least {self.min_num_rows} rows, got {len(table)}"
            )
        if target_column is None:
            target_column = str(table.columns[-1])
        if target_column not in table.columns:
            raise ValueError(f"target column '{target_column}' not found in the table")
        if self.max_num_features is not None:
            num_feature_columns = table.shape[1] - 1
            if num_feature_columns > self.max_num_features:
                raise TableSkippedError(
                    f"table has {num_feature_columns} feature columns, "
                    f"exceeds limit {self.max_num_features}"
                )

        working_table = table.copy()
        if self.max_num_rows is not None and len(working_table) > self.max_num_rows:
            working_table = working_table.sample(
                n=self.max_num_rows,
                replace=False,
                random_state=self._next_random_state(),
            ).reset_index(drop=True)
        if shuffle_table:
            working_table = working_table.sample(
                frac=1.0, random_state=self._next_random_state()
            ).reset_index(drop=True)

        fit_rows = self._resolve_fit_rows(len(working_table), fit_size)
        fit_df = working_table.iloc[:fit_rows].copy()
        predict_df = working_table.iloc[fit_rows:].copy()
        fit_df, predict_df = self._prepare_frames(fit_df, predict_df, target_column)

        chunk_size = (
            len(predict_df) if predict_chunk_size is None else int(predict_chunk_size)
        )
        if chunk_size <= 0:
            raise ValueError("predict_chunk_size must be a positive integer")

        return fit_df, predict_df, target_column, is_regression, chunk_size


class RPTParquetDataset(IterableDataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        fit_size: Optional[Union[int, float]],
        tokenizer: Tokenizer,
        target_column: Optional[str] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        drop_constant_columns: bool = True,
        max_num_columns: int = 500,
        max_num_features: Optional[int] = None,
        min_num_rows: int = 2,
        max_num_rows: Optional[int] = None,
        query_size_range: Optional[tuple[int, int]] = None,
        auto_select_target: bool = False,
        skip_ineligible_target: bool = False,
        numeric_nan_ratio_threshold: float = 0.5,
        categorical_unique_ratio_threshold: float = 0.2,
        balance_classification_tasks: bool = False,
        random_seed: int = 42,
        regression_keyword: str = "regression",
        streaming_read_batch_size: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root directory does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"root directory is not a directory: {self.root_dir}"
            )

        self.target_column = target_column
        self.fit_size = fit_size
        self.tokenizer = tokenizer
        self.predict_chunk_size = predict_chunk_size
        self.shuffle_table = shuffle_table
        self.drop_constant_columns = drop_constant_columns
        self.max_num_columns = max_num_columns
        self.max_num_features = max_num_features
        self.min_num_rows = min_num_rows
        self.max_num_rows = max_num_rows
        self.query_size_range = query_size_range
        self.auto_select_target = auto_select_target
        self.skip_ineligible_target = skip_ineligible_target
        self.numeric_nan_ratio_threshold = numeric_nan_ratio_threshold
        self.categorical_unique_ratio_threshold = categorical_unique_ratio_threshold
        self.balance_classification_tasks = balance_classification_tasks
        self.random_seed = random_seed
        self.regression_keyword = regression_keyword.lower()
        self.streaming_read_batch_size = self._resolve_streaming_read_batch_size(
            streaming_read_batch_size
        )
        self.parquet_files = sorted(self.root_dir.rglob("*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"no parquet files found under {self.root_dir}")

    def _infer_is_regression(self, parquet_path: Path) -> bool:
        return self.regression_keyword in parquet_path.as_posix().lower()

    def _exceeds_feature_limit(self, table: pd.DataFrame) -> bool:
        if self.max_num_features is None:
            return False
        return table.shape[1] - 1 > self.max_num_features

    def _resolve_streaming_read_batch_size(
        self, streaming_read_batch_size: Optional[int]
    ) -> int:
        if streaming_read_batch_size is not None:
            batch_size = int(streaming_read_batch_size)
            if batch_size <= 0:
                raise ValueError("streaming_read_batch_size must be a positive integer")
            return batch_size

        inferred_batch_size = self.min_num_rows
        if self.max_num_rows is not None:
            inferred_batch_size = max(inferred_batch_size, int(self.max_num_rows))
        if self.query_size_range is not None:
            inferred_batch_size = max(
                inferred_batch_size,
                max(int(self.query_size_range[0]), int(self.query_size_range[1])) + 1,
            )
        return inferred_batch_size

    def _iter_table_chunks(
        self, parquet_path: Path
    ) -> Iterator[tuple[int, pd.DataFrame]]:
        parquet_file = pq.ParquetFile(parquet_path)
        for chunk_idx, record_batch in enumerate(
            parquet_file.iter_batches(batch_size=self.streaming_read_batch_size)
        ):
            table = record_batch.to_pandas()
            if len(table) < self.min_num_rows:
                continue
            yield chunk_idx, table

    def _read_probe_table(self, parquet_path: Path) -> Optional[pd.DataFrame]:
        for _, table in self._iter_table_chunks(parquet_path):
            return table
        return None

    @staticmethod
    def _first_non_null_value(series: pd.Series):
        non_null = series[series.notna()]
        if non_null.empty:
            return None
        return non_null.iloc[0]

    @classmethod
    def _is_date_like_column(cls, series: pd.Series) -> bool:
        dtype_str = str(series.dtype).lower()
        if any(token in dtype_str for token in ("date", "time", "timestamp")):
            return True

        value = cls._first_non_null_value(series)
        return isinstance(
            value,
            (datetime.date, datetime.time, datetime.datetime, pd.Timestamp),
        )

    @classmethod
    def _is_numeric_column(cls, series: pd.Series) -> bool:
        return is_numeric_dtype(series) and not is_bool_dtype(series)

    def _get_target_candidates(
        self, table: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        regression_candidates = []
        classification_candidates = []
        num_rows = max(len(table), 1)

        for column_name in table.columns:
            column = table[column_name]
            if column.notna().sum() == 0:
                continue
            if self._is_date_like_column(column):
                continue

            if self._is_numeric_column(column):
                if column.isna().mean() <= self.numeric_nan_ratio_threshold:
                    regression_candidates.append(str(column_name))
                continue

            unique_ratio = column.nunique(dropna=True) / num_rows
            if unique_ratio <= self.categorical_unique_ratio_threshold:
                classification_candidates.append(str(column_name))

        return regression_candidates, classification_candidates

    def _is_eligible_target_column(
        self,
        table: pd.DataFrame,
        target_column: str,
        is_regression: bool,
    ) -> bool:
        regression_candidates, classification_candidates = self._get_target_candidates(
            table
        )
        if is_regression:
            return target_column in regression_candidates
        return target_column in classification_candidates

    def _choose_target_column(self, candidates: list[str], seed: int) -> str:
        rng = np.random.default_rng(seed)
        return candidates[int(rng.integers(0, len(candidates)))]

    def _build_auto_target_specs(
        self, parquet_files: list[Path]
    ) -> list[dict[str, object]]:
        regression_specs = []
        classification_specs = []

        for file_idx, parquet_path in enumerate(parquet_files):
            table = self._read_probe_table(parquet_path)
            if table is None:
                continue
            if self._exceeds_feature_limit(table):
                continue

            regression_candidates, classification_candidates = (
                self._get_target_candidates(table)
            )
            if regression_candidates:
                regression_specs.append(
                    {
                        "source_path": parquet_path,
                        "target_column": self._choose_target_column(
                            regression_candidates, self.random_seed + file_idx
                        ),
                        "is_regression": True,
                    }
                )
            if classification_candidates:
                classification_specs.append(
                    {
                        "source_path": parquet_path,
                        "target_column": self._choose_target_column(
                            classification_candidates,
                            self.random_seed + len(parquet_files) + file_idx,
                        ),
                        "is_regression": False,
                    }
                )

        if (
            self.balance_classification_tasks
            and regression_specs
            and classification_specs
            and len(classification_specs) < len(regression_specs)
        ):
            num_extra_specs = len(regression_specs) - len(classification_specs)
            extra_indices = np.random.default_rng(self.random_seed).choice(
                len(classification_specs), size=num_extra_specs, replace=True
            )
            classification_specs.extend(
                [classification_specs[int(index)].copy() for index in extra_indices]
            )

        all_specs = regression_specs + classification_specs
        np.random.default_rng(self.random_seed).shuffle(all_specs)
        return all_specs

    def _iter_batches_from_table(
        self,
        parquet_path: Path,
        table: pd.DataFrame,
        target_column: str,
        is_regression: bool,
        seed: int,
    ) -> Iterator[dict[str, object]]:
        try:
            dataset = RPTTableDataset(
                table=table,
                target_column=target_column,
                fit_size=self.fit_size,
                is_regression=is_regression,
                tokenizer=self.tokenizer,
                predict_chunk_size=self.predict_chunk_size,
                shuffle_table=self.shuffle_table,
                drop_constant_columns=self.drop_constant_columns,
                max_num_columns=self.max_num_columns,
                max_num_features=self.max_num_features,
                min_num_rows=self.min_num_rows,
                max_num_rows=self.max_num_rows,
                query_size_range=self.query_size_range,
                random_seed=seed,
            )
        except TableSkippedError:
            return
        except Exception as exc:
            raise ValueError(
                f"failed to build batches from {parquet_path}: {exc}"
            ) from exc

        try:
            for batch_idx in range(len(dataset)):
                try:
                    batch = dataset[batch_idx]
                except UnicodeDecodeError as exc:
                    print(
                        f"Skipping table {parquet_path} due to UnicodeDecodeError: {exc}"
                    )
                    return
                batch["source_path"] = str(parquet_path)
                batch["target_column"] = target_column
                yield batch
        finally:
            del dataset

    def _iter_batches_for_file(
        self,
        parquet_path: Path,
        target_column: str,
        is_regression: bool,
        seed_offset: int,
    ) -> Iterator[dict[str, object]]:
        for chunk_idx, table in self._iter_table_chunks(parquet_path):
            if self._exceeds_feature_limit(table):
                continue
            if self.skip_ineligible_target and not self._is_eligible_target_column(
                table, target_column, is_regression
            ):
                continue
            yield from self._iter_batches_from_table(
                parquet_path=parquet_path,
                table=table,
                target_column=target_column,
                is_regression=is_regression,
                seed=self.random_seed + seed_offset + chunk_idx,
            )

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker_info = get_worker_info()
        parquet_files = self.parquet_files
        if worker_info is not None:
            parquet_files = parquet_files[worker_info.id :: worker_info.num_workers]
            if not parquet_files:
                return

        yielded_any = False
        if self.auto_select_target:
            for spec_idx, spec in enumerate(
                self._build_auto_target_specs(parquet_files)
            ):
                for batch in self._iter_batches_for_file(
                    parquet_path=Path(spec["source_path"]),
                    target_column=str(spec["target_column"]),
                    is_regression=bool(spec["is_regression"]),
                    seed_offset=spec_idx * 10_000,
                ):
                    yielded_any = True
                    yield batch
        else:
            for file_idx, parquet_path in enumerate(parquet_files):
                for chunk_idx, table in self._iter_table_chunks(parquet_path):
                    if self._exceeds_feature_limit(table):
                        continue
                    target_column = (
                        self.target_column
                        if self.target_column is not None
                        else str(table.columns[-1])
                    )
                    is_regression = self._infer_is_regression(parquet_path)
                    if (
                        self.skip_ineligible_target
                        and not self._is_eligible_target_column(
                            table, target_column, is_regression
                        )
                    ):
                        continue
                    for batch in self._iter_batches_from_table(
                        parquet_path=parquet_path,
                        table=table,
                        target_column=target_column,
                        is_regression=is_regression,
                        seed=self.random_seed + file_idx * 10_000 + chunk_idx,
                    ):
                        yielded_any = True
                        yield batch

        if not yielded_any and worker_info is None:
            raise ValueError(f"no eligible parquet tables found under {self.root_dir}")
