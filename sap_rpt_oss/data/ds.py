import hashlib
import json
import os
from collections import OrderedDict
from math import ceil
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from sap_rpt_oss.configs import TableRulesConfig
from sap_rpt_oss.data.rules import (
    classify_target_column,
    exceeds_feature_limit,
    filter_table_frames,
    get_target_candidates,
)
from sap_rpt_oss.data.tokenizer import Tokenizer


class TableSkippedError(ValueError):
    pass


def _resolve_int_env(env_var: str, fallback: int) -> int:
    """Return int from env_var if set and parseable, otherwise fallback."""
    raw = os.getenv(env_var)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _resolve_table_rules(
    table_rules: Optional[TableRulesConfig],
    *,
    drop_constant_columns: bool,
    max_num_columns: int,
    max_num_features: Optional[int],
    min_num_rows: int,
    numeric_nan_ratio_threshold: float = 0.5,
    categorical_unique_ratio_threshold: float = 0.2,
) -> TableRulesConfig:
    if table_rules is not None:
        return table_rules

    return TableRulesConfig(
        drop_constant_columns=drop_constant_columns,
        max_num_columns=max_num_columns,
        max_num_features=max_num_features,
        min_num_rows=min_num_rows,
        numeric_nan_ratio_threshold=numeric_nan_ratio_threshold,
        categorical_unique_ratio_threshold=categorical_unique_ratio_threshold,
    )


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
        table_rules: Optional[TableRulesConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.table_rules = _resolve_table_rules(
            table_rules,
            drop_constant_columns=drop_constant_columns,
            max_num_columns=max_num_columns,
            max_num_features=max_num_features,
            min_num_rows=min_num_rows,
        )
        self.drop_constant_columns = self.table_rules.drop_constant_columns
        self.max_num_columns = self.table_rules.max_num_columns
        self.max_num_features = self.table_rules.max_num_features
        self.min_num_rows = self.table_rules.min_num_rows
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
        if exceeds_feature_limit(table, self.max_num_features):
            num_feature_columns = table.shape[1] - 1
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
        fit_df, predict_df = filter_table_frames(
            fit_df,
            predict_df,
            target_column,
            drop_constant_columns=self.drop_constant_columns,
            max_num_columns=self.max_num_columns,
            random_seed=self.random_seed,
        )

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
        auto_select_target: bool = True,
        numeric_nan_ratio_threshold: float = 0.5,
        categorical_unique_ratio_threshold: float = 0.2,
        balance_classification_tasks: bool = False,
        random_seed: int = 42,
        streaming_read_batch_size: Optional[int] = None,
        table_rules: Optional[TableRulesConfig] = None,
        replay_buffer_size: int = 50_000,
        probe_cache_size: int = 100_000,
        discover_parquet: bool = True,
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
        self.table_rules = _resolve_table_rules(
            table_rules,
            drop_constant_columns=drop_constant_columns,
            max_num_columns=max_num_columns,
            max_num_features=max_num_features,
            min_num_rows=min_num_rows,
            numeric_nan_ratio_threshold=numeric_nan_ratio_threshold,
            categorical_unique_ratio_threshold=categorical_unique_ratio_threshold,
        )
        self.drop_constant_columns = self.table_rules.drop_constant_columns
        self.max_num_columns = self.table_rules.max_num_columns
        self.max_num_features = self.table_rules.max_num_features
        self.min_num_rows = self.table_rules.min_num_rows
        self.max_num_rows = max_num_rows
        self.query_size_range = query_size_range
        self.auto_select_target = auto_select_target
        self.numeric_nan_ratio_threshold = (
            self.table_rules.numeric_nan_ratio_threshold
        )
        self.categorical_unique_ratio_threshold = (
            self.table_rules.categorical_unique_ratio_threshold
        )
        self.balance_classification_tasks = balance_classification_tasks
        self.random_seed = random_seed
        self.streaming_read_batch_size = self._resolve_streaming_read_batch_size(
            streaming_read_batch_size
        )
        # Subclasses that read from a manifest (RPTShardDataset) skip the
        # rglob over millions of tiny files.
        if discover_parquet:
            self.parquet_files = sorted(self.root_dir.rglob("*.parquet"))
            if not self.parquet_files:
                raise FileNotFoundError(
                    f"no parquet files found under {self.root_dir}"
                )
        else:
            self.parquet_files = []

        self._regression_task_count = 0
        self._classification_task_count = 0
        self._balance_rng = np.random.default_rng(self.random_seed + 7919)
        # Incremented on every __iter__ so the file order is reshuffled per epoch
        # (with persistent_workers the same worker object is re-iterated each
        # epoch, so this advances correctly).
        self._epoch = -1

        # Cache sizes: env var beats config-passed value beats default.
        self.replay_buffer_size = _resolve_int_env(
            "RPT_REPLAY_BUFFER_SIZE", replay_buffer_size
        )
        self.probe_cache_size = _resolve_int_env(
            "RPT_PROBE_CACHE_SIZE", probe_cache_size
        )

    def _should_skip_for_balance(self, is_regression: bool) -> bool:
        if not self.balance_classification_tasks:
            return False
        total = self._regression_task_count + self._classification_task_count
        if total < 50:
            return False
        if is_regression:
            share = self._regression_task_count / total
        else:
            share = self._classification_task_count / total
        if share <= 0.5:
            return False
        skip_prob = 2.0 * (share - 0.5)
        return bool(self._balance_rng.random() < skip_prob)

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
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            for chunk_idx, record_batch in enumerate(
                parquet_file.iter_batches(batch_size=self.streaming_read_batch_size)
            ):
                table = record_batch.to_pandas()
                if len(table) < self.min_num_rows:
                    continue
                yield chunk_idx, table
        except (OSError, pa.ArrowException) as exc:
            print(f"Skipping parquet file {parquet_path} due to read error: {exc}")
            return

    def _read_probe_table(self, parquet_path: Path) -> Optional[pd.DataFrame]:
        for _, table in self._iter_table_chunks(parquet_path):
            return table
        return None

    def _choose_target_column(self, candidates: list[str], seed: int) -> str:
        rng = np.random.default_rng(seed)
        return candidates[int(rng.integers(0, len(candidates)))]

    def _target_seed(self, table_id: str, salt: int) -> int:
        """Deterministic per-table seed, stable across readers and epochs.

        Seeding target selection by table_id (not file position) makes the
        choice reproducible and identical between the raw-file and shard
        readers, which is what lets the two be checked for parity.
        """
        digest = hashlib.blake2b(
            f"{table_id}:{salt}".encode("utf-8"), digest_size=8
        ).digest()
        return (self.random_seed + int.from_bytes(digest, "big")) % (2**32 - 1)

    def _stream_specs_from_tables(
        self, table_stream: Iterator[tuple[str, str, pd.DataFrame]]
    ) -> Iterator[tuple[dict[str, object], pd.DataFrame]]:
        """Shared auto-select + class-balancing core for both readers.

        ``table_stream`` yields ``(table_id, source_path, table)`` in the order
        to consume (the caller does any per-epoch shuffling). For each table this
        picks at most one regression and one classification target (seeded by
        ``table_id`` so the choice is stable and reader-independent), emits those
        specs, and — when ``balance_classification_tasks`` is on — replays cached
        minority-class tables until the two task counters even out.

        Replays are served from an in-memory LRU keyed by ``table_id`` (capacity
        ``probe_cache_size`` entries); a replay whose table has been evicted is
        skipped (best-effort oversampling). There is no disk re-read on a cache
        miss, which keeps the shard path I/O-free and bounds raw-path memory.
        """
        replay_buffer_size = self.replay_buffer_size
        probe_cache_size = self.probe_cache_size
        replay_rng = np.random.default_rng(self.random_seed + 3)

        # (table_id, source_path, target) — lightweight FIFO history.
        history_reg: list[tuple[str, str, str]] = []
        history_cls: list[tuple[str, str, str]] = []
        probe_cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()

        def cache_put(table_id: str, table: pd.DataFrame) -> None:
            if table_id in probe_cache:
                probe_cache.move_to_end(table_id)
                return
            probe_cache[table_id] = table
            while len(probe_cache) > probe_cache_size:
                probe_cache.popitem(last=False)

        def push_history(
            buf: list, table_id: str, source_path: str, target: str
        ) -> None:
            buf.append((table_id, source_path, target))
            if len(buf) > replay_buffer_size:
                buf.pop(0)

        def replay_one(
            buf: list,
        ) -> Optional[tuple[str, str, str, pd.DataFrame]]:
            if not buf:
                return None
            idx = int(replay_rng.integers(0, len(buf)))
            table_id, source_path, target = buf[idx]
            table = probe_cache.get(table_id)
            if table is None:
                return None
            probe_cache.move_to_end(table_id)
            return table_id, source_path, target, table

        for table_id, source_path, table in table_stream:
            if table is None:
                continue
            cache_put(table_id, table)
            if exceeds_feature_limit(table, self.max_num_features):
                continue

            regression_candidates, classification_candidates = get_target_candidates(
                table,
                numeric_nan_ratio_threshold=self.numeric_nan_ratio_threshold,
                categorical_unique_ratio_threshold=self.categorical_unique_ratio_threshold,
            )

            specs_for_table: list[tuple[str, bool]] = []
            if regression_candidates:
                target = self._choose_target_column(
                    regression_candidates, self._target_seed(table_id, 0)
                )
                specs_for_table.append((target, True))
                push_history(history_reg, table_id, source_path, target)
            if classification_candidates:
                target = self._choose_target_column(
                    classification_candidates, self._target_seed(table_id, 1)
                )
                specs_for_table.append((target, False))
                push_history(history_cls, table_id, source_path, target)

            np.random.default_rng(self._target_seed(table_id, 2)).shuffle(
                specs_for_table
            )

            for target_column, is_regression in specs_for_table:
                spec = {
                    "source_path": source_path,
                    "table_id": table_id,
                    "target_column": target_column,
                    "is_regression": is_regression,
                }
                if is_regression:
                    self._regression_task_count += 1
                else:
                    self._classification_task_count += 1
                yield spec, table

                if not self.balance_classification_tasks:
                    continue

                # Drain replays from the minority side until counters even out
                # (or that side has nothing cached to replay).
                while (
                    self._regression_task_count != self._classification_task_count
                ):
                    minority_is_reg = (
                        self._regression_task_count
                        < self._classification_task_count
                    )
                    minority_history = (
                        history_reg if minority_is_reg else history_cls
                    )
                    replayed = replay_one(minority_history)
                    if replayed is None:
                        break
                    rtable_id, rsource_path, rtarget, rtable = replayed
                    rspec = {
                        "source_path": rsource_path,
                        "table_id": rtable_id,
                        "target_column": rtarget,
                        "is_regression": minority_is_reg,
                    }
                    if minority_is_reg:
                        self._regression_task_count += 1
                    else:
                        self._classification_task_count += 1
                    yield rspec, rtable

    def _stream_auto_target_specs(
        self, parquet_files: list[Path], epoch: int = 0
    ) -> Iterator[tuple[dict[str, object], pd.DataFrame]]:
        """Raw-file auto-select: shuffle file order per epoch, probe each file's
        first chunk, and delegate target selection + balancing to
        :meth:`_stream_specs_from_tables`."""
        order_rng = np.random.default_rng([self.random_seed, epoch])
        shuffled_indices = np.arange(len(parquet_files))
        order_rng.shuffle(shuffled_indices)

        def table_stream() -> Iterator[tuple[str, str, pd.DataFrame]]:
            for shuffled_idx in shuffled_indices:
                parquet_path = parquet_files[int(shuffled_idx)]
                table = self._read_probe_table(parquet_path)
                if table is None:
                    continue
                yield parquet_path.stem, str(parquet_path), table

        yield from self._stream_specs_from_tables(table_stream())

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
                table_rules=self.table_rules,
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

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker_info = get_worker_info()
        parquet_files = self.parquet_files
        if worker_info is not None:
            parquet_files = parquet_files[worker_info.id :: worker_info.num_workers]
            if not parquet_files:
                return

        self._regression_task_count = 0
        self._classification_task_count = 0
        self._epoch += 1

        yielded_any = False
        if self.auto_select_target:
            for spec_idx, (spec, probe_table) in enumerate(
                self._stream_auto_target_specs(parquet_files, epoch=self._epoch)
            ):
                for batch in self._iter_batches_from_table(
                    parquet_path=Path(spec["source_path"]),
                    table=probe_table,
                    target_column=str(spec["target_column"]),
                    is_regression=bool(spec["is_regression"]),
                    seed=self.random_seed + spec_idx * 10_000,
                ):
                    yielded_any = True
                    yield batch
            if not yielded_any and worker_info is None:
                raise ValueError(
                    f"no eligible parquet tables found under {self.root_dir}"
                )
            return

        for file_idx, parquet_path in enumerate(parquet_files):
            for chunk_idx, table in self._iter_table_chunks(parquet_path):
                if exceeds_feature_limit(table, self.max_num_features):
                    continue
                target_column = (
                    self.target_column
                    if self.target_column is not None
                    else str(table.columns[-1])
                )
                task_type = classify_target_column(
                    table[target_column],
                    num_rows=len(table),
                    numeric_nan_ratio_threshold=self.numeric_nan_ratio_threshold,
                    categorical_unique_ratio_threshold=self.categorical_unique_ratio_threshold,
                )
                if task_type == "regression":
                    is_regression = True
                elif task_type == "classification":
                    is_regression = False
                else:
                    # target column is to be discarded -> skip whole table
                    continue
                if self._should_skip_for_balance(is_regression):
                    continue
                if is_regression:
                    self._regression_task_count += 1
                else:
                    self._classification_task_count += 1
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


class RPTShardDataset(RPTParquetDataset):
    """Streaming dataset over repacked Arrow-IPC shards.

    Consumes the ``manifest.json`` produced by ``scripts/repack_t4_shards.py``
    instead of rglob-ing millions of tiny parquet files. Each shard record holds
    one table as an Arrow-IPC ("feather") ``table_bytes`` blob; this reader
    deserializes each blob back into a DataFrame and reuses
    :class:`RPTParquetDataset`'s auto-select + balancing
    (:meth:`_stream_specs_from_tables`) and batch construction
    (:meth:`_iter_batches_from_table`). It always uses auto-select, which is the
    paper-aligned pretraining default.
    """

    def __init__(self, manifest_path: Union[str, Path], *args, **kwargs):
        manifest_path = Path(manifest_path).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"shard manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        self.manifest_path = manifest_path
        self.shards_root = manifest_path.parent
        self.shard_entries = list(manifest.get("shards", []))
        self.blob_format = manifest.get("blob_format", "feather")
        # Reuse the parent config setup but skip the rglob over raw files.
        kwargs["discover_parquet"] = False
        super().__init__(self.shards_root, *args, **kwargs)
        if not self.shard_entries:
            raise FileNotFoundError(f"shard manifest lists no shards: {manifest_path}")
        # Tables buffered for the streaming shuffle (light: holds blobs, not
        # DataFrames). Env-overridable for memory tuning across workers.
        self._shard_shuffle_buffer = _resolve_int_env("RPT_SHARD_SHUFFLE_BUFFER", 2048)

    @staticmethod
    def _shuffle_buffered(
        stream: Iterator, buffer_size: int, rng: np.random.Generator
    ) -> Iterator:
        """Streaming shuffle: sequential input, randomized output, bounded memory.

        Standard reservoir-style shuffle buffer (cf. tf.data.shuffle). Lets the
        shard be read sequentially (good page-cache locality) while still
        presenting tables in a shuffled order to the model.
        """
        if buffer_size <= 1:
            yield from stream
            return
        buf: list = []
        for item in stream:
            if len(buf) < buffer_size:
                buf.append(item)
            else:
                j = int(rng.integers(0, buffer_size))
                yield buf[j]
                buf[j] = item
        # Drain the remainder, shuffled (manual Fisher-Yates — items hold bytes).
        for k in range(len(buf) - 1, 0, -1):
            j = int(rng.integers(0, k + 1))
            buf[k], buf[j] = buf[j], buf[k]
        yield from buf

    def _iter_shard_records(
        self, shard_path: Path
    ) -> Iterator[tuple[str, str, bytes]]:
        """Stream ``(table_id, source_path, blob)`` SEQUENTIALLY from one shard.

        Sequential mmap access preserves read locality; random access into a
        large on-disk shard thrashes the page cache (multi-second stalls).
        Deserialization + shuffling happen downstream, so the shuffle buffer
        holds light blobs rather than DataFrames.
        """
        try:
            reader = pa.ipc.open_file(pa.memory_map(str(shard_path), "r"))
        except (OSError, pa.ArrowException) as exc:
            print(f"Skipping shard {shard_path} due to read error: {exc}")
            return
        for batch_idx in range(reader.num_record_batches):
            record_batch = reader.get_batch(batch_idx)
            table_ids = record_batch.column("table_id").to_pylist()
            source_paths = record_batch.column("source_path").to_pylist()
            blobs = record_batch.column("table_bytes")
            for i in range(record_batch.num_rows):
                blob = blobs[i].as_py()
                if blob is not None:
                    yield table_ids[i], source_paths[i], blob

    def _blob_to_frame(self, table_id: str, blob: bytes) -> Optional[pd.DataFrame]:
        try:
            return pa.ipc.open_file(pa.BufferReader(blob)).read_all().to_pandas()
        except (pa.ArrowException, ValueError) as exc:
            print(f"Skipping table {table_id}: {exc}")
            return None

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker_info = get_worker_info()
        shard_entries = self.shard_entries
        if worker_info is not None:
            shard_entries = shard_entries[worker_info.id :: worker_info.num_workers]
            if not shard_entries:
                return

        self._regression_task_count = 0
        self._classification_task_count = 0
        self._epoch += 1

        # Reshuffle shard order each epoch; a streaming shuffle buffer over the
        # sequentially-read records gives intra-shard shuffling without random
        # mmap access.
        order_rng = np.random.default_rng([self.random_seed, self._epoch])
        order = np.arange(len(shard_entries))
        order_rng.shuffle(order)
        buf_rng = np.random.default_rng([self.random_seed, self._epoch, 13])

        def records() -> Iterator[tuple[str, str, bytes]]:
            for oi in order:
                entry = shard_entries[int(oi)]
                yield from self._iter_shard_records(self.shards_root / entry["path"])

        def table_stream() -> Iterator[tuple[str, str, pd.DataFrame]]:
            for table_id, source_path, blob in self._shuffle_buffered(
                records(), self._shard_shuffle_buffer, buf_rng
            ):
                df = self._blob_to_frame(table_id, blob)
                if df is None or len(df) < self.min_num_rows:
                    continue
                yield table_id, source_path, df

        yielded_any = False
        for spec_idx, (spec, table) in enumerate(
            self._stream_specs_from_tables(table_stream())
        ):
            for batch in self._iter_batches_from_table(
                parquet_path=Path(spec["source_path"]),
                table=table,
                target_column=str(spec["target_column"]),
                is_regression=bool(spec["is_regression"]),
                seed=self.random_seed + spec_idx * 10_000,
            ):
                yielded_any = True
                yield batch

        if not yielded_any and worker_info is None:
            raise ValueError(
                f"no eligible tables found in shards under {self.shards_root}"
            )


class MixedRPTDataset(IterableDataset):
    """Bernoulli-mix two RPTParquetDataset streams.

    Each draw selects the primary dataset with probability `mixing_ratio`,
    otherwise the secondary. Used for paper Appendix A.4 stage-2 curriculum
    where T4 contributes 80% and the Ma et al. data 20%.
    """

    def __init__(
        self,
        primary: "RPTParquetDataset",
        secondary: "RPTParquetDataset",
        mixing_ratio: float = 0.8,
        random_seed: int = 42,
    ):
        super().__init__()
        if not 0.0 <= mixing_ratio <= 1.0:
            raise ValueError(
                f"mixing_ratio must be in [0, 1]; got {mixing_ratio}"
            )
        self.primary = primary
        self.secondary = secondary
        self.mixing_ratio = float(mixing_ratio)
        self.random_seed = random_seed

    def __iter__(self) -> Iterator[dict[str, object]]:
        rng = np.random.default_rng(self.random_seed)
        it_primary = iter(self.primary)
        it_secondary = iter(self.secondary)
        while True:
            pick_primary = rng.random() < self.mixing_ratio
            try:
                if pick_primary:
                    yield next(it_primary)
                else:
                    yield next(it_secondary)
            except StopIteration:
                # fall back to the other stream so training does not stall when
                # one dataset finishes first
                try:
                    yield next(it_secondary if pick_primary else it_primary)
                except StopIteration:
                    return
