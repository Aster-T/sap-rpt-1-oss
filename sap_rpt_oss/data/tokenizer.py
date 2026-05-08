# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import warnings
from typing import Collection, Literal, Optional, Union

import numpy as np
import pandas as pd
import pyarrow
import torch
from sklearn.preprocessing import StandardScaler

from sap_rpt_oss.constants import (
    QUANTILE_DIMENSION_DEFAULT,
    embedding_model_to_dimension_and_pooling,
)
from sap_rpt_oss.data.sentence_embedder import SentenceEmbedder
from sap_rpt_oss.utils.lru_cache import LRU_Cache

DEFAULT_SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Tokenizer:
    QUANTILE_DIMENSION = QUANTILE_DIMENSION_DEFAULT
    # Kept for backwards-compatibility with older external code that reads the
    # class attribute. The instance attribute (set in __init__) is the source
    # of truth.
    sentence_embedding_model_name = DEFAULT_SENTENCE_EMBEDDING_MODEL
    embedding_dim = embedding_model_to_dimension_and_pooling[
        DEFAULT_SENTENCE_EMBEDDING_MODEL
    ][0]

    def __init__(
        self,
        regression_type: Literal["reg-as-classif", "l2"] = "reg-as-classif",
        classification_type: Literal[
            "cross-entropy", "clustering", "clustering-cosine"
        ] = "cross-entropy",
        num_regression_bins=16,
        random_seed=None,
        is_valid: Optional[bool] = None,
        clip_quantile: float = 0.02,
        sentence_embedding_model_name: str = DEFAULT_SENTENCE_EMBEDDING_MODEL,
        sentence_embedder_device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ):
        self.regression_type = regression_type
        self.classification_type = classification_type
        self.random_seed = random_seed
        self.num_regression_bins = num_regression_bins
        self.verbose = verbose

        if is_valid is not None:
            warnings.warn(
                "Tokenizer(is_valid=...) is deprecated; pass clip_quantile instead. "
                "is_valid=True is equivalent to clip_quantile=0.005 and "
                "is_valid=False to clip_quantile=0.02.",
                DeprecationWarning,
                stacklevel=2,
            )
            clip_quantile = 0.005 if is_valid else 0.02
        self.is_valid = bool(is_valid) if is_valid is not None else False
        if not 0.0 <= clip_quantile < 0.5:
            raise ValueError(f"clip_quantile must be in [0, 0.5); got {clip_quantile}")
        self.clip_quantile = float(clip_quantile)

        if (
            sentence_embedding_model_name
            not in embedding_model_to_dimension_and_pooling
        ):
            raise ValueError(
                f"Unknown sentence embedding model: {sentence_embedding_model_name}. "
                f"Known models: {list(embedding_model_to_dimension_and_pooling)}"
            )
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.embedding_dim = embedding_model_to_dimension_and_pooling[
            sentence_embedding_model_name
        ][0]

        self.sentence_embedder = SentenceEmbedder(
            self.sentence_embedding_model_name,
            device=sentence_embedder_device,
        )
        self.cache = LRU_Cache(max_size=int(os.getenv("LRU_CACHE_SIZE", 10_000_000)))

    def texts_to_tensor(self, texts: Collection[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float16)
        # Make sure we're actually dealing with texts
        texts = [str(x) for x in texts]

        result_dict = {text: self.cache[text] for text in set(texts)}
        missing_texts = [text for text, result in result_dict.items() if result is None]

        if missing_texts:
            missing_embeddings = self.sentence_embedder.embed(missing_texts)

            for text, emb in zip(missing_texts, missing_embeddings):
                self.cache[text] = emb
                result_dict[text] = emb

        results = np.stack([result_dict[text] for text in texts])
        return torch.tensor(results, dtype=torch.float16)

    @staticmethod
    def stable_unique_with_first_indices(
        values: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns unique values in first-occurrence order, together with their positional indices.

        We intentionally avoid np.unique here because it sorts values, which crashes for mixed
        Python scalar types such as floats and strings contained in object-typed target columns.
        """
        is_first_occurrence = ~values.duplicated(keep="first")
        first_indices = np.flatnonzero(is_first_occurrence.to_numpy())
        unique_values = values.iloc[first_indices].to_numpy(dtype=object)
        return unique_values, first_indices

    @staticmethod
    def value_or_nan(values: Union[pd.Series, np.ndarray]):
        if isinstance(values, pd.Series):
            values = values.values
        return torch.tensor(np.where(np.isnan(values), np.array(0), values + 1))

    def standard_scale_column(self, y_train: pd.DataFrame, y_test: pd.DataFrame):
        """
        Expects input to be a dataframe with just one column, and returns the column standardized as flat numpy array,
        plus mean and std as torch tensors.
        TODO: maybe we should return and use torch tensors everywhere.
        """
        train_data = y_train.astype(float).values
        test_data = y_test.astype(float).values

        train_data = np.where(
            np.isfinite(train_data), np.clip(train_data, -1e100, 1e100), np.nan
        )
        test_data = np.where(
            np.isfinite(test_data), np.clip(test_data, -1e100, 1e100), np.nan
        )

        finite_train = train_data[np.isfinite(train_data)]
        if finite_train.size == 0:
            total_rows = len(train_data) + len(test_data)
            labels = np.zeros(total_rows, dtype=np.float32)
            return labels, torch.tensor(0.0), torch.tensor(1.0)

        # Clip outliers between [clip_quantile, 1 - clip_quantile] (paper Section 3.1
        # uses 2% / 98% for pretraining; inference path uses 0.5% / 99.5%).
        q_lo, q_hi = self.clip_quantile, 1.0 - self.clip_quantile
        vmin, vmax = np.nanquantile(train_data, [q_lo, q_hi])
        train_data = np.clip(train_data, vmin, vmax)
        test_data = np.clip(test_data, vmin, vmax)

        fill_value = np.nanmean(train_data)
        if not np.isfinite(fill_value):
            fill_value = float(np.mean(finite_train))
        train_data = np.where(np.isnan(train_data), fill_value, train_data)
        test_data = np.where(np.isnan(test_data), fill_value, test_data)

        scaler = StandardScaler()
        transformed_train_data = scaler.fit_transform(train_data)
        transformed_test_data = scaler.transform(test_data)
        labels = np.concatenate([transformed_train_data, transformed_test_data])[:, 0]
        target_mean = scaler.mean_[0]
        target_std = np.sqrt(scaler.var_)[0]
        return labels, torch.tensor(target_mean), torch.tensor(target_std)

    def quantize_column(self, y_context: pd.DataFrame, y_query: pd.DataFrame):
        """
        Creates num_bins=self.num_regression_bins bins based on column[train_indices] data, and then puts all the column data into them.
        More specifically, for each element in column, it returns:
        - lower_bound_index: in [0, num_bins - 2].
        - delta: in [0, 1] that gives the position in the interval; so the "real" bin is lower_bound_index + delta in [0, num_bins - 1]
        - bin_index: the discretized value in [0, num_bins - 1], which is the class it belongs to for classification loss
        All of the above are returned as numpy arrays of length len(column). Furthermore, this function also returns:
        - quantiles: The num_bins midpoint quantiles of the train data, so that quantiles[bin_index]
            is a piecewise constant approximation of column
        This is done by:
        * Computing n=num_bins symmetric quantiles of a (in positions 1/2n, 3/2n, ..., (2n-1)/2n) and maps b to the
            corresponding quantile intervals (of which there are n-1, also putting anything before 1/2n and after
            (2n-1)/2n in the extreme bins).
        * Values for lower_bound_index and delta are:
            - constant index = 0 and delta = 0 until q0 = q(1/2n)
            - index = 0 and delta linear in [0, 1] from q0 to q1 = q(3/2n)
            - index = 1 and delta linear in [0, 1] from q1 to q2
            - ...
            - index = n-2 and delta linear in [0, 1] from q_{n-2} to q_{n-1}
            - constant index = n-2 and delta = 1 from q_{n-1} to +inf
        It also returns the "real" bin it gets discretized to is int(round(index + delta)).

        Example: if num_bins = 5, we compute the five quantiles 10%, 30%, 50%, 70%, 90%.
        Then the first 10% of data gets index 0 and delta 0, the next 20% gets index 0 and delta linear in [0, 1], etc.,
        and the last 10% gets index 3 and delta 1.
        Remark that even if indices are at most 3, this corresponds to five bins, because the "real" label is
        int(round(index + delta)) which, in this example, is in [0, 1, 2, 3, 4].
        """
        a = y_context.values.flatten()
        b = np.concatenate([a, y_query.values.flatten()])
        num_bins = self.num_regression_bins

        q = np.linspace(
            1 / (2 * num_bins), (2 * num_bins - 1) / (2 * num_bins), num_bins
        )
        quantiles = np.quantile(a, q)
        extended_quantiles = np.concatenate(([np.min(a)], quantiles, [np.max(a)]))

        # Digitize b to find which interval each value belongs to
        indices = np.digitize(b, extended_quantiles) - 1
        # The above values are in [-1, n + 1], because digitize returns 0 or len(extended_quantiles)
        # for values outside the range

        indices = np.clip(indices, 1, num_bins - 1)

        # Compute delta
        lower_bounds = extended_quantiles[indices]
        upper_bounds = extended_quantiles[indices + 1]
        delta = (b - lower_bounds) / np.maximum(upper_bounds - lower_bounds, 1e-10)
        delta = np.clip(delta, 0, 1)

        lower_bound_index = indices - 1
        bin_index = np.round(lower_bound_index + delta).astype(int)

        return lower_bound_index, delta, bin_index, quantiles

    def build_labels(
        self, y_context: pd.DataFrame, y_query: pd.DataFrame, is_clustering=False
    ):
        """
        A bit like OrdinalEncoder fit on train, but for test, if we find an unseen label,
        instead of raising a ValueError, we assign an integer larger than the largest seen label.
        This avoids the (tiny) information leakage that if e.g. we see only labels 0 and 2 in train,
        then we know there is at least one row in test with label 1.
        """
        sorted_value_to_count = y_context.iloc[:, 0].value_counts()
        # get most frequent `self.QUANTILE_DIMENSION - 2` labels and their counts

        if is_clustering is False:
            sorted_value_to_count = sorted_value_to_count.iloc[
                : self.QUANTILE_DIMENSION - 2
            ]

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        shuffled_labels = np.random.permutation(sorted_value_to_count.index)

        label_classes = list(shuffled_labels)
        # labels get indices from 0 to QUANTILE_DIMENSION - 3 (inclusive)
        # index QUANTILE_DIMENSION - 2 is reserved for everything else
        # in masking, those indices are increased by 1 to indicate that 0 is a mask
        # and then QUANTILE_DIMENSION - 1 is the max index
        labels_idx = np.arange(0, len(label_classes))
        label_to_index = {label: idx for label, idx in zip(label_classes, labels_idx)}
        y_concat = pd.concat([y_context, y_query]).values.flatten()
        result = np.asarray(
            [label_to_index.get(y, self.QUANTILE_DIMENSION - 2) for y in y_concat]
        )
        return result, np.asarray(label_classes)

    def time_to_seconds(self, t: Union[datetime.time, pyarrow.time64]):
        if t is None or pd.isna(t):
            return np.nan
        try:
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6
        except (AttributeError, TypeError, ValueError):
            if self.verbose:
                print("Expected time found", type(t), t)
            return np.nan

    def convert_type_(
        self, context_df: pd.DataFrame, query_df: pd.DataFrame, column_name: str
    ):
        dt = str(context_df[column_name].dtype)
        if dt.startswith("time64") or (
            dt == "object"
            and isinstance(context_df[column_name].iloc[0], datetime.time)
        ):
            # time type; either pyarrow.time64[us] (whole column dtype) or datetime.time (column is object)
            context_df[column_name] = context_df[column_name].apply(
                self.time_to_seconds
            )
            query_df[column_name] = query_df[column_name].apply(self.time_to_seconds)
            dt = "float64"
        elif (
            dt.startswith("date")
            or dt.startswith("timestamp")
            or (
                dt == "object"
                and isinstance(context_df[column_name].iloc[0], datetime.date)
            )
        ):
            # date type; either pyarrow.date32[day] (whole column dtype) or datetime.date (column is object)
            context_df[column_name] = pd.to_datetime(
                context_df[column_name], errors="coerce"
            )
            query_df[column_name] = pd.to_datetime(
                query_df[column_name], errors="coerce"
            )
            dt = "datetime64[ns]"
        elif dt.split("[")[0] not in {
            "int64",
            "int32",
            "int16",
            "int8",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
            "float64",
            "float32",
            "float16",
            "datetime64",
            "double",
            "date32",
            "timestamp",
        }:
            if dt not in [
                "bool",
                "str",
                "string[pyarrow]",
                "bool[pyarrow]",
                "category",
                "string",
                "object",
            ]:
                if self.verbose:
                    print(f"Data type {dt} not recognized! Defaulting to string")
            elif dt == "object" and not isinstance(
                context_df[column_name].iloc[0], str
            ):
                is_null = context_df[column_name].isnull()
                if not is_null.iloc[0]:
                    value = context_df[column_name].iloc[0]
                elif is_null.all():
                    if self.verbose:
                        print("Warning, all column is null!")
                    value = "skip_other_warning"
                else:
                    value = context_df[column_name][~is_null].iloc[0]
                if not isinstance(value, str):
                    if self.verbose:
                        print(
                            f"Warning, dtype is object, but first non-null value is {type(value)}. Converting to str."
                        )
            dt = "object"
        return dt

    def process_target(self, data, y_context, y_query, classification_or_regression):
        data["target_delta"] = torch.zeros(
            len(y_context) + len(y_query), dtype=torch.float32
        )
        if classification_or_regression == "regression":
            y_context = y_context.astype(float)
            y_query = y_query.astype(float)

            # Here it's numeric, so we also need to quantize it
            if self.regression_type != "l2":
                labels_lower_bin, delta_labels, labels, label_classes = (
                    self.quantize_column(y_context, y_query)
                )
                data["target"] = torch.tensor(labels_lower_bin)
                # For regression ('reg-as-classif'), we also need the delta,
                # a float in [0, 1]; morally, the "real" target is data['target'] + data['delta']
                data["target_delta"] = torch.tensor(delta_labels, dtype=torch.float32)
            else:
                label_classes = np.zeros(self.QUANTILE_DIMENSION - 2)

            if self.regression_type == "l2":
                labels, _, _ = self.standard_scale_column(y_context, y_query)
                data["target"] = torch.tensor(labels)
        else:
            is_clustering = "clustering" in self.classification_type
            labels_lower_bin, label_classes = self.build_labels(
                y_context, y_query, is_clustering=is_clustering
            )
            labels = labels_lower_bin
            data["target"] = torch.tensor(labels_lower_bin)
            # by default save text embeddings for the target column, inside the model we clear it
            texts = pd.concat([y_context, y_query]).iloc[:, -1]
            data["text_embeddings"][:, -1] = self.texts_to_tensor(texts.astype(str))
            # It could happen that multiple classes have the same embedding, for example if they
            # are only distinguished by casing and our text embedding model is case-insensitive,
            # or maybe if one class is 0 (int) and one is "0" (string) -
            # well, let's hope that doesn't happen, but who knows.
            # In that case, add a prefix to classes and embed again
            unique_classes, unique_indices = self.stable_unique_with_first_indices(
                texts
            )
            should_be_unique_embeddings = data["text_embeddings"][:, -1].numpy()[
                unique_indices
            ]
            unique_label_embeddings = np.unique(should_be_unique_embeddings, axis=0)
            if len(unique_label_embeddings) < len(unique_classes):
                remapped_unique_classes = {
                    c: f"{i}_{c}" for i, c in enumerate(unique_classes)
                }
                modified_column = texts.apply(remapped_unique_classes.get)
                data["text_embeddings"][:, -1] = self.texts_to_tensor(
                    modified_column.astype(str)
                )

        context_size = len(y_context)
        data["text_embeddings"][context_size:, -1] = 0
        data["target"][context_size:] = -100
        data["target_delta"][context_size:] = 0.0
        return data, labels, label_classes

    def replace_inf_values(self, column_values: pd.Series):
        array_values = column_values.values
        if not np.isfinite(array_values).any():
            clipped_values = np.full(array_values.shape, np.nan)
        else:
            max_value = array_values[np.isfinite(array_values)].max()
            min_value = array_values[np.isfinite(array_values)].min()
            clipped_values = np.clip(array_values, min_value - 1, max_value + 1)
        return pd.Series(clipped_values, index=column_values.index)

    def process_features(self, X_context, X_query, data):
        total_length = len(X_context) + len(X_query)
        for column_index, c in enumerate(X_context.columns):
            str_dtype = self.convert_type_(X_context, X_query, c)
            column_values = pd.concat([X_context[c], X_query[c]])

            if str_dtype == "object":
                data["text_embeddings"][:, column_index] = self.texts_to_tensor(
                    column_values.astype(str)
                )
            elif str_dtype.split("[")[0] in ["datetime64", "date32", "timestamp"]:
                data["date_year_month_day_weekday"][:, column_index, 0] = (
                    self.value_or_nan(
                        column_values.dt.year.clip(2000, 2050) - 2000
                    ).int()
                )
                data["date_year_month_day_weekday"][:, column_index, 1] = (
                    self.value_or_nan(column_values.dt.month - 1).int()
                )
                data["date_year_month_day_weekday"][:, column_index, 2] = (
                    self.value_or_nan(column_values.dt.day - 1).int()
                )
                data["date_year_month_day_weekday"][:, column_index, 3] = (
                    self.value_or_nan(column_values.dt.weekday).int()
                )
            else:
                # Switch to float, potentially away from arrow types?
                # Probably not needed.
                del column_values
                context_values = X_context[c].astype(float)
                query_values = X_query[c].astype(float)

                if self.regression_type == "l2":
                    context_values = context_values.replace([np.inf, -np.inf], np.nan)
                    query_values = query_values.replace([np.inf, -np.inf], np.nan)
                    col_mean_value = context_values.mean()
                    context_values = context_values.fillna(value=col_mean_value)
                    query_values = query_values.fillna(value=col_mean_value)
                    col_values_normalized, _, _ = self.standard_scale_column(
                        context_values.to_frame(), query_values.to_frame()
                    )
                    data["number_normalized"][:, column_index] = torch.tensor(
                        col_values_normalized
                    )
                else:
                    column_labels_lower_bin = np.zeros(total_length, dtype=int)
                    column_delta_labels = np.zeros(total_length, dtype=float)
                    # np.quantile doesn't like infinity for some reason.
                    # Since exact values don't matter, we just replace inf with max+1 and -inf with min-1.
                    context_values = self.replace_inf_values(context_values)
                    query_values = self.replace_inf_values(query_values)
                    nan_mask = pd.concat(
                        [context_values.isnull(), query_values.isnull()]
                    ).values
                    (
                        column_labels_lower_bin[~nan_mask],
                        column_delta_labels[~nan_mask],
                        _,
                        _,
                    ) = self.quantize_column(
                        context_values.dropna().to_frame(),
                        query_values.dropna().to_frame(),
                    )
                    # assign nan value to the last bin `QUANTILE_DIMENSION - 1`
                    column_labels_lower_bin[nan_mask] = self.QUANTILE_DIMENSION - 1
                    data["number_percentile_floor"][:, column_index] = torch.tensor(
                        column_labels_lower_bin
                    )
                    data["number_percentile_delta"][:, column_index] = torch.tensor(
                        column_delta_labels
                    )
        return data

    def __call__(
        self,
        X_context: pd.DataFrame,
        y_context: pd.DataFrame,
        X_query: pd.DataFrame,
        y_query: pd.DataFrame,
        classification_or_regression,
    ):

        # Drop columns that are entirely null in the training indices
        # (for numeric columns, it causes a crash; for categorical ones, it might work, but it's likely not worth it)
        X_context = X_context.dropna(axis=1, how="all").copy()
        X_query = X_query[X_context.columns].copy()

        total_length = len(X_context) + len(X_query)
        num_columns = len(X_context.columns) + 1  # Add one for the target column

        data = {
            "column_embeddings": self.texts_to_tensor(
                [str(x) for x in X_context.columns] + [str(y_context.columns[0])]
            ),
            "text_embeddings": torch.zeros(
                (total_length, num_columns, self.embedding_dim), dtype=torch.float16
            ),
            "date_year_month_day_weekday": torch.zeros(
                (total_length, num_columns, 4), dtype=torch.int64
            ),
            "target": torch.zeros(total_length, dtype=torch.float32),
        }
        if self.regression_type == "l2":
            data["number_normalized"] = torch.full(
                (total_length, num_columns), dtype=torch.float32, fill_value=-100
            )
        else:
            data["number_percentile_floor"] = torch.full(
                (total_length, num_columns), dtype=torch.int64, fill_value=-100
            )
            data["number_percentile_delta"] = torch.zeros(
                (total_length, num_columns), dtype=torch.float32
            )

        data, labels, label_classes = self.process_target(
            data, y_context, y_query, classification_or_regression
        )
        data = self.process_features(X_context, X_query, data)

        return data, torch.tensor(labels), label_classes
