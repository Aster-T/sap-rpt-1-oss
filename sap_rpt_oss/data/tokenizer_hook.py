# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""Data-side hook that builds the per-cell text channel `data['celltext_embeddings']`.

Only `number` and `date` cells are injected (text/categorical/bool already go through
the existing content channel and are skipped). dtype routing **reuses the tokenizer's
own `convert_type_`** so it agrees with the content channel. The MiniLM encoder is
reused via `tokenizer.texts_to_tensor` (LRU-cached); only non-empty serializations are
encoded, so missing cells keep a zero vector. The target column (last) is never injected.

`add_placebo_channel` produces the same SHAPE and the same non-zero MASK but with
random, content-independent vectors — the control that isolates "extra parameters /
extra non-zero inputs" from genuine semantic gain.
"""
from typing import Optional

import numpy as np
import torch

from sap_rpt_oss.data.cell_text_serializer import (
    EMPTY,
    fit_numeric,
    serialize_date,
    serialize_number,
)


def _route(str_dtype: str) -> str:
    """Map a tokenizer dtype string to a celltext route: skip / date / number."""
    if str_dtype == "object":
        return "skip"  # text / categorical / bool -> already in content channel
    base = str_dtype.split("[")[0]
    if base in ("datetime64", "date32", "timestamp", "date", "datetime"):
        return "date"
    return "number"


def _serialize_column(col, X_context, X_query, kind, mode, num_quantile_bins) -> list:
    ctx = list(X_context[col])
    qry = list(X_query[col])
    if kind == "date":
        return [serialize_date(col, v, mode) for v in ctx + qry]
    state = fit_numeric(ctx, num_quantile_bins=num_quantile_bins) if mode == "s2" else None
    return [serialize_number(col, v, mode, state) for v in ctx + qry]


def _empty_channel(tokenizer, X_context, X_query) -> torch.Tensor:
    total = len(X_context) + len(X_query)
    num_columns = len(X_context.columns) + 1  # + target
    return torch.zeros((total, num_columns, tokenizer.embedding_dim), dtype=torch.float16)


def add_celltext_channel(
    tokenizer,
    X_context,
    X_query,
    data,
    mode: str = "s2",
    num_quantile_bins: int = 10,
):
    channel = _empty_channel(tokenizer, X_context, X_query)
    for col_index, col in enumerate(X_context.columns):
        str_dtype = tokenizer.convert_type_(X_context, X_query, col)
        kind = _route(str_dtype)
        if kind == "skip":
            continue
        strs = _serialize_column(col, X_context, X_query, kind, mode, num_quantile_bins)
        nonempty = [i for i, s in enumerate(strs) if s != EMPTY]
        if not nonempty:
            continue
        vecs = tokenizer.texts_to_tensor([strs[i] for i in nonempty])  # (k, dim) fp16
        idx = torch.as_tensor(nonempty, dtype=torch.long)
        channel[idx, col_index] = vecs.type(channel.dtype)
    channel[:, -1] = 0  # never inject the target column
    data["celltext_embeddings"] = channel
    return data


def add_placebo_channel(
    tokenizer,
    X_context,
    X_query,
    data,
    mode: str = "s2",
    num_quantile_bins: int = 10,
):
    channel = _empty_channel(tokenizer, X_context, X_query)
    seed = (getattr(tokenizer, "random_seed", None) or 0) + 911
    rng = np.random.default_rng(seed)
    dim = tokenizer.embedding_dim
    for col_index, col in enumerate(X_context.columns):
        str_dtype = tokenizer.convert_type_(X_context, X_query, col)
        kind = _route(str_dtype)
        if kind == "skip":
            continue
        strs = _serialize_column(col, X_context, X_query, kind, mode, num_quantile_bins)
        nonempty = [i for i, s in enumerate(strs) if s != EMPTY]
        if not nonempty:
            continue
        rand = rng.standard_normal((len(nonempty), dim)).astype("float16")
        idx = torch.as_tensor(nonempty, dtype=torch.long)
        channel[idx, col_index] = torch.from_numpy(rand)
    channel[:, -1] = 0
    data["celltext_embeddings"] = channel
    return data
