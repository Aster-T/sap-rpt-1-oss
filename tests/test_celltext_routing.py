# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""Gate 3: the celltext hook injects ONLY number/date columns (text/categorical/bool
are skipped — they already go through the content channel), the target column stays
zero, and the placebo channel has the identical non-zero mask. Needs the cached MiniLM
(CPU), no RPT checkpoint. Standalone or pytest."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sap_rpt_oss.data.tokenizer import Tokenizer  # noqa: E402
from sap_rpt_oss.data.tokenizer_hook import (  # noqa: E402
    add_celltext_channel,
    add_placebo_channel,
)

_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        _TOK = Tokenizer(
            regression_type="l2",
            classification_type="cross-entropy",
            random_seed=42,
            sentence_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            sentence_embedder_device="cpu",
            verbose=False,
        )
    return _TOK


def _frames():
    cols = {
        "i": [1, 2, 3, 4],
        "f": [1.5, 2.5, 3.5, 4.5],
        "d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "s": ["a", "b", "c", "d"],
        "b": [True, False, True, False],
    }
    ctx = pd.DataFrame(cols)
    qry = pd.DataFrame(
        {
            "i": [5, 6],
            "f": [5.5, 6.5],
            "d": pd.to_datetime(["2020-05-01", "2020-06-01"]),
            "s": ["e", "f"],
            "b": [True, False],
        }
    )
    return ctx, qry


def test_only_number_and_date_injected():
    tok = _tok()
    ctx, qry = _frames()
    data = {}
    add_celltext_channel(tok, ctx.copy(), qry.copy(), data, mode="s2", num_quantile_bins=4)
    ch = data["celltext_embeddings"]
    # shape: total=6 rows, cols = 5 features + 1 target = 6, dim = 384
    assert ch.shape == (6, 6, tok.embedding_dim), ch.shape
    injected = [j for j in range(ch.shape[1]) if ch[:, j].abs().sum().item() > 0]
    # column order: i(0)=int, f(1)=float, d(2)=date, s(3)=str, b(4)=bool, target(5)
    assert injected == [0, 1, 2], f"expected int/float/date only, got {injected}"
    assert ch[:, -1].abs().sum().item() == 0.0, "target column must stay zero"
    # every non-skipped cell should be non-zero (no missing here)
    assert (ch[:, 0].abs().sum(-1) > 0).all()


def test_s0_s1_also_route_correctly():
    tok = _tok()
    for mode in ("s0", "s1"):
        ctx, qry = _frames()
        data = {}
        add_celltext_channel(tok, ctx.copy(), qry.copy(), data, mode=mode)
        ch = data["celltext_embeddings"]
        injected = [j for j in range(ch.shape[1]) if ch[:, j].abs().sum().item() > 0]
        assert injected == [0, 1, 2], f"mode {mode}: got {injected}"


def test_placebo_same_mask():
    tok = _tok()
    ctx, qry = _frames()
    real, plac = {}, {}
    add_celltext_channel(tok, ctx.copy(), qry.copy(), real, mode="s2", num_quantile_bins=4)
    add_placebo_channel(tok, ctx.copy(), qry.copy(), plac, mode="s2", num_quantile_bins=4)
    real_mask = (real["celltext_embeddings"].abs().sum(-1) > 0)
    plac_mask = (plac["celltext_embeddings"].abs().sum(-1) > 0)
    assert (real_mask == plac_mask).all(), "placebo must share the celltext mask"
    # but the vectors differ (content-independent random)
    assert not (real["celltext_embeddings"] == plac["celltext_embeddings"]).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"gate 3: {len(fns)}/{len(fns)} passed")
