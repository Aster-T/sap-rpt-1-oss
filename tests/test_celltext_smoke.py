# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""Gate 4 (from-scratch / no gated checkpoint): end-to-end that the tokenizer emits
`celltext_embeddings` for each mode and a freshly-built RPT consumes it through a
forward pass without error and with the right shapes. Exercises the full (4)+(5)
wiring on CPU. The gated-checkpoint fit/predict smoke needs HF login + ~80GB GPU and
is part of the experiment phase."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sap_rpt_oss.constants import ModelSize  # noqa: E402
from sap_rpt_oss.data.tokenizer import Tokenizer  # noqa: E402
from sap_rpt_oss.model.torch_model import RPT  # noqa: E402

DIM = 384


def _table(n=40):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "income": rng.normal(50000, 15000, n).round(2),
            "signup": pd.Timestamp("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
            "city": rng.choice(["NYC", "LA", "SF"], n),
            "label": rng.integers(0, 2, n),
        }
    )
    X = df.drop(columns=["label"])
    y = df[["label"]]
    return X.iloc[:32], y.iloc[:32], X.iloc[32:], y.iloc[32:]


def _tok(add_cell_text, mode="s2"):
    return Tokenizer(
        regression_type="l2",
        classification_type="cross-entropy",
        random_seed=42,
        sentence_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        sentence_embedder_device="cpu",
        add_cell_text=add_cell_text,
        cell_text_serialization=mode,
        cell_text_bins=10,
        verbose=False,
    )


def test_end_to_end_each_mode():
    torch.manual_seed(0)
    model = RPT(
        ModelSize.tiny,
        regression_type="l2",
        classification_type="cross-entropy",
        add_cell_text=True,
    ).eval()

    tok = _tok(add_cell_text=True)
    for mode in ("s0", "s1", "s2", "placebo"):
        tok.cell_text_serialization = mode
        Xc, yc, Xq, yq = _table()
        ncols = len(Xc.columns) + 1
        total = len(Xc) + len(Xq)
        data, labels, _ = tok(Xc, yc, Xq, yq, "classification")
        assert "celltext_embeddings" in data, f"{mode}: channel missing"
        assert tuple(data["celltext_embeddings"].shape) == (total, ncols, DIM)
        with torch.no_grad():
            out = model(data, is_regression=False, labels=labels)
        assert out is not None
        # the model returns (logits/preds, loss, metric)-like tuple; loss finite
        loss = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        if loss is not None:
            assert torch.isfinite(torch.as_tensor(loss)).all(), f"{mode}: non-finite loss"
        print(f"  ok mode={mode} celltext shape={tuple(data['celltext_embeddings'].shape)}")


def test_off_path_unaffected():
    # add_cell_text=False tokenizer -> no celltext_embeddings; an add_cell_text=True
    # model still runs (the forward block is skipped when the key is absent).
    torch.manual_seed(0)
    model = RPT(
        ModelSize.tiny, regression_type="l2",
        classification_type="cross-entropy", add_cell_text=True,
    ).eval()
    tok = _tok(add_cell_text=False)
    Xc, yc, Xq, yq = _table()
    data, labels, _ = tok(Xc, yc, Xq, yq, "classification")
    assert "celltext_embeddings" not in data
    with torch.no_grad():
        out = model(data, is_regression=False, labels=labels)
    assert out is not None
    print("  ok off-path: no celltext channel, forward runs")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"gate 4: {len(fns)}/{len(fns)} passed")
