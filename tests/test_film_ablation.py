# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import torch

from sap_rpt_oss.constants import ModelSize
from sap_rpt_oss.model.torch_model import RPT


def _make_dummy_batch(num_rows: int = 8, num_cols: int = 4, embed_dim: int = 384):
    return {
        "column_embeddings": torch.randn(num_cols, embed_dim, dtype=torch.float16),
        "text_embeddings": torch.zeros(
            num_rows, num_cols, embed_dim, dtype=torch.float16
        ),
        "date_year_month_day_weekday": torch.zeros(
            num_rows, num_cols, 4, dtype=torch.long
        ),
        "number_normalized": torch.randn(num_rows, num_cols),
        "target": torch.zeros(num_rows),
        "target_delta": torch.zeros(num_rows),
    }


def test_sum_default_no_film_module():
    m = RPT(
        model_size=ModelSize.mini,
        regression_type="l2",
        classification_type="cross-entropy",
    )
    assert not hasattr(m.embeddings, "film") or m.embeddings.film is None


def test_weight_sharing_active_by_default():
    m = RPT(
        model_size=ModelSize.base,
        regression_type="l2",
        classification_type="cross-entropy",
    )
    ids = {id(layer) for layer in m.in_context_encoder}
    assert len(ids) == 1, "weight sharing not active"
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # base shared model should be far below the unshared 172M figure.
    assert trainable < 60_000_000, f"unexpected trainable param count: {trainable}"


def test_weight_sharing_disabled_yields_independent_layers():
    m = RPT(
        model_size=ModelSize.mini,
        regression_type="l2",
        classification_type="cross-entropy",
        weight_sharing=False,
    )
    ids = {id(layer) for layer in m.in_context_encoder}
    assert len(ids) == m.config.num_hidden_layers


def test_film_init_equivalent_to_sum():
    """FiLM zero-init + column_embeds bias must match sum mode forward output."""
    torch.manual_seed(42)
    m_sum = RPT(
        model_size=ModelSize.mini,
        regression_type="l2",
        classification_type="cross-entropy",
        combination_type="sum",
    )
    m_sum.eval()

    torch.manual_seed(42)
    m_film = RPT(
        model_size=ModelSize.mini,
        regression_type="l2",
        classification_type="cross-entropy",
        combination_type="film",
    )
    m_film.eval()
    m_film.load_state_dict(m_sum.state_dict(), strict=False)

    batch = _make_dummy_batch()
    with torch.no_grad():
        out_sum = m_sum.embeddings(batch, is_regression=True)
        out_film = m_film.embeddings(batch, is_regression=True)
    diff = (out_sum - out_film).abs().max().item()
    print(f"sum vs film init max abs diff: {diff:.2e}")
    assert diff < 1e-5, f"FiLM init not equivalent to sum (max diff={diff})"
