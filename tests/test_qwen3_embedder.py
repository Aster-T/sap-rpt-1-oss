# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


@pytest.mark.skipif(
    not os.environ.get("RUN_QWEN3_TEST"),
    reason="set RUN_QWEN3_TEST=1 to actually download Qwen3 (~1.2GB)",
)
def test_qwen3_embed_shape():
    from sap_rpt_oss.data.sentence_embedder import SentenceEmbedder

    se = SentenceEmbedder("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    emb = se.embed(["hello world", "tabular learning"])
    assert emb.shape == (2, 1024)


@pytest.mark.skipif(
    not os.environ.get("RUN_QWEN3_TEST"),
    reason="set RUN_QWEN3_TEST=1 to actually download Qwen3 (~1.2GB)",
)
def test_qwen3_rpt_construction():
    from sap_rpt_oss.constants import ModelSize
    from sap_rpt_oss.data.tokenizer import Tokenizer
    from sap_rpt_oss.model.torch_model import RPT

    tok = Tokenizer(
        sentence_embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        sentence_embedder_device="cpu",
    )
    assert tok.embedding_dim == 1024

    m = RPT(
        model_size=ModelSize.mini,
        sentence_embedding_dim=tok.embedding_dim,
    )
    # adapter shape should be (hidden=256, 1024)
    assert m.embeddings.column_remapping.weight.shape == (256, 1024)
    assert m.embeddings.content_remapping.weight.shape == (256, 1024)
