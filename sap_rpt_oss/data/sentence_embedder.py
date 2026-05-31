# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from sap_rpt_oss.constants import embedding_model_to_dimension_and_pooling


class SentenceEmbedder:
    def __init__(self, sentence_embedding_model_name, batch_size=512, device=None):
        super().__init__()
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.model = AutoModel.from_pretrained(sentence_embedding_model_name)
        self.embedding_dimension, self.pooling_method = (
            embedding_model_to_dimension_and_pooling[sentence_embedding_model_name]
        )
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model_name)
        if self.pooling_method == "last_token":
            # Qwen3-Embedding officially recommends left-padding so that the last
            # position always corresponds to the final non-pad token.
            self.tokenizer.padding_side = "left"
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def pooling(self, model_output, attention_mask):
        """
        model_output[1] claims to contain the "pooled value", which would
        already have the correct shape, but actually the manual says _not_
        to use it, and instead do the either average or first token [CLS]
        pooling, depending on the model.
        Either case, returns a tensor of shape (batch_size, embedding_dim)
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        if self.pooling_method == "mean":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .type(token_embeddings.dtype)
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        if self.pooling_method == "last_token":
            if self.tokenizer.padding_side == "left":
                return token_embeddings[:, -1].type(token_embeddings.dtype)
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = token_embeddings.size(0)
            return token_embeddings[
                torch.arange(batch_size, device=token_embeddings.device), seq_lengths
            ].type(token_embeddings.dtype)
        assert self.pooling_method == "cls"
        return token_embeddings[:, 0].type(token_embeddings.dtype)

    @torch.no_grad()
    def embed_sentences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        results = []
        for start_idx in range(0, len(input_ids), self.batch_size):
            these_ids = input_ids[start_idx : start_idx + self.batch_size].to(
                self.device
            )
            this_mask = attention_mask[start_idx : start_idx + self.batch_size].to(
                self.device
            )
            model_output = self.model(these_ids, this_mask)
            results.append(self.pooling(model_output, this_mask))
        res = torch.concat(results)
        if res.dtype != self.model.dtype or self.dtype != torch.float16:
            res = res.type(torch.float16)
        return res

    def embed(self, texts: List[str]):
        if not len(texts):
            return []
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        embeddings = self.embed_sentences(encoded.input_ids, encoded.attention_mask)
        embeddings = embeddings.cpu().numpy()
        if self.dtype != torch.float16:
            embeddings = embeddings.astype("float16")
        return embeddings


class RemoteEmbedder:
    """Embed via a TEI (text-embeddings-inference) server using its
    OpenAI-compatible ``/v1/embeddings`` endpoint.

    Drop-in for :class:`SentenceEmbedder`: exposes ``.embed(texts)`` returning a
    ``(len(texts), dim)`` float16 array. The heavy transformer forward runs in
    the TEI service (batched, on GPU), so the data workers are thin HTTP clients
    and the embedding leaves the training process entirely.

    Convention note: TEI follows the model's Sentence-Transformers config, so for
    ``sentence-transformers/all-MiniLM-L6-v2`` it returns mean-pooled **and
    L2-normalized** vectors — unlike the local :class:`SentenceEmbedder` (raw
    mean pooling). Keep training and inference on the same backend.
    """

    def __init__(
        self,
        sentence_embedding_model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        batch_size: int = 128,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.embedding_dimension, self.pooling_method = (
            embedding_model_to_dimension_and_pooling[sentence_embedding_model_name]
        )
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._client_pid: Optional[int] = None

    def _get_client(self):
        # Create the client lazily and per-process so each forked DataLoader
        # worker gets its own httpx connection pool (httpx pools don't survive
        # fork).
        pid = os.getpid()
        if self._client is None or self._client_pid != pid:
            import httpx
            from openai import OpenAI

            # trust_env=False so a local TEI endpoint is NOT routed through an
            # ambient HTTP(S)_PROXY (e.g. clash); proxying localhost returns 502.
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=httpx.Client(trust_env=False, timeout=self.timeout),
            )
            self._client_pid = pid
        return self._client

    def embed(self, texts: List[str]):
        if not len(texts):
            return []
        client = self._get_client()
        out: list = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            # TEI rejects empty strings; substitute a single space so indices
            # still line up with the inputs.
            chunk = [t if t else " " for t in chunk]
            resp = client.embeddings.create(
                model=self.sentence_embedding_model_name, input=chunk
            )
            for item in sorted(resp.data, key=lambda d: d.index):
                out.append(item.embedding)
        return np.asarray(out, dtype=np.float16)
