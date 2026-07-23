import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_embeddings():
    name = "nlp.embeddings"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "nlp" / "embeddings.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EMB = _load_embeddings()


# ── helpers ───────────────────────────────────────────────────────────────────

def _adaptive_tok(batch, **kwargs):
    b = len(batch)
    return {
        "input_ids":      torch.zeros(b, 4, dtype=torch.long),
        "attention_mask": torch.ones(b, 4),
    }


def _adaptive_model(**kwargs):
    b = kwargs["input_ids"].shape[0]
    out = MagicMock()
    out.last_hidden_state = torch.randn(b, 4, 768)
    return out


# ── tests: mean_pool ──────────────────────────────────────────────────────────

def test_mean_pool_output_shape():
    hidden = torch.randn(3, 5, 8)
    mask   = torch.ones(3, 5)
    out = EMB.mean_pool(hidden, mask)
    assert out.shape == (3, 8)


def test_mean_pool_all_ones_mask_equals_token_mean():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # batch=1, seq=2, dim=2
    mask   = torch.ones(1, 2)
    out = EMB.mean_pool(hidden, mask)
    expected = torch.tensor([[2.0, 3.0]])  # (1+3)/2, (2+4)/2
    assert torch.allclose(out, expected)


def test_mean_pool_zero_mask_does_not_divide_by_zero():
    hidden = torch.randn(2, 4, 8)
    mask   = torch.zeros(2, 4)
    out = EMB.mean_pool(hidden, mask)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_mean_pool_partial_mask_ignores_masked_tokens():
    # Only first token valid — output should equal first token's values
    hidden = torch.tensor([[[1.0, 2.0], [9.0, 9.0]]])
    mask   = torch.tensor([[1.0, 0.0]])
    out = EMB.mean_pool(hidden, mask)
    expected = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(out, expected)


# ── tests: embed_texts ────────────────────────────────────────────────────────

def test_embed_texts_output_shape():
    tok = MagicMock(side_effect=_adaptive_tok)
    mod = MagicMock(side_effect=_adaptive_model)
    result = EMB.embed_texts(tok, mod, ["apple earnings", "tesla drops"])
    assert result.shape == (2, 768)


def test_embed_texts_empty_returns_zero_matrix():
    tok = MagicMock()
    mod = MagicMock()
    result = EMB.embed_texts(tok, mod, [])
    assert result.shape == (0, 768)


def test_embed_texts_single_text():
    tok = MagicMock(side_effect=_adaptive_tok)
    mod = MagicMock(side_effect=_adaptive_model)
    result = EMB.embed_texts(tok, mod, ["one headline"])
    assert result.shape == (1, 768)


def test_embed_texts_output_is_numpy_array():
    tok = MagicMock(side_effect=_adaptive_tok)
    mod = MagicMock(side_effect=_adaptive_model)
    result = EMB.embed_texts(tok, mod, ["test"])
    assert isinstance(result, np.ndarray)


def test_embed_texts_batches_correctly():
    n = EMB.BATCH_SIZE + 2
    texts = [f"headline {i}" for i in range(n)]
    tok = MagicMock(side_effect=_adaptive_tok)
    mod = MagicMock(side_effect=_adaptive_model)
    result = EMB.embed_texts(tok, mod, texts)
    assert result.shape == (n, 768)
