import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_sentiment():
    name = "nlp.sentiment"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "nlp" / "sentiment.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SENT = _load_sentiment()


# ── helpers ───────────────────────────────────────────────────────────────────

def _adaptive_tok(batch, **kwargs):
    b = len(batch)
    return {
        "input_ids":      torch.zeros(b, 4, dtype=torch.long),
        "attention_mask": torch.ones(b, 4, dtype=torch.long),
    }


def _adaptive_model(logit_row=None):
    """Return a model mock whose output adapts to the input batch size."""
    def side_effect(**kwargs):
        b = kwargs["input_ids"].shape[0]
        row = logit_row if logit_row is not None else [2.0, 0.5, 0.5]
        out = MagicMock()
        out.logits = torch.tensor([row] * b, dtype=torch.float)
        return out
    m = MagicMock(side_effect=side_effect)
    return m


# ── tests: batch_inference ────────────────────────────────────────────────────

def test_batch_inference_returns_one_dict_per_text():
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(), ["Apple earnings", "Tesla drops"])
    assert len(results) == 2


def test_batch_inference_dict_has_pos_neg_neu_keys():
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(), ["test headline"])
    assert set(results[0].keys()) == {"pos", "neg", "neu"}


def test_batch_inference_probabilities_sum_to_one():
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(), ["a", "b", "c"])
    for r in results:
        assert abs(r["pos"] + r["neg"] + r["neu"] - 1.0) < 1e-5


def test_batch_inference_probabilities_are_non_negative():
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(), ["x", "y"])
    for r in results:
        assert r["pos"] >= 0 and r["neg"] >= 0 and r["neu"] >= 0


def test_batch_inference_empty_input_returns_empty_list():
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(), [])
    assert results == []


def test_batch_inference_handles_more_texts_than_batch_size():
    n = SENT.BATCH_SIZE + 3
    texts = [f"headline {i}" for i in range(n)]
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(), texts)
    assert len(results) == n


def test_batch_inference_dominant_logit_gives_highest_prob():
    # logits [5, 0, 0] → pos dominates after softmax
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(logit_row=[5.0, 0.0, 0.0]), ["text"])
    assert results[0]["pos"] > results[0]["neg"]
    assert results[0]["pos"] > results[0]["neu"]


def test_batch_inference_negative_logit_gives_highest_neg_prob():
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(logit_row=[0.0, 5.0, 0.0]), ["text"])
    assert results[0]["neg"] > results[0]["pos"]
    assert results[0]["neg"] > results[0]["neu"]


def test_batch_inference_single_text():
    tok = MagicMock(side_effect=_adaptive_tok)
    results = SENT.batch_inference(tok, _adaptive_model(), ["one headline"])
    assert len(results) == 1
    assert abs(results[0]["pos"] + results[0]["neg"] + results[0]["neu"] - 1.0) < 1e-5
