import importlib.util
import sys
from pathlib import Path

import torch
import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_train_gnn():
    name = "graph.train_gnn"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "graph" / "train_gnn.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GNN = _load_train_gnn()


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    """5-node chain graph with 768-dim features (matches FinBERT output)."""
    x = torch.randn(5, 768)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return x, edge_index


@pytest.fixture
def model():
    return GNN.GraphSAGE(
        in_channels=768,
        hidden_channels=GNN.HIDDEN_DIM,
        out_channels=GNN.OUT_DIM,
    )


# ── tests: GraphSAGE.forward ──────────────────────────────────────────────────

def test_forward_output_shape(model, small_graph):
    x, edge_index = small_graph
    model.train()
    out = model(x, edge_index)
    assert out.shape == (5, GNN.OUT_DIM)


def test_forward_disconnected_graph():
    m = GNN.GraphSAGE(in_channels=4, hidden_channels=8, out_channels=2)
    x = torch.randn(3, 4)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = m(x, edge_index)
    assert out.shape == (3, 2)


def test_forward_single_node():
    m = GNN.GraphSAGE(in_channels=4, hidden_channels=8, out_channels=2)
    x = torch.randn(1, 4)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = m(x, edge_index)
    assert out.shape == (1, 2)


def test_forward_output_is_float_tensor(model, small_graph):
    x, edge_index = small_graph
    out = model(x, edge_index)
    assert out.dtype == torch.float32


# ── tests: GraphSAGE.encode ───────────────────────────────────────────────────

def test_encode_output_shape(model, small_graph):
    x, edge_index = small_graph
    out = model.encode(x, edge_index)
    assert out.shape == (5, GNN.OUT_DIM)


def test_encode_no_grad(model, small_graph):
    x, edge_index = small_graph
    out = model.encode(x, edge_index)
    assert not out.requires_grad


def test_encode_sets_eval_mode(model, small_graph):
    x, edge_index = small_graph
    model.train()
    model.encode(x, edge_index)
    assert not model.training


def test_encode_is_deterministic(model, small_graph):
    x, edge_index = small_graph
    out1 = model.encode(x, edge_index)
    out2 = model.encode(x, edge_index)
    assert torch.allclose(out1, out2)


# ── tests: masked_feature_loss ────────────────────────────────────────────────

def test_masked_feature_loss_returns_scalar():
    torch.manual_seed(0)
    z = torch.randn(10, GNN.OUT_DIM)
    x = torch.randn(10, 768)
    loss = GNN.masked_feature_loss(z, x)
    assert loss.shape == torch.Size([])


def test_masked_feature_loss_is_non_negative():
    torch.manual_seed(1)
    z = torch.randn(10, GNN.OUT_DIM)
    x = torch.randn(10, 768)
    loss = GNN.masked_feature_loss(z, x)
    assert loss.item() >= 0


def test_masked_feature_loss_all_masked_still_scalar():
    torch.manual_seed(2)
    z = torch.randn(10, GNN.OUT_DIM)
    x = torch.zeros(10, 10)
    loss = GNN.masked_feature_loss(z, x, mask_ratio=1.0)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_masked_feature_loss_custom_in_dim():
    torch.manual_seed(3)
    z = torch.randn(5, GNN.OUT_DIM)
    x = torch.randn(5, 32)  # smaller feature dim
    loss = GNN.masked_feature_loss(z, x)
    assert loss.item() >= 0
