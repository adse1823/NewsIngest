import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GRAPH_PATH = "./data/graph.pt"
GNN_EMBEDDINGS_PATH = "./data/gnn_embeddings.parquet"
HIDDEN_DIM = 128
OUT_DIM = 64
EPOCHS = 100
LR = 1e-3


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index)


def masked_feature_loss(z: torch.Tensor, x: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
    n, d = x.shape
    mask = torch.rand(n, d) < mask_ratio
    proj = torch.nn.Linear(OUT_DIM, d).to(z.device)
    x_hat = proj(z)
    return F.mse_loss(x_hat[mask], x[mask])


def main():
    data: Data = torch.load(GRAPH_PATH)
    x = data.x
    edge_index = data.edge_index
    tickers = data.tickers

    in_dim = x.shape[1]
    model = GraphSAGE(in_dim, HIDDEN_DIM, OUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    proj = torch.nn.Linear(OUT_DIM, in_dim)
    all_params = list(model.parameters()) + list(proj.parameters())
    optimizer = torch.optim.Adam(all_params, lr=LR)

    log.info("Training GraphSAGE: %d nodes, %d edges, %d epochs", x.shape[0], edge_index.shape[1], EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        n, d = x.shape
        mask = torch.rand(n, d) < 0.2
        x_hat = proj(z)
        loss = F.mse_loss(x_hat[mask], x[mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            log.info("Epoch %d/%d — loss: %.6f", epoch, EPOCHS, loss.item())

    embeddings = model.encode(x, edge_index).numpy()

    df = pd.DataFrame(
        embeddings,
        columns=[f"gnn_dim_{i}" for i in range(OUT_DIM)],
    )
    df.insert(0, "ticker", tickers)
    df.to_parquet(GNN_EMBEDDINGS_PATH, index=False)
    log.info("Saved GNN embeddings %s to %s", embeddings.shape, GNN_EMBEDDINGS_PATH)


if __name__ == "__main__":
    main()
