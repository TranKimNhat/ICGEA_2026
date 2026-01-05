"""
GNN.py
=====
Minimal message-passing GNN (no torch_geometric dependency).

Encoder:
  (node_x, edge_index, edge_attr) -> (H_nodes, g_global)

- node_x: [N, F]
- edge_index: [2, E] (src, dst)
- edge_attr: [E, A]

Designed to work with dynamic topology (E varies over time), suitable for PPO/MAPPO.

Dependencies:
- torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, n_hidden: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(n_hidden, 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class GNNConfig:
    node_dim: int
    edge_dim: int
    hidden_dim: int = 128
    msg_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.0


class MPNNEncoder(nn.Module):
    """
    Simple MPNN:
      h0 = node_mlp(node_x)
      m_ij = msg_mlp([h_src, edge_attr])
      agg_j = sum_{i->j} m_ij
      h <- h + update_mlp([h, agg])
    """
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
        self.node_mlp = MLP(cfg.node_dim, cfg.hidden_dim, hidden_dim=cfg.hidden_dim, n_hidden=2, dropout=cfg.dropout)
        self.msg_mlp = MLP(cfg.hidden_dim + cfg.edge_dim, cfg.msg_dim, hidden_dim=cfg.hidden_dim, n_hidden=2, dropout=cfg.dropout)
        self.up_mlp = MLP(cfg.hidden_dim + cfg.msg_dim, cfg.hidden_dim, hidden_dim=cfg.hidden_dim, n_hidden=2, dropout=cfg.dropout)
        self.ln = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          H: [N, hidden_dim]
          g: [hidden_dim] (mean pooled)
        """
        # Ensure types + numerical safety (avoid NaN/Inf poisoning)
        node_x = torch.nan_to_num(node_x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        edge_index = edge_index.long()
        edge_attr = torch.nan_to_num(edge_attr.float(), nan=0.0, posinf=0.0, neginf=0.0)

        N = node_x.shape[0]
        if N == 0:
            H0 = torch.zeros((0, self.cfg.hidden_dim), device=node_x.device, dtype=node_x.dtype)
            g0 = torch.zeros((self.cfg.hidden_dim,), device=node_x.device, dtype=node_x.dtype)
            return H0, g0
        H = self.node_mlp(node_x)  # [N, d]
        H = torch.nan_to_num(self.ln(H), nan=0.0, posinf=0.0, neginf=0.0)

        if edge_index.numel() == 0:
            g = H.mean(dim=0)
            return H, g

        src = edge_index[0]
        dst = edge_index[1]

        for _ in range(self.cfg.n_layers):
            msg_in = torch.cat([H[src], edge_attr], dim=-1)  # [E, d + A]
            M = self.msg_mlp(msg_in)                         # [E, msg_dim]

            # aggregate by destination
            agg = torch.zeros((N, self.cfg.msg_dim), device=H.device, dtype=H.dtype)
            agg.index_add_(0, dst, M)

            up_in = torch.cat([H, agg], dim=-1)
            H = H + self.up_mlp(up_in)  # residual
            H = torch.nan_to_num(self.ln(H), nan=0.0, posinf=0.0, neginf=0.0)

        g = H.mean(dim=0)
        return H, g
