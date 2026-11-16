"""Stage 1 MIL head and Sultani ranking loss (CVPR 2018).

Exact approved architecture (Execution plan §3.4, Stage 1 MIL explained §5–6):
  Head:  in_dim (2048) -> 512 (ReLU, Dropout 0.6) -> 32 (Dropout 0.6) -> 1 (Sigmoid)
         1,065,537 parameters (~1.07 M) for in_dim=2048.
  Loss:  mean over all (anomaly, normal) pairs in batch of max(0, 1 - max s_a + max s_n)
       + lambda_smooth * sum_{a, i} (s_{a,i} - s_{a,i+1})^2
       + lambda_sparse * sum_{a, i} s_{a,i}
       (L2 weight decay 1e-3 is applied by Adagrad).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILHead(nn.Module):
    """3-layer Sultani-style MIL MLP applied independently to each segment/clip."""

    def __init__(
        self,
        in_dim: int = 2048,
        hidden1: int = 512,
        hidden2: int = 32,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout

        self.fc1 = nn.Linear(in_dim, hidden1)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [..., in_dim] features to anomaly scores [...] in (0, 1)."""
        h = F.relu(self.fc1(x))
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return torch.sigmoid(self.fc3(h)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SultaniMILLoss(nn.Module):
    """All-pairs Sultani MIL hinge ranking loss with temporal smoothness and sparsity."""

    def __init__(
        self,
        lambda_smooth: float = 8e-5,
        lambda_sparse: float = 8e-5,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.margin = margin

    def forward(
        self,
        anom_scores: torch.Tensor,
        norm_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss and component dict.

        Args:
            anom_scores: [B_a, T] scores in (0, 1) for anomaly videos.
            norm_scores: [B_n, T] scores in (0, 1) for normal videos.
        """
        if anom_scores.ndim != 2 or norm_scores.ndim != 2:
            raise ValueError(
                f"Expected 2D [B, T] score tensors, got {anom_scores.shape} and {norm_scores.shape}"
            )

        max_a = anom_scores.max(dim=1).values  # [B_a]
        max_n = norm_scores.max(dim=1).values  # [B_n]

        # Compare every anomaly video with every normal video in the batch (B_a * B_n pairs)
        pairwise = F.relu(self.margin - max_a.unsqueeze(1) + max_n.unsqueeze(0))
        ranking = pairwise.mean()

        if anom_scores.shape[1] > 1:
            smoothness = self.lambda_smooth * (
                (anom_scores[:, :-1] - anom_scores[:, 1:]) ** 2
            ).sum()
        else:
            smoothness = anom_scores.new_tensor(0.0)

        sparsity = self.lambda_sparse * anom_scores.sum()

        total = ranking + smoothness + sparsity
        parts = {
            "total": float(total.detach().item()),
            "ranking": float(ranking.detach().item()),
            "smoothness": float(smoothness.detach().item()),
            "sparsity": float(sparsity.detach().item()),
            "mean_max_anom": float(max_a.detach().mean().item()),
            "mean_max_norm": float(max_n.detach().mean().item()),
        }
        return total, parts
