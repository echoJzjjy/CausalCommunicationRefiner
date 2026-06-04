from __future__ import annotations

import torch
import torch.nn.functional as F


class EdgeMLPCritic(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def score_regression_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target) ** 2
    if weight is not None:
        loss = loss * weight
        return loss.sum() / weight.sum().clamp_min(1e-8)
    return loss.mean()


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    if pred.numel() < 2:
        return torch.zeros((), device=pred.device)
    diffs = target.unsqueeze(0) - target.unsqueeze(1)
    pairs = diffs > 0
    if not pairs.any():
        return torch.zeros((), device=pred.device)
    pred_margin = pred.unsqueeze(0) - pred.unsqueeze(1)
    return F.relu(margin - pred_margin[pairs]).mean()
