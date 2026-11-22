"""Architectures for flow-feature intrusion detection."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain ReLU MLP with BatchNorm. Pruned parameters: all Linear weights."""

    def __init__(self, d_in: int, widths=(512, 512, 256), n_cls: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, d = [], d_in
        for w in widths:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        return self.head(self.body(x))


class CNN1D(nn.Module):
    """1-D CNN over the feature vector (a common NIDS baseline architecture)."""

    def __init__(self, d_in: int, ch=(64, 128), hidden: int = 256, n_cls: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch[0], 3, padding=1), nn.BatchNorm1d(ch[0]), nn.ReLU(),
            nn.Conv1d(ch[0], ch[1], 3, padding=1), nn.BatchNorm1d(ch[1]), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(nn.Linear(ch[1] * (d_in // 2), hidden), nn.BatchNorm1d(hidden),
                                nn.ReLU())
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1)).flatten(1)
        return self.head(self.fc(h))


def build_model(arch: str, d_in: int) -> nn.Module:
    if arch == "mlp":
        return MLP(d_in)
    if arch == "mlp_small":
        return MLP(d_in, widths=(128, 128))
    if arch == "cnn":
        return CNN1D(d_in)
    raise ValueError(arch)


def prunable(model: nn.Module) -> dict[str, nn.Parameter]:
    """Weights subject to pruning: every Linear/Conv weight (biases and BN are kept dense)."""
    return {n: p for n, p in model.named_parameters()
            if n.endswith("weight") and p.dim() > 1}
