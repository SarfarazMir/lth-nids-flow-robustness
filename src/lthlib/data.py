"""Data loading, splitting and preprocessing for flow-feature NIDS datasets.

Preprocessing is a monotone per-feature map  z = (sign(x)*log1p|x| - mu) / sd  fitted on the
SOURCE training split only. Because the map is strictly increasing per feature, attack
constraints stated in raw feature space ("the attacker may only increase feature j", "stay
inside the observed range") translate exactly into constraints in model-input space.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "data_cache")


def load_frame(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(CACHE, f"{name}.parquet"))


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("label", "y")]


@dataclass
class Split:
    X: np.ndarray  # float32, raw features
    y: np.ndarray  # int64
    label: np.ndarray  # original string labels (for per-class analysis)


def stratified_split(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed: int = 0) -> dict[str, Split]:
    """Stratified (by original label) random split into train/val/test."""
    rng = np.random.default_rng(seed)
    feats = feature_names(df)
    idx = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("label", sort=True):
        ii = rng.permutation(g.index.to_numpy())
        n_tr = int(round(fracs[0] * len(ii)))
        n_va = int(round(fracs[1] * len(ii)))
        idx["train"].append(ii[:n_tr])
        idx["val"].append(ii[n_tr:n_tr + n_va])
        idx["test"].append(ii[n_tr + n_va:])
    out = {}
    for k, parts in idx.items():
        ii = rng.permutation(np.concatenate(parts))
        sub = df.loc[ii]
        out[k] = Split(sub[feats].to_numpy(np.float32), sub["y"].to_numpy(np.int64),
                       sub["label"].to_numpy())
    return out


def subsample(split: Split, frac: float, seed: int) -> Split:
    """Stratified (binary) subsample used for the low-label target regime."""
    if frac >= 1.0:
        return split
    rng = np.random.default_rng(seed)
    keep = []
    for c in (0, 1):
        ii = np.flatnonzero(split.y == c)
        keep.append(rng.choice(ii, max(1, int(round(frac * len(ii)))), replace=False))
    ii = rng.permutation(np.concatenate(keep))
    return Split(split.X[ii], split.y[ii], split.label[ii])


class Preprocessor:
    def __init__(self):
        self.mu = self.sd = self.lo = self.hi = None

    @staticmethod
    def _slog(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X: np.ndarray) -> "Preprocessor":
        L = self._slog(X.astype(np.float64))
        self.mu = L.mean(0)
        self.sd = L.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (L - self.mu) / self.sd
        self.lo, self.hi = Z.min(0), Z.max(0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((self._slog(X.astype(np.float64)) - self.mu) / self.sd).astype(np.float32)

    def state(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("mu", "sd", "lo", "hi")}


def to_tensors(split: Split, pre: Preprocessor, device) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.from_numpy(pre.transform(split.X)).to(device),
            torch.from_numpy(split.y).to(device))


def class_weights(y: np.ndarray, device) -> torch.Tensor:
    cnt = np.bincount(y, minlength=2).astype(np.float64)
    w = cnt.sum() / (2.0 * np.maximum(cnt, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)
