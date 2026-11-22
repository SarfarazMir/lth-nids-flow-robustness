"""Shared experiment setup: domains, preprocessing, threat models and the evaluation bundle."""
from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import torch

from . import data as D
from .attacks import ThreatModel, evasion_eval
from .flowattack import FlowAttack, flow_eval
from .train import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = os.path.join(D.ROOT, "results")

# Each domain = (cache file, split seed). Splits are fixed (seed 0) for every run so that all
# methods and training seeds see identical train/val/test partitions.
DOMAINS = {
    "ddos19_d1": "ddos2019_day1",
    "ddos19_d2": "ddos2019_day2",
    "ids17": "ids2017",
}

# Calibrated on dense MLPs (results/raw/calibrate_eps.json): grids span the transition from
# ~0% to ~100% evasion on both source domains.
EPS_GRID = {"unconstrained": (0.05, 0.1, 0.15, 0.25), "constrained": (0.25, 0.5, 1.0, 2.0)}


def environment() -> dict:
    return dict(python=sys.version.split()[0], torch=torch.__version__,
                cuda=torch.version.cuda, gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                platform=platform.platform(), numpy=np.__version__)


class Domain:
    """A dataset with fixed stratified 70/15/15 split (split seed 0)."""

    _cache: dict = {}

    def __init__(self, key: str):
        if key not in Domain._cache:
            df = D.load_frame(DOMAINS[key])
            Domain._cache[key] = (D.stratified_split(df, seed=0), D.feature_names(df))
        self.key = key
        self.splits, self.features = Domain._cache[key]


class Bundle:
    """Tensors of a domain transformed with a given preprocessor (fit on some training set)."""

    def __init__(self, dom: Domain, pre: D.Preprocessor, train_frac: float = 1.0, frac_seed: int = 0):
        tr = D.subsample(dom.splits["train"], train_frac, frac_seed)
        self.Xtr, self.ytr = D.to_tensors(tr, pre, DEVICE)
        self.Xva, self.yva = D.to_tensors(dom.splits["val"], pre, DEVICE)
        self.Xte, self.yte = D.to_tensors(dom.splits["test"], pre, DEVICE)
        self.cw = D.class_weights(tr.y, DEVICE)
        self.n_train = len(tr.y)
        # raw (untransformed) features: needed by the flow-consistent attack
        self.Xtr_raw = torch.tensor(tr.X, dtype=torch.float32, device=DEVICE)
        self.Xte_raw = torch.tensor(dom.splits["test"].X, dtype=torch.float32, device=DEVICE)
        self.flow = FlowAttack(dom.features, pre, DEVICE)


def make_pre(dom: Domain) -> D.Preprocessor:
    return D.Preprocessor().fit(dom.splits["train"].X)


def threat_models(features, pre: D.Preprocessor):
    return {k: ThreatModel(features, pre.lo, pre.hi, k, DEVICE) for k in ("unconstrained", "constrained")}


def attack_subset(X, y, n=3000, seed=0, X_raw=None):
    """Fixed random subset of malicious test flows (same subset for every method/seed).
    Returns the model-input tensor, or (input, raw) if X_raw is given."""
    idx = torch.nonzero(y == 1).flatten()
    g = torch.Generator(device="cpu").manual_seed(seed)
    sel = idx[torch.randperm(len(idx), generator=g)[:n].to(idx.device)]
    return X[sel] if X_raw is None else (X[sel], X_raw[sel])


FLOW_RHOS = (0.1, 0.25, 0.5, 1.0, 2.0)


def robust_eval(model, X_att, tms, eps=EPS_GRID, steps=50, restarts=2, surrogate=None,
                flow=None, feature_attacks=True, sanity=True) -> dict:
    """flow: optional (FlowAttack, raw malicious flows) -> traffic-space attack (primary metric)."""
    out = {}
    if flow is not None:
        fa, X_raw = flow
        r = flow_eval(model, fa, X_raw, FLOW_RHOS)
        out.update({f"flow/{m}": v for m, v in r.items()})
    if not feature_attacks:
        return out
    for k, tm in tms.items():
        grid = eps[k]
        r = evasion_eval(model, X_att, tm, grid, steps=steps, restarts=restarts)
        out.update({f"{k}/{m}": v for m, v in r.items()})
        # area under the ESR-vs-eps curve, normalized to [0, 1] (trapezoid on the grid)
        e = [0.0] + list(grid)
        s = [0.0] + [r[f"ESR@{x}"] for x in grid]
        out[f"{k}/ESR_AUC"] = sum((e[i + 1] - e[i]) * (s[i + 1] + s[i]) / 2 for i in range(len(grid))) / grid[-1]
        if surrogate is not None:
            r = evasion_eval(model, X_att, tm, grid, steps=steps, restarts=1, source_model=surrogate)
            out.update({f"{k}/transfer_{m}": v for m, v in r.items() if m != "clean_DR"})
        if not sanity:
            continue
        # sanity checks at the 2nd grid point: FGSM must not beat PGD; CW-loss PGD as a 2nd attack
        mid = grid[1]
        out[f"{k}/fgsm_ESR@{mid}"] = evasion_eval(model, X_att, tm, (mid,), attack="fgsm")[f"ESR@{mid}"]
        out[f"{k}/cw_ESR@{mid}"] = evasion_eval(model, X_att, tm, (mid,), steps=steps, restarts=1,
                                                loss="cw")[f"ESR@{mid}"]
    return out


def clean_eval(model, bundles: dict) -> dict:
    out = {}
    for name, (X, y) in bundles.items():
        out.update({f"{name}/{k}": v for k, v in evaluate(model, X, y).items()})
    return out


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
