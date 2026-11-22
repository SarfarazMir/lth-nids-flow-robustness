"""HYDRA-style robust pruning (Sehwag et al., NeurIPS 2020), re-implemented for MLPs.

Pipeline: (1) adversarially pre-train the dense net (done by the caller); (2) freeze weights and
learn real-valued importance scores s with the adversarial training loss, where the forward pass
uses w * TopK(s) and gradients reach s through a straight-through estimator (as in edge-popup,
Ramanujan et al. 2020); scores are initialized proportional to |w| (HYDRA's scaled init);
(3) fix the mask TopK(s) and adversarially fine-tune the surviving weights (caller).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .attacks import pgd
from .flowattack import craft
from .models import prunable
from .prune import _global_topk, keep_count


class _STEMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_float, scores):
        return mask_float

    @staticmethod
    def backward(ctx, g):
        return None, g


class ScoredModel(nn.Module):
    """Wraps a model; prunable weights are multiplied by a top-k mask of learnable scores."""

    def __init__(self, base: nn.Module, sparsity: float, layerwise: bool = True):
        super().__init__()
        self.base = base
        self.sparsity = sparsity
        self.layerwise = layerwise
        self.scores = nn.ParameterDict()
        self.key = {}
        for n, p in prunable(base).items():
            k = n.replace(".", "__")
            self.key[n] = k
            s = p.detach().abs()
            self.scores[k] = nn.Parameter(s / s.max().clamp_min(1e-12))
        for p in base.parameters():
            p.requires_grad_(False)

    def current_masks(self) -> dict:
        sc = {n: self.scores[k].detach().abs() for n, k in self.key.items()}
        if self.layerwise:
            out = {}
            for n, s in sc.items():
                keep = max(1, int(round((1 - self.sparsity) * s.numel())))
                thr = torch.topk(s.flatten(), keep).values[-1]
                out[n] = s >= thr
            return out
        base = {n: torch.ones_like(s, dtype=torch.bool) for n, s in sc.items()}
        return _global_topk(sc, base, keep_count(base, self.sparsity))

    def forward(self, x):
        masks = self.current_masks()
        params = dict(self.base.named_parameters())
        buffers = dict(self.base.named_buffers())
        for n, k in self.key.items():
            s = self.scores[k]
            m = _STEMask.apply(masks[n].float(), s)
            params[n] = params[n] * m
        return functional_call(self.base, {**params, **buffers}, (x,))


def learn_scores(base, X, y, tm, sparsity, *, epochs=5, lr=0.1, bs=1024, eps=1.0, steps=10,
                 cw=None, seed=0, layerwise=True, adv=None):
    """Optimize HYDRA importance scores with the adversarial loss; returns a boolean mask dict."""
    sm = ScoredModel(base, sparsity, layerwise).to(X.device)
    opt = torch.optim.SGD(sm.scores.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    n = len(X)
    steps_per_epoch = math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(X.device)
        for b in range(steps_per_epoch):
            idx = perm[b * bs:(b + 1) * bs]
            xb, yb = X[idx], y[idx]
            att = (yb == 1).nonzero().flatten()
            if len(att) and adv is not None and adv.get("kind") == "flow":
                xa = craft(sm, adv["fa"], adv["X_raw"][idx][att], adv["rho"], steps=steps, seed=b)
            elif len(att):
                xa = pgd(sm, xb[att], yb[att], tm, eps, steps=steps, restarts=1)
            if len(att):
                xb = xb.clone()
                xb[att] = xa
            sm.train()
            sm.base.eval()  # BN statistics stay fixed while scores are learned
            loss = F.cross_entropy(sm(xb), yb, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return {n: m.clone() for n, m in sm.current_masks().items()}
