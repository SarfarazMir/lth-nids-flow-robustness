"""Mask utilities and pruning criteria.

A mask set is a dict {param_name: bool tensor} over `models.prunable(model)`.
Sparsity is always the fraction of ZERO entries over all prunable weights.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import prunable


def full_masks(model) -> dict[str, torch.Tensor]:
    return {n: torch.ones_like(p, dtype=torch.bool) for n, p in prunable(model).items()}


@torch.no_grad()
def apply_masks(model, masks) -> None:
    for n, p in prunable(model).items():
        if n in masks:
            p.mul_(masks[n])


def sparsity(masks) -> float:
    tot = sum(m.numel() for m in masks.values())
    return 1.0 - sum(int(m.sum()) for m in masks.values()) / tot


def layer_density(masks) -> dict[str, float]:
    return {n: float(m.float().mean()) for n, m in masks.items()}


def _global_topk(scores: dict, masks: dict, keep_total: int) -> dict:
    """Keep the `keep_total` highest-scoring weights among those currently unmasked."""
    flat = torch.cat([torch.where(masks[n], scores[n], torch.full_like(scores[n], -float("inf"))).flatten()
                      for n in scores])
    if keep_total >= flat.numel():
        return {n: masks[n].clone() for n in masks}
    thr = torch.topk(flat, keep_total, sorted=True).values[-1]
    new = {}
    for n in scores:
        new[n] = masks[n] & (scores[n] >= thr)
    # resolve ties so the exact budget is met
    excess = sum(int(m.sum()) for m in new.values()) - keep_total
    if excess > 0:
        for n in new:
            tie = (scores[n] == thr) & new[n]
            k = min(excess, int(tie.sum()))
            if k:
                idx = tie.flatten().nonzero().flatten()[:k]
                v = new[n].flatten().clone()
                v[idx] = False
                new[n] = v.view_as(new[n])
                excess -= k
            if excess == 0:
                break
    return new


def keep_count(masks, target_sparsity: float) -> int:
    tot = sum(m.numel() for m in masks.values())
    return int(round((1.0 - target_sparsity) * tot))


def magnitude_masks(model, masks, target_sparsity: float) -> dict:
    """Global magnitude pruning (Han et al. 2015; Frankle & Carbin 2019) to `target_sparsity`."""
    scores = {n: p.detach().abs() for n, p in prunable(model).items()}
    return _global_topk(scores, masks, keep_count(masks, target_sparsity))


def random_masks_like(ref_masks, seed: int) -> dict:
    """Random mask with the SAME per-layer density as `ref_masks` (layer-wise random pruning)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for n, m in ref_masks.items():
        k = int(m.sum())
        perm = torch.randperm(m.numel(), generator=g)[:k]
        v = torch.zeros(m.numel(), dtype=torch.bool)
        v[perm] = True
        out[n] = v.view_as(m).to(m.device)
    return out


def shuffle_masks_within_layer(masks, seed: int) -> dict:
    return random_masks_like(masks, seed)


def erk_random_masks(model, target_sparsity: float, seed: int) -> dict:
    """Random pruning with Erdos-Renyi-Kernel layer densities (Evci et al. 2020)."""
    params = prunable(model)
    raw = {n: (sum(p.shape) / p.numel()) for n, p in params.items()}
    tot = sum(p.numel() for p in params.values())
    keep = (1 - target_sparsity) * tot
    dense = set()
    while True:
        budget = keep - sum(params[n].numel() for n in dense)
        denom = sum(raw[n] * params[n].numel() for n in params if n not in dense)
        eps = budget / denom
        over = [n for n in params if n not in dense and eps * raw[n] > 1]
        if not over:
            break
        dense.update(over)
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for n, p in params.items():
        d = 1.0 if n in dense else eps * raw[n]
        k = int(round(d * p.numel()))
        v = torch.zeros(p.numel(), dtype=torch.bool)
        v[torch.randperm(p.numel(), generator=g)[:k]] = True
        out[n] = v.view_as(p).to(p.device)
    return out


# ---------------------------------------------------------------- pruning at initialization
def _batches(X, y, bs, n_batches, gen_seed=0):
    g = torch.Generator(device="cpu").manual_seed(gen_seed)
    for _ in range(n_batches):
        idx = torch.randint(0, len(X), (bs,), generator=g).to(X.device)
        yield X[idx], y[idx]


def snip_masks(model, X, y, target_sparsity, cw=None, n_batches=10, bs=1024) -> dict:
    """SNIP (Lee et al., ICLR 2019): saliency |g * w| at initialization, one shot."""
    m = copy.deepcopy(model).train()
    params = prunable(m)
    acc = {n: torch.zeros_like(p) for n, p in params.items()}
    for xb, yb in _batches(X, y, bs, n_batches):
        m.zero_grad()
        F.cross_entropy(m(xb), yb, weight=cw).backward()
        for n, p in params.items():
            acc[n] += (p.grad * p).abs().detach()
    base = {n: torch.ones_like(p, dtype=torch.bool) for n, p in params.items()}
    return _global_topk(acc, base, keep_count(base, target_sparsity))


def grasp_masks(model, X, y, target_sparsity, cw=None, n_batches=10, bs=1024, T=200.0) -> dict:
    """GraSP (Wang et al., ICLR 2020): score = -w * (H g); remove highest scores."""
    m = copy.deepcopy(model).train()
    params = prunable(m)
    names = list(params)
    plist = [params[n] for n in names]
    g0 = [torch.zeros_like(p) for p in plist]
    for xb, yb in _batches(X, y, bs, n_batches, 1):
        gr = torch.autograd.grad(F.cross_entropy(m(xb) / T, yb, weight=cw), plist)
        g0 = [a + b.detach() for a, b in zip(g0, gr)]
    acc = [torch.zeros_like(p) for p in plist]
    for xb, yb in _batches(X, y, bs, n_batches, 2):
        gr = torch.autograd.grad(F.cross_entropy(m(xb) / T, yb, weight=cw), plist, create_graph=True)
        z = sum((a * b).sum() for a, b in zip(gr, g0))
        hg = torch.autograd.grad(z, plist)
        acc = [a + (-p.detach() * h.detach()) for a, p, h in zip(acc, plist, hg)]
    # GraSP removes the weights with the LARGEST score -> keep the smallest; negate for top-k.
    scores = {n: -s for n, s in zip(names, acc)}
    base = {n: torch.ones_like(p, dtype=torch.bool) for n, p in params.items()}
    return _global_topk(scores, base, keep_count(base, target_sparsity))


def synflow_masks(model, d_in, target_sparsity, iters=100) -> dict:
    """SynFlow (Tanaka et al., NeurIPS 2020): data-free iterative synaptic-flow pruning."""
    m = copy.deepcopy(model).double().eval()  # eval: BN uses running stats (identity at init)
    params = prunable(m)
    with torch.no_grad():
        for n, p in m.state_dict().items():
            if p.dtype.is_floating_point:
                p.abs_()
    masks = {n: torch.ones_like(p, dtype=torch.bool) for n, p in params.items()}
    x = torch.ones(1, d_in, dtype=torch.float64, device=next(m.parameters()).device)
    tot = sum(p.numel() for p in params.values())
    for k in range(1, iters + 1):
        apply_masks(m, masks)
        m.zero_grad()
        m(x).sum().backward()
        scores = {n: (p.grad * p).abs().detach() for n, p in params.items()}
        s_k = 1 - (1 - target_sparsity) ** (k / iters)
        masks = _global_topk(scores, masks, int(round((1 - s_k) * tot)))
    return {n: v for n, v in masks.items()}


def lamp_masks(model, masks, target_sparsity: float) -> dict:
    """Magnitude pruning with LAMP scores (Lee et al., ICLR 2021): layer-adaptive global pruning.

    For weights of one layer sorted ascending by w^2, score(u) = w_u^2 / sum_{v >= u} w_v^2.
    Already-pruned weights are excluded from the normalizer.
    """
    scores = {}
    for n, p in prunable(model).items():
        w2 = (p.detach() * masks[n]).flatten() ** 2
        order = torch.argsort(w2)
        sorted_w2 = w2[order]
        tail = torch.flip(torch.cumsum(torch.flip(sorted_w2, [0]), 0), [0])
        s = torch.empty_like(w2)
        s[order] = sorted_w2 / tail.clamp_min(1e-12)
        scores[n] = s.view_as(p)
    return _global_topk(scores, masks, keep_count(masks, target_sparsity))
