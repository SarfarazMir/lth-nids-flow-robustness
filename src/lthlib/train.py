"""Masked training loop (optionally adversarial) and evaluation metrics."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .attacks import ThreatModel, pgd
from .flowattack import craft
from .prune import apply_masks


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, masks, X, y, *, epochs=10, lr=1e-3, wd=1e-4, bs=1024, cw=None, seed=0,
          rewind_iter=None, adv: dict | None = None, tm: ThreatModel | None = None,
          X_val=None, y_val=None, log=None):
    """Train `model` with fixed `masks` (weights outside the mask are held at 0).

    adv: None or dict(eps=..., steps=..., ratio=...) for PGD adversarial training (Madry et al.)
         where attack-class samples in each batch are replaced by PGD examples under `tm`
         (mixed with clean samples according to `ratio` = fraction of attack samples perturbed).
    Returns (model, rewind_state) where rewind_state is a deepcopy of the state_dict after
    `rewind_iter` optimizer steps (None if rewind_iter is None).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(X)
    steps_per_epoch = math.ceil(n / bs)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    apply_masks(model, masks)
    rewind_state = copy.deepcopy(model.state_dict()) if rewind_iter == 0 else None
    it = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(X.device)
        model.train()
        for b in range(steps_per_epoch):
            idx = perm[b * bs:(b + 1) * bs]
            xb, yb = X[idx], y[idx]
            if adv is not None:
                att = (yb == 1).nonzero().flatten()
                if adv.get("ratio", 1.0) < 1.0:
                    att = att[torch.rand(len(att), device=att.device) < adv["ratio"]]
                if len(att) and adv.get("kind") == "flow":
                    # traffic-space adversarial training (flow-consistent injection attack)
                    xa = craft(model, adv["fa"], adv["X_raw"][idx][att], adv["rho"],
                               n_random=adv.get("n_random", 16), steps=adv.get("steps", 10),
                               seed=it)
                elif len(att):
                    xa = pgd(model, xb[att], yb[att], tm, adv["eps"], steps=adv.get("steps", 10),
                             restarts=1)
                if len(att):
                    xb = xb.clone()
                    xb[att] = xa
                model.train()
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            apply_masks(model, masks)
            it += 1
            if rewind_iter is not None and it == rewind_iter:
                rewind_state = copy.deepcopy(model.state_dict())
        if log is not None and X_val is not None and (ep == epochs - 1 or ep % 5 == 0):
            m = evaluate(model, X_val, y_val)
            log(f"  ep {ep + 1}/{epochs} loss={loss.item():.4f} val_bacc={m['bacc']:.4f} "
                f"[{time.time() - t0:.0f}s]")
    return model, rewind_state


@torch.no_grad()
def scores(model, X, bs=8192):
    model.eval()
    return torch.cat([F.softmax(model(X[i:i + bs]), 1)[:, 1] for i in range(0, len(X), bs)])


@torch.no_grad()
def evaluate(model, X, y, thr=0.5) -> dict:
    p = scores(model, X).cpu().numpy()
    yt = y.cpu().numpy()
    yp = (p >= thr).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    out = dict(bacc=(tpr + tnr) / 2, DR=tpr, FPR=1 - tnr, acc=(tp + tn) / len(yt),
               macro_f1=f1_score(yt, yp, average="macro"))
    if len(np.unique(yt)) == 2:
        out["roc_auc"] = roc_auc_score(yt, p)
        out["pr_auc_benign"] = average_precision_score(1 - yt, 1 - p)  # minority class
    return out
