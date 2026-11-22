"""Evasion attacks on flow-feature NIDS classifiers, in model-input (standardized log) space.

Threat models
-------------
* `unconstrained`: L_inf ball of radius eps around x over ALL features, projected onto the
  feature-wise domain box [lo, hi] observed on the source training split. This is the standard
  ML robustness setting; it over-approximates attacker capability in NIDS.
* `constrained` (semi-realistic): only features the attacker can influence from the sending side
  are mutable (forward-direction volume/size/timing and flow-level timing/rates); backward-
  direction statistics, TCP flags and receiver-side window fields are immutable; volume, size and
  timing features may only INCREASE (padding, extra packets, added delay -- an attacker cannot
  remove packets or bytes that the attack needs). Rates may move either way. Domain box applies.
  Because the input map is strictly increasing per feature, "raw value may only increase" is
  exactly "input coordinate may only increase". Inter-feature arithmetic dependencies (e.g.
  Flow Bytes/s = bytes / duration) are NOT enforced; this is a feature-space (not problem-space)
  attack and is reported as such.

Only ATTACK-class flows are perturbed (goal: be classified BENIGN = class 0).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

INCREASE_ONLY = [
    "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd Header Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "Average Packet Size", "Avg Fwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "act_data_pkt_fwd",
    "Active Mean", "Active Max", "Idle Mean", "Idle Max",
]
BIDIRECTIONAL = ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Down/Up Ratio"]


class ThreatModel:
    def __init__(self, features, lo, hi, kind: str, device):
        self.kind = kind
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float32, device=device)
        d = len(features)
        if kind == "unconstrained":
            self.mut = torch.ones(d, device=device)
            self.inc_only = torch.zeros(d, dtype=torch.bool, device=device)
        elif kind == "constrained":
            mut = [f in INCREASE_ONLY or f in BIDIRECTIONAL for f in features]
            self.mut = torch.tensor(mut, dtype=torch.float32, device=device)
            self.inc_only = torch.tensor([f in INCREASE_ONLY for f in features], device=device)
        else:
            raise ValueError(kind)

    def project(self, x_adv, x0, eps):
        d = (x_adv - x0).clamp(-eps, eps) * self.mut
        d = torch.where(self.inc_only, d.clamp_min(0.0), d)
        x = x0 + d
        # never push a coordinate outside the domain box, but do not move clean points that
        # already lie outside it (possible on a shifted target domain)
        lo = torch.minimum(self.lo, x0)
        hi = torch.maximum(self.hi, x0)
        return torch.max(torch.min(x, hi), lo)


def _loss(logits, y, kind):
    if kind == "ce":
        return F.cross_entropy(logits, y, reduction="none")
    # CW margin loss: maximize (best wrong logit - true logit)
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y[:, None], -1e9)
    return other.max(1).values - true


def pgd(model, x, y, tm: ThreatModel, eps, steps=50, alpha=None, loss="ce", restarts=1,
        rand_init=True):
    """L_inf PGD (Madry et al. 2018) with restarts; returns the worst-case x per sample."""
    model.eval()
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    best = x.clone()
    best_l = torch.full((len(x),), -float("inf"), device=x.device)
    for r in range(restarts):
        if rand_init:
            xa = tm.project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
        else:
            xa = x.clone()
        for _ in range(steps):
            xa.requires_grad_(True)
            l = _loss(model(xa), y, loss)
            g, = torch.autograd.grad(l.sum(), xa)
            xa = tm.project(xa.detach() + alpha * g.sign(), x, eps)
        with torch.no_grad():
            lo = model(xa)
            # success indicator dominates, then margin
            l_fin = _loss(lo, y, "cw") + 1e3 * (lo.argmax(1) != y).float()
            upd = l_fin > best_l
            best[upd] = xa[upd]
            best_l[upd] = l_fin[upd]
    return best.detach()


def fgsm(model, x, y, tm: ThreatModel, eps):
    return pgd(model, x, y, tm, eps, steps=1, alpha=eps, rand_init=False)


@torch.no_grad()
def predict(model, X, bs=8192):
    model.eval()
    return torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])


def evasion_eval(model, X_att, tm: ThreatModel, eps_list, steps=50, restarts=2, bs=4096,
                 attack="pgd", loss="ce", source_model=None):
    """Evaluate evasion on attack-class flows X_att (true label 1).

    Returns per-eps: detection rate under attack (robust DR) and evasion success rate (ESR) =
    fraction of clean-detected attacks that the perturbation flips to BENIGN.
    If `source_model` is given, adversarial points are crafted on it and transferred (black-box).
    """
    crafter = source_model if source_model is not None else model
    y = torch.ones(len(X_att), dtype=torch.long, device=X_att.device)
    clean_pred = predict(model, X_att)
    det = clean_pred == 1
    out = {"clean_DR": float(det.float().mean())}
    for eps in eps_list:
        preds = []
        for i in range(0, len(X_att), bs):
            xb, yb = X_att[i:i + bs], y[i:i + bs]
            if attack == "fgsm":
                xa = fgsm(crafter, xb, yb, tm, eps)
            else:
                xa = pgd(crafter, xb, yb, tm, eps, steps=steps, restarts=restarts, loss=loss)
            preds.append(predict(model, xa))
        p = torch.cat(preds)
        out[f"DR@{eps}"] = float((p == 1).float().mean())
        out[f"ESR@{eps}"] = float(((p == 0) & det).float().sum() / det.float().sum().clamp_min(1))
    return out
