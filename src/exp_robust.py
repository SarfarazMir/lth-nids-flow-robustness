"""RQ1(b): pruning WITH adversarial training, at matched sparsity and training budget.

Methods (all adversarially trained with PGD under the chosen AT threat model):
  dense_at      dense adversarial training (Madry et al. 2018)
  aimp          IMP with weight rewinding, adversarial training in every round (Cosentino 2019 style)
  aimp_reinit   A-IMP mask + fresh init (control)
  random_at     random mask with A-IMP's layer-wise density + rewind weights (control)
  omp_at        one-shot global magnitude pruning of dense_at + adversarial fine-tuning
  lamp_at       one-shot LAMP pruning of dense_at + adversarial fine-tuning (Piras et al. 2025 baseline)
  hydra         HYDRA: score learning with adversarial loss on dense_at + adversarial fine-tuning
Plus the trivial input-level baseline:
  immutable_std dense, standard training, attacker-mutable input features removed (set to 0)
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import torch
import torch.nn as nn

from lthlib import prune as P
from lthlib.attacks import INCREASE_ONLY, BIDIRECTIONAL
from lthlib.common import (DEVICE, RESULTS, Bundle, Domain, Logger, attack_subset, clean_eval,
                           environment, make_pre, robust_eval, threat_models)
from lthlib.data import save_json, to_tensors
from lthlib.hydra import learn_scores
from lthlib.models import build_model
from lthlib.train import set_seed, train

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="ids17")
ap.add_argument("--eval_domains", default="ddos19_d1,ddos19_d2,ids17")
ap.add_argument("--arch", default="mlp")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=16)
ap.add_argument("--rate", type=float, default=0.2)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--rewind_frac", type=float, default=0.05)
ap.add_argument("--eval_rounds", default="4,7,10,13,16")
ap.add_argument("--at_tm", default="flow", choices=["flow", "constrained", "unconstrained"])
ap.add_argument("--at_eps", type=float, default=1.0, help="eps (feature-space AT) or rho (flow AT)")
ap.add_argument("--at_steps", type=int, default=5)
ap.add_argument("--at_bs", type=int, default=4096)
ap.add_argument("--at_lr", type=float, default=2e-3)
ap.add_argument("--hydra_epochs", type=int, default=5)
ap.add_argument("--baseline_rounds", default="", help="rounds for baselines (default: eval_rounds)")
ap.add_argument("--no_sanity", action="store_true", help="skip FGSM/CW sanity checks")
ap.add_argument("--wide_probe", action="store_true",
                help="immutable baseline: find changeable features with many (k, s, dd) probes, not one")
ap.add_argument("--immutable_only", action="store_true", help="train/evaluate only the immutable baseline")
ap.add_argument("--methods", default="aimp_reinit,random_at,omp_at,lamp_at,hydra")
ap.add_argument("--n_attack", type=int, default=3000)
ap.add_argument("--tag", default="")
args = ap.parse_args()

name = f"{args.src}_{args.arch}_s{args.seed}_{args.at_tm}{args.at_eps}{args.tag}"
out_dir = os.path.join(RESULTS, "raw", "robust")
os.makedirs(out_dir, exist_ok=True)
log = Logger(os.path.join(RESULTS, "..", "logs", f"robust_{name}.log"))
log(f"args={vars(args)} env={environment()}")

src = Domain(args.src)
pre = make_pre(src)
B = Bundle(src, pre)
tms = threat_models(src.features, pre)
X_att, X_att_raw = attack_subset(B.Xte, B.yte, args.n_attack, seed=0, X_raw=B.Xte_raw)
FLOW = (B.flow, X_att_raw)
evalsets = {"src_test": (B.Xte, B.yte)}
for k in [d for d in args.eval_domains.split(",") if d and d != args.src]:
    evalsets[f"{k}_all"] = to_tensors(Domain(k).splits["test"], pre, DEVICE)
d_in = B.Xtr.shape[1]
steps_per_epoch = -(-B.n_train // args.at_bs)
rewind_iter = max(1, int(args.rewind_frac * args.epochs * steps_per_epoch))
if args.at_tm == "flow":
    ADV = dict(kind="flow", fa=B.flow, X_raw=B.Xtr_raw, rho=args.at_eps, steps=args.at_steps, n_random=32)
else:
    ADV = dict(eps=args.at_eps, steps=args.at_steps)
TR = dict(epochs=args.epochs, lr=args.at_lr, bs=args.at_bs, cw=B.cw, adv=ADV, tm=tms.get(args.at_tm))
records = []


def record(method, rnd, masks, model, t_train):
    r = dict(method=method, round=rnd, sparsity=P.sparsity(masks), train_time_s=t_train)
    r.update(clean_eval(model, evalsets))
    r.update(robust_eval(model, X_att, tms, flow=FLOW, sanity=not args.no_sanity))
    records.append(r)
    log(f"{method:13s} r={rnd:2d} sp={r['sparsity']:.3f} bacc={r['src_test/bacc']:.4f} "
        f"ESR_AUC(u)={r['unconstrained/ESR_AUC']:.3f} ESR_AUC(c)={r['constrained/ESR_AUC']:.3f} "
        f"flow={r['flow/ESR_AUC']:.3f} "
        f"t={t_train:.0f}s")
    save_json(dict(args=vars(args), env=environment(), records=records), os.path.join(out_dir, f"{name}.json"))


# ---------------------------------------------------------------- trivial input-level baseline
class InputMask(nn.Module):
    def __init__(self, net, keep):
        super().__init__()
        self.net = net
        self.register_buffer("keep", keep)

    def forward(self, x):
        return self.net(x * self.keep)


if args.at_tm == "flow":
    # features the flow-consistent attack can change (determined empirically from the attack map)
    Xp = B.Xtr_raw[:2000]
    one = torch.ones(len(Xp), dtype=torch.float64, device=DEVICE)
    changed = (B.flow.apply(Xp, one, 700 * one, 1e5 * one) != Xp).any(0)
    if args.wide_probe:  # the single probe misses e.g. Min Packet Length (lowered by a small injected packet)
        Xp = B.Xtr_raw.double()
        one = torch.ones(len(Xp), dtype=torch.float64, device=DEVICE)
        kmax, ddmax = B.flow.limits(Xp, 2.0)
        for k in (one, kmax):
            for s in (0.0, 1.0, 700.0, 1460.0):
                for dd in (0 * one, 1e5 * one, ddmax):
                    changed |= (B.flow.apply(Xp, k, s * one, dd) != Xp).any(0)
    keep = (~changed).float()
else:
    keep = torch.tensor([0.0 if (f in INCREASE_ONLY or f in BIDIRECTIONAL) else 1.0 for f in src.features],
                        device=DEVICE)
log(f"immutable baseline keeps {int(keep.sum())}/{len(keep)} features: "
    f"{[f for f, k in zip(src.features, keep.tolist()) if k]}")
set_seed(args.seed)
net = InputMask(build_model(args.arch, d_in), keep).to(DEVICE)
t = time.time()
net, _ = train(net, P.full_masks(net), B.Xtr, B.ytr, epochs=args.epochs, lr=args.lr, cw=B.cw, seed=args.seed)
record("immutable_std", 0, P.full_masks(net), net, time.time() - t)
if args.immutable_only:
    log("done")
    raise SystemExit

# ---------------------------------------------------------------- dense AT + A-IMP
set_seed(args.seed)
model = build_model(args.arch, d_in).to(DEVICE)
masks = P.full_masks(model)
t = time.time()
model, theta_k = train(model, masks, B.Xtr, B.ytr, seed=args.seed, rewind_iter=rewind_iter, **TR)
record("dense_at", 0, masks, model, time.time() - t)
dense_at = copy.deepcopy(model.state_dict())
eval_rounds = [int(x) for x in args.eval_rounds.split(",")]
aimp_masks = {}
for rnd in range(1, args.rounds + 1):
    masks = P.magnitude_masks(model, masks, 1 - (1 - args.rate) ** rnd)
    model.load_state_dict(theta_k)
    t = time.time()
    model, _ = train(model, masks, B.Xtr, B.ytr, seed=args.seed, **TR)
    aimp_masks[rnd] = {k: v.clone() for k, v in masks.items()}
    if rnd in eval_rounds:
        record("aimp", rnd, masks, model, time.time() - t)
    else:
        log(f"aimp r={rnd} sp={P.sparsity(masks):.3f} (not evaluated)")
torch.save(dict(aimp_masks={r: {k: v.cpu() for k, v in m.items()} for r, m in aimp_masks.items()},
                theta_k=theta_k, dense_at=dense_at), os.path.join(out_dir, f"{name}_state.pt"))

# ---------------------------------------------------------------- baselines at matched sparsity
for rnd in [int(x) for x in (args.baseline_rounds or args.eval_rounds).split(",")]:
    m_ref = aimp_masks[rnd]
    s = P.sparsity(m_ref)
    for meth in [m for m in args.methods.split(",") if m]:
        set_seed(args.seed)
        net = build_model(args.arch, d_in).to(DEVICE)
        lr = args.at_lr
        t = time.time()
        if meth == "aimp_reinit":
            set_seed(args.seed + 777)
            net = build_model(args.arch, d_in).to(DEVICE)
            mk = m_ref
        elif meth == "random_at":
            net.load_state_dict(theta_k)
            mk = P.random_masks_like(m_ref, seed=args.seed + rnd)
        elif meth == "omp_at":
            net.load_state_dict(dense_at)
            mk = P.magnitude_masks(net, P.full_masks(net), s)
            lr = args.at_lr / 10
        elif meth == "lamp_at":
            net.load_state_dict(dense_at)
            mk = P.lamp_masks(net, P.full_masks(net), s)
            lr = args.at_lr / 10
        elif meth == "hydra":
            net.load_state_dict(dense_at)
            mk = learn_scores(net, B.Xtr, B.ytr, tms.get(args.at_tm), s, epochs=args.hydra_epochs,
                              eps=args.at_eps, steps=args.at_steps, cw=B.cw, seed=args.seed, adv=ADV,
                              bs=args.at_bs)
            for p in net.parameters():
                p.requires_grad_(True)
            lr = args.at_lr / 10
        else:
            raise ValueError(meth)
        net, _ = train(net, mk, B.Xtr, B.ytr, seed=args.seed, **{**TR, "lr": lr})
        record(meth, rnd, mk, net, time.time() - t)
log("done")
