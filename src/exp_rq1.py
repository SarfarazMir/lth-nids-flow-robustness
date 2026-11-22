"""RQ1 characterization: clean accuracy and adversarial robustness of pruned subnetworks.

For one (source domain, architecture, seed):
  1. Train dense model; keep init (theta_0), rewind point (theta_k) and trained weights.
  2. IMP with weight rewinding (Frankle et al. 2020): prune 20% of remaining weights per round
     (global magnitude), rewind surviving weights to theta_k, retrain. -> one ticket per round.
  3. Baselines at selected sparsities (same training budget, same seed):
       imp_reinit  - IMP mask, fresh random init               (Frankle & Carbin 2019 control)
       random_lw   - random mask with IMP's layer-wise density, rewound weights theta_k
       erk         - random mask, ERK layer densities, theta_0
       snip/grasp/synflow - pruning at initialization, theta_0
       omp_ft      - one-shot magnitude pruning of the trained dense net + fine-tuning
  4. Every network: clean metrics on source test, temporal-shift test (CIC-DDoS2019 other day)
     and other datasets if present; evasion robustness (PGD-50 x2 restarts, unconstrained and
     constrained threat models, eps grid), FGSM/CW sanity checks and black-box transfer attacks
     from an independently trained dense surrogate.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import torch

from lthlib import prune as P
from lthlib.common import (DEVICE, RESULTS, Bundle, Domain, Logger, attack_subset, clean_eval,
                           environment, make_pre, robust_eval, threat_models)
from lthlib.data import save_json, to_tensors
from lthlib.models import build_model
from lthlib.train import set_seed, train

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="ddos19_d1")
ap.add_argument("--eval_domains", default="ddos19_d2,ids17")
ap.add_argument("--arch", default="mlp")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--rate", type=float, default=0.2)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--rewind_frac", type=float, default=0.05, help="rewind point as fraction of training")
ap.add_argument("--eval_rounds", default="4,7,10,13,16,20")
ap.add_argument("--baseline_rounds", default="", help="rounds for baselines (default: eval_rounds)")
ap.add_argument("--no_surrogate", action="store_true", help="skip surrogate + transfer attacks")
ap.add_argument("--no_sanity", action="store_true", help="skip FGSM/CW sanity checks")
ap.add_argument("--baselines", default="imp_reinit,random_lw,erk,snip,grasp,synflow,omp_ft")
ap.add_argument("--n_attack", type=int, default=3000)
ap.add_argument("--tag", default="")
args = ap.parse_args()

name = f"{args.src}_{args.arch}_s{args.seed}{args.tag}"
out_dir = os.path.join(RESULTS, "raw", "rq1")
os.makedirs(out_dir, exist_ok=True)
log = Logger(os.path.join(RESULTS, "..", "logs", f"rq1_{name}.log"))
log(f"args={vars(args)} env={environment()}")

src = Domain(args.src)
pre = make_pre(src)
B = Bundle(src, pre)
tms = threat_models(src.features, pre)
X_att, X_att_raw = attack_subset(B.Xte, B.yte, args.n_attack, seed=0, X_raw=B.Xte_raw)
FLOW = (B.flow, X_att_raw)
evalsets = {"src_test": (B.Xte, B.yte)}
for k in [d for d in args.eval_domains.split(",") if d and d != args.src]:
    try:
        dom = Domain(k)
    except FileNotFoundError:
        log(f"domain {k} unavailable - skipped")
        continue
    # zero-shot evaluation on another domain's held-out test split, with SOURCE preprocessing
    evalsets[f"{k}_all"] = to_tensors(dom.splits["test"], pre, DEVICE)

d_in = B.Xtr.shape[1]
steps_per_epoch = -(-B.n_train // 1024)
rewind_iter = max(1, int(args.rewind_frac * args.epochs * steps_per_epoch))
TR = dict(epochs=args.epochs, lr=args.lr, cw=B.cw, X_val=B.Xva, y_val=B.yva)
records = []


def record(method, rnd, masks, model, t_train):
    t0 = time.time()
    r = dict(method=method, round=rnd, sparsity=P.sparsity(masks), train_time_s=t_train,
             layer_density=P.layer_density(masks))
    r.update(clean_eval(model, evalsets))
    r.update(robust_eval(model, X_att, tms, surrogate=surrogate, flow=FLOW, sanity=not args.no_sanity))
    r["eval_time_s"] = time.time() - t0
    records.append(r)
    log(f"{method:11s} r={rnd:2d} sp={r['sparsity']:.3f} bacc={r['src_test/bacc']:.4f} "
        f"F1={r['src_test/macro_f1']:.4f} ESR(u)={r['unconstrained/ESR_AUC']:.3f} "
        f"ESR(c)={r['constrained/ESR_AUC']:.3f} ESR(flow)={r['flow/ESR_AUC']:.3f} "
        + " ".join(f"{k.split('/')[0]}={v:.3f}" for k, v in r.items() if k.endswith('_all/bacc')))
    save_json(dict(args=vars(args), env=environment(), rewind_iter=rewind_iter,
                   n_train=B.n_train, records=records), os.path.join(out_dir, f"{name}.json"))


# ---------------------------------------------------------------- surrogate for transfer attacks
surrogate = None
if not args.no_surrogate:
    set_seed(10_000 + args.seed)
    surrogate = build_model(args.arch, d_in).to(DEVICE)
    surrogate, _ = train(surrogate, P.full_masks(surrogate), B.Xtr, B.ytr, seed=10_000 + args.seed, **TR)
    surrogate.eval()

# ---------------------------------------------------------------- dense + IMP
set_seed(args.seed)
model = build_model(args.arch, d_in).to(DEVICE)
theta0 = copy.deepcopy(model.state_dict())
masks = P.full_masks(model)
t = time.time()
model, theta_k = train(model, masks, B.Xtr, B.ytr, seed=args.seed, rewind_iter=rewind_iter, log=log, **TR)
record("dense", 0, masks, model, time.time() - t)
dense_trained = copy.deepcopy(model.state_dict())

eval_rounds = [int(x) for x in args.eval_rounds.split(",")]
imp_masks, imp_weights = {}, {}
for rnd in range(1, args.rounds + 1):
    target = 1 - (1 - args.rate) ** rnd
    masks = P.magnitude_masks(model, masks, target)
    model.load_state_dict(theta_k)
    t = time.time()
    model, _ = train(model, masks, B.Xtr, B.ytr, seed=args.seed, **TR)
    imp_masks[rnd] = {k: v.cpu() for k, v in masks.items()}
    imp_weights[rnd] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if rnd in eval_rounds:
        record("imp", rnd, masks, model, time.time() - t)
    else:
        r = dict(method="imp", round=rnd, sparsity=P.sparsity(masks))
        r.update(clean_eval(model, evalsets))
        records.append(r)
        log(f"imp         r={rnd:2d} sp={r['sparsity']:.3f} bacc={r['src_test/bacc']:.4f} (clean only)")
torch.save(dict(imp_masks=imp_masks, imp_weights=imp_weights, theta0=theta0, theta_k=theta_k, dense=dense_trained),
           os.path.join(out_dir, f"{name}_state.pt"))

# ---------------------------------------------------------------- baselines at matched sparsity
for rnd in [int(x) for x in (args.baseline_rounds or args.eval_rounds).split(",")]:
    m_imp = {k: v.to(DEVICE) for k, v in imp_masks[rnd].items()}
    s = P.sparsity(m_imp)
    for meth in [b for b in args.baselines.split(",") if b]:
        set_seed(args.seed)
        net = build_model(args.arch, d_in).to(DEVICE)
        if meth == "imp_reinit":
            set_seed(args.seed + 777)
            net = build_model(args.arch, d_in).to(DEVICE)
            mk = m_imp
        elif meth == "random_lw":
            net.load_state_dict(theta_k)
            mk = P.random_masks_like(m_imp, seed=args.seed + rnd)
        elif meth == "erk":
            net.load_state_dict(theta0)
            mk = P.erk_random_masks(net, s, seed=args.seed + rnd)
        elif meth == "snip":
            net.load_state_dict(theta0)
            mk = P.snip_masks(net, B.Xtr, B.ytr, s, cw=B.cw)
        elif meth == "grasp":
            net.load_state_dict(theta0)
            mk = P.grasp_masks(net, B.Xtr, B.ytr, s, cw=B.cw)
        elif meth == "synflow":
            net.load_state_dict(theta0)
            mk = P.synflow_masks(net, d_in, s)
            mk = {k: v.to(DEVICE) for k, v in mk.items()}
        elif meth == "omp_ft":
            net.load_state_dict(dense_trained)
            mk = P.magnitude_masks(net, P.full_masks(net), s)
        else:
            raise ValueError(meth)
        t = time.time()
        if meth == "omp_ft":  # fine-tune: same epochs, 10x smaller peak LR
            net, _ = train(net, mk, B.Xtr, B.ytr, seed=args.seed,
                           **{**TR, "lr": args.lr / 10})
        else:
            net, _ = train(net, mk, B.Xtr, B.ytr, seed=args.seed, **TR)
        record(meth, rnd, mk, net, time.time() - t)
log("done")
