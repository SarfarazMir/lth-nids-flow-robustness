"""RQ2 characterization: transferring pruned subnetworks across NIDS datasets / capture days.

Requires an exp_rq1 run on the SOURCE domain (provides the IMP masks, theta_0, theta_k and the
trained dense weights). For a TARGET domain and a target-label budget (fraction of the target
training split), trains on the target:

  dense_scratch   dense net, fresh init (seed)                      - no transfer
  dense_ft        dense net initialized from SOURCE-trained weights  - standard transfer learning
  tkt_rewind      source IMP mask + source rewind weights theta_k    - ticket transfer (Morcos et al. 2019)
  tkt_reinit      source IMP mask + fresh random init                - mask-only transfer
  tkt_ft          source IMP mask + source-trained sparse weights    - sparse fine-tuning
  rand_rewind     random mask with the ticket's layer-wise density + theta_k  (control)
  snip_tgt        SNIP mask computed on target data at init          - cheap target-specific pruning

All conditions share the training budget (epochs) and seed. The input preprocessor is refit on
the target training subset (what a practitioner would do); the feature schema is identical.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import torch

from lthlib import prune as P
from lthlib.common import (DEVICE, RESULTS, Bundle, Domain, Logger, attack_subset, clean_eval,
                           environment, robust_eval, threat_models)
from lthlib.data import Preprocessor, save_json, subsample
from lthlib.models import build_model
from lthlib.train import set_seed, train

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="ddos19_d1")
ap.add_argument("--tgt", default="ids17")
ap.add_argument("--arch", default="mlp")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--src_tag", default="")
ap.add_argument("--fracs", default="0.01,0.1,1.0")
ap.add_argument("--rounds", default="7,10,13,16")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--conds", default="dense_scratch,dense_ft,tkt_rewind,tkt_reinit,tkt_ft,rand_rewind,snip_tgt")
ap.add_argument("--n_attack", type=int, default=3000)
ap.add_argument("--no_robust", action="store_true")
ap.add_argument("--tag", default="")
args = ap.parse_args()

src_name = f"{args.src}_{args.arch}_s{args.seed}{args.src_tag}"
state = torch.load(os.path.join(RESULTS, "raw", "rq1", f"{src_name}_state.pt"), map_location=DEVICE)
name = f"{args.src}__{args.tgt}_{args.arch}_s{args.seed}{args.tag}"
out_dir = os.path.join(RESULTS, "raw", "rq2")
os.makedirs(out_dir, exist_ok=True)
log = Logger(os.path.join(RESULTS, "..", "logs", f"rq2_{name}.log"))
log(f"args={vars(args)}")

tgt = Domain(args.tgt)
records = []
rounds = [int(r) for r in args.rounds.split(",")]


def to_dev(sd):
    return {k: v.to(DEVICE) for k, v in sd.items()}


for frac in [float(f) for f in args.fracs.split(",")]:
    pre = Preprocessor().fit(subsample(tgt.splits["train"], frac, args.seed).X)
    B = Bundle(tgt, pre, train_frac=frac, frac_seed=args.seed)
    tms = threat_models(tgt.features, pre)
    X_att, X_att_raw = attack_subset(B.Xte, B.yte, args.n_attack, seed=0, X_raw=B.Xte_raw)
    FLOW = (B.flow, X_att_raw)
    d_in = B.Xtr.shape[1]
    # keep the number of optimizer steps comparable across budgets: at least 300 steps
    steps = -(-B.n_train // 1024)
    epochs = max(args.epochs, -(-300 // steps))
    TR = dict(epochs=epochs, lr=args.lr, cw=B.cw)
    log(f"frac={frac} n_train={B.n_train} epochs={epochs}")

    def run(cond, rnd, net, mk, lr=None):
        set_seed(args.seed)
        t = time.time()
        net, _ = train(net, mk, B.Xtr, B.ytr, seed=args.seed, **{**TR, **({"lr": lr} if lr else {})})
        r = dict(cond=cond, frac=frac, n_train=B.n_train, round=rnd, sparsity=P.sparsity(mk),
                 train_time_s=time.time() - t)
        r.update(clean_eval(net, {"tgt_test": (B.Xte, B.yte)}))
        if not args.no_robust:
            r.update(robust_eval(net, X_att, tms, flow=FLOW, feature_attacks=False))
        records.append(r)
        log(f"{cond:13s} frac={frac:<5} r={rnd:2d} sp={r['sparsity']:.3f} bacc={r['tgt_test/bacc']:.4f} "
            f"F1={r['tgt_test/macro_f1']:.4f}"
            + ("" if args.no_robust else f" flow_ESR_AUC={r['flow/ESR_AUC']:.3f}"))
        save_json(dict(args=vars(args), env=environment(), records=records),
                  os.path.join(out_dir, f"{name}.json"))

    def fresh(seed_off=0):
        set_seed(args.seed + seed_off)
        return build_model(args.arch, d_in).to(DEVICE)

    conds = args.conds.split(",")
    if "dense_scratch" in conds:
        net = fresh()
        run("dense_scratch", 0, net, P.full_masks(net))
    if "dense_ft" in conds:
        net = fresh()
        net.load_state_dict(to_dev(state["dense"]))
        run("dense_ft", 0, net, P.full_masks(net), lr=args.lr / 10)
    for rnd in rounds:
        mk = {k: v.to(DEVICE) for k, v in state["imp_masks"][rnd].items()}
        if "tkt_rewind" in conds:
            net = fresh()
            net.load_state_dict(to_dev(state["theta_k"]))
            run("tkt_rewind", rnd, net, mk)
        if "tkt_reinit" in conds:
            run("tkt_reinit", rnd, fresh(777), mk)
        if "tkt_ft" in conds:
            net = fresh()  # the trained source ticket (round rnd), fine-tuned on the target
            net.load_state_dict(to_dev(state["imp_weights"][rnd]))
            run("tkt_ft", rnd, net, mk, lr=args.lr / 10)
        if "rand_rewind" in conds:
            net = fresh()
            net.load_state_dict(to_dev(state["theta_k"]))
            run("rand_rewind", rnd, net, P.random_masks_like(mk, seed=args.seed + rnd))
        if "snip_tgt" in conds:
            net = fresh()
            run("snip_tgt", rnd, net, P.snip_masks(net, B.Xtr, B.ytr, P.sparsity(mk), cw=B.cw))
log("done")
