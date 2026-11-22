"""Explainable-AI analysis: which input features do dense networks and pruned tickets rely on?

Integrated Gradients (Sundararajan et al., 2017) on the attack-vs-benign logit margin,
baseline = training mean (the all-zeros vector in standardized input space), 64 steps,
on the same 3,000 fixed malicious test flows that are attacked in the robustness study.

Networks (5 seeds, both source datasets):
  dense, IMP tickets (rounds 4/7/10/13/16; weights saved by exp_rq1.py) and, retrained
  deterministically with the exact exp_rq1.py recipe, ERK and re-initialized IMP masks
  (rounds 7/10/13/16). The retrained networks are checked against the logged balanced accuracy.

Per network: global importance g_j = mean |IG_j| (normalized to sum 1), and
  * share of importance on attacker-changeable features (wide probe of the flow attack),
  * concentration: share of the top-5 features and the effective number of features 1/sum(g^2),
  * Spearman rank correlation with the dense network of the same seed,
  * completeness check: mean |sum_j IG_j - (F(x) - F(baseline))| relative to |F(x) - F(baseline)|.
Outputs results/tables/xai_*.csv|md, results/raw/xai/xai_records.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from lthlib import prune as P
from lthlib.common import DEVICE, RESULTS, Bundle, Domain, attack_subset, make_pre
from lthlib.flowattack import MSS
from lthlib.models import build_model
from lthlib.train import evaluate, set_seed, train

SEEDS = [0, 1, 2, 3, 4]
RAW = os.path.join(RESULTS, "raw")
OUT = os.path.join(RAW, "xai")
os.makedirs(OUT, exist_ok=True)
LEVEL = {4: 59.0, 7: 79.0, 10: 89.3, 13: 94.5, 16: 97.2}


def changeable(B):
    X = B.Xtr_raw.double()
    one = torch.ones(len(X), dtype=torch.float64, device=X.device)
    kmax, ddmax = B.flow.limits(X, 2.0)
    ch = torch.zeros(X.shape[1], dtype=torch.bool, device=X.device)
    for k in (one, kmax):
        for s in (0.0, 1.0, 700.0, MSS):
            for dd in (0 * one, 1e5 * one, ddmax):
                ch |= (B.flow.apply(X, k, s * one, dd) != X).any(0)
    return ch.cpu().numpy()


def integrated_gradients(model, X, steps=64, bs=500):
    model.eval()
    out, err = [], []
    alphas = (torch.arange(steps, device=X.device, dtype=torch.float32) + 0.5) / steps  # midpoint rule
    for i in range(0, len(X), bs):
        x = X[i:i + bs]
        tot = torch.zeros_like(x)
        for a in alphas:
            z = (a * x).requires_grad_(True)
            lo = model(z)
            m = (lo[:, 1] - lo[:, 0]).sum()
            g, = torch.autograd.grad(m, z)
            tot += g
        ig = x * tot / steps
        with torch.no_grad():
            f = model(x)
            f0 = model(torch.zeros_like(x))
            d = (f[:, 1] - f[:, 0]) - (f0[:, 1] - f0[:, 0])
        err.append(((ig.sum(1) - d).abs() / d.abs().clamp_min(1e-6)).detach().cpu())
        out.append(ig.detach().cpu())
    return torch.cat(out).numpy(), float(torch.cat(err).median())


def summarize(ig):
    g = np.abs(ig).mean(0)
    g = g / g.sum()
    return g


def main():
    recs, imps = [], {}
    for src in ("ids17", "ddos19_d1"):
        dom = Domain(src)
        pre = make_pre(dom)
        B = Bundle(dom, pre)
        ch = changeable(B)
        X_att = attack_subset(B.Xte, B.yte, 3000, seed=0)
        feats = dom.features
        steps_per_epoch = -(-B.n_train // 1024)
        TR = dict(epochs=10, lr=1e-3, cw=B.cw, X_val=B.Xva, y_val=B.yva)
        logged = {}
        for s in SEEDS:
            d = json.load(open(os.path.join(RAW, "rq1", f"{src}_mlp_s{s}.json")))
            for r in d["records"]:
                logged[(s, r["method"], r["round"])] = r["src_test/bacc"]
        for s in SEEDS:
            st = torch.load(os.path.join(RAW, "rq1", f"{src}_mlp_s{s}_state.pt"), map_location=DEVICE, weights_only=False)
            d_in = B.Xtr.shape[1]
            nets = []
            net = build_model("mlp", d_in).to(DEVICE)
            net.load_state_dict(st["dense"])
            nets.append(("dense", 0, net))
            for rnd in LEVEL:
                net = build_model("mlp", d_in).to(DEVICE)
                net.load_state_dict({k: v.to(DEVICE) for k, v in st["imp_weights"][rnd].items()})
                nets.append(("imp", rnd, net))
            for rnd in (7, 10, 13, 16):
                m_imp = {k: v.to(DEVICE) for k, v in st["imp_masks"][rnd].items()}
                sp = P.sparsity(m_imp)
                # ERK (exp_rq1.py recipe)
                set_seed(s)
                net = build_model("mlp", d_in).to(DEVICE)
                net.load_state_dict(st["theta0"])
                mk = P.erk_random_masks(net, sp, seed=s + rnd)
                net, _ = train(net, mk, B.Xtr, B.ytr, seed=s, **TR)
                nets.append(("erk", rnd, net))
                # IMP mask, re-initialized
                set_seed(s + 777)
                net = build_model("mlp", d_in).to(DEVICE)
                net, _ = train(net, m_imp, B.Xtr, B.ytr, seed=s, **TR)
                nets.append(("imp_reinit", rnd, net))
            dense_g = None
            for meth, rnd, net in nets:
                bacc = evaluate(net, B.Xte, B.yte)["bacc"]
                ref = logged.get((s, meth, rnd))
                ig, cerr = integrated_gradients(net, X_att)
                g = summarize(ig)
                if meth == "dense":
                    dense_g = g
                imps[(src, s, meth, rnd)] = g
                srt = np.sort(g)[::-1]
                recs.append(dict(src=src, seed=s, method=meth, round=rnd, sparsity=LEVEL.get(rnd, 0.0),
                                 bacc=bacc, bacc_logged=ref, valid=bacc >= 0.9,
                                 share_changeable=float(g[ch].sum()), top5_share=float(srt[:5].sum()),
                                 eff_features=float(1 / (g ** 2).sum()),
                                 spearman_vs_dense=float(spearmanr(g, dense_g).correlation),
                                 ig_completeness_err=cerr))
                print(f"{src} s{s} {meth:10s} r{rnd:2d} bacc={bacc:.4f} (logged {ref}) "
                      f"chg={recs[-1]['share_changeable']:.3f} top5={recs[-1]['top5_share']:.3f} "
                      f"rho={recs[-1]['spearman_vs_dense']:.3f} cerr={cerr:.3f}", flush=True)
        json.dump(dict(features=feats, changeable=ch.tolist()), open(os.path.join(OUT, f"{src}_features.json"), "w"))
    df = pd.DataFrame(recs)
    df.to_json(os.path.join(OUT, "xai_records.json"), orient="records", indent=1)
    np.save(os.path.join(OUT, "importance.npy"), {"|".join(map(str, k)): v for k, v in imps.items()}, allow_pickle=True)


if __name__ == "__main__":
    main()
