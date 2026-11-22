"""Follow-up analyses that need no new training runs (except the corrected immutable baseline).

1. Gradient-masking checks (seed 0, where the surrogate and sanity attacks were run):
   transfer attack <= direct attack, FGSM <= PGD, CW-loss PGD ~ CE-loss PGD.
2. Mechanism: per-layer density of each pruning method and the input-feature connectivity of the
   IMP tickets (how many first-layer weights reach attacker-changeable features).
3. Residual of the 'ignore changeable features' baseline: which features the flow attack can change
   beyond the single probe used to build the mask (k=1, s=700 B, dd=0.1 s).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from analyze import RAW, TAB, load

OUT = []
say = OUT.append
SEEDS = [0, 1, 2, 3, 4]
LEVEL = {4: "59.0", 7: "79.0", 10: "89.3", 13: "94.5", 16: "97.2", 20: "98.8"}


def gradient_masking():
    say("# 1. Gradient-masking checks (seed 0 networks; n = networks with valid clean accuracy)\n")
    rows = []
    for kind in ("rq1",):
        df = load(kind)
        df = df[(df["seed"] == 0) & (df["src_test/bacc"] >= 0.9)]
        for src, d in df.groupby("src"):
            for tm, grid in (("unconstrained", (0.05, 0.1, 0.15, 0.25)), ("constrained", (0.25, 0.5, 1.0, 2.0))):
                mid = grid[1]
                n = 0
                viol_tr = viol_fg = 0
                gap_tr, gap_fg, gap_cw = [], [], []
                for _, r in d.iterrows():
                    if pd.isna(r.get(f"{tm}/ESR@{mid}")):
                        continue
                    n += 1
                    for e in grid:
                        dt = r.get(f"{tm}/transfer_ESR@{e}")
                        if pd.notna(dt):
                            gap_tr.append(dt - r[f"{tm}/ESR@{e}"])
                            viol_tr += dt > r[f"{tm}/ESR@{e}"] + 0.01
                    fg = r.get(f"{tm}/fgsm_ESR@{mid}")
                    if pd.notna(fg):
                        gap_fg.append(fg - r[f"{tm}/ESR@{mid}"])
                        viol_fg += fg > r[f"{tm}/ESR@{mid}"] + 0.01
                    cw = r.get(f"{tm}/cw_ESR@{mid}")
                    if pd.notna(cw):
                        gap_cw.append(cw - r[f"{tm}/ESR@{mid}"])
                rows.append({"source": src, "threat model": tm, "networks": n,
                             "transfer − direct ESR (mean, pp)": 100 * np.mean(gap_tr),
                             "max": 100 * np.max(gap_tr),
                             "cases transfer > direct (+1pp)": f"{viol_tr}/{len(gap_tr)}",
                             "FGSM − PGD (mean, pp)": 100 * np.mean(gap_fg),
                             "cases FGSM > PGD (+1pp)": f"{viol_fg}/{len(gap_fg)}",
                             "CW − CE PGD (mean, pp)": 100 * np.mean(gap_cw)})
    say(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".1f"))


def layer_densities():
    say("\n\n# 2a. Layer-wise density (% of weights kept per layer; mean over 5 seeds)\n")
    say("Layers: L1 = input→512 (38,912 weights), L2 = 512→512 (262,144), L3 = 512→256 (131,072), "
        "L4 = 256→2 (512).\n")
    names = {"body.0.weight": "L1", "body.3.weight": "L2", "body.6.weight": "L3", "head.weight": "L4"}
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "rq1", "*.json"))):
        d = json.load(open(f))
        if d["args"]["seed"] not in SEEDS:
            continue
        for r in d["records"]:
            if "layer_density" in r and r["round"] in (7, 10, 13, 16):
                rows.append({"src": d["args"]["src"], "method": r["method"], "round": r["round"],
                             **{names[k]: 100 * v for k, v in r["layer_density"].items()}})
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(["imp", "random_lw", "erk", "snip", "omp_ft"])]
    g = df.groupby(["src", "round", "method"])[["L1", "L2", "L3", "L4"]].mean().reset_index()
    g["pruned %"] = g["round"].map(LEVEL)
    for c in ["L1", "L2", "L3", "L4"]:
        g[c] = g[c].map(lambda v: f"{v:.3f}" if v < 1 else f"{v:.1f}")
    # kept-weight counts in the narrowest layer
    say(g[["src", "pruned %", "method", "L1", "L2", "L3", "L4"]].to_markdown(index=False))
    df["L2 weights kept"] = df["L2"] / 100 * 262144
    df["L3 weights kept"] = df["L3"] / 100 * 131072
    h = df[df["round"] == 16].groupby(["src", "method"])[["L2 weights kept", "L3 weights kept"]].agg(["mean", "min"])
    say("\n### Weights left in the hidden layers at 97.2% pruning (mean and minimum over seeds)\n")
    say(h.round(0).to_markdown())


def input_connectivity(mutable):
    say("\n\n# 2b. Input connectivity of IMP tickets (first layer, CIC-IDS2017 and CIC-DDoS2019)\n")
    say(f"Attacker-changeable features: {len(mutable)} of 76 (share of first-layer weights in a dense "
        f"or uniformly random mask: {100 * len(mutable) / 76:.1f}%).\n")
    rows = []
    for src in ("ids17", "ddos19_d1"):
        for s in SEEDS:
            st = torch.load(os.path.join(RAW, "rq1", f"{src}_mlp_s{s}_state.pt"), map_location="cpu", weights_only=False)
            W = st["dense"]["body.0.weight"].abs()  # trained dense first layer (512, 76)
            for rnd in (7, 10, 13, 16):
                m = st["imp_masks"][rnd]["body.0.weight"].bool()
                per_feat = m.sum(0).numpy()
                rows.append({"src": src, "seed": s, "pruned %": LEVEL[rnd],
                             "first-layer density %": 100 * m.float().mean().item(),
                             "inputs fully cut": int((per_feat == 0).sum()),
                             "changeable inputs cut": int((per_feat[mutable] == 0).sum()),
                             "share of kept L1 weights on changeable inputs %": 100 * per_feat[mutable].sum() / per_feat.sum(),
                             "dense: share of |W1| on changeable inputs %": 100 * W[:, mutable].sum().item() / W.sum().item()})
    df = pd.DataFrame(rows)
    g = df.groupby(["src", "pruned %"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    say(g.to_markdown(index=False, floatfmt=".1f"))


def mutability_probe():
    """Which features can the flow attack change at all? Probe many (k, s, dd) on all training flows."""
    from lthlib.common import Domain, make_pre
    from lthlib.flowattack import FlowAttack, MSS
    say("\n\n# 3. Features the flow attack can change (CIC-IDS2017 training flows)\n")
    src = Domain("ids17")
    pre = make_pre(src)
    fa = FlowAttack(src.features, pre, "cpu")
    X = torch.tensor(src.splits["train"].X, dtype=torch.float64)
    one = torch.ones(len(X), dtype=torch.float64)
    # the probe used to build the immutable baseline (exp_robust.py): first 2000 flows, k=1, s=700, dd=1e5
    Xp = X[:2000]
    o = torch.ones(2000, dtype=torch.float64)
    probe = (fa.apply(Xp, o, 700 * o, 1e5 * o) != Xp).any(0)
    kmax, ddmax = fa.limits(X, 2.0)
    changed = torch.zeros(X.shape[1], dtype=torch.bool)
    for k in (one, kmax):
        for s in (0.0, 1.0, 700.0, MSS):
            for dd in (0 * one, 1e5 * one, ddmax):
                changed |= (fa.apply(X, k, s * one, dd) != X).any(0)
    extra = [f for f, c, p in zip(src.features, changed.tolist(), probe.tolist()) if c and not p]
    say(f"- Probe used for the baseline marks {int(probe.sum())} features as changeable.")
    say(f"- A wider probe (k ∈ {{1, max}}, payload ∈ {{0, 1, 700, 1460}} B, delay ∈ {{0, 0.1 s, max}}, all "
        f"{len(X):,} training flows) marks {int(changed.sum())}.")
    say(f"- Features the baseline kept although the attack can change them: {extra}")
    for f in extra:
        j = src.features.index(f)
        n = 0
        for s in (0.0, 1.0, 700.0, MSS):
            n = max(n, int((fa.apply(X, one, s * one, 0 * one)[:, j] != X[:, j]).sum()))
        say(f"  - {f}: changes for up to {n:,} of {len(X):,} training flows ({100 * n / len(X):.1f}%) with one injected packet")
    return [i for i, c in enumerate(changed.tolist()) if c], probe


if __name__ == "__main__":
    gradient_masking()
    layer_densities()
    mut, probe = mutability_probe()
    input_connectivity([i for i, c in enumerate(probe.tolist()) if c])
    open(os.path.join(TAB, "mechanism_summary.md"), "w", encoding="utf-8").write("\n".join(OUT))
    print("\n".join(OUT))
