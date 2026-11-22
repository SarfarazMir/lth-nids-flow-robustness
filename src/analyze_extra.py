"""Extra analyses for the manuscript: sensitivity, effect sizes, omnibus tests and efficiency.

1. Sensitivity of the main finding (IMP ticket vs dense, CIC-IDS2017) to the collapse threshold.
2. Evasion success per attack budget rho (dense, IMP, ERK, SNIP; CIC-IDS2017).
3. Effect sizes: matched-pairs rank-biserial correlation for the main pooled comparisons.
4. Friedman omnibus test over the seven standard-training conditions (blocks = seed x level,
   79-94.5 %, where no condition collapsed), with Kendall's W.
5. Efficiency of IMP tickets: non-zero weights, FLOPs, storage (dense fp32 vs CSR), and measured
   single-thread CPU latency of a dense-masked vs a CSR sparse forward pass.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from analyze import RAW, TAB, load

SEEDS = [0, 1, 2, 3, 4]
LEVEL = {4: "59.0", 7: "79.0", 10: "89.3", 13: "94.5", 16: "97.2", 20: "98.8"}
OUT = []
say = OUT.append


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[(~np.isnan(d)) & (d != 0)]
    if not len(d):
        return np.nan
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def rq1():
    df = load("rq1")
    return df[df["seed"].isin(SEEDS)]


def sensitivity(df):
    say("# 1. Sensitivity of 'IMP ticket − dense' (flow ESR-AUC, CIC-IDS2017, levels 79–97.2 %) to the collapse threshold\n")
    d = df[df["src"] == "ids17"]
    dn = d[d["method"] == "dense"].set_index("seed")["flow/ESR_AUC"]
    rows = []
    for thr in (0.0, 0.80, 0.90, 0.95, 0.98):
        e = d[(d["method"] == "imp") & d["round"].isin([7, 10, 13, 16]) & (d["src_test/bacc"] >= thr)]
        diff = (e["flow/ESR_AUC"].to_numpy() - dn.reindex(e["seed"]).to_numpy())
        diff = diff[~np.isnan(diff)]
        rows.append({"threshold (bacc ≥)": thr, "pairs": len(diff), "mean Δ (pp)": 100 * diff.mean(),
                     "ticket worse": f"{int((diff > 0).sum())}/{len(diff)}",
                     "Wilcoxon p": wilcoxon(diff).pvalue, "rank-biserial r": rank_biserial(diff)})
    say(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4g"))


def per_budget(df):
    say("\n\n# 2. Evasion success rate (%) per flow-attack budget ρ, CIC-IDS2017 (mean ± sd over valid networks)\n")
    d = df[(df["src"] == "ids17") & (df["src_test/bacc"] >= 0.9)]
    rows = []
    for meth, rnd in [("dense", 0), ("imp", 13), ("erk", 13), ("snip", 13), ("imp", 16), ("erk", 16), ("snip", 16)]:
        e = d[(d["method"] == meth) & (d["round"] == rnd)]
        row = {"network": meth if rnd == 0 else f"{meth} {LEVEL[rnd]}%", "n": len(e)}
        for rho in (0.1, 0.25, 0.5, 1.0, 2.0):
            v = 100 * e[f"flow/ESR@{rho}"]
            row[f"ρ={rho}"] = f"{v.mean():.1f} ± {v.std():.1f}"
        rows.append(row)
    say(pd.DataFrame(rows).to_markdown(index=False))


def effect_sizes(df):
    say("\n\n# 3. Effect sizes (matched-pairs rank-biserial r; +1 = first condition always higher), flow ESR-AUC, valid networks, levels 79–97.2 %\n")
    rows = []
    for src, d in df[df["src_test/bacc"] >= 0.9].groupby("src"):
        imp = d[(d["method"] == "imp") & d["round"].isin([7, 10, 13, 16])].set_index(["seed", "round"])["flow/ESR_AUC"]
        dn = d[d["method"] == "dense"].set_index("seed")["flow/ESR_AUC"]
        rows.append({"src": src, "comparison": "imp − dense",
                     "r": rank_biserial(imp.to_numpy() - dn.reindex(imp.index.get_level_values("seed")).to_numpy())})
        for m in ["imp_reinit", "random_lw", "erk", "snip", "omp_ft"]:
            o = d[(d["method"] == m) & d["round"].isin([7, 10, 13, 16])].set_index(["seed", "round"])["flow/ESR_AUC"]
            j = pd.concat([o, imp], axis=1, keys=["o", "i"]).dropna()
            rows.append({"src": src, "comparison": f"{m} − imp", "r": rank_biserial(j["o"] - j["i"])})
    say(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"))


def friedman(df):
    say("\n\n# 4. Friedman test over 7 standard-training conditions (flow ESR-AUC; blocks = seed × level 79/89.3/94.5 %)\n")
    rows = []
    meths = ["imp", "imp_reinit", "random_lw", "erk", "snip", "omp_ft"]
    for src, d in df.groupby("src"):
        blocks = []
        for s in SEEDS:
            dn = d[(d["method"] == "dense") & (d["seed"] == s)]["flow/ESR_AUC"].iloc[0]
            for rnd in (7, 10, 13):
                vals = [dn]
                for m in meths:
                    v = d[(d["method"] == m) & (d["seed"] == s) & (d["round"] == rnd)]["flow/ESR_AUC"]
                    vals.append(v.iloc[0] if len(v) else np.nan)
                blocks.append(vals)
        B = np.array(blocks)
        B = B[~np.isnan(B).any(1)]
        stat, p = friedmanchisquare(*B.T)
        n, k = B.shape
        W = stat / (n * (k - 1))
        mean_rank = rankdata(B, axis=1).mean(0)
        rows.append({"src": src, "blocks": n, "χ²": stat, "p": p, "Kendall W": W,
                     **{f"rank {m}": r for m, r in zip(["dense"] + meths, mean_rank)}})
    say(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"))
    say("\n(Mean rank 1 = least evadable, 7 = most evadable.)")


def efficiency():
    say("\n\n# 5. Efficiency of IMP tickets (CIC-IDS2017, seed 0; single CPU thread)\n")
    from lthlib.models import build_model
    torch.set_num_threads(1)
    st = torch.load(os.path.join(RAW, "rq1", "ids17_mlp_s0_state.pt"), map_location="cpu", weights_only=False)
    rows = []
    for rnd in [0, 4, 7, 10, 13, 16]:
        net = build_model("mlp", 76)
        net.load_state_dict(st["dense"] if rnd == 0 else st["imp_weights"][rnd])
        net.eval()
        lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
        nnz = sum(int((m.weight != 0).sum()) for m in lin)
        tot = sum(m.weight.numel() for m in lin)
        dense_bytes = 4 * tot
        csr_bytes = sum(4 * int((m.weight != 0).sum()) * 2 + 4 * (m.weight.shape[0] + 1) for m in lin)
        res = {}
        for bs in (1, 1024):
            x = torch.randn(bs, 76)
            with torch.no_grad():
                for _ in range(20):
                    net(x)
                ts = []
                for _ in range(200 if bs == 1 else 50):
                    t = time.perf_counter(); net(x); ts.append(time.perf_counter() - t)
            res[f"dense-kernel latency bs={bs} (ms)"] = 1e3 * np.median(ts)
            # sparse forward: CSR matmul for every Linear (BN/ReLU unchanged), eval mode
            W = [m.weight.detach().to_sparse_csr() for m in lin]
            b = [m.bias.detach() for m in lin]

            def sparse_forward(z):
                li = 0
                for mod in list(net.body) + [net.head]:
                    if isinstance(mod, torch.nn.Linear):
                        nonlocal_i = li
                        z = (W[nonlocal_i] @ z.T).T + b[nonlocal_i]
                        li += 1
                    else:
                        z = mod(z)
                return z

            with torch.no_grad():
                assert torch.allclose(sparse_forward(x), net(x), atol=1e-4)
                for _ in range(20):
                    sparse_forward(x)
                ts = []
                for _ in range(200 if bs == 1 else 50):
                    t = time.perf_counter(); sparse_forward(x); ts.append(time.perf_counter() - t)
            res[f"CSR latency bs={bs} (ms)"] = 1e3 * np.median(ts)
        rows.append({"pruned %": "0 (dense)" if rnd == 0 else LEVEL[rnd], "non-zero weights": nnz,
                     "MFLOPs / flow": 2 * nnz / 1e6, "storage dense fp32 (KB)": dense_bytes / 1024,
                     "storage CSR (KB)": csr_bytes / 1024, **res})
    say(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"))


if __name__ == "__main__":
    df = rq1()
    sensitivity(df)
    per_budget(df)
    effect_sizes(df)
    friedman(df)
    efficiency()
    open(os.path.join(TAB, "extra_summary.md"), "w", encoding="utf-8").write("\n".join(OUT))
    print("\n".join(OUT))
