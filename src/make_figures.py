"""Publication figures for the manuscript (results/figures/paper_fig*.png, 300 dpi).
Markers and line styles differ per method so that every figure is readable in greyscale."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from analyze import FIG, RAW, load

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 300, "savefig.bbox": "tight"})
SEEDS = [0, 1, 2, 3, 4]
STY = {"dense": ("k", "--", ""), "imp": ("#1f4e9c", "-", "o"), "imp_reinit": ("#d9822b", "-", "s"),
       "random_lw": ("#3a923a", ":", "^"), "erk": ("#b0302a", "-.", "D"), "snip": ("#7d4fa3", "--", "v"),
       "omp_ft": ("#7a5a3a", ":", "x")}
NAME = {"dense": "Dense", "imp": "IMP ticket (LTH)", "imp_reinit": "IMP mask, re-init.", "random_lw": "Random (IMP layer density)",
        "erk": "ERK random", "snip": "SNIP", "omp_ft": "One-shot magnitude + FT"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300)
    plt.close(fig)


def fig1_concept():
    """Kept weights per layer: dense vs IMP vs ERK at 97.2 % (CIC-IDS2017, mean over seeds)."""
    # exact per-layer densities from the saved runs (mean over seeds), not rounded percentages
    def raw_density(method, sp=0.972):
        rows = []
        for s in SEEDS:
            with open(os.path.join(RAW, "rq1", f"ids17_mlp_s{s}.json")) as fh:
                for e in json.load(fh)["records"]:
                    if e["method"] == method and abs(float(e["sparsity"]) - sp) < 0.002 and "layer_density" in e:
                        rows.append([e["layer_density"][k] for k in ("body.0.weight", "body.3.weight", "body.6.weight", "head.weight")])
        assert len(rows) == len(SEEDS), (method, len(rows))
        return list(100 * np.mean(rows, axis=0))
    dens = {"(a) Dense": [100, 100, 100, 100], "(b) IMP ticket, 97.2 % pruned": raw_density("imp"),
            "(c) ERK random, 97.2 % pruned": raw_density("erk")}
    sizes = [76 * 512, 512 * 512, 512 * 256, 256 * 2]
    widths = [76, 512, 512, 256, 2]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (name, d) in zip(axes, dens.items()):
        xs = np.arange(5)
        for li in range(4):
            kept = d[li] / 100 * sizes[li]
            lw = 0.4 + 7 * np.log10(1 + kept) / np.log10(1 + 262144)
            ax.fill_between([xs[li] + 0.12, xs[li + 1] - 0.12], 0.5 - lw / 22, 0.5 + lw / 22, color="#1f4e9c", alpha=0.8)
            ax.text((xs[li] + xs[li + 1]) / 2, 0.5 + lw / 22 + 0.05 + 0.07 * (li % 2), f"{kept:,.0f}", ha="center", fontsize=6.3)
        for x, w in zip(xs, widths):
            h = 0.12 + 0.5 * np.log10(w) / np.log10(512)
            ax.add_patch(FancyBboxPatch((x - 0.1, 0.5 - h / 2), 0.2, h, boxstyle="round,pad=0.01", fc="#e7ecf3", ec="k", lw=0.6))
            ax.text(x, 0.5 - h / 2 - 0.07, str(w), ha="center", fontsize=6.5)
        ax.set_xlim(-0.4, 4.4); ax.set_ylim(0, 1.05); ax.axis("off")
        ax.set_title(name, fontsize=8.5)
    fig.text(0.5, -0.02, "Numbers between layers: weights kept (mean over 5 seeds, CIC-IDS2017). Layer sizes under boxes.",
             ha="center", fontsize=7.5)
    save(fig, "paper_fig1_concept.png")


def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.axis("off")
    boxes = [("1. Data", "CIC-IDS2017,\nCIC-DDoS2019;\n76 shared features;\n70/15/15 split"),
             ("2. Train & prune", "5 seeds; IMP ticket,\n3 controls, ERK, SNIP;\nadversarial: A-IMP,\nOMP, LAMP, HYDRA"),
             ("3. Attack", "flow-consistent:\npackets k, payload s,\ndelay; 16 recomputed\nrelations; ρ ≤ 2"),
             ("4. Transfer", "mask / init /\nweights reused;\n1 % or 100 %\ntarget labels"),
             ("5. Analysis", "paired Wilcoxon\n+ Holm, Friedman;\nlayer densities;\nIntegrated Gradients")]
    w, gap = 0.18, 0.03
    for n, (head, txt) in enumerate(boxes):
        x = n * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.05), w, 0.9, boxstyle="round,pad=0.01", fc="#e7ecf3", ec="#1f3864", lw=0.8))
        ax.text(x + w / 2, 0.83, head, ha="center", va="center", fontsize=7.5, weight="bold", color="#1f3864")
        ax.text(x + w / 2, 0.42, txt, ha="center", va="center", fontsize=5.9, linespacing=1.35)
        if n < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.5), xytext=(x + w + 0.006, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#1f3864", lw=1))
    ax.set_xlim(-0.01, 1.04); ax.set_ylim(0, 1)
    save(fig, "paper_fig2_pipeline.png")


def fig3_main():
    df = load("rq1")
    df = df[df["seed"].isin(SEEDS)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, src, title in [(axes[0], "ids17", "(a) CIC-IDS2017"), (axes[1], "ddos19_d1", "(b) CIC-DDoS2019")]:
        d = df[(df["src"] == src) & (df["src_test/bacc"] >= 0.9)]
        dn = d[d["method"] == "dense"]["flow/ESR_AUC"] * 100
        ax.axhspan(dn.mean() - dn.std(), dn.mean() + dn.std(), color="0.85", lw=0)
        ax.axhline(dn.mean(), color="k", ls="--", lw=0.9, label="Dense (± sd)")
        for m in ["imp", "imp_reinit", "random_lw", "erk", "snip", "omp_ft"]:
            e = d[(d["method"] == m) & d["round"].isin([4, 7, 10, 13, 16])]
            s = e.groupby("round").agg(x=("sparsity", "mean"), y=("flow/ESR_AUC", "mean"), sd=("flow/ESR_AUC", "std"), n=("seed", "nunique"))
            s = s[s["n"] >= 3]
            c, ls, mk = STY[m]
            ax.errorbar(100 * s["x"], 100 * s["y"], yerr=100 * s["sd"], color=c, ls=ls, marker=mk, ms=3.5, lw=1, capsize=2, label=NAME[m])
        ax.set_xlabel("weights removed (%)"); ax.set_title(title)
    axes[0].set_ylabel("flow ESR-AUC (%)  ↓ more robust")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    # (c) per-budget curve
    ax = axes[2]
    d = df[(df["src"] == "ids17") & (df["src_test/bacc"] >= 0.9)]
    rhos = [0.1, 0.25, 0.5, 1.0, 2.0]
    for m, rnd in [("dense", 0), ("imp", 16), ("erk", 16), ("snip", 16)]:
        e = d[(d["method"] == m) & (d["round"] == rnd)]
        y = [100 * e[f"flow/ESR@{r}"].mean() for r in rhos]
        sd = [100 * e[f"flow/ESR@{r}"].std() for r in rhos]
        c, ls, mk = STY[m]
        ax.errorbar(rhos, y, yerr=sd, color=c, ls=ls if m != "dense" else "--", marker=mk or "o", ms=3.5, lw=1, capsize=2,
                    label=NAME[m] + ("" if m == "dense" else " 97.2 %"))
    ax.set_xscale("log"); ax.set_xticks(rhos); ax.set_xticklabels([str(r) for r in rhos])
    ax.set_xlabel("attack budget ρ"); ax.set_ylabel("evasion success (%)"); ax.set_title("(c) CIC-IDS2017, per budget")
    ax.legend(fontsize=6.3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "paper_fig3_robustness.png")


def fig4_layers():
    rows = []
    names = {"body.0.weight": "L1 76→512", "body.3.weight": "L2 512→512", "body.6.weight": "L3 512→256", "head.weight": "L4 256→2"}
    for s in SEEDS:
        d = json.load(open(os.path.join(RAW, "rq1", f"ids17_mlp_s{s}.json")))
        for r in d["records"]:
            if "layer_density" in r and r["round"] in (13, 16) and r["method"] in ("imp", "erk", "snip"):
                for k, v in r["layer_density"].items():
                    rows.append(dict(method=r["method"], round=r["round"], layer=names[k], density=100 * v))
    df = pd.DataFrame(rows).groupby(["method", "round", "layer"])["density"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    hatches = {"imp": "", "erk": "///", "snip": "..."}
    for ax, rnd, t in [(axes[0], 13, "(a) 94.5 % of weights removed"), (axes[1], 16, "(b) 97.2 % of weights removed")]:
        layers = list(names.values())
        for i, m in enumerate(["imp", "erk", "snip"]):
            v = [df[(df["method"] == m) & (df["round"] == rnd) & (df["layer"] == L)]["density"].iloc[0] for L in layers]
            ax.bar(np.arange(4) + (i - 1) * 0.27, v, 0.27, color=STY[m][0], hatch=hatches[m], ec="k", lw=0.4, label=NAME[m])
        ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(layers, fontsize=7)
        ax.set_title(t); ax.axhline(100, color="0.6", lw=0.5)
    axes[0].set_ylabel("weights kept in layer (%)"); axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "paper_fig4_layers.png")


def fig4_layers_col():
    """Single-column (3.4 in) version of the layer-density figure for the IEEE layout."""
    rows = []
    names = {"body.0.weight": "L1", "body.3.weight": "L2", "body.6.weight": "L3", "head.weight": "L4"}
    for s in SEEDS:
        d = json.load(open(os.path.join(RAW, "rq1", f"ids17_mlp_s{s}.json")))
        for r in d["records"]:
            if "layer_density" in r and r["round"] in (13, 16) and r["method"] in ("imp", "erk", "snip"):
                for k, v in r["layer_density"].items():
                    rows.append(dict(method=r["method"], round=r["round"], layer=names[k], density=100 * v))
    df = pd.DataFrame(rows).groupby(["method", "round", "layer"])["density"].mean().reset_index()
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.5}):
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=True)
        hatches = {"imp": "", "erk": "///", "snip": "..."}
        short = {"imp": "IMP ticket", "erk": "ERK", "snip": "SNIP"}
        for ax, rnd, t in [(axes[0], 13, "(a) 94.5 % removed"), (axes[1], 16, "(b) 97.2 % removed")]:
            layers = list(names.values())
            for i, m in enumerate(["imp", "erk", "snip"]):
                v = [df[(df["method"] == m) & (df["round"] == rnd) & (df["layer"] == L)]["density"].iloc[0] for L in layers]
                ax.bar(np.arange(4) + (i - 1) * 0.27, v, 0.27, color=STY[m][0], hatch=hatches[m], ec="k", lw=0.3, label=short[m])
            ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(layers)
            ax.set_title(t); ax.axhline(100, color="0.6", lw=0.5)
        axes[0].set_ylabel("weights kept (%)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.08))
        fig.tight_layout(pad=0.3)
        save(fig, "paper_fig4_layers_col.png")


def fig5_xai():
    p = os.path.join(RAW, "xai", "xai_records.json")
    if not os.path.exists(p):
        print("no XAI records yet")
        return
    df = pd.DataFrame(json.load(open(p)))
    df = df[df["valid"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (col, lab) in zip(axes, [("share_changeable", "importance on attacker-\nchangeable features (%)"),
                                     ("top5_share", "importance in top-5 features (%)"),
                                     ("spearman_vs_dense", "Spearman ρ with dense\nfeature ranking")]):
        d = df[df["src"] == "ids17"]
        dn = d[d["method"] == "dense"][col]
        f = 1 if col == "spearman_vs_dense" else 100
        if col != "spearman_vs_dense":
            ax.axhspan(f * (dn.mean() - dn.std()), f * (dn.mean() + dn.std()), color="0.85", lw=0)
            ax.axhline(f * dn.mean(), color="k", ls="--", lw=0.9, label="Dense")
        for m in ["imp", "imp_reinit", "erk"]:
            e = d[d["method"] == m].groupby("sparsity")[col].agg(["mean", "std"])
            c, ls, mk = STY[m]
            ax.errorbar(e.index, f * e["mean"], yerr=f * e["std"], color=c, ls=ls, marker=mk, ms=3.5, lw=1, capsize=2, label=NAME[m])
        ax.set_xlabel("weights removed (%)"); ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Integrated-Gradients attributions on malicious CIC-IDS2017 test flows (5 seeds)", fontsize=8.5, y=1.02)
    fig.tight_layout()
    save(fig, "paper_fig5_xai.png")


def fig6_at():
    rob = load("robust")
    rob = rob[rob["seed"].isin(SEEDS) & (rob["src_test/bacc"] >= 0.9)]
    # drop the single-seed pilot runs (OMP/LAMP/HYDRA at 59 % and 79 %, seed 0 only)
    rob = rob[~(rob["method"].isin(["omp_at", "lamp_at", "hydra"]) & (rob["sparsity"] < 0.88))]
    std = load("rq1")
    std = std[(std["src"] == "ids17") & std["seed"].isin(SEEDS) & (std["src_test/bacc"] >= 0.9)]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    pts = [("Dense (standard)", std[std["method"] == "dense"], "k", "o"),
           ("IMP ticket (standard), 79–97.2 %", std[(std["method"] == "imp") & std["round"].isin([7, 10, 13, 16])], "#1f4e9c", "o"),
           ("Dense adversarial", rob[rob["method"] == "dense_at"], "#b0302a", "*"),
           ("A-IMP, 59–97.2 %", rob[rob["method"] == "aimp"], "#d9822b", "s"),
           ("OMP/LAMP/HYDRA + adv. FT", rob[rob["method"].isin(["omp_at", "lamp_at", "hydra"])], "#7d4fa3", "^"),
           ("Immutable-input (corrected)", rob[rob["method"] == "immutable_std_immfix"], "#3a923a", "D")]
    for lab, e, c, mk in pts:
        big = mk == "*"
        ax.scatter(100 * e["flow/ESR_AUC"], 100 * e["src_test/bacc"], s=60 if big else 14, c=c, marker=mk, label=lab,
                   alpha=0.85, lw=0.4 if big else 0, ec="k" if big else None, zorder=5 if big else 2)
    ax.set_xlabel("flow ESR-AUC (%)  ↓"); ax.set_ylabel("balanced accuracy (%)  ↑")
    ax.set_title("CIC-IDS2017: robustness vs accuracy (each point = one network)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    save(fig, "paper_fig6_adversarial.png")


def fig7_transfer():
    df = load("rq2")
    df = df[df["seed"].isin(SEEDS) & df["frac"].isin([0.01, 1.0]) & (df["tgt"] != "ddos19_d2") & df["round"].isin([0, 10, 13, 16])]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    conds = [("tkt_rewind", "#1f4e9c", "-", "o", "Ticket (rewind)"), ("tkt_reinit", "#d9822b", "-", "s", "Mask only"),
             ("tkt_ft", "#7a5a3a", ":", "x", "Sparse FT (source weights)"), ("rand_rewind", "#3a923a", ":", "^", "Random mask"),
             ("snip_tgt", "#7d4fa3", "--", "v", "SNIP on target")]
    for ax, ((src, tgt, frac), d) in zip(axes, df.groupby(["src", "tgt", "frac"])):
        sc = d[d["cond"] == "dense_scratch"].set_index("seed")["tgt_test/bacc"]
        for c, col, ls, mk, lab in conds:
            e = d[d["cond"] == c].copy()
            e["delta"] = 100 * (e["tgt_test/bacc"] - sc.reindex(e["seed"]).to_numpy())
            s = e.groupby("round").agg(x=("sparsity", "mean"), y=("delta", "mean"), sd=("delta", "std"))
            ax.errorbar(100 * s["x"], s["y"], yerr=s["sd"], color=col, ls=ls, marker=mk, ms=3, lw=1, capsize=2, label=lab)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_yticks([0, -1, -10, -50]); ax.set_yticklabels(["0", "−1", "−10", "−50"])
        pair = {"ddos19_d1": "DDoS", "ids17": "IDS17"}
        ax.set_title(f"{pair[src]}→{pair[tgt]}, {int(frac * 100)} % labels", fontsize=8)
        ax.set_xlabel("weights removed (%)")
    axes[0].set_ylabel("Δ balanced acc. vs dense\nfrom scratch (pp, symlog)")
    axes[0].legend(fontsize=5.8, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "paper_fig7_transfer.png")


if __name__ == "__main__":
    fig1_concept(); fig2_pipeline(); fig3_main(); fig4_layers(); fig4_layers_col(); fig5_xai(); fig6_at(); fig7_transfer()
    print("figures written")
