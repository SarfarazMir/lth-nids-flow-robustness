"""Extra result figures (learning curves, confusion matrices, budget curves).

  res_accuracy_vs_rounds.png   clean balanced accuracy of the IMP ticket at all 20 rounds (both datasets)
  res_esr_vs_budget.png        evasion success vs flow-attack budget rho (CIC-IDS2017)
  res_confusion.png            confusion matrices + attack outcome, dense vs 97.2 % ticket (seed 0)
  res_learning_curves.png      per-epoch curves (standard training), both datasets
  res_learning_curves_at.png   per-epoch curves (adversarial training), CIC-IDS2017
Inputs: results/raw/rq1/*.json, results/raw/curves/*.json
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
RAW = os.path.join(ROOT, "raw")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
SEEDS = range(5)
NAME = {"ids17": "CIC-IDS2017", "ddos19_d1": "CIC-DDoS2019"}
COL = {"dense": "#333333", "imp": "#C0392B", "erk": "#2471A3", "snip": "#7D3C98", "dense_at": "#333333",
       "aimp": "#C0392B"}


def recs(src):
    out = []
    for s in SEEDS:
        p = os.path.join(RAW, "rq1", f"{src}_mlp_s{s}.json")
        out += [dict(r, seed=s) for r in json.load(open(p))["records"]]
    return out


def accuracy_vs_rounds():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, src in zip(axes, ("ids17", "ddos19_d1")):
        R = recs(src)
        dense = np.array([r["src_test/bacc"] for r in R if r["method"] == "dense"])
        sp, m, sd = [], [], []
        for rnd in range(1, 21):
            v = [r["src_test/bacc"] for r in R if r["method"] == "imp" and r["round"] == rnd]
            s = [r["sparsity"] for r in R if r["method"] == "imp" and r["round"] == rnd]
            sp.append(100 * np.mean(s)); m.append(np.mean(v)); sd.append(np.std(v))
        sp, m, sd = map(np.array, (sp, m, sd))
        ax.axhspan(dense.mean() - dense.std(), dense.mean() + dense.std(), color="grey", alpha=0.25)
        ax.axhline(dense.mean(), color="#333", lw=1, label=f"dense ({dense.mean():.4f})")
        ax.errorbar(sp, m, yerr=sd, color=COL["imp"], marker="o", ms=4, capsize=2, label="IMP ticket")
        ax.axhline(0.90, color="#B26A00", ls="--", lw=1, label="collapse threshold 0.90")
        ax.set_xlabel("weights removed (%)")
        ax.set_ylabel("clean balanced accuracy")
        ax.set_title(f"({'ab'[src != 'ids17']}) {NAME[src]}")
        ax.set_ylim(0.45, 1.01)
        ax.legend(loc="lower left", fontsize=8.5)
        ins = ax.inset_axes([0.42, 0.35, 0.36, 0.33])
        k = sp <= 95
        ins.errorbar(sp[k], m[k], yerr=sd[k], color=COL["imp"], marker="o", ms=3, capsize=2)
        ins.axhline(dense.mean(), color="#333", lw=1)
        ins.set_title("zoom: 20-95 %", fontsize=8)
        ins.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "res_accuracy_vs_rounds.png"), dpi=220)
    plt.close(fig)


def esr_vs_budget():
    rhos = [0.1, 0.25, 0.5, 1.0, 2.0]
    R = recs("ids17")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, rnd, lev in zip(axes, (13, 16), ("94.5", "97.2")):
        for meth, lab in (("dense", "dense"), ("imp", f"IMP ticket {lev} %"), ("erk", f"ERK {lev} %"),
                          ("snip", f"SNIP {lev} %")):
            rows = [r for r in R if r["method"] == meth and (meth == "dense" or r["round"] == rnd)
                    and r["src_test/bacc"] >= 0.9 and "flow/ESR@0.1" in r]
            M = np.array([[100 * r[f"flow/ESR@{x}"] for x in rhos] for r in rows])
            ax.errorbar(rhos, M.mean(0), yerr=M.std(0, ddof=1), marker="o", capsize=3, label=lab, color=COL[meth],
                        ls="--" if meth == "dense" else "-")
        ax.set_xscale("log")
        ax.set_xticks(rhos)
        ax.set_xticklabels([str(x) for x in rhos])
        ax.set_xlabel("attack budget rho (extra traffic relative to the flow)")
        ax.set_title(f"({'ab'[rnd == 16]}) {lev} % of weights removed")
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("evasion success rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "res_esr_vs_budget.png"), dpi=220)
    plt.close(fig)


def confusion():
    d = json.load(open(os.path.join(RAW, "curves", "confusion_ids17_s0.json")))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, key, title in zip(axes[:2], ("dense", "imp_97.2"), ("(a) Dense network", "(b) IMP ticket, 97.2 % removed")):
        M = np.array(d[key]["clean_test"])
        ax.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:,}\n({100 * M[i, j] / M[i].sum():.1f} %)", ha="center", va="center",
                        color="white" if (i == j) else "black", fontsize=9.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["benign", "attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["benign", "attack"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
    ax = axes[2]
    names = ["Dense", "Ticket 97.2 %"]
    caught = [d["dense"]["detected_before"], d["imp_97.2"]["detected_before"]]
    ev = [d["dense"]["evaded_rho1"], d["imp_97.2"]["evaded_rho1"]]
    n = d["dense"]["attack_flows"]
    still = [c - e for c, e in zip(caught, ev)]
    missed = [n - c for c in caught]
    x = np.arange(2)
    ax.bar(x, still, color="#2E7D32", label="still detected after attack")
    ax.bar(x, ev, bottom=still, color="#C0392B", label="evaded by the attack (rho = 1)")
    ax.bar(x, missed, bottom=np.array(still) + np.array(ev), color="#AAAAAA", label="missed even without attack")
    for i in range(2):
        ax.text(i, still[i] / 2, f"{still[i]:,}", ha="center", color="white", fontsize=9)
        ax.text(i, still[i] + ev[i] / 2, f"{ev[i]:,}\n({100 * ev[i] / caught[i]:.1f} % of caught)", ha="center",
                color="white", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("of 3,000 attack flows")
    ax.set_title("(c) Outcome under the flow attack", fontsize=10)
    ax.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "res_confusion.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _curves(pattern):
    data = {}
    for p in sorted(glob.glob(os.path.join(RAW, "curves", pattern))):
        for r in json.load(open(p)):
            data.setdefault((r["method"], r["round"]), []).append(r)
    return data


def learning_curves():
    shades = {4: "#F1948A", 7: "#E74C3C", 10: "#C0392B", 13: "#922B21", 16: "#641E16", 20: "#2C0A08"}
    lev = {4: "59", 7: "79", 10: "89.3", 13: "94.5", 16: "97.2", 20: "98.8"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for row, src in enumerate(("ids17", "ddos19_d1")):
        D = _curves(f"std_{src}_s*.json")
        for col, (key, lab) in enumerate((("train_loss", "training loss"), ("val_loss", "validation loss"),
                                          ("val_bacc", "validation balanced accuracy"))):
            ax = axes[row, col]
            for (m, rnd), runs in sorted(D.items(), key=lambda kv: (kv[0][0] != "dense", kv[0][0], kv[0][1])):
                if m == "erk":
                    continue
                Y = np.array([[h[key] for h in r["history"]] for r in runs])
                ep = np.arange(1, Y.shape[1] + 1)
                c = "#222" if m == "dense" else shades[rnd]
                lbl = "dense" if m == "dense" else f"ticket {lev[rnd]} %"
                ax.plot(ep, Y.mean(0), color=c, lw=2 if m == "dense" else 1.4, ls="--" if m == "dense" else "-",
                        marker="o", ms=3, label=lbl)
                if rnd != 20:
                    ax.fill_between(ep, Y.mean(0) - Y.std(0, ddof=1), Y.mean(0) + Y.std(0, ddof=1), color=c, alpha=0.12)
            ax.set_xlabel("epoch")
            ax.set_title(f"{NAME[src]}: {lab}", fontsize=10)
            if key == "val_bacc":
                ax.set_ylim(0.85 if src == "ids17" else 0.9, 1.0)
            else:
                ax.set_yscale("log")
            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "res_learning_curves.png"), dpi=220)
    plt.close(fig)

    D = _curves("adv_ids17_s*.json")
    if not D:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    lev2 = {0: "dense_at", 7: "A-IMP 79 %", 13: "A-IMP 94.5 %", 16: "A-IMP 97.2 %"}
    cs = {0: "#222", 7: "#E74C3C", 13: "#922B21", 16: "#2C0A08"}
    for ax, (key, lab) in zip(axes, (("train_loss", "(a) training loss (tampered batches)"),
                                     ("val_loss", "(b) validation loss (clean data)"),
                                     ("val_bacc", "(c) validation balanced accuracy"))):
        for (m, rnd), runs in sorted(D.items(), key=lambda kv: kv[0][1]):
            Y = np.array([[h[key] for h in r["history"]] for r in runs])
            ep = np.arange(1, Y.shape[1] + 1)
            ax.plot(ep, Y.mean(0), color=cs[rnd], marker="o", ms=3, ls="--" if rnd == 0 else "-", label=lev2[rnd])
            ax.fill_between(ep, Y.mean(0) - Y.std(0, ddof=1), Y.mean(0) + Y.std(0, ddof=1), color=cs[rnd], alpha=0.12)
        ax.set_xlabel("epoch")
        ax.set_title(lab, fontsize=10)
        if key != "val_bacc":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.8, 1.0)
            ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "res_learning_curves_at.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    accuracy_vs_rounds()
    esr_vs_budget()
    confusion()
    learning_curves()
    print("done")
