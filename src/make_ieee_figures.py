"""Compact vector figures (PDF, two-column page widths): fragility curve and study workflow.

  ieee_fragility.pdf   when tickets become easy to fool: clean accuracy (all 20 rounds) and flow ESR-AUC
                       (evaluated rounds) of the IMP ticket vs the dense network, CIC-IDS2017
  ieee_workflow.pdf    compact overall workflow of the study (data -> Experiments A/B/C -> evaluation -> analysis)
Inputs: results/raw/rq1/*.json
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
RAW = os.path.join(ROOT, "raw")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42})
COLLAPSE = 0.90


def recs(src):
    out = []
    for s in range(5):
        out += [dict(r, seed=s) for r in json.load(open(os.path.join(RAW, "rq1", f"{src}_mlp_s{s}.json")))["records"]]
    return out


def fragility():
    R = recs("ids17")
    dense = {r["seed"]: r for r in R if r["method"] == "dense"}
    d_acc = np.array([r["src_test/bacc"] for r in dense.values()])
    d_esr = 100 * np.array([r["flow/ESR_AUC"] for r in dense.values()])
    rounds = np.arange(1, 21)
    acc_m, acc_s, sp = [], [], []
    for rnd in rounds:
        rows = [r for r in R if r["method"] == "imp" and r["round"] == rnd]
        v = np.array([r["src_test/bacc"] for r in rows])
        acc_m.append(v.mean()); acc_s.append(v.std(ddof=1)); sp.append(100 * np.mean([r["sparsity"] for r in rows]))
    acc_m, acc_s = np.array(acc_m), np.array(acc_s)
    ev_r, ev_m, ev_s, gap = [], [], [], []
    for rnd in (4, 7, 10, 13, 16):
        rows = [r for r in R if r["method"] == "imp" and r["round"] == rnd and r["src_test/bacc"] >= COLLAPSE]
        v = 100 * np.array([r["flow/ESR_AUC"] for r in rows])
        g = [100 * (r["flow/ESR_AUC"] - dense[r["seed"]]["flow/ESR_AUC"]) for r in rows]
        ev_r.append(rnd); ev_m.append(v.mean()); ev_s.append(v.std(ddof=1)); gap.append(np.mean(g))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.1), sharex=True, gridspec_kw=dict(hspace=0.22))
    # only rounds 4, 7, 10, 13, 16 and 20 were attacked; 59 % showed no consistent effect (4/5 seeds worse, CI incl. 0)
    zones = [(0.5, 6.5, "#E8F5E9", "no consistent\neffect (59 %)"), (6.5, 13.5, "#FFF3E0", "hidden\nfragility"),
             (13.5, 16.5, "#FDECEA", "breakdown"), (16.5, 20.5, "#EEEEEE", "collapse")]
    for ax in (a1, a2):
        for x0, x1, c, _ in zones:
            ax.axvspan(x0, x1, color=c, lw=0, zorder=0)
        ax.grid(alpha=0.3, lw=0.4)
    for x0, x1, _, lab in zones:
        a2.text((x0 + x1) / 2, 93, lab, ha="center", va="top", fontsize=6.5, style="italic")

    a1.axhspan(d_acc.mean() - d_acc.std(ddof=1), d_acc.mean() + d_acc.std(ddof=1), color="0.6", alpha=0.35, lw=0)
    a1.axhline(d_acc.mean(), color="k", ls="--", lw=0.9, label="dense network")
    a1.errorbar(rounds, acc_m, yerr=acc_s, color="#1F4E9A", marker="o", ms=2.6, lw=1, capsize=1.5,
                label="IMP lottery ticket")
    a1.axhline(COLLAPSE, color="#B26A00", ls=":", lw=0.9, label="collapse threshold (0.90)")
    a1.set_ylim(0.45, 1.0)
    a1.set_ylabel("clean balanced accuracy")
    a1.legend(loc="lower left", frameon=True, framealpha=0.9)
    a1.set_title("(a) Accuracy on normal traffic: flat up to 94.5 % pruning", loc="left", pad=3)

    a2.axhspan(d_esr.mean() - d_esr.std(ddof=1), d_esr.mean() + d_esr.std(ddof=1), color="0.6", alpha=0.35, lw=0)
    a2.axhline(d_esr.mean(), color="k", ls="--", lw=0.9)
    a2.errorbar(ev_r, ev_m, yerr=ev_s, color="#C0392B", marker="s", ms=3, lw=1.1, capsize=1.5)
    for x, y, s, g in zip(ev_r, ev_m, ev_s, gap):
        a2.annotate(f"+{g:.1f}", (x, y + s), textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=6.5, color="#C0392B", fontweight="bold" if x >= 7 else "normal")
    a2.text(18.5, 62, "98.8 %:\nall tickets\ncollapsed", ha="center", va="center", fontsize=6.5, color="0.35")
    a2.set_ylim(35, 96)
    a2.set_ylabel("flow ESR-AUC (%), lower = better")
    a2.set_title("(b) Evasion success under the flow attack: rises from 79 %", loc="left", pad=3)
    ticks = [1, 4, 7, 10, 13, 16, 20]
    a2.set_xticks(ticks)
    a2.set_xticklabels([f"{sp[t - 1]:.0f}" if sp[t - 1] < 89 else f"{sp[t - 1]:.1f}" for t in ticks])
    a2.set_xlim(0.5, 20.5)
    a2.set_xlabel("weights removed (%)  (IMP rounds 1-20)")
    fig.savefig(os.path.join(FIG, "ieee_fragility.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, "ieee_fragility.png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc, ec, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.9))
    title, *body = text.split(chr(10))
    lh = 0.052  # body line height in axes units at this figure size
    top = y + h / 2 + (len(body) * lh + 0.07) / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs, fontweight="bold")
    ax.text(x + w / 2, top - 0.075, chr(10).join(body), ha="center", va="top", fontsize=fs - 0.5, linespacing=1.3)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#1F3864"))


def workflow():
    fig = plt.figure(figsize=(7.16, 2.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.012, 1.012); ax.set_ylim(0.04, 1.0); ax.axis("off")
    P, E = ("#E8EEF7", "#1F3864"), ("#EAF5E9", "#2E7D32")
    _box(ax, 0.005, 0.08, 0.15, 0.88, "Data preparation\nCIC-IDS2017,\nCIC-DDoS2019\n76 shared flow\nfeatures; clean,\n"
         "de-duplicate;\n70/15/15 split;\nsigned log +\nstandardization", *E)
    _box(ax, 0.175, 0.08, 0.14, 0.88, "Detector\nMLP 76-512-\n512-256-2\n432,640 prunable\nweights\n5 seeds\n3,000 fixed\n"
         "attack flows", *P)
    ys = (0.68, 0.38, 0.08)
    exps = ["Exp. A: standard tickets\nIMP + rewinding, 20 rounds;\n5 controls (re-init, random,\nERK, SNIP, one-shot)",
            "Exp. B: adversarial training\ndense AT, A-IMP (16 rounds),\nOMP, LAMP, HYDRA;\nimmutable-feature detector",
            "Exp. C: ticket transfer\nsource ticket -> target data\n(1 % or 100 % labels);\n7 starting conditions"]
    for y, t in zip(ys, exps):
        _box(ax, 0.345, y, 0.215, 0.28, t, *P)
    _box(ax, 0.59, 0.08, 0.19, 0.88, "Evaluation\nevery attacked network:\nclean balanced acc.\nflow-consistent attack\n"
         "($\\rho$ = 0.1 ... 2) -> ESR-AUC\nPGD reference (A, B)\ncollapse rule:\nbal. acc. < 0.90", *P)
    _box(ax, 0.81, 0.08, 0.185, 0.88, "Analysis\npaired Wilcoxon\n+ Holm, Friedman\nweights per layer\n"
         "Integrated Gradients\nstorage, latency", "#FBE9E7", "#B23A1E")
    _arrow(ax, (0.155, 0.52), (0.175, 0.52))
    for y in ys:
        _arrow(ax, (0.315, 0.52), (0.345, y + 0.14))
        _arrow(ax, (0.56, y + 0.14), (0.59, 0.52))
    _arrow(ax, (0.78, 0.52), (0.81, 0.52))
    fig.savefig(os.path.join(FIG, "ieee_workflow.pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(FIG, "ieee_workflow.png"), dpi=250, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    fragility()
    workflow()
    print("done")
