"""Aggregate RQ1/RQ2 JSON results into tables and figures.

Statistics: per (method, sparsity level) mean +- std over seeds. Paired comparisons vs a
reference condition use the Wilcoxon signed-rank test over all matched (seed, sparsity) pairs,
Holm-Bonferroni corrected across methods, with the median paired difference as effect size.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (len(p) - i) * p[k])
        adj[k] = min(1.0, running)
    return adj


def paired_tests(df, key_cols, cond_col, ref, metrics):
    rows = []
    for m in metrics:
        ref_df = df[df[cond_col] == ref].set_index(key_cols)[m]
        ps, tmp = [], []
        for c in sorted(df[cond_col].unique()):
            if c == ref:
                continue
            other = df[df[cond_col] == c].set_index(key_cols)[m]
            j = pd.concat([ref_df, other], axis=1, keys=["ref", "x"]).dropna()
            if len(j) < 3:
                continue
            d = (j["x"] - j["ref"]).to_numpy()
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ps.append(p)
            tmp.append(dict(metric=m, cond=c, ref=ref, n_pairs=len(d), median_diff=np.median(d),
                            mean_diff=d.mean(), wins=int((d > 0).sum()), p=p))
        for r, pa in zip(tmp, holm(ps) if ps else []):
            r["p_holm"] = pa
            rows.append(r)
    return pd.DataFrame(rows)


def load(kind):
    recs = []
    for f in sorted(glob.glob(os.path.join(RAW, kind, "*.json"))):
        d = json.load(open(f))
        a = d["args"]
        for r in d["records"]:
            r = {k: v for k, v in r.items() if k != "layer_density"}
            r.update(seed=a["seed"], arch=a["arch"], src=a["src"], tgt=a.get("tgt"))
            if a.get("tag") and "method" in r:  # tagged runs (e.g. _immfix) are separate conditions
                r["method"] += a["tag"]
            recs.append(r)
    return pd.DataFrame(recs)


def rq1(kind="rq1", ref="imp"):
    """Sparsity sweep tables/figures. kind='rq1' (standard training, reference IMP) or
    kind='robust' (adversarial training, reference A-IMP)."""
    df = load(kind)
    if df.empty:
        print(f"no {kind} results")
        return
    df = df[df["round"].isin([0, 4, 7, 10, 13, 16, 20])]
    df["sp"] = df["sparsity"].round(3)
    cols = [c for c in df.columns if c.endswith(("/bacc", "/macro_f1", "ESR_AUC")) or "transfer_ESR" in c
            or c.startswith("flow/ESR@")]
    cols = [c for c in cols if df[c].notna().any()]
    g = df.groupby(["src", "method", "round"])
    agg = g[cols + ["sp"]].agg(["mean", "std", "count"])
    os.makedirs(TAB, exist_ok=True)
    agg.to_csv(os.path.join(TAB, f"{kind}_summary.csv"))
    # compact markdown: bacc, ESR AUC (u,c) mean+-sd
    show = ["src_test/bacc", "flow/ESR_AUC", "unconstrained/ESR_AUC", "constrained/ESR_AUC"]
    show += [c for c in df.columns if c.endswith("_all/bacc")]
    lines = []
    for src, d in df.groupby("src"):
        lines.append(f"\n### {kind} source = {src} (mean ± sd over seeds; n = seeds)\n")
        lines.append("| method | round | sparsity | n | " + " | ".join(show) + " |")
        lines.append("|" + "---|" * (4 + len(show)))
        for (meth, rnd), e in d.groupby(["method", "round"]):
            cells = []
            for c in show:
                v = e[c].dropna()
                cells.append(f"{v.mean():.4f} ± {v.std():.4f}" if len(v) else "–")
            lines.append(f"| {meth} | {rnd} | {e['sp'].mean():.3f} | {e['seed'].nunique()} | " + " | ".join(cells) + " |")
    # paired tests vs IMP at matched (seed, round), rounds where baselines exist
    tests = []
    for src, d in df[df["round"] > 0].groupby("src"):
        t = paired_tests(d, ["seed", "round"], "method", ref,
                         ["src_test/bacc", "flow/ESR_AUC", "unconstrained/ESR_AUC", "constrained/ESR_AUC"])
        t.insert(0, "src", src)
        tests.append(t)
    # dense vs each sparse method, per round (does pruning change robustness?)
    if tests:
        T = pd.concat(tests)
        T.to_csv(os.path.join(TAB, f"{kind}_paired_vs_{ref}.csv"), index=False)
        lines.append(f"\n### Paired Wilcoxon vs {ref} (pairs = seed × sparsity level; Holm-corrected)\n")
        lines.append(T.to_markdown(index=False, floatfmt=".4g"))
    open(os.path.join(TAB, f"{kind}_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    # figures
    os.makedirs(FIG, exist_ok=True)
    for src, d in df.groupby("src"):
        metrics = ["src_test/bacc", "flow/ESR_AUC", "unconstrained/ESR_AUC", "constrained/ESR_AUC"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.8))
        dense_name = "dense" if kind == "rq1" else "dense_at"
        dense = d[d["method"] == dense_name]
        for ax, m in zip(axes, metrics):
            for meth, e in d[~d["method"].isin([dense_name, "immutable_std"])].groupby("method"):
                s = e.groupby("round").agg(x=("sparsity", "mean"), y=(m, "mean"), sd=(m, "std")).dropna()
                ax.errorbar(1 - s["x"], s["y"], yerr=s["sd"], marker="o", ms=3, capsize=2, label=meth)
            if len(dense):
                ax.axhline(dense[m].mean(), color="k", ls="--", lw=1, label=dense_name)
            imm = d[d["method"] == "immutable_std"]
            if len(imm):
                ax.axhline(imm[m].mean(), color="gray", ls=":", lw=1, label="immutable features only")
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("density (fraction of weights kept)")
            ax.set_title(f"{src}: {m}")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{kind}_{src}.png"), dpi=150)
        plt.close(fig)
    print("\n".join(lines))


def rq2():
    df = load("rq2")
    if df.empty:
        print("no rq2 results")
        return
    df["pair"] = df["src"] + "→" + df["tgt"]
    show = ["tgt_test/bacc", "tgt_test/macro_f1", "flow/ESR_AUC"]
    show = [c for c in show if c in df.columns]
    lines = []
    for (pair, frac), d in df.groupby(["pair", "frac"]):
        lines.append(f"\n### RQ2 {pair}, target label fraction = {frac} (mean ± sd over seeds)\n")
        lines.append("| cond | round | sparsity | n | " + " | ".join(show) + " |")
        lines.append("|" + "---|" * (4 + len(show)))
        for (c, rnd), e in d.groupby(["cond", "round"]):
            cells = [f"{e[m].mean():.4f} ± {e[m].std():.4f}" for m in show]
            lines.append(f"| {c} | {rnd} | {e['sparsity'].mean():.3f} | {e['seed'].nunique()} | " + " | ".join(cells) + " |")
    tests = []
    for (pair, frac), d in df[df["round"] > 0].groupby(["pair", "frac"]):
        t = paired_tests(d, ["seed", "round"], "cond", "tkt_rewind", show)
        t.insert(0, "frac", frac)
        t.insert(0, "pair", pair)
        tests.append(t)
    if tests:
        T = pd.concat(tests)
        T.to_csv(os.path.join(TAB, "rq2_paired_vs_tkt_rewind.csv"), index=False)
        lines.append("\n### Paired Wilcoxon vs tkt_rewind (pairs = seed × sparsity; Holm-corrected)\n")
        lines.append(T.to_markdown(index=False, floatfmt=".4g"))
    os.makedirs(TAB, exist_ok=True)
    open(os.path.join(TAB, "rq2_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all")
    a = ap.parse_args()
    if a.what in ("all", "rq1"):
        rq1()
    if a.what in ("all", "rq2"):
        rq2()
    if a.what in ("all", "robust"):
        rq1("robust", ref="aimp")
