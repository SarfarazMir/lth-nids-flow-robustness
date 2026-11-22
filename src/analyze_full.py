"""5-seed analysis of the full (lean) study: the tables, tests and figures the paper uses.

Only conditions run for all five seeds (0-4) enter the main tables; pilot-only conditions
(seed 0: GraSP, SynFlow, extra levels, d1->d2 pair, 10% labels, A-IMP controls) are reported by
analyze.py as n = 1.

Validity rule (stated before the tests were run): a network whose clean balanced accuracy on its
own test set is below 0.90 has collapsed (it no longer detects attacks reliably), so its evasion
success rate is not meaningful (ESR is computed only over flows the model detects). Such networks
are excluded from robustness statistics and counted separately. Every test is also repeated with
all networks included (sensitivity analysis).

Statistics:
  * per level: mean ± sd over seeds; paired difference vs a reference with a t-based 95% CI and the
    number of seeds in which the difference is positive (with n = 5 a per-level Wilcoxon test can
    never reach p < 0.05, so per-level results are descriptive);
  * pooled: Wilcoxon signed-rank over all (seed, level) pairs, Holm-corrected within each family.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon

from analyze import FIG, TAB, holm, load

SEEDS = [0, 1, 2, 3, 4]
VALID = 0.90
LEVEL = {4: "59.0", 7: "79.0", 10: "89.3", 13: "94.5", 16: "97.2", 20: "98.8"}
OUT = []


def say(s=""):
    OUT.append(s)


def ms(v, pct=False):
    v = pd.Series(v).dropna()
    if not len(v):
        return "–"
    f = 100 if pct else 1
    return f"{f * v.mean():.1f} ± {f * v.std():.1f}" if pct else f"{v.mean():.4f} ± {v.std():.4f}"


def ci(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan, np.nan
    h = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def pooled(rows):
    """rows: list of dicts with key 'd' (paired differences); adds p and Holm-adjusted p."""
    ps = []
    for r in rows:
        d = np.asarray(r.pop("d"), float)
        d = d[~np.isnan(d)]
        r["n_pairs"] = len(d)
        r["mean_diff"] = d.mean() if len(d) else np.nan
        r["median_diff"] = np.median(d) if len(d) else np.nan
        r["positive"] = f"{int((d > 0).sum())}/{len(d)}"
        r["p"] = wilcoxon(d).pvalue if len(d) >= 3 and np.any(d != 0) else 1.0
        ps.append(r["p"])
    for r, pa in zip(rows, holm(ps) if ps else []):
        r["p_holm"] = pa
    return pd.DataFrame(rows)


def md(df, fmt=".3g"):
    return df.to_markdown(index=False, floatfmt=fmt)


# ======================================================================== Experiment A (RQ1)
def exp_a():
    df = load("rq1")
    df = df[df["seed"].isin(SEEDS)]
    df["valid"] = df["src_test/bacc"] >= VALID
    methods = ["imp", "imp_reinit", "random_lw", "erk", "snip", "omp_ft"]
    say("# Experiment A — standard training (5 seeds)\n")
    say(f"Validity rule: networks with clean balanced accuracy < {VALID} are 'collapsed'; their ESR is "
        "excluded (counts shown as 'collapsed').\n")
    res = {}
    for src, d in df.groupby("src"):
        dense = d[d["method"] == "dense"].set_index("seed")
        say(f"\n## Source = {src}\n")
        say(f"Dense: bacc {ms(dense['src_test/bacc'])}; flow ESR-AUC {ms(dense['flow/ESR_AUC'], True)}%; "
            f"unconstrained {ms(dense['unconstrained/ESR_AUC'], True)}%; constrained {ms(dense['constrained/ESR_AUC'], True)}%\n")
        rows = []
        for meth in methods:
            for rnd in [4, 7, 10, 13, 16, 20]:
                e = d[(d["method"] == meth) & (d["round"] == rnd)]
                if e["seed"].nunique() < 5:
                    continue
                v = e[e["valid"]]
                rows.append({"method": meth, "pruned %": LEVEL[rnd], "n": len(e),
                             "collapsed": int((~e["valid"]).sum()),
                             "bacc": ms(e["src_test/bacc"]),
                             "flow ESR-AUC %": ms(v["flow/ESR_AUC"], True),
                             "uncon. ESR-AUC %": ms(v["unconstrained/ESR_AUC"], True),
                             "con. ESR-AUC %": ms(v["constrained/ESR_AUC"], True)})
        say(md(pd.DataFrame(rows)))
        # --- IMP vs dense, per level (same seed)
        say(f"\n### {src}: IMP ticket minus dense network of the same seed (flow ESR-AUC, percentage points)\n")
        rows = []
        for rnd in [4, 7, 10, 13, 16, 20]:
            e = d[(d["method"] == "imp") & (d["round"] == rnd) & d["valid"]].set_index("seed")
            diff = (e["flow/ESR_AUC"] - dense["flow/ESR_AUC"]).dropna() * 100
            m, lo, hi = ci(diff)
            b = (e["src_test/bacc"] - dense["src_test/bacc"]).dropna() * 100
            rows.append({"pruned %": LEVEL[rnd], "valid seeds": len(diff), "Δ flow ESR (pp)": m,
                         "95% CI": f"[{lo:.1f}, {hi:.1f}]", "seeds worse than dense": f"{int((diff > 0).sum())}/{len(diff)}",
                         "Δ bacc (pp)": b.mean()})
        say(md(pd.DataFrame(rows), ".1f"))
        # --- pooled tests
        for include_all in (False, True):
            fam = []
            dd = d if include_all else d[d["valid"]]
            imp = dd[dd["method"] == "imp"].set_index(["seed", "round"])
            dn = dd[dd["method"] == "dense"].set_index("seed")
            for metric in ["flow/ESR_AUC", "unconstrained/ESR_AUC", "constrained/ESR_AUC", "src_test/bacc"]:
                x = imp[imp.index.get_level_values("round").isin([7, 10, 13, 16])][metric]
                ref = dn[metric].reindex(x.index.get_level_values("seed")).to_numpy()
                fam.append({"comparison": "imp − dense", "metric": metric, "d": x.to_numpy() - ref})
                for meth in methods[1:]:
                    o = dd[dd["method"] == meth].set_index(["seed", "round"])[metric]
                    j = pd.concat([o, imp[metric]], axis=1, keys=["o", "imp"]).dropna()
                    j = j[j.index.get_level_values("round").isin([7, 10, 13, 16])]
                    fam.append({"comparison": f"{meth} − imp", "metric": metric, "d": (j["o"] - j["imp"]).to_numpy()})
            T = pooled(fam)
            T.insert(0, "src", src)
            res[(src, include_all)] = T
            say(f"\n### {src}: pooled paired Wilcoxon over seeds × levels 79–97.2% "
                f"({'ALL networks (sensitivity)' if include_all else 'valid networks only (primary)'}); Holm over the table\n")
            say(md(T[["comparison", "metric", "n_pairs", "mean_diff", "median_diff", "positive", "p", "p_holm"]], ".4g"))
        # --- zero-shot transfer to other datasets
        other = [c for c in d.columns if c.endswith("_all/bacc") and d[c].notna().any()]
        say(f"\n### {src}: zero-shot balanced accuracy on other datasets (no retraining)\n")
        rows = []
        for meth, rnd in [("dense", 0)] + [("imp", r) for r in (4, 7, 10, 13, 16, 20)]:
            e = d[(d["method"] == meth) & (d["round"] == rnd)]
            rows.append({"network": meth if meth == "dense" else f"imp {LEVEL[rnd]}%",
                         **{c.split("_all")[0]: ms(e[c]) for c in other}})
        say(md(pd.DataFrame(rows)))
    pd.concat(res.values()).to_csv(os.path.join(TAB, "full5_expA_tests.csv"), index=False)
    # figure: flow ESR-AUC and bacc vs sparsity
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for col, (src, d) in enumerate(df.groupby("src")):
        dense = d[d["method"] == "dense"]
        for row, (m, lab) in enumerate([("flow/ESR_AUC", "flow attack ESR-AUC (valid networks)"),
                                        ("src_test/bacc", "clean balanced accuracy")]):
            ax = axes[row, col]
            for meth in methods:
                e = d[d["method"] == meth]
                if m != "src_test/bacc":
                    e = e[e["valid"]]
                s = e.groupby("round").agg(x=("sparsity", "mean"), y=(m, "mean"), sd=(m, "std"), n=("seed", "nunique"))
                s = s[s["n"] >= 3].dropna(subset=["y"])
                ax.errorbar(100 * s["x"], s["y"], yerr=s["sd"], marker="o", ms=3, capsize=2, label=meth)
            ax.axhline(dense[m].mean(), color="k", ls="--", lw=1, label="dense")
            ax.set_xlabel("% of weights removed")
            ax.set_title(f"{src}: {lab}", fontsize=10)
        axes[0, col].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "full5_expA.png"), dpi=150)
    plt.close(fig)


# ======================================================================== Experiment B (robust)
def exp_b():
    df = load("robust")
    df = df[df["seed"].isin(SEEDS)]
    df["valid"] = df["src_test/bacc"] >= VALID
    std = load("rq1")
    std = std[(std["src"] == "ids17") & std["seed"].isin(SEEDS)]
    say("\n\n# Experiment B — adversarial (flow) training, CIC-IDS2017 (5 seeds)\n")
    rows = []
    for meth, rnds in [("immutable_std", [0]), ("immutable_std_immfix", [0]), ("dense_at", [0]), ("aimp", [4, 7, 10, 13, 16]),
                       ("omp_at", [10, 13, 16]), ("lamp_at", [10, 13, 16]), ("hydra", [10, 13, 16])]:
        for rnd in rnds:
            e = df[(df["method"] == meth) & (df["round"] == rnd)]
            v = e[e["valid"]]
            rows.append({"method": meth, "pruned %": LEVEL.get(rnd, "0"), "n": len(e),
                         "collapsed": int((~e["valid"]).sum()), "bacc": ms(e["src_test/bacc"]),
                         "flow ESR-AUC %": ms(v["flow/ESR_AUC"], True),
                         "uncon. ESR-AUC %": ms(v["unconstrained/ESR_AUC"], True),
                         "con. ESR-AUC %": ms(v["constrained/ESR_AUC"], True)})
    say(md(pd.DataFrame(rows)))
    # tests: (1) AT network vs standard-training IMP at the same level & seed  (does AT fix it?)
    #        (2) pruned AT vs dense AT (does pruning keep the fix?)   (3) immutable vs dense std
    fam = []
    v = df[df["valid"]]
    dat = v[v["method"] == "dense_at"].set_index("seed")
    stdimp = std[(std["method"] == "imp") & (std["src_test/bacc"] >= VALID)].set_index(["seed", "round"])
    for meth in ["aimp", "omp_at", "lamp_at", "hydra"]:
        e = v[(v["method"] == meth) & v["round"].isin([10, 13, 16])].set_index(["seed", "round"])
        for metric in ["flow/ESR_AUC", "src_test/bacc"]:
            ref = dat[metric].reindex(e.index.get_level_values("seed")).to_numpy()
            fam.append({"comparison": f"{meth} − dense_at", "metric": metric, "d": e[metric].to_numpy() - ref})
        j = pd.concat([e["flow/ESR_AUC"], stdimp["flow/ESR_AUC"]], axis=1, keys=["at", "std"]).dropna()
        fam.append({"comparison": f"{meth} − standard IMP (same level)", "metric": "flow/ESR_AUC",
                    "d": (j["at"] - j["std"]).to_numpy()})
    dstd = std[std["method"] == "dense"].set_index("seed")
    for name in ["immutable_std", "immutable_std_immfix"]:
        imm = df[df["method"] == name].set_index("seed")
        for metric in ["flow/ESR_AUC", "src_test/bacc"]:
            fam.append({"comparison": f"{name} − dense (standard)", "metric": metric,
                        "d": (imm[metric] - dstd[metric]).dropna().to_numpy()})
            fam.append({"comparison": f"{name} − dense_at", "metric": metric,
                        "d": (imm[metric] - dat[metric]).dropna().to_numpy()})
    T = pooled(fam)
    T.to_csv(os.path.join(TAB, "full5_expB_tests.csv"), index=False)
    say("\n### Pooled paired Wilcoxon (pairs = seed × level 89.3/94.5/97.2%, or seed for dense comparisons); Holm over the table\n")
    say("Note: comparisons with only 5 pairs cannot reach p < 0.0625 by construction; read their 'positive' column.\n")
    say(md(T[["comparison", "metric", "n_pairs", "mean_diff", "median_diff", "positive", "p", "p_holm"]], ".4g"))


# ======================================================================== Experiment C (RQ2)
def exp_c():
    df = load("rq2")
    df = df[df["seed"].isin(SEEDS) & df["frac"].isin([0.01, 1.0]) & df["round"].isin([0, 10, 13, 16])
            & (df["tgt"] != "ddos19_d2")]
    df["pair"] = df["src"] + " → " + df["tgt"]
    say("\n\n# Experiment C — reusing tickets on another dataset (5 seeds)\n")
    conds = ["tkt_rewind", "tkt_reinit", "tkt_ft", "rand_rewind", "snip_tgt"]
    fams = []
    for (pair, frac), d in df.groupby(["pair", "frac"]):
        say(f"\n## {pair}, target labels = {int(frac * 100)}%\n")
        sc = d[d["cond"] == "dense_scratch"].set_index("seed")
        ft = d[d["cond"] == "dense_ft"]
        say(f"dense_scratch bacc {ms(sc['tgt_test/bacc'])}; dense_ft bacc {ms(ft['tgt_test/bacc'])}\n")
        rows = []
        for c in conds:
            for rnd in [10, 13, 16]:
                e = d[(d["cond"] == c) & (d["round"] == rnd)].set_index("seed")
                diff = (e["tgt_test/bacc"] - sc["tgt_test/bacc"]).dropna() * 100
                m, lo, hi = ci(diff)
                rows.append({"condition": c, "pruned %": LEVEL[rnd], "n": len(e), "bacc": ms(e["tgt_test/bacc"]),
                             "Δ vs scratch (pp)": f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                             "seeds better than scratch": f"{int((diff > 0).sum())}/{len(diff)}",
                             "flow ESR-AUC %": ms(e.loc[e["tgt_test/bacc"] >= VALID, "flow/ESR_AUC"], True)})
        say(md(pd.DataFrame(rows)))
        for c in conds + ["dense_ft"]:
            e = d[d["cond"] == c].set_index(["seed", "round"])
            ref = sc["tgt_test/bacc"].reindex(e.index.get_level_values("seed")).to_numpy()
            fams.append({"pair": pair, "labels %": int(frac * 100), "comparison": f"{c} − dense_scratch",
                         "metric": "tgt_test/bacc", "d": e["tgt_test/bacc"].to_numpy() - ref})
    T = pooled(fams)
    T.to_csv(os.path.join(TAB, "full5_expC_tests.csv"), index=False)
    say("\n### Pooled paired Wilcoxon vs dense_scratch, balanced accuracy (pairs = seed × level); Holm over the table\n")
    say(md(T[["pair", "labels %", "comparison", "n_pairs", "mean_diff", "median_diff", "positive", "p", "p_holm"]], ".4g"))
    # figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=False)
    for ax, ((pair, frac), d) in zip(axes, df.groupby(["pair", "frac"])):
        sc = d[d["cond"] == "dense_scratch"]["tgt_test/bacc"]
        for c in conds:
            s = d[d["cond"] == c].groupby("round").agg(x=("sparsity", "mean"), y=("tgt_test/bacc", "mean"),
                                                         sd=("tgt_test/bacc", "std"))
            ax.errorbar(100 * s["x"], s["y"], yerr=s["sd"], marker="o", ms=3, capsize=2, label=c)
        ax.axhline(sc.mean(), color="k", ls="--", lw=1, label="dense_scratch")
        ax.set_title(f"{pair}, {int(frac * 100)}% labels", fontsize=9)
        ax.set_xlabel("% of weights removed")
    axes[0].set_ylabel("target balanced accuracy")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "full5_expC.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    exp_a()
    exp_b()
    exp_c()
    open(os.path.join(TAB, "full5_summary.md"), "w", encoding="utf-8").write("\n".join(OUT))
    print("\n".join(OUT))
