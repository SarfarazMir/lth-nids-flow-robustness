"""Workflow diagrams and flowcharts for the methodology section (drawn with matplotlib).

Writes results/figures/diag_*.png. Every box states what the code does (exp_rq1.py, exp_robust.py,
exp_rq2.py, lthlib/*); numbers are the configuration actually used.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})

C = dict(proc="#E8EEF7", edge="#1F3864", data="#EAF5E9", dedge="#2E7D32", dec="#FFF4E0", decedge="#B26A00",
         out="#FBE9E7", oedge="#B23A1E", grey="#F2F2F2", gedge="#555555", term="#DDE7F6")


_BOXES = []  # (patch, text) pairs drawn since the last save, checked for text overflow


def box(ax, x, y, w, h, text, kind="proc", fs=9.5, bold=False):
    fc, ec = {"proc": (C["proc"], C["edge"]), "data": (C["data"], C["dedge"]),
              "out": (C["out"], C["oedge"]), "grey": (C["grey"], C["gedge"]),
              "term": (C["term"], C["edge"])}[kind]
    r = 0.25 if kind == "term" else 0.06
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                           fc=fc, ec=ec, lw=1.4)
    ax.add_patch(patch)
    t = ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True,
                fontweight="bold" if bold else "normal", color="#111")
    _BOXES.append((patch, t))


def diamond(ax, x, y, w, h, text, fs=9.5):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                         fc=C["dec"], ec=C["decedge"], lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, label=None, lpos=0.5, color="#1F3864", style="-|>", rad=0.0, loff=(0.08, 0)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        lx = p[0] + (q[0] - p[0]) * lpos + loff[0]
        ly = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(lx, ly, label, fontsize=8.5, color=color, ha="left", va="center", style="italic")


def poly(ax, pts, color="#1F3864", label=None, lxy=None):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.3)
    arrow(ax, pts[-2], pts[-1], color=color)
    if label and lxy:
        ax.text(*lxy, label, fontsize=8.5, color=color, style="italic", ha="center")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    r = fig.canvas.get_renderer()
    for patch, t in _BOXES:
        b, tb = patch.get_window_extent(r), t.get_window_extent(r)
        if tb.x0 < b.x0 or tb.x1 > b.x1 or tb.y0 < b.y0 or tb.y1 > b.y1:
            print(f"WARNING {name}: text overflows its box: {t.get_text()[:50]!r}")
    _BOXES.clear()
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 1. overall workflow
def overall():
    fig, ax = canvas(11, 8.6)
    box(ax, 5.5, 8.1, 8.6, 0.6, "Raw flow files: CIC-IDS2017 (8 CSV files) and CIC-DDoS2019 (day 1, day 2)", "data", bold=True)
    box(ax, 5.5, 7.1, 8.6, 0.9, "PHASE 1 - Data preparation (Algorithm 1)\nremove identity columns -> keep 76 shared flow features -> clean\n-> "
        "70/15/15 stratified split  ->  signed-log + standard scaling", fs=9.3)
    arrow(ax, (5.5, 7.8), (5.5, 7.53))
    box(ax, 5.5, 6.1, 8.6, 0.75, "PHASE 2 - Detector and attack set-up\nMLP 76-512-512-256-2 (432,640 prunable weights)   |   "
        "3,000 fixed attack flows   |   5 seeds (0-4)", fs=9.3)
    arrow(ax, (5.5, 6.77), (5.5, 6.48))
    xs = [1.9, 5.5, 9.1]
    titles = ["EXPERIMENT A\nStandard lottery tickets\n(Algorithms 2, 3)",
              "EXPERIMENT B\nTickets + adversarial training\n(Algorithms 4, 5)",
              "EXPERIMENT C\nTicket transfer to a new dataset\n(Algorithm 6)"]
    bodies = ["Round 0: train dense, save\nrewind copy (5 % of run)\nRounds 1-20: prune 20 %,\nrewind, retrain\n"
              "+ 5 controls (imp_reinit,\nrandom_lw, ERK, SNIP, omp_ft)",
              "Baselines: drop changeable\nfeatures; dense_at\nA-IMP rounds 1-16\n(attack inside every step)\n"
              "+ omp_at, lamp_at, HYDRA",
              "Source run from Exp. A\n-> target data (1 % or 100 %)\n7 conditions: scratch,\nfine-tune, ticket rewind,\n"
              "ticket reinit, ticket FT,\nrandom mask, SNIP on target"]
    for x, t, b in zip(xs, titles, bodies):
        box(ax, x, 4.95, 3.3, 0.85, t, "term", fs=9.3, bold=True)
        box(ax, x, 3.55, 3.3, 1.75, b, fs=8.8)
        arrow(ax, (x, 4.52), (x, 4.43))
        poly(ax, [(5.5, 5.72), (5.5, 5.55), (x, 5.55), (x, 5.38)])
    box(ax, 5.5, 1.95, 10.4, 0.85, "Every attacked network: clean accuracy + flow-consistent evasion attack (Algorithm 7)\n"
        "PGD constrained / unconstrained in Experiments A and B; other IMP rounds: clean accuracy only\n"
        "collapse rule: balanced accuracy < 0.90 = excluded from robustness comparisons", fs=9.0)
    for x in xs:
        arrow(ax, (x, 2.67), (x, 2.33))
    box(ax, 5.5, 0.75, 10.4, 0.85, "PHASE 3 - Analysis\nstatistics (paired Wilcoxon + Holm, rank-biserial r, Friedman)   |   "
        "mechanism (weights per layer)\nexplainable AI (Integrated Gradients)   |   efficiency (storage, latency)", "out", fs=9.0)
    arrow(ax, (5.5, 1.57), (5.5, 1.18))
    save(fig, "diag_overall_workflow.png")


# ------------------------------------------------------------------ 2. preprocessing pipeline
def preprocessing():
    fig, ax = canvas(12, 3.4)
    steps = [("Raw flow\nrecords", "data"), ("Drop identity\ncolumns\n(Flow ID, IPs,\nports, time)", "proc"),
             ("Keep 76\nshared flow\nfeatures", "proc"), ("Bernoulli\nsample\n(nominal\n30,000)", "proc"),
             ("Drop inf/NaN;\nde-duplicate\nfeatures +\noriginal label", "proc"), ("Binary label:\n0 = benign\n1 = attack", "proc"),
             ("Split 70/15/15\nstratified by\noriginal label", "proc"), ("Signed log\n+ standard\nscaling (train)", "proc"),
             ("Model\ninputs", "data")]
    n = len(steps)
    w = 1.12
    xs = [0.65 + i * (10.7 / (n - 1)) for i in range(n)]
    for i, ((t, k), x) in enumerate(zip(steps, xs)):
        box(ax, x, 2.05, w, 1.35, t, k, fs=7.9)
        if i:
            arrow(ax, (xs[i - 1] + w / 2, 2.05), (x - w / 2, 2.05))
    ax.text(6.0, 0.75, "CIC-IDS2017: 198,307 training flows   |   CIC-DDoS2019 (day 1): 111,440 training flows   |   "
            "same split and same 3,000 attack test flows for all seeds", ha="center", fontsize=9, style="italic")
    save(fig, "diag_preprocessing.png")


# ------------------------------------------------------------------ 3. MLP architecture
def architecture():
    fig, ax = canvas(11, 3.6)
    layers = [("Input\n76 flow\nfeatures", 76, "data"), ("Linear 76x512\nBatchNorm\nReLU", 512, "proc"),
              ("Linear 512x512\nBatchNorm\nReLU", 512, "proc"), ("Linear 512x256\nBatchNorm\nReLU", 256, "proc"),
              ("Linear 256x2\n(output)", 2, "proc"), ("Softmax\nbenign /\nattack", 2, "out")]
    xs = [0.8, 2.75, 4.7, 6.65, 8.6, 10.2]
    for i, ((t, n, k), x) in enumerate(zip(layers, xs)):
        box(ax, x, 2.2, 1.45, 1.5, t, k, fs=8.8)
        if i:
            arrow(ax, (xs[i - 1] + 0.73, 2.2), (x - 0.73, 2.2))
    w = ["L1: 38,912", "L2: 262,144", "L3: 131,072", "L4: 512"]
    for x, t in zip([2.75, 4.7, 6.65, 8.6], w):
        ax.text(x, 1.15, t + " weights", ha="center", fontsize=8.6, color=C["oedge"])
    ax.text(5.5, 0.45, "Prunable: the 4 Linear weight matrices (432,640 weights). Biases and BatchNorm are never pruned.",
            ha="center", fontsize=9, style="italic")
    save(fig, "diag_architecture.png")


# ------------------------------------------------------------------ 4. IMP flowchart
def imp():
    fig, ax = canvas(10, 12.2)
    x0 = 3.6
    box(ax, x0, 11.7, 3.2, 0.55, "START (one seed, one dataset)", "term", bold=True)
    box(ax, x0, 10.85, 5.6, 0.75, "Create MLP with random weights (seed); save copy theta_0\nMask = all 1s")
    arrow(ax, (x0, 11.42), (x0, 11.23))
    box(ax, x0, 9.8, 5.6, 0.95, "ROUND 0: train 10 epochs (1,940 / 1,090 steps)\nat step 97 / 54 (5 % of training) save the\nrewind copy theta_k (weights, biases, BatchNorm)")
    arrow(ax, (x0, 10.47), (x0, 10.28))
    box(ax, x0, 8.8, 5.6, 0.6, "Full test of the dense network  ->  REFERENCE", "out")
    arrow(ax, (x0, 9.32), (x0, 9.1))
    box(ax, x0, 7.95, 5.6, 0.6, "round = 1 (first pruning round)")
    arrow(ax, (x0, 8.5), (x0, 8.25))
    box(ax, x0, 7.05, 5.6, 0.7, "Step 1: target sparsity = 1 - 0.8^round\n-> number of weights that survive")
    arrow(ax, (x0, 7.65), (x0, 7.4))
    box(ax, x0, 6.0, 5.6, 0.95, "Step 2: rank the still-alive weights of ALL layers\ntogether by |w|; keep the largest; mark the rest 0\n"
        "(new mask = old mask AND keep; a 0 stays 0)")
    arrow(ax, (x0, 6.7), (x0, 6.48))
    box(ax, x0, 5.0, 5.6, 0.65, "Step 3: rewind - load theta_k (network is full again)")
    arrow(ax, (x0, 5.52), (x0, 5.33))
    box(ax, x0, 4.15, 5.6, 0.6, "Step 4: weights = weights x mask")
    arrow(ax, (x0, 4.67), (x0, 4.45))
    box(ax, x0, 3.2, 5.6, 0.85, "Step 5: train 10 epochs with a fresh optimizer and\nschedule; re-apply the mask after EVERY step")
    arrow(ax, (x0, 3.85), (x0, 3.63))
    diamond(ax, x0, 2.05, 3.6, 1.0, "round in\n{4,7,10,13,16,20}?")
    arrow(ax, (x0, 2.77), (x0, 2.55))
    box(ax, 8.2, 2.05, 2.6, 0.75, "Full test\n(clean + 3 attacks)", "out")
    arrow(ax, (x0 + 1.8, 2.05), (8.2 - 1.3, 2.05), "yes", loff=(-0.3, 0.17))
    box(ax, x0, 0.85, 3.6, 0.55, "Clean test only", "grey", fs=9)
    arrow(ax, (x0, 1.55), (x0, 1.13), "no", loff=(0.08, 0.02))
    diamond(ax, 8.2, 0.75, 2.4, 0.9, "round < 20?")
    arrow(ax, (x0 + 1.8, 0.85), (8.2 - 1.2, 0.8))
    arrow(ax, (8.2, 1.67), (8.2, 1.2))
    poly(ax, [(9.4, 0.75), (9.75, 0.75), (9.75, 7.05), (x0 + 2.8, 7.05)], label="yes: round = round + 1",
         lxy=(8.6, 7.4))
    box(ax, 8.2, -0.6, 3.2, 0.65, "END: save all masks + weights;\nbuild the 5 controls (Alg. 3)", "term", fs=8.8)
    arrow(ax, (8.2, 0.3), (8.2, -0.27), "no", loff=(0.08, 0))
    ax.set_ylim(-1.05, 12.2)
    save(fig, "diag_imp_flowchart.png")


# ------------------------------------------------------------------ 5. controls
def controls():
    fig, ax = canvas(12.5, 5.0)
    cols = [("imp_reinit", "ticket's own mask", "NEW random weights\n(seed + 777)", "10 epochs, lr 1e-3",
             "Do the rewound\nstarting values matter?"),
            ("random_lw", "random positions,\nsame count per layer\nas the ticket", "rewind copy theta_k", "10 epochs, lr 1e-3",
             "Do the exact\npositions matter?"),
            ("ERK", "random positions,\nERK layer amounts\n(balanced layers)", "theta_0 (untrained)", "10 epochs, lr 1e-3",
             "Does a balanced layer\nlayout behave differently?"),
            ("SNIP", "top |gradient x weight|\non 10 batches,\nbefore training", "theta_0 (untrained)", "10 epochs, lr 1e-3",
             "Does pruning before\ntraining differ?"),
            ("omp_ft", "smallest |w| of the\ntrained dense net,\nremoved in one shot", "trained dense network", "fine-tune 10 epochs,\nlr 1e-4",
             "Gradual vs one-shot\npruning?")]
    rows = [4.35, 3.35, 2.45, 1.7, 0.7]
    labels = ["Control", "Mask", "Starting\nweights", "Training", "Question it\nanswers"]
    hs = [0.5, 0.85, 0.55, 0.55, 0.75]
    for r, lab in zip(rows, labels):
        ax.text(0.05, r, lab, fontsize=9, fontweight="bold", va="center")
    for i, c in enumerate(cols):
        x = 2.55 + i * 2.15
        box(ax, x, rows[0], 1.95, hs[0], c[0], "term", bold=True, fs=9.5)
        box(ax, x, rows[1], 1.95, hs[1], c[1], fs=8.3)
        box(ax, x, rows[2], 1.95, hs[2], c[2], "data", fs=8.3)
        box(ax, x, rows[3], 1.95, hs[3], c[3], "grey", fs=8.3)
        box(ax, x, rows[4], 1.95, hs[4], c[4], "out", fs=8.3)
        for j in range(4):
            arrow(ax, (x, rows[j] - hs[j] / 2 - 0.03), (x, rows[j + 1] + hs[j + 1] / 2 + 0.03))
    save(fig, "diag_controls.png")


# ------------------------------------------------------------------ 6. flow-consistent attack
def attack():
    fig, ax = canvas(10, 10.6)
    x0 = 3.9
    box(ax, x0, 10.15, 5.8, 0.6, "Input: attack flow x (raw features) caught by the detector,\nbudget rho", "data")
    box(ax, x0, 9.15, 5.8, 0.85, "Limits: extra forward packets k <= max(1, ceil(rho x n_fwd))\npayload per packet s in [0, 1460] bytes\n"
        "extra delay dd <= rho x (duration + 0.1 s)", fs=9)
    arrow(ax, (x0, 9.85), (x0, 9.58))
    box(ax, x0, 8.05, 5.8, 0.85, "(a) Candidate list: 7 fixed corner cases\n+ 128 random (k, s, dd) combinations")
    arrow(ax, (x0, 8.72), (x0, 8.48))
    box(ax, x0, 6.85, 5.8, 1.05, "For every candidate: recompute ALL dependent features\nas CICFlowMeter would (totals, means, sizes, "
        "std, rates,\nratios, header length, IAT mean/total); keep the\nrest (flags, backward traffic, IAT std/max/min)", fs=9)
    arrow(ax, (x0, 7.62), (x0, 7.38))
    diamond(ax, x0, 5.55, 4.4, 1.05, "any candidate\nclassified BENIGN?")
    arrow(ax, (x0, 6.32), (x0, 6.08))
    box(ax, 8.6, 5.55, 2.2, 0.75, "EVADED\n(attack succeeded)", "out")
    arrow(ax, (x0 + 2.2, 5.55), (8.6 - 1.1, 5.55), "yes", loff=(-0.4, 0.17))
    box(ax, x0, 4.2, 5.8, 0.95, "(b) Refine from 2 starting points: 30 Adam steps (size 0.2)\non a smooth (sigmoid) version of (k, s, dd) towards\n"
        "'benign'; then try k rounded down and up")
    arrow(ax, (x0, 5.02), (x0, 4.68), "no", loff=(0.08, 0))
    diamond(ax, x0, 2.85, 4.4, 1.05, "classified\nBENIGN?")
    arrow(ax, (x0, 3.72), (x0, 3.38))
    poly(ax, [(x0 + 2.2, 2.85), (8.6, 2.85), (8.6, 5.17)], color=C["oedge"], label="yes", lxy=(8.85, 3.9))
    box(ax, x0, 1.55, 3.6, 0.6, "DETECTED (attack failed)", "grey")
    arrow(ax, (x0, 2.32), (x0, 1.85), "no", loff=(0.08, 0))
    box(ax, x0, 0.45, 7.6, 0.6, "Repeat for rho = 0.1, 0.25, 0.5, 1.0, 2.0  ->  ESR(rho) = evaded / caught  ->  ESR-AUC", "term", fs=9)
    save(fig, "diag_flow_attack.png")


# ------------------------------------------------------------------ 7. adversarial training step
def at_step():
    fig, ax = canvas(11, 6.2)
    box(ax, 1.6, 5.4, 2.6, 0.75, "Batch of 4,096\ntraining flows", "data")
    box(ax, 5.5, 5.4, 3.6, 0.75, "Switch network to test mode\n(BatchNorm frozen)")
    arrow(ax, (2.9, 5.4), (3.7, 5.4))
    box(ax, 5.5, 4.0, 5.6, 1.25, "For every ATTACK flow in the batch, attack the CURRENT\nnetwork (budget rho = 1.0):\n"
        "32 random (k, s, dd) -> keep the most benign-looking\n-> 5 Adam steps (step size 0.2) -> round k", fs=9)
    arrow(ax, (5.5, 5.02), (5.5, 4.63))
    box(ax, 5.5, 2.55, 5.6, 0.9, "Replace each attack flow by its tampered version\n(label stays 'attack'; benign flows unchanged)")
    arrow(ax, (5.5, 3.37), (5.5, 3.0))
    box(ax, 5.5, 1.3, 5.6, 0.75, "Switch to training mode -> loss -> AdamW step\n-> OneCycle step -> weights x mask")
    arrow(ax, (5.5, 2.1), (5.5, 1.68))
    box(ax, 9.7, 1.3, 2.2, 0.75, "Next batch\n(490 steps total)", "term")
    arrow(ax, (8.3, 1.3), (8.6, 1.3))
    poly(ax, [(9.7, 1.68), (9.7, 5.4), (7.3, 5.4)], color=C["gedge"])
    save(fig, "diag_adversarial_training.png")


# ------------------------------------------------------------------ 8. transfer protocol
def transfer():
    fig, ax = canvas(12, 6.4)
    box(ax, 2.0, 5.85, 3.4, 0.65, "SOURCE run (Experiment A,\nsame seed)", "term", bold=True)
    arts = ["masks of rounds 10, 13, 16 (89.3/94.5/97.2 %)", "rewind copy theta_k", "trained dense network",
            "trained tickets"]
    for i, a in enumerate(arts):
        box(ax, 2.0, 5.0 - i * 0.62, 3.7, 0.48, a, "data", fs=8.2)
    arrow(ax, (2.0, 5.52), (2.0, 5.25))
    box(ax, 10.0, 5.85, 3.4, 0.65, "TARGET dataset\n(the other one)", "term", bold=True)
    box(ax, 10.0, 4.85, 3.4, 0.7, "take 1 % or 100 % of its\ntraining split (stratified)", "data", fs=8.8)
    box(ax, 10.0, 3.85, 3.4, 0.7, "refit scaling on that data;\n>= 300 training steps", fs=8.8)
    arrow(ax, (10.0, 5.52), (10.0, 5.2))
    arrow(ax, (10.0, 4.5), (10.0, 4.2))
    box(ax, 6.0, 2.75, 3.4, 0.62, "train on target, test on target\n(clean + flow attack)", "grey", fs=8.8)
    arrow(ax, (3.85, 3.3), (4.3, 2.95))
    arrow(ax, (8.3, 3.5), (7.7, 2.95))
    conds = ["dense_scratch\nrandom init, no mask\n(REFERENCE)", "dense_ft\nsource dense,\nlr / 10",
             "tkt_rewind\nsource mask +\ntheta_k", "tkt_reinit\nsource mask +\nnew random init",
             "tkt_ft\nsource trained\nticket, lr / 10", "rand_rewind\nrandom mask\n(same layer\ncounts) + theta_k",
             "snip_tgt\nSNIP mask from\nTARGET data"]
    for i, c in enumerate(conds):
        x = 0.95 + i * 1.68
        box(ax, x, 0.85, 1.55, 1.25, c, "out" if i == 0 else "proc", fs=7.8)
        arrow(ax, (6.0, 2.44), (x, 1.5), color=C["gedge"])
    save(fig, "diag_transfer.png")


if __name__ == "__main__":
    overall()
    preprocessing()
    architecture()
    imp()
    controls()
    attack()
    at_step()
    transfer()
