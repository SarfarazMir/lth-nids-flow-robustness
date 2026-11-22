"""Graphical abstract of the study (5:2 landscape, 300 dpi; PNG, TIFF and PDF).

All numbers come from the saved results (results/tables/full5_summary.md, results/raw/rq1/*):
dense ESR-AUC 44.6 %, ticket 50.1 / 50.7 / 53.5 / 73.8 % at 79 / 89.3 / 94.5 / 97.2 % sparsity,
ticket BA 0.994 / 0.994 / 0.993 / 0.962 (dense 0.992); A-IMP BA 0.984-0.986 with 0.8-1.3 % ESR-AUC up to 94.5 %;
ticket layer counts at 97.2 %: 12,079 / 10 / 29 / 60 weights.
Usage: python src/make_graphical_abstract.py <output dir> [<output dir> ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "Arial", "font.size": 11, "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
NAVY, BLUE, RED, GREEN, GREY, LIGHT = "#1f3b6e", "#3a6fb5", "#c0392b", "#2e8b57", "#6b6b6b", "#eef2f8"

W, H = 13.28, 5.312   # inches -> 3984 x 1594 px at 300 dpi (ratio 2.5)
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def panel(x, w, title, color):
    ax.add_patch(FancyBboxPatch((x, 0.78), w, 3.95, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=LIGHT, ec=color, lw=1.6))
    ax.add_patch(FancyBboxPatch((x, 4.33), w, 0.40, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=color, lw=1.6))
    ax.text(x + w / 2, 4.53, title, ha="center", va="center", color="white", fontsize=12.5, weight="bold")


def arrow(x0, x1, y=2.75):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=NAVY))


ax.text(W / 2, 5.05, "Lottery-ticket pruning of deep NIDS: accurate yet easier to evade",
        ha="center", va="center", fontsize=16, weight="bold", color=NAVY)

# ---------------- panel 1: compress
x1, w1 = 0.15, 3.0
panel(x1, w1, "1  Compress (IMP, global ranking)", NAVY)


def layers(x0, y0, counts, widths, label):
    xs = [x0 + i * 0.55 for i in range(5)]
    for i, wd in enumerate(widths):
        h = 0.25 + 0.55 * (wd / 512) ** 0.5
        ax.add_patch(Rectangle((xs[i] - 0.06, y0 - h / 2), 0.12, h, fc="white", ec="black", lw=0.8))
    for i, c in enumerate(counts):
        th = 0.03 + 0.42 * (min(c, 262144) ** 0.5) / 512
        ax.add_patch(Rectangle((xs[i] + 0.07, y0 - th / 2), 0.41, th, fc=BLUE, ec=BLUE, alpha=0.85))
        ax.text(xs[i] + 0.275, y0 + max(th / 2, 0.12) + 0.08, f"{c:,}", ha="center", fontsize=8.2)
    ax.text(x0 + 1.1, y0 - 0.62, label, ha="center", fontsize=10.5, weight="bold")


layers(x1 + 0.4, 3.45, [38912, 262144, 131072, 512], [76, 512, 512, 256, 2], "Dense MLP (432,640 weights)")
layers(x1 + 0.4, 1.75, [12079, 10, 29, 60], [76, 512, 512, 256, 2], "Ticket, 97.2 % pruned")
ax.text(x1 + w1 / 2, 0.98, "hidden layers keep ~10 and ~29 weights", ha="center", fontsize=9.5, color=RED, style="italic")

arrow(x1 + w1 + 0.05, x1 + w1 + 0.33)

# ---------------- panel 2: attack
x2, w2 = 3.55, 2.75
panel(x2, w2, "2  Flow-consistent attack", RED)
items = [("+ forward packets", r"$k \leq \max(1, \lceil \rho\, n_f \rceil)$"),
         ("+ payload bytes", "0–1460 B per packet"),
         ("+ delay", r"$\Delta d \leq \rho\, (d + 10^5\ \mu s)$")]
for i, (a, b) in enumerate(items):
    y = 3.78 - i * 0.70
    ax.add_patch(FancyBboxPatch((x2 + 0.18, y - 0.29), w2 - 0.36, 0.58, boxstyle="round,pad=0.02",
                                fc="white", ec=RED, lw=1.1))
    ax.text(x2 + w2 / 2, y + 0.11, a, va="center", ha="center", fontsize=11, weight="bold", color=RED)
    ax.text(x2 + w2 / 2, y - 0.15, b, va="center", ha="center", fontsize=9.5, color=GREY)
ax.text(x2 + w2 / 2, 1.5, "dependent CICFlowMeter features\nrecomputed (rates, means, sizes)",
        ha="center", va="center", fontsize=10)
ax.text(x2 + w2 / 2, 1.0, "budgets ρ = 0.1 … 2", ha="center", fontsize=9.5, color=GREY)

arrow(x2 + w2 + 0.05, x2 + w2 + 0.33)

# ---------------- panel 3: finding (bar chart)
x3, w3 = 6.7, 3.55
panel(x3, w3, "3  Hidden fragility (CIC-IDS2017)", "#8e2b20")
cax = fig.add_axes([(x3 + 0.55) / W, 1.55 / H, (w3 - 0.8) / W, 2.5 / H])
labels = ["Dense", "79 %", "89.3 %", "94.5 %", "97.2 %"]
esr = [44.6, 50.1, 50.7, 53.5, 73.8]
ba = ["0.992", "0.994", "0.994", "0.993", "0.962"]
cols = [GREY, BLUE, BLUE, BLUE, RED]
bars = cax.bar(labels, esr, color=cols, width=0.62)
for b, v, a in zip(bars, esr, ba):
    cax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5, weight="bold")
    cax.text(b.get_x() + b.get_width() / 2, 4, f"BA\n{a}", ha="center", fontsize=7.6, color="white", weight="bold")
cax.set_ylim(0, 88); cax.set_ylabel("Evasion ESR-AUC (%)", fontsize=10)
cax.tick_params(labelsize=9); cax.spines[["top", "right"]].set_visible(False); cax.set_facecolor(LIGHT)
ax.text(x3 + w3 / 2, 1.0, "79–94.5 %: near-equal BA, higher evasion in every seed\n"
        "97.2 %: +29.2 pp evasion (BA 0.962)", ha="center", va="center", fontsize=8.9,
        color="#8e2b20", weight="bold")

arrow(x3 + w3 + 0.05, x3 + w3 + 0.33)

# ---------------- panel 4: remedy
x4, w4 = 10.65, 2.48
panel(x4, w4, "4  Remedy", GREEN)
ax.text(x4 + w4 / 2, 3.85, "Adversarial IMP\n(trained with the flow attack)", ha="center", va="center",
        fontsize=10.5, weight="bold", color=GREEN)
ax.text(x4 + w4 / 2, 2.95, "up to 94.5 % sparsity:", ha="center", fontsize=9.8)
ax.text(x4 + w4 / 2, 2.45, "BA 0.984–0.986", ha="center", fontsize=12, weight="bold")
ax.text(x4 + w4 / 2, 1.98, "evasion 0.8–1.3 %", ha="center", fontsize=12, weight="bold", color=GREEN)
ax.text(x4 + w4 / 2, 1.42, "(dense adversarial training:\nBA 0.985, evasion 1.2 %)", ha="center", va="center",
        fontsize=8.8, color=GREY)

ax.text(W / 2, 0.38, "Matched controls implicate the thinned hidden layers.  Standard ℓ∞ (PGD) attacks miss the "
        "breakdown at 97.2 %  →  test pruned detectors with feature-consistent attacks before deployment.",
        ha="center", va="center", fontsize=10.2, color=NAVY)

for out in sys.argv[1:]:
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "LTH_NIDS_Graphical_Abstract")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base + ".pdf")
print("saved", round(300 * W), "x", round(300 * H), "px")
