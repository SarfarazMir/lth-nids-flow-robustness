"""Relabelled copies of three workflow diagrams of src/make_diagrams.py.

The drawings are identical to the diag_* versions; only labels that refer to the original algorithm numbers or to
internal code names are replaced by the numbering and names used in the long-form paper
(Algorithm 1 = IMP, 2 = flow attack, 3 = adversarial training, 4 = transfer).
Writes results/figures/els_overall_workflow.png, els_imp_flowchart.png and els_controls.png.

Usage: python src/make_els_diagrams.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "make_diagrams.py")
code = open(SRC, encoding="utf-8").read()

REPLACE = [
    ("PHASE 1 - Data preparation (Algorithm 1)", "PHASE 1 - Data preparation (Section 3.3)"),
    ("(Algorithms 2, 3)", "(Algorithm 1)"),
    ("(Algorithms 4, 5)", "(Algorithm 3)"),
    ("(Algorithm 7)", "(Algorithm 2)"),
    ("(Algorithm 6)", "(Algorithm 4)"),
    ("+ 5 controls (imp_reinit,\\nrandom_lw, ERK, SNIP, omp_ft)", "+ 5 controls (IMP re-init.,\\nRandom-LW, ERK, SNIP, OMP)"),
    ("features; dense_at", "features; dense AT"),
    ("+ omp_at, lamp_at, HYDRA", "+ OMP, LAMP, HYDRA"),
    ("build the 5 controls (Alg. 3)", "then build the 5 controls"),
    ('("imp_reinit", "ticket', '("IMP re-init.", "ticket'),
    ('("random_lw", "random', '("Random-LW", "random'),
    ('("omp_ft", "smallest', '("One-shot (OMP)", "smallest'),
    ('"diag_overall_workflow.png"', '"els_overall_workflow.png"'),
    ('"diag_imp_flowchart.png"', '"els_imp_flowchart.png"'),
    ('"diag_controls.png"', '"els_controls.png"'),
]
for a, b in REPLACE:
    assert code.count(a) == 1, (a, code.count(a))
    code = code.replace(a, b)
code = code.split('if __name__ == "__main__":')[0]
ns = {"__file__": SRC, "__name__": "make_diagrams_els"}
exec(compile(code, SRC, "exec"), ns)
ns["overall"]()
ns["imp"]()
ns["controls"]()
