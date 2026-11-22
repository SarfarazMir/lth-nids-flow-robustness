#!/bin/bash
# Every step of the study, in the order used. Run from the repository root with Git Bash or any POSIX shell.
# Set PYTHON to choose the interpreter (default: python). Long steps are resumable; see README.md.
set -e
PY=${PYTHON:-python}
cd "$(dirname "$0")"

# 0. Data (optional: data_cache/ is included). Paths to the raw CIC downloads:
#    $PY src/prepare_data.py ids2017 --root <MachineLearningCVE folder>
#    $PY src/convert_ddos2019_to_parquet.py --train_csv <CSV-01-12> --test_csv <CSV-03-11> --out <dir>
#    $PY src/prepare_data.py ddos2019 --day 1 --ddos_root <dir>
#    $PY src/prepare_data.py ddos2019 --day 2 --ddos_root <dir>

# 1. Checks and budget calibration
$PY src/test_flowattack.py
$PY src/calibrate_eps.py

# 2. Seed 0 pilot (superset of conditions): Experiments A and C on both datasets, Experiment B on CIC-IDS2017
SEEDS=0 ROBUST_SRCS=ids17 bash src/run_pilot.sh

# 3. Seeds 1-4 (stage 3, Experiment B on CIC-DDoS2019, was not run: logs/full_done/rob_ddos_s* = SKIPPED)
bash src/run_full.sh

# 4. Corrected immutable-feature baseline (wide perturbation probe)
for s in 0 1 2 3 4; do
  (cd src && $PY exp_robust.py --src ids17 --seed $s --wide_probe --immutable_only --no_sanity --tag _immfix)
done

# 5. Learning curves and Integrated Gradients
(cd src && $PY learning_curves.py)
(cd src && $PY analyze_xai.py)

# 6. Tables and statistical tests
(cd src && $PY analyze.py && $PY analyze_full.py && $PY analyze_extra.py && $PY analyze_mechanism.py)

# 7. Figures
(cd src && $PY make_figures.py && $PY make_extra_figures.py && $PY make_ieee_figures.py && $PY make_diagrams.py && $PY make_els_diagrams.py)
$PY src/make_graphical_abstract.py results/graphical_abstract
