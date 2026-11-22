# Lottery-ticket NIDS under flow-consistent evasion: code, models and results

This repository contains the code, the preprocessed data, the trained models and all raw and aggregated results of the study
*"Statistical Evaluation of Adversarial Robustness in Lottery-Ticket-Pruned Deep Network Intrusion Detection Systems Under Flow-Consistent Attacks"* (under review).

The study asks whether lottery tickets found by iterative magnitude pruning (IMP) of a flow-based deep network intrusion
detector stay robust to evasion. It uses a flow-consistent attack that adds forward packets, payload and delay within a
relative budget and recomputes the dependent CICFlowMeter statistics, on CIC-IDS2017 and CIC-DDoS2019 with five seeds.

| Experiment | Question | Script | Results |
|---|---|---|---|
| A | When do tickets become easy to evade, and which ingredient (initialization, positions, layer allocation, pruning schedule) is implicated? Five matched-sparsity controls. | `src/exp_rq1.py` | `results/raw/rq1/` |
| B | Do adversarial training (dense AT, adversarial IMP) and robust pruning (OMP, LAMP, HYDRA-style) remove the weakness? Immutable-feature baseline. CIC-IDS2017 only. | `src/exp_robust.py` | `results/raw/robust/` |
| C | Does a ticket found on one NIDS dataset help on another (1 % or 100 % of target labels, both directions)? | `src/exp_rq2.py` | `results/raw/rq2/` |
| Mechanism | Per-layer weight counts and Integrated Gradients attributions. | `src/analyze_xai.py`, `src/analyze_mechanism.py` | `results/raw/xai/`, `results/tables/mechanism_summary.md` |

**Research questions and folder names.** The code uses an older numbering than the paper. The paper's RQ1 (when do tickets
become easy to evade?) and RQ2 (which ingredient is implicated?) are both Experiment A (`exp_rq1.py`, `results/raw/rq1/`);
adversarial training and robust pruning are Experiment B (`exp_robust.py`, `results/raw/robust/`); the paper's **RQ3**
(does a ticket transfer to another dataset?) is Experiment C, in **`exp_rq2.py` and `results/raw/rq2/`**. Likewise, files and
tables named `rq1_*` / `rq2_*` refer to Experiments A / C.

Headline result (CIC-IDS2017, five seeds; `results/tables/full5_summary.md`): integrated flow-attack evasion (ESR-AUC) of the
97.2 % ticket is 73.8 % versus 44.6 % for the dense detector; tickets are more evadable in every seed at 79, 89.3, 94.5 and 97.2 % sparsity.

## Repository layout

```
src/                        all code (see "Scripts" below)
  lthlib/                   library: data split and scaling, MLP, pruning (IMP, ERK, SNIP, LAMP, ...), HYDRA-style scores,
                            training, PGD/FGSM attacks, the flow-consistent attack (flowattack.py), evaluation
data_cache/                 preprocessed datasets used by every experiment (Parquet) + preprocessing metadata (*_meta.json)
results/raw/                every run's raw output (JSON) and the saved model checkpoints (*.pt)
results/tables/             aggregated tables, statistical tests and summaries (Markdown / CSV)
results/figures/            all figures and diagrams
logs/                       training and evaluation logs of every run; logs/full_done/ holds the run-driver markers
reproduce.sh                every step of the study, in order
```

## Code

Library (`src/lthlib/`):

| Module | Content |
|---|---|
| `data.py` | loading the cached datasets, the stratified 70/15/15 split, label subsampling, signed-log + standard scaling (`Preprocessor`), class weights |
| `common.py` | experiment setup shared by all scripts: dataset domains, the evaluation bundle, the fixed 3,000-flow attack pool, PGD budget grids, clean and robust evaluation, run-environment logging |
| `models.py` | the MLP 76-512-512-256-2 (and a 1-D CNN that was not used in the study) |
| `prune.py` | masks and pruning criteria: global magnitude (IMP), one-shot magnitude, random with given layer densities, ERK, SNIP, GraSP, SynFlow, LAMP |
| `hydra.py` | HYDRA-style score learning under the adversarial loss, re-implemented for the MLP |
| `train.py` | masked training (AdamW, one-cycle schedule, class-weighted loss), optional adversarial training, evaluation metrics |
| `attacks.py` | feature-space PGD and FGSM (unconstrained and restricted to attacker-influenceable features), CW-loss variant, transfer attacks |
| `flowattack.py` | the flow-consistent attack: packet/payload/delay budget, recomputation of the dependent CICFlowMeter statistics, candidate search, ESR and ESR-AUC; the cheaper search used inside adversarial training (`craft`) |

Scripts (`src/`):

| Script | Purpose |
|---|---|
| `convert_ddos2019_to_parquet.py`, `prepare_data.py` | data pipeline (see "Data") |
| `test_flowattack.py` | identity check of the flow-feature recomputation and dense-model ESR |
| `calibrate_eps.py` | PGD budget calibration on the seed-0 dense networks |
| `exp_rq1.py`, `exp_robust.py`, `exp_rq2.py` | Experiments A, B and C |
| `run_pilot.sh`, `run_full.sh` | run drivers (seed 0; seeds 1-4) |
| `learning_curves.py` | retrains selected networks to log per-epoch curves and a confusion matrix |
| `analyze_xai.py` | Integrated Gradients attributions |
| `analyze.py` | per-condition summaries of Experiments A-C (`rq1_*`, `rq2_*`, `robust_*` tables and plots) |
| `analyze_full.py` | five-seed tables and pooled Wilcoxon/Holm tests of Experiments A-C (`full5_*`) |
| `analyze_extra.py` | collapse-threshold sensitivity, seed-level tests, effect sizes, Friedman tests, per-budget results, storage/FLOPs/latency |
| `analyze_mechanism.py` | gradient-masking checks, per-layer densities, attacker-changeable features, first-layer input connectivity, and which features the flow attack changes beyond the single probe of the original immutable-feature baseline (the reason for the corrected `_immfix` runs) |
| `make_figures.py`, `make_extra_figures.py`, `make_ieee_figures.py`, `make_diagrams.py`, `make_els_diagrams.py`, `make_graphical_abstract.py` | figures and diagrams (see "Figures") |
| `time_at.py`, `time_flow.py` | one-off timing probes used to size the runs (adversarial-training epoch and flow-attack evaluation cost); they only print timings |

## Setup

Python 3.12.7 with the versions used for all runs:

```
pip install -r requirements.txt
```

All runs used one NVIDIA GTX 1660 Ti (6 GB, CUDA 12.1) under Windows 11; a GPU is recommended for the experiments
(the analyses and figures run on a CPU). Bash scripts were run with Git Bash; set `PYTHON` to choose the interpreter.

## Data

The preprocessed datasets that all experiments use are included in `data_cache/`
(`ids2017.parquet`, `ddos2019_day1.parquet`, `ddos2019_day2.parquet`), so the experiments can be rerun without the raw data.
The split (70/15/15, stratified by the original label, seed 0) and the 3,000 attack flows per dataset are drawn from these files
by the code, identically for every run.

To rebuild `data_cache/` from the raw data (both datasets are published by the Canadian Institute for Cybersecurity,
University of New Brunswick: https://www.unb.ca/cic/datasets/ids-2017.html and https://www.unb.ca/cic/datasets/ddos-2019.html):

1. CIC-IDS2017: the eight `MachineLearningCVE` CSV files.
   `python src/prepare_data.py ids2017 --root <folder with the 8 CSV files>`
2. CIC-DDoS2019: the CSV folders `CSV-01-12` (day 1) and `CSV-03-11` (day 2), converted to Parquet first:
   `python src/convert_ddos2019_to_parquet.py --train_csv <CSV-01-12> --test_csv <CSV-03-11> --out <dir>`, then
   `python src/prepare_data.py ddos2019 --day 1 --ddos_root <dir>` and `python src/prepare_data.py ddos2019 --day 2 --ddos_root <dir>`.

Preprocessing keeps the 76 flow statistics shared by both datasets, drops identifiers (flow ID, IPs, ports, timestamp),
Bernoulli-thins rows towards a nominal 30,000 per file (CIC-IDS2017: benign and attack rows separately; CIC-DDoS2019: attack
rows only), removes rows with infinite or missing values and rows duplicated in the 76 features plus the original label,
and keeps feature vectors that occur with different labels (their counts are in `data_cache/*_meta.json`).
The split is not feature-disjoint: a small number of test rows have an exact copy in the training data.

If you use the data, cite the dataset papers: I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward generating a new intrusion
detection dataset and intrusion traffic characterization," ICISSP 2018; I. Sharafaldin, A. H. Lashkari, S. Hakak, A. A. Ghorbani,
"Developing realistic distributed denial of service (DDoS) attack dataset and taxonomy," ICCST 2019.
The datasets remain under the terms of their publisher; the MIT licence of this repository covers the code, models and results.

## Reproducing the study

`reproduce.sh` lists every command in the order used. In short:

| Step | Command | Output |
|---|---|---|
| Data | `prepare_data.py` (see above) | `data_cache/` |
| Attack check | `python src/test_flowattack.py` | `results/raw/flowattack_validation.json` |
| PGD budgets | `python src/calibrate_eps.py` | `results/raw/calibrate_eps.json` |
| Seed 0 (pilot, superset of conditions) | `bash src/run_pilot.sh` with `SEEDS=0 ROBUST_SRCS=ids17` | `results/raw/{rq1,rq2,robust}/*_s0*` |
| Seeds 1-4 | `bash src/run_full.sh` (resumable; markers in `logs/full_done/`) | `results/raw/{rq1,rq2,robust}/` |
| Corrected immutable-feature baseline | `python src/exp_robust.py --src ids17 --seed S --wide_probe --immutable_only --no_sanity --tag _immfix` for S = 0..4 | `results/raw/robust/*_immfix.json` |
| Learning curves | `python src/learning_curves.py` | `results/raw/curves/` |
| Integrated Gradients | `python src/analyze_xai.py` | `results/raw/xai/` |
| Tables and tests | `analyze.py`, `analyze_full.py`, `analyze_extra.py`, `analyze_mechanism.py` | `results/tables/` |
| Figures | `make_figures.py`, `make_extra_figures.py`, `make_ieee_figures.py`, `make_diagrams.py`, `make_els_diagrams.py`, `make_graphical_abstract.py <folder>` | `results/figures/` |

On Windows, set `PYTHONIOENCODING=utf-8` so that the analysis scripts can print their tables to the console.
Rerunning the analyses reproduces every table exactly except the CPU latency columns of `extra_summary.md`, which are
re-measured on each run (the included file holds the measurements reported in the paper).
On the GPU above, the seed-0 pilot took about 3 hours and seeds 1-4 about 5 hours (timestamps in `logs/`). Seeds fix initialization, data order, random masks and attack sampling;
results on other hardware or library versions can differ slightly in the last digits.

Experiment B on CIC-DDoS2019 (stage 3 of `run_full.sh`) was not run: its markers in `logs/full_done/rob_ddos_s*` contain
"SKIPPED". Delete them and rerun `run_full.sh` to run it.

## Results

### Raw results (`results/raw/`)

| Path | Content |
|---|---|
| `rq1/<src>_mlp_s<seed>.json` | Experiment A: `args`, environment, and one record per network (dense, IMP round 1-20, controls) with clean metrics (balanced accuracy, DR, FPR, ROC-AUC, ...), cross-dataset BA, flow-attack ESR per budget and ESR-AUC, PGD ESR per budget and ESR-AUC, layer densities |
| `robust/ids17_mlp_s<seed>_flow1.0.json` | Experiment B: dense AT, A-IMP rounds 4/7/10/13/16, OMP/LAMP/HYDRA with adversarial fine-tuning, immutable-feature baseline (`*_immfix.json`: the corrected baseline) |
| `rq2/<src>__<tgt>_mlp_s<seed>.json` | Experiment C: seven conditions per target label budget and round, target BA and flow ESR-AUC |
| `xai/` | Integrated Gradients attributions (`importance.npy`), per-network summaries (`xai_records.json`), feature lists |
| `curves/` | per-epoch learning curves (standard and adversarial training) and a confusion matrix |
| `calibrate_eps.json`, `flowattack_validation.json` | PGD budget calibration; flow-attack identity check and dense ESR |
| `_aborted_pilot_v1/`, `_aborted_pilot_v2_concurrent/`, `_stopped_pilot_v3/` | early pilot runs that were stopped or superseded (kept for completeness; not used in any table) |

Source names: `ids17` = CIC-IDS2017, `ddos19_d1` / `ddos19_d2` = CIC-DDoS2019 day 01-12 / 03-11. Sparsity of IMP round r is
1 - 0.8^r (rounds 4, 7, 10, 13, 16, 20 = 59.0, 79.0, 89.3, 94.5, 97.2, 98.8 %). A network with balanced accuracy below 0.90 is
"collapsed" and excluded from the robustness comparisons.

### Models (`*_state.pt`)

PyTorch dictionaries of `state_dict`s for the MLP 76-512-512-256-2 (`src/lthlib/models.py`):

- `results/raw/rq1/<src>_mlp_s<seed>_state.pt`: `dense` (trained dense network), `theta0` (initialization), `theta_k`
  (rewind point, 5 % of training), `imp_masks[r]` and `imp_weights[r]` (mask and trained weights of the IMP ticket after round r = 1..20).
- `results/raw/robust/ids17_mlp_s<seed>_flow1.0_state.pt`: `dense_at` (adversarially trained dense network), `theta_k`, `aimp_masks[r]` (r = 1..16).

Control networks, OMP/LAMP/HYDRA networks and Experiment C networks were evaluated when trained and not saved; the scripts retrain them.

```python
import sys, torch
sys.path.insert(0, "src")                   # run from the repository root
from lthlib.models import MLP
s = torch.load("results/raw/rq1/ids17_mlp_s0_state.pt", map_location="cpu", weights_only=False)
net = MLP(76); net.load_state_dict(s["imp_weights"][16]); net.eval()   # the 97.2 % ticket, seed 0
```

Inputs must be scaled as in training (signed log, then standardization fitted on the training split: `lthlib.data.Preprocessor`,
used through `lthlib.common.Domain` / `make_pre`).

### Tables (`results/tables/`)

| File | Content | Script |
|---|---|---|
| `full5_summary.md` | five-seed results and pooled tests of all three experiments (the source of the paper's numbers) | `analyze_full.py` |
| `full5_expA_tests.csv`, `full5_expB_tests.csv`, `full5_expC_tests.csv` | every paired Wilcoxon test with mean/median difference, number of positive pairs, p and Holm-corrected p | `analyze_full.py` |
| `extra_summary.md` | collapse-threshold sensitivity, seed-level tests, effect sizes, Friedman tests, per-budget evasion, storage/FLOPs/latency | `analyze_extra.py` |
| `mechanism_summary.md` | gradient-masking checks (seed 0), per-layer densities and hidden-layer weight counts, features the flow attack can change, first-layer input connectivity of the tickets | `analyze_mechanism.py` |
| `rq1_summary.md` / `.csv`, `rq1_paired_vs_imp.csv` | Experiment A: mean ± s.d. per condition and round; paired differences to the IMP ticket | `analyze.py` |
| `robust_summary.md` / `.csv`, `robust_paired_vs_aimp.csv` | Experiment B: the same, paired with A-IMP | `analyze.py` |
| `rq2_summary.md`, `rq2_paired_vs_tkt_rewind.csv` | Experiment C: the same, paired with ticket rewind (includes the seed-0 pilot conditions: 10 % labels, round 7, day-2 target) | `analyze.py` |
| `analyze_stdout.txt` | console output of an earlier `analyze.py` run (library warnings only) | `analyze.py` |

### Figures (`results/figures/`)

| Files | Content | Script |
|---|---|---|
| `paper_fig1_concept.png` | weights kept per layer: dense, ticket and ERK at 97.2 % | `make_figures.py` |
| `paper_fig2_pipeline.png` | preprocessing pipeline (compact) | `make_figures.py` |
| `paper_fig3_robustness.png` | flow ESR-AUC versus sparsity for the ticket and the controls, both datasets; per-budget panel | `make_figures.py` |
| `paper_fig4_layers.png`, `paper_fig4_layers_col.png` | share of weights kept per layer (two layouts) | `make_figures.py` |
| `paper_fig5_xai.png` | Integrated Gradients attributions | `make_figures.py` |
| `paper_fig6_adversarial.png` | robustness versus accuracy of all adversarially trained networks | `make_figures.py` |
| `paper_fig7_transfer.png` | target balanced accuracy of the transfer conditions | `make_figures.py` |
| `ieee_fragility.pdf/.png`, `ieee_workflow.pdf/.png` | fragility curve (clean BA and flow ESR-AUC over the IMP rounds) and compact workflow, vector versions | `make_ieee_figures.py` |
| `res_accuracy_vs_rounds.png`, `res_esr_vs_budget.png`, `res_learning_curves.png`, `res_learning_curves_at.png`, `res_confusion.png` | clean BA at all rounds, evasion versus budget, learning curves (standard and adversarial training), confusion matrices before/after the attack | `make_extra_figures.py` |
| `diag_*.png` | workflow, preprocessing, architecture, IMP, controls, flow-attack, adversarial-training and transfer diagrams | `make_diagrams.py` |
| `els_overall_workflow.png`, `els_imp_flowchart.png`, `els_controls.png` | the same three diagrams with the algorithm numbering of the long-form paper | `make_els_diagrams.py` |
| `full5_expA.png`, `full5_expC.png` | overview plots of the five-seed Experiments A and C | `analyze_full.py` |
| `rq1_ids17.png`, `rq1_ddos19_d1.png`, `robust_ids17.png` | overview plots of Experiments A and B | `analyze.py` |

The graphical abstract is written by `make_graphical_abstract.py <output folder>` and is not stored here.

### Logs (`logs/`)

One log per run (`rq1_*`, `rq2_*`, `robust_*`, including the `_immfix` runs), the preprocessing logs (`prep_*.log`), the
learning-curve and attribution logs, and the two driver logs (`pilot_driver.log`, `full_driver.log`) with start and end times
of every job. `logs/full_done/` holds the resumable-driver markers (the `rob_ddos_s*` markers contain "SKIPPED").
`data_cache/*_meta.json` record, per source file, the rows read and kept and the rows removed by each cleaning step.

## Licence

Code, models and results: MIT (see `LICENSE`). Datasets: see "Data".
