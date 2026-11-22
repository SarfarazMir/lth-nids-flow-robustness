# 1. Sensitivity of 'IMP ticket − dense' (flow ESR-AUC, CIC-IDS2017, levels 79–97.2 %) to the collapse threshold

|   threshold (bacc ≥) |   pairs |   mean Δ (pp) | ticket worse   |   Wilcoxon p |   rank-biserial r |
|---------------------:|--------:|--------------:|:---------------|-------------:|------------------:|
|                 0    |      20 |        12.41  | 20/20          |    1.907e-06 |                 1 |
|                 0.8  |      20 |        12.41  | 20/20          |    1.907e-06 |                 1 |
|                 0.9  |      20 |        12.41  | 20/20          |    1.907e-06 |                 1 |
|                 0.95 |      20 |        12.41  | 20/20          |    1.907e-06 |                 1 |
|                 0.98 |      15 |         6.813 | 15/15          |    6.104e-05 |                 1 |


# 2. Evasion success rate (%) per flow-attack budget ρ, CIC-IDS2017 (mean ± sd over valid networks)

| network    |   n | ρ=0.1      | ρ=0.25      | ρ=0.5       | ρ=1.0       | ρ=2.0       |
|:-----------|----:|:-----------|:------------|:------------|:------------|:------------|
| dense      |   5 | 29.8 ± 1.1 | 34.3 ± 3.4  | 40.5 ± 4.1  | 47.0 ± 3.3  | 56.4 ± 6.2  |
| imp 94.5%  |   5 | 40.1 ± 5.2 | 42.1 ± 4.7  | 50.2 ± 2.7  | 56.6 ± 4.6  | 64.6 ± 5.5  |
| erk 94.5%  |   5 | 31.0 ± 3.1 | 33.1 ± 3.4  | 41.3 ± 3.9  | 51.1 ± 2.3  | 61.6 ± 3.2  |
| snip 94.5% |   5 | 32.3 ± 2.5 | 35.1 ± 1.4  | 43.2 ± 2.4  | 47.6 ± 4.1  | 58.8 ± 3.2  |
| imp 97.2%  |   5 | 60.4 ± 9.0 | 67.2 ± 10.6 | 71.8 ± 11.3 | 77.6 ± 12.3 | 82.9 ± 13.9 |
| erk 97.2%  |   5 | 28.0 ± 2.1 | 30.2 ± 2.6  | 37.9 ± 4.8  | 45.7 ± 3.9  | 55.0 ± 4.2  |
| snip 97.2% |   5 | 32.3 ± 9.0 | 34.2 ± 8.9  | 42.8 ± 7.8  | 48.9 ± 6.2  | 58.3 ± 6.3  |


# 3. Effect sizes (matched-pairs rank-biserial r; +1 = first condition always higher), flow ESR-AUC, valid networks, levels 79–97.2 %

| src       | comparison       |     r |
|:----------|:-----------------|------:|
| ddos19_d1 | imp − dense      |  0.09 |
| ddos19_d1 | imp_reinit − imp |  0.10 |
| ddos19_d1 | random_lw − imp  |  0.58 |
| ddos19_d1 | erk − imp        |  0.13 |
| ddos19_d1 | snip − imp       |  0.45 |
| ddos19_d1 | omp_ft − imp     |  0.10 |
| ids17     | imp − dense      |  1.00 |
| ids17     | imp_reinit − imp | -0.38 |
| ids17     | random_lw − imp  | -0.19 |
| ids17     | erk − imp        | -0.87 |
| ids17     | snip − imp       | -0.69 |
| ids17     | omp_ft − imp     | -0.38 |


# 4. Friedman test over 7 standard-training conditions (flow ESR-AUC; blocks = seed × level 79/89.3/94.5 %)

| src       |   blocks |    χ² |       p |   Kendall W |   rank dense |   rank imp |   rank imp_reinit |   rank random_lw |   rank erk |   rank snip |   rank omp_ft |
|:----------|---------:|------:|--------:|------------:|-------------:|-----------:|------------------:|-----------------:|-----------:|------------:|--------------:|
| ddos19_d1 |       15 |  7.49 | 0.278   |      0.0832 |         3.87 |       3.13 |              4.27 |             4.4  |        4.2 |        4.87 |          3.27 |
| ids17     |       15 | 22.3  | 0.00106 |      0.248  |         2.4  |       5.47 |              3.87 |             5.27 |        4   |        3.73 |          3.27 |

(Mean rank 1 = least evadable, 7 = most evadable.)


# 5. Efficiency of IMP tickets (CIC-IDS2017, seed 0; single CPU thread)

| pruned %   |   non-zero weights |   MFLOPs / flow |   storage dense fp32 (KB) |   storage CSR (KB) |   dense-kernel latency bs=1 (ms) |   CSR latency bs=1 (ms) |   dense-kernel latency bs=1024 (ms) |   CSR latency bs=1024 (ms) |
|:-----------|-------------------:|----------------:|--------------------------:|-------------------:|---------------------------------:|------------------------:|------------------------------------:|---------------------------:|
| 0 (dense)  |             432640 |          0.865  |                  1.69e+03 |           3.39e+03 |                            0.192 |                   1.24  |                                9.7  |                       53.7 |
| 59.0       |             177209 |          0.354  |                  1.69e+03 |           1.39e+03 |                            0.19  |                   0.612 |                               10.5  |                       27.1 |
| 79.0       |              90731 |          0.181  |                  1.69e+03 |         714        |                            0.191 |                   0.446 |                               10.1  |                       18.7 |
| 89.3       |              46454 |          0.0929 |                  1.69e+03 |         368        |                            0.191 |                   0.353 |                                9.54 |                       13.9 |
| 94.5       |              23785 |          0.0476 |                  1.69e+03 |         191        |                            0.19  |                   0.315 |                               10.5  |                       11   |
| 97.2       |              12178 |          0.0244 |                  1.69e+03 |         100        |                            0.19  |                   0.288 |                               10.6  |                       11.3 |