# 1. Gradient-masking checks (seed 0 networks; n = networks with valid clean accuracy)

| source    | threat model   |   networks |   transfer − direct ESR (mean, pp) |   max | cases transfer > direct (+1pp)   |   FGSM − PGD (mean, pp) | cases FGSM > PGD (+1pp)   |   CW − CE PGD (mean, pp) |
|:----------|:---------------|-----------:|-----------------------------------:|------:|:---------------------------------|------------------------:|:--------------------------|-------------------------:|
| ddos19_d1 | unconstrained  |         46 |                              -13.8 |  39.8 | 4/184                            |                    -0.6 | 0/46                      |                     -0.3 |
| ddos19_d1 | constrained    |         46 |                              -22.8 |   0.1 | 0/184                            |                    -1.0 | 0/46                      |                     -0.2 |
| ids17     | unconstrained  |         44 |                              -20.7 |  11.0 | 3/176                            |                    -4.2 | 0/44                      |                     -0.8 |
| ids17     | constrained    |         44 |                              -20.1 |  20.5 | 7/176                            |                   -11.8 | 0/44                      |                     -1.9 |


# 2a. Layer-wise density (% of weights kept per layer; mean over 5 seeds)

Layers: L1 = input→512 (38,912 weights), L2 = 512→512 (262,144), L3 = 512→256 (131,072), L4 = 256→2 (512).

| src       |   pruned % | method    |   L1 |     L2 |     L3 |    L4 |
|:----------|-----------:|:----------|-----:|-------:|-------:|------:|
| ddos19_d1 |       79   | erk       | 57.3 | 14.8   | 22.2   | 100   |
| ddos19_d1 |       79   | imp       | 68.7 | 17.1   | 14.2   |  73.9 |
| ddos19_d1 |       79   | omp_ft    | 67.5 | 16.3   | 16.4   |  57.2 |
| ddos19_d1 |       79   | random_lw | 68.7 | 17.1   | 14.2   |  73.9 |
| ddos19_d1 |       79   | snip      | 41.7 | 17.4   | 21.6   |  92   |
| ddos19_d1 |       89.3 | erk       | 29.2 |  7.5   | 11.3   | 100   |
| ddos19_d1 |       89.3 | imp       | 63.7 |  5.9   |  4.4   |  70   |
| ddos19_d1 |       89.3 | omp_ft    | 62.1 |  5.4   |  6.1   |  49.5 |
| ddos19_d1 |       89.3 | random_lw | 63.7 |  5.9   |  4.4   |  70   |
| ddos19_d1 |       89.3 | snip      | 29   |  7.5   | 11.5   |  86.9 |
| ddos19_d1 |       94.5 | erk       | 14.8 |  3.8   |  5.7   | 100   |
| ddos19_d1 |       94.5 | imp       | 55.1 |  0.565 |  0.432 |  59.1 |
| ddos19_d1 |       94.5 | omp_ft    | 54.4 |  0.48  |  0.878 |  38.7 |
| ddos19_d1 |       94.5 | random_lw | 55.1 |  0.565 |  0.432 |  59.1 |
| ddos19_d1 |       94.5 | snip      | 19.8 |  3.1   |  5.8   |  81.1 |
| ddos19_d1 |       97.2 | erk       |  7.4 |  1.9   |  2.9   | 100   |
| ddos19_d1 |       97.2 | imp       | 31   |  0.007 |  0.023 |  12.3 |
| ddos19_d1 |       97.2 | omp_ft    | 31.1 |  0.004 |  0.034 |   7.1 |
| ddos19_d1 |       97.2 | random_lw | 31   |  0.007 |  0.023 |  12.3 |
| ddos19_d1 |       97.2 | snip      | 13   |  1.2   |  2.8   |  74.3 |
| ids17     |       79   | erk       | 57.3 | 14.8   | 22.2   | 100   |
| ids17     |       79   | imp       | 68.7 | 17.1   | 14.4   |  76.4 |
| ids17     |       79   | omp_ft    | 66.1 | 16.4   | 16.5   |  62   |
| ids17     |       79   | random_lw | 68.7 | 17.1   | 14.4   |  76.4 |
| ids17     |       79   | snip      | 46.2 | 17.4   | 20.3   |  92.3 |
| ids17     |       89.3 | erk       | 29.2 |  7.5   | 11.3   | 100   |
| ids17     |       89.3 | imp       | 63   |  5.9   |  4.7   |  73.4 |
| ids17     |       89.3 | omp_ft    | 58.8 |  5.8   |  6.3   |  53.2 |
| ids17     |       89.3 | random_lw | 63   |  5.9   |  4.7   |  73.4 |
| ids17     |       89.3 | snip      | 32.4 |  7.5   | 10.5   |  87   |
| ids17     |       94.5 | erk       | 14.8 |  3.8   |  5.7   | 100   |
| ids17     |       94.5 | imp       | 53.7 |  0.684 |  0.571 |  67.1 |
| ids17     |       94.5 | omp_ft    | 48.7 |  1     |  1.4   |  40.7 |
| ids17     |       94.5 | random_lw | 53.7 |  0.684 |  0.571 |  67.1 |
| ids17     |       94.5 | snip      | 22.3 |  3     |  5.1   |  81.6 |
| ids17     |       97.2 | erk       |  7.4 |  1.9   |  2.9   | 100   |
| ids17     |       97.2 | imp       | 31   |  0.004 |  0.022 |  11.8 |
| ids17     |       97.2 | omp_ft    | 30.3 |  0.048 |  0.143 |  17.1 |
| ids17     |       97.2 | random_lw | 31   |  0.004 |  0.022 |  11.8 |
| ids17     |       97.2 | snip      | 14.7 |  1.2   |  2.3   |  74   |

### Weights left in the hidden layers at 97.2% pruning (mean and minimum over seeds)

|                            |   ('L2 weights kept', 'mean') |   ('L2 weights kept', 'min') |   ('L3 weights kept', 'mean') |   ('L3 weights kept', 'min') |
|:---------------------------|------------------------------:|-----------------------------:|------------------------------:|-----------------------------:|
| ('ddos19_d1', 'erk')       |                          5019 |                         5019 |                          3764 |                         3764 |
| ('ddos19_d1', 'imp')       |                            19 |                           15 |                            31 |                           20 |
| ('ddos19_d1', 'omp_ft')    |                            10 |                            7 |                            45 |                           16 |
| ('ddos19_d1', 'random_lw') |                            19 |                           15 |                            31 |                           20 |
| ('ddos19_d1', 'snip')      |                          3086 |                         2902 |                          3669 |                         3521 |
| ('ids17', 'erk')           |                          5019 |                         5019 |                          3764 |                         3764 |
| ('ids17', 'imp')           |                            10 |                            6 |                            29 |                           23 |
| ('ids17', 'omp_ft')        |                           126 |                          118 |                           187 |                          153 |
| ('ids17', 'random_lw')     |                            10 |                            6 |                            29 |                           23 |
| ('ids17', 'snip')          |                          3070 |                         2620 |                          3027 |                         2752 |


# 3. Features the flow attack can change (CIC-IDS2017 training flows)

- Probe used for the baseline marks 25 features as changeable.
- A wider probe (k ∈ {1, max}, payload ∈ {0, 1, 700, 1460} B, delay ∈ {0, 0.1 s, max}, all 198,307 training flows) marks 26.
- Features the baseline kept although the attack can change them: ['Min Packet Length']
  - Min Packet Length: changes for up to 99,762 of 198,307 training flows (50.3%) with one injected packet


# 2b. Input connectivity of IMP tickets (first layer, CIC-IDS2017 and CIC-DDoS2019)

Attacker-changeable features: 25 of 76 (share of first-layer weights in a dense or uniformly random mask: 32.9%).

| src       |   pruned % |   first-layer density % |   inputs fully cut |   changeable inputs cut |   share of kept L1 weights on changeable inputs % |   dense: share of |W1| on changeable inputs % |
|:----------|-----------:|------------------------:|-------------------:|------------------------:|--------------------------------------------------:|----------------------------------------------:|
| ddos19_d1 |       79.0 |                    68.7 |                0.0 |                     0.0 |                                              33.1 |                                          32.9 |
| ddos19_d1 |       89.3 |                    63.7 |                0.0 |                     0.0 |                                              33.1 |                                          32.9 |
| ddos19_d1 |       94.5 |                    55.1 |                0.0 |                     0.0 |                                              32.9 |                                          32.9 |
| ddos19_d1 |       97.2 |                    31.0 |                0.0 |                     0.0 |                                              32.8 |                                          32.9 |
| ids17     |       79.0 |                    68.7 |                0.0 |                     0.0 |                                              33.4 |                                          33.1 |
| ids17     |       89.3 |                    63.0 |                0.0 |                     0.0 |                                              33.4 |                                          33.1 |
| ids17     |       94.5 |                    53.7 |                0.0 |                     0.0 |                                              33.5 |                                          33.1 |
| ids17     |       97.2 |                    31.0 |                0.0 |                     0.0 |                                              33.5 |                                          33.1 |