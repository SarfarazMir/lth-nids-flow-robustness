#!/bin/bash
# Pilot v2 (flow-consistent attack included). Seed-major: RQ1 (2 sources) + RQ2 (3 pairs) per seed,
# then the adversarially-trained pruning comparison (exp_robust) per seed.
PY=${PYTHON:-python}   # set PYTHON to use a specific interpreter
cd "$(dirname "$0")"
LOCK=../logs/driver.lock
if [ -e "$LOCK" ]; then echo "another driver holds $LOCK (pid $(cat $LOCK)); refusing to start"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
SEEDS=${SEEDS:-"0 1 2"}
ROBUST_SRCS=${ROBUST_SRCS:-"ids17 ddos19_d1"}
for seed in $SEEDS; do
  for src in ids17 ddos19_d1; do
    $PY exp_rq1.py --src $src --eval_domains ddos19_d2,ids17,ddos19_d1 --seed $seed --rounds 20 --epochs 10 || echo "FAILED rq1 $src $seed"
  done
  $PY exp_rq2.py --src ids17 --tgt ddos19_d1 --seed $seed || echo "FAILED rq2 ids17->ddos19_d1 $seed"
  $PY exp_rq2.py --src ddos19_d1 --tgt ids17 --seed $seed || echo "FAILED rq2 ddos19_d1->ids17 $seed"
  $PY exp_rq2.py --src ddos19_d1 --tgt ddos19_d2 --seed $seed || echo "FAILED rq2 ddos19_d1->ddos19_d2 $seed"
  echo "SEED_DONE $seed $(date)"
done
echo PILOT_DONE
for seed in $SEEDS; do
  for src in $ROBUST_SRCS; do
    $PY exp_robust.py --src $src --seed $seed || echo "FAILED robust $src $seed"
  done
  echo "ROBUST_SEED_DONE $seed $(date)"
done
echo ROBUST_DONE
