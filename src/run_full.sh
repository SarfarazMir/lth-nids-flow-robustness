#!/bin/bash
# Full study, lean design. Seed 0 = pilot (superset, run by run_pilot.sh).
# Priority order, so an interrupted night still leaves the most useful results:
#   stage 1: seeds 1,2   Exp A (both datasets) + Exp C (two cross-dataset pairs) + Exp B (CIC-IDS2017)
#   stage 2: seeds 3,4   same
#   stage 3: seeds 0-4   Exp B on CIC-DDoS2019
# Resumable: each finished job leaves a marker in logs/full_done/ and is skipped on restart.
PY=${PYTHON:-python}   # set PYTHON to use a specific interpreter
cd "$(dirname "$0")"
LOCK=../logs/driver.lock
if [ -e "$LOCK" ]; then echo "another driver holds $LOCK (pid $(cat $LOCK)); refusing to start"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
DONE=../logs/full_done; mkdir -p "$DONE"

RQ1="--rounds 20 --epochs 10 --eval_rounds 4,7,10,13,16,20 --baseline_rounds 7,10,13,16
     --baselines imp_reinit,random_lw,erk,snip,omp_ft --no_surrogate --no_sanity"
RQ2="--fracs 0.01,1.0 --rounds 10,13,16"
ROB="--eval_rounds 4,7,10,13,16 --baseline_rounds 10,13,16 --methods omp_at,lamp_at,hydra --no_sanity"

job() {  # job <id> <command...>
  local id=$1; shift
  if [ -e "$DONE/$id" ]; then echo "SKIP $id (done)"; return; fi
  echo "START $id $(date)"
  if "$@"; then touch "$DONE/$id"; echo "OK $id $(date)"; else echo "FAILED $id $(date)"; fi
}

core_seed() {
  local s=$1
  job rq1_ids17_s$s   $PY exp_rq1.py --src ids17 --eval_domains ddos19_d2,ids17,ddos19_d1 --seed $s $RQ1
  job rq1_ddos_s$s    $PY exp_rq1.py --src ddos19_d1 --eval_domains ddos19_d2,ids17,ddos19_d1 --seed $s $RQ1
  job rq2_ids2ddos_s$s $PY exp_rq2.py --src ids17 --tgt ddos19_d1 --seed $s $RQ2
  job rq2_ddos2ids_s$s $PY exp_rq2.py --src ddos19_d1 --tgt ids17 --seed $s $RQ2
  job rob_ids17_s$s   $PY exp_robust.py --src ids17 --seed $s $ROB
  echo "CORE_SEED_DONE $s $(date)"
}

for s in ${STAGE1:-1 2}; do core_seed $s; done
echo STAGE1_DONE
for s in ${STAGE2:-3 4}; do core_seed $s; done
echo STAGE2_DONE
for s in ${STAGE3:-0 1 2 3 4}; do
  job rob_ddos_s$s $PY exp_robust.py --src ddos19_d1 --seed $s $ROB
done
echo STAGE3_DONE
echo FULL_DONE $(date)
