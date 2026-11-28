#!/bin/bash
# Closed loop: sim-trained CNN (model_v1, thr 0.9, 1 step) as the recovery entry trigger,
# best four-leg recipe (D7.9), original wire at 10 cm, seeds $1..$2, 3 processes.
# usage: run_closed_loop.sh FIRST LAST OUTFILE [extra flags]
cd /home/chirag/Entanglement-recovery
REC="--max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
first=$1; last=$2; out=$3; shift 3
seq $first $last | xargs -P 3 -I{} bash -c "uv run python scripts/d9_detector_closed_loop.py --act $* --kind closed --seed {} --pair all --start-x-range -0.485 -0.35 --original-wire --wire-height 0.10 $REC --data-dir ${DATA_DIR:-logs/d9_detector/scratch_cl} 2>&1 | grep -E '^closed_|DETECT (per_leg|first)|rror' | sed 's/^/s{} /'" > $out
