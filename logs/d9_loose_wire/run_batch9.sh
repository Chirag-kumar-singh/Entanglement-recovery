#!/bin/bash
# Condition B (3 cm slack, tightening 2 s, anchor solref 0.02, overtravel 0) on seeds 10-29,
# best four-leg recipe, oracle trigger. Baseline = stored D7.9 results for the same seeds.
cd /home/chirag/Entanglement-recovery
R="--pair all --start-x-range -0.485 -0.35 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
export R
seq 10 29 | xargs -P 2 -I{} bash -c 'uv run python scripts/d9_loose_wire_recovery.py --slack-cm 3 --tighten-time 2 --anchor-solref 0.02 --overtravel-cm 0 $R --seed {} --out-dir logs/d9_loose_wire/rec/B_10_29 2>&1 | grep "^seed="' > logs/d9_loose_wire/run_batch9.txt
