#!/bin/bash
# Original wire + overlap check, best four-leg recipe, seeds 0-9 (baseline for the overlap numbers;
# verdicts should equal the stored D7.9 results).
cd /home/chirag/Entanglement-recovery
R="--pair all --start-x-range -0.485 -0.35 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
export R
seq 0 9 | xargs -P 2 -I{} bash -c 'uv run python scripts/d9_loose_wire_recovery.py --original-wire $R --seed {} --out-dir logs/d9_loose_wire/rec/original 2>&1 | grep "^seed="' > logs/d9_loose_wire/run_batch7.txt
