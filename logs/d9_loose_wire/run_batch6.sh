#!/bin/bash
# Step 2: best four-leg recipe (D7.9) on the loose-to-tight wire, seeds 0-9, oracle trigger.
# Conditions: slack 0 (control: my wire build, taut), slack 3 cm tightening in 0.5 s / 2 s / never.
# Baseline (original wire, same seeds) is the stored D7.9 result in logs/d7_9_combined/four_0_39.txt.
cd /home/chirag/Entanglement-recovery
R="--pair all --start-x-range -0.485 -0.35 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
export R
{
for cond in "0 2" "3 0.5" "3 2" "3 0"; do for s in 0 1 2 3 4 5 6 7 8 9; do echo "$cond $s"; done; done
} | xargs -P 2 -L 1 bash -c 'uv run python scripts/d9_loose_wire_recovery.py --slack-cm $0 --tighten-time $1 $R --seed $2 --out-dir logs/d9_loose_wire/rec/s$0_T$1 2>&1 | grep "^seed=" | sed "s/^/slack$0 T$1 /"' > logs/d9_loose_wire/run_batch6.txt
