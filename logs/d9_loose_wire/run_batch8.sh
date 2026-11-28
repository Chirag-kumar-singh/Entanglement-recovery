#!/bin/bash
# Softer anchors, best four-leg recipe, seeds 0-9, oracle trigger. Conditions (slack cm, tighten s,
# anchor solref, overtravel cm):
#   A  0 2 0.02 0   validity control: should reproduce the original wire (D7.9)
#   B  3 2 0.02 0   loose, original-stiffness anchor
#   C  3 2 0.01 1   loose, softer-than-0.005 anchor with 1 cm pre-tension (best in the probe)
#   D  0 2 0.01 1   control for C
cd /home/chirag/Entanglement-recovery
R="--pair all --start-x-range -0.485 -0.35 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
export R
{
for cond in "0 2 0.02 0" "3 2 0.02 0" "3 2 0.01 1" "0 2 0.01 1"; do for s in 0 1 2 3 4 5 6 7 8 9; do echo "$cond $s"; done; done
} | xargs -P 2 -L 1 bash -c 'uv run python scripts/d9_loose_wire_recovery.py --slack-cm $0 --tighten-time $1 --anchor-solref $2 --overtravel-cm $3 $R --seed $4 --out-dir logs/d9_loose_wire/rec/s$0_sr$2_ot$3 2>&1 | grep "^seed=" | sed "s/^/slack$0 sr$2 ot$3 /"' > logs/d9_loose_wire/run_batch8.txt
