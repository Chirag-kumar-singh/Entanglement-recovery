#!/bin/bash
cd /home/chirag/Entanglement-recovery
D=logs/d7_10_rear
C="--pair all --start-x-range -0.485 -0.35 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --new-legs FL FR RL RR"
uv run python scripts/d6_1_front_pair_wire.py $C --seeds 0-39 --out-dir $D/C_full_0_39 > $D/C_full_0_39.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C --seeds 40-79 --out-dir $D/C_full_40_79 > $D/C_full_40_79.txt 2>&1
echo done > $D/DONE_FULL
