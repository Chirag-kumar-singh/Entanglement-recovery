#!/bin/bash
cd /home/chirag/Entanglement-recovery
B="--pair all --start-x-range -0.485 -0.35 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $B --seeds 0-39 --out-dir logs/d7_6_four_leg_half/s0_39 > logs/d7_6_four_leg_half/s0_39.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $B --seeds 40-79 --out-dir logs/d7_6_four_leg_half/s40_79 > logs/d7_6_four_leg_half/s40_79.txt 2>&1
echo done > logs/d7_6_four_leg_half/DONE
