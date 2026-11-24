#!/bin/bash
cd /home/chirag/Entanglement-recovery
D=logs/d7_9_combined
C="--start-x-range -0.485 -0.35 --jobs 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30"
F="--pair front --max-attempts FL 2 FR 2"
A="--pair all --max-attempts FL 2 FR 2 RL 2 RR 2 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C $F --seeds 0-39 --out-dir $D/front_0_39 > $D/front_0_39.txt 2>&1
echo 1 > $D/DONE1
uv run python scripts/d6_1_front_pair_wire.py $C $F --seeds 40-79 --out-dir $D/front_40_79 > $D/front_40_79.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C $A --seeds 0-39 --out-dir $D/four_0_39 > $D/four_0_39.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C $A --seeds 40-79 --out-dir $D/four_40_79 > $D/four_40_79.txt 2>&1
echo done > $D/DONE
