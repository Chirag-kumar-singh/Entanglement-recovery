#!/bin/bash
cd /home/chirag/Entanglement-recovery
D=logs/d7_10_rear
SEEDS="4,8,11,18,20,21,31,32,35,46,64,67,73,75,0,1,2,3,5,7,9,10,12,13"
C="--pair all --start-x-range -0.485 -0.35 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30"
uv run python scripts/d6_1_front_pair_wire.py $C --seeds $SEEDS --new-legs FL FR RL RR --support-lift-trigger FL FR --out-dir $D/A_newlegs_all > $D/A_newlegs_all.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C --seeds $SEEDS --new-legs FL FR --out-dir $D/B_lift_all_steps > $D/B_lift_all_steps.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C --seeds $SEEDS --new-legs FL FR RL RR --out-dir $D/C_both > $D/C_both.txt 2>&1
echo done > $D/DONE
