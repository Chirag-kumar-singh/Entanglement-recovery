#!/bin/bash
cd /home/chirag/Entanglement-recovery
D=logs/d7_14_gap3
C="--pair all --start-x-range -0.485 -0.35 --seeds 0-79 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 3 --out-dir $D/gap3 > $D/gap3.txt 2>&1
echo done > $D/DONE
