#!/bin/bash
cd /home/chirag/Entanglement-recovery
D=logs/d7_12_height_gate
C="--pair all --start-x-range -0.485 -0.35 --seeds 0-79 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C --min-body-z RL 0.29 RR 0.29 --out-dir $D/z029 > $D/z029.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C --min-body-z RL 0.30 RR 0.30 --out-dir $D/z030 > $D/z030.txt 2>&1
echo done > $D/DONE
