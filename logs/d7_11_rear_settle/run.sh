#!/bin/bash
cd /home/chirag/Entanglement-recovery
D=logs/d7_11_rear_settle
C="--pair all --start-x-range -0.485 -0.35 --seeds 0-79 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 24 --out-dir $D/gap24 > $D/gap24.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 12 --max-roll-deg RL 5 RR 5 --out-dir $D/roll5 > $D/roll5.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 24 --max-roll-deg RL 5 RR 5 --out-dir $D/gap24_roll5 > $D/gap24_roll5.txt 2>&1
echo done > $D/DONE
