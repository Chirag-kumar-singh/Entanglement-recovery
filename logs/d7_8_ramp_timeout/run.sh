#!/bin/bash
cd /home/chirag/Entanglement-recovery
B="--pair front --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 2 --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3"
D=logs/d7_8_ramp_timeout
uv run python scripts/d6_1_front_pair_wire.py $B --support-lift-ramp-steps 12 --out-dir $D/ramp12 > $D/ramp12.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $B --step-gate-timeout 30 --out-dir $D/to30 > $D/to30.txt 2>&1
echo done > $D/DONE
