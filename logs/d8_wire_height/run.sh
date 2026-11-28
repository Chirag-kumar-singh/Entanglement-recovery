#!/bin/bash
cd /home/chirag/Entanglement-recovery
D=logs/d8_wire_height
C="--pair all --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
for h in 0.06 0.08 0.12 0.14; do
  uv run python scripts/d6_1_front_pair_wire.py $C --wire-height $h --out-dir $D/h$h > $D/h$h.txt 2>&1
done
echo done > $D/DONE
