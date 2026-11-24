#!/bin/bash
cd /home/chirag/Entanglement-recovery
B="--pair front --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 2 --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR"
uv run python scripts/d6_1_front_pair_wire.py $B --out-dir logs/d7_5_small_lift/half --support-lift-thigh=-0.2 --support-lift-calf=0.3 > logs/d7_5_small_lift/half.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $B --out-dir logs/d7_5_small_lift/tenth --support-lift-thigh=-0.1 --support-lift-calf=0.15 > logs/d7_5_small_lift/tenth.txt 2>&1
uv run python scripts/d6_1_front_pair_wire.py $B --out-dir logs/d7_5_small_lift/calf --support-lift-thigh=0 --support-lift-calf=0.3 > logs/d7_5_small_lift/calf.txt 2>&1
