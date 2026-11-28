#!/bin/bash
# Detector dataset v0 (small): seeds 200-237, kept apart from the tuning/eval seeds 0-159.
#   walk     taut wire at 6/8/10/12 cm, recovery off, 140 steps      (seeds 200-211)
#   loose    loose-to-tight wire (3 cm slack, 2 s, solref 0.02), recovery off, 160 steps (212-219)
#   recovery taut 10 cm wire, best four-leg recipe, 320 steps          (220-231)
#   nowire   wire at 50 cm (never touched), recovery off, 200 steps    (232-237)
cd /home/chirag/Entanglement-recovery
REC="--max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
BASE="--pair all --start-x-range -0.485 -0.35"
H=(0.06 0.08 0.10 0.12)
{
for s in $(seq 200 211); do echo "walk $s ${H[$((s % 4))]} --original-wire --entry-delay-steps 1000000000 --max-steps 140"; done
for s in $(seq 212 219); do echo "loose $s 0.10 --slack-cm 3 --tighten-time 2 --anchor-solref 0.02 --entry-delay-steps 1000000000 --max-steps 160"; done
for s in $(seq 220 231); do echo "recovery $s 0.10 --original-wire $REC"; done
for s in $(seq 232 237); do echo "nowire $s 0.5 --original-wire --entry-delay-steps 1000000000 --max-steps 200"; done
} | xargs -P 3 -L 1 bash -c 'kind=$0; seed=$1; h=$2; shift 2; uv run python scripts/d9_detector_data.py --kind $kind --seed $seed --wire-height $h '"$BASE"' "$@" --out-dir logs/d9_detector/scratch --data-dir logs/d9_detector/data_v0 2>&1 | grep -E "^(walk|loose|recovery|nowire)_|Error|rror:"'  > logs/d9_detector/run_data_v0.txt
