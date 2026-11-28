#!/bin/bash
# Detector dataset v1 (larger): seeds 300-479, apart from tuning/eval seeds 0-159 and v0 (200-237).
#   walk     taut wire at 6/8/10/12 cm, two start ranges, recovery off, 140 steps (300-359)
#   loose    loose-to-tight wire, slack 2/3/4 cm x tighten 1/2/3 s, recovery off   (360-407)
#   recovery taut 10 cm wire, best four-leg recipe, 320 steps                      (408-447)
#   nowire   wire at 50 cm (never touched), recovery off, 200 steps                (448-479)
# 3 processes max, one process per seed (laptop memory limit).
cd /home/chirag/Entanglement-recovery
REC="--max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
H=(0.06 0.08 0.10 0.12)
SL=(2 3 4)
TT=(1 2 3)
X1="--start-x-range -0.485 -0.35"
X2="--start-x-range -0.55 -0.40"
{
for s in $(seq 300 359); do x=$X1; [ $((s/4%2)) = 1 ] && x=$X2; echo "walk $s ${H[$((s % 4))]} $x --original-wire --entry-delay-steps 1000000000 --max-steps 140"; done
for s in $(seq 360 407); do echo "loose $s 0.10 $X1 --slack-cm ${SL[$((s % 3))]} --tighten-time ${TT[$((s/3 % 3))]} --anchor-solref 0.02 --entry-delay-steps 1000000000 --max-steps 160"; done
for s in $(seq 408 447); do echo "recovery $s 0.10 $X1 --original-wire $REC"; done
for s in $(seq 448 479); do echo "nowire $s 0.5 $X1 --original-wire --entry-delay-steps 1000000000 --max-steps 200"; done
} | xargs -P 3 -L 1 bash -c 'kind=$0; seed=$1; h=$2; shift 2; uv run python scripts/d9_detector_data.py --kind $kind --seed $seed --wire-height $h --pair all "$@" --out-dir logs/d9_detector/scratch --data-dir logs/d9_detector/data_v1 2>&1 | grep -E "^(walk|loose|recovery|nowire)_|Error|rror:"' > logs/d9_detector/run_data_v1.txt
