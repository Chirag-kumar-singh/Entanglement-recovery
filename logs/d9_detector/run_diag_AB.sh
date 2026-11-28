#!/bin/bash
# Diagnostics on seeds 10-49 with model_v2 (sequential, 3 processes each):
#  A: detector triggers FR/FL only, oracle triggers RR/RL
#  B: detector triggers all legs but a leg's alarm is ignored until its oracle contact happened
cd /home/chirag/Entanglement-recovery
logs/d9_detector/run_closed_loop.sh 10 49 logs/d9_detector/closed_loop/diagA_front_det_s10_49.txt --model logs/d9_detector/model_v2.pt --act-legs FR FL
logs/d9_detector/run_closed_loop.sh 10 49 logs/d9_detector/closed_loop/diagB_noearly_s10_49.txt --model logs/d9_detector/model_v2.pt --no-early
