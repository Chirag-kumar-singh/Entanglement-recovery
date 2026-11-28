#!/bin/bash
# usage: run_delay.sh <delay_steps> <seeds, e.g. 0-9>
cd /home/chirag/Entanglement-recovery
source logs/d9_entry_delay/common.sh
d=$1; s=$2
uv run python scripts/d6_1_front_pair_wire.py $C --seeds $s --entry-delay-steps $d \
  --out-dir logs/d9_entry_delay/scratch/d${d} > logs/d9_entry_delay/delay${d}_seeds${s}.txt 2>&1
