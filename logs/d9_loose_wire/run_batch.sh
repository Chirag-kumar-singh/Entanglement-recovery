#!/bin/bash
# Loose-to-tight wire probe: seeds 0-2 x slack {0.5,1.0} cm x tighten time {0.5,2,4} s,
# plus a never-tighten control (T=0) at slack 1.0. 2 processes at a time.
cd /home/chirag/Entanglement-recovery
{
for s in 0 1 2; do
  for k in 0.5 1.0; do for T in 0.5 2.0 4.0; do echo "$s $k $T"; done; done
  echo "$s 1.0 0"
done
} | xargs -P 2 -L 1 bash -c 'uv run python scripts/d9_loose_wire_probe_log.py --seed $0 --slack-cm $1 --tighten-time $2 --out logs/d9_loose_wire/raw 2>&1 | grep "^s[0-9]_"' > logs/d9_loose_wire/run_batch.txt
