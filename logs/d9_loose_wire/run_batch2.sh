#!/bin/bash
# Stiffer anchor (solref 0.005), tight wire at 15 cm with 1 cm overtravel, more slack.
# seeds 0-2 x slack {3,5} cm x tighten time {0.5,2,4} s + never-tighten controls. 2 at a time.
cd /home/chirag/Entanglement-recovery
{
for s in 0 1 2; do
  for k in 3 5; do for T in 0.5 2.0 4.0 0; do echo "$s $k $T"; done; done
done
} | xargs -P 2 -L 1 bash -c 'uv run python scripts/d9_loose_wire_probe_log.py --seed $0 --slack-cm $1 --tighten-time $2 --wire-height 0.15 --anchor-solref 0.005 --overtravel-cm 1 --steps 500 --out logs/d9_loose_wire/raw2 2>&1 | grep "^s[0-9]_"' > logs/d9_loose_wire/run_batch2.txt
