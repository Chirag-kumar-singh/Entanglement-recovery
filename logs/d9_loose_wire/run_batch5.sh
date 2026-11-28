#!/bin/bash
# Step 3: TCN probe on the clean 10 cm loose-to-tight wire (anchor solref 0.005, overtravel 0).
# seeds 0-2 x slack {2,3} x tighten {2,4,0(never)}; the T=0.5 runs were copied from raw3.
cd /home/chirag/Entanglement-recovery
{
for s in 0 1 2; do for k in 2 3; do for T in 2.0 4.0 0; do echo "$s $k $T"; done; done; done
} | xargs -P 2 -L 1 bash -c 'uv run python scripts/d9_loose_wire_probe_log.py --seed $0 --slack-cm $1 --tighten-time $2 --anchor-solref 0.005 --overtravel-cm 0 --wire-height 0.10 --steps 500 --out logs/d9_loose_wire/raw5 2>&1 | grep "^s[0-9]_"' > logs/d9_loose_wire/run_batch5.txt
