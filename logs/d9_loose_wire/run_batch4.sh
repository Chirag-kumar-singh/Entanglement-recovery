#!/bin/bash
# Step 1 confirmation: two overlap-free candidates from run_batch3 on seeds 1-2 (T=0.5 s, worst case).
#   A: solref 0.01, overtravel 1 cm    B: solref 0.005, overtravel 0
cd /home/chirag/Entanglement-recovery
{
for s in 1 2; do for k in 2 3; do echo "$s $k 0.01 1"; echo "$s $k 0.005 0"; done; done
} | xargs -P 2 -L 1 bash -c 'uv run python scripts/d9_loose_wire_probe_log.py --seed $0 --slack-cm $1 --anchor-solref $2 --overtravel-cm $3 --tighten-time 0.5 --wire-height 0.10 --steps 500 --out logs/d9_loose_wire/raw3 2>&1 | grep "^s[0-9]_"' > logs/d9_loose_wire/run_batch4.txt
