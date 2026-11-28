#!/bin/bash
# Step 1: find an anchor setting that tightens the wire at 10 cm with NO robot/wire overlap.
# seed 0, tighten 0.5 s (fastest = worst case), slack {2,3} cm x anchor solref {0.005,0.01,0.02}
# x overtravel {0,1} cm, plus slack-0 controls (taut from the start) with the stiff anchor.
cd /home/chirag/Entanglement-recovery
{
for k in 2 3; do for sr in 0.005 0.01 0.02; do for ot in 0 1; do echo "$k $sr $ot"; done; done; done
for ot in 0 1; do echo "0 0.005 $ot"; done
} | xargs -P 2 -L 1 bash -c 'uv run python scripts/d9_loose_wire_probe_log.py --seed 0 --slack-cm $0 --anchor-solref $1 --overtravel-cm $2 --tighten-time 0.5 --wire-height 0.10 --steps 500 --out logs/d9_loose_wire/raw3 2>&1 | grep "^s[0-9]_"' > logs/d9_loose_wire/run_batch3.txt
