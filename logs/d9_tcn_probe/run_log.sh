#!/bin/bash
# One process per seed (env build leaks memory). Wire runs, then no-contact controls.
cd /home/chirag/Entanglement-recovery
for s in 0 1 2 3 4 5 6 7; do
  uv run python scripts/d9_tcn_probe_log.py --seed $s --out logs/d9_tcn_probe/raw 2>&1 | grep "^seed"
done
for s in 0 1; do
  uv run python scripts/d9_tcn_probe_log.py --seed $s --wire-height 0.5 --out logs/d9_tcn_probe/raw 2>&1 | grep "^seed"
done
