#!/usr/bin/env bash
set -e
cd /home/arc01/mjlab
for s in 0 7; do
  uv run python scripts/d5_9_calf_target_diagnostic.py \
    --seed "$s" \
    --max-recovery-steps-override 48 \
    --no-ablate-hip-wire \
    --out-dir "logs/d5_9_diagnostics/seed${s}" \
    > "logs/d5_9_diagnostics/seed${s}_console.log" 2>&1
  echo "seed $s exit=$?"
done
echo "ALL_DONE"
