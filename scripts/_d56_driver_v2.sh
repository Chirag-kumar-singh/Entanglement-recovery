#!/usr/bin/env bash
set -e
cd /home/arc01/mjlab
# D5.6 follow-up: 3 never-latch seeds + 3 success seeds, same config as D5.5.
for s in 7 10 13 0 3 8; do
  uv run python scripts/d5_6_base_pose_diagnostic.py \
    --seed "$s" \
    --max-recovery-steps-override 48 \
    --no-ablate-hip-wire \
    --out-dir "logs/d5_6_diagnostics_v2/seed${s}" \
    > "logs/d5_6_diagnostics_v2/seed${s}_console.log" 2>&1
  echo "seed $s exit=$?"
done
echo "ALL_DONE"
