# Handoff: where the project stands and what to do next

Read this first on a new machine. Written 2026-11-28.

## Project in one paragraph

Unitree Go2 walks (frozen RL walking policy) into a wire that entangles its legs; a scripted
per-leg recovery FSM (`ResidualRecoveryAction` + `MultiLegRecoveryAction`) steps the legs
over it. Entanglement recovery itself is paused at about 75% (best four-leg setup "D7.9":
30/40 on seeds 10-49, 125/160 on seeds 0-159). The current work (D9) replaces the
simulation's oracle trigger, the wire contact sensor, with a proprioception-based detector
trained in simulation. All results and exact commands are in
`D6_MULTI_LEG_RECOVERY_SUMMARY.md` (read the D9 sections at the end).

## Best baseline (always compare against this)

D7.9 recipe, original wire at 10 cm, 4 legs:

    --pair all --start-x-range -0.485 -0.35 --max-attempts FL 2 FR 2 RL 2 RR 2
    --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04
    --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2
    --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30
    --support-lift-trigger FL FR

Stored results: `logs/d7_9_combined/four_0_39.txt`, `four_40_79.txt`. Seeds 0-159 and 10-49
are the tuning and evaluation seeds; detector data uses 200-237, 300-479 and 500-599.

## D9 results so far (seeds 10-49, closed loop)

| Trigger | PASS | FAIL_FRONT | FAIL_REAR |
|---|---|---|---|
| Oracle contact sensor (D7.9) | 30/40 | 2 | 8 |
| CNN model_v1 (trained on walking data) | 19/40 | 3 | 18 |
| CNN model_v2 (v1 + 100 closed-loop episodes) | 24/40 | 3 | 13 |

- Detector delay matters: a late trigger costs seeds (100 ms delay: 16/40 vs 30/40).
- The shipped TCN from `go2-entanglement-tcn` is too slow zero-shot on sim logs.
- model_v2 misses nothing and is on time; the remaining gap is early (30) and false (10)
  rear-leg alarms. Cause is hypothesised (foot touches the wire before the labelled calf2),
  not proven.
- BUG FIXED, do not repeat: detector legs are ordered FR FL RR RL, the controller's are
  FL FR RL RR. Always map by leg name (done in `scripts/d9_detector_closed_loop.py`).
  Anything under `logs/d9_detector/closed_loop/invalid_swapped_legs/` used the wrong order.

## What is in the repo

- `src/mjlab/tasks/recovery/mdp/actions.py`: two opt-in additions, `entry_delay_steps` and
  `external_trigger`. Defaults reproduce the original behaviour exactly.
- `scripts/d6_1_front_pair_wire.py`: the episode runner (`--seed N --debug` opens a viewer).
- `scripts/d9_*`: the D9 tooling.
  - `d9_loose_wire_*`: loose-to-tight wire and overlap monitor.
  - `d9_tcn_probe_*`: runs the external TCN on sim logs.
  - `d9_detector_data.py`: records features and oracle labels per episode.
  - `d9_detector_train.py`: trains the causal CNN and evaluates it.
  - `d9_detector_closed_loop.py`: runs a model as the recovery trigger.
- `logs/d9_detector/`: datasets (`data_v0`, `data_v1`, `data_v2cl`), models
  (`model_v1.pt`, `model_v2.pt`), run scripts (`run_data_v1.sh`, `run_closed_loop.sh`),
  results. Regenerable scratch and raw dumps are git-ignored.
- `logs/rsl_rl/go2_velocity/2026-09-17_13-00-16/model_9900.pt`: the frozen walking policy.

## Not in the repo (copy it over by hand)

`/home/chirag/Downloads/go2-entanglement-tcn-main` (the TCN detector project).
`scripts/d9_tcn_probe_eval.py` imports it from a hardcoded path (`PKG`); edit that path.
Only that probe script needs it. The sim-trained detector does not.

## Setup on the new machine

    git clone git@github.com:Chirag-kumar-singh/Entanglement-recovery.git
    cd Entanglement-recovery && uv sync
    uv run python scripts/d6_1_front_pair_wire.py --pair all --seed 10 <D7.9 flags>   # sanity

Check one oracle seed against `logs/d7_9_combined/four_0_39.txt` before trusting anything:
physics should match, but confirm on the new GPU/CPU.

## Working rules (carried over from earlier sessions)

- Always `uv run`, never bare python. Run `make format` / `make type` per CLAUDE.md. Note
  `make check` runs `ruff check --fix` over the whole repo and rewrites old files; check
  `git status` afterwards and revert unrelated changes.
- Memory: one recovery env peaks about 1.7 GB RSS and an in-process seed sweep leaks. Use one
  fresh process per seed and at most 3 concurrent sim processes (a 15 GB laptop crashed at 7).
- Compare every new idea against the best verified baseline above, on the same seeds, and put
  the exact command next to each result in `D6_MULTI_LEG_RECOVERY_SUMMARY.md`.
- From papers and Quad-SDK borrow control logic (structure, triggers, ordering), never their
  numeric values; the robots differ. Tune on our own sweeps.
- Everything new must be opt-in and leave existing behaviour untouched. Start experiments
  with small seed counts (10, then 40) before 160.

## Next steps (in priority order)

1. Another DAgger round: collect with model_v2 as trigger on new seeds (600-699), retrain on
   v1 + all valid closed-loop data, re-test seeds 10-49. Each round gained about 5 seeds.
2. Rear diagnostic: detector on FR/FL, oracle on RR/RL (flags `--act-legs FR FL`) to see how
   much of the 6-seed gap is the rear legs. `--no-early` is the other diagnostic.
3. Try a rear-leg alarm rule (2 consecutive steps, or a higher threshold).
4. Consider adding calf1 and foot to the oracle label, since early alarms look like foot
   contacts the label does not cover.
5. Open: wire pass-through into calf1 on some episodes; vary walking commands in the data;
   final check of the winning detector on seeds 0-159 (not used for any selection).
