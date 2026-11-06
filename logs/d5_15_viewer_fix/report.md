# D5.15 — Fix Diagnostic Viewer Double-Stepping

## 1. Bug recap (established in D5.14)

`NativeMujocoViewer._execute_step` (`src/mjlab/viewer/base.py`) calls the
policy callback once per rendered tick and steps the environment exactly
once with whatever action the callback returns:

```python
obs = self.env.get_observations()
actions = self.policy(obs)
self.env.step(actions)
```

Both `scripts/d5_3_extended_recovery_window.py` and
`scripts/d5_13_entry_hold_ablation.py` passed a `viewer_policy` callback
that itself called a `do_step(i, action)` helper — which internally called
`env.step(action)` — and then returned that same `action` back to the
viewer, which stepped the environment a second time. Net effect: every
rendered tick advanced the simulation by **two** environment steps instead
of one, which explained the viewer-vs-headless numerical divergence seen in
D5.3/D5.11/D5.13.

## 2. Files changed and why

- [scripts/d5_3_extended_recovery_window.py](scripts/d5_3_extended_recovery_window.py)
- [scripts/d5_13_entry_hold_ablation.py](scripts/d5_13_entry_hold_ablation.py)

In both files the previous monolithic `do_step(step, action)` (which called
`env.step(action)` internally and then built the logged row) was split into:

- `do_step_pre(step) -> prev_phase`: pre-step bookkeeping that must read
  state from *before* this step's `env.step()` call (e.g.
  `newly_entangled`/`enter_line` detection).
- `do_step_post(step, prev_phase) -> row`: all post-step bookkeeping and the
  row dict, run *after* `env.step()` for that step has executed. Content and
  semantics are unchanged from the original inline body.
- `do_step(step, action)`: thin wrapper used only by the **headless** path —
  `do_step_pre` → `env.step(action)` (exactly once) → `do_step_post`. This
  preserves the headless API and behavior exactly.

The viewer's `viewer_policy(_obs)` callback was rewritten as a stateful
closure (`pending = {"step": None, "prev_phase": None}`). On each call it:

1. If a previous step is pending, finalizes it by calling `do_step_post`
   for that step (using the env state the viewer's own prior `env.step()`
   call already produced), appends/prints the row, and raises
   `SystemExit(0)` if that was the last step (D5.13 also preserves the
   original `env.reset_buf`/`env.reset_terminated` check here).
2. Computes the next step index, calls `do_step_pre` for it, stores it as
   `pending`, and returns a **zero-action tensor** for the viewer to apply.

The callback itself never calls `env.step()`. The viewer performs exactly
one `env.step()` call per tick, as its contract requires.

`get_errors` was run on both files after the change: no errors.

## 3. Headless before/after comparison

Reference commands (unchanged from D5.14):

```
uv run python scripts/d5_3_extended_recovery_window.py --seed 0 \
  --max-recovery-steps-override 48 --no-ablate-hip-wire \
  --out-dir logs/d5_15_viewer_fix/d53_baseline_headless_AFTER

uv run python scripts/d5_13_entry_hold_ablation.py --seed 0 --max-steps 150 \
  --condition baseline \
  --out-dir logs/d5_15_viewer_fix/d513_baseline_headless_AFTER
```

- `d53_baseline_headless_BEFORE/` = copy of pre-fix
  `logs/d5_14_parity_audit/d53_baseline_headless/`.
- `d53_baseline_headless_AFTER/` = freshly generated with the fixed script,
  same seed/config.
- `diff trajectory.csv` and `diff run_log.txt` between BEFORE/AFTER: **no
  differences ("IDENTICAL")**.
- Same procedure and result for D5.13's `baseline` condition
  (`d513_baseline_headless_BEFORE/` vs `d513_baseline_headless_AFTER/`):
  **IDENTICAL**.

Recovery metrics (unchanged, both scripts, both before/after):

```
ENTER RECOVERY step=45
EXIT RECOVERY step=88 elapsed=45 timeout=False
stage1_first_latch_step=84
stage2_first_latch_step=84
```

`--help` output for both scripts was re-checked post-fix and is unchanged
(no CLI flags added, removed, or renamed).

**Conclusion: the refactor has zero effect on the headless execution path.**
No controller, actuator, gain, wire-geometry, checkpoint, or stepping-rate
behavior was touched for headless runs.

## 4. Viewer validation (Task 6)

### D5.3 corrected viewer

Exact command run:

```
uv run python scripts/d5_3_extended_recovery_window.py --seed 0 \
  --max-recovery-steps-override 48 \
  --out-dir logs/d5_15_viewer_fix/d53_viewer_AFTER \
  --no-ablate-hip-wire --debug
```

The run printed `[D5.3] step=...` lines up through step 199 (200 steps
total), then:

```
[D5.3] wrote logs/d5_15_viewer_fix/d53_viewer_AFTER/trajectory.csv
  ENTER RECOVERY step=45
  EXIT RECOVERY step=88 elapsed=45 timeout=False
  stage1_first_latch_step=84
  stage2_first_latch_step=84
malloc(): unsorted double linked list corrupted
```

Exit code: **134 (SIGABRT)** — see crash notes below.

**Stepping-rate confirmation:** `logs/d5_15_viewer_fix/d53_viewer_AFTER/trajectory.csv`
was written in full (201 rows, matching the expected 200 steps + header)
before the crash. It is **byte-for-byte identical**
(`diff` → no output) to `logs/d5_15_viewer_fix/d53_baseline_headless_AFTER/trajectory.csv`,
and the printed `ENTER RECOVERY`/`EXIT RECOVERY`/latch-step summary matches
the headless run exactly. This is the strongest possible confirmation that
the viewer now advances exactly one environment step per tick — previously
(pre-fix, per D5.14) viewer and headless trajectories for the same
condition/seed diverged numerically; now they match exactly, step for step.

### D5.13 viewer (optional tooling check, per task instructions — not used as
a recovery baseline)

`--help` was checked first (unchanged). Command run:

```
uv run python scripts/d5_13_entry_hold_ablation.py --seed 0 --max-steps 150 \
  --condition baseline --out-dir logs/d5_15_viewer_fix/d513_viewer_AFTER --debug
```

The run printed step logs through step 149 (150 steps total), wrote
`trajectory.csv`, and printed the same `ENTER RECOVERY`/`EXIT RECOVERY`/latch
summary as the headless `baseline` run, then hit a different teardown
failure (GLX `X_GLXSwapBuffers` / `GLXBadDrawable` error on window close,
process then hung until an externally applied timeout, exit code 124).

`logs/d5_15_viewer_fix/d513_viewer_AFTER/trajectory.csv` (151 rows) is
**byte-for-byte identical** to
`logs/d5_15_viewer_fix/d513_baseline_headless_AFTER/trajectory.csv`. Same
conclusion as D5.3: stepping rate is now correct.

## 5. Teardown crashes (reported separately, per task instructions — NOT fixed)

Two distinct teardown failures were observed, both occurring **only after**
all simulation steps completed and the CSV/run_log were fully written:

- D5.3 viewer: exits with `malloc(): unsorted double linked list corrupted`
  and process exit code 134 (SIGABRT).
- D5.13 viewer: window-close triggers an X11/GLX `GLXBadDrawable` error on
  `X_GLXSwapBuffers`, and the process does not exit on its own (had to be
  killed via `timeout`, observed exit code 124).

Both are consistent with the previously documented pattern (D5.11/D5.13/D5.14)
that `NativeMujocoViewer`'s native window/GL teardown is unreliable in this
environment. Neither crash occurs until after the script's own Python-level
logic (stepping, logging, CSV writing) has completed successfully. These are
pre-existing, native/display teardown issues, separate from the
double-stepping bug this task targeted, and were **not modified or fixed** —
per the task's explicit scope, only the stepping-rate bug was in scope.

## 6. Explicit confirmation: no production controller behavior changed

- No change to `_apply_hybrid_actuators`, `_update_fsm`, actuator gains, or
  any FSM thresholds.
- No change to wire geometry (`ISOLATED_FL_WIRE_CFG`,
  `ISOLATED_FL_WIRE_SPAWN_Y`), checkpoint path, or `max_recovery_steps`.
- No change to `env.cfg.auto_reset`, `env.reset_buf`/`reset_terminated`
  handling semantics (D5.13's viewer path preserves the original
  termination-detection logic verbatim).
- No blending/hold/gating logic added; `install_entry_hold` and its
  `hold_state` monkeypatch machinery (D5.13) are untouched.
- No policy/checkpoint retraining; no quad-sdk changes.
- Evidence: byte-identical headless CSV/run_log before vs. after the fix
  (Section 3), and byte-identical viewer-vs-headless CSVs after the fix
  (Section 4) — the only change in behavior is that the viewer's trajectory
  now *matches* the headless trajectory instead of diverging from it.

## 7. Scope / stopping point

This task fixed only the viewer double-stepping bug in the two named
scripts. No new recovery-controller experiment (e.g. further sag
mitigation, blending designs referenced in D5.13's report) was started, per
the task's explicit instruction to stop after this fix and its validation.
