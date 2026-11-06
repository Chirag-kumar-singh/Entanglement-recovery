# D5.14 — Recovery Behavior Parity Audit

## Summary of conclusion (read first)

Two separate, independent issues were conflated by comparing the two
viewer visualizations directly:

1. **The `entry_hold` condition is a genuine regression**, confirmed by
   headless, deterministic data (not a visualization artifact): with a
   5-step hold, the FL leg never clears the wire in this seed
   (`crossing_achieved` never becomes True), versus the unmodified
   baseline which does cross and recover. This matches and quantifies
   what D5.13's report already concluded qualitatively.
2. **Both scripts' `--debug` viewer mode has a pre-existing bug
   (present in D5.3 too, not introduced by D5.13) that double-steps the
   environment on every rendered tick**, making the viewer's visual
   behavior diverge from the headless, authoritative rollout in BOTH
   scripts. This means any visual impression of "the D5.3 viewer looked
   like it cleared the wire more often" is not reliable evidence on its
   own — it may reflect the double-step artifact rather than genuine
   per-condition behavior. The headless comparison below is the
   authoritative basis for the "regression" conclusion, independent of
   this bug.

## Task 1 — Verified source differences

### Environment / wire geometry / configuration
Both scripts build via `go2_wire_recovery_residual_env_cfg(...)` with
identical arguments: `affected_leg_idx=0`, `num_envs=1`,
`recovery_mode="hybrid"`, `play=True`,
`wire_cfg=ISOLATED_FL_WIRE_CFG`, `spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y`
(both construct `D3GroundConfig(..., use_isolated_wire=True)`,
confirmed in [scripts/d5_3_extended_recovery_window.py](scripts/d5_3_extended_recovery_window.py#L162) and
[scripts/d5_10_stumble_ablation.py](scripts/d5_10_stumble_ablation.py#L94-L96), the latter reused unchanged
by D5.13). `experimental_peak_triggered_thigh_stop` defaults to
`False` in `D3GroundConfig` and is not overridden by either script.
**No difference.**

### Hip-wire contact ablation and collision settings
D5.3's `build_ablation_env` takes an explicit `ablate_hip_wire: bool`
parameter and, when True, composes a `spec_fn` that adds
`robot/FL_hip <-> wire_seg_*` body-pair excludes (see
[scripts/d5_3_extended_recovery_window.py](scripts/d5_3_extended_recovery_window.py#L119-L135)). D5.13 reuses
`build_d510_env` ([scripts/d5_10_stumble_ablation.py](scripts/d5_10_stumble_ablation.py#L81-L90)), which has **no
`ablate_hip_wire` parameter at all** and never adds any exclude — hip-
wire contact is unconditionally enabled. The reference commands given
use `--no-ablate-hip-wire` for D5.3 (its default), which matches
D5.13's unconditional behavior. **No difference given the specific
commands used**, but D5.13 cannot reproduce D5.3's `--ablate-hip-wire
True` case at all (no such flag exists) — a real asymmetry in
capability, not in the specific comparison requested.

### Recovery cap and FSM behavior
D5.3 exposes `--max-recovery-steps-override` (default 48, passed
through to `max_recovery_steps`); the reference command passes `48`
explicitly. D5.13 hardcodes `max_recovery_steps_override=48` inside
`run()` (via `build_d510_env(d3_cfg, device, max_recovery_steps_override=48)`,
[scripts/d5_13_entry_hold_ablation.py](scripts/d5_13_entry_hold_ablation.py#L225)) with no CLI override. **Both
resolve to 48 for the given commands; no difference in outcome**, but
D5.13 cannot be run with a different cap without editing source.
`_update_fsm()` itself is never touched by either script. **No
difference in FSM logic.**

### Position-actuator disabling and auxiliary actuator commands
D5.3's baseline rollout uses the real, completely unmodified
`_apply_hybrid_actuators`. D5.13's `baseline` condition also leaves it
completely unmodified (`install_entry_hold` is only called when
`condition == "entry_hold"`, [scripts/d5_13_entry_hold_ablation.py](scripts/d5_13_entry_hold_ablation.py#L231-L232)).
**No difference between D5.3 and D5.13's `baseline` condition** —
confirmed empirically below (byte-identical headless trajectories).

D5.13's `entry_hold` condition (see Task 2) is a genuinely different
control behavior, present ONLY when `--condition entry_hold` is passed.

### Recovery-entry timing and entry-hold implementation
Recovery-entry timing (`_update_fsm`'s wire-contact-triggered
transition) is identical in all cases — confirmed empirically below
(`ENTER RECOVERY step=45` in all three headless runs). The entry-hold
implementation itself is described fully in Task 2.

### Reset behavior, simulation stepping, viewer/debug execution
- **`auto_reset`**: D5.3's `build_ablation_env` does not set
  `env_cfg.auto_reset` (leaves the framework default, `True`). D5.13's
  reused `build_d510_env` explicitly sets `env_cfg.auto_reset = False`
  ([scripts/d5_10_stumble_ablation.py](scripts/d5_10_stumble_ablation.py#L99)). This is a **real
  difference**, but it only affects behavior *after* a termination/
  time-out fires (silent auto-reset-and-continue in D5.3 vs. halt in
  D5.13); it does not affect the pre-termination dynamics compared
  below, and neither run in this audit hit a termination within the
  compared window.
- **Simulation stepping in `--debug` mode**: both scripts' `viewer_policy`
  callback internally calls `do_step(i, action)`, which itself calls
  `env.step(action)` — and then **returns `action` to the viewer**,
  which immediately calls `self.env.step(actions)` AGAIN inside
  `NativeMujocoViewer._execute_step` (confirmed by reading
  [src/mjlab/viewer/base.py](src/mjlab/viewer/base.py#L310-L317): `actions = self.policy(obs); self.env.step(actions)`).
  **This means both D5.3 and D5.13's `--debug` mode step the
  environment TWICE per rendered tick** — once inside the script's own
  `do_step`, and once more via the viewer's own post-policy-call
  `env.step`. This bug is identical in both scripts (same `do_step`/
  `viewer_policy` pattern, see [scripts/d5_3_extended_recovery_window.py](scripts/d5_3_extended_recovery_window.py#L297-L301)
  vs. [scripts/d5_13_entry_hold_ablation.py](scripts/d5_13_entry_hold_ablation.py#L316-L319)) and is the direct cause
  of the viewer-vs-headless divergence reported in both D5.11 and the
  prior turn's D5.13 viewer check — **not a difference between the two
  scripts, and not specific to `entry_hold`.**

### Checkpoint path and policy configuration
Both scripts resolve the checkpoint/policy entirely through
`go2_wire_recovery_residual_env_cfg`'s internal defaults — neither
script passes an explicit checkpoint path or overrides policy
construction. **No difference.**

## Task 2 — Does `entry_hold` change anything beyond delaying the handoff?

Verified against [scripts/d5_13_entry_hold_ablation.py](scripts/d5_13_entry_hold_ablation.py#L95-L224)
(`install_entry_hold`): **yes, three concrete behavioral changes, all
already documented and intentional, confirmed again here directly
against source**:

1. The position-actuator gain/bias zeroing and auxiliary-actuator
   gain-enabling that normally happen on `newly_entangled` are deferred
   until `recovery_elapsed >= hold_steps` (a logical delay).
2. During the hold, `apply_actions` is wrapped to forcibly overwrite
   `sim.data.ctrl` for the FL thigh/calf position actuators with the
   FIXED joint position captured at the instant of entry — this is an
   added control signal, not merely "nothing happens for N steps": it
   actively holds the leg in place, counteracting whatever the live
   walking-policy target would otherwise have commanded during the
   hold.
3. The auxiliary thigh-velocity/calf-torque commands are explicitly
   zeroed during the hold (`aux_active = in_recovery & (~holding)`) —
   so no assistance is given toward clearing the wire during the hold
   window.

The peak-detector initialization, FSM entry/exit transition detection,
and the post-hold handoff values (full-magnitude, no ramp) are
unmodified relative to baseline. **No other FSM, threshold, gain, or
geometry change exists.**

## Task 3 — `--help` output (verified, no invented flags)

```
$ uv run python scripts/d5_3_extended_recovery_window.py --help
--seed INT (default: 0)
--device {None}|STR (default: None)
--max-steps INT (default: 200)
--max-recovery-steps-override INT (default: 48)
--ablate-hip-wire, --no-ablate-hip-wire (default: False)
--out-dir STR (default: '')
--debug, --no-debug (default: False)

$ uv run python scripts/d5_13_entry_hold_ablation.py --help
--seed INT (default: 0)
--device {None}|STR (default: None)
--max-steps INT (default: 150)
--condition {baseline,entry_hold} (default: baseline)
--hold-steps INT (default: 5)
--out-dir STR (default: '')
--debug, --no-debug (default: False)
```

D5.13 has **no `--ablate-hip-wire` or `--max-recovery-steps-override`
flag** — these are not configurable from its CLI (hardcoded as
described in Task 1). No flags were invented beyond what `--help`
confirms.

## Task 4 — Controlled headless comparison

Since the specific reference commands both resolve to the same
hip-wire setting (not ablated) and the same recovery cap (48), with
identical checkpoint/wire-geometry/env-construction, a controlled,
headless (non-viewer) comparison was performed using seed 0 for:
D5.3's baseline, D5.13's `baseline` condition, and D5.13's `entry_hold`
condition (`hold_steps=5`).

Commands executed:
```
uv run python scripts/d5_3_extended_recovery_window.py --seed 0 --max-recovery-steps-override 48 --no-ablate-hip-wire --out-dir logs/d5_14_parity_audit/d53_baseline_headless
uv run python scripts/d5_13_entry_hold_ablation.py --seed 0 --max-steps 150 --condition baseline --out-dir logs/d5_14_parity_audit/d513_baseline_headless
uv run python scripts/d5_13_entry_hold_ablation.py --seed 0 --max-steps 150 --condition entry_hold --hold-steps 5 --out-dir logs/d5_14_parity_audit/d513_entry_hold_headless
```

Results (window: recovery entry step 45 through step 100):

| | D5.3 baseline | D5.13 baseline | D5.13 entry_hold(5) |
|---|---|---|---|
| `crossing_achieved` ever True | **Yes**, first at step 84 | **Yes**, first at step 84 | **No**, never |
| max `foot_height_above_wire` | 0.0933 | 0.0933 | 0.0048 |
| max `calf_forward_dist` | 0.2253 | 0.2253 | −0.0032 (never crosses to far side) |
| `base_z` minimum | 0.2040 (step 54) | 0.2040 (step 54) | 0.1824 (step 91) |
| FL ground-contact-lost steps (45-100) | 48/56 | 48/56 | 45/56 |
| Recovery exit reason | crossing+release (success) | crossing+release (success) | cap_timeout (failure) |
| Final phase at end of window | 2 (POST_RECOVERY) | 2 (POST_RECOVERY) | 2 (POST_RECOVERY, but never genuinely "recovered") |

**D5.3's baseline and D5.13's `baseline` condition are byte-identical**
on every measured quantity — confirming the env-construction paths are
behaviorally equivalent for the comparison requested (the `auto_reset`
and extra-sensor differences noted in Task 1 have no effect on this
window). **D5.13's `entry_hold` condition is a genuine, substantial
regression relative to both baselines**: it never achieves
`crossing_achieved`, the FL foot barely rises above the wire (0.0048 m
vs. ≈0.093 m) and never advances past it (`calf_forward_dist` stays
negative), and the run exits via the 48-step timeout rather than a
genuine recovery — not merely a "lower success probability" but a
qualitatively failed crossing attempt in this seed.

Note: the `base_z` minimum is actually slightly *less severe* in
`entry_hold` (0.1824 vs 0.2040) because the hold delays and slightly
dampens the initial sag (as already found in D5.13's original report);
this does not translate into a successful or even improved crossing —
the leg simply never generates enough clearance/forward progress to
reach the far side of the wire within the 48-step budget.

## Task 5 — Viewer reproduction commands and caveat

Exact, verified commands (both previously executed):
```
uv run python scripts/d5_3_extended_recovery_window.py --seed 0 --max-recovery-steps-override 48 --out-dir logs/d5_3_extended_recovery_window/_viz_seed0_baseline --no-ablate-hip-wire --debug
uv run python scripts/d5_13_entry_hold_ablation.py --seed 0 --max-steps 150 --condition entry_hold --hold-steps 5 --out-dir logs/d5_13_entry_hold_ablation/_viz_seed0_hold5 --debug
```

**Both are affected by the double-step bug identified in Task 1** and
the D5.13 `entry_hold` viewer run additionally crashed on exit with
code 139 (segfault) in this environment, consistent with the D5.3
viewer run's own exit code 139 on record in this session. Per the
task's instructions and the established pattern from D5.11/D5.13, the
viewer is **not used as authoritative numerical evidence** here; the
headless comparison in Task 4 is the sole basis for this report's
conclusions. The double-step bug means that *any* prior visual
impression from comparing the D5.3 and D5.13 viewer runs side-by-side
is unreliable on its own and should not be used to judge the relative
severity of the regression — the headless data above should be used
instead.

## Conclusion

The apparent regression ("D5.13's `entry_hold` visualization fails to
get the FL leg over the wire") **is real and is attributable to the
5-step hold itself**, not to any configuration mismatch between the two
scripts — D5.3's baseline and D5.13's `baseline` condition were
confirmed byte-identical in a controlled headless comparison using
matched seed, checkpoint, wire geometry, hip-wire setting, and recovery
cap. The entry-hold mechanism, while successfully preventing the
initial base-height sag during the hold window itself (as D5.13 already
reported), delays the handoff to the auxiliary commands just long
enough that, in this seed, the leg no longer has sufficient time/
momentum within the 48-step recovery budget to clear and cross the
wire — a genuine, quantified failure mode, not a visualization or
configuration artifact.

Separately (and independently of the above conclusion), the viewer
(`--debug`) mode in BOTH D5.3 and D5.13 double-steps the environment on
every rendered tick due to a pattern bug in how `viewer_policy` is
implemented in these diagnostic scripts (calling `env.step()` directly
inside the policy callback, which the viewer then steps again). This
explains the previously observed viewer-vs-headless numerical
divergence in both scripts and means neither script's `--debug` output
should be treated as authoritative. This bug pre-dates D5.13 (present
in D5.3 since at least this session) and is reported here as a
diagnostic-tooling issue, not a production controller issue; per this
task's scope, it is not fixed here.

No production controller code, actuator gains, torques, velocity
commands, FSM thresholds, rewards, or the frozen checkpoint were
modified. No new recovery strategy, smooth blending, velocity gating, or
additional hold variant was implemented. This comparison stops here.
