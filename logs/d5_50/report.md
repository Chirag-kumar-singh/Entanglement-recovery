# D5.50 — First Calf–Wire Contact Counterfactual

## Status

Stopped after Part A feasibility validation.

No controlled baseline-vs-counterfactual comparison was run, because the exact first-contact pair could be identified but could not be suppressed cleanly in the current runtime without introducing a contact-scope confound or breaking the normal recovery trigger.

## Verified baseline facts

- Seed: 0
- First raw FL calf–wire contact step: 43
- First raw contact pair: `wire_seg_2_geom <-> robot/FL_calf2_collision`
- First-contact pair force norm (raw MuJoCo contact force): 130.4653 N
- Baseline RECOVERY entry step: 44
- D5.49 shared restore snapshot corresponds to the post-step-43 state, immediately before the baseline's step-44 recovery entry.

## Trigger dependency

Inspected source confirms that RECOVERY entry is driven directly by the `wire_contact` sensor in `ResidualRecoveryAction._update_fsm`.

- `wire_contact` sensor primary match: `wire.geom_names()`
- `wire_contact` sensor secondary match: `FL_calf2_collision`
- `entering_recovery = is_normal & in_contact`

Therefore, removing the exact first calf–wire contact also removes the normal RECOVERY trigger unless entry is explicitly replayed as a diagnostic intervention.

## Why the obvious D4.8-style exclude is not valid here

The D4.8 mechanism uses `MjSpec.add_exclude(bodyname1=..., bodyname2=...)`, which is body-pair scoped, not geom-pair scoped.

That is acceptable for the FL hip diagnostic because the relevant hip body has one active collision geom for the wire interaction.
It is not cleanly valid for the FL calf onset pair because the `robot/FL_calf` body contains two active collision geoms:

- `FL_calf1_collision`
- `FL_calf2_collision`

So excluding `robot/FL_calf <-> wire_seg_2` would disable both calf geoms against that segment, not only the first-contact pair `wire_seg_2_geom <-> robot/FL_calf2_collision`.
That changes unrelated potential contacts and violates the task's requirement for a pair-specific intervention.

## Runtime pair-filter probe

I probed MuJoCo's Python callback API and confirmed that `mujoco.set_mjcb_contactfilter` exists.
However, in this environment's `mujoco_warp` stepping/contact pipeline, installing a filter callback that returns `1` for the exact onset pair did not remove that pair from the live contact set:

- Before filter: target pair present = `True`
- After filter + `forward()` + `scene.update()` + `sense()`: target pair present = `True`

So the available Python callback path is not an effective geom-pair suppression mechanism for this runtime.

## Feasibility decision

A valid D5.50 comparison would require all of the following:

- the same restored post-step-43 simulator state,
- a change affecting only `wire_seg_2_geom <-> robot/FL_calf2_collision`,
- comparable recovery activation timing,
- no unintended suppression of sibling calf or later wire contacts.

I could satisfy at most two of those at once:

- body-pair exclude preserves deterministic setup but is too broad,
- explicit trigger replay could preserve timing but does not fix the contact-scope confound,
- the Python geom-pair callback is not effective in the current `mujoco_warp` runtime.

## Evidence vs interpretation

### Evidence

- The exact first-contact pair is identified and measured: `wire_seg_2_geom <-> robot/FL_calf2_collision` at step 43 with force norm 130.4653 N.
- The normal trigger is coupled to `FL_calf2_collision` wire contact.
- `robot/FL_calf` contains both `FL_calf1_collision` and `FL_calf2_collision`, so body-level exclusion is over-broad.
- The available Python geom-pair contact-filter callback does not suppress the target pair in this runtime.

### Interpretation

Because no clean pair-specific contact intervention is available here, any D5.50 counterfactual run would either:

- alter additional contacts beyond the intended onset pair, or
- require a trigger replay layered on top of a broadened contact change.

That would make the comparison invalid for the question the task asks.

## Conclusion

- The comparison is inconclusive or invalid because of trigger, contact, or state-restoration confounds.

## Recommended next step

If this question still matters, the next valid path is infrastructure work rather than another seed-0 comparison:

- either run the diagnostic in a runtime that honors true geom-pair contact filtering,
- or add an engine-level pair-specific exclusion path for `mujoco_warp` before re-attempting D5.50.
