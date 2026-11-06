# D5.46 — Forward-Bias Activation-State Audit

## Verdict
The forward-bias gate activates at env step 78 for seed 0 and 80 for seed 11. Seed 0 keeps converting that bias into a crossing; seed 11 does not, because its forward progress stalls almost immediately after activation and the calf never reaches positive signed forward distance.

## Measured activation window
- Seed 0: first active at env step 78 (recovery_elapsed 35), active for 11 recovery steps through env step 88.
- Seed 11: first active at env step 80 (recovery_elapsed 30), active for 18 recovery steps through env step 97.
- The actuator command sign is unchanged: the forward-bias phase still uses a negative thigh velocity target, just with smaller magnitude than the full sign reversal.

## What the activation window shows
- Seed 0 enters the bias window with the calf still behind the wire but with a strongly positive body forward velocity. The calf continues advancing and crosses a few steps later.
- Seed 11 enters the bias window from a less favorable state: the calf is still behind the wire, but the body forward velocity is already negative and the calf-forward-distance improvement stalls almost immediately.
- D5.45 does not log base vertical velocity, so the post-activation vertical comparison cannot be made from this experiment alone. The closest vertical evidence remains D5.44, where seed 11 entered recovery with a much more negative vertical velocity than seed 0.

## Earliest loss of useful forward progress
- Seed 11’s best calf-forward distance after activation occurs one step after the switch (env step 81). After that point, calf-forward distance reverses and drifts back toward the wire instead of moving decisively forward.
- In practical terms, the useful forward-progress window ends within one to two steps of activation: the bias changes the shape of the motion, but not the sign of the net outcome.
- Seed 0, by contrast, keeps converting the same bias into a positive excursion and later release.

## Supported trigger hypothesis
The combined measurements from D5.43 and D5.45 support a more restrictive state-dependent trigger than the current one: require sustained contact and a non-negative base forward velocity before enabling the forward-bias phase. That fits the observed split better than a magnitude-only change, because seed 0 activates from a forward-moving, contact-sustained state while seed 11’s contact history is fragmented near entry and its body forward velocity is already negative by the time the forward bias turns on. The data do not support a stronger claim than that without a new log that includes base vertical velocity in the D5.45-style experiment.

## Files written
- [logs/d5_46/activation_alignment.csv](logs/d5_46/activation_alignment.csv)
- [logs/d5_46/report.md](logs/d5_46/report.md)
