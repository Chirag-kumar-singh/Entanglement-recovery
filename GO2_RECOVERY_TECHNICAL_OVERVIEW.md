# Go2 wire-entanglement recovery: technical overview

Status as of 2026-11-26. Simulation only (mjlab / MuJoCo). All numbers come from the
experiment log in `D6_MULTI_LEG_RECOVERY_SUMMARY.md` (sections D6 and D7.x), where each
result sits next to the exact command that produced it.

---

## 0. The 5-minute version

1. **Problem.** A Unitree Go2 walks forward into a thin wire. A foot or shin gets
   hooked, the gait stalls, and the robot trips. We want it to get over the wire and keep
   walking, with the walking policy itself left untouched.
2. **Approach.** Keep the learned walking policy frozen. Add a small state machine per leg:
   when the wire touches a leg, replace that leg's joint targets with a scripted foot path
   (back and up, forward over the wire, down), then hand the leg back to the policy.
3. **Where we started.** One leg at a time: about 128/136 seeds pass. Two legs and four
   legs were much worse (front pair 36%, four legs 29-39% on varied approaches).
4. **What moved the numbers.** Timing rules for when each step starts, retries, and
   *holding the body up on the legs that are not stepping* (the "support lift").
5. **Where we are.** Front pair 74/80 (92%). Four legs 60/80 (75%) on the tuning seeds and
   65/80 (81%) on fresh seeds never used for tuning, so 125/160 (78%) overall.
6. **What did not work, and why that matters.** Five different rear-leg fixes all failed on
   fresh seeds. The rear stage is sensitive, not fixable by one more timing flag.
7. **Honest limits.** Contact detection uses the simulator's contact sensor (not
   proprioception as in the paper), pass/fail is geometric, and the stance with the lift is
   still a little stiffer and raised compared with normal standing.

---

## 1. The problem and the success test

- **Robot:** Unitree Go2, 12 actuated joints (hip ab/ad, thigh, calf per leg), 50 Hz control
  (0.02 s per policy step).
- **Wire:** a thin wire across the robot's path. Each leg has its own contact sensor
  against the wire.
- **Start condition ("varied approaches"):** the start position along the approach is
  `x = lo + (hi - lo) * frac(seed * 0.618...)` with `(lo, hi) = (-0.485, -0.35)` m
  (`--start-x-range`), a golden-ratio spread over about one gait cycle. The gait phase when
  the wire is hit therefore differs between seeds. A seed changes the start position only,
  not the robot or the wire.
- **Pass (geometric):** every tested leg ends across the wire, no fall (base height
  and roll stay sane), the robot is stable and walks on. A scripted step counts as a real
  lift when the foot peaks at about 0.15 m or higher.
- **Verdicts:** `PASS`, `FAIL_FRONT` (front stage fails, rear legs never reach the wire),
  `FAIL_REAR`.

Pass/fail does not measure how smooth the crossing looks. That is why stability metrics
(roll, pitch, angular velocity, body height) were added in D7.5 onwards.

---

## 2. System architecture

```
 obs (48) ──► frozen walking policy π ──► 12 joint targets ──┐
                                                             ▼
        wire contact sensors ──► per-leg FSM ──► per-leg override ──► PD joint targets ──► MuJoCo
                                     ▲                  ▲
                      multi-leg wrapper: serialization, gates, support lift
```

**Frozen walking policy.** An RL policy (PPO, checkpoint `model_9900.pt`), never retrained
here. Input is 48 numbers:

$$
o_t=[\,v_b\ (3),\ \omega_b\ (3),\ g_b\ (3),\ q-q_0\ (12),\ \dot q\ (12),\ a_{t-1}\ (12),\ c\ (3)\,]
\in\mathbb{R}^{48}
$$

(base linear and angular velocity, gravity in the body frame, joint position relative to
the default, joint velocity, previous action, velocity command). Output is a 12-vector of
normalised joint actions turned into position targets by

$$
q^{*}_j = q_{0,j} + s_j\,a_j
$$

with default pose $q_0$ (thigh 0.9 rad, calf −1.8 rad) and per-joint action scale $s_j$.

**Per-leg recovery controller** (`ResidualRecoveryAction`, `mdp/actions.py`), one per leg,
in three phases: `NORMAL_WALK`, `RECOVERY`, `POST_RECOVERY`.

- *Enter* `RECOVERY` when that leg's wire-contact sensor fires.
- In `RECOVERY` the leg's thigh and calf targets are replaced by the scripted step
  (section 3). The ab/ad joint and the other three legs keep following the policy.
- *Exit* when the crossing is achieved (calf cleared the wire and foot is past it), or a
  step cap (`--max-recovery-steps 96`) is hit.

**Multi-leg wrapper** (`MultiLegRecoveryAction`, `mdp/multi_leg.py`): runs one controller
per leg and adds the cross-leg logic: serialization, start gates and the support lift.

---

## 3. The scripted step: kinematics and trajectory

Each leg is modelled as a two-link planar chain in its sagittal plane (origin at the thigh
joint, $x$ forward, $z$ up) with thigh angle $\theta$, calf angle $\varphi$ and
link lengths $l_1=l_2=0.213$ m.

**Forward kinematics** (foot position):

$$
x=-\big(l_1\sin\theta+l_2\sin(\theta+\varphi)\big),\qquad
z=-\big(l_1\cos\theta+l_2\cos(\theta+\varphi)\big)
$$

**Inverse kinematics** (knee-back branch, $\varphi<0$). With $d=\lVert p\rVert$ clamped to
a reachable range:

$$
\cos\varphi=\frac{d^2-l_1^2-l_2^2}{2\,l_1 l_2},\qquad
\varphi=-\arccos(\cdot),\qquad
\alpha=\operatorname{atan2}(-x,-z),\qquad
\theta=\alpha-\tfrac{\varphi}{2}
$$

(for equal link lengths). Results are clamped to the joint limits.

**Foot path.** From the foot position $p_0=(x_0,z_0)$ at the moment the step starts:

$$
p_1=(x_0-b,\ z_0+h),\quad p_2=(x_0+A,\ z_0+h),\quad p_3=(x_0+A,\ z_{\text{stand}})
$$

with retraction $b=0.12$ m, lift height $h=0.20$ m, advance $A=0.30$ m (0.18 m for FR, which
cut falls from 12 to 2 over 60 seeds). Three segments run for $n_1=4$, $n_2=5$, $n_3=4$
policy steps, each with a cosine ease so velocity is zero at the ends:

$$
s(u)=\tfrac12\big(1-\cos(\pi\,\mathrm{clip}(u,0,1))\big),\qquad
p(k)=p_0+s_1(p_1-p_0)+s_2(p_2-p_1)+s_3(p_3-p_2)
$$

with $s_i=s\big((k+1-\textstyle\sum_{j<i}n_j)/n_i\big)$. After the path the leg is blended
back to the policy over 6 steps. The 13-step swing is $n_1+n_2+n_3$.

Idea in words: pull the foot back so it unhooks, raise it above the wire, reach forward
past it, and put it down.

---

## 4. When does each step start? (the part that took most of the work)

The step path is fixed. What changes the pass rate is the **start rule**, because the
walking policy is still running on the other legs and the gait phase matters.

| Rule | What it does | Flag |
|---|---|---|
| Touchdown window | start when the leg's own foot has been on the ground 17-19 steps | `--td-window 17 19` |
| Gate timeout | start anyway after this many steps in recovery if the gate never opens | `--step-gate-timeout 30` (was 60) |
| Serialization | a leg waits while another leg is mid-step, plus a gap after it lands | `--serialize-gap-steps 12` |
| Support gate | wait while another support foot is in the air (height above 0.04 m) | `--support-gate-z 0.04` |
| Gait gate (front) | start only if the two support feet landed within 6 steps | `--step-gate FL 6 FR 6` |
| Retry | up to 2 attempts per leg if the leg is not across | `--max-attempts FL 2 ...` |
| Crossing fallback | latch the crossing when the foot has been past the wire for 10 steps with no contact | `--crossing-fallback 10` |

Crossing is a two-stage latch: calf clearance (calf on the far side and lifted above the
wire) and foot crossed (foot at least 0.02 m past the wire). The fallback covers seeds where
the leg crosses visually but the calf never rises above the wire.

---

## 5. The support lift (main contribution of D7)

**Observation.** When one leg steps, the three standing legs carry the robot and the frozen
policy lets the body sag to 0.14-0.18 m (standing height is 0.308 m). Sag is a policy effect,
not a wire effect: a forced step with no wire at all gave 0.205 m, with the wire 0.208 m,
walking only 0.307 m.

**Fix.** Add a small joint offset to the standing legs while a trigger leg is stepping.
For leg $k$ not stepping:

$$
\Delta q_{k,\text{thigh}}=\lambda_k(t)\,\delta_{\text{thigh}},\qquad
\Delta q_{k,\text{calf}}=\lambda_k(t)\,\delta_{\text{calf}}
$$

with $\delta_{\text{thigh}}=-0.2$ rad, $\delta_{\text{calf}}=+0.3$ rad (the "half lift"; the
"full lift" is −0.4 / +0.6). The level $\lambda_k\in[0,1]$ ramps toward 1 while any trigger
leg is busy and toward 0 otherwise, at most $1/R$ per step with ramp length $R=12$:

$$
\lambda_k \leftarrow \lambda_k+\operatorname{clip}\!\big(\mathbb 1[\text{trigger leg busy}\wedge k\text{ not busy}]-\lambda_k,\ -\tfrac1R,\ \tfrac1R\big)
$$

In action space this is $a_j\mathrel{+}=\lambda_k\,\delta_j/s_j$, so the physical target
shift is $\lambda_k\,\delta_j$ radians. **The trigger is the front legs only**
(`--support-lift-trigger FL FR`): lifting during rear steps hurt the rear stage.

**Size of the effect (from the FK above, default pose $\theta=0.9,\ \varphi=-1.8$).**
Foot depth below the hip is $0.265$ m. With the half lift ($\theta=0.7,\ \varphi=-1.5$) it is
$0.311$ m (+4.7 cm); with the full lift ($\theta=0.5,\ \varphi=-1.2$) it is $0.350$ m (+8.5 cm).
That is the most the legs could raise the body if they held the pose. Measured (four legs,
passing runs): the body only rises about 1.3 cm on average (0.315 m against 0.302 m with no
lift), because the policy partly resists.

**Why the ramp mattered.** A 3-step ramp launched the body up (0.30 to 0.38 m in 12 steps in
one seed) and looked like a jump. Using 12 steps cut peak roll and angular velocity back
close to the no-lift values.

---

## 6. How we got here (timeline)

| Stage | What we did | What we learned |
|---|---|---|
| D1-D3 | Position-residual approximation of the paper's controller, then a hybrid actuator version | The walking policy was overriding the recovery command; real actuator-level override was needed |
| D4-D5.4 | Isolated one leg, added geometric crossing logic | First genuine recovery completions; the failure was repeatable and diagnosable |
| D5.5x | Stall analysis (foot held by the wire, policy stuck in a standing-like state) | Led to the scripted foot-space step instead of tuning thresholds |
| D5.58 | Scripted IK step, retry, per-leg timing | **128/136 seeds pass, one leg** |
| D6 | Two legs and four legs (`MultiLegRecoveryAction`) | Rear pair 98%, front pair 36%, four legs 29%; start timing was the weak spot |
| D6 new rules | Touchdown window, gap, gates, crossing fallback | Front pair 36% to 57.5% (46/80) |
| D7.0-D7.2 | Forced-step test with no wire, then body-support lift | Sag is a policy effect; lift gave front pair 46/80 to 65/80 |
| D7.5-D7.9 | Smaller lift, slower ramp, shorter timeout, combined | Front pair 74/80; four legs 60/80; stability back near baseline |
| D7.10-D7.15 | Five rear-stage ideas, then fresh seeds | None beat D7.9; fresh-seed check showed some gains were noise |

One lesson worth saying out loud: an early comparison of the lift against a weak baseline
wrongly concluded it did not help. Re-testing on the project's best setup showed 32/40 against
20/40. Every later test used the best verified setup and recorded its command.

---

## 7. Results

**Pass rates (varied approaches, seeds 0-79 unless noted).**

| Setup | Before | Now |
|---|---|---|
| One leg (D5.58) | 128/136 | unchanged |
| Rear pair | 39/40 (98%) | unchanged |
| Front pair | 29/80 (36%) | **74/80 (92%)** |
| Four legs | 31/80 (39%, new start rules, no lift) | **60/80 (75%)** |
| Four legs, fresh seeds 80-159 | not tested | **65/80 (81%)** |
| Four legs, all 160 seeds | | **125/160 (78%)** |

**Front pair, seeds 0-79:** no lift 46/80, half lift 62/80, full lift 65/80, half lift with
the 12-step ramp and 30-step timeout 74/80. Steps on seeds 0-39 alone: half lift 29/40,
+ 12-step ramp 35/40, + 30-step timeout 37/40, both 38/40 (36/40 on seeds 40-79).

**Stability, four legs, seeds 0-39 passing runs** (peak roll/pitch in degrees, angular
velocity RMS in rad/s, steps 40 to the end):

| Lift | Passes /40 | Roll | Pitch | Ang. vel. |
|---|---|---|---|---|
| none | 13 | 6.4 | 7.8 | 0.55 |
| full | 28 | 9.8 | 15.3 | 0.71 |
| half | 19 | 5.8 | 12.0 | 0.59 |
| half + 12-step ramp + 30-step timeout | 29 | 5.4 | 8.9 | 0.45 |

The combined setting has more passes than the full lift's 28 and roll, pitch and
angular velocity close to the no-lift baseline.

---

## 8. Current best recipe (four legs)

`--td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6
--crossing-fallback 10 --new-legs FL FR --max-attempts FL 2 FR 2 RL 2 RR 2
--max-recovery-steps 96 --step-gate-timeout 30 --support-lift-thigh=-0.2
--support-lift-calf=0.3 --support-lift-ramp-steps 12 --support-lift-trigger FL FR`

Full viewer and sweep commands for every result are in `D6_MULTI_LEG_RECOVERY_SUMMARY.md`
(D7.9 for the best setting). Example viewer, seed 80 (a clean pass):

```
uv run python scripts/d6_1_front_pair_wire.py --pair all --start-x-range -0.485 -0.35 \
  --seed 80 --debug --out-dir logs/d7_viewer/best_fourleg_seed80 \
  --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 \
  --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
  --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 \
  --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 \
  --support-lift-trigger FL FR
```

---

## 9. What did not work

| Idea | Result | Take-away |
|---|---|---|
| Lift-and-slide step (no retract first) | 2/40 | retracting first lets the calf clear the wire |
| Rear legs use the new start rules and the lift (D7.10) | 47/80 against 60 | rear stage is sensitive |
| Longer serialization gap, 24 steps (D7.11) | 36/80 | waiting while pressed on the wire hurts |
| Rear roll gate 5 deg | 61/80, no real change | the gate rarely fires |
| Rear body-height gate (D7.12) | 54/80 and 48/80 | delays the second rear leg, passes fall |
| Gap 6 and gap 3 (D7.13-D7.15) | 66/80 and 55/80 on seeds 0-79; gap 6 gave 60/80 against 65/80 on fresh seeds | the apparent win was seed noise |
| Hold/freeze the body, partner hold, roll gate (front) | no effect | see D6 summary |

The pattern: on a set of 80 seeds, a +6 or −6 difference is within chance (about 4 seeds
of binomial noise at 75%), and the seeds flip a lot between settings. Fresh seeds are what
separate real gains from noise.

---

## 10. Limitations

- **Contact detection is an oracle.** The recovery trigger reads the simulator's wire-contact
  sensor. The paper (Yim et al. 2023) detects entanglement from hip torque
  (about 2.5 N·m), which is proprioceptive and works on real hardware. Replacing the oracle
  with a torque detector has not been done.
- **Simulation only**, with a frozen walking policy and one wire.
- **Pass/fail is geometric.** A crossing can pass while looking stiff or unstable. With the
  lift the robot stands about 1 cm higher on average than without it and pitches more
  (8.8 deg against 7.8 deg without lift, four legs).
- **Rear stage after the front stage is the main loss** (about 24 of 35 four-leg failures).
  Rear legs alone pass 98%; of the seeds that reach the rear stage, about 84% pass
  (125 of 149).
- **Several settings were tuned on seeds 0-79.** The gap and the combined setting were
  re-checked on fresh seeds; the lift sizes were not.
- **Behaviour differs from the paper's video.** In the paper the robot keeps a level body and
  uses walking momentum. Our lift is a fixed offset standing in for that.

---

## 11. Next steps (ranked)

1. **Rear stage**, with a different idea from the timing rules: a higher rear foot lift (failing
   rear steps peak at about 0.11-0.12 m against 0.16-0.17 m in passing ones).
2. **Second front leg (FL)** starts with the wire already above its calf; the retract-then-lift
   goes up into the wire.
3. **Proprioceptive trigger** (hip torque) to remove the sensor oracle.
4. **Residual RL on top of the frozen policy**, if the rule-based approach stays below about
   85% on the rear stage.
5. **Spirit-40 vs Go2 geometry** comparison, to understand why the paper's robot keeps its body
   level.

---

## 12. Questions you may get

- **"Why not retrain the walking policy?"** The policy is the baseline that works; the
  question is how far a thin recovery layer can go without touching it. Retraining is the
  fallback if the rear stage stays low.
- **"Is 78% good?"** It is 75% on the tuning seeds and 81% on fresh ones, from 33-39% before the
  lift. Single legs pass 94%. The remaining failures are mostly the second pair of legs.
- **"Does it work on hardware?"** Not shown. The contact sensor in the trigger is a
  simulator oracle.
- **"How do you know it isn't overfitted to the seeds?"** Fresh seeds 80-159 were never used
  for tuning: D7.9 scored 65/80 there. A gap-6 improvement that looked real on seeds 0-79
  reversed on them, which is why the default stayed at 12.
- **"What is a seed?"** A start position along the gait cycle, so the gait phase at wire
  contact differs. Different seeds do not use different robot or wire parameters.
- **"Why a lift only on the front-leg steps?"** Lifting during rear steps lowered the rear
  pass rate in D7.3 and D7.10.
- **"Why is the body higher with the lift?"** The lift pulls the standing legs straighter. The
  full lift raised it about 1.7 cm and wobbled the robot; the half lift with a slow ramp raises it
  about 1.3 cm with stability close to no lift.
