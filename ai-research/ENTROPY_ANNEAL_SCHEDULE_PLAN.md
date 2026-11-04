# Entropy anneal schedule with insolvency-cost steps (OpenAI Five style)

**2026-11-04 extension, adopted from g611:** append a third explicit reward step
at **k=100**: **0.005/event, cap 0.05 → 0.0025/event, cap 0.025**. Preserve the
existing k=0 and k=50 steps and all entropy coefficients/counters. The user
confirmed k=100 rather than an immediate reward change. Version 1.2.0 records
this append-only extension; older signed 1.1.0 schedules remain unchanged.
Entropy holds at each unapplied boundary, including k=100, until the existing
`insolvency-step` restart applies it. The g611 revision also sets epsilon to 0
and PPG beta floor/cap to 0.25/0.003, with clean accepted phases lowering beta
below KL 0.0015. See the current commands in
[the launch instructions](NEUTRAL_SELFPLAY_G1290_RESTART.md).
The original two-stage rationale and recipe below describe the earlier version.

Implemented for the pending g570 continuation; see the
[launch and reward-step commands](NEUTRAL_SELFPLAY_G1290_RESTART.md).
The user selected an initial entropy of **0.005**, including the revised fork
recipe, and starts the schedule at **g571**. The pending g570 settings retain
epsilon **0.02**, T1.3 three-action normal openings, T1.5 two-action post-fork
openings, and the other g570 changes. These supersede the older fork/epsilon
examples below. No long training or evaluation is launched by the helpers.

Two things move together in the planned final phase:
- Entropy coefficient: 0.005 → 0.001 in all four groups.
- Insolvency cost: 0.02 per event (cap 0.075) → 0.005 per event (cap 0.05).

It applies to the neutral g1290 restart, which starts at 0.005 entropy and
0.02/0.075 insolvency. The current fork recipe also starts at **0.005**.

## 1. What OpenAI Five did

| Item | OpenAI Five / Rerun |
| --- | --- |
| Entropy coefficient | 0.01 → 0.001 in both runs; the baseline experiments stayed at 0.01 (Table 2) |
| Shape | "x → y indicates a smooth monotonic transition (usually a linear change over one to three days)" (Table 2 note) |
| Duration | "Each hyperparameter change was applied gradually over the course of 1-2 days, corresponding to several thousand iterations" (Figure 7) |
| When | Rerun: iteration 54k, day 42 of about two months, so roughly three-quarters of the way through |
| Changed at the same time | Learning rate 5e-5 → 5e-6 and GAE horizon 180 s → 360 s |
| Reward shaping | Team spirit, a reward-shaping weight, was also changed gradually: 0.3 → 0.8 from iteration 15k in Rerun (Figure 7) |
| Rationale (Appendix O) | The bonus "prevents premature convergence to suboptimal policy"; 0.01 was best early in training; "we set it to an initial value and lower it during training" |

Takeaways:
- The changes were one-way and gradual.
- They came late, as part of a planned final phase.
- They covered optimisation settings and reward shaping together.
- Absolute coefficients don't transfer between codebases. The ratio (10×) and
  the timing do.

## 2. Decisions

| Setting | Value |
| --- | --- |
| Entropy, start → final | 0.005 → 0.001 in all four groups (5×) |
| Entropy shape | Linear in the coefficient, then hold at 0.001 |
| Entropy ramp | 100 healthy iterations, about 13 h at the restart's ~7.7 min average iteration |
| Entropy step | −0.00004 per healthy iteration |
| Insolvency, start → final | 0.02/event, cap 0.075 → 0.005/event, cap 0.05 |
| Insolvency shape | Two reviewed steps: 0.02/0.075 → 0.01/0.075 at k = 0, then 0.01/0.075 → 0.005/0.05 at k = 50 |
| Advance rule | k increases by 1 only after a committed iteration that passes the existing entropy-controller health check; otherwise it holds. It also holds at k = 50 until insolvency step 2 has been applied |
| Start | An explicit generation, planned as the last quarter of a run. One restart turns on the entropy schedule and applies insolvency step 1 |
| Entropy off | Returns entropy to the 0.005 preset at the next iteration. Insolvency changes only through explicit reverse steps (section 6) |
| Not changed | Reward weights (win 1.0, placement 0.1, VP 0.001, VP cap 180), pending g570 ε 0.02, opening temperature, forks, LR controller, teacher phase |

## 3. Why these numbers

### Entropy

**Final value.** 5× takes you to OpenAI's final 0.001. In your logs, entropy
was already sliding at 0.005 (to 60–72% of the g187–196 level by g1300). So
0.001 means continued sharpening, not a new equilibrium.

**Ramp length.** In your logs, lowering the coefficient acts slowly and raising
it acts fast:
- After g850 cut the controller's coefficients to a fixed 0.005, canal-action
  entropy kept drifting down for 300+ iterations: 0.94 at g560, 0.78 at g1000,
  0.65–0.71 at g1200–g1290.
- Raising to 0.1 moved canal-action entropy +13% in 4 iterations. Raising to
  0.015 moved it +2.5% in 4.

So lowering doesn't need a ramp for stability. The ramp is there so that a
problem shows up while the coefficient is still near 0.005, and so that health
holds can pause it. 100 iterations is about half a day of wall clock, the same
order as OpenAI's 1–2 days.

**Timing.** Appendix O says the bonus prevents premature convergence, so lower
it only after the main learning is done.
- Your g1290 restart is still gaining about 23 PL points per 100 iterations
  (95% interval ±9, g10–g250).
- The old run was still gaining about 16 per 100 at g1000–g1300.
- With ±42 rating noise per slot, it takes about 250 iterations of slots to
  confirm a plateau.

So plan it the way OpenAI did: start it as the last ~25% of a run whose length
you've decided in advance, rather than waiting for a plateau trigger.

### Insolvency: why two steps, not a per-iteration ramp

- **Every reward change is a restart.** The insolvency cost is part of the
  reward spec, and the reward spec is part of the run signature and of the
  model's utility config. Each change goes through a reviewed source transition
  at a restart. There, `utility.migrate()` swaps the utility config and
  invalidates the utility calibration, and the reward transition refreshes PPG
  replay. A per-iteration ramp would trigger all of that every iteration.
- **Only listed transitions are allowed.** `utility.event_cost_upgrade` accepts
  only transitions on its list. Two of them are 0.02/0.075 → 0.01/0.075 (the
  g1320 step) and 0.01/0.075 → 0.005/0.05 (the g1322 step). That is exactly the
  path needed here, with no new entries.
- **The term is small in behaviour.** In the restart at 0.02/event, the average
  cost was 0.00015 per learner trajectory: 7 events across 960 trajectories at
  g199. Halving the cost at g850 moved insolvency rates by only a few tenths of
  a point. Two steps are as gradual as this term needs.
- **Timing.** Step 1 rides on the same restart that turns the entropy schedule
  on. Step 2 at the ramp midpoint gives the critic 100 iterations on the final
  reward (50 of ramp plus 50 of evaluation) before the decision.

## 4. Schedule

    α_g(k) = s_g + (f_g − s_g) · min(1, k / N)

- `k` is the number of healthy iterations since the start.
- `N` is `ramp_iterations` (100).
- `s_g` and `f_g` are the start and final entropy values for group `g`.

| k | before start | 0 | 25 | 50 | 75 | ≥100 |
| --- | --- | --- | --- | --- | --- | --- |
| Entropy α, all four groups | 0.0050 | 0.0050 | 0.0040 | 0.0030 | 0.0020 | 0.0010 |
| Insolvency, per event / cap | 0.02 / 0.075 | 0.01 / 0.075 | 0.01 / 0.075 | 0.005 / 0.05 | 0.005 / 0.05 | 0.005 / 0.05 |

### Config

The applied insolvency stays where it already lives: top-level `insolvency`,
`search.reward`, and `teacher_phase.search.reward`. The schedule block records
the planned steps so they can be validated and reported.

```json
"update": {
  "entropy_weight": 0.005,
  "fixed_entropy": {"contract": "fixed-group-entropy/1.2.0",
    "weights": {"canal_action": 0.005, "canal_other": 0.005, "rail_action": 0.005, "rail_other": 0.005}},
  "entropy_schedule": {
    "contract": "entropy-anneal-schedule/1.1.0",
    "mode": "off",
    "start_generation": null,
    "reset_generation": null,
    "start": {"canal_action": 0.005, "canal_other": 0.005, "rail_action": 0.005, "rail_other": 0.005},
    "final": {"canal_action": 0.001, "canal_other": 0.001, "rail_action": 0.001, "rail_other": 0.001},
    "ramp_iterations": 100,
    "shape": "linear",
    "advance_on": "healthy-entropy-update",
    "measure_every": 5,
    "insolvency_start": {"event_cost": 0.02, "max_cost": 0.075},
    "insolvency_steps": [
      {"at_progress": 0,  "insolvency": {"event_cost": 0.01,  "max_cost": 0.075}},
      {"at_progress": 50, "insolvency": {"event_cost": 0.005, "max_cost": 0.05}}
    ]
  }
}
```

### Validation

Entropy:
- `start` equals the `fixed_entropy` weights: **0.005** for the current fork
  recipe as explicitly requested.
- `0 < final_g ≤ start_g` in every group.
- `ramp_iterations` is an integer ≥ 1. `shape` is `linear`. `advance_on` is the
  fixed string above.
- In `off` mode, `start_generation` is null. A new `on` or reset operation
  requires a start no earlier than the next generation to collect. Normal resume
  retains its original start. `reset_generation` binds an explicit reset to a
  committed boundary and prevents repeating it after a restart.
- `entropy_weight` and the controller's `alpha_base` stay equal to `start`. The
  existing `entropy_weight == alpha_base` check therefore still holds. These two
  fields identify the recipe; they are not the applied value.
- `entropy_control` stays `mode: off`. Revised and neutral recipes require
  that.

Insolvency:
- `insolvency_start` equals the run's insolvency before the schedule starts.
- Each step is an `event_cost_upgrade` transition from the previous spec.
  Steps are in increasing `at_progress` order, and the first is at 0.
- The top-level `insolvency` must equal the spec for the latest step whose
  `at_progress` ≤ k. Once a step has been applied, `search.reward` and
  `teacher_phase.search.reward` must both equal `reward_spec(config)`.
  During the documented midpoint wait, the previously applied spec remains
  active. `off` and an explicit reverse step preserve their signed applied-step
  receipt; they never silently restore the original reward.

### State

Saved with the exploration state:

```json
"entropy_schedule": {"started_at": 1450, "progress": 37, "held": 3, "last_generation": 1490,
  "applied": {"canal_action": 0.00352, "canal_other": 0.00352, "rail_action": 0.00352, "rail_other": 0.00352},
  "insolvency_steps_applied": 1, "waiting_for_insolvency_step": false}
```

### Rules

The implementation also stores the schedule contract/config, actual insolvency
spec and reset count in the checkpoint-bound state. `applied` is the next
iteration's coefficient; reports separately record the coefficient actually
used by the completed iteration. Pending source transitions are replayed before
restoring the schedule, including toggles before the first new checkpoint.

1. Collection and PPO for an iteration both use α(k), with k read when the
   iteration starts. The behaviour record and the update then agree.
2. After the iteration commits, k increases by 1 if `healthy` is true.
   Otherwise `held` increases by 1.
3. `healthy` is the flag `league.py` already builds for the entropy controller
   (around line 638). It is true when all of the following hold:
   - economic health is ok
   - optimizer steps > 0
   - no KL early stop
   - no PPG KL-guard trip
   - categorical KL ≤ 1.5 × target_kl
4. **Reward step gate.** When k reaches a step's `at_progress` but the matching
   insolvency hasn't been applied yet, k stops advancing.
   `waiting_for_insolvency_step` is reported, and training continues at the
   current α and insolvency until you restart with the helper. The run never
   changes the reward on its own.
5. On resume, k, `held` and the applied-step count come from saved state. An
   interrupted iteration never advances.
6. `off` returns entropy to the start values at the next iteration. It does not
   touch insolvency. Turning it `on` again resumes from the saved k, unless the
   helper is told to reset.
7. Once k ≥ N, entropy holds at the final value and insolvency at the last step.

## 5. Code touch points

| File | Change |
| --- | --- |
| `recovery/exploration.py` | Add the schedule contract and `validate_schedule`, including the insolvency-step checks. `effective_alphas` returns α(k) when on. `effective()` then passes those as the behaviour alphas. `validate()` keeps the fixed-preset checks and also validates the schedule |
| `recovery/updates.py`, `entropy_term` | The check `behavior.entropy_alphas == weights` currently compares against the fixed preset. It must compare against this generation's scheduled α |
| `recovery/config.py` lines 590 and 920 | Add `entropy_schedule` to the allowed extra `update` keys |
| `recovery/trained_startup.py`, `validate` | Runs on every config validation (`config.py` around 649) and currently requires insolvency to be exactly 0.02/0.075. Accept 0.01/0.075 and 0.005/0.05 only when the schedule is on and its state records that step as applied |
| `recovery/league.py`, around line 638 | Advance k using the existing `healthy` flag, and hold at a step boundary until the matching reward is applied. Persist the state, add an `entropy_schedule` report block (including `waiting_for_insolvency_step`) and a console line |
| `recovery/league.py`, around line 555 | With policy observations off, the fixed probe isn't evaluated, so the restart reports per-group raw entropy as None. While the schedule is on, run the probe report-only every `measure_every` iterations (about 21 s per run) |
| `recovery/utility.py` | No change. Both transitions are already on the `event_cost_upgrade` list, and `migrate()` handles the model |
| Resume and state | Persist the schedule state. Validate that k only increases |
| Helper | `scripts/configure-entropy-schedule.py` with three commands: `on --start next/<gen>` (turns on entropy and applies insolvency step 1), `insolvency-step` (applies the next step; requires k ≥ its `at_progress`), and `off` (entropy only). Each insolvency step sets `insolvency`, `search.reward` and `teacher_phase.search.reward` together, as `training_revision_g1320.configure` does. Follow the g1320/g1322 revision-helper pattern: preserve weights, optimizer and RNG state, and saved entropy baselines. Apply each step at an iteration boundary; resume already refuses an uncommitted iteration |

## 6. Monitoring and stop rule

During the ramp and for 50 iterations after it, watch:
- per-group raw entropy (from the probe) and the pooled probe entropy
- KL and learning rate
- the held count and `waiting_for_insolvency_step`
- family and industry fingerprints
- economic health
- **insolvent seats and events per seat.** The restart at 0.02/0.075 runs at
  about 1% insolvent seats.

**After an insolvency step.** If insolvent seats roughly double, to above about
2%, go back one step: 0.005/0.05 → 0.01/0.075 is already on the list (the g1340
transition). There is no listed transition back to 0.02/0.075; from 0.01/0.075
the list only allows 0.02/0.08. Returning to 0.02/0.075 would need a new list
entry.

**Decision 50 iterations after the ramp ends.** Run a one-off head-to-head
between the latest checkpoint and the checkpoint from the start of the schedule:
- Use 2v2 tables (new, new, start, start) with seat rotations, and at least 256
  games. That gives roughly ±30 rating points at 95%.
- If the new checkpoint is even or ahead (finishes ahead in ≥ 50% of
  comparisons), keep both changes.
- If it's clearly behind (95% interval entirely below 50%), turn the entropy
  schedule off first, since it is the bigger change. Undo the insolvency steps
  only if insolvency itself rose.

## 7. Interactions

- **Teacher phase.** Keep it off from the start of the ramp until 50 iterations
  after it ends.
  - Both are end-of-run sharpening moves, so overlapping them hides which one
    did what.
  - Each insolvency step also invalidates the utility calibration.
    `calibration_ready` gates teacher collection, teacher value and search
    teaching, and the reward-keyed teacher buffer is invalidated as well.
- **PPG after a reward step.** The first PPG phase after each step trains on one
  age of data instead of two, because the previous iteration's replay has the
  old reward.
- **Utility-head input.** The utility head reads each seat's remaining
  insolvency budget. Step 2 lowers that budget at the start of each game from
  0.075 to 0.05. The recalibration after the step covers it.
- **Search reward.** `search.reward` and `teacher_phase.search.reward` move with
  the learner reward at each step. Frozen opponents keep their own reward
  identities.
- **Learning rate.** OpenAI cut its learning rate 10× at the same moment. Your
  KL controller would undo a direct LR cut, because it raises LR whenever KL
  falls below 0.005 (0.5 × target_kl). If you want that step too, lower
  `target_kl` instead. That is not part of this plan.
- **Diversity.** Expect play to become more repeatable and more one-track; the
  drift you saw at 0.005 continues. With Gumbel search, a sharper prior also
  makes the 16 root candidates more repeatable from game to game. Use the
  schedule for final strength, not during the new IL run's style-forming phase.
- **Ratings.** These are unaffected, because rating games use game results
  only.
- **Rollout exploration.** ε, opening temperature and forks stay unchanged.
  OpenAI's entropy change was also only to the loss coefficient.

## 8. Tests

Entropy:
- α(k) hits both endpoints, decreases monotonically, has the linear values at
  k = 25, 50 and 75, and holds at k ≥ N.
- An unhealthy iteration doesn't advance k and increments `held`.
- Resuming mid-ramp reproduces the same α sequence. An interrupted iteration
  doesn't advance.
- Behaviour-record alphas equal PPO alphas in every generation, and a mismatch
  is rejected.
- Validation rejects `start` ≠ the fixed preset, `final` > `start`,
  `ramp_iterations` < 1, and a `start_generation` in the past.
- `mode: off` reproduces today's 0.005 loss exactly on a fixture.
- Current fork recipe: a start of 0.005 is accepted; a mismatching preset is rejected.
- In reports, the applied α equals `effective_alpha`.

Insolvency:
- Only transitions on the `event_cost_upgrade` list are accepted as steps.
- k holds at 50 until step 2 is applied, and resumes advancing afterwards.
- After each step, `search.reward` and `teacher_phase.search.reward` equal
  `reward_spec(config)`, and a mismatch is rejected.
- After each step, the utility calibration receipt is not ready until a fresh
  calibration.
- Resuming across a step restart preserves k, α and the applied-step count.
- Config validation accepts only listed specs with the schedule block; runtime
  restoration verifies the applied step and its committed progress. Turning
  entropy off keeps an already-applied reward step and its receipt.
- `off` leaves insolvency unchanged.

## 9. Rollout

1. Land the code with `mode: off`. Check that reports still show entropy 0.005
   and insolvency 0.02/0.075.
2. Choose the start generation (the last ~25% of the planned run). In one
   restart, use the helper to turn on the entropy schedule and apply insolvency
   step 1 (0.01/0.075).
3. At k = 50, the schedule waits. Restart with `insolvency-step` to apply step 2
   (0.005/0.05).
4. Ramp entropy to k = 100, then hold at 0.001.
5. Make the decision 50 iterations after the ramp ends.

## Sources

- Berner et al., *Dota 2 with Large Scale Deep Reinforcement Learning* (arXiv
  1912.06680): Table 2, Figure 7, Appendices A, C and O.
  https://arxiv.org/abs/1912.06680
- Appendix notes with the Appendix O exploration quotes:
  https://kiiiiii123.github.io/2020/05/31/post_67.html
