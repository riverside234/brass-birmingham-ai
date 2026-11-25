# g5402 optimizer revision

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g5402 (export
`joint7-v11-2-g4970-rail025-soft8-beta0.02-g5402-20261124T231937105321Z`). Adopted once with
`--optimizer-revision` at a committed boundary, then kept (`src/brass_ai/recovery/training_revision_g5402.py`).

## Request (2026-11-24)

1. PPO `grad_clip` 1.0 -> 5.83. PPG gets a separate `grad_clip`, kept at 1.0.
2. Every PPG `value_phase` report logs gradient clipping: mean, median and P95 of the pre-clip total norm that
   `clip_grad_norm_` returns, plus the number and percentage of steps that clipped, over all attempted PPG optimizer
   steps (rolled-back steps included). Logging only, at minimal cost.
3. At the restart, the PPO LR drops by 20% and that LR becomes the base (1.0).
4. The soft policy head uses the PPO LR.
5. Every ownership/auxiliary head uses the normal PPG LR, with the same warm-up and decay.
6. Follow-up: a one-time rescale of PPO's Adam state at the restart, so the steps don't grow after the clip change.

## What the evidence showed before the change

From the export's 128 summarised iterations (g5270-g5401):

| | value |
| --- | --- |
| PPO pre-clip gradient norm, mean of iteration means | 4.77 (range 4.46-7.24) |
| PPO pre-clip norm, mean of iteration P95s | 5.86 |
| PPO steps clipped at 1.0 | all of them, every iteration (46 of 46) |
| PPO rate | 6e-6 (base 6e-6, controller multiplier 1.0 in 123 of 128 iterations) |
| PPO KL per update | .0066 mean against the .01 target; 1 early stop |
| PPG peak rate | 3.9e-6 (.65 x the PPO rate) |
| PPG iterations keeping any steps | 57%; 17 kept of 31 attempted steps on average |
| Era heads' Adam rate | 1e-4 in PPO; 1e-4 x the PPG schedule factor in PPG |
| Soft head's Adam rate | 1e-4 |

So the clip at 1.0 normalised every PPO step, and 5.83 sits at the typical P95: about 8% of steps will still
clip.

## What changes

| Item | Before | After |
| --- | --- | --- |
| PPO clip (`update.grad_clip`) | 1.0 | 5.83 |
| PPG clip (`optimizer_revision.ppg_grad_clip`, passed to `value_phase.fit`) | `update.grad_clip` (1.0) | 1.0, its own |
| PPO base rate (`learning_rate`) | 6e-6 at multiplier 1.0 | 4.8e-6 at multiplier 1.0 |
| PPO floor (g1307) | 4e-6 (min multiplier .667) | 4e-6 (min multiplier .833) |
| PPG peak | 3.9e-6 | 3.12e-6 (still .65 x the applied PPO rate) |
| Soft head rate | 1e-4 | the PPO rate of the same PPO step (4.8e-6) |
| Era heads in PPG | 1e-4 x schedule factor | the PPG rate of the same step: 3.12e-6 x the same factor |
| Era heads in PPO (preservation) | 1e-4 | the PPO rate of the same step (4.8e-6) |
| PPO Adam moments (follow-up) | built on gradients clipped to norm 1 | rescaled once to the new clip: `exp_avg` x 4.744, `exp_avg_sq` x 22.96 |
| PPG report | no clip metrics | `value_phase.gradient_clipping` |

The 20% cut is taken from the rate in force, base x multiplier, read from the committed iteration's report
(`committed_multiplier`). The multiplier is 1.0 now, so the new base is 4.8e-6.

On the adoption generation's committed checkpoint, `resume.apply_optimizer_config` calls
`training_revision_g5402.rebase_ppo` once. It sets the PPO Adam group's warm-up `base_lr` to 4.8e-6, the controller
multiplier to 1.0, the KL window to empty and the cooldown to 3. It also multiplies every parameter's `exp_avg` and
`exp_avg_sq` by the recorded rescale factors (next section). Adam's step counts, the warm-up counter, the VP health
window and the data signature are kept. A crash at g5402 reloads the same checkpoint and rebases it the same way.
Later checkpoints are already rebased and are not rescaled again, and any other mismatch is refused.

## Design choices

- **The cut counts as one controller cut.** 20% is exactly the controller's own cut (1/1.25). So, as after any cut,
  the three-iteration KL window restarts and the three-iteration cooldown runs. No raise can then come from KL
  measured under the old clip.
- **The 4e-6 floor stays absolute.** g1307 defines it as a 4e-6 floor on the rate, and `applied()` tests it that way.
  With the base at 4.8e-6 the controller has one cut of room left (4.8e-6 / 1.25 = 3.84e-6, held at 4e-6). Scaling
  the floor with the base (3.2e-6) would be a separate change.
- **PPG follows the cut.** PPG's peak has always been .65 x the PPO rate actually applied, so it falls by 20% too.
- **The era heads' PPO-phase steps take the PPO rate.** Those steps are output preservation while PPO moves the
  trunk, the same loss the main value heads get at the PPO rate. "Same LR as the main network in the phase" was read
  as the intent of items 4 and 5. Keeping them at 1e-4 is the alternative.
- **The rates are linked per step, not copied.** The soft head reads `learning_rate.upcoming(ppo_adam, ppo=True)`,
  which is the rate `scheduled_step` applies to that same PPO step. Warm-up and controller changes therefore carry
  through. The era heads receive the main rate in `begin()`: PPG's peak times the PPG schedule factor, including the
  head-only steps after the trunk stops.
- **Settings are not rewritten.** The soft head's settings (rate 1e-4, clip 1.0) and the local-scoring optimizer
  block stay as they are, so the soft state stamp, the companion checkpoint and the label buffer carry over. The
  1e-4 is now only an upper bound that `begin()` checks.
- **The paused search-value head is untouched.** It takes no steps while paused.

## Head rates fall a lot

| Head | Before | After | Ratio |
| --- | --- | --- | --- |
| Era heads, PPG peak | 1e-4 | 3.12e-6 | 1/32 |
| Era heads, PPO | 1e-4 | 4.8e-6 | 1/21 |
| Soft head | 1e-4 | 4.8e-6 | 1/21 |

These heads have their own Adams, so they will move 20-30x more slowly per step. Expect `aux_ce` and the soft KL to
change far less per iteration. The ownership stop may accept more often, because 16 steps at these rates barely move
the CE.

## Adam rescale (follow-up request, 2026-11-24)

Every PPO gradient the Adam has seen had norm 1.0. Unclipped they will have norm about 4.7. The first moment adapts
in about 10 steps (beta1 .9). The second moment takes about 1000 steps (beta2 .999), which is about 22 iterations at
46 steps each.

Without a rescale the step size would be larger than today until it catches up. The table is a global estimate,
from the norm distribution above:

| Iterations after adoption | Step size x today without the rescale (with the .8 rate) |
| --- | --- |
| 1 | 2.7 |
| 2 | 2.2 |
| 5 | 1.6 |
| 10 | 1.3 |
| 20 | 1.0 |
| 40 | 0.86 (settling at 0.8) |

KL grows roughly with the square of the step, so the first iterations would hit the .01 PPO KL guard early, and the
controller would cut to the 4e-6 floor.

So the rebase also rescales the moments once. Each old step's gradient G was scaled to norm 1.0. Under the new clip
it would have been scaled to min(n, 5.83), where n is its pre-clip norm. So, as estimates of the moments under the
new clip:

- `exp_avg` x E[min(n, 5.83)];
- `exp_avg_sq` x E[min(n, 5.83)^2].

The norm n follows a normal fit (pooled mean and variance) of the committed iterations' PPO norm statistics in
`state.recent_training` (`committed_gradient_norms`). The fit requires that every one of their steps clipped at 1.0.

`adam_rescale` computes the expectations in closed form for a normal capped at 5.83. The inputs and factors are
recorded in `optimizer_revision.adam_rescale`, and `validate` recomputes them. On the g5402 state the fit uses
g5400-g5401, 92 steps with norm mean 4.765 and std 0.708:

| Factor | Value |
| --- | --- |
| `exp_avg` | x 4.744 |
| `exp_avg_sq` | x 22.96 (= 4.791^2) |

The update direction m/sqrt(v) changes by 4.744 / 4.791 = 0.99, so the first steps are 0.8x today's, as asked, and
the moments keep their per-parameter shape. A test trains an Adam on norm-4.8 gradients clipped to 1.0 and then
moves the clip to 5.83. Without the rescale the next 46 steps are over 2.5x larger; with it they stay within 10% of
today's size.

Cost: one multiply over the PPO Adam's moment tensors, once, at load. The rescale is a global estimate.
Per-parameter moments see the same clip factor, but the factor's correlation with each step's direction is
ignored, which is small with a norm standard deviation of 15%.

## Logging (item 2)

`value_phase.fit` keeps each attempted step's `clip_grad_norm_` result as a device scalar. That adds no host sync per
step; the existing `error_if_nonfinite` check already syncs. `clipping_report` moves the scalars to the host once per
phase. Every PPG report, including early returns and not-due phases, carries:

```
gradient_clipping: {max_norm, steps, clipped_steps, clipped_percent,
                    pre_clip_norm: {mean, median, p95}, scope}
```

`steps` equals `attempted_steps`: rolled-back intervals and KL restores are included. A step clips when its
pre-clip norm exceeds `max_norm`, the same rule PPO's `gradient_clip_steps` uses. The value phase is copied whole
into the retention summaries, so trends need no extra export. The training log adds `value_clip=k/n@1` and
`value_gnorm=<median>`.

## Cost

| | Cost |
| --- | --- |
| Time per iteration | about 0: one `torch.stack` and copy per PPG phase, one float per PPO step |
| CPU, memory, data | none |
| Collection | unchanged; trainer-only |

At adoption the boundary is journalled as `configuration-history/optimizer-revision-5402` and any interrupted
iteration is archived, as with the earlier trainer-only revisions.

## Checks after g5403

- `optimizer_resume.optimizer_revision` in the first iteration's report: rate 6e-6 -> 4.8e-6, cooldown 3,
  `adam_rescale` 4.744 / 22.96. The training log prints the same.
- `actor.learning_rates` [4.8e-6]. `actor.gradient_clip_steps` should fall from 46 to about 4.
- `actor.lr_controller.kl`: about 0.64x of today's .0066, roughly .004, since KL scales about with the square of the
  step. An early stop or a cut in the first iterations would mean the rescale missed.
- `value_phase.gradient_clipping`: `max_norm` 1.0 with `steps` equal to `attempted_steps`.
- `local_scoring_auxiliary.companion_lr`: PPO 4.8e-6 (or the controller's rate), PPG the last scheduled PPG rate;
  `companion_lr_source` is `main-network-step-rate`.
- `teacher_phase` soft report `learning_rate`: `source` `ppo-step-rate`, `last` equal to `actor.learning_rates[0]`.

## Tests

`tests/Recovery/test_optimizer_revision_g5402.py` (10 tests):

- the recipe: clip, rate, floor, idempotence, `previous`, validation;
- forward-only signature transitions;
- the committed multiplier;
- the committed norms, and the rescale factors against a sample of the same normal;
- the one-time rebase, with the moments scaled by exactly the recorded factors and not again later, and that a cut
  after it reaches the 4e-6 floor;
- the step size across the clip change, with and without the rescale;
- the soft head's step at the PPO rate;
- the era heads' linked `begin`;
- `clipping_report`;
- a full `ppg_g210.fit` with a PPG clip of .5 and linked era heads: every attempted step is counted, and the heads'
  rate equals the value phase's attempted rates.

The 20 related suites ran 207 tests. Six fail identically on unmodified HEAD, so this change did not cause them:

- the two g660 `test_ppg_g210` tests, whose export is no longer in `results/`;
- two `latest_model_config` assertions;
- one test that needs 16 CPU slots;
- one VP-chart assertion.

They are listed in `docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md`.

The dry run on the export's `state.json` (scratchpad `dryrun_g5402.py`) checks:

- the changed keys are `learning_rate`, `optimizer_revision`, `ppo_lr_controller.min_multiplier` and
  `update.grad_clip`;
- labels, beta, the soft settings, local scoring and the value phase are kept;
- the transition is True forward and False backward;
- the g4970 script's fixed flags stay no-ops.

## Execution identity

Recorded as `a7aad72d19526a7c12271cf769a99e011c5bf3ff3dae17ffba4205953691ff4b`, with the Adam rescale, and with
predecessor `2df2cf08…`. The
committed `2df2cf08` did not match HEAD's tree: the plot commits c7d53b80 and fc640bd7 changed `vp_plot.py`,
`rating_plot.py` and `teacher_vp.py` after it was recorded. This record covers them.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 5402 --iterations 10 --share-target 0.065 --optimizer-revision
```

Later continuations may keep or drop `--optimizer-revision`; the adopted block is kept either way.
