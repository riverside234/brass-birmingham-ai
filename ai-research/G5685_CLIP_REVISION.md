# g5685 clip revision

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g5685 in the newest export
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g5685-20261125T184134235433Z`). The clip changes are adopted once with
`--clip-revision` at a committed boundary, then kept (`src/brass_ai/recovery/training_revision_g5685.py`). Turning
off the era heads' norm summary is code-only.

## Request (2026-11-25)

1. PPG `grad_clip` 2.4, with the Adam state rescaled.
2. PPO `grad_clip` 6.5, with the Adam state rescaled.
3. Disable and skip the era heads' (Canal/Rail/Rail-end) own PPO- and PPG-phase P98 norm measurement (added at g5466).

## Changes

| | before (g5466 recipe) | after |
| --- | --- | --- |
| PPO clip (`update.grad_clip`) | 5.83 (g5402) | 6.5 |
| PPG clip (`ppg_grad_clip`) | 2.2 (g5466) | 2.4 |
| PPO Adam | built on gradients clipped at 5.83 | moments rescaled once at adoption |
| PPG Adam | built on gradients clipped at 2.2 | moments rescaled once at adoption |
| Era-head norm summary (`local_scoring_auxiliary.gradient_norms.ppo/ppg`) | measured every head step, reported per phase | not measured or reported |

Nothing else changes. That includes the rates and schedules, the era heads' and soft head's own clips (1.0) and the
rest of g5466: Rail-end preserved in PPO, the era heads on the fixed PPO rate in PPG. The era heads keep
`gradient_norm_max`, which they reported before g5466. Their norms ran 0.009-0.04 against their 1.0 clip at g5683-g5684.

## Evidence: the last 30 committed iterations (g5655-g5684)

| | PPO (clip 5.83) | PPG (clip 2.2) |
| --- | --- | --- |
| steps | 1,354 | 865 |
| pre-clip mean (step-weighted) | 4.78 | 1.34 |
| per-iteration P98, median | 6.52 | 2.17 |
| largest norm | 10.7 | 4.33 |
| steps above the old clip (logged) | 9.0% | 5.9% |
| ... the fit's prediction at the old clip | 10.0% | 6.5% |
| ... the fit's prediction at the new clip | 2.2% | 3.8% |

## Adam rescales

The method is the one g5402 and g5466 used. Clipping scales the gradient to `min(n, c)` along its direction. Once,
on the adoption generation's checkpoint:
- each `exp_avg` is multiplied by `E[min(n, new)] / E[min(n, old)]`;
- each `exp_avg_sq` is multiplied by `E[min(n, new)^2] / E[min(n, old)^2]`.

Each iteration's norm is fitted as a normal, and the fits are pooled by steps:
- PPO uses the logged mean and standard deviation (`actor.gradient_norms`). This fit matched the clip rate better:
  10.0% against 9.0% logged. A fit to the mean and P98 predicted 11.7%.
- PPG uses the logged mean and P98 (`value_phase.gradient_clipping`), the same fit as g5466.

`validate` requires each fit to reproduce the logged clip rate at the old clip within 10 points. Both factors must be
above 1.

| | exp_avg | exp_avg_sq | hook |
| --- | --- | --- | --- |
| PPO 5.83 -> 6.5 | x1.0075 | x1.0187 | `resume.apply_optimizer_config` -> `rescale_ppo` |
| PPG 2.2 -> 2.4 | x1.0077 | x1.0230 | `ppg_g210.initialize` -> `rescale_ppg` |

Each receipt is stored in its Adam group (`brass_ppo_adam_rescale_g5685`, `brass_ppg_adam_rescale_g5685`), so a
later checkpoint is not rescaled again. A crashed attempt reloads the same checkpoint and rescales it the same way.
The PPG group already holds g5466's receipt, which keeps g5466's own rescale a no-op. The factors are computed again
at adoption, from the 30 iterations before the adoption generation. Adopted at g5685, they are the ones above.

Both factors are small because both clips move only within the tail, where 6-9% of steps sat. Adam's settled step
size, `exp_avg / sqrt(exp_avg_sq)`, changes by about −0.2% for PPO (1.0075 / sqrt(1.0187)) and −0.4% for PPG. The
effect is to give high-norm batches up to 11% (PPO) and 9% (PPG) more weight instead of trimming them. Without the
rescale, the first steps after the change would be off by about 1-2% until Adam's second moment caught up.

## Cost

- Compute: none. The same steps, rates and data. The rescale is one multiply per Adam tensor, once.
- Dropping the era-head norm summary saves one device scalar per head step (about 60-100 per iteration) and one
  host transfer. That is negligible.
- Risk: the steps on the 2-4% of batches above the new clips get up to 11% (PPO) and 9% (PPG) larger. The PPO LR
  controller reacts to KL as before. PPG keeps its loss and KL checks and its held-out early stop.

## Checks after adoption

- The training log prints `g5685 clip revision: PPO clip 5.83 -> 6.5` with the PPO Adam factors (about 1.0075 and
  1.0187), and the first PPG phase prints `g5685 clip revision: PPG clip 2.2 -> 2.4`. That PPG report carries
  `adam_rescale` with `phase: ppg`.
- `actor.gradient_clip_steps / actor.gradient_norms.count` falls from about 9% to about 2%. The KL guard, KL early
  stop and controller decisions should stay where they were.
- `value_phase.gradient_clipping.max_norm` 2.4 and `clipped_percent` near 4% (it was 5.9%). Watch the KL checks and
  beta.
- `local_scoring_auxiliary` no longer has `gradient_norms`; `gradient_norm_max` stays.

## Tests

- `tests/Recovery/test_clip_revision_g5685.py` (7):
  - the recipe: both clips move and nothing else does, adoption happens once, a configuration it does not follow
    is refused, and invalid blocks are refused;
  - forward-only signature transitions;
  - both fits, checked against a Monte Carlo sample and the logged clip rates;
  - the norms read from the run's files (skips, too few, the wrong clip);
  - each Adam rescaled once, with the receipt surviving a checkpoint; g5466's PPG receipt left alone;
  - the PPO hook in `apply_optimizer_config`;
  - the era heads keep no per-phase norm summary.
- `test_auxiliary_revision_g5466`: the era heads' per-phase norm test is removed, since that measurement is gone.

The related suites were run together: 20 modules, 203 tests, covering g5402, g5466, g4770, both iteration-speed
modules, g3140 signatures, the PPG g210 phase, local scoring and its PPG and resume paths, the shadow runtime and
resume, the LR controller, the policy anchor, g4551, g4475, g3318, the GAE value phase and g3423. 10 fail, all already
in `docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md` with the same messages:
- `test_ppg_g210.Restored` ×2
- `test_ppo_lr_controller.FixedRecipe` ×2
- `test_policy_anchor.Configuration` ×2
- `test_gae_value_phase` ×3
- `test_table_revision_g4551.SelfPlay` ×1

## Execution identity

`991862cad5dbbcf7084b5d8d786c20f36d67500939bfde942ac45a8f2652faf3`, with predecessor `3aea4236...`. The entry is
`g5685_clip_revision` in `configs/learning/b07-replay-location-review.json`.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 5685 --iterations 10 --share-target 0.065 --clip-revision
```

Use the run's committed generation in `--expected-generation`. `--optimizer-revision` and `--auxiliary-revision` are
already adopted and are not needed again.
