# g6230 soft-head revision

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g6230 in the newest export
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g6230-20261127T052928674970Z`). The changes are adopted once with
`--soft-head-revision` at a committed boundary, then kept (`src/brass_ai/recovery/training_revision_g6230.py`,
block `soft_head_revision`).

## Request (2026-11-26, revised 2026-11-27)

1. Mask the soft head's target and output, and normalize each over only the legal actions with `visits > 0`.
2. A sample with at most one such action (a forced move, an anomalous step) gets soft loss weight 0, and its loss is
   not computed.
3. The soft head's LR is 1e-4 for 100 iterations from the continuation, then back to the current PPO-linked setting.
4. ~~The soft head trains on fixed targets from each sample's collection-time logits (`old_logits`) and its search
   bonus.~~ Rolled back on 2026-11-27, before any lab use: the soft head still uses the current logits.
5. The soft held-out KL compares the soft head with targets built the same way for each held-out sample. It is
   computed only once, after PPO.
6. PPG `grad_clip` 2.5, with the Adam state rescaled.

## Changes

| | before (g3423 to g5685) | after |
| --- | --- | --- |
| Soft target | `softmax((current logits + bonus) / 4)` over all legal actions | `softmax((current logits + bonus) / 4)` over the legal actions with visits > 0, renormalized there, zero elsewhere |
| Soft head output | log-softmax over all legal actions | log-softmax over the same visited actions |
| Rows with at most 1 visited action | trained like the others | weight 0, no loss computed; a step with none left skips the head |
| Soft held-out KL | 3 checks (before PPO, after PPO, after PPG) against `stored target ** (1/4)` | 1 check, after PPO, against targets built like the training ones |
| Soft head rate | the PPO step rate (6e-6) | 1e-4 for g6230-g6329, then the PPO step rate again from g6330 |
| PPG clip | 2.4 (g5685) | 2.5, with a one-time PPG Adam rescale |

The following do not change:
- the soft head's temperature (4), weight (8 x beta), clip (1.0) and warm-up (finished);
- its committed head and Adam state (the g3423 block and its settings stamp are untouched);
- the main policy's teacher targets;
- the label rows, buffers and beta;
- PPO's clip (6.5) and rates.

### Soft target and masking

As before, the target is rebuilt at each step from the current policy's logits and held fixed for the gradient. The
bonus is the teacher term's own: `teacher_term.search_bonus` at the g3270 label scale (.2), the same vector the main
teacher target uses. What changes is the support (`SoftPolicy.masked_loss`):
- the target is `softmax((current logits + bonus) / T)` taken over the visited legal actions, renormalized there and
  zero elsewhere;
- the head's logits get the same visited mask before their log-softmax.

Each drawn row's visited mask is built once per iteration from its label counts (`soft_policy.visited_support`). It
is staged on the GPU beside the bonus (`StagedRows` extra `visited`), and the host path builds the same tensor.

A row needs at least 2 visited actions (`MIN_VISITED`). With one visited action, the masked softmax is 1 on both
sides and the KL is exactly 0. With none, it is undefined. A step's soft loss is the sum of the eligible rows' KL,
divided by all of the step's rows. The other rows have weight 0 and no soft-loss work is spent on them, while the
eligible rows keep the weight they had. A step whose rows are all weight 0 takes no soft head step and sends no soft
gradient to the trunk (`skipped_steps`).

With Gumbel root search, the visited actions are the considered set: `effective_k` is at most 16, and often far
below the legal width. So the soft head now learns only the order among the actions the search looked at. It is not
exported and never chooses a move. Its gradient still reaches the shared scorer and trunk at 8 x beta.

### Held-out KL, once after PPO

`TeacherPhase.after_ppo` calls `SoftPolicy.heldout_masked` on the validation-table long labels. The target is built
from the policy's logits in the same forward, plus the bonus, over the visited actions. For each row with at least 2
visited actions, it computes KL(target || head softmax over the visited actions); the other rows are counted in
`skipped_rows`. It also reports `dropped_unvisited_mass`, the share of the unmasked target that the mask removes.

The checks before PPO and after PPG are dropped. The policy's own held-out measures before and after the iteration
stay; their logits come from `policy_logits`, as they do without a soft head. In the 128 iterations of this export's
history, PPG moved the soft held-out KL by a median 0.00016 (0 in 49 of them), against 0.0013 for PPO.

This KL is a different measure from the old `heldout.soft.after.kl` (about 0.046), which used the softened stored
target over all legal actions. The two numbers are not comparable.

### Soft head rate

For the 100 iterations g6230-g6329, the head's own Adam steps at a fixed 1e-4 (`SoftPolicy.use_rate`). From g6330 it
follows the PPO step rate again (`follow`, g5402's link). Its Adam moments are kept: Adam normalizes the gradient
scale, and a rate change needs no moment rescale. The head's steps become about 16 times larger (6e-6 -> 1e-4). This
is the rate it trained at from g3423 to g5402. Its gradient norms were 0.05-0.12 against its 1.0 clip at g6229, with
0 clipped steps.

### PPG clip 2.5

The PPG Adam rescale uses the same method as g5466 and g5685: a normal fit of each of the last 30 committed
iterations' logged mean and P98 (`value_phase.gradient_clipping`), pooled by steps.

| PPG window g6200-g6229 (clip 2.4) | |
| --- | --- |
| attempted steps | 942 |
| pre-clip mean | 1.48 (per-iteration P98 median 2.58, largest 11.4) |
| steps above 2.4 | 7.5% logged, 7.7% from the fit |
| predicted above 2.5 | 6.2% |
| exp_avg / exp_avg_sq | x1.0048 / x1.0140 |

The rescale runs once on the adoption generation's checkpoint (`ppg_g210.initialize` -> `rescale_ppg`). Its receipt
is stored in the PPG Adam group beside g5466's and g5685's, which keep their own rescales no-ops. The factors are
small because 2.5 moves the clip only within the top 6-8% of norms.

## Cost

- Compute: negligible. Building one visited mask per drawn row takes a host pass over its counts, and staging it takes
  about a megabyte of GPU memory. The soft loss covers only eligible rows. Dropping two of the three soft held-out
  checks saves under 0.1 s per iteration; the whole after-PPO check took a median 0.03 s.
- Training signal: the soft head and its 8 x beta trunk gradient no longer see the unvisited actions.
- The 16x larger soft head steps last 100 iterations.
- PPG steps on the 6-8% of batches above 2.4 get up to 4% larger.

## Checks after adoption

- The log prints the g6230 summary line, `soft-head-revision-g6230: PPG clip 2.4 -> 2.5` (factors about 1.0048 /
  1.0140), and per iteration `soft_heldout_after_ppo=` with `soft_rate=0.0001`.
- `teacher_phase.term.soft.learning_rate.source` is `g6230-fixed-0.0001-until-g6330`, and `last` is 1e-4. From g6330
  they are `ppo-step-rate` and the PPO rate.
- `teacher_phase.term.soft.masked_targets` shows the rows drawn, `eligible_rows`, `weight_zero_rows` (expected
  rare), `skipped_steps` (expected 0) and `visited_actions` (the support size).
- `teacher_phase.heldout.soft` has only `after_ppo` (`rows`, `skipped_rows`, `kl`, `by_phase`,
  `dropped_unvisited_mass`) and its seconds.
- `value_phase.gradient_clipping.max_norm` is 2.5 with `clipped_percent` near 6%.

## Tests

`tests/Recovery/test_soft_head_revision_g6230.py` (11):
- the recipe: one block, the PPG clip, the rate window, refusals;
- forward-only transitions;
- the PPG fit and its norm reader;
- the PPG Adam rescaled once after the earlier receipts;
- the visited support and masked target math;
- `TeacherTerm.apply`:
  - with the policy moved after the rows were drawn, the loss matches a hand computation from the current logits,
    not from the collection logits;
  - weight-0 rows, and a step of only weight-0 rows skipped;
  - the host path equal to the staged one;
  - an unrevised head unchanged;
- the teacher phase: one soft check after PPO, the 1e-4 rate, and the held-out KL against a hand computation.

After the rollback, 10 focused modules were run: this file, g5685, g3423 soft, g4475 after-PPO, g3270, g3140, the
shadow runtime, g6120, g4138 and g3471. All 116 tests pass. The first version had also run 20 related modules (221
tests); its only failures were the 9 already listed in `docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md`.

## Execution identity

`4aaa5fba8c3631ee5f0d3e4fd9301fe24fc535a7cf8b0bc6012e68ec85059df2`, with predecessor `9876fe7c...` (the 95dd6394
version with the collection-logit targets, never run). The entry is `g6230_soft_head_revision` in
`configs/learning/b07-replay-location-review.json`.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 6230 --iterations 10 --share-target 0.065 --soft-head-revision
```

Use the run's committed generation in `--expected-generation`. The earlier revision flags are already adopted and are
not needed again.
