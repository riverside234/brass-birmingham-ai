# g6230 soft-head revision

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g6230 in the newest export
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g6230-20261127T052928674970Z`). The changes are adopted once with
`--soft-head-revision` at a committed boundary, then kept (`src/brass_ai/recovery/training_revision_g6230.py`,
block `soft_head_revision`).

## Request (2026-11-26)

1. Mask the soft head's target and output, and normalize each over only the legal actions with `visits > 0`.
2. A sample with at most one such action (a forced move, an anomalous step) gets soft loss weight 0, and its loss is
   not computed.
3. The soft head's LR is 1e-4 for 100 iterations from the continuation, then back to the current PPO-linked setting.
4. The soft head trains on fixed targets built from each sample's collection-time logits (`old_logits`) and its search
   bonus.
5. The soft held-out KL compares the soft head with fixed targets built the same way for each held-out sample. It is
   computed only once, after PPO.
6. PPG `grad_clip` 2.5, with the Adam state rescaled.

## Changes

| | before (g3423 to g5685) | after |
| --- | --- | --- |
| Soft target | `softmax((current logits + bonus) / 4)` over all legal actions, rebuilt every step | fixed per row: `softmax((label logits + bonus) / 4)` over the legal actions with visits > 0, zero elsewhere |
| Soft head output | log-softmax over all legal actions | log-softmax over the same visited actions |
| Rows with at most 1 visited action | trained like the others | weight 0, no loss computed; a step with none left skips the head |
| Soft held-out KL | 3 checks (before PPO, after PPO, after PPG) against `stored target ** (1/4)` | 1 check, after PPO, against the fixed targets |
| Soft head rate | the PPO step rate (6e-6) | 1e-4 for g6230-g6329, then the PPO step rate again from g6330 |
| PPG clip | 2.4 (g5685) | 2.5, with a one-time PPG Adam rescale |

The following do not change:
- the soft head's temperature (4), weight (8 x beta), clip (1.0) and warm-up (finished);
- its committed head and Adam state (the g3423 block and its settings stamp are untouched);
- the main policy's teacher targets, which are still rebuilt from the current policy;
- the label rows, buffers and beta;
- PPO's clip (6.5) and rates.

### Soft target and masking

The label logits are the policy logits the collector stored with each label (`label['logits']`), so they are the
collection-time `old_logits`. The bonus is the teacher term's own: `teacher_term.search_bonus` at the g3270 label
scale (.2), the same vector the main teacher target uses. For each row (`soft_policy.fixed_target`):
- the target is `softmax((logits + bonus) / T)` taken over the visited legal actions, renormalized there and zero
  elsewhere;
- the head's logits get the same visited mask before their log-softmax (`SoftPolicy.fixed_loss`).

The target depends only on the stored label, so it does not move with the policy. The teacher term builds it once per
drawn row per iteration and stages it on the GPU beside the bonus (`StagedRows` extras `soft_target` and
`visited`). The host path builds the same tensors.

A row needs at least 2 visited actions (`MIN_VISITED`). With one visited action, the masked softmax is 1 on both
sides and the KL is exactly 0. With none, it is undefined. A step's soft loss is the sum of the eligible rows' KL,
divided by all of the step's rows: the other rows have weight 0 and no forward work is spent on their soft loss,
while the eligible rows keep the weight they had. A step whose rows are all weight 0 takes no soft head step and
sends no soft gradient to the trunk (`skipped_steps`).

With Gumbel root search, the visited actions are the considered set: `effective_k` is at most 16, and often far
below the legal width. So the soft head now learns only the order among the actions the search looked at. It is not
exported and never chooses a move. Its gradient still reaches the shared scorer and trunk at 8 x beta, now toward the
collection-time improved policy restricted to the visited actions.

### Held-out KL, once after PPO

`TeacherPhase.after_ppo` calls `SoftPolicy.heldout_fixed` on the validation-table long labels. For each row with at
least 2 visited actions it computes KL(fixed target || head softmax over the visited actions); the other rows are
counted in `skipped_rows`. The checks before PPO and after PPG are dropped. The policy's own held-out measures before
and after the iteration stay; their logits come from `policy_logits`, as they do without a soft head.

In the 128 iterations of this export's history, PPG moved the soft held-out KL by a median 0.00016 (0 in 49 of
them), against 0.0013 for PPO.

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

- Compute: small. One host softmax per drawn row per iteration builds the fixed targets, beside the existing bonus
  computation. That is a few thousand of the 13,355 pooled rows, estimated under 0.3 s. Staging the two extra
  vectors takes a few MB of GPU memory. The soft loss now covers only the visited columns of eligible rows. Dropping
  two of the three soft held-out checks saves under 0.1 s per iteration; the whole after-PPO check took a median
  0.03 s.
- Training signal: the soft target no longer tracks the current policy. Labels up to 8 iterations old keep their
  collection-time targets.
- The 16x larger soft head steps last 100 iterations.
- PPG steps on the 6-8% of batches above 2.4 get up to 4% larger.

## Checks after adoption

- The log prints the g6230 summary line, `soft-head-revision-g6230: PPG clip 2.4 -> 2.5` (factors about 1.0048 /
  1.0140), and per iteration `soft_heldout_after_ppo=` with `soft_rate=0.0001`.
- `teacher_phase.term.soft.learning_rate.source` is `g6230-fixed-0.0001-until-g6330`, and `last` is 1e-4. From g6330
  they are `ppo-step-rate` and the PPO rate.
- `teacher_phase.term.soft.fixed_targets` shows the rows built, `eligible_rows`, `weight_zero_rows` (expected rare),
  `skipped_steps` (expected 0), `visited_actions` (the support size) and `dropped_unvisited_mass` (the share of the
  unmasked target on unvisited actions).
- `teacher_phase.heldout.soft` has only `after_ppo` (`rows`, `skipped_rows`, `kl`, `by_phase`) and its seconds.
- `value_phase.gradient_clipping.max_norm` is 2.5 with `clipped_percent` near 6%.

## Tests

`tests/Recovery/test_soft_head_revision_g6230.py` (11):
- the recipe: one block, the PPG clip, the rate window, refusals;
- forward-only transitions;
- the PPG fit and its norm reader;
- the PPG Adam rescaled once after the earlier receipts;
- the fixed target's math;
- `TeacherTerm.apply` with fixed targets: the loss matches a hand computation, weight-0 rows, a step of only weight-0
  rows skipped, the host path equal to the staged one, and an unrevised head unchanged;
- the teacher phase: one soft check after PPO, the 1e-4 rate, and the held-out KL against a hand computation.

The related suites were run together: 20 modules, 221 tests, covering g5685, g5466, g5402, g3423 soft, g4475
after-PPO, g3270, g4770, g3140, PPG g210, the shadow runtime and resume, g4551, g4138, g3471, g3528, g6120, local
scoring PPG, the GAE value phase and the teacher-phase revision. 9 fail, all already in
`docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md` with the same messages:
- `test_ppg_g210.Restored` ×2
- `test_gae_value_phase` ×3
- `test_teacher_phase_revision.DistinctUsageTests` ×3
- `test_table_revision_g4551.SelfPlay` ×1

## Execution identity

`9876fe7cdd82cd9c7658d880574b203953b3b640febbfe0f1e93802a22c79a75`, with predecessor `0fb8d4ce...`. The entry is
`g6230_soft_head_revision` in `configs/learning/b07-replay-location-review.json`.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 6230 --iterations 10 --share-target 0.065 --soft-head-revision
```

Use the run's committed generation in `--expected-generation`. The earlier revision flags are already adopted and are
not needed again.
