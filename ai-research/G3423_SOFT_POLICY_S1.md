# g3423 soft policy S1: a soft head trained on the long-search labels

- **Date:** 2026-11-18. **Run:** `joint7-v11-2-g1190-restart-3`, newest committed generation g3423.
- **Adopted with** `--soft-policy` at a committed boundary (planned: g3423). It needs the g3270 teacher revision, which the run has.
- **Design:** step S1 of [SOFT_POLICY_TARGET_PLAN.md](SOFT_POLICY_TARGET_PLAN.md). This file covers what was built, what to read in the reports and how to continue the run.

## What it does

- **The head.** A trainer-owned `Linear(192→1)` without bias reads the policy scorer's GELU layer (`policy_scorer.1`). It is not on `FourSeatNet`, not exported, and never chooses a move.
  - At adoption it copies `policy_scorer.2.weight / 4`, so its first output is `softmax(main logits / 4)`.
  - Its raw logits differ from `main / 4` by one constant, the main bias / 4. That constant changes no softmax.
- **Target and rows.**
  - Target: `softmax((current logits + search bonus) / 4)`, from the same detached logits and staged bonus the teacher term uses to rebuild its own target.
  - Rows: the teacher term's 128 long-label rows of each PPO step. PPO rows stay out: the PPO-row factor is 0 until S4.
- **Loss and weight.**
  - Loss: the mean KL from that target to the head, over the 128 rows.
  - Weight on the actor parameters: 8β (8 × 0.0312 = 0.25 at g3422). It follows β if β changes.
- **Gradient path.**
  - The hidden layer comes from a forward hook on the same forward pass. The main logits therefore come from the unchanged `_score_policy` call.
  - The soft loss gets its own `autograd.grad` call: over the head, plus the actor parameters once the trunk multiplier is above 0.
  - The teacher's own call is unchanged. It keeps the graph (`retain_graph`) only when the soft loss reaches the trunk.
  - The head steps its own Adam (learning rate 1e-4, no weight decay, gradient clip 1.0) inside the same `apply`, and only with teaching on. In shadow mode the soft term is measured but does not step.
- **Warm-up.**
  - Head-only for 2 trained iterations: the hidden layer is detached, so no soft gradient reaches the shared scorer or the trunk.
  - Then the trunk multiplier is 0.2, 0.4, 0.6, 0.8, 1.0 (`scale_gradient`).
  - An iteration counts only if the head stepped.
- **State.**
  - JSON lists of exact float32 values in the teacher phase's payload, as `soft_policy` beside `weight`. state.json and the checkpoint hold the same dict.
  - Contents: the head, its copy at adoption, the Adam moments and step, and the warm-up counter (about 770 numbers, ~20 KB of the 18.6 MB state).
  - A fresh head is created only at the adoption generation, so a lost state fails loudly instead of restarting silently.
- **Unchanged:**
  - the teacher term's loss, gradient, share, β and its qualification;
  - labels and buffers;
  - collection, CPU workers and lab-native;
  - the model and its export;
  - the value phase and the ownership stop.

### Details the plan left open

| Point | Choice | Why |
| --- | --- | --- |
| How the hidden layer is read | A forward hook on `policy_scorer.1` during the existing forward | The main logits stay bit-identical by construction, with no copy of the scorer's code to keep in sync. Exactly one capture per forward is required. |
| The gradient the head's Adam sees | The unweighted soft-loss gradient, clipped at 1.0 | Adam's step does not depend on a constant factor, so 8β only decides what reaches the trunk. |
| Adam | torch's single-tensor Adam formula on one 192-vector | Its state is three vectors and a step count, so it fits the JSON payload exactly. |
| Validation tables | `teacher_phase.heldout.soft` before/after, from the forwards that already compute the pre/post logits | No extra forward. The target is the stored search target raised to 1/4 and renormalized, so before and after share one fixed target. |

## What to read

| Field | Meaning |
| --- | --- |
| `teacher_phase.term.soft.stage`, `multiplier`, `next_multiplier` | `head_only`, `ramp` or `full`, this iteration and the next |
| `teacher_phase.term.soft.kl.first`, `.last`, `.median` | Soft KL on each step's label rows, before that step |
| `teacher_phase.term.soft.share` | 8β × the soft gradient norm on the shared state/entity encoders ÷ the PPO actor norm, on the measured steps (every fourth). It is 0 while head-only; about 0.04 at multiplier 1 (*estimate*). |
| `teacher_phase.term.soft.head_change` | ‖w − w₀‖ / ‖w₀‖ from the copy at adoption. It must leave 0 in the first iteration. |
| `teacher_phase.term.soft.distance_from_main` | ‖w − main/4‖ / ‖main/4‖ after the update: how far the soft readout has moved from a tempered copy of the main one |
| `teacher_phase.term.soft.head_gradient_norm`, `clipped_steps`, `seconds` | Size of the head's gradient (clip 1.0), and host time |
| `teacher_phase.heldout.soft.before.kl`, `.after.kl`, `.delta`, `.by_phase` | The head on the 4 validation tables, before they train |
| log line | `soft=stage:multiplier soft_kl=first->last soft_share=… soft_moved=head_change soft_heldout=before->after`, after `heldout_pull=` |

## Cost

- **Time.**
  - Head-only iterations: under 0.1 s each. That covers a 192-wide matmul, a KL over the legal moves, a backward through that matmul only, and one small GPU sync per step.
  - From multiplier 0.2 on: about +0.4–0.5 s (*estimate*), for a second backward over the same 128-row graph. For scale, at g3422 the teacher term took 2.83 s: 2.35 s of staging plus 0.48 s for its 48 steps.
- **Memory:** a few MB on the GPU, and ~20 KB of state.
- **No change to:** search or CPU work, held-out data, or collection.

## Checks run (laptop, fixture models)

`tests/Recovery/test_soft_policy_g3423.py`, 12 tests:

- **Search rows train the head during warm-up.** Multiplier 0, PPO-row factor 0, teaching on, one `apply` step compared with the same step with the soft term off:
  1. The head has a nonzero gradient and moves after its Adam step.
  2. Every model parameter's gradient is bit-identical (`torch.equal`) to the soft-off step.
  3. The teacher's loss, norm, share and cosine are identical, and the soft trunk norm and share are exactly 0.
- **Control at multiplier 1.** The gradient on `policy_scorer.0` changes while `policy_scorer.2` (the main readout) gets none. The teacher's loss is unchanged, and the soft trunk norm is applied at 8β.
- **Shadow mode.** It measures without stepping and without advancing the warm-up.
- **The stratified long-and-short path is refused.**
- **Hidden forward.** The hook leaves the logits bit-identical, and `heldout` returns exactly `policy_logits`' values.
- **The copied head.**
  - It equals main/4 up to the constant −bias/4, and has the same softmax.
  - KL < 1e-6 with no bonus.
  - With a bonus on the visited moves only, the KL is positive and the never-visited move gets a nonzero gradient (plan section 2).
- **Gradient scaling.** At multiplier 0.5 the trunk gradient is half that at 1, while the head's gradient is identical.
- **State.**
  - JSON round trip: the head, Adam and origin are restored exactly.
  - Generation, corruption, fresh-only-at-adoption and S4 guards.
  - The multiplier sequence is 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.
- **Inside `ppo_update`.** The head steps on every teacher batch and nothing reaches the trunk while head-only.
- **`TeacherPhase`.** One iteration's payload carries the head into the next `prepare`, and the validation-table soft KL moves while the policy's KL does not.
- **Recipe.**
  - Adopted once, and `is_on` still recognizes the recipe.
  - Forward-only `signature_transition`.
  - Refused without the g3270 revision; a changed block fails validation.

The g3270 (18), g3318 (6), g3140 (11), teacher-on Stage A, B07 shadow-resume and search-head-refresh suites still pass. Three `test_teacher_phase_revision.DistinctUsageTests` tests fail with `V11 link topology differs: CanalBelperDerby` inside `distill_update`'s board-topology check. That path does not reach this change.

## Code

| File | Change |
| --- | --- |
| `recovery/soft_policy.py` (new) | Head, copy initialization, hidden capture, soft target and loss, its own gradient call, Adam, warm-up multiplier, JSON state and `carry`, validation-table measurement, report |
| `recovery/training_revision_g3423.py` (new) | The `soft_policy` block: validation, `configure` (needs g3270 and teaching on), `previous` peel, forward-only `project`, `settings`, tag `soft-t4-w8-s1` |
| `recovery/teacher_term.py` | `soft=` argument. `apply` captures the hidden layer, computes the soft loss and its own gradients (`retain_graph` on the teacher's call only at multiplier > 0), adds 8β × the soft gradient beside the teacher's, steps the head and records. `report()['soft']`. The stratified path is refused. |
| `recovery/teacher_phase.py` | Creates the head in `prepare`, measures the validation tables before and after, commits `soft_policy` in the payload (carried if no iteration ran) |
| `recovery/config.py`, `rust_backend/lab_shadow_config.py` | Validation; the peel in `is_shadow`; the projection in `signature_transition` |
| `recovery/league.py` | Log line |
| `scripts/continue-rust-shadow.py`, `continue-rust-g1491.py`, `continue-rust-g1190-restart-2.py` | `--soft-policy`, recipe filename tag, the adoption is journaled as `soft-policy` (pending work archived, as for value revisions), help text |

The execution identity is now `259d39df…` (search auxiliary 2.0.0). Its predecessors `e62799b4…` (S1 with search auxiliary 1.0.0, committed) and `0dcb6b30…` (g3318–g3423) are kept, so the g3423 checkpoint resumes. S1 itself changes no Rust; the search auxiliary adds one CPU-teacher overlay patch, so the continuation rebuilds the CPU teacher once.

## The run at g3423

From the export: full reports for g3419–g3422, plus the retention summaries for g3291–g3418.

- **Recipe.**
  - g3140 items `time, validation, heads, targets`; `ownership-stop` since g3318; `weight=0.15` since g3418.
  - The g3270 teacher revision; ratings every 20 iterations.
  - Deferred: `weight=0.5, vp, rail-end`.
- **Teacher term.**
  - β 0.0312 and share 0.10 (target 0.10).
  - PPO took all 48 steps in every iteration since g3318.
  - The term takes 2.8 s.

**Validation tables, before they train:**

| | g3291–g3317 (27) | g3318–g3417 (100) | g3418–g3422 (5) |
| --- | ---: | ---: | ---: |
| KL to the search target (median) | 0.215 | 0.193 | 0.159 |
| Top-1 agreement | 0.871 | 0.887 | 0.881 |
| `delta.all` (mean ± s.e.) | +0.0098 ± 0.0013 | +0.0077 ± 0.0008 | +0.0087 ± 0.0031 |
| `pull.all` (mean ± s.e.) | −0.0009 ± 0.0013 | +0.0001 ± 0.0008 | −0.0009 ± 0.0034 |

The update still moves the policy away from the search on new positions about as far as it moves at all. That is the problem the soft term (and later the distillation phase) is aimed at.

**Ownership stop, g3318–g3417:**
- The stop decided 50 of 100 iterations (`heldout_ce_increased`), and 43 kept 0 head steps.
- Owner CE falls by about 0.001 per value phase.
- Since `weight=0.15` (g3418–g3422) the CE is 0.967, against a median of 0.982 before.

**PL ratings:**

| Generation | g3310 | g3330 | g3350 | g3370 | g3390 | g3410 |
| --- | --- | --- | --- | --- | --- | --- |
| Rating | 186 | 204 | 204 | 226 | 197 | 210 |
| 95% interval | — | — | — | [197, 261] | — | [173, 252] |

**Iteration time:** median 199 s (PPO 48 s, value phase about 20 s, collection 70–90 s).

## Continue plan

**Leg 1: adopt S1 and run through the warm-up to the g3430 rating (7 iterations).**

This command was never run. It is replaced by the one below, which also adopts the search auxiliary at the same boundary ([G3423_SEARCH_AUXILIARY.md](G3423_SEARCH_AUXILIARY.md)):

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3423 --iterations 7 --share-target 0.1 --soft-policy --value-revision search-aux
```

| Iteration | Multiplier | Soft gradient into the trunk |
| --- | --- | --- |
| g3423→g3424 | 0 | none (head-only) |
| g3424→g3425 | 0 | none (head-only) |
| g3425→g3426 | 0.2 | ramp |
| g3426→g3427 | 0.4 | ramp |
| g3427→g3428 | 0.6 | ramp |
| g3428→g3429 | 0.8 | ramp |
| g3429→g3430 | 1.0 | full |

Ending the leg at the g3430 rating also exercises the resume path when leg 2 starts: a checkpoint that carries `soft_policy`.

What to check:
- **After g3424** (first iteration, head-only):
  - `soft=head_only:0.0`, `soft_moved` above 0, `soft_share=0`;
  - `soft_heldout` after below before on most iterations;
  - the teacher's own numbers in their usual range (β 0.031, share about 0.10, 48 PPO steps);
  - `term_seconds` within about 0.1 s of 2.8 s.
- **If `soft_moved` stays 0:** stop. The head's gradient path is broken.
- **During the ramp (g3426–g3430):**
  - `soft_share` rising toward about 0.04;
  - PPO still at 48 steps with no `kl_early_stop`, and `epochs[*].probe_categorical_kl` at 0.006–0.008 against 0.01;
  - `term_seconds` up by at most about 0.5 s.

**Leg 2: 20 iterations at full weight, to the g3450 rating.** The setting is stored with the run, so no flag is needed.

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3430 --iterations 20 --share-target 0.1
```

What to read at g3450:
- **`heldout.soft`:** whether the head's KL on the validation tables keeps falling, which would mean it generalizes beyond the rows it trains on.
- **`heldout.delta` and `pull`** against the g3318–g3417 baseline (+0.0077 and +0.0001).
  - Over 20 iterations their standard error is about 0.0017, because one iteration's spread is 0.008. Only a change of about 0.004 or more will show.
  - S1 is expected to be smaller than that. Its trunk share (about 0.04) is under half the teacher term's 0.10.
- **The ratings at g3430 and g3450** against the 197–226 band of g3330–g3410. Their intervals are about ±40.

These readings are what to watch, not gates. S1's job is to train the head and prove its path, so the later steps can use it.

**After g3450:** the next build steps, from the plan.

| Option | What it does | Cost | Code / timing |
| --- | --- | --- | --- |
| S4 | PPO rows join through the one-visit target (factor 1) | About +1 s per iteration | `updates.py` only. It can be built while leg 2 runs and adopted at g3450. |
| Distillation phase (build step 4), then S2 | The label term moves out of the clipped PPO step into its own phase with a KL budget, and the soft term goes with it | About +0.5 s in the phase for the soft part | The step aimed directly at the zero pull. It is the larger change. |
| Deferred value items `weight=0.5`, `vp`, `rail-end` | — | No extra time | Best adopted one at a time, after S1's ramp (from g3430), so the soft numbers and the auxiliary numbers do not change in the same iterations |
