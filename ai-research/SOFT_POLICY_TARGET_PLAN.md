# Auxiliary soft policy target (KataGo style): plan

- **Date:** 2026-11-18. **Run:** `joint7-v11-2-g1190-restart-3` at g3318.
- **Evidence:** the g3318 export: full reports for g3314–g3317, plus the g3270–g3313 summaries from its retention history. Also the training code and the three references at the end.
- **Scope:** a design. No code or recipe was changed. Numbers marked *estimate* are derived; all others are read from the reports.
- **Status (2026-11-18):** S1 is implemented for adoption at g3423 with `--soft-policy` (`recovery/soft_policy.py`, `recovery/training_revision_g3423.py`). What was built, what to read and the continue plan are in [G3423_SOFT_POLICY_S1.md](G3423_SOFT_POLICY_S1.md). S2–S4 are not built.
- **Revision (same day):**
  - S1 now trains the head on the 128 long-search label rows of each PPO step only; the PPO-game target moves to a later step, S4.
  - The gradient on unvisited moves is corrected (section 2).
  - The head's gradient path and a warm-up test are specified (sections 3 and 10).

## Summary

- **What.** Add a second policy output, used only in training, that predicts the policy target raised to the power 1/4 and renormalized (temperature T = 4). It is weighted 8 times the policy loss of the same rows. As in KataGo, it shares the whole policy path and adds only a last layer: one `Linear(192→1)` without bias on the policy scorer's hidden layer, 192 parameters. It is not exported and never chooses a move.
- **S1 trains it on search games only.** Each PPO step's soft loss uses the 128 long-search label rows the teacher term already draws, with target `softmax((current logits + search bonus) / 4)`. PPO rows wait for S4.
- **Why.** On the validation tables the policy already picks the search's top move 88% of the time, yet its KL to the search target is 0.21, and each update raises that KL on new positions by 0.010. A sharp target trains mostly the top move. The soft target trains how the search ranks the other moves, and how much probability they hold together.
- **The head's gradient has a path of its own.** The teacher term differentiates an explicit list of model parameters, and the head is not on it. S1 therefore computes the head's gradient in a separate call. A test shows that during warm-up the search rows change the head and nothing else (section 10).
- **Later steps.**
  - S2: the search part moves into the distillation phase (plan step 4).
  - S3: short-search labels for the soft head only (your decision).
  - S4: PPO rows, through a one-visit target.
- **Cost of S1:** about 0.5 s per iteration (*estimate*). No search, CPU-worker, lab-native, export or held-out change.
- **What to expect.** Inside the PPO step the soft term is as weak as the teacher term is today: about 0.04 of the trunk gradient (*estimate*). S1 trains the head and proves its path; S2 is where the search part can matter.

## 1. What the references do

| Source | Target | Weight | Head | Reported effect |
| --- | --- | --- | --- | --- |
| [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#auxiliary-soft-policy-target), in main runs since March 2023 (v1.12.0) | The policy target raised to 1/T, T = 4, renormalized. The code adds 1e-7, masks, takes the 0.25 power and renormalizes ([metrics_pytorch.py](https://github.com/lightvector/KataGo/blob/master/python/katago/train/metrics_pytorch.py)). | 8 × the policy loss, with the same per-sample weight. The 8 makes up for the small gradients of a softened target once it is roughly fitted. | One more output channel of the policy head's last 1×1 convolution; every earlier policy layer is shared ([model_pytorch.py](https://github.com/lightvector/KataGo/blob/master/python/katago/train/model_pytorch.py)). | No numbers. The doc says it speeds up policy learning considerably and also helps the main policy head, because the net must learn the relative weight of the weaker moves. Thanks to policy-target pruning, those weights carry real search information. |
| [Chessformer](https://arxiv.org/html/2409.12272v2), appendix B | The policy target at temperature 4 | 8; still small next to the main policy loss, since the temperature mutes it | A soft policy head | No ablation |
| [Sayuri v0.6.0](https://github.com/CGLemon/Sayuri/releases/tag/v0.6.0) | The release notes list soft policy among new output targets | — | — | No details in the notes |

In KataGo and the Chessformer paper it is a training target only; play uses the ordinary policy.

## 2. Why it fits this run

| Number (g3270–g3313, 44 iterations, validation tables before they train) | Median |
| --- | ---: |
| Top-move agreement between the policy and the stored search target | 0.88 |
| KL from the search target to the policy, before the update | 0.213 |
| Change of that KL across the iteration's update (`heldout.delta`) | +0.010 |
| Pull toward the search on these new positions (movement minus the KL change) | −0.0013 |

- **The search's knowledge does not reach new positions.** The policy agrees with the search's best move most of the time. Even so, each update moves it as far from the search as it moves at all: there is no pull. The g2910 review found the same before g3270, with a pull of about 0.0006.
- **A sharp target trains the top move and little else.** Under cross-entropy, a move with almost no target mass gets almost no gradient beyond "lower". The soft target asks the net to say how much the search likes the second, fifth and eighth move as well.
- **The search does have that information.** Each long search spreads 640 simulations over 8 Gumbel-sampled candidates (27 to 187 visits each). A fast search spreads 96 over the same 8. Gumbel top-k sampling includes moves the prior ranks low, and the label's completed values give every legal move a mass.

### What the soft target asks of unvisited moves

The policy is sharp: entropy 0.53 nats on PPO decisions, over a median of 73 legal moves (p95 214). T = 4 lifts its long tail, so about three quarters of a soft target's mass sits on moves the search never visited.

- **Their order among themselves is already right.** All unvisited moves share one completed value, the mixed value, so they share one bonus B. For each of them the target is exp((z + B)/4)/Z_t and the copied head's prediction is exp(z/4)/Z_p. The ratio target/prediction is the same constant c for every unvisited move of a row, so the head starts with their relative order exactly right.
- **Their total is not, so they do get a gradient.** c equals 1 only when the visited moves' bonuses leave the unvisited group's share unchanged. Whenever the search rates its candidates above or below the mixed value, softmax normalization moves probability between the visited and the unvisited group.
  - Each unvisited move then gets the gradient (1 − c) × its soft probability. They all move together, in the same direction.
  - Their gradients reach the trunk through every unvisited candidate's hidden vector.
  - What they get no signal about is their order among themselves.
- **Size, from a synthetic check shaped like this run.** Entropy 0.54 nats over 73 moves; 8 candidates with 27, 80 and 187 visits; candidate values spread 0.05; 4,000 rows.
  - c differs from 1 in 99% of rows (1 − c: median 0.03, p10 −0.08, p90 0.20).
  - Unvisited moves hold 0.745 of the copied head's mass against 0.725 of the target's (medians).
  - They carry a median 35% of the gradient's L1 norm (p10 11%, p90 43%), but only 2% of its squared L2 norm, spread thin over about 65 moves.
- **The visited moves carry the rest:** their ranking among the 8 candidates, and the other side of the shift between the two groups. That part is muted by the temperature, which is what KataGo's weight of 8 compensates for.
- **Why T stays 4.** A lower T would give the visited moves more of the mass, but would weaken the emphasis on the lower-ranked ones. T = 4 keeps KataGo's balance and the meaning of its weight.

## 3. The head

- **Shape.** The policy scorer is `Linear(512+160→192) → GELU → Linear(192→1)` per candidate move. The soft head is a second `Linear(192→1)` on the same GELU output, without a bias. A constant added to every move changes no softmax, so a bias would only collect rounding noise: its gradient is the sum of the row's gradients over the legal moves, which is zero. KataGo's last policy layer has no bias either. This is KataGo's arrangement: everything is shared except the last layer, so the shared features must serve both readouts. A head with its own hidden layer could fit the soft target without changing those features, which would defeat the purpose.
- **Initialization.** At adoption the head copies the main last layer's weights (`policy_scorer.2.weight`) divided by 4, and drops its bias. Its first output is therefore `softmax(main logits / 4)`.
  - A target with no search bonus then has KL 0 at once.
  - The first gradients come only from where the search disagrees with the policy: the candidates' ranking, and the share the unvisited moves hold (section 2).
  - A zero or random start would instead spend its first iterations pushing arbitrary gradients into the trunk.
- **Ownership.** The head belongs to the trainer, like the era heads. It is not added to `FourSeatNet` and is not in the ONNX export or the CPU workers. Its weights, its Adam state and its warm-up counters are persisted in the teacher phase's checkpoint payload, beside the term's β state (`weight`).
- **Forward.** A trainer-side function returns the main logits and the scorer's hidden layer. It uses the same operations, in the same order, as `teacher_term._score_policy` (`encode_state`, `prepare_candidates`, `score_prepared`). The main logits stay bit-identical, and the soft logits come from the same hidden tensor, so there is no extra pass.
- **Gradient path.** The teacher term's `apply` builds an explicit tuple of actor parameters (names starting with `ACTOR_PREFIXES`) and calls `torch.autograd.grad(loss, parameters)`. The head is not in that tuple, and `autograd.grad` fills no `.grad`. A soft loss simply added to the teacher's loss would therefore never reach the head: it would stay a copy for ever, silently. So:
  - **Its own gradient call.** The soft loss gets a separate call over the head's parameters, plus the actor parameters once the trunk multiplier is above 0. The teacher's own call keeps the graph for it (`retain_graph`) only in that case.
  - **The teacher term is unchanged.** Its gradient, loss, share and β are computed exactly as today.
  - **The head's own step.** Its gradient goes to the head's own Adam (learning rate 1e-4 as for the era heads, no weight decay, gradient clip 1.0). That Adam steps inside the same `apply`, and only when the teacher term applies gradients (mode `on`).
- **Warm-up (the local-scoring rule).** For the first 2 iterations the trunk multiplier is 0. The multiplier then rises to 1 over 5 iterations: 0.2, 0.4, 0.6, 0.8, 1.0.
  - **Multiplier 0** is done by detaching the hidden layer before the head. No soft gradient can reach the shared scorer or the trunk, and the head-only iterations cost no extra backward pass.
  - **Between 0 and 1,** `scale_gradient` scales the hidden layer's gradient. At 1 the hidden layer is used as is.

## 4. S1 target and weight

`soft target = softmax((current logits + search bonus) / 4)`

- **Rows.** The teacher term's drawn rows only: 128 long-search labels per PPO step, no repeats inside the update, age-weighted from the 8-iteration pool, 6,144 presentations per update. Validation-table rows train from the next iteration, as for the main term. No PPO row enters the soft loss in S1: the PPO-row factor is 0.
- **Target.** This is exactly the rebuilt label target raised to 1/4 and renormalized, KataGo's formula applied to this run's target. No epsilon is needed, because every legal move already has mass.
  - The bonus is the staged vector `search_bonus` builds today: label scale 0.20 × (50 + most visits) × completed value.
  - The current logits are those of the same forward pass, detached. Illegal moves stay at zero.
- **Validation report.** For this report only, the soft target is the stored search target raised to 1/4, so that the before and after measurements share one fixed target.
- **Loss.** The mean KL from the soft target to the soft head over the 128 rows; its gradient is the same as cross-entropy's.
- **Weight.** As in KataGo, 8 × the policy loss already on these rows. The teacher term weighs its label KL by β (0.030 at g3317), so the soft term weighs **8β**, and it follows β if β is re-measured.
- **Rough size.** About 0.04 of the PPO actor gradient on the shared encoders (*estimate*), half the teacher term's 0.088.
  - The temperature divides the soft gradient by about 16 (once in the target's logit differences, once in the head's copied weights), and the 8 restores half of it.
  - The report measures it on the same sampled steps as the teacher's share (every fourth step).
  - It enters neither the teacher's share, nor its β qualification, nor its adaptive β.

Every PPO step is clipped today (48 of 48, gradient norm median 4.6 against a clip of 1.0), so the soft term takes part of the step's direction, not more step. PPO's learning-rate controller already measures KL after the whole step, teacher included, and it will see the soft term the same way.

## 5. S1, step by step

Inside `TeacherTerm.apply`, for each PPO step's batch of 128 staged rows:

1. **Forward** through the hidden-returning function: main logits (bit-identical to today) and the hidden layer.
2. **Teacher term as today:** rebuilt target, label KL, its own gradient call over the actor parameters, β, its norm and share.
3. **Soft target** from the same logits and the same staged bonus, detached.
4. **Soft prediction:** the head on the hidden layer. The hidden layer is detached at multiplier 0 and gradient-scaled between 0 and 1.
5. **Soft loss:** the mean KL over the 128 rows.
6. **Soft gradients:** one call over the head's parameters, plus the actor parameters when the multiplier is above 0.
7. **Apply (mode on):**
   - add 8β × the soft actor-parameter gradients to `.grad`, beside the teacher's;
   - clip the head's gradient to 1.0;
   - step the head's Adam.
8. **Record:**
   - the soft KL at the first and last step;
   - the soft trunk norm and its share on the measured steps;
   - the head's distance from the copied layer divided by 4;
   - the multiplier.

On the 4 validation tables, the soft head's KL to the softened stored target is measured before and after the update, in the forwards the teacher's held-out measurement already runs.

Nothing else changes: the PPO microbatches, collection, the value phase, the ownership stop, the search-value head and rating. The soft term stops when the teacher term's batches stop, including when PPO stops early.

The soft term runs on the staged, long-only path this run uses. The stratified `long-and-short` path of g1491 is refused while the soft term is on; S3 adds short labels in its own way.

## 6. S2: with the distillation phase (plan step 4)

- **What moves.** When step 4 takes the label term out of PPO, the soft term goes with it. Each phase step's loss becomes `label KL + 8 × soft KL` on the same 512 rows. The phase's own parameter list does not contain the head either, so the same separate gradient call applies. The head keeps its Adam.
- **The KL budget.** The budget is measured on the main policy (the PPO probe rows). The soft term moves the main policy only through shared features, so it spends part of the budget, and the phase's learning-rate multiplier settles where the total lands on the budget. That is the trade: some distillation movement goes to features for the weaker moves, which KataGo reports helps the main head learn.

## 7. S3: short-search labels for the soft head only

The 1,967 fast searches of each iteration choose moves but leave no label (g3317). At 1.94 labels per searched turn they would give about 3,800 labels, three times the 1,228 long labels (*estimate*).

Your g3270 decision keeps short labels out of policy training. This item keeps them out of the main term: they feed only the soft head.

- **Recording.** Use the g1491 short-label path (`long-and-short`) in a separate stream that the main teacher term never draws from.
- **Target.** Recompute at label scale 0.20 (the g2910 review's section 2.2), then soften as in S1.
- **Pool.** Keep 4 iterations, not 8: the labels are noisier, and this halves the memory.
- **Rows per step.** Draw 128 short rows per step beside the long ones (in the PPO step), or add them to the phase's batches with label-term weight 0 (S2).
- **Weight.** The same 8β (8 in the phase).

## 8. S4: PPO rows, through a one-visit target

PPO games join once S1 runs. A PPO game has nothing beyond the policy except one fact per decision: how the chosen move turned out. Treat that as a search with one visit and apply the search's own formula.

| Field | Search label | PPO decision |
| --- | --- | --- |
| Visits | counts from the search | 1 on the chosen move |
| Value of a visited move | mean return of its simulations | V(s) + Â: the frozen critic's value plus the GAE advantage, before PPO normalizes it (utility units) |
| Mixed value (unvisited moves) | from the visited moves and the prior | (V(s) + V(s) + Â) / 2 |
| Bonus | 0.20 × (50 + most visits) × completed value | 0.20 × 51 × completed value |
| Chosen move minus any other | — | **5.1 × Â** |

`soft target (PPO row) = softmax((current logits + 5.1 × Â on the chosen move) / 4)`

- **The one-visit scale is the evidence weighting.** In the units of the g2910 review (section 2.2), the bonus per unit of value is:

  | Source | Bonus per unit of value |
  | --- | ---: |
  | Long search | 47 |
  | Short search at label scale 0.20 | 16 |
  | PPO decision (one visit) | 10, i.e. 5.1 per unit of advantage |

- **Advantage noise averages out.** The advantage's spread is 0.21 at g3317, so the tilt on the chosen move is about ±1.1 logits, or ±0.27 after softening. That is the noise of one game. Each update presents 47,800 decisions twice.
- **Rows covered.** All nonforced learner decisions. Moves played under the opening temperature or cotton exploration keep their tilt; forced decisions are skipped.
- **Weight.** 8 × the PPO surrogate's weight of 1, times the PPO-row factor: 0 in S1, 1 in S4. The factor can be lowered on its own. Rough trunk share: 0.01–0.04 (*estimate*).
- **What PPO rows add.** The head sees PPO's whole state distribution, about 95,600 presentations per update against 6,144 label presentations, plus a small softened push on the chosen move. They cannot teach the ranking of weak moves, because PPO rarely plays them; only the search rows can.
- **Not used: a softened behaviour policy.** At the start of the update it equals the policy itself. The head matches it by dividing its readout by 4, so the trunk learns nothing.
- **Code.** The PPO microbatches use the hidden-returning forward, keep the raw GAE advantages before `normalize_advantages` overwrites them, and add the PPO-row soft term to the same backward.

## 9. Cost

| Item | Time per iteration | Search / CPU | Memory | Held-out data | What changes in the stack |
| --- | --- | --- | --- | --- | --- |
| S1 | About +0.5 s (*estimate*). The teacher term's 48 steps take about 0.5 s today, beyond its one-time 2.5 s staging. S1 adds a 192→1 layer on 128 rows per step (192 of the policy path's 83,232 multiply-adds per candidate, 0.23%), a KL over the legal moves and, after the warm-up, a second backward over the same rows. Under 0.1 s while head-only. | none | a few MB on the GPU; 192 parameters plus Adam in the checkpoint | none | trainer Python only: no model, export, CPU-worker, lab-native or bundle change; execution identity re-recorded |
| S2 | About +0.5 s inside the phase (*estimate*) | none | none | none | part of step 4 |
| S3 | About +3–4 s for 128 more label rows per step (the teacher term's rows cost 3.0 s today, staging included). About +63 helper CPU-seconds to encode 3,800 rows (16.6 ms each, 4 helpers), mostly overlapped with the tables' own search (*estimate*). | none: the fast searches already run | about +1.7 GB host memory for 4 iterations at about 110 KB per row; about +0.9 GB GPU for the staged rows (916 MB per 6,144 rows today) | none | recording switch, soft-only label stream, sampler |
| S4 | About +1 s (*estimate*): the extra layer and KL on 95,600 PPO presentations, and the PPO-row share on sampled steps. Today PPO's forward-backward takes 15.3 s. | none | a few MB | none | `updates.py` |

## 10. Build order and tests

| Step | Change | Time per iteration | Needs |
| --- | --- | --- | --- |
| S1 | The head, trained on the teacher term's 128 long-label rows per step; warm-up; reports. Adopted with `--soft-policy` at a committed boundary. | about +0.5 s | Python only |
| S4 | PPO rows join through the one-visit target (PPO-row factor 1) | about +1 s | S1 |
| S2 | The soft term moves into the distillation phase with the label term | about +0.5 s in the phase | plan step 4 |
| S3 | Short-search labels feed the soft head only | about +3–4 s, +1.7 GB host | S1; best once S1's warm-up is over (7 iterations) |

- **Adoption of S1** follows the g3270 pattern:
  - a top-level `soft_policy` block: T 4, weight 8, rows `long-labels`, PPO-row factor 0, head copied from `policy_scorer.2.weight` / 4 with no bias, Adam 1e-4, warm-up 2 + 5;
  - a revision module named for its adoption generation;
  - the `is_shadow` peel and a forward-only projection;
  - the flag, set once and stored with the run.
- **Unchanged at adoption:** labels, buffers and β.
- **The way back** is the checkpoint before adoption, as for the other revisions.

**Tests for S1:**

| Test | What it proves |
| --- | --- |
| **Search rows train the head during warm-up.** See the setup and checks below. | The head's separate gradient path works, and warm-up leaves the model's gradients untouched. |
| Hidden-returning forward | Main logits bit-identical to `_score_policy` on the same rows. |
| Copied head | KL 0 on a target with no search bonus. Positive KL and a nonzero gradient on unvisited moves when only visited moves get a bonus (section 2). |
| Gradient scaling | At multiplier 0.5 the soft gradient on the actor parameters is half the gradient at multiplier 1, to rounding. |
| Checkpoint | Round trip of the head, its Adam and its warm-up counters; resume continues the ramp. |
| Recipe | `--soft-policy` validates, persists and only moves forward; `is_shadow` still recognizes the recipe. |

**Search rows train the head during warm-up.**

Setup:
- the fixture model and a teacher term with staged long-label rows whose visited moves have different completed values, so the copied head does not already fit;
- soft term on, PPO-row factor 0, trunk multiplier 0, mode on;
- one `apply` step, compared with the same step with the soft term off.

Checks:
1. The head's weight has a nonzero gradient, and the weight differs after the head's Adam step. The head has no bias to check (section 3).
2. The gradient accumulated on every model parameter (`policy_scorer.0`, the action encoder, the entity projections, the state and entity encoders) is bit-identical to the soft-off step.
3. The teacher's own loss, gradient norm and share for the step are identical with the soft term on or off.
4. Control: at multiplier 1 the gradient on `policy_scorer.0` differs from the soft-off step, so the test can see a soft gradient when there is one.

## 11. What to read and watch

| Field | Today | What it shows |
| --- | --- | --- |
| `teacher_phase.term.soft.stage`, `multiplier` | new | head-only, ramp or full |
| `teacher_phase.term.soft.kl` (first and last step) | new | how well the head fits the label rows inside an update |
| `teacher_phase.term.soft.share` | new | the trunk gradient the soft term supplies (*estimate* about 0.04 once the multiplier is 1) |
| `teacher_phase.term.soft.head_distance` | new | how far the head has moved from the copied main layer / 4; it should leave 0 in the first head-only iteration |
| `teacher_phase.heldout.soft` before/after | new | the soft head on positions it has not trained on |
| `teacher_phase.heldout.delta.all`, `pull.all` | +0.010, −0.0013 | The point of the change: the update should stop moving the policy away from the search on new positions. Delta should fall and pull should rise. |
| `teacher_phase.heldout.before.all.kl`, `top1` | 0.213, 0.88 | the slower trend |
| `actor.optimizer_steps`, `kl_early_stop`, `epochs[*].probe_categorical_kl` | 48 of 48, no stop, 0.006–0.008 against 0.01 | whether the soft term pushes PPO into its KL stop |
| PL rating (every 20 iterations) | g3290: 186.7; g3310: 204.0 (95% interval 158–256) | strength |

| Risk | Sign | Response |
| --- | --- | --- |
| The soft term moves the main policy | PPO stops before 48 steps, or the probe KL per step rises | Unlikely at a share of about 0.04. If it happens, go back to the checkpoint before adoption, as for the other revisions. |
| The head stays a copy | `head_distance` at 0 after the first iteration | The gradient path is broken; the warm-up test exists to catch this before adoption |
| One-visit targets add mostly noise (S4) | The PPO-row share well above the search share for many iterations, while `heldout.soft` does not fall | Lower the PPO-row factor below 1 |
| Short labels too noisy (S3) | Soft KL on the validation tables' short rows rises | Label scale 0.15 for them, as the g2910 review suggests |

## 12. Where it touches the code

| File | Change |
| --- | --- |
| `recovery/soft_policy.py` (new) | Head, copy initialization, the hidden-returning forward, the soft target and loss, warm-up counters and multiplier, Adam, persisted state, report |
| `recovery/teacher_term.py` | `apply`: hidden-returning forward; the soft loss on the drawn rows; a separate gradient call over the head's parameters (and the actor parameters above multiplier 0), with `retain_graph` on the teacher's own call only then; 8β into `.grad`; the head's step; the measurements. `heldout_measure` adds the soft KL. The stratified path is refused while the soft term is on. |
| `recovery/teacher_phase.py` | Create the head at adoption, carry it in the term's state and checkpoint, pass the settings |
| revision module (new), `recovery/config.py` | The `soft_policy` block: validation, configuration, the `previous` peel, forward-only projection, filename tag, summary |
| `rust_backend/lab_shadow_config.py` | Peel in `is_shadow`, projection in `signature_transition` |
| `recovery/league.py` | Log line: soft KL, share, stage |
| `scripts/continue-rust-shadow.py`, `continue-rust-g1491.py`, `continue-rust-g1190-restart-2.py` | `--soft-policy` |
| tests | Section 10 |
| S4 only: `recovery/updates.py` | PPO microbatches with the hidden-returning forward, raw advantages kept, the PPO-row soft term |

Not touched: `learning/model.py`, the export, the CPU workers, `lab-native`. The execution identity is re-recorded last.

## 13. Not chosen

| Option | Why not |
| --- | --- |
| Adding the soft loss to the teacher's loss in one gradient call | The call lists the actor parameters only; the head would never train. |
| Warm-up by multiplying the gradient by 0 | Detaching gives the same result with certainty, and costs no backward pass. |
| PPO rows from the start | Your call for S1. Their one-visit signal is weak and repeats PPO's own; S4 adds them once the search path is shown to work. |
| Softened one-hot of the chosen move (for S4) | A one-hot stays a one-hot at any temperature. It is behaviour cloning of the run's own samples. |
| A soft head with its own hidden layer | It could fit without shaping the shared features. KataGo shares everything but the last layer. |
| Exporting the head or using it in search | KataGo and the Chessformer paper use it only in training. Exporting would change the ONNX contract and the CPU workers. |
| Training it in the value phase | The value phase's rows carry one candidate, not the legal moves. Its trunk steps are undone whenever an interval is rejected, which kept 0 steps in 17 of 39 iterations (g3275–g3313). |
| Raising the soft weight inside the PPO step until it matters | The g2910 review's section 3: a larger share inside the clipped PPO step takes movement from PPO. The distillation phase's budget (S2) is the place for more weight. |

## 14. Estimates and open points

- The trunk share in section 4 assumes small target differences and the copied head. The report measures it from the first iteration at multiplier 1.
- The unvisited-move numbers in section 2 come from a synthetic check, not from the run's labels. The structure holds exactly; the percentages depend on how sharp the prior is and how far the candidates' values spread.
- No source gives a number for the effect. KataGo tested it at the end of 2022 and put it into its main run's nets in v1.12.0; the Chessformer paper has no ablation.
- S2's share of the KL budget is not known until step 4 runs.
- **A later source:** labels on positions of the ordinary games, from the g2910 review's section 10. A short search on sampled PPO positions would give PPO games real search targets for both heads.

## References

- KataGo methods, "Auxiliary Soft Policy Target": https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#auxiliary-soft-policy-target
- KataGo training code: `python/katago/train/metrics_pytorch.py` (soft target and loss) and `model_pytorch.py` (policy head outputs)
- D. Monroe, P. A. Chalmers, *Mastering Chess with a Transformer Model*, arXiv:2409.12272, appendix B: https://arxiv.org/html/2409.12272v2
- Sayuri v0.6.0 release notes: https://github.com/CGLemon/Sayuri/releases/tag/v0.6.0
- The completed-value improved policy used by this run's search: [GUMBEL_IS_MCTS_REFERENCE.md](GUMBEL_IS_MCTS_REFERENCE.md); label scale and rebuilt targets: [TRAINING_IMPROVEMENT_PLAN_G2910.md](TRAINING_IMPROVEMENT_PLAN_G2910.md) sections 2.2, 2.3 and 3; [G3270_TEACHER_STEPS_2_3.md](G3270_TEACHER_STEPS_2_3.md)
