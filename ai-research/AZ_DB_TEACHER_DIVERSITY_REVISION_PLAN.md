# AZ_db-inspired diversity for the Rust search teacher

Date: 2026-10-30 (America/Chicago).

Status: **Design only.** This document revises the completed B02.5, B03 and B04 implementations for a later teacher activation. It does not enable teaching, alter the ongoing backend migration, or authorize a training campaign. DOMiNO ordinary training is already implemented; reuse it.

## 1. Scope and relationship to existing work

Implement a strategy-conditioned, diversity-aware **policy-only** teacher using ideas from AZ_db. Keep the existing PPO/DCPG/PPG trainer, DOMiNO objective, Rust Gumbel IS-MCTS and four-seat hidden-information rules.

The inputs to this revision are:

- [Teacher search phase revision plan](TEACHER_SEARCH_PHASE_REVISION_PLAN.md), including its implemented T1/B02.5 and T2/B04 contracts.
- [Migration plan B02.5, B03 and B04](RUST_AI_BACKEND_MIGRATION_PLAN.md#b025--revise-the-teacher-distillation-phase).
- [B02.5 report](RUST_BACKEND_B025_REPORT.md), [B03 report](RUST_BACKEND_B03_REPORT.md) and [B04 report](RUST_BACKEND_B04_REPORT.md).
- [DOMiNO implementation](DOMINO_DIVERSITY_IMPLEMENTATION.md) and [its design](DOMINO_DIVERSITY_TRAINING_PLAN.md).

Treat B01-B04 as implemented foundations. Their existing receipts prove their original scope; they do not automatically qualify the changed teacher objective. The checked-in reports still distinguish local implementation from Linux/GPU qualification. At implementation time, use the actual current receipts rather than assuming either that all server gates passed or that none passed.

This document supersedes the teacher's neutral-only restriction and its task-only search objective **only for the new, explicitly versioned teacher profile**. The ordinary DOMiNO recipe, production game search, old teacher profile and historical records retain their existing meaning. The earlier DOMiNO plan's decision to leave teaching unchanged remains valid for the old profile.

The four adopted ideas are:

| Idea | Project implementation |
| --- | --- |
| A policy conditioned on a strategy identifier | Reuse the seven existing DOMiNO contexts; keep one identifier per seat for the entire teacher game. |
| Diversity measured from actual play | Reuse the existing 448-dimensional strategic occupancy and its frozen nearest-centroid direction. |
| Diversity inside planning | Include root-player intrinsic return in search evaluation and backup, with a qualified continuation estimate. |
| Conditional imitation of search | Distill each improved action distribution into the policy under that label's original context. |

Do not add a separate policy population, seven weight copies, chess-style PUCT, a new matchmaker, max-over-latents selection, an oracle, or a search over all seven strategies at every position. Do not replace DOMiNO's adaptive quality coefficients with AZ_db's fixed coefficient or add a second diversity objective/controller.

## 2. Paper basis and adaptation boundary

Primary source: Zahavy et al., [*Diversifying AI: Towards Creative Chess with AlphaZero*, arXiv:2308.09175v2](https://arxiv.org/html/2308.09175v2#S3), especially Section 3, equations 4-8 and the search description immediately after equation 8.

The paper conditions shared policy, task-value and intrinsic-value outputs on a latent identifier. Behavioral diversity uses occupancy features. Search combines task and intrinsic value and accumulates the planning player's intrinsic rewards along simulated paths. Its reference player uses task reward only. The root latent remains fixed throughout a search. Policy training imitates search distributions; the paper also trains value outputs. These are the borrowed mechanisms. Its attractive/repulsive distance objective, fixed mixing coefficients, alternating two-player treatment and multi-player-selection procedure are not transplanted here.

All Brass-specific choices below are engineering adaptations, not paper results. In particular, four-seat absolute utilities, causal insolvency accounting, DOMiNO's episodic normalization, bounded scalar-value continuation, Gumbel improved-policy labels and policy-only teacher loss require independent tests. No chess Elo or puzzle result predicts the benefit in Brass.

## 3. Current implementation findings

These are observations from the checkout inspected on 2026-10-30, not instructions to undo concurrent updates.

| Component | Existing behavior | Required delta |
| --- | --- | --- |
| `recovery/teacher_phase_config.py` | `teacher-phase/1.0.0`; off/shadow/on; 28 train + 4 held-out tables; fixed search profile | Add a new profile with explicit context and objective contracts; keep v1 readable. |
| `teacher_phase.py`, `lab_teacher.py` | Separate teacher jobs, frozen current learner, dedicated seed cursor, separate barrier | Bind per-seat contexts and a frozen diversity evaluator bundle to every job. |
| `lab-native/src/teacher.rs` | All teacher contexts are zero; labels also write zero style/intent | Use the assigned seat context for real play and label it accurately. |
| `lab_teacher_inference.py` | Explicitly rejects non-neutral context; exports logits and four-seat utility | Admit verified contexts and a separate read-only intrinsic prediction request. |
| `teacher_labels.py`, `teacher_buffer.py` | Neutral-only validation; admission and cache reload both encode zero context | Validate and reconstruct the stored context in both paths. |
| `teacher_term.py` | Actor-only KL inside PPO; no teacher critic graph | Preserve this training boundary and use the correctly encoded conditional rows. |
| B04 core overlay | Export-only hooks; selection and backup unchanged | Add a separately fingerprinted teacher objective extension; stop describing this extension as export-only. |
| `domino_features.py` | 112 public features per phase; 448-dimensional occupancy | Port the exact event calculation and normalization, not an approximate industry-count bonus. |
| `domino_state.py` | Committed centroids, six dual logits, readiness, nearest-repulsive directions | Read and freeze these inputs; teacher games must never advance them. |
| `domino_auxiliary.py` | Training-only scalar intrinsic head for the acting observer, with PPO/PPG Adam and rollback | Reuse its weights read-only; add target-version/retained-fit metadata before search can consume it. |
| `lab_continuation.py` | Explicit restoration checks for base/PPG/local-scoring/teacher state | Add explicit DOMiNO restoration and comparison. The inspected `restore()` does not instantiate `Diversity`; hashing raw checkpoint parts alone is insufficient. |

Two details prevent a simple "replace neutral with focus" patch:

1. The intrinsic head predicts **one acting observer's continuation**, not four seats' intrinsic returns. An opponent's leaf prediction cannot be used as the root player's intrinsic value.
2. The head is fitted to a frozen direction from an earlier incoming DOMiNO state. The checkpoint also contains subsequently advanced centroids. Pairing the saved scalar head with those newer directions silently changes what its output means.

Address both before enabling diversity-aware teacher updates. Context-only search is a useful separately identified control, not completion of this plan.

## 4. Preserve the existing teacher and training budgets

| Setting | Required behavior |
| --- | --- |
| Activation | Default `off`; later qualified `shadow`, then explicit `on`. No automatic activation during migration. |
| Teacher tables | 28 training + 4 whole-game held-out tables per enabled iteration. |
| Seats | Four copies of the frozen incoming learner weights, with independently assigned contexts. |
| Reserved turns | Existing four phases and one reserved turn per seat per phase. |
| Phases | Canal rounds 1-4 / 5-8; Rail rounds 1-6 / 7-8. |
| Search | Gumbel IS-MCTS selection machinery, K16, c1=50, c2=0.1, T1 priors and Gumbel draws. |
| Resources | One session, one thread, one pending path; existing CPU/GPU admission. |
| Time | One 4-second whole-turn allowance, including intrinsic work, both actions, replans and inference waits. Preserve the 0.4-second finalization reserve and 30-second failure watchdog. |
| Unsearched teacher decisions | Raw conditional T1 policy, epsilon=0, no opening tempering or branches. |
| Labels | Root, validated action-two continuation and replanned root; complete legal-action distribution; completed-trajectory evidence. |
| Buffer | Four collection ages, weights 4:3:2:1; 64 rows per PPO step; existing dedicated sampling RNG. |
| Teacher gradient | Existing shared-encoder share target 0.25, measured against the actual DOMiNO-combined PPO actor gradient. |
| Value learning | `teacher_value.enabled=false`; no teacher rows in any critic or PPG dataset. |

The ordinary recipe is loaded from the actual committed run. Do not regenerate it from historical defaults. Preserve PPO, PPG cadence/epochs/LR/beta, exploration/openings/branching, reward, PFSP, curriculum, ratings, anchors, economic baselines, GAE and snapshot evidence.

No ordinary seed, game count, population admission or controller progress is advanced by a teacher game. Preserve `search_wall_fraction=null` where present; the new phase continues using its own per-turn budget rather than legacy cumulative wall-fraction scheduling.

## 5. Strategy assignment and search opponent assumptions

### 5.1 Actual teacher games

Reuse the existing mapping without changing feature dimensions:

| ID | Context | Probability per neural seat-game |
| --- | --- | --- |
| 0 | neutral | 1/7 |
| 1 | CottonMill | 1/7 |
| 2 | Manufacturer | 1/7 |
| 3 | Pottery | 1/7 |
| 4 | Brewery | 1/7 |
| 5 | IronWorks | 1/7 |
| 6 | CoalMine | 1/7 |

Draw independently with replacement per final physical seat. Keep the result through the whole game, every staged decision and retries. Do not require distinct IDs at a table or change the ID after a bad search.

Use a **teacher-specific** deterministic stream, derived from the run seed, immutable teacher collection identity, split, game identity and physical seat. Reuse the context encoding/validation, not `ordinary_game_index` or the ordinary assignment stream. Avoid circular job hashes: derive a seed identity first, sample assignments, then seal the full job including them.

Store a teacher seat-context receipt containing ID, exact style/intent, encoding contract, model/export identity, physical seat and assignment probability. Use teacher-specific scope rather than forging an ordinary learner receipt. Teacher seats remain ineligible for ordinary training even though all use learner weights.

### 5.2 Initial search assumption: root-latent self-play

For a searched turn owned by seat `r`, freeze that seat's strategy `z_r` throughout the search. Preserve the existing search adapter's use of one root context for simulated actors. Thus all simulated priors use the root model and root context, while the real teacher game's other seats still play their independently assigned contexts on their actual turns.

This follows the root-latent planning idea and avoids exposing the real opponents' strategy assignments to the planner. It is an explicit opponent-model approximation. It does not mean that the four actual teacher seats share a strategy.

Use diversity return for the **root owner only**. Other players' search selection continues to use their own task-utility components and conditional priors. Do not make opponents maximize the root's diversity reward, negate a scalar utility as in two-player chess, or silently switch to actual-opponent-context planning. A richer four-agent planning model is a separate revision.

Keep the existing permitted-information sampler. Search must not access the authoritative game's hidden hands, future deck, private opponent recall or future outcome through the native lane or descriptor callback.

## 6. Freeze a semantically complete evaluator bundle

### 6.1 Retained-fit receipt for the existing intrinsic head

Add versioned metadata alongside the existing DOMiNO state; do not reset or replace the head. Record:

- The exact intrinsic target definition, descriptor manifest, normalization and task-reward regime.
- The direction matrix and nearest IDs actually used to build the fitted MC targets, with their source generation and centroid snapshot hash.
- Retained PPG fit generation, accepted steps and actual fitted rows/whole-game support by strategy and phase.
- The descriptor input convention and fingerprint identities used during fitting.
- The retained head identity and the base encoder identity at that retained state.

Receipt changes participate in the existing PPG keep-best/rollback transaction. An attempted or rejected interval cannot mark the head as newly fitted. A phase with zero retained steps preserves the previous receipt if the previous head is restored. A partly supported new fit must expose missing strategies rather than claim all seven were fitted.

Do not infer the fitting direction from the newest saved centroids. Old checkpoints lack this receipt: they remain fully resumable, but cannot supply a diversity search value until an ordinary accepted PPG fit records it. This is a teacher-readiness limitation, not a reason to reset DOMiNO or perform an extra training phase.

### 6.2 Collection snapshot

At incoming committed generation `G`, freeze:

```text
base model and export                       = actual incoming learner at G
intrinsic head                              = actual saved companion at G
intrinsic reward direction d_search         = direction in its retained-fit receipt
task mixing coefficients c                  = current committed DOMiNO coefficients
descriptor/context/normalization contracts  = admitted contracts
own-seat fingerprints                       = frozen current-model fingerprints by z
```

The head predicts the unweighted intrinsic return; changing `c` does not reinterpret that prediction. Using `d_search` from the last retained fit is deliberately conservative. It may lag the direction used by the current ordinary PPO update. Report this lag and the direction/coefficient differences; never describe the two objectives as identical when they differ.

The current base/head pair can have undergone subsequent DCPG-preserving updates. Record both its current bytes and its last fresh-target fit receipt. Preservation is not proof of exact value invariance or prediction quality.

Give the complete collection bundle its own hash. A base `model_id` is insufficient because the companion, directions, coefficients and fingerprints are outside the base policy identity. Freeze the bundle for the collection, not separately for each worker, action or PPO step. Neither teacher data nor live optimizer activity may mutate it.

Neutral has `c[0]=1` and zero intrinsic contribution exactly. Bypass intrinsic inference and continuation work for neutral roots. Do not approximate neutral with a sigmoid near one.

### 6.3 Readiness

Retain the existing latest-non-null **task utility** readiness rule and its explicit-false behavior. For the diversity profile additionally require:

1. DOMiNO is in `apply` mode and its current committed evidence is ready.
2. The head, receipt, descriptor manifest and reward regime are compatible and finite.
3. A retained fit supplies actual intrinsic training support for all six non-neutral strategies; missing support is reported explicitly.
4. The read-only search inference wrapper and normalization are qualified under this profile.

If these prerequisites are absent, defer the new collection with a precise reason. Existing compatible buffered labels can still be used under the buffer rules. Do not silently change a diversity job into neutral or task-only search, fabricate a fit, or copy readiness across a changed reward/descriptor regime. A missing companion or corrupted receipt is an integrity error, not a normal deferral.

These checks establish a usable, correctly identified estimator. Log its held-out MSE versus a zero predictor, bias and support; do not claim that a retained PPG step proves it is accurate. Empirical adequacy belongs to the bounded qualification and later strength/diversity trial, not a newly invented per-label quality filter.

## 7. Exact event reward and causal normalization

Reuse `domino_features.event()` semantics: public **predecision** state plus the selected main-action family. Do not substitute post-action snapshots. Include all ordinary-equivalent main actions, including a forced main action; payment, bonus, debt and continuation sub-decisions contribute no separate descriptor event.

For phase `p` and root strategy `z`, the reward is:

```text
r_div(s, a; z) = dot(x(s, a), d_search[z, p]) / (4 * N_phase[p])
```

Use the existing family scaling, clipping, slot order and 112-coordinate layout exactly. Do not add the strategy ID, hidden information, final VP/winner, checkpoint identity or an explicit industry quota to the descriptor. The six industry names remain context labels, not required build targets.

The ordinary implementation normalizes by realized complete-game main-action counts. The inspected four-player rules give each player one action in Canal round 1 and two in every later turn, with eight rounds per era. For completed standard games this gives:

```text
N_phase = [7, 8, 12, 4]
```

Before using these counts in search, prove equality with the ordinary replay extractor on complete native/reference games, including pass, scout, insolvency, forced choices and era transitions. This turns the existing full-game normalization into a causal constant for this admitted ruleset without modifying ordinary DOMiNO.

Reject unsupported truncated/variant game semantics at profile admission. Never divide by the number of events seen so far, the simulated prefix length or a hidden future realized count. If the equivalence proof fails, this profile is not ready; a revised normalization would require a separate DOMiNO objective migration.

Record an event once when its chosen main-action edge is successfully applied. Staged cancellation, retries and failed application must not duplicate rewards. Keep a separate intrinsic prefix accumulator; do not overload the monotone, nonnegative insolvency ledger with signed diversity rewards.

## 8. Leaf values and four-seat search backup

### 8.1 Reuse the scalar head without inventing four-seat outputs

The current companion computes a scalar from the encoded state and the acting observer's descriptor embedding. Reuse that exact computation in eval/no-grad mode. The base task-value path continues using its existing unknown seat descriptors.

For the intrinsic call only, construct the root owner's known self descriptor: learner role, lag zero and its frozen current-model context fingerprint. Use the same relative-seat-zero convention as `Diversity.prediction`. Do not replace this descriptor with unknown or feed another seat's fingerprint. Record differences between the fit-time and inference-time fingerprints. Descriptor metadata remains critic-only and does not enter policy features.

At a nonterminal leaf where the root owner is the acting observer, evaluate the scalar head directly. At a terminal leaf, its remaining intrinsic value is zero.

### 8.2 Bounded continuation to a root-owned decision

When the main search leaf belongs to another actor, the scalar head cannot supply the root's intrinsic tail. The initial proposed adapter uses a short, explicit **continuation rollout** on the sampled world:

1. Keep the main search leaf and its task-utility prediction unchanged.
2. From that leaf, sample raw T1, epsilon-zero policies under the frozen root context until the first nonterminal decision owned by the root player, or terminal.
3. Resolve chance through the same permitted sampled world, including authorized hypothetical draws. Do not consult the real future deck.
4. At the first root-owned decision, bootstrap the existing scalar intrinsic head before selecting that decision's action. Terminal contributes zero.
5. This rollout supplies only the root's intrinsic continuation estimate. It creates no teacher labels, tree visits, ordinary rows or plan-support evidence of its own.

Use a separate deterministic search-continuation RNG namespace. Do not consume ordinary-game/action/PPO RNG, and preserve existing Gumbel draw ordering. Keep actual teacher seat assignments fixed; the rollout's root-context opponent assumption is the one defined in section 5.2.

Charge every step, encoding, inference wait and head evaluation to the original work deadline. Bound total traversal plus continuation work by the admitted search depth limit (currently 256 staged steps). If a continuation cannot complete before its bound, discard that unfinished trajectory atomically. Do not count a task-only partial backup as a completed diversity trajectory or bootstrap an opponent scalar as the root scalar.

This is a project-specific way to reuse the existing head, not a claim that the paper used this adapter. It can substantially reduce completed trajectories under four seconds. Its cost is an explicit B04 go/no-go measurement. If it is impractical, leave diversity teaching off and revise the leaf estimator separately; do not hide a shorter-horizon reward under the same contract.

### 8.3 Separate return channels

For any backed-up node `n`, let `L` be the original task leaf and `B` the root-owned bootstrap boundary reached above. Maintain:

```text
G_task[j, n] = existing task leaf/terminal return for seat j
               minus causal insolvency cost from n to L, exactly as today

G_div[r, n]  = sum of root-owned event rewards from n to L
               + intrinsic continuation from L to B
               + V_div(B, z_r)                 # zero at terminal

G_search[r, n] = c[z_r] * G_task[r, n]
                  + (1 - c[z_r]) * G_div[r, n]
G_search[j, n] = G_task[j, n]                    # j != r
```

The task component stops at `L`; it does not charge insolvency from the extra intrinsic-only rollout. That rollout's future task consequences are already represented by the original task value. Within the main path, mix **after** existing task-cost subtraction, so the root's task cost is weighted by `c` once, not accidentally left unweighted or charged twice.

Back up suffix returns from each node. An action-two continuation label must not include action one's earlier intrinsic reward or earlier insolvency cost. Do not add a full-game occupancy score at every edge or include the already played game prefix in a reward-to-go head.

Store task and intrinsic sums alongside the combined search sums, with one shared completed-trajectory count and watermark. All three channels, plan evidence and visit counts commit atomically or not at all. Intrinsic returns are signed; do not clamp the combined result to a win-probability range. Opponent nodes still select their own absolute task component. There is no alternating sign flip.

Cache pure neural outputs by their complete input/evaluator identities. Path-dependent rewards and stochastic continuation results are not pure neural outputs and must not be stored in a shared NN cache. A node's initial completed-Q fill value is frozen when initialized; repeated trajectories contribute their own completed returns, not an overwritten initial value.

### 8.4 Recommendations and labels use the same objective

Feed the combined statistics into the existing Gumbel selection/completed-Q machinery. Preserve complete-plan legality/support, completed-round recommendation snapshots, actual-successor checks and fallback rules.

Diversity intentionally changes recommendations relative to task-only search. The required parity is:

- Legacy/off profile: unchanged behavior.
- New profile with neutral/zero intrinsic contribution and identical contexts, draws and count budgets: task-only equivalence.
- Enabled profile: recommendations and exported targets use the same combined objective, while their completed-round versus final-watermark distinction remains unchanged.

Do not change only the final label while the tree continues searching a different objective.

## 9. Label mathematics, identities and admission

Retain the existing all-legal improved-policy equation with combined root-owner returns:

```text
p        = softmax(raw_T1_logits)
Q[a]     = combined_return_sum[a] / n[a]                if n[a] > 0
v_mix    = (v + N * sum_visited(p[a] * Q[a])
                / sum_visited(p[a])) / (1 + N)
Q_fill[a]= Q[a] if visited else v_mix
target   = softmax(raw_T1_logits + 0.1 * (50 + max(n)) * Q_fill)
```

`N=sum(n)`; a node with no evidence is not labelled. `v` is the corresponding combined node value. Use the search's own float64 implementation; Python recomputes it independently. Preserve raw units, legal masking, no Q normalization, no prior floor and no Gumbel noise inside the target. Do not substitute normalized visit counts or renormalize over only visited actions.

Add to the existing label fields:

- Strategy ID, exact style/intent, context contract and seat-context receipt.
- Static profile identity, dynamic evaluator-bundle identity and intrinsic target-receipt identity.
- Root owner, search context and opponent-model convention.
- Base task `reward_id` separately from the intrinsic objective hash.
- Task and intrinsic node values and per-edge mean returns, combined value/returns, coefficient and direction identity.
- Main versus intrinsic-continuation work, aborted incomplete trajectories and completed work watermark.

The existing `reward_id` still identifies the ordinary combined task reward. Do not overwrite it with a diversity reward or mix the two meanings in utility readiness.

Validate that all return channels have the same evidence, are finite and recombine correctly. Recompute target and incoming logits with the **recorded context**. Keep the current target tolerance (max absolute difference 1e-5), probability normalization tolerance (1e-6), and logits tolerance (atol 2e-5, rtol 1e-5), subject to an explicitly documented device parity result.

Validate assignments against the independently frozen job; a reply cannot authorize its own strategy or evaluator identity. Actual action-two identity must include context, objective bundle, intrinsic prefix/normalization state and the existing public/recall/legal/revision/task-ledger bindings. Replanning spends the same original clock and cannot change the objective mid-turn.

Preserve completed-trajectory admission: valid partial-round, fallback-played and late-finalized labels survive. No low-diversity, task-quality, two-visit, coverage or timeout quality gate is added. Cancellation, wrong owner/model/context/bundle, nonfinite output, invalid successor and incomplete evidence remain integrity failures.

## 10. Policy-only training and the four-age buffer

The teacher loss remains:

```text
L_teacher = beta_teacher * mean_rows KL(target(. | s, z) || policy(. | s, z))
```

Keep the current raw-T1 forward and actor-parameter gradient allowlist. No teacher loss gradient enters task-value, utility, descriptor, local-scoring or intrinsic companion heads. No teacher MC targets are created. Reading a frozen head during collection does not train it.

Shared encoder updates can still move both critics indirectly. Existing DCPG preservation, clipping, the PPO KL guard, LR controller, PPG cloning safeguards and qualified-anchor checks remain authoritative. Teacher gradients are not exempted. Distinguish teacher KL from raw-policy movement and from ordinary behavior-probability reconstruction.

Teacher rows must stay out of ordinary actor/critic data, PPG, auxiliary scoring, DOMiNO centroid/quality/dual evidence, PFSP outcomes, curriculum counters and ratings. They are consumed only by this separate supervised policy term.

Preserve 64 age-weighted label draws and four ages. Encode actual contexts on both initial admission and disk reload. Never relabel a legacy neutral row as a focus row, average different contexts into a neutral target, or train a held-out shard.

Separate two identities:

- **Static buffer compatibility:** schema, contexts, descriptor/regime, objective algorithm, search profile and producer. An incompatible change starts a fresh teacher window through a reviewed transition.
- **Per-collection bundle:** weights, companion, fitting direction and coefficient snapshot. These normally change between generations and are retained on each row. They must not accidentally flush the four-age buffer every iteration.

Keep the existing age-level policy staleness test. Compute additional diagnostics by age/strategy: correction KL, policy drift, direction angle/norm change, coefficient change and target-receipt lag. These diagnostics do not silently add new rejection thresholds. Old labels retain their original frozen target; do not recompute them using today's centroids.

Archive any previous teacher beta calibration when adopting this materially different objective, then measure the new objective in shadow using the existing two-iteration procedure. Do not reuse the neutral teacher's gradient-scale receipt. This changes only teacher calibration, not PPO LR state, PPG cloning beta, DOMiNO dual state or entropy baselines.

## 11. Versioning and proposed configuration

Use a new, strictly validated configuration contract. The names below are proposed identifiers, not currently implemented commands or accepted schemas:

```json
{
  "contract": "teacher-phase/2.0.0",
  "mode": "off",
  "diversity": {
    "contract": "teacher-domino-search/1.0.0",
    "assignment": "teacher-uniform-seven-seat-game/1.0.0",
    "context_encoding": "industry-context-4-style-8-intent/1.0.0",
    "objective": "domino-retained-fit-direction",
    "coefficient_source": "current-committed-domino",
    "opponent_model": "root-latent-self-play",
    "reward_owner": "root-only",
    "intrinsic_leaf": "bounded-root-owner-bootstrap/1.0.0",
    "descriptor_input": "known-root-self-unknown-task",
    "normalization": "four-player-phase-actions-7-8-12-4/1.0.0",
    "missing_readiness": "defer-collection"
  }
}
```

This is the extension portion of the block, not a complete replacement config. Retain the table/search/buffer/term settings in section 4.

Suggested related contracts:

| Surface | Proposed identity |
| --- | --- |
| Label source | `search-v1-domino-improved-policy/1.0.0` |
| Buffer | `teacher-label-buffer/2.0.0` |
| Teacher native producer | `brass-lab-teacher/2.0.0` |
| Objective algorithm | `brass-gumbel-is-mcts/domino-count-v1` |
| Context-aware objective session | `brass-gumbel-is-mcts/domino-session-v1` |
| Retained intrinsic fit metadata | `domino-search-fit/1.0.0` |
| Frozen inference bundle | `teacher-domino-evaluator/1.0.0` |

Keep `method=search_v1`; additionally record the unchanged base selection algorithm/session identities. The new objective must not pretend to be the old task-only algorithm. Admit this exact extension at a narrow teacher boundary; do not relax the general B01 method checks or allow arbitrary `focus_policy`, composite, subtree reuse, search_v2 or V205 overrides. Industry context is unrelated to the native focus/composite search variants.

Maintain historical v1 validation and dispatch. A mode/share-only toggle cannot smuggle in the v2 schema or new objective; adoption is a separate reviewed transition. If the old teacher block has not yet been adopted, allow a direct explicit v2-off adoption rather than requiring fake v1 state.

## 12. B02.5, B03 and B04 implementation revisions

### 12.1 B02.5: configuration, conditional labels and PPO term

1. Add v2 config/adoption, immutable context assignments and evaluator-bundle planning.
2. Extend readiness with retained-fit metadata without changing ordinary DOMiNO learning.
3. Implement conditional label validation, cache reconstruction and target-channel checking.
4. Preserve the existing policy-only teacher term and add strategy/objective telemetry.
5. Extend the generation/hash-checked helper; default off. Keep ordinary RNG, teacher seed cursor and training progress exact.

Primary files: `src/brass_ai/recovery/teacher_{phase_config,phase,labels,buffer,term}.py`, `config.py`, `history.py`, `resume.py`, `scripts/configure-teacher-phase.py`, and `scripts/export-training-reports.py`.

Add small isolated modules for the teacher context/bundle contract where that avoids coupling generic teacher code to mutable DOMiNO state. Add retained-fit metadata hooks in `domino_auxiliary.py`/PPG rollback only where necessary; no new optimizer, loss coefficient or training schedule.

### 12.2 B03: DOMiNO-aware full-state restoration and separate admission

1. Instantiate and restore `Diversity` on the actual saved recipe in `lab_continuation.restore`, attach it to the same phase/companion structure as the production trainer, and compare the live objects to the saved payload.
2. Explicitly verify intrinsic head weights, both intrinsic Adam states, dual Adam/logits, centroids, support, quality, regime/archive and committed generation. Verify that no DOMiNO adoption or PFSP context reset is replayed.
3. Inventory the required fingerprint manifests, evaluator bundles, retained-fit receipts and teacher shards with hashes. Protect live dependencies from retention; retain full restoration/rollback dependencies, not merely JSON summaries.
4. Extend teacher admission independently of ordinary replay. Equal model weights or a valid ordinary context receipt cannot make a teacher row ordinary data.
5. Preserve the diagnostic fork's sealed origin/path map and source checkpoint bytes. Old checkpoints lacking optional teacher-fit metadata still restore their full training state; report teacher readiness as unavailable.

Primary files: `src/brass_ai/rust_backend/lab_{admission,continuation,snapshot}.py`, the B03 runner and tests, plus existing narrow resume/source-review routes. Recheck current code before changing these because migration work is in progress.

### 12.3 B04: native objective extension and GPU service

1. Replace neutral literals with verified actual-game contexts at the native teacher boundary.
2. Add a lab-only objective adapter for descriptor events, intrinsic prefixes, the bounded root-owner continuation and atomic multi-channel backup.
3. Extend binary inference requests with optional read-only intrinsic output and complete evaluator/context identity. Reuse encoded state/trunk output when both values are needed; do not reload the model or run seven forwards for one request.
4. Keep task-value descriptors unknown. The intrinsic wrapper uses its separate root-self descriptor; it must not bypass `Brass::new` by stuffing a full privileged descriptor table into the existing task context.
5. Extend exported root/continuation/replan evidence and actual-successor validation, including both raw return channels.
6. Recalibrate trajectory cost on the actual device for the new profile. Root-owner continuation work must be included before freezing N.
7. Preserve CPU/thread admission, shared-slot retirement, cancellation and deadlines. Different strategies can batch under the same base weights; cache keys also include the intrinsic bundle and exact context.

Primary files: `lab-native/src/teacher.rs`, `lab-native/core-overlay/teacher.rs`, new lab-only objective/continuation modules, `src/brass_ai/rust_backend/lab_teacher{,_inference,_overlay}.py` and the existing inference transport/service adapters. If shared tree/trajectory hooks are required, keep the disabled path unchanged and test it independently.

The existing B04 overlay changes exports only. This revision changes backup semantics for an opt-in profile and therefore needs a new real source/build fingerprint and qualification. Preserve the sealed B01 source and old B04 packages. Use a detached supplement with checked hook matches or a narrowly scoped typed extension; never rewrite historical archives or relabel a binary. Refresh affected transfer manifests and evidence chains only when the implementation exists.

## 13. Continuation, cache ownership and later activation

Implementation and activation are separate operations. Complete the current backend migration without implicitly turning this feature on.

At a later clean committed boundary:

1. Lock the run and read its actual generation `G`, checkpoint hash, signature, effective DOMiNO state and existing teacher mode.
2. Adopt the new profile as off through the narrow reviewed route. Preserve weights, every optimizer/RNG, iteration/game/seed counters, population/PFSP, ratings, active KL reference, fixed probe, economic baselines, DOMiNO and PPG state.
3. If old teacher shards exist, retire them under the new static profile identity; keep owned old files until the new transition/checkpoint is durably committed and required rollback dependencies are retained. Never delete ordinary PPG caches for a teacher-only change.
4. Record retained-fit metadata on subsequent normal accepted PPG work. Do not launch additional PPG just to make the teacher ready.
5. After the revised B02.5/B03/B04 checks and the existing B05/B07 gates pass, run the bounded isolated shadow qualification and calibrate teacher beta.
6. Enable production teaching later with an explicit mode toggle. No automatic date, generation or near-end-of-run switch is introduced.

Use the current label generation convention: incoming committed `G` produces the collection associated with `G`, and its update commits `G+1`. Do not shift existing buffer ages while adding a fit-generation field. Fit generation, collection generation and checkpoint generation must be separately named.

A restart discards in-flight trees, continuation rollouts, owners and clocks. It restores sealed complete labels and the exact frozen bundle needed to verify them. Test the crash after source-transition commit but before the first new checkpoint, as well as partially filled buffers and repeated helper invocation.

Do not publish a runnable production command for an unimplemented flag in this plan. The implementation deliverable must provide tested preflight/adoption/shadow/on/off/continuation commands using the then-current committed generation and the existing IL path. Do not hardcode the historical g1326/g1324 generation or rerun a prior reward/DOMiNO migration.

## 14. Reporting and expected cost

Retain existing teacher metrics and add the following to full reports, compact history and the report exporter:

| Area | Required measurements |
| --- | --- |
| Identity | Profile, collection generation, model/export/head hashes, target-fit generation, context and objective bundle IDs. |
| Context support | Planned/completed seat-games and labels by strategy, phase, split and label kind; separate unique games from correlated decision labels. |
| Objective | `c`, direction norm/nearest ID, direction lag/change, task versus intrinsic Q scale and contribution. |
| Intrinsic estimator | Retained fitting support, descriptor fingerprint provenance, ordinary held-out MSE/bias versus zero, wrapper parity and readiness reason. |
| Work | Main search steps, intrinsic-continuation steps/requests, terminal versus root-owned bootstrap counts, incomplete work, N/K and completed trajectories/rounds. |
| Budget | Work/finalization/wait time, deadline hits, overruns, watchdogs, original-clock replan counts and cancellation. |
| Training | Per-strategy teacher KL/top-1 before and after, age-level staleness plus strategy diagnostics, presentations/distinct labels used, gradient share/cosine and ordinary PPO KL. |
| Runtime | Collection wall time, lane-seconds, GPU request/batch distribution, queue wait, encoding/head time, memory and teacher-term time. |
| Outcome | Whole-game strategy separation and task quality by ID, with matched setups and uncertainty when an evaluation is eventually run. |

With unchanged table counts, the old planning estimate remains approximately 868 training labels per iteration (28 x 31), or about 124 per strategy in expectation. This is not a guarantee of evidence or uniform labels. There are only 112 training seat-games and 16 held-out seat-games per collection; some held-out strategies will be absent. Report absent support as missing, and pool descriptive held-out results across the retained window with their original target identities.

There are still at most 512 reserved searched turns and 2,048 nominal search lane-seconds per enabled iteration (32 x 16 x 4). Root/replanned-root calls can outnumber turns, but share their turn's allowance. Diversity does not multiply this allowance by seven. Intrinsic continuation consumes part of it and may sharply reduce N; total table cost also includes unsearched play, transport and finalization. The earlier plan's wall-time and label estimates cannot be treated as new measurements.

The existing 512-wide model's companion is documented as 34,945 parameters, but obtain the actual size from the admitted checkpoint. Loading that head is small relative to the base network; extra rollout/encoding/inference is the important unknown. No sevenfold model residency is required, and the deployed ONNX interface need not change for the lab PyTorch service.

Use empirical whole-game measures to distinguish seven strategic policies from seven labels or local action noise. Evaluate separation within/between IDs, a shuffled-ID null, and descriptors with industry counts removed. Keep task reward, VP/insolvency and neutral historical ratings visible. Lower distillation KL, more labels or larger descriptor distances do not establish stronger play.

## 15. Focused regressions and bounded validation

### 15.1 Required correctness tests

1. **Assignments:** IID seven-way stream, all IDs reachable, seat rotation, restart/retry identity, separate teacher/ordinary streams, fixed ID across a game and no sevenfold model copies.
2. **Context propagation:** actual play, root-latent simulation, both actions, sub-stages, fallback and label reload use the intended context. Corrupt context/model/role/bundle fields fail independently.
3. **Descriptors:** native/Python event equality; predecision semantics; exact slot/family scales; phase boundary and 7/8/12/4 normalization; forced/pass/scout cases; no hidden/context-ID leakage.
4. **Reward/version binding:** fitted old direction versus advanced centroid fixture; rejected/zero-step/partially supported fits; no forged readiness; current coefficients do not change the intrinsic target's meaning.
5. **Four-seat backup:** hand-calculated task/intrinsic returns, bias-free seat indexing, root-only reward, no sign flips, causal insolvency charged once, action-two suffixes and terminal zero bootstrap.
6. **Scalar continuation:** leaf owned by root, by another actor, forced debt and terminal; stop before the first root-owned action; separate RNG; no ordinary rows/labels/tree counts from rollout work.
7. **Atomic timing:** deadline during continuation or GPU wait, depth limit, canceled node insertion, no partial component/visit publication, no replenished clock, valid earlier evidence survives an unfinished trajectory.
8. **Gumbel labels:** complete legal support, all three label kinds, zero/partial-round evidence, fallback/overrun retention, separate return-channel recombination and Python/Rust target parity.
9. **Controls:** old/off route unchanged; neutral and zero-diversity count-budget parity with matching contexts/draws; enabling diversity can change recommendations without corrupting support or target provenance.
10. **Inference:** batch sizes 1 and 2, heterogeneous strategies, candidate chunking including 1 and 772 candidates, nonterminal/terminal behavior, exact self-descriptor mapping, same base weights with different heads/directions cannot cross caches.
11. **Training:** teacher actor gradients only; shared-trunk preservation and existing KL guard participation; no intrinsic/value-head gradients from teacher loss; CPU fixed-data off/shadow equivalence for weights, all Adam states and RNG.
12. **Resume/admission:** explicit live DOMiNO restore, both companion Adams and dual state, teacher-fit receipt, four-age conditional reload, missing/tampered shard, helper idempotence, source-transition crash gap, commit-bound cleanup and rollback.

### 15.2 Small isolated measurement sequence

Run these during implementation, not while writing this plan:

1. Authored Python/Rust fixtures with fake clocks and recorded draws. These cover rare continuation/replan/cancellation cases without waiting for full games.
2. A small root corpus spanning four phases and all seven contexts. Compare the neutral control, context-only task search and diversity search with fixed trajectory counts/draws for mathematical correctness. These controls are diagnostic profiles, not new default training modes.
3. On the qualified lab device, recalibrate with the existing bounded two-root/eight-completed-trajectory approach, expanded only enough to include the intrinsic continuation path. Then compare equal 4-second/K16 budgets on the same roots, hardware and counterbalanced order. Report N and overhead instead of equating higher average batch size with throughput.
4. Reuse B04's two-table isolated shadow smoke: one training-split and one held-out-split table, at the normal profile budget. Missing strategies/label kinds are supplemented by fixtures, not an unbounded game campaign.
5. B05 fixed-data bounded updates, sufficient to exercise the existing KL/PPG acceptance and rollback boundaries, then checkpoint/reload with the full DOMiNO/teacher state. B07 retains the previously specified bounded isolated shadow canary; reduced fixtures must be labelled as such.

Predeclare the finite root/game/update counts, output directory and hardware in each receipt. Do not modify source-run checkpoints or count smoke data toward production. End-to-end diversity and playing-strength claims require a later, separately authorized matched training/evaluation experiment.

## 16. Implementation order and completion criteria

| Step | Revision area | Deliverable |
| --- | --- | --- |
| TD0 | B02.5 + B03 | Reinspect current migration/DOMiNO changes; pin contracts; prove normalization; explicit complete DOMiNO restore and target-fit receipt design. |
| TD1 | B02.5 + B04 | Context assignment, conditional inference/labels/reload and task-only diagnostic control. Teacher remains off for production. |
| TD2 | B04 | Read-only intrinsic evaluator, bounded root-owner continuation, multi-channel backup, extended identity and real source fingerprint. |
| TD3 | B02.5 + B03 | Policy-only training, conditional buffer, receipt retention, telemetry, exact resume and narrow adoption helper. |
| TD4 | B04 -> B05/B07 | Focused tests, GPU calibration, two-table shadow, fixed-data update/restart and inherited canary gates. Publish measured limitations and tested commands. |

Complete means the new profile can be selected explicitly, resumes the current DOMiNO-trained run exactly, produces conditional diversity-aware labels under the same shared budget, and trains only the policy through the existing guarded term. Neutral equivalence, correct non-neutral return accounting, canceled-work exclusion and conditional cache reload must all be demonstrated.

Keep production teaching off until the revised implementation and relevant existing migration gates are qualified. Do not mark completion based on context-only plumbing, successful model loading, seven observed IDs or a lower imitation loss. Ordinary DOMiNO, unrelated migration updates and the installed game remain intact.
