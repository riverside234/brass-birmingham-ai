# DOMiNO-Based Strategic Diversity Training: Implementation Plan

Date: 2026-10-29

Status: design only. This document does not change training code, configurations, checkpoints, or an active run. Implementation and validation remain future work.

## 1. Objective and requested decisions

Train several coherent, competitive whole-game strategies within the existing conditional policy. Success means different investment, development, financing, sales, and network-expansion trajectories across games, with adequate playing quality. Different industry counts or higher action entropy alone are insufficient.

Use seven strategy IDs, sampled independently and uniformly at the start of each neural seat's game, including neutral. Each ID has probability **1/7**. There is no 25% neutral allocation, no 50% neutral allocation, and no requirement that a four-seat table contain all seven IDs.

Apply the same sampling to **every frozen neural opponent selected for a history table**, including the IL checkpoint when it is selected from that pool. Select the checkpoint through the existing PFSP sampler first; independently select its strategy ID afterward. Frozen models remain frozen and never become PPO actor-data sources.

For this revision, retain the existing, checkpoint-compatible industry-context encodings and their existing exploration-distribution semantics. The IDs are:

| ID | Existing context | Probability per neural seat-game | DOMiNO learning role |
| --- | --- | --- | --- |
| 0 | `neutral` | 1/7 | Task-only reference policy |
| 1 | `CottonMill` | 1/7 | Task return plus adaptive diversity objective |
| 2 | `Manufacturer` | 1/7 | Task return plus adaptive diversity objective |
| 3 | `Pottery` | 1/7 | Task return plus adaptive diversity objective |
| 4 | `Brewery` | 1/7 | Task return plus adaptive diversity objective |
| 5 | `IronWorks` | 1/7 | Task return plus adaptive diversity objective |
| 6 | `CoalMine` | 1/7 | Task return plus adaptive diversity objective |

This mapping implements the requested random industry focus without silently changing the meaning of old checkpoint inputs. The names provide existing conditioning and exploration priors; they do not impose industry quotas, force builds, or award a fixed bonus for the named industry. The new diversity objective must distinguish broader strategies. Removing industry conditioning altogether would be a separate, explicitly versioned change.

## 2. Paper review and methodological boundary

Primary source: Zahavy et al., **Discovering Policies with DOMiNO: Diversity Optimization Maintaining Near Optimality**, ICLR 2023, [paper](https://arxiv.org/pdf/2205.13521), [HTML](https://arxiv.org/html/2205.13521v2). The methodological checks for this plan cover Sections 2–3, Appendix B including Algorithm 1 and its implementation listings, and Appendix D's limitations.

DOMiNO conditions a shared policy on an episode-level latent, separates task and diversity value estimates, and adapts their relative weights using a performance constraint. Its published implementation samples latents uniformly and reserves one task-only reference. Diversity uses expected behavior features and nearest-policy separation. The reference is learned, not a known optimum. The published implementation uses V-trace; this project retains PPO. See the paper's equations 4, 6, and 9 and Appendix B.

Call the implementation **DOMiNO-based episodic PPO**, not a literal reproduction or a guaranteed near-optimality method. The finite games, strategic feature manifest, industry-conditioned behavior, opening correction, multi-player self-play, and DCPG/PPG integration below are project adaptations. Results must establish their usefulness here.

Do not substitute a fixed diversity-bonus coefficient or a reward on the strategy label itself for the adaptive constrained objective. Do not add DGPO's diversity-threshold switch that disables task reward. Also do not claim that DOMiNO fixes the task coefficient at one: its mixture weight is adaptive.

## 3. Current implementation inspected

| Existing path | Relevant behavior and required integration |
| --- | --- |
| `src/brass_ai/recovery/ordinary.py` | Learner contexts currently use a 12-position cycle: six neutral and one of each industry. Historical policies are explicitly constructed with `focus='neutral'`. Plan context assignments after determining actual seats. |
| `src/brass_ai/step5/policy.py` | `context_for()` maps the seven supported contexts to four style and eight intent values. `mixed` also has a separate 12-position fallback. Exploration uses industry intent to weight legal action groups. Both assignment paths need consistent contracts. |
| `src/brass_ai/recovery/execution.py` | Resolves seat rotations, installs planned contexts, creates policy identities, and seals collection jobs. Contexts must be explicit rather than dependent on worker scheduling. |
| `src/brass_ai/recovery/branching.py` | Saves/restores seat contexts at a fork. Strategy IDs must be inherited together with those contexts. |
| `src/brass_ai/recovery/updates.py` and `actor_value.py` | Verify behavior probabilities, prepare learner/critic rows, compute advantages, and execute PPO/PPG. Current attribution relies in several places on a single learner policy identity. |
| `src/brass_ai/recovery/gae.py` | Uses undiscounted, per-seat nonforced chains and preserves exact Monte Carlo behavior at lambda 1. Forced transitions' consequences are included between eligible decisions. |
| `src/brass_ai/recovery/critic_data.py` | Builds seat descriptors using cached policy fingerprints. A neutral fingerprint must not be presented as the exact fingerprint of a newly focused historical opponent. |
| `src/brass_ai/recovery/quality_pool.py` and `pfsp_pool.py` | Sample model identities and estimate model-level outrank difficulty. Context randomization changes the distribution underlying that estimate. |
| `src/brass_ai/recovery/ppg_g210.py`, `value_preservation.py`, and the local-scoring companion | Provide current PPG, DCPG, auxiliary-state, and keep-best patterns to extend. |
| `src/brass_ai/recovery/resume.py`, `history.py`, and `league.py` | Reviewed transitions, full checkpoint state, commit ordering, and compact reporting. |

The inspected `configs/step5/joint7-v11-2-grouped-lr-critic-training-g1324.json` currently specifies a **two-iteration PPG buffer and a phase every iteration**, two maximum epochs, held-out checks every 16 steps, and KL checks every 32 steps. Do not reintroduce older eight- or sixteen-iteration settings from discussion history.

The same config specifies epsilon 0.08, three T2 opening action decisions, 1% branching with a 50% T2 / 50% exploration-q fork mixture, fixed action entropy 0.015 and other-decision entropy 0.005, insolvency event cost 0.005 and cap 0.05, and fixed snapshot critic weight 1.0. These are inspection findings, not instructions to overwrite a later run's settings. A future transition must read the selected run's actual committed config and preserve unrelated differences.

Preserve table composition and any already-qualified v4-share transition, historical capacity and admission cadence, seat rotation, rating panels and retirement receipts, evaluation policy settings, teacher settings, reward definitions, model architecture, and inference batching. Rule-based v4 seats have no neural strategy ID.

## 4. Episode assignment, historical opponents, and identities

### 4.1 One independent draw per neural seat-game

Create a versioned sampler such as `ordinary-strategy-uniform/1.0.0`:

1. Construct table type, learner roles, frozen opponent model identities, and seat rotation exactly as before.
2. For every neural seat, use a dedicated deterministic RNG stream keyed by the run seed, committed ordinary-game index, final physical seat, and sampling contract.
3. Draw `z = randrange(7)` exactly once. Store it in the immutable job before collection starts.
4. Obtain style/intent through the existing context mapping. Hold the assignment throughout the game, including payment, continuation, and forced decisions.
5. Retries reuse the job. A branch inherits all four assignments; resetting post-fork opening counters does not resample strategies.

The context RNG must not consume PFSP, setup, action-sampling, or PPO RNG draws. Do not derive IDs from table-index modulo patterns, assign a single ID to the entire table, or redraw an ID because its behavior is weak. Independent draws allow two seats to receive the same ID. Uniformity is an expectation across many games, not an exact per-collection quota.

Apply this rule to four current-model seats on self-play tables; current learners plus every frozen neural seat on history tables; and current learners on v4 tables. Evaluation and teacher paths retain their explicit existing contexts unless separately requested.

### 4.2 Frozen-model capability checks

Before activation, enumerate all selectable history/IL checkpoints and verify that they admit the current seven context encodings and legal-action interface. Executability does not prove that every context already produces a strong or distinct strategy.

Unsupported context contracts must produce a readiness error naming the checkpoint. Do not silently fall back to neutral, relabel its inputs, remove it from the pool, or convert it into a trainable learner. Already compatible historical models require no new intrinsic head for inference.

### 4.3 Separate roles from behavior identities

Record at least:

- `model_id` and checkpoint digest;
- explicit `training_role` and actor-eligibility origin;
- `behavior_family_id`, including exploration/opening/context contracts;
- `strategy_id`, context name, and context-encoding contract;
- `seat_policy_id`, binding the behavior family and actual context;
- original ordinary-game index, physical seat, parent/fork identity, and assignment receipt.

Audit every existing equality check against the one learner `policy_id`, including collection, replay admission, sampling reports, critic descriptors, and PFSP receipts. Use the planned role and verified incoming learner model/behavior family to admit PPO data. Neither sharing a model hash nor receiving a focus ID makes an opponent a learner.

Keep model residency and inference queues keyed by actual weights; seven contexts do not require seven copies of each model. Every request still carries its own correct style/intent and must return to its original seat and model.

### 4.4 Context-aware critic descriptors

Cache raw-policy fingerprints by `(model_id, strategy_id, probe_hash, context_contract, fingerprint_contract)`. Reuse the fixed probe positions but apply the declared context when calculating each fingerprint. This is a derived cache, not a rewrite of the IL/anchor probe.

Compute frozen fingerprints lazily before the first collection requiring them, with batching and reusable caches. For current-model seats, derive all needed contexts from the incoming model before freezing that collection. Store the matching fingerprint identity in each seat descriptor. Preserve descriptor dimensions and dropout policy.

Update both Python and the selected native collection adapter; never assume a fallback path is unused. The existing native search file named `focus.rs` concerns search behavior and is not evidence that neural industry assignment is already implemented there.

## 5. Strategic behavior representation

Add a separate, versioned **training descriptor manifest**. It does not change Features-4 or the NN input layout. Descriptors are computed from ordinary replay observations and selected legal actions, in the acting player's perspective.

Use the existing phase boundaries, confirmed in `teacher_schedule.py`:

| Phase | Rounds |
| --- | --- |
| Early Canal | Canal 1–4 |
| Late Canal | Canal 5–8 |
| Early Rail | Rail 1–6 |
| Late Rail | Rail 7–8 |

Partition descriptors by these phases so early development and late development do not collapse to the same final count. Proposed feature families are:

| Family | Initial content | Purpose |
| --- | --- | --- |
| Action allocation | Main-action family indicators, including build/develop/sell/loan/network/scout/pass | Spending turns differently |
| Industrial investment | Own installed industry counts and levels; development position by industry | Distinguish investment paths without fixed industry rewards |
| Realization | Own flipped/unflipped assets by industry | Distinguish investment from realized production |
| Financing | Public own cash, income, and spending measures | Capture financing and reinvestment timing |
| Network/geography | Own links, connected footprint, and own asset locations using canonical map IDs | Capture expansion and positioning |

For every component, specify the exact replay source field, perspective, static normalization, clipping bound, missing-value behavior, and unit test. Use observed predecision state and selected actions; do not invent an after-action observation where replay provides none. A build request is not proof that an asset was realized.

Exclude strategy IDs, industry-intent/style inputs, model IDs, seats as arbitrary labels, seeds, policy fingerprints, opponent identities, hidden opponent information, and final win/VP labels from the diversity descriptor. Otherwise trivial identifying information or outcome differences could masquerade as strategic diversity. Opponent actions must not directly earn the learner a diversity bonus; their effects on public state remain part of the game.

Use fixed family weights and normalizations. Obtain any data-dependent scale from a train-only pilot, then freeze and hash it before `apply`. Never normalize independently per strategy or continuously rescale features during training: either would change the meaning of stored centroid distances.

### 5.1 Explicit finite-game occupancy convention

For a complete seat-game `g`, let `x_t` be the normalized feature vector at its main-action decision `t`, `p(t)` its phase, and `N[g,p]` the number of that seat's main-action decisions in phase `p`. Define a phase-block vector:

```text
phi_t = put_in_phase_block(p(t), x_t) / (4 * N[g,p(t)])
psi_game = sum_t phi_t
```

Each completed phase has equal weight; extra payment/continuation records do not inflate its importance. Record the normalizers and all phase support. An interrupted/incomplete game is not a complete DOMiNO sample. Handle genuine engine-terminal edge cases explicitly rather than dividing by zero or renormalizing missing phases silently.

This is a finite-episode, phase-weighted adaptation. Do not claim a stationary infinite-horizon occupancy estimate. The descriptor itself uses observable information; the completed-game normalization is training bookkeeping and is never supplied to the actor as future information.

Initialize each strategy centroid from real completed training seat-games. Maintain an EMA over per-strategy collection means, with equal weight per seat-game and no update when a strategy has no support. Count distinct whole games as well as seat-games because seats on the same table are correlated.

## 6. Intrinsic reward and adaptive quality constraint

The following specifies the project's numerical contract; the referenced paper equations establish its origin. Use float64 for controller statistics and float32 for training tensors.

### 6.1 Published nearest-policy reward, with frozen statistics

At iteration start, freeze the admitted centroids and nearest neighbors:

```text
j(i) = argmin over j != i of squared_norm(psi[i] - psi[j])
d[i] = stop_gradient(psi[i] - psi[j(i)])
r_div[t] = dot(phi_t, d[z])                 # z > 0
r_div[t] = 0                               # z == 0
```

Use the repulsive reward corresponding to equation 6 as the initial implementation. Neutral participates as a neighbor/reference centroid but receives no intrinsic reward itself. Mask self-distance with positive infinity, including when all distances are zero. Resolve equal-distance ties deterministically.

Implement this published per-policy direction with the other centroids held fixed. Do not silently replace it by backpropagating a sum of all pair distances, adding incoming-neighbor terms, or counting the same direction twice. Unit-test against the derivative of the corresponding local half-squared-distance objective with a fixed neighbor.

The optional equation-8 Van der Waals variant is not enabled initially. If implemented later, its distance target and distance normalization must be versioned and calibrated; do not invent a target distance in changing feature units.

Identical centroids give zero repulsive reward. Report this honestly. Existing distinct context priors and independent trajectories may break symmetry, but success is not guaranteed. Do not seed fabricated centroid differences and report them as observed strategy diversity. Any parameter perturbation or additional symmetry-breaking method needs a separately identified experiment.

### 6.2 Task-quality metric

Keep the existing undiscounted combined task return, including its existing insolvency cost, as `J`. Estimate `J[z]` from complete ordinary learner seat-games under the same collection regime. Never substitute intrinsic return, normalized PPO advantages, PL points, or a mixture of different reward versions.

The proposed initial quality constraint is:

```text
J[z] >= alpha * J[0],  z = 1..6
alpha = 0.95                         # project starting value, not a proven optimum
```

`J[0]` is the current learned neutral strategy, not the frozen IL model or the LR controller's KL reference. The scalar is a combined-return ratio; it does not mean 95% win rate, 95% of Elo, or a guaranteed maximum 5% strength loss. Require a finite, positive, adequately supported reference estimate before activating the relative constraint. If that condition fails, remain task-only and report why; do not silently shift rewards to make the ratio convenient.

Report results by table type, opponent regime, and strategy. Uniform assignments must be independent of table type and opponent draw. A small number of v4 games cannot establish a separate per-strategy v4 quality guarantee. Use paired fixed-opponent evaluation for release qualification; a changing self-play reference can deteriorate together with the other strategies.

### 6.3 Adaptive coefficients and update sign

For nonneutral strategies, maintain unconstrained controller parameters `m[z]`:

```text
c[z] = sigmoid(m[z])
L_dual = mean_{supported z=1..6} c[z] * stop_gradient(J[z] - alpha * J[0])
```

Minimize `L_dual` with a separate optimizer. Below-target performance must increase `m` and therefore the task coefficient; above-target performance decreases it. Initialize `m=0`, giving `c=0.5`, when activation has sufficient evidence. This initialization is an experimental training change, not protection against transient regression.

For neutral, bypass the controller: `c[0]=1` exactly and the intrinsic coefficient is zero. Do not compute `sigmoid(1)` and call it task-only. Do not give neutral a trainable dual parameter.

Freeze all coefficients for the full PPO iteration. Perform one controller update per successfully committed iteration using the previous committed EMA estimates, then incorporate that iteration's new observations for the next generation. No minibatch, replay age, branch, validation check, or restart constitutes an extra controller observation.

Proposed project defaults are EMA decay 0.9 and dual Adam LR 0.01 per eligible committed iteration. These time units differ from the paper's learner updates; tune them through the planned short experiment rather than claiming the numerical defaults were reproduced. Archive the initial settings and every reviewed change. Reject nonfinite statistics/updates without modifying the last valid state.

Before initial activation, require at least 32 training seat-games and 16 distinct ordinary games per ID in the current statistics regime. Bootstrap with task-only training and zero intrinsic reward while collecting genuine estimates. Held-out games do not train centroids or the controller. Preserve support counts and report any strategy that prevents readiness.

## 7. PPO, GAE, DCPG, and PPG integration

### 7.1 Separate task and intrinsic targets

Keep the official win, rank, VP, utility, and local-scoring targets unchanged. Do not bake diversity into the existing utility head, teacher utility, ratings, or economic-health metrics.

Add one scalar intrinsic value estimate for the acting player's continuation, conditioned on its actual strategy through the existing encoded context. Use a training-only companion head attached to the current state representation and the acting-seat descriptor embedding. Reuse the recorded stable/relative seat mapping. The companion must not require new inference outputs or alter the existing policy input layout.

Follow the existing auxiliary-companion serialization pattern: zero-initialize the final layer, use an isolated initialization RNG, store its parameters and PPO/PPG optimizer state in the full checkpoint, and leave old actor/critic parameters and Adam moments intact. A separate weights-only policy export need not contain this training-only head.

For ordinary learner rows, derive intrinsic reward-to-go from the frozen reward coefficients and complete recorded trajectory. A main-action event's intrinsic reward is counted once. Carry rewards across intervening forced decisions when constructing the nonforced actor chain. Never assign an auxiliary reward independently to every payment subchoice.

### 7.2 Advantage combination

Compute existing task advantages `A_task` without alteration. Compute intrinsic advantages `A_div` separately. Start the intrinsic stream at undiscounted Monte Carlo/lambda 1; the existing task GAE gate provides no evidence that the new intrinsic critic is calibrated.

```text
A_combined = c[z] * A_task + (1 - c[z]) * A_div
```

Combine unnormalized streams, then apply the current single advantage-normalization operation over eligible actor rows. Do not independently standardize each stream or each strategy, and do not multiply the task/intrinsic weights again inside the PPO loss. Record raw stream magnitudes and the resulting effective contributions.

Preserve actual behavior likelihoods, legal masks, forced-action exclusion, and current PPO sampling. Preserve the existing T2 opening objective and its detached importance weights, using the combined advantage in the same position. Keep raw-T1 movement and behavior-policy verification separate. This remains the current opening approximation; it is not a proof of exact on-policy DOMiNO learning.

### 7.3 Delayed value fitting

The inspected run uses DCPG preservation during PPO. Extend its frozen pre-PPO output snapshot with the intrinsic scalar on eligible learner critic rows. Preserve it with MSE through the non-detached trunk and its companion head. Do not resume fresh task-value fitting inside PPO.

Fit the intrinsic critic on fresh targets in PPG, retaining the run's phase cadence, learning-rate source/schedule, task losses, cloning objective, KL checks, and acceptance rules. The initial intrinsic target-loss and preservation coefficients are both 1.0 within the existing value-weight convention, explicitly recorded as new objective coefficients.

For the first implementation, intrinsic fitting uses **only the current iteration's ordinary learner training rows**. Older PPG ages, historical/snapshot seats, IL/bot seats, held-outs, and branches have no intrinsic fitting target. Existing task-value fitting keeps its current eligibility across ages and roles. Mask the auxiliary loss explicitly and normalize/report its valid sample support; never convert a missing auxiliary label into a zero target.

This current-only choice avoids training the new intrinsic critic on obsolete centroid-based rewards or pretending old policies generated fresh on-policy intrinsic experience. It is an intentional project adaptation, not an unnoticed truncation of the task PPG buffer. If current auxiliary support is inadequate, report it rather than silently relabeling older targets.

PPG keep-best, rejected-interval rollback, interruption recovery, and acceptance snapshots must include the intrinsic head and its optimizer state. The current task held-out loss remains the selection criterion; intrinsic held-out error is a separate diagnostic and cannot override policy KL safeguards or task-value acceptance.

Cloning and KL probes must retain each row's actual context. Report support and drift by strategy in addition to the existing overall groups. If adding minimum strategy quotas to an existing stratified sampler, retain the intended aggregate weighting with explicit inclusion-weight correction; do not quietly turn the overall KL into an unweighted average of uneven strata. Preserve the current KL thresholds and baseline-capture rules. Leave the fixed IL/qualified-anchor probe and its operative limits unchanged.

This initial integration requires an enabled, supported PPG/DCPG path. Reject an `apply` configuration that has nowhere to fit the intrinsic critic; do not automatically re-enable a disabled value phase or change its cadence.

### 7.4 Branches and historical critic rows

Branches keep their inherited strategy/context, behavior provenance, opening reset, and current task actor/critic eligibility. They do not enter centroid estimates, quality-reference estimates, or dual updates: forks are selected counterfactual continuations, not independent complete games.

For this first version, retain the existing **task-only** objective on branch actor rows; do not apply a partial-game intrinsic reward or attenuate their task advantage by `c[z]`. Report that scope explicitly. A later branch-inclusive diversity objective would require prefix accounting, the same full-game normalization, and branch-sampling correction. Fork moves and copied prefixes remain excluded as before.

Historical seats keep their existing task-critic eligibility and snapshot weight, but receive no learner intrinsic targets and no PPO actor updates. A history model's assigned ID is behavior metadata, not evidence for the current learner's strategy centroid.

## 8. PFSP and fair quality evidence

Keep one PFSP member per checkpoint and the current model-selection function, capacity, admission cadence, and pairwise outrank definition. Do not expand a 24-model pool into 168 context entries or weight a model more heavily because it has more contexts.

Under the new ordinary sampling contract:

```text
P(history model=m, strategy=z) = P_PFSP(m) / 7
```

Use each historical seat outcome once with the existing learner/opponent comparisons. Aggregate the model's live difficulty estimate over its uniformly sampled contexts; additionally report context-stratified counts and outcomes. Keep the planned model probability and context probability separately in receipts.

At adoption, archive the old neutral-opponent PFSP score state and start a **new context-regime accumulator** using the existing configured prior and half-life. Preserve membership, checkpoint references, admission ages, IL qualification/retirement receipts, and the archived comparisons. This is an explicit estimator migration because neutral-only difficulty is not the same target quantity as uniform-context difficulty. It does change initial sampling weights; report that rather than claiming unchanged effective draws across adoption. Within the new regime, context RNG must not perturb a given PFSP draw stream.

Do not use sparse per-context win estimates to introduce a second context-difficulty sampler; all seven IDs remain uniform as requested.

For DOMiNO statistics, ordinary pool turnover is expected nonstationarity handled by the estimator. Changes to reward, descriptor manifest, context mapping, or table-share regime require a new statistics regime and readiness check. Do not reset the controller merely because another historical checkpoint is admitted.

## 9. Configuration and checkpoint contracts

Add one strictly validated block. The following is a proposed schema, not a configuration accepted by today's trainer:

```json
{
  "diversity": {
    "contract": "domino-episodic-ppo/1.0.0",
    "mode": "shadow",
    "context_contract": "ordinary-strategy-uniform/1.0.0",
    "strategy_count": 7,
    "reference_strategy": 0,
    "sampling": "independent-uniform-per-neural-seat-game",
    "historical_contexts": "same-seven-contexts",
    "descriptor_contract": "brass-strategic-occupancy/1.0.0",
    "descriptor_manifest_sha256": "REQUIRED_FROM_VALIDATED_MANIFEST",
    "objective": "nearest-repulsive",
    "quality_ratio": 0.95,
    "statistics_ema_decay": 0.9,
    "minimum_seat_games_per_strategy": 32,
    "minimum_distinct_games_per_strategy": 16,
    "dual_learning_rate": 0.01,
    "dual_initial_logit": 0.0,
    "intrinsic_gamma": 1.0,
    "intrinsic_lambda": 1.0,
    "intrinsic_value_weight": 1.0,
    "intrinsic_preservation_weight": 1.0,
    "intrinsic_fit_scope": "current-ordinary-learner-training-rows",
    "branch_objective": "existing-task-only"
  }
}
```

Reject unknown fields, unsupported modes, invalid types/nonfinite values, duplicate or reordered context mappings, a missing descriptor manifest, and impossible reference IDs. Changing a contract or manifest is a reviewed transition, not a silent edit.

- `off`: legacy behavior and no DOMiNO mutation; regression-test parity.
- `shadow`: new requested uniform neural-seat assignments and telemetry, but no intrinsic actor/critic updates or changes to base losses. This changes collection behavior, so it still needs an explicit transition.
- `apply`: uniform assignments plus the objective once its evidence is ready.

A full checkpoint must save controller logits/Adam, centroids, quality EMAs, support counts, initialization status, descriptor hash, context map, statistics regime, last consumed generation, companion parameters/optimizers, and adoption receipts. Distinguish this training-state identity from the frozen policy's weights-only inference identity.

Derive a generation's pending statistics from its frozen input state and stage them transactionally. Publish exactly once with the successful checkpoint commit. Retrying an interrupted iteration must not advance the sampler, EMA, dual optimizer, or support counters twice.

## 10. Continuation and cache migration

Provide a new generation-bound configuration/transition helper during implementation. It must load and lock the real committed state, verify the full checkpoint hash, show the before/after contract changes, and reject a stale expected generation. Do not hard-code g1324 as the future adoption point.

Preserve existing model weights, optimizer moments, LR multiplier/warmup, ordinary-game and iteration progress, RNG state, population membership, checkpoint lineage, active KL reference and protected probabilities, rating history, economic baselines, and unrelated GAE/snapshot controls. Initialize only the new companion/controller state and the explicitly archived/new PFSP estimator regime.

Use a separate RNG namespace for new assignment and head initialization. Adding the head must not consume the PPO RNG stream or reinitialize existing layers. Existing behavior verification should fail on incompatible pending collections; archive/recollect them through the established mechanism instead of editing their labels.

Start a fresh PPG cache generation at adoption because role/context provenance and auxiliary schema change. Preserve the value optimizer and schedule. Archive the previous manifest, stage new caches, and expire old owned caches **only after the new checkpoint commits**. An adoption failure must still permit recovery of the prior checkpoint and cache. Preserve the original IL dataset and its paths.

Tag health evidence by collection regime and invoke existing mechanisms for reviewed regime changes. Do not silently lower economic baselines, reset the LR multiplier, reinterpret old EV/R-squared evidence, or change GAE lambda just because diversity is enabled. Uniform-context quality is new evidence, not a substitute for prior neutral evaluation.

Base policy/ONNX outputs and feature dimensions stay unchanged. Policy bundle metadata must record supported context semantics. Strict loaders must reject missing required DOMiNO training state on a purported DOMiNO resume; legacy weights-only opponents remain loadable through their explicit old contract.

## 11. Reporting and measurement

Add a compact `diversity` section to each iteration report and retained history:

- contract/mode, incoming and outgoing statistics generations, and readiness reason;
- assigned and completed seat-games by ID, table type, role, and physical seat;
- history model/context counts and the PFSP context-regime receipt;
- task-return mean/support, neutral reference, threshold, coefficient before/after, and constraint residual per learned ID;
- pairwise centroid distances, nearest neighbors, descriptor hash, EMA support, zero-distance/tie counts;
- raw task/intrinsic advantage scales and combined contribution; no latent-wise normalization hidden in telemetry;
- intrinsic reward/return distribution, clipping or nonfinite events, fitted rows/steps, and held-out intrinsic error;
- ordinary versus branch scopes and auxiliary exclusions;
- existing task EV/R-squared, PPO raw/behavior KL, PPG safeguards, and economic metrics;
- centroid preparation, context-fingerprint work, companion training time, RAM/VRAM, cache bytes, and collection/training throughput.

Console additions should be short, for example `DOMiNO=apply ids=7/7 ready=1 min_dist=... quality=...`. A classifier's ability to guess the ID, entropy increase, or a larger centroid distance does not establish playing strength.

Expected extra work is descriptor extraction, a small companion head, and context-specific fingerprints. Frozen fingerprints are reusable; a changing learner can require up to six additional context passes over the existing fingerprint probe compared with a neutral-only calculation. These are neural evaluations, not extra game rollouts. Measure this cost rather than claiming a free feature. Store sparse phase-indexed descriptor events instead of four mostly-zero dense phase blocks on every replay row; never cache autograd graphs or duplicate policy weights per context.

For behavioral evaluation, compare phase-resolved trajectories on held-out matched setups and rotated seats. Report within-ID variation, between-ID separation, and actual representative game traces. Recompute diversity using held-out games and keep a same-policy random-label null comparison to expose seed noise. A useful result should remain when industry-count features are omitted from the evaluation metric; otherwise it may only rediscover the existing industry bias.

Evaluate each ID's task quality and the intended uniform mixture against the same frozen opponents, including randomized-context histories where appropriate. Keep current neutral rating records comparable; do not silently feed mixed-context results into an unchanged PL identity. Any new mixed-policy rating must have its own declared evaluation identity. Deployment sampling and qualification are separate from training allocation.

## 12. Focused regressions

### Assignment and historical behavior

- Enumerated IDs map to the exact existing style/intent arrays; neutral is included.
- Large deterministic sampling fixtures approach 1/7 per ID with a predetermined statistical tolerance. Do not demand exact counts from IID draws.
- Results reproduce across worker counts, queue order, retries, and restart; no draws consume opponent/action RNG streams.
- Every historical seat, including duplicate checkpoint draws and IL, receives its planned context. Learner/opponent roles remain correct when weights happen to match.
- Seat rotation, branch inheritance, three post-fork opening decisions, and exploration-q weighting remain correct.
- The collection fallback and native adapter cannot silently reinstall the old 12-position cycle or neutral history.

### Algorithm and feature correctness

- Hand-worked centroids verify nearest neighbors, self masking, deterministic ties, neutral participation, and exact zero at coincidence.
- Finite differences with a fixed neighbor verify the local repulsive reward direction and scale. Do not test against a different global-gradient objective.
- Dual update below the quality target raises task weight; above target lowers it. Neutral is exactly task-only.
- Unsupported/zero/negative reference quality, missing strategy support, and nonfinite updates leave last valid controller state intact.
- Features are invariant to seat renumbering with equivalent relative game state and to changes in irrelevant seed/model/strategy labels.
- Swapping early/late action order changes phase descriptors even with identical final industry totals.
- Normalization gives equal whole-game/phase weight and cannot be inflated by extra payment records. Forced-action credit is counted once.
- Holding out a game excludes all its seats and branch family from fitting centroids/controller/auxiliary heads according to the existing whole-game split.

### Training and checkpoint correctness

- Separate advantages combine once before the existing normalization; coefficients are frozen for an update.
- Actual epsilon/temperature behavior reconstructs exactly; opening PPO still uses its existing raw-T1 ratio and detached importance denominator.
- DCPG preserves old task outputs and the new scalar without fitting fresh targets during PPO; the intended preservation gradients reach the trunk.
- Intrinsic PPG gradients occur only on current ordinary learner rows; no snapshot, older-age, branch, or missing-label row becomes an auxiliary target.
- Rejected PPG intervals restore model, auxiliary head, and all relevant Adam state together.
- Context-aware fingerprint caches reject a different model, context, or probe; per-model inference routing stays correct.
- Cloning/probe selection retains true context and correct aggregate weights; reported per-strategy support agrees with actual samples.
- A restart mid-bootstrap, after activation, and around a failed commit reproduces the next assignments, coefficients, consumed statistics, and training result.
- Adoption preserves old tensor values/moments/RNG, starts only the declared new state, and does not delete recovery caches before commit.
- `off` parity, old checkpoint readability, strict new-checkpoint completeness, and unchanged policy export outputs are covered.

## 13. Implementation stages and bounded validation

| Stage | Work | Exit evidence |
| --- | --- | --- |
| D0 | Contracts, context capability inventory, source-field audit, descriptor fixtures, paper-equation tests | Reviewed mapping and exact feature/quality definitions |
| D1 | Uniform learner/history assignment, role identities, fingerprint caches, native path, PFSP regime | Deterministic collection/provenance tests and a short collection smoke |
| D2 | Train-only phase statistics, intrinsic returns, adaptive coefficients, reporting | Numerical tests, zero/unsupported-case behavior, shadow logs |
| D3 | Intrinsic companion, PPO advantages, DCPG/PPG masks and rollback | Authored training regression and exact restart parity |
| D4 | Generation-bound transition, cache handling, exports, report collection, lab documentation | Dry-run receipt plus isolated end-to-end smoke |
| D5 | Explicitly launched empirical trial and qualification | Supported strategic separation and task-quality evidence |

For the implementation smoke, use a new output directory and cloned/synthetic fixtures, never the active production run. Limit native collection to 28 ordinary tables: 14 self-play, 12 history, and 2 v4, with the same authored setup schedule for the control and new collector. This smoke mix is a bounded test fixture, not a new production curriculum. Rotate seats and verify all historical model/context execution paths.

Random draws in a tiny smoke need not cover every ID in every stratum. Use deterministic authored fixtures for exhaustive coverage and report real random support separately. Keep neural inference and actor/critic shapes representative, cap optimization to a small fixed number of steps, and perform a save/reload/retry check. Report wall time, fingerprint cache misses, inference request counts, auxiliary cost, and maximum memory where available.

Do not launch a long run or strength-evaluation campaign merely to implement this plan. A later trial must explicitly measure both quality and whole-game diversity; a smoke establishes integration, not stronger or more varied play.

## 14. Suggested file ownership

New modules should separate configuration/state, assignment, feature descriptors, statistics/rewards, and the intrinsic companion, for example `recovery/domino_config.py`, `domino_sampling.py`, `domino_features.py`, `domino_state.py`, and `domino_auxiliary.py`. Keep them small enough to test without launching a game engine.

Integrate through the inspected ordinary/execution/policy, actor-value/updates/GAE, critic fingerprint, PFSP, PPG preservation/cache, and resume/commit paths rather than adding an unverified parallel trainer. Register exact source-review transitions and extend compact report/export handling so the new evidence survives retention.

Update lab instructions with the helper that actually exists after implementation. Until then, this file contains no runnable production continuation command and grants no permission to mutate an existing run.

## 15. Definition of done

Implementation is complete when uniform seven-way sampling applies to all requested neural seats; frozen opponents retain correct weights and roles; the adaptive DOMiNO objective and independent intrinsic critic are numerically tested; PPO/PPG/branch/resume invariants hold; and an isolated bounded smoke passes with useful telemetry.

Experimental success is a separate result: several IDs must demonstrate repeatable, meaningfully different whole-game strategies under held-out matched conditions while satisfying the declared quality criteria. Report fewer qualified styles if seven inputs yield fewer than seven distinct strong strategies. Neither DOMiNO's name nor uniform sampling proves that result.
