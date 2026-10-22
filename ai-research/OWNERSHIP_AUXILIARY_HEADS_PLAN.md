# Local Ownership and Scoring Auxiliary Heads: Implementation Plan

Date: 2026-10-22  
Status: proposed implementation; this document does not enable the feature or alter a training run.

## 1. Objective and decisions

Add KataGo-inspired, training-only supervision for:

1. The effective owner and official industry VP at every industry slot at the end of the current era.
2. The owner and official network VP of every applicable connection at the end of the current era.

Continue the existing v11.2 self-play run without fresh imitation learning, replacement of its learned core, or changes to its game-playing interface. Preserve the completed IL dataset/model and frozen population.

**Latest user decision:** on the first successful continuation with this feature, retire the existing PPG buffer and start an eight-iteration fresh PPG round. From committed g560, collect g561-568 and run the first new PPG phase at g568, followed by g576. Do not reuse g553-560 for that phase. This supersedes the earlier suggestion to preserve that buffer and run PPG at g564.

The existing reward, PPO objective, exploration, opponent selection, evaluation budgets and safeguards remain in force. The auxiliary outputs do not become rewards, action filters, sale gates, or LR/GAE qualification gates.

## 2. Verified starting point: g560

Evidence directory:

[Latest export](../results/joint7-v11-2-grouped-lr-critic-g0560-20261022T202410715648Z)

Read state.json, recorded-config.json, iteration-0559/report.json and retention-history.json. The report's iteration index is zero-based; state.iteration is the completed generation count.

| Item | Recorded value |
| --- | --- |
| Committed generation / ordinary games | 560 / 179,200 |
| Active run | artifacts/step5/joint7-v11-2-grouped-lr-critic |
| Latest full resume checkpoint | generation-0560.pt |
| Latest model ID | 521a49306214f652932320069a8d0fff1f013bfa019a21859325145c9e548be9 |
| Checkpoint SHA-256 | ff10e995a94b1a27ee605505bea2427bc05348f45a7bd9fc4b88cab66cb3c83d |
| Recorded implementation | b95f45e2b0be6672a4efb7a3cabf9e2fab3e152a667e7f229b6c09ab5cb01c71 |
| IL run, unchanged | artifacts/step5/features4-v11-2-batch7-il |
| Core architecture | brass-four-seat-mlp-resnet/11.2.0; medium preset: width 512, eight residual blocks |
| Features / action input | Features-4; existing 327-wide action input |
| Local banks | 49 slots, 69 connections; 64-dimensional entity embeddings |
| Ordinary table counts | 160 self-play, 154 history, 6 v4 per 320 games |
| Population | 24 |
| PPO | two epochs; batch 2,048; microbatch 512; candidate-cell cap 131,072 |
| g560 PPO steps / effective LR | 52 / 6.25e-6; configured base LR 6e-6 |
| Learner exploration / openings / branching | epsilon 0; temperature 2 for existing opening rules; branching 0.015 |
| Reward | win 1.0, placement 0.1, VP 0.001 capped at 180 |
| Insolvency | event_cost 0.020; max_cost 0.080 |
| Snapshot critic weight | fixed 1.0 |
| Reported self-play GAE lambda | 0.95; preserve saved per-table gates too |
| KL reference | g440, still active at g560 |
| PPG | enabled; eight iterations, uniform weights; batch 4,096 |
| PPG acceptance / patience | 16 steps / 1 |
| PPG selection | one-SE coverage retention; epoch cap 2, permitted range 2-6 |
| PPG policy safeguards | exact overall and age-0 KL caps 0.002; saved beta 8 |
| Existing PPG clock | origin g540; last phase g556; formerly next g564 |
| Existing buffer to retire | g553-560 |
| Disabled features | teacher search, v4 search, monitor/broader evaluations and temperature comparison |

At g560, collection took 192.38 s and PPO took 83.81 s. PPG was not scheduled: zero value optimizer steps, although cache/diagnostic work occupied 9.23 s. Scheduled PPG phases at g548 and g556 took 225.11 s and 269.40 s, retaining 112 and 144 steps respectively. These are reference observations, not controlled performance benchmarks.

The g560 buffer reports 17,698,768,640 critic bytes and 3,691,341,832 policy-action bytes; its pre-commit peak was 24,107,832,660 bytes. Cleanup must measure actual files rather than assume these numbers equal reclaimable storage.

The export contains reports and checkpoint references, not the live generation-0560.pt file. Actual full-state restore and optimizer parity must be verified on the lab machine or with an authorized copy. Do not claim that reading this export verifies checkpoint contents.

## 3. Label contract: an era-end local scoring map

Introduce a separate versioned contract, proposed name:

- local-scoring-targets/1.0.0

It must bind the board topology, canonical entity ordering, scoring rules, producer implementation and game/branch identity. Keep future outcomes outside the observation and action feature schemas.

### 3.1 Industry targets

For each of the 49 stable slot IDs, at the scoring instant of the observation's current era:

- Effective owner: one of NONE, seat 0, seat 1, seat 2, seat 3.
- An occupied, flipped industry receives its actual owner's class.
- An empty or unflipped slot receives NONE.
- A flipped industry with zero industry VP still receives its owner's class; it may matter to network scoring.
- Record the exact official industry VP contribution, attributed to that owner, with zero for other seats.
- A zero VP target must not be used to infer NONE.

These are **slot-final-state targets**, including slots empty at observation time. If a tile is overbuilt, use the replacement actually present at scoring. Do not confuse this target with survival or flipping of the original tile instance. Keep instance IDs in the audit record where needed to verify replacements.

### 3.2 Connection targets

For every canonical connection ID:

- Mask connections that cannot exist under the current era's rules.
- For applicable connections, classify NONE or the actual owning seat at scoring.
- Record that connection's official network VP for its owner; other seats receive zero.
- A built zero-point connection has an owner and zero VP.
- Use the authoritative scoring rule, including all endpoints, merchant/external-location contributions and any special topology. Do not approximate points from a heuristic or assume every connection has only two endpoints.

Different players can benefit from the same surrounding industries through different connections. No zero-sum or exclusive-region assumption is allowed.

### 3.3 Time, seats and masks

- Supervise the end of the **current era only** in v1. Do not add a second Rail forecast to Canal rows in this revision.
- Capture after any required final settlement/liquidation, before scoring cleanup or era transition removes/replaces assets.
- Canal and Rail produce distinct receipts. A surviving industry may contribute in both eras.
- Final Rail has no income/debt settlement under the current rules; do not invent one.
- Store labels in stable seat order; transform owner classes and four-seat VP targets using the row's existing observer-relative mapping.
- A missing receipt means unavailable supervision, never an all-NONE/all-zero target.
- Empty slots and unbuilt, era-valid connections are valid negative examples.
- An incomplete game is not made eligible for PPO by having a partial scoring receipt.

### 3.4 Scoring audit

Build labels from the official scoring path or confirmed scoring events. Reconcile, per player and era:

industry contributions + connection contributions + explicit bonus/adjustment terms = actual era scoring change.

Also reconcile whole-game VP, allowing bonuses or insolvency deductions posted outside that scoring transaction. Do not fold unrelated penalties into local industry/link labels. Do not clip these labels at the reward's 180-VP cap.

Retain exact integer targets and validity masks. Validate non-negativity and storage bounds for local scoring values rather than silently clipping or casting overflow.

## 4. Collection and replay integration

### Authoritative capture

The current Host.Advance invokes EndRound, which can cross an era boundary; terminal Result contains aggregate scores only. Taking the first Rail observation or final result alone cannot reliably reconstruct the Canal map.

Implement passive scoring capture at the actual scoring transaction, after liquidation and before cleanup. If EndRound combines these operations, inspect its event sequence and capture at the correct internal boundary; a generic pre-EndRound snapshot is not sufficient proof of correctness.

Expose a training-only label receipt through a separate capability/RPC. Never add future information to Observe, choices, Features-4 tensors, policy inputs, or trusted game-state hashes that drive gameplay.

A recording-enabled host must produce identical observations, choices, actions, scoring, game RNG progression and branch checkpoint restoration to recording-disabled execution for identical inputs. Capture must not consume randomness.

The existing pinned host manifest is historical evidence: do not overwrite its hash to impersonate the new binary. Bind the new recorder capability/build manifest separately, with a narrowly reviewed admission route proving unchanged gameplay/observation semantics. If host producer admission requires an additional compatibility receipt, implement and test that receipt explicitly; do not weaken general hash checks.

### Replay storage

Add an optional, strictly validated game-level auxiliary-target envelope or an atomically bound companion receipt. Choose one representation consistently before implementation; the preferred representation is one game-level envelope, with separate derived PPG sidecars.

- A game envelope contains at most the two era maps plus producer/identity/audit metadata.
- Old replays and the completed IL dataset remain valid without it.
- Newly collected apply-mode games require the applicable complete receipts; producer failures are explicit, not silently masked as legacy data.
- Bind the envelope to the replay digest and validation seal. Extend cloning, family assembly, serialization and checked-cache paths accordingly.
- Inference must never inspect this field.

Branch continuations use their own future receipts. A fork in Rail needs only a Rail target for its rows. Restored pre-fork events must not cause duplicate capture or copy the parent's divergent future. Keep parent and descendants in one holdout family; exclude copied prefixes and the excluded fork move from auxiliary fitting.

Generate and normalize targets once per game/era. Do not repeat host scoring, scan the full game, or store a complete target map per decision.

## 5. Model design: training-only companion heads

Keep the core FourSeatNet/UtilityNet architecture, parameter names, Features-4 layout, policy scorer, outcome heads and exported interface unchanged.

Create a separate LocalScoringHeads module, owned by the trainer rather than registered inside the core model. This avoids changing old frozen model IDs or requiring every history/IL opponent to acquire unused heads.

Proposed version:

- local-scoring-heads/1.0.0

For each entity kind (industry slot, connection), use one shared MLP:

~~~text
global input = concat(core context [512],
                      existing relative seat-descriptor embeddings [4 * 32])
entity input = existing local embedding [64]

hidden = GELU(W_global * global_input + W_entity * entity_input + bias)  # width 64

owner_logits = Linear(hidden, 5)
vp_predictions = Linear(hidden, 4)  # normalized expected contribution per seat
~~~

Compute the global projection once per position, then broadcast across entities. Separate slot/link weights; share weights within each kind. Use the existing descriptor inputs/dropout decisions, not new private information or independently sampled dropout.

This design adds 45,705 parameters per kind, or **91,410 auxiliary parameters**, for the inspected width-512 core. Verify that count in tests; report core and auxiliary counts separately. No backbone or action-branch widening is required.

Use linear VP outputs trained against scaled official contributions. Do not obtain expected VP by multiplying independently predicted ownership and unconditional VP: that can introduce an incorrect independence assumption. Cross-head consistency is initially a diagnostic.

Initialize the new heads with a private deterministic seed derived from the adoption checkpoint and head contract. Initialize final projections to zero; preserve all global Python/NumPy/Torch/CUDA RNG streams.

Refactor the training forward path to reuse one core encoding for the official and auxiliary losses. Do not run an additional complete encoder for every head. Preserve the old public forward output and state-only critic fast path; any touched architecture source requires an exact equivalence receipt and parity tests.

At deployment, collection, evaluation and ONNX inference, execute only the existing core. New auxiliary outputs are optional diagnostic exports, outside the default game-playing graph.

## 6. Losses, optimizer ownership and staged activation

Use the existing critic-training row stream, including its eligible learner/snapshot/branch rows. Do not add bot or IL observer rows, or alter PPO actor eligibility. A learner observation may predict all four players' public scoring outcomes without becoming opponent-policy training data.

For each row:

- Average ownership cross-entropy across valid entities, separately for slots and links.
- Average normalized VP Smooth L1 loss across valid entities and four seats.
- Normalize by full effective batch sizes across microbatches.
- Unavailable legacy targets contribute zero. Do not rescale a small labeled fraction to pretend that every row was labeled.
- Apply existing row/snapshot weights once; do not also weight sampling.
- Start without class oversampling. Report NONE-class dominance and positive-class metrics to detect trivial predictors.

Proposed initial auxiliary term, outside the existing value_weight multiplier:

~~~text
L_aux = 0.05 * slot_owner_CE
      + 0.05 * link_owner_CE
      + 0.01 * slot_VP_Huber
      + 0.01 * link_VP_Huber

VP target/prediction normalization: divide by 20.
~~~

These are conservative experimental starting coefficients, not values validated by KataGo for Brass. Preserve the existing official value/utility loss coefficients.

### Optimizers

The current PPO continuation code requires exactly one core AdamW group. **Do not append auxiliary parameters to that group or rebuild it.**

- Restore the existing core PPO AdamW and core value Adam exactly.
- Maintain separate small auxiliary-head optimizers for PPO and PPG, analogous to the core's two optimizer histories.
- Core parameters receive combined original-plus-auxiliary gradients, but are stepped only once by the existing phase's core optimizer.
- Companion optimizers update only auxiliary parameters. They cannot step a core parameter.
- Initial companion base LR: 1e-4, Adam, zero weight decay. Keep this explicitly configurable and validate it in the smoke; it does not change either core LR.
- Preserve the core's existing gradient clipping. Clip companion-head gradients separately at 1.0 and report both.
- Skip companion steps entirely when no auxiliary targets exist; clear stale gradients.
- No extra training epochs or standalone auxiliary phase in v1.

For PPG, apply its existing within-phase warmup/decay factors to the companion base LR and reset the PPG companion moments at phase start, matching the existing core value-Adam reset policy. Ordinary continuation itself does not reset either core optimizer.

### Staging

Persist the adoption generation G and a committed-iteration age:

- First two newly committed iterations: train new heads on detached core representations only.
- Next five iterations: scale the auxiliary gradient entering shared core representations by 1/5, 2/5, 3/5, 4/5, 1.
- Head gradients keep their configured loss weights throughout.
- Retries and restarts do not advance this clock.

Implement gradient scaling without changing forward values, including descriptor embeddings. Modes off, shadow and apply must be strictly validated: shadow remains head-only; off retains state but performs no auxiliary work. Disabling auxiliary learning later requires an explicit reviewed config transition and does not restart PPO/PPG.

## 7. PPO/PPG safeguards and acceptance

- PPO keeps actual behavior probabilities, opening-row T1 ratios, detached importance weights, epsilon 0, legal masks and forced-action filtering.
- Exact behavior-policy and raw-T1 movement checks include any shared-core movement from auxiliary losses.
- The LR controller still measures post-PPO KL before PPG. Auxiliary-only head steps do not count toward its optimizer-step or warmup evidence.
- PPG freezes the same post-PPO raw-T1 policy. Both overall and age-0 0.002 caps stay unchanged.
- PPG minimum-loss reference, paired one-SE test, coverage preference, patience, epoch votes and GAE evidence continue to use the existing official critic metrics. Auxiliary accuracy cannot waive a failure.
- On every retained-state snapshot, rollback or beta retry, move the core weights, auxiliary weights, core value Adam and PPG companion Adam together. Keep distinct-label usage aligned with the actually retained path.
- PPG does not update PPO companion moments. The PPO companion state persists for the next iteration.
- Keep all critic-held-out families out of auxiliary fitting, including snapshot and branch rows from those families. Their existing eligible actor updates remain unchanged: this is a critic holdout, not an untouched policy test set.

## 8. Checkpoint and identity contract

Store the companion bundle inside a strictly validated, versioned training metadata entry, proposed name:

- training.local_scoring_auxiliary
- local-scoring-training/1.0.0

Include architecture/config hashes, topology/label contract, head tensors, both companion optimizer states, adoption generation/checkpoint identity, initialization seed, activation progress and retained sample counters. Use tensors and primitive metadata compatible with weights-only loading.

Keep the core state_dict and model_id definition unchanged. Add a separate auxiliary-bundle identity; the full checkpoint hash binds both. Head-only updates must not create a fake new PFSP or PL player identity.

Old weight-only population checkpoints need no bundle. An old full resume checkpoint may omit it only when the exact reviewed adoption route initializes it. Once adoption has produced a full checkpoint, missing/corrupt companion state is a hard resume error.

Restore and preserve:

- all existing core weights, PPO Adam moments/steps and value Adam;
- RNG, next_seed, context counters, iteration/game counts and source lineage;
- PFSP identities, weights/evidence, population retention and original IL identity;
- active g440 KL reference/probability cache and calibrated allowance state;
- LR multiplier/window/cooldown and completed warmup;
- GAE/per-table evidence, fixed snapshot weight and archived R2 evidence;
- economic and entropy baselines, anchor-promotion entropy-floor progress;
- rating history, retired-panel receipts and evaluation schedule.

No reference promotion or baseline recalibration is caused merely by enabling these heads.

## 9. One-time fresh PPG round and safe cache retirement

This section implements the user's latest buffer instruction.

### Transaction order

1. Lock the run; require the expected committed generation and checkpoint hash.
2. Validate the old full checkpoint, source/config route, optimizer states, live file ownership and pending-work status.
3. Prepare an adoption receipt bound to g560 (or the explicitly selected current generation). It records the old PPG clock/cache receipts and the exact reset.
4. Durably commit the reviewed transition using the existing restart-safe source-adoption mechanism.
5. Only after that commit, retire audited files in this run's owned .ppg-eight-cache and any obsolete .ppg-value-cache. Preserve ownership markers. Do not touch IL/rehearsal/probe caches, population, checkpoints or unrelated directories.
6. Begin new collection. Write new cache files transactionally and expire them only after their consuming generation commits.

Audit all targets before deletion. Refuse unknown files, symlinks, path escapes or an ownership mismatch before deleting anything. Report file counts and actual bytes released. Use native filesystem operations; no broad shell wildcard cleanup.

Persist an idempotent reset receipt. A restart after adoption but during cleanup resumes cleanup safely, skips already-deleted owned files, and **does not reset the clock again or delete newly committed g561+ caches**.

### Reset scope

From g560:

- New PPG buffer/cache origin: g560.
- First eligible phase: g568, using fresh g561-568.
- Following phase: g576.
- Require eight newly committed iterations before the first new phase; no short-buffer substitute at g564.
- Keep the current epoch cap 2 and allowed range 2-6; archive old epoch-vote history and begin a new PPG-round vote window.
- Keep saved clone beta 8 and its hard maximum 32. Preserve safety history unless a separately documented schema conversion is needed; do not relax any KL limit.
- Preserve both existing core optimizer states at adoption. The next actual PPG phase still performs its normal configured phase-start Adam reset.
- Preserve PPO/GAE/health/anchor evidence and global training counters.

The adopted state must explicitly authorize ignoring the retired rolling-buffer references still embedded in the old g560 checkpoint. Do not delete those files and then attempt an ordinary old-cache restore. Test the crash boundary before the first new checkpoint carefully.

Archive old cache **metadata**, not a second hidden copy of the 20+ GB payload. Once deletion completes, returning to the old PPG buffer requires an external backup; retaining g560 weights alone does not recreate it. Resuming under the adopted fresh-buffer receipt remains supported.

## 10. New label cache and retention

Keep the existing large critic-row dtype and policy-input arrays unchanged where possible. Store one compact label map per game/era, referenced by existing game/context identities through a small, checksummed sidecar.

Store owner class plus scalar official contribution and masks compactly; reconstruct the four-seat target vector during bounded CPU preparation. Do not allocate a dense entity target tensor for every historical decision on the GPU.

- Bind sidecars to their replay/era receipts, target contract, core input identities and holdout-family membership.
- Commit auxiliary sidecars with the generation checkpoint and PPG cache receipts.
- Rebuild only from verified authoritative sources; never fabricate absent scoring maps.
- Delete expired sidecars only after checkpoint commit, with their matching PPG age.
- Keep any source receipts needed to restore retained full checkpoints.
- The reset in Section 9 removes old PPG payloads only once. Normal eight-age expiration resumes afterwards.
- Completed IL data remain usable for their existing losses, with auxiliary supervision absent. No dataset regeneration or filename change.

## 11. Configuration and reviewed continuation

Add one block to the recorded run signature, proposed name local_scoring_auxiliary, containing:

- training/head/target contracts and mode;
- current-era horizon and canonical topology identity;
- head width 64, four loss weights and VP scale;
- companion optimizer settings;
- two head-only iterations and five shared-gradient ramp iterations;
- an explicit fresh-PPG adoption policy, bound to the selected starting checkpoint.

Generate the new config by deep-copying the committed run's recorded config and adding this block. Do not reconstruct it from today's general defaults.

Implement a narrow reviewed-source/config migration from the recorded implementation above. Preserve existing reviewed routes; do not grant transitive approval to arbitrary edits. Validate every non-auxiliary config field for exact equality.

If interrupted work exists beyond g560, use the existing inspected recovery workflow. Archive only verified uncommitted work under its existing rules; do not change a completed generation or silently discard an ambiguous pending checkpoint.

## 12. Implementation work packages

| Package | Work and likely files | Completion criterion |
| --- | --- | --- |
| AO0: contracts and scoring capture | Host scoring observer/RPC, producer capability receipt, new recovery/local_scoring_targets.py; collector execution.py and branching.py | Exact era maps and scoring reconciliation; gameplay/RNG parity |
| AO1: replay and cache | recovery/replay.py, checked seals, execution family publication, value_buffer.py companion sidecars | Old replay admission, strict new-label validation, branch-safe and restart-safe storage |
| AO2: companion model and losses | New learning/local_scoring_heads.py and recovery/local_scoring_auxiliary.py; training forward/updates.py integration | Dimensions/counts, seat permutations, gradients and no auxiliary inference execution |
| AO3: phase integration | actor_value.py, ppg_auxiliary.py, ppg_selection.py, training checkpoint metadata | Joint retained-state rollback, unchanged official acceptance and KL safeguards |
| AO4: adoption and PPG reset | config.py, resume.py, league.py, source-review registries, bounded configuration helper | Preserve g560 state; retire old buffer after durable adoption; next phase g568 |
| AO5: reporting and verification | diagnostics/history/report exporter, focused tests, isolated smoke, docs/MLP_V11_LAB.md | Auditable labels/usage/runtime; reproducible continuation instructions |

Inspect current files again at implementation time and preserve changes made since this plan. Do not modify deployment/gameplay code solely to add outputs that are not consumed there.

## 13. Focused validation

### Labels and game semantics

Test all owner classes; observer-seat rotations; empty and unflipped slots; flipped zero-VP industries; overbuilding; liquidation; Canal clearing; surviving industries scoring in both eras; final Rail settlement behavior; zero-point links; era-inapplicable links; merchant/special/multi-endpoint connection scoring.

Check exact per-seat official scoring reconciliation, topology/order mismatches, tampered receipts, missing-vs-zero labels, parent/branch divergence and interrupted final-action handling. Recorder on/off must preserve full deterministic game traces and RNG.

### Models, gradients and identities

Test B=1 and B=2, all 49/69 entities, partial/all-missing masks and VP magnitudes above the reward cap. Verify 91,410 proposed parameters and finite gradients.

Before the first shared update, old policy logits, official values and utility must match exactly where deterministic operations permit, with tight documented tolerances otherwise. Old IL/frozen models and ONNX outputs remain loadable and unchanged.

Prove that head-only/shadow work does not alter core tensors, core Adam states, controller counters or RNG. Apply mode must send auxiliary gradients into the intended context/entity/descriptor encoders without stepping any core parameter twice.

### Continuation and rollback

Use a real isolated checkpoint fixture with nonzero Adam moments and populated controller state:

- Adopt, stop/restart during each transition/cleanup boundary, and resume.
- Compare uninterrupted and restarted runs, including both companion optimizers and activation age.
- Verify old checkpoint bytes remain unchanged.
- Verify exactly eight new committed iterations are needed for fresh PPG.
- Exercise one-SE rejection, both KL-cap failures, beta retries and selected-state restoration with companion weights/moments.
- Confirm missing active companion state fails; an old history opponent without it loads.
- Verify cleanup cannot delete active anchor/probe/population files or g561+ data.
- Verify enabling/disabling auxiliary learning does not reset the original baselines.

### Bounded smoke and benchmark

Use a new isolated output directory, authored edge-case fixtures plus a few deterministic real host games. Do not run a production training fork or a full rating campaign.

Simulate the eight-iteration fresh round with reduced rows/game counts while retaining production semantic checks. Exercise a complete PPO/PPG checkpoint/restart cycle.

Run same-input before/after microbenchmarks on available hardware, reporting label generation time, extra forward/backward time, main/auxiliary optimizer time, PPG validation time, CPU peak RSS, GPU peak allocation if available, cache bytes and collection wall time. Keep core sampling indices and random streams fixed; exclude first-call compilation/warmup from steady-state comparisons.

No fixed percentage speedup is promised. Auxiliary heads add training work; the goal is better learning per game and ultimately per wall-clock hour.

## 14. Reporting

Retain the following in full reports and compact history:

- mode, contracts, adoption generation, activation age and shared-gradient scale;
- target producer/topology identity; complete/missing/masked era/game/entity counts;
- ownership CE, Brier score, per-class/support metrics and confusion counts, separated by era and table type;
- per-seat local VP MAE/RMSE, aggregate scoring error and audit discrepancies;
- slot/link NONE frequency, current-empty/current-occupied support, and positive-target errors;
- distinct games, era maps, rows and entity labels available, attempted and actually retained; do not call correlated entity targets independent games;
- core/auxiliary gradient norms, loss contributions and companion steps/LRs;
- original PPO/PPG KL and acceptance statistics, with auxiliary rollback counts;
- labeling/training/validation times and CPU/GPU/disk overhead;
- old-buffer retirement receipt, bytes released and new PPG origin/next phase.

Use existing held-out ordinary game families for auxiliary diagnostics. They remain excluded from auxiliary critic fitting; describe their existing actor-use scope accurately.

Extend export-training-reports.py to include compact auxiliary contracts, transition/reset receipts and metrics. Do not export multi-gigabyte arrays or private simulator snapshots by default.

## 15. Intended lab procedure after implementation

**The new helper and config below are proposed deliverables. They do not exist merely because this plan was written. Do not run these commands until implementation and focused validation are complete.**

If the live run is still committed at g560:

~~~bash
cd /data/not_backed_up/yxu209/ai-training

python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 560 --apply && \
python scripts/configure-local-scoring-aux.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 560 \
  --mode apply \
  --fresh-ppg-buffer \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-local-scoring-g560.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-local-scoring-g560.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 560 --iterations 20
~~~

The configuration helper must validate and prepare only the new config/preview; it must not delete the buffer or rewrite checkpoint tensors. Training startup performs the reviewed, locked, durable adoption and subsequent cache retirement.

Expected progression: g561-580; first fresh PPG phase g568, second g576. This is the user's continuation procedure, not a validation run for the implementer.

If training advances before deployment, bind the helper to the new actual committed generation G; use G+8 for the first fresh phase. Refuse a stale expected generation rather than rewinding. Repeating the train command after a later interruption must not repeat the fresh-buffer reset; update the expected-generation guard to the actual committed generation.

## 16. Research basis and limits

- [KataGo: Accelerating Self-Play Learning in Go, Sections 4.1 and 5.2](https://arxiv.org/pdf/1902.10565): local ownership and score supervision provide finer feedback. The ablation removes ownership and score together; it does not isolate ownership's individual contribution or establish Brass gains.
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782): shared-task gradients can interfere. This motivates conservative weights and diagnostics; PCGrad itself is not included in v1.
- [Current one-SE PPG recipe](PPG_ONE_SE_G540.md) and [g540 continuation fixes](SELFPLAY_G540_DEBUG.md): preserve the accepted training/safety semantics, except for the explicitly requested one-time PPG buffer/round reset.

The implementation is complete only when the resume, scoring, rollback and bounded-smoke checks pass. Improved ownership accuracy alone is not proof of stronger play; subsequent normal PL evaluations and wall-clock learning trends must establish that separately.

