# Masked ISMCTS: one search design for trained AI and v3

**1 October training selection:** both requested eight-game screens are complete. **v3 + Masked ISMCTS** remains the imitation and rule-based training/evaluation default because it beat original S1 at the shared **one-second / two-private-worker** budget. Main lab collection allows **16 heavy tables × 2 workers**. S1 is retained as a frozen diagnostic control; neural search/distillation remain **Masked ISMCTS + NN**. The game patch is unchanged. See [comparison results and the rail-era follow-up](../docs/V3_S1_TRAINING.md) and [current conda guide](../docs/TRAINING_RECOVERY_LAB.md). F0–F6 are implemented and archived in game release 1.4.11; older proposal/status paragraphs below are historical.

30 September 2026. **Status: M0–M5 implemented locally.** Game package 1.4.9, direct-one-second bank `bank-927c8ddec148897fa542` and fresh recipe 2.2.0 replace the old active S1/S2/S3/R4/R5 workflow. See [exit evidence](../docs/MASKED_ISMCTS_EXIT_CHECKS.md) and the [conda guide](../docs/TRAINING_RECOVERY_LAB.md). Lab throughput and playing strength remain to be measured.

**Strength follow-up, 30 September:** [MASKED_ISMCTS_STRENGTH_FIX_PLAN.md](MASKED_ISMCTS_STRENGTH_FIX_PLAN.md) proposes correcting unseen-edge/continuation selection, reducing v3 candidate breadth and rollout cost, adding guarded heuristic Q normalization, and separating heuristic settings from the NN frontier. Isolated diagnostics found no repeated first-action visits at 2/5/12 seconds on one stored position. These follow-up changes are not implemented; the installed package and frozen bank remain unchanged.

**Model/batching revision:** start **fresh training of a residual MLP (MLP-ResNet), with at most 2,000,000 total parameters** for the main model. Do not use existing neural checkpoints. Later goal/personality additions may slightly exceed that ceiling, with a conservative total limit of **2,100,000 (5% above 2M)**. CPU batching is required for Masked ISMCTS + NN. Add dynamic action masking and policy-guided top-K expansion with progressive unpruning. The architecture, initialization, batching and pruning rules below supersede the earlier instruction to retain the existing dense 1.90M model as the main backbone.

## 1. Decisions

| Use | Selected design |
| --- | --- |
| New trained AI's primary search | **Masked ISMCTS + NN**: information-set tree search, permitted-observation policy priors and four-seat neural values |
| Main training architecture | **MLP-ResNet ≤2,000,000 total parameters**, including state/action encoders, every head, biases, normalization and any context parameters; proposed initial layout: **1,840,109** |
| Initialization | Fresh random weights and new training state; no existing neural checkpoint for initialization, teachers or the starting opponent pool |
| Later goal/personality extension | Aim to stay below 2M; allow at most **2.10M total** for the extra heads/context, without increasing the base trunk |
| Required inference path | CPU-prepared batches and a working **batched CPU NN evaluator**, as well as lab GPU batching; batch size is separate from the optimizer minibatch |
| Branching control | Dynamic rules-derived legal masks; policy top-K as an initial expansion shortlist, with protected alternatives and progressive unpruning |
| Current six v3 strategies | **Masked ISMCTS + existing v3 heuristics**: same search contract, existing completion/cash/merchant-beer evaluation and style preferences; no untrained network substitution |
| Game time options | Preserve **Easy 2s / Mid 5s / Hard 12s**, from `DifficultyCatalog.cs`; preserve the old-save 10s compatibility setting |
| Training v3 allowance | **1,000 ms per whole turn**, including both actions and replanning. Direct configuration, with no timing-equivalence calibration or qualification tournament |
| Active search choices after migration | Only `masked_ismcts`. Remove S0/S1/S2/S3 and separate R4 algorithm choices from active planners, configurations and commands |
| Search references | OpenSpiel ISMCTS and the original ISMCTS papers/example; Daiyu remains the neural model/batching reference |

“Masked ISMCTS” is our explicit application contract, not a claim that every implementation named ISMCTS already has these information restrictions. Mask both permitted inputs and the decisions made during simulations. Preserve four independent objectives; never combine the other three players into a coalition.

A policy-only evaluator, greedy legal completion and a cached legal incumbent are supporting components, not alternative search algorithms. Original Hard may remain a vendor benchmark with zero training weight. Old immutable banks/results stay available as historical evidence, but their S1/S2 opponents are removed from the active training pool after migration.

## 2. Papers and reference implementations

The following primary sources were reviewed. The design choices in subsequent sections are project requirements, rather than claims that these references implement our Steam rules or guarantee stronger Brass play.

| Reference | Useful material | Intended use |
| --- | --- | --- |
| Cowling, Powley and Whitehouse, [Information Set Monte Carlo Tree Search, 2012](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) | Information sets, determinization per iteration, legal-action availability and shared statistics; strategy-fusion limitations | Algorithm definition and information/legality review |
| [google-deepmind/open_spiel — ISMCTS, v2.0.2](https://github.com/google-deepmind/open_spiel/blob/v2.0.2/open_spiel/python/algorithms/ismcts.py) | Information-state keys, resampling callback, evaluator prior/value hooks, PUCT and visit-based selection | Primary code reference; preserve Apache-2.0 notices. Existing project dependency is already 2.0.2 |
| [AI Factory's author-written implementation article](https://www.aifactory.co.uk/newsletter/2013_01_reduce_burden.htm) and [GitHub-hosted mirror of its Python example](https://gist.github.com/kjlubick/8ea239ede6a026a61f4d) | Small readable ISMCTS loop and game interface: randomize hidden state, legal moves, apply and terminal result | Readable secondary reference for the C# port; the gist is a mirror, not the authors' maintained repository. Preserve its attribution if code is reused |
| Cowling, Powley and Whitehouse, [Parallelization of ISMCTS, 2014](https://edpowley.com/academic/papers/wcci2014-ismcts-parallelization.pdf) | Independent root trees and aggregation of root statistics | Reference for private per-worker trees; select worker counts for this project rather than importing a reported speedup |
| Goodman, [Re-determinizing IS-MCTS in Hanabi, 2019](https://arxiv.org/abs/1902.06075) | Hidden-information leakage through opponent models; learned leaf evaluation | Information-boundary caution. Do not add another algorithm branch or transplant cooperative Hanabi assumptions |
| He et al., [Deep Residual Learning, 2015](https://arxiv.org/abs/1512.03385) | Residual connections | Architectural reference for skip connections; our dense, structured-feature network is a project design, not an image ResNet or a pretrained vision model |
| Cazenave, [Batch Monte Carlo Tree Search, 2021](https://arxiv.org/abs/2104.04278) | Batched leaf inference and separate inference-cache/tree statistics | Reference for the batching mechanism. Its GPU/Go results do not establish CPU speedups in Brass |
| Huang and Ontañón, [Invalid Action Masking, 2020/2022](https://arxiv.org/abs/2006.14171) | Consistent legal masking for policy-gradient training | Support for legal masks; not a justification for treating low-prior legal moves as illegal |
| Chaslot et al., [Progressive Strategies for MCTS, 2008](https://cris.maastrichtuniversity.nl/en/publications/progressive-strategies-for-monte-carlo-tree-search/) | Initially restricted branching followed by progressive unpruning | Reference for reopening policy-deferred legal actions; the proposed K values below are Brass engineering defaults |
| [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) | Explicit inference thread pools and controls | Configure CPU inference threads within the same budget as simulation workers and Unity; do not assume runtime defaults fit this workload |

Do not copy ordinary perfect-information MCTS into the hidden-card game and merely mask network inputs. Also do not import OpenSpiel's two-player AlphaZero trainer as our four-player training loop.

During implementation, add a new reference/component manifest containing the selected upstream commit, file hashes, license notices and adaptation list. Resolve the 2.0.2 tag to an immutable commit. Preserve the original R0 pins; do not rewrite them to imply that this revision was already implemented.

## 3. Starting point in this repository

The closest existing implementation is `src/brass_ai/searchlab/ismcts.py`. Its `MaskedISMCTS` already uses OpenSpiel information-state nodes, PUCT/prior/resampling hooks, independently masked opponent policies, and completed-simulation backups. `spiel.py` supplies reconstruction, private recall, semantic actions and explicit chance handling. Reuse those parts as the basis of the main search, rather than beginning with another generic MCTS repository.

Required improvements before it becomes the baseline:

- Move it out of an optional/comparison namespace and remove dependencies on S2 proposal/search classes.
- Replace permanent proposal-width truncation with dynamic legality masks and progressive policy top-K expansion over the complete legal list, including enabling actions and every industry.
- Add asynchronous submit/poll inference and bounded pending simulation paths so a CPU search worker can produce a batch instead of blocking after each individual NN request.
- Expose completed root visit distributions as a new distillation target. The current R4 path explicitly does not export them as teachers.
- Preserve four-seat values and consistent seat mapping through parallel aggregation and terminal scoring.
- Reuse one turn's tree/budget across resource choices and the second action; invalidate it on actual observation/model/style incompatibility.
- Port the same semantics to an authored C# implementation for the current v3 game patch and research worker. The Unity patch must not require Python or OpenSpiel at runtime.

Keep R1 legality/restoration/private-observation contracts and reuse the R2 feature/target meanings. The current `learning/model.py` implements `daiyu-four-seat-dense/1.0.0`, with no residual blocks. Its 1,897,389-parameter medium model stays archived and is not used in the new run. The new residual architecture needs a new checkpoint identity and fresh initialization as specified in section 6; audited heuristic demonstration datasets remain reusable.

## 4. Tree, beliefs and masked simulation

### Information-state identity

Each root search operates for one actual decision-maker. Key a node by that player's permitted public/private observation, permitted recall, phase, actions remaining, stable seat identity and legal semantic choices. Include rules/model/style versions in the cache namespace. Exclude sampled opponent hands, unrevealed future cards, real hidden checkpoints, simulator routing UUIDs and world-sampling seeds.

Recall contains only what that player previously observed. Loading a save without an available private observation history must not recover it from secret engine events. Use the available permitted view and conservative uncertainty instead.

Two hidden worlds indistinguishable to the root share a node and its action statistics. Different newly observed hands or public opponent actions may produce different future nodes. They cannot change a current action before that information is observed.

### Belief sampling

At each iteration, sample a hypothetical world consistent with the root observation: card multiplicities, known hand, public discards, hidden setup allocations, hand/deck counts, wild supplies and era reshuffles. Search RNG is separate from the real game RNG. Future cards are sampled within that hypothetical world, never obtained from the live deck.

Advance through the actual rules engine and scheduler. Spending-based order can produce zero to six intervening opponent turns before the next ordinary root turn. Draws, income, forced debt choices, round boundaries and the Canal/Rail transition must resolve normally. A simulation cannot evaluate an unfinished payment as a settled action.

### Opponent decisions

Use **single-observer root search with independently masked, fixed opponent policies** as the initial Masked ISMCTS design. Other seats act through their own allowed observations and legal choices. The root tree does not optimize opponent nodes with returns conditioned on the root's private hand.

For v3, use cheap observation-based heuristic rollout policies. For trained AI, use frozen neural rollout policies from the population, with public styles or a broad sampled style mixture. Sample an opponent identity once per simulation and keep it stable throughout that simulation. Never invoke another full timed search recursively for every opponent choice.

This models plausible self-interested replies; it does not claim equilibrium play or optimal human prediction. Recording a four-seat value vector does not itself make opponent-statistic optimization safe.

## 5. Selection, expansion and complete-turn execution

Tree edges represent semantic choices, including action starts, payments, sales, resource sources and forced decisions. This uses the existing complete legal interface and avoids enumerating a huge fixed menu of whole turns before search starts.

One iteration:

1. Sample a permitted hidden world and locate the matching root information node.
2. Select existing legal edges using normalized action priors and the root player's own value component.
3. Expand a previously unvisited legal edge according to progressive widening, with positive exploration support across legal choices.
4. Complete any pending action and the remaining current-turn actions with permitted rollout policies before an ordinary leaf evaluation. Resolve replies until an appropriate settled root boundary; terminal results take precedence.
5. Evaluate the root's permitted leaf observation using the selected evaluator. Back up only a completed iteration; interrupted work cannot fabricate losses or visits.

Use PUCT inside ISMCTS, with legal-set/availability accounting where action availability varies. It is an internal selection rule, not a second user-selectable algorithm. Start with one further root-decision boundary as the normal rollout horizon. Longer budgets can deepen the same information-set tree, bounded by a clear meaningful-turn/choice limit; no S3 mode returns.

Progressive expansion must preserve loan/develop/scout setup moves, build–sell sequences, alternative beer sources and all legal Pottery actions. Use the dynamic masks and policy top-K schedule below; the shortlist is a search-allocation decision and never replaces the complete legal interface.

Choose among completed root visits with a modest persistent style influence and difficulty-controlled sampling over credible alternatives. Search should be able to prefer a tactically sound move over a style prior. Keep the v3 cash/credit/completion safeguards where feasible; distinguish a financial preference from a rules restriction.

Publish a fast complete legal incumbent before expensive work. Extract a complete current-action/turn continuation from information-consistent branches, validate it on an isolated reconstruction, and revalidate against fresh legal choices during execution. Reuse matching tree information for the second action with **the original deadline**. New information or a changed revision invalidates incompatible continuations; it never grants another full time allowance.

### 5.1 Dynamic Action Masking

Keep three concepts explicit in the request/replay contract:

| Mask | Meaning | Source |
| --- | --- | --- |
| Legal action mask | Whether a semantic choice is legal now, for this acting seat and phase | Complete R1 rules interface using that actor's permitted information |
| Padding mask | Whether a batched candidate cell represents a real choice | Batch collation, never a game rule |
| Search expansion mask | Whether a legal choice is currently opened for tree expansion | Policy top-K, protected alternatives and progressive widening |

Refresh legality after each state transition: action selection, card/payment/resource choice, each sale and merchant-beer choice, forced debt resolution, draw, and era/round change. Reuse a cached mask only for the exact compatible permitted state/phase/revision. The existing candidate provider already enumerates legal choices; preserve that interface rather than introducing a huge fixed action vocabulary just to mask it again.

Apply the legal/padding masks before normalization and sampling in both training and inference. Illegal/padded choices have exactly zero probability. For PPO, replay the recorded legal set/order when computing old/new probabilities and entropy. A terminal state bypasses action inference; an unexpected empty nonterminal legal set is an error, not an invented pass. A forced singleton can advance without a policy forward pass when no value estimate is needed.

Root-equivalent hidden worlds must produce the same current root legal set. Sampled opponents use their own legal lists privately. Never union opponent-only choices or sampled hidden feasibility into the real root mask. Legal availability counts in ISMCTS track actual availability, not how often top-K admitted an edge. Financial caution, personality and a low prior must not turn a legal action into an illegal one.

### 5.2 Policy Top-K Pruning with progressive unpruning

Adopt **temporary policy top-K pruning of tree expansion**, retaining every legal choice in the provider and base policy. Score all legal choices in candidate batches; K saves downstream simulation/expansion work, not the initial full-legal prior computation. Chunk large lists and normalize over the complete list, rather than silently truncating it to fit a tensor.

Initial proposed settings:

| Setting | Initial value / rule |
| --- | --- |
| Root policy shortlist | Top 16 legal semantic choices; use all if fewer |
| Interior root-observer node shortlist | Top 8; use all if fewer |
| Widening | `K(N) = min(number_legal, K0 + floor(2 * sqrt(N)))`, where N counts completed node visits |
| Protected additions | Add phase-relevant setup, recovery, sale/merchant-beer and industry representatives even when outside K |
| Tail exploration | Admit an outside-shortlist legal choice at the first expansion opportunity after the initial incumbent, then reserve every fourth new expansion for the remaining tail when nonempty |
| Tail distribution | Mix policy weighting with a nonzero uniform component over the deferred legal choices; use recorded independent RNG streams across replicas |

The active set is the union of the top-K shortlist, protected choices, already expanded legal edges and tail admissions. K is not a hard total-child cap. Rank by the legal NN prior with any later bounded root style bias, while preserving unbiased family/industry coverage. Keep previously expanded compatible children; never erase completed evidence because rankings changed. Single-observer opponent rollout policies still sample their own full legal distribution; do not create optimized opponent trees or permanently prune their plausible replies.

At relevant phases, protected coverage includes a feasible loan/develop/scout enabling choice, a feasible sale and merchant-beer choice, and at least one available build representative for each industry including Pottery. This is coverage, not a command to take those moves. Preserve different card costs and resource sources where they affect future play. Do not merge distinct discard choices merely because they look strategically similar; exact equivalence needs a rules/continuation argument.

Broad or unreliable priors favor earlier widening; a larger time budget increases K through completed visits. Deferred choices remain legal and reachable, but finite time still cannot guarantee exploration of every tactic. Report deferred prior mass, family/industry coverage, outside-K admissions and selected outside-K actions. In distillation, retain full-legal priors and record the search expansion mask separately; an unexpanded action is not an illegal or proven-bad label. Use a small comparison with top-K disabled within the same Masked ISMCTS backend to detect harmful pruning, not a revived multi-algorithm tournament.

## 6. Two evaluators, one algorithm

| Component | v3 heuristic evaluator | Trained NN evaluator |
| --- | --- | --- |
| Priors | Softmax of permitted v3 proposal scores and bounded persistent style preferences | Full-legal-choice logits conditioned on allowed state/style; root-only bounded goal bias after the later personality stage |
| Leaves | Existing industry completion, route, finance, beer/merchant and terminal evaluation, mapped to a bounded documented utility scale | Existing four-seat win-share outputs; placement/VP remain auxiliary predictions |
| Opponent rollouts | Cheap masked heuristic policies, no recursive timed search | Frozen masked neural policies, independently attributed to their own observations |
| Terminal result | Exact four-player utility and tie handling | Same |
| Runtime | Authored C#; CPU-only | Required batched CPU inference, plus shared lab GPU inference, with identical observation/choice contracts |

Heuristic utilities are not calibrated win probabilities. All replicas must use the same documented scale, otherwise PUCT and root-statistic merging become inconsistent. At a neural leaf, use the root component of the correctly mapped four-seat output; do not combine private views from four players into a privileged input.

### 6.1 Main model: MLP-ResNet and parameter limits

The main-training ceiling is **2,000,000 total learned parameters**, not two million trunk parameters plus heads, and not merely parameters currently receiving gradients. Count every unique weight/bias, normalization affine parameter, embedding/context projection and policy/value head, including frozen parameters. The later goal/personality model must include its extra heads/context in the same count, with a maximum of **2,100,000 total**. This 5% allowance is only for those later extensions, not a larger Step 5 trunk. Optimizer state, activations and search trees are separate memory costs. Quantization changes storage, not the parameter count. Offline AURORA encoders are separate tools, not hidden parts of the playing network.

Use the current structured feature dimensions, **3,207 state features and 184 candidate features**, as the planning baseline. No convolution or image encoder is needed. The following concrete layout has **1,840,109 parameters** by layer-shape calculation:

| Component | Proposed structure | Parameters |
| --- | --- | ---: |
| State/context stem | Linear 3207 → 304, GELU | 975,232 |
| Shared residual trunk | Four blocks: `x + Linear304(GELU(Linear304(LayerNorm304(x))))` | 744,192 |
| Final state normalization | LayerNorm 304 | 608 |
| Candidate encoder | Linear 184 → 128, GELU, LayerNorm 128 | 23,936 |
| Candidate policy scorer | Concatenate state 304 + candidate 128; Linear 432 → 128, GELU, Linear 128 → 1 | 55,553 |
| Four-seat value group | Linear 304 → 128, GELU, Linear 128 → 12 | 40,588 |
| **Total** | Biases and normalization parameters included | **1,840,109** |

Each residual block has two 304 → 304 affine layers and one pre-normalization, with an identity skip. Prefer LayerNorm over batch-dependent normalization so another leaf in the batch cannot change a position's prediction. Start with no dropout in the search evaluator and optionally zero-initialize the last affine layer of each new residual branch for a stable identity start. Preserve `win_share_4`, `placement_4`, `vp_4`, their existing transforms and stable-seat mapping.

Encode the state once and score its variable candidates in batches/chunks. The 128-wide scorer limits work repeated per candidate. This shape leaves **159,891 parameters** below 2M; an initial allowance of up to 100,000 for later goal/style extensions would total at most **1,940,109** with unchanged features. Aim to fit there first; if later context/goal needs exceed it, the complete extended model may reach 2.10M without widening the base trunk. Recount after every schema change. Step 5 need not add an unused goal head now. A changed input dimension invalidates the numerical total above and requires a fresh count before accepting the configuration.

During implementation, assert `sum(p.numel() for p in model.parameters()) <= 2_000_000` for Step 5 at construction, checkpoint validation and export; the explicitly versioned later goal/personality variant uses `<= 2_100_000`. Record total and trainable counts separately, and itemize any parameters above 2M as extension overhead. Give the base architecture a new explicit ID, proposed `brass-four-seat-mlp-resnet/1.0.0`. Retire the 8M architecture from the active main-training roadmap; smaller ablations also remain within the base cap. The proposed count is a design calculation, not evidence of a built or trained model or stronger play.

### 6.2 Fresh training and the later handoff

Initialize all residual-model weights from a declared seed, with fresh optimizer, scheduler, scaler, counters and population state. **Do not load any existing neural checkpoint**, including bootstrap files, old dense models or previously trained compatible models. Do not use them for policy/value distillation or as starting neural opponents. New-run metadata records random initialization, the seed and no parent checkpoint; preserve earlier files as archived evidence.

Fresh initialization can still be followed by the planned audited **heuristic demonstration imitation phase**, then self-play; demonstration data is not pretrained weights. Reuse eligible v3 demonstration observations, executed legal choices and terminal outcomes, or regenerate them using the new 1s Masked ISMCTS v3 bank. Enforce the existing rules/schema/provenance and setup-based split checks. Exclude old neural behavior/search targets and neural-generated replay from the fresh run rather than indirectly importing their policies.

Build the neural population only from snapshots produced by this new run. Before historical snapshots exist, assign their planned share to current-policy self-play; retain the 40% → 20% → 10% → 5% v3-table curriculum. New Masked ISMCTS teachers use this run's frozen neural snapshots, with no imported checkpoint teacher. Resuming an interrupted instance of this same run is allowed after normal lineage/schema checks; that is not a pretrained initialization.

The later personality stage still begins **after this fresh Step 5 run finishes**, using its selected newly trained residual checkpoint. It does not substitute an old experiment's model. Keep 16 visible presets / 128 hidden variants, with the later 2.10M total allowance and its own versioned fine-tuning run.

## 7. CPU/GPU allocation and the one-second lab rule

| Role | Allowance and resource policy |
| --- | --- |
| Game v3 | Existing 2/5/12-second options; old-save compatibility 10s; default one CPU search worker initially |
| Lab v3 demonstrations/opponent tables | **1s per whole turn**, warm persistent workers; initially 8 concurrent heavy games × 4 private search replicas |
| Lab trained-NN search | Same Masked ISMCTS backend; start with a separately declared 1s lab preset, CPU batch preparation and shared batched GPU/CPU NN inference; keep desktop presets independent |
| Ordinary policy-only self-play | Existing fast neural actor collection; policy-only is a throughput/control mode, not another search algorithm |

The 1s v3 training allowance is an accepted requirement. **Remove `lab-calibrate`, the 25-game qualification prerequisite, the 80% reference-work test and automatic fallback to a 3s teacher for this new path.** Do not replace them with another budget-admission test. Configure `milliseconds: 1000` and activate the new training bank directly with explicit source/rules provenance. Record that its training use is user-directed and that its playing strength is unmeasured; do not reuse old qualification reports as proof for the changed policy.

Use independent root trees with private simulated worlds, RNG and node storage. Sum completed root visits and four-seat return sums by compatible legal semantic ID; do not average replica means equally when their completed counts differ. Share the same deadline across replicas and both actions. Bound global heavy search workers at 32. When all 32 are active, initially allow 24 ordinary game workers and reserve the remaining eight of the reported 64 CPUs for inference/encoding/coordination/learning; ordinary collection can return to its existing 32-worker setting when search releases capacity. Account for inference threads in the same allocation. Do not fan out every legal action into another process or add nested BLAS threads.

### 7.1 Required CPU batching for Masked ISMCTS + NN

CPU batching is a required implementation component, including when the lab uses GPU inference. It has two levels: CPU workers prepare multiple independent pending simulations and batch their feature/candidate work; a CPU NN evaluator can run a batch of positions directly when CPU is selected. Merely increasing PPO `minibatch_size`, adding simulator processes, or batching only on CUDA does not meet this requirement.

Use an asynchronous leaf/prior request interface over the existing inference service:

1. Advance independent simulation paths on the CPU until they need a policy/value prediction; retain bounded, privately owned path/world snapshots.
2. Submit immutable permitted observations, complete legal choices/masks, observer/style/model IDs, continuation tickets and the original turn deadline. Continue other pending paths instead of synchronously waiting on each NN call.
3. Collate compatible same-model requests into variable-position/candidate batches. Encode a state once, group similar candidate counts to reduce padding, and score long lists in chunks with a global legal normalization. Never drop choices to meet a batch-size limit.
4. Run the batch on CPU or GPU through the same evaluator contract, route each result to its exact request, recheck expiry/cancellation/model/observation, then resume that private simulation.
5. Back up each distinct completed simulation once. Expired or cancelled paths release reservations and private buffers without contributing completed visits or returns.

Bound pending paths per replica. The current synchronous evaluator must gain submit/poll or an equivalent cooperative batch loop; one worker must be able to fill a small CPU batch even during one live game. Rules operations with process-global engine state remain serialized within their owning host. Pending paths need isolated saved state and restoration; they must not interleave mutable engine contexts or create a new OS process for every leaf.

Use temporary in-flight edge reservations to discourage repeatedly selecting an unresolved path. Keep reservation counts separate from completed counts; if virtual-mean estimates are used for selection, remove them exactly on completion/cancellation. They never enter replay teachers or final root-visit statistics. Coalesce identical NN inputs into one inference request, but do not multiply one identical pending simulation into several fictitious visits. Different genuinely simulated worlds may share a cached prediction while retaining their own completed path evidence.

### 7.2 Initial batch/resource presets

These are initial implementation settings, not measured speed claims or new timing-admission gates:

| Execution setting | Position batch cap | Maximum queue wait | CPU/resource setting |
| --- | ---: | ---: | --- |
| Laptop CPU NN search | 8 | 0.5ms | Start with one search worker, up to eight pending paths, one persistent evaluator, two intra-op threads and one inter-op thread |
| Lab CPU NN evaluator | 16 | 1ms | Start with four intra-op threads, one inter-op thread; charge these against reserved CPU capacity |
| Lab GPU NN evaluator | 64 | 1ms | Batch across games/replicas; CPU preparation still required; initially up to four pending paths per search replica |

Flush when full, on the queue-wait limit, when no other useful search work can run, or when the turn deadline approaches. The first root request must not wait for an impossible full batch. The original whole-turn allowance includes encoding, queueing, inference, simulation and validation. Position caps are distinct from candidate-cell/memory limits: initially use up to 8,192 padded candidate cells per CPU batch and the existing lab GPU capacity, splitting candidate work rather than truncating legality. Batch size one remains a valid low-latency fallback within the same batching implementation.

Keep one warm model/evaluator per necessary device/model cache entry, explicit bounded thread pools and no per-leaf model loads. A CPU comparison may try batch caps 1/4/8/16 and modest thread counts on a small representative position set; retain useful batches based on completed search work and latency, not a desire to fill memory. Intel GPU remains a later selectable inference device with its own batch limits. Heuristic v3 does not need NN inference batching, but shares batched work scheduling, private root replicas, masking and deadline handling.

### 7.3 Other throughput improvements and telemetry

Reconstruct one sampled world per iteration and advance it in place where safe, rather than rebuilding a host at every tree edge. Reuse checked feature inputs, exact permitted-view NN caches, warm workers and compatible tree continuations. Separate cached NN predictions from search visit/return statistics. Keys include model/architecture, observer, permitted history, legal order and style; neither batch membership nor real hidden state may change a prediction.

Report position/candidate batch sizes, padding ratio, mask density, pending/duplicate/expired requests, cache hit rate, legal-generation/encoding/queue/inference/host times, completed simulations and distinct root alternatives per second, and p50/p95 whole-turn latency. Track worker occupancy and GPU utilization as diagnostics, not success targets. The goals are more useful completed search and adequate candidate coverage within the existing time limits; batched CPU inference is required but a speedup still needs measurement.

Stop dispatching work before the deadline and return the legal incumbent if too little search completes. A slow non-preemptible engine/inference call may finish late; record actual latency and do not restart the turn clock or silently change the 1s setting. Release each worker only after its actual computation ends. Node/choice/memory caps are guards, not the old 240-node planning target.

## 8. Planned v3 game patch

Implemented in separately versioned package 1.4.9:

- Add the C# Masked ISMCTS engine and evaluator/policy interfaces, shared by the research worker and `BrassLocalAI` deployment sources. Replace the active beam/reply planner while retaining reusable rules, greedy completion, cash, conversion and tactic helpers.
- Route `TurnCoordinator` to this backend through the existing `AiExecutionScheduler` and `IAiExecutionBackend` lifecycle. Preserve cancellation, typed tickets, detached incumbents, stale-response rejection and buffer release.
- Preserve `DifficultyCatalog` values and save contract. Keep the six v3 strategy labels and their existing preferences. Change the active algorithm/profile version in a new package; do not relabel the release v4 merely because its search changed.
- Validate old saved assignment hashes against their original embedded profiles, then map a detached runtime profile to Masked ISMCTS. Preserve saved strategy/intent/difficulty identities. Never rewrite immutable archived profiles or continue old search merely to load an old save.
- Preserve the current UI enhancements and execution resource management: city highlighting, link replay, completion statistics, custom-game choices, temporary frame cap and restoration. Avoid changing the core rules or public game flow.
- Build against the current game patch staging inputs and produce a separately versioned, rollback-capable patch package with source/binary fingerprints and installation instructions. Select the next free package version when implementing; do not assume another patch has not been installed meanwhile.

Research remains built against its pinned rules runtime; the game patch uses its separately supported deployment fork. Both use equivalent Masked ISMCTS semantics and matching scoring helpers. Research optimization remains four-player. Preserve existing two-/three-player loading/play compatibility without claiming optimization for those counts.

## 9. Training, evaluation and distillation migration

Keep audited heuristic imitation after random initialization, nonforced PPO actor normalization, separate critic sampling, bounded populations, checkpoint cleanup and smaller evaluation cadence. Preserve the v3-table curriculum 40% → 20% → 10% → 5%; current-policy and this new run's historical self-play remain dominant as training matures. Section 6's fresh-start restriction applies to training, teacher generation and evaluation lineups: no pre-existing neural checkpoint enters the new workflow.

Freeze a new Masked ISMCTS v3 bank with all six styles. Five training identities enter the active pool at the direct 1s budget; Pottery Flex's **opponent identity** remains evaluation-only. Pottery actions and learner exploration remain fully available. Update demonstration generation, routine evaluation and new bank provenance together. Do not silently reuse changed actions under the old v3 policy ID.

For the trained AI, replace the main S2 teacher schedule with Masked ISMCTS teachers. Search remains bounded to a small wall-time share. Do not require all ordinary self-play decisions to spend a second searching; use fast neural actors for throughput and the one selected search backend for teacher generation and playing-strength evaluation.

Create a new teacher contract for **completed ISMCTS root visits**, including legal order, model/observer identity, simulation counts, priors, normalized visits, sampled-world counts, clock/worker settings and source fingerprint. Do not present these as the old fresh-finalist confirmation labels. A meaningful-choice teacher needs completed evidence on at least two alternatives; empty, interrupted or singleton-only statistics produce no search-distillation update.

Initially mix a modest amount of normalized completed visits with the full-support base policy prior for distillation. This reduces pressure to erase unvisited industries under short searches. Record the mixture weight and compare learning/economics on the existing small development panels. Search external actions/visit probabilities are not PPO neural behavior likelihoods; keep their distillation loss separate. Full completed self-play games still provide four-seat terminal critic targets.

Use a new training recipe/run identity, proposed `step5-training/2.2.0`, with random initialization, `search_backend: masked_ismcts`, the explicit residual architecture and 2,000,000 base parameter cap, evaluator/batching/worker/clock settings, masking/pruning settings and new opponent/teacher contracts. Do not pass an existing `--initial` checkpoint or copy a previous run's population/optimizer state. Preserve old runs and their source signatures. Keep neural inference, hashing, batching and retention compatibility checks; dropping a timing tournament does not permit malformed replays or hidden information.

Retire separate S2/R4/S3 comparison tournaments and S3-only distillation commands. Maintain the existing 8-game monitor / 16-game broader panel and shortlist-only strength evaluation. Compare the selected search against the same frozen NN without search when useful; there is no requirement to run or promote one of the retired search algorithms.

## 10. Implementation order and component map

| Package | Work and principal files | Completion criterion |
| --- | --- | --- |
| M0: contracts/references | New reference pin/map and `configs/contracts/` search/teacher/model schema; extract neutral observation/turn/deadline helpers | One algorithm identity; defined permitted inputs, four-seat mapping, separate mask kinds and total-parameter cap |
| M1: main NN baseline | Add versioned, randomly initialized MLP-ResNet in `learning/model.py` with fresh-run/checkpoint/export contracts; promote `searchlab/ismcts.py` into `src/brass_ai/search/`; extend `learning/batching.py` and `recovery/inference.py` for async CPU batches; retain `searchlab/spiel.py` as an adapter; add progressive top-K and visit targets | ≤2M residual model; working batched CPU Masked ISMCTS + NN, private opponent policies, dynamic masks and bounded pending paths/root forest |
| M2: v3 C# implementation | New shared C# search source; replace active `BrassLegacyAI/Planner.cs` and `BrassLocalAI/Policy/Planner.cs` algorithms; reuse `Evaluator`, `CashManagement`, `Conversion`, `Tactics` | Current-turn planning, masked replies and legal incumbents use ISMCTS semantics in research and deployment |
| M3: game package | `Policy/Coordinator.cs`, profile catalog/versioning, Unity bridge, build/package/install manifests | All six v3 strategies use the new backend; exact three time options and old saves preserved |
| M4: direct 1s lab bank | New immutable opponent bank/pool/worker transfer; recovery demonstration/execution/league/evaluation routing | Active v3 tables use 1,000ms without a calibration/report-admission tournament; old bank evidence untouched |
| M5: training and retirement | Step 5 fresh residual-model initialization, heuristic imitation, new-run search teachers/population, active configs/CLI, lab guide, references and runtime bundles | No pre-existing neural checkpoint/replay imported; audited heuristic data and new-run snapshots only; no active S1/S2/S3/R4 choices/import dependencies or old optional experiment schedule |

M1 and M2 implement one contract in Python and C#, not two competing search designs. Finish M3/M4 before switching the bank used for demonstrations. Finish M5 before starting a long new search-distillation run. Preserve all productive recovery work; there is no need to redo R0/R1/R2 or the later personality plan.

After reusable helpers are extracted, remove the old active search cores and imports: S1 beam selection, `BrassLegacyAI/ReplySearch.cs`, old `step5/search.py` algorithms, `searchlab/s3.py`, S2/S3/R4 variants in `recovery/parallel.py`, and the `r4r5.py` active comparison interface. Historical frozen bank sources/results remain immutable. Existing OpenSpiel adapter and batching infrastructure can remain implementation dependencies without being a selectable R4 search variant.

## 11. Focused implementation checks

Use a compact engineering check, not another lengthy search tournament:

1. Root private-world invariance and opponent-policy masking, including recall/cache keys, hidden setup discards and era reshuffles.
2. Complete legal two-action sequences, payments/debt, enabling actions, merchant beer and a feasible Pottery sale; all four seats remain active.
3. Interrupted simulations leave no fake visits/returns; root forests merge counts with correct seats and private channels.
4. Game patch compilation and a short real turn/save-load/cancellation check with the existing 2/5/12 settings; resource/UI lifecycle remains intact.
5. A small residual NN teacher/optimizer/resume/export check demonstrates the base total count ≤2,000,000, fresh initialization with no existing-checkpoint import, valid same-run resume, separate search distillation and source/behavior provenance; cleanup preserves current models and datasets. Later goal variants separately enforce ≤2,100,000 total.
6. Batched versus individual CPU predictions agree within declared numeric tolerance across observers, candidate counts and masks; cancelled reservations leave no visits, and a single live game can issue a nontrivial batch without deadlock or shared-host state leakage.
7. Dynamic masks match the complete legal provider through multi-step actions. Top-K does not alter legality or PPO probabilities; protected Pottery/setup/merchant choices and tail unpruning remain reachable. A small same-backend top-K-disabled comparison checks coverage and useful work without a long tournament.

**No training-v3 clock calibration or 25-game budget qualification is part of these checks.** The one-second allowance is configured directly as requested. Strength, improved enjoyment and equal-strength personalities remain measured outcomes of training and later playtests, not prerequisites for selecting this search design.

## 12. Deliverables for the implementation turn

The implementation provides the freshly initialized ≤2M MLP-ResNet and fresh-run data/population rules, the single Masked ISMCTS search package with required CPU batching, dynamic masks and progressive policy top-K, the built v3 game patch with unchanged time choices, the frozen direct-1s lab opponent bank/private runtime transfer, new Step 5 teacher/configuration wiring, retirement of old active search modes and revised conda commands. The later goal/personality stage has a 2.10M total limit. Historical pins, banks and results remain immutable. Later goal/personality extensions retain the separate 2.10M total cap.

**1 October 2026 implementation:** F0–F6 of the [Masked ISMCTS strength fix](MASKED_ISMCTS_STRENGTH_FIX_PLAN.md) are implemented. [Package 1.4.11 and its evidence](../docs/MASKED_ISMCTS_STRENGTH_1.4.11.md) preserve game clocks/UI/rules and introduce `bank-981744ddae4aaf07f1cf`. Neural training remains fresh 1.84M MLP-ResNet with batched raw-value Masked ISMCTS and a supported observation frontier; heuristic v3 uses a separate short horizon and supported sibling normalization. [Lab commands](../docs/TRAINING_RECOVERY_LAB.md) use new run/data names.
