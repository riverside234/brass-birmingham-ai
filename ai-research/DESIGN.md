# Four-player opponent design

**30 September implementation update:** M0–M5 of [Masked ISMCTS](MASKED_ISMCTS_REVISION_PLAN.md#10-implementation-order-and-component-map) are implemented. Current training uses a fresh 1,840,109-parameter MLP-ResNet, CPU/GPU batching, separate legal/padding/expansion masks and completed root-visit teachers. Package 1.4.9 switches all six v3 styles to the shared C# backend, preserving 2/5/12 and old-save 10 seconds plus resource/UI management. The new lab bank uses one second directly. Historical S1/S2/S3/R4 variants and checkpoint-import commands below are archival. Use the [current lab guide](../docs/TRAINING_RECOVERY_LAB.md) and [exit record](../docs/MASKED_ISMCTS_EXIT_CHECKS.md).

**30 September 2026 selected revision:** [Masked ISMCTS](MASKED_ISMCTS_REVISION_PLAN.md) supersedes the earlier search-stage/algorithm-comparison recommendations below. The new trained AI's baseline is **Masked ISMCTS + NN**; current v3 uses the same search with its existing heuristics. Planned game patch keeps 2/5/12-second options; lab v3 uses 1 second directly without calibration. Preserve the model, privacy, personality and four-player requirements. The implementation update above records the current runtime.

**Updated model choice:** fresh training of an **MLP-ResNet with ≤2M total parameters**, initially 1,840,109 by the proposed layer shapes. No existing neural checkpoints enter initialization, distillation or the starting population. CPU inference batching is required; dynamic legality masks and progressive policy top-K control search branching. Later goal/personality additions may reach **2.10M total**, with no widening of the base trunk. See sections 5–7 of the selected plan for the concrete design.

Design recommendation, 27 September 2026. No new model, inference runtime, or mod was installed during this follow-up research.

29 September implementation update: [R4/R5](../docs/R4_R5_SEARCH_LAB.md) now provide the required additional OpenSpiel and S3 experiments, with independent outputs and frozen-checkpoint evaluation alongside Step 5. Current population training and the installed game remain unchanged. Promotion requires correctness and measured equal-budget benefit; [exit evidence](../docs/R4_R5_EXIT_CHECKS.md) does not establish stronger or human-like play.

Implementation update: [R2's scaffold](../docs/R2_TRAINING.md) now provides fresh models at 473,229 / 1,897,389 / 7,797,901 parameters, four-seat targets, permitted-view batching, training/resume and checked CPU ONNX export. This implements the architecture experiment below, not calibrated playing strength or personalities. The [exit record](../docs/R2_EXIT_CHECKS.md) separates local CPU checks from the still-unverified lab/GPU and live-game work.

Revised thinking-time design: [SEARCH_DESIGN.md](SEARCH_DESIGN.md) specifies complete-turn planning, sampled opponent responses, information restrictions, adaptive time allocation and controlled experiments. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) integrates those stages with the rules interface, state restoration, game bridge and training work. These are proposed designs, not measured strength or speed results.

Selected implementation references: **Daiyu AI first, with OpenSpiel for multiplayer search mechanics and comparison experiments.** [REFERENCE_INTEGRATION.md](REFERENCE_INTEGRATION.md) records pinned revisions, source files to adapt, four-player corrections, the additional OpenSpiel adapter and delivery gates.

The earlier complete-turn planning proposal remains part of this design. The [unified requirements checklist](IMPLEMENTATION_PLAN.md#unified-requirements-from-the-earlier-search-design) connects its planning, uncertainty, personality, timing, model-size and rules requirements to implementation and acceptance. Daiyu/OpenSpiel components are selected to implement these requirements.

The [legacy Hard optimization plan](LEGACY_HARD_AI_PLAN.md) adds an earlier CPU-only opponent family: scored rules and complete-turn planning with Balanced, Rail Tempo, Cotton, Goods-and-Beer, Supplier and Pottery Flex profiles. All six are tuned and evaluated specifically for four-player games; the adapter preserves the existing AI path for two-/three-player games. This family shares the new legal interface and planner, preserves separately labeled stock benchmarks, and is intended to supply stronger heuristic opponents for training without requiring a neural model.

**User priorities**

Train and optimize for four players. A human normally faces three AI opponents. All inference should run locally on the Core Ultra 5 135U laptop; the RTX lab server is for training. Longer thinking times are acceptable when they improve play. The aim is enjoyable, varied opponents with coherent personalities. Easy, Medium, and Hard should all be available, with new Easy measurably stronger than the original Hard. That strength requirement is an acceptance target, not an achieved result.

**Recommended starting point**

One shared **MLP-ResNet policy/value model with at most 2,000,000 total parameters** for main training, initialized freshly from a declared seed. Include state/candidate encoders, all heads, biases, normalization and context parameters in that count, even when frozen. Preserve the existing style/intent input for each seat. Policy outputs score variable legal choices; value outputs estimate outcomes for all four seats from the observing player's permitted information. The neural population grows from this fresh run's own snapshots, alongside the declining v3 heuristic curriculum.

Use Masked ISMCTS to share information-state statistics across sampled hidden worlds, connect both actions through the same tree, and simulate opponents from their own permitted observations. Evaluate settled leaves with the NN; longer budgets may deepen the same tree. A larger network is not required to adopt this design.

Use a 3,207 → 304 state stem, four pre-LayerNorm residual MLP blocks of width 304, a 184 → 128 candidate encoder, a 128-wide policy scorer and a 128-wide four-seat value group. This proposed shape totals **1,840,109 parameters** and leaves 159,891 below 2M. It is a new architecture: the current 416-wide R2 dense model has no residual blocks and is not reused as weights. Encode the board once per position and score candidates in batches/chunks. Preserve connectivity, resource access, industry development, public cash/income, order, spending, era, own cards, remaining actions, allowed history and soft intent. Use LayerNorm so predictions do not depend on other positions in a batch; stable seat identity remains distinct from turn order.

Adapt Daiyu's existing encoder/scorer structure and batched training/search workflow to our observation and legal-choice contracts. Replace scalar/root-relative value shortcuts with separately learned win-share, placement and auxiliary score estimates for all four stable seats. OpenSpiel's actor-specific return-vector backup is the multiplayer reference; its information-set interfaces guide a separate search comparison. Build the initial simulator as a C# host over the Steam rules engine and validate it against the declared rules contract. An optional Daiyu Rust backend must first pass rules and trace parity checks. These choices reuse reference code without making a second game engine or OpenSpiel runtime a prerequisite for local play.

MLP-ResNet is the selected main architecture. Any smaller residual ablation stays within the main-training cap; the previous 8M comparison is retired from the active roadmap. Retain the existing R2 model files only as historical artifacts. Fresh training may use audited heuristic demonstration data, followed by self-play and search teachers generated from the new run; it does not import old neural weights or their replay targets.

| Selected model stage | Parameters | Float32 weights | Float16 weights |
| --- | ---: | ---: | ---: |
| Proposed fresh MLP-ResNet | 1,840,109 | About 7.36 MB | About 3.68 MB |
| Main-training ceiling | 2,000,000 | 8 MB | 4 MB |
| Later goal/personality ceiling | 2,100,000 | 8.4 MB | 4.2 MB |

These are parameter-storage calculations, not total RAM requirements or measurements. The runtime, activations, game state, and search tree require additional memory. Validate the effect of reduced precision on move ranking and match strength. The earlier 1–4 MB suggestion was a compact prototype budget, not a hardware ceiling.

**Intel graphics inference**

Intel lists the 135U's integrated Intel Graphics GPU with four Xe cores and support for OpenVINO, ONNX Runtime, and DirectML. It also lists an Intel AI Boost NPU. [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/237328/intel-core-ultra-5-processor-135u-12m-cache-up-to-4-40-ghz/specifications.html).

OpenVINO explicitly supports inference on Intel integrated GPUs. Its guidance includes float16 inference and batching requests. [GPU documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html).

The deployment requires CPU rules/search workers and a **batched CPU NN evaluator**; Intel GPU is another possible inference device. CPU workers gather bounded asynchronous leaf requests, encode permitted observations and collate variable legal candidates. Start laptop CPU batches at up to eight positions with a 0.5ms flush limit and bounded inference threads; lab GPU batches start at 64 with 1ms flush. Avoid waiting for a full batch when the root/deadline needs a result. Benchmark completed search work and latency while Unity runs. The GPU shares laptop power, memory and rendering resources, so support does not establish a speedup. NPU optimization can wait.

The earlier tests measured the stock rules and stock AI on CPU. They did not test neural inference or the GPU driver/runtime. A read-only CIM query for adapter/driver details was denied by the local environment; no driver or inference software was changed.

**Existing code and pretrained models checked**

| Source | Verified useful material | Limits for this project |
| --- | --- | --- |
| [Yuiffy/brass-birmingham-daiyu-ai](https://github.com/Yuiffy/brass-birmingham-daiyu-ai) — primary reference | Already supports 2–4 players; Rust game engine, PyTorch state/action encoders, candidate scorer, self-play and batched search. | Checkpoints are excluded and strong human-level play is not established. Some search paths infer opponents' values by distributing non-root win mass; replace these with four-seat predictions. Engine/schema differ from Steam. |
| [VyqreL/brass-birmingham-rl-model](https://github.com/VyqreL/brass-birmingham-rl-model) | MaskablePPO training code, with 512/512/256 policy and value layers, and a separate Python game implementation. | `brass_env.py` computes the opponent as `1 - current_player`, retaining a two-player reward assumption despite accepting four players. No weights appeared in the checked tree and no releases were listed. It is a reference, not a validated four-player training base. |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) — multiplayer companion reference | N-player MCTS/max-n-style backup, information-state resampling, ISMCTS and bot evaluation scaffolding. | Requires a game adapter and explicit masking of evaluators/bot callbacks. Full-state MCTS does not solve hidden information. The standard Python AlphaZero trainer rejects non-two-player games; ISMCTS is listed as lightly tested. |
| [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | Established implementation of PPO with legal-action masking; useful for a training baseline. | Requires a correct four-player wrapper or per-player trajectories. Simply alternating four players' rewards inside a normal single-agent trajectory is not the intended training setup. |

I did not find a verified, downloadable, strong four-player Brass checkpoint that can be plugged into this Steam installation. This is a bounded search result, not a claim that none exists anywhere. Reuse architecture/algorithm ideas and train weights against the verified engine. Even an available Brass checkpoint would need matching state and action encodings, compatible rules, and independent strength tests. The promising repositories were read, not installed or executed.

Inspected repository tree revisions: Daiyu `387840dd7a784f2c05fe561cb381e4f7d7bc590b`; VyqreL `7ec8b5fff9f60445a00b2384b81952b942118b43`. Sources for specific findings: [Daiyu model](https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/training/model.py), [Daiyu status](https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/README.md), [PPO environment](https://github.com/VyqreL/brass-birmingham-rl-model/blob/7ec8b5fff9f60445a00b2384b81952b942118b43/brass_env.py).

OpenSpiel was additionally inspected at `48401890ee9857e611678302371378175a8e4c6b`: [n-player MCTS](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/mcts.py), [ISMCTS](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/ismcts.py), and [AlphaZero trainer](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/alpha_zero/alpha_zero.py). The integration plan turns Daiyu and OpenSpiel into the selected working references; the other repositories remain optional comparisons.

**Thinking time and difficulty**

Use the following staged planner, specified in detail in [SEARCH_DESIGN.md](SEARCH_DESIGN.md):

1. First produce a complete legal incumbent turn, then generate diverse alternatives. Preserve enabling combinations such as loan-then-build and develop-then-build even when the first action looks weak by itself.
2. Compare those plans across matched samples of unseen cards and plausible opponent behavior. Choose one current-turn plan across those samples; do not allow the second action to depend on an unrevealed opponent hand.
3. Simulate the real scheduler to the root player's next decision. Round spending can reorder players, so forecasting exactly one turn for each opponent is insufficient. Resolve relevant income, debt and era transitions.
4. Use fast, independently motivated, observation-based opponent policies. Evaluate nonterminal leaves from the root player's permitted information. Implement future observation-tree search as an additional experiment; select it for release only if it beats the simpler planner at the same laptop time budget.
5. Recheck finalists on fresh scenarios, then use personality to select among credible near-best plans. Commit one complete action, refresh the live state and validate the continuation with the remaining turn budget.

Extra time can broaden candidate turns, reduce uncertainty about responses, or extend the useful horizon. Allocate it according to what is changing the decision. Repeating a biased opponent model or inaccurate value estimate more often will not fix it. Stop early when a choice is clear or further distinctions are too small to improve the play experience.

Keep the entire search fair: exclude actual opponent hands, future deck order, private seeds and the identities of face-down setup discards. Each simulated opponent sees its own allowed view; even its search statistics must not indirectly exploit another player's private cards. Persistent statics, state restoration and the audited rules defects must be addressed before trusting search-generated training data.

| Difficulty | Proposed maximum thinking budget per AI's whole turn | Intended behavior |
| --- | ---: | --- |
| Easy | 1–2 seconds | Complete-turn planning with limited scenario comparison and calibrated variation |
| Medium | 3–5 seconds | Broader plans and replies; improved resource and turn-order timing |
| Hard | 8–12 seconds | More carefully compared scenarios and selective deeper planning where validated |

These are configurable design budgets, not measured speed or established strength levels. One deadline covers the whole turn, including candidate generation, simulation, inference and validation; action two does not start a fresh budget. Return early and provide a Play Now action using the current complete incumbent. Three opponents at the Hard limit imply 24–36 seconds of thinking between human turns, plus animations. Benchmark an optional 20–30-second limit separately; three 30-second turns can mean 90 seconds of thinking. Do not label it stronger before testing.

Difficulty combines calibrated policy/checkpoint, search effort and credible-choice variation; a separate speed setting controls waiting. Use the fresh residual model and one Masked ISMCTS backend. Recompute dynamic legal masks at each semantic phase; policy top-K starts with 16 root/eight interior choices, with protected setup/industry choices and progressive unpruning. Keep complete legal priors and do not confuse deferred moves with illegal actions. Small comparisons can vary batching and top-K within this backend at the existing whole-turn caps. Test CPU/GPU backends while Unity runs, with rotated seats, varied setups and human feedback; improvement remains measured.

**Personality and enjoyable play**

| Personality | Typical preference |
| --- | --- |
| Industrialist | Industry development and planned sequences of profitable sales |
| Networker | Expansion, contested connections, and link scoring |
| Supplier | Coal, iron, and beer timed around demand from the table |
| Investor | Cash flexibility, income stability, and longer-term investments |

Personality is a soft preference, not an instruction to repeat one strategy regardless of the board. Store a modest strategic intent per seat, train with it, and preserve it across turns until conditions justify changing it. Search compares viable ways to pursue that intent; bounded style preferences select among credible near-best complete turns. Calibrate acceptable estimated quality loss separately for each difficulty. Maintain match-strength checks because neural values are imperfect estimates, and check that extra search does not make every personality behave identically.

After this fresh Step 5 run finishes, reuse its selected newly trained residual checkpoint for one shared style-conditioned policy/value/goal model. Preserve that learned backbone and four-seat heads; add the small goal/context extension in a separate versioned run. Aim for at most 100,000 added parameters, which would keep the proposed base below 2M; if necessary, the complete extended model may reach 2.10M without widening the trunk. The [personality plan](PERSONALITIES_AND_DIFFICULTY_PLAN.md) specifies labels, losses and compatibility. Bounded root goal priors support soft intent; candidate coverage remains necessary because finite search can miss tactics.

An AURORA-inspired lab workflow will learn per-seat trajectory descriptors and retain distinct capable profiles in a versioned quality-diversity archive. A completed-game descriptor is separate from the style chosen before play. Select tested style codes or distill qualified profiles into the shared network; keep the behavior encoder offline. This is proposed work, and learned latent separation is not proof of useful or human-like styles. Goal text uses public context by default; exposing private-hand-conditioned intent is a separate player-visible hints setting.

The intended roster is **16 player-facing archetypes, each with eight hidden qualified micro-variants (128 total)**. Quality-filtered clustering/FPS selects tested representatives and nearby coherent members. Sample one member per seat at game creation, keep it stable through the game and save/load, and qualify both individual members and their sampling mixtures at each difficulty. The four authored profiles remain the starting pilot; the full roster uses the same shared model and compact style records. The [detailed selection plan](PERSONALITIES_AND_DIFFICULTY_PLAN.md#5-aurora-inspired-discovery-of-additional-personalities) separates visible archetype diversity from subtle replayability within an archetype.

Each seat gets its own style, saved intent, random seed, and allowed observation. Calibrate every style into the same strength band for a given difficulty using mixed tables and rotated seats. Benchmark actual style diversity, repeated openings, failed plans, score margins, and qualitative human feedback. Shared model weights do not require identical decisions or cooperative objectives. Broadly human-like preferences can be designed; reliable imitation of human decisions would benefit from human move data and playtesting. Begin these additions with the completed self-trained AI and a small pilot, then apply the release strength gates to the selected profiles.

**Four-player training and acceptance**

Train on four-player games only, including early prototypes. Sample opponents from previous checkpoints, different styles, and the stock bot as a baseline. Rotate seats; vary legal game setup and random seeds; keep whole validation games separate from training. Attribute rewards and trajectories to the correct player. Train per-seat outcome estimates and use final standings plus score-related signals carefully enough that players still pursue good outcomes when victory becomes unlikely. Preserve observation boundaries for all seats and lab teachers. Distill stronger search into the fast policy, and train value estimates on the observable afterstates and rollout horizons used during laptop play. Search targets must reflect evaluated quality: raw visit counts from an elimination scheduler are not automatically suitable policy labels.

Use Daiyu as the scaffold for this workflow, with new weights and versioned four-seat targets throughout data generation, losses, inference and export. Do not inherit two-player example commands or discard valid trajectories merely for early networks, repeat loans, low cash or losing. Such filters can erase sound strategies and narrow the personalities. OpenSpiel comparisons must use the same four-player rules, information access, objective and total turn budget as S2.

Version the rules with the model and dataset. The [rules audit](../rules-audit/BUG_FIX_PLAN.md) identifies defects that can affect legality, results and simulation state; fix relevant issues consistently in simulation and live execution before large training runs. Preserve an explicitly labeled legacy baseline. Stock crashes and illegal legacy moves must be reported rather than treated as clean evidence of strategic superiority.

Maintain an immutable bank of improved legacy profiles and evaluate on both mixed and matching-style tables. Their tactic preferences are conditional, based on the BGG sources and tests in the legacy plan; they do not impose fixed openings or a universal five-VP-per-action threshold. Keep weak guarded stock occasional once stronger partners exist. Report the stronger bank separately from the original-Hard difficulty target, and reserve some tactical configurations for held-out testing.

New Easy exceeding stock Hard must be tested rather than assumed. A basic test is one new Easy seat against three stock Hard seats across many held-out seeds and rotated seats; 25% is the equal-strength first-place baseline. Require a meaningful improvement with uncertainty bounds, plus corroborating rank/score and mixed-table results. Repeat for every personality, and then check the Medium/Easy and Hard/Medium ordering. Exact equal skill across personalities cannot be guaranteed for every matchup.

Human playtesting decides whether the resulting games are enjoyable. Success includes varied credible strategies, coherent recovery from setbacks, fair information access, believable competition for resources, and manageable waiting time. Model size and stock-bot win rate alone cannot establish that outcome.
