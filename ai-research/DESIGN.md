# Four-player opponent design

Design recommendation, 27 September 2026. No new model, inference runtime, or mod was installed during this follow-up research.

Follow-up: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) details the rules interface, action continuations, state restoration, information boundary, runtime integration, and training milestones. It recommends new weights trained through search and a diverse opponent population; stock Hard remains a benchmark and occasional opponent. Its phase-based legal choices refine the complete-candidate representation below.

**User priorities**

Train and optimize for four players. A human normally faces three AI opponents. All inference should run locally on the Core Ultra 5 135U laptop; the RTX lab server is for training. Longer thinking times are acceptable when they improve play. The aim is enjoyable, varied opponents with coherent personalities. Easy, Medium, and Hard should all be available, with new Easy measurably stronger than the original Hard. That strength requirement is an acceptance target, not an achieved result.

**Recommended starting point**

One shared policy/value model, initially around two million trainable parameters, with a personality input for each AI seat. The policy scores complete legal candidate moves; value outputs estimate outcomes for all four seats. Add bounded search that respects hidden information and plans both actions of the acting player's turn. Train with a diverse population of opponents in four-player matches from the beginning.

Start with ordinary dense layers and residual blocks (an MLP) over structured features. Use separate state and candidate-action encoders, a shared move scorer, and per-seat outcome/score outputs. Widths in the 256–512 range are sensible experiment settings; the final parameter count depends on the feature schema. Encode the board once per position and score candidate moves in a batch. The representation must include connectivity, resource access, industry development, cash/income, relative seat order, era, cards visible to the acting player, and remaining actions. It does not use screenshots or text prompts.

A compact graph network or attention encoder is a later comparison if the simple model struggles to represent interactions between cities, links, industries, and players. Architecture and parameter count should be selected by held-out match performance at the same laptop time budget. A larger model is not automatically a stronger playing system.

| Experiment | Parameters | Float32 weights | Float16 weights |
| --- | ---: | ---: | ---: |
| Small baseline | 0.5 million | About 2 MB | About 1 MB |
| Preferred first serious model | 2 million | About 8 MB | About 4 MB |
| Larger comparison | 8 million | About 32 MB | About 16 MB |

These are parameter-storage calculations, not total RAM requirements or measurements. The runtime, activations, game state, and search tree require additional memory. Validate the effect of reduced precision on move ranking and match strength. The earlier 1–4 MB suggestion was a compact prototype budget, not a hardware ceiling.

**Intel graphics inference**

Intel lists the 135U's integrated Intel Graphics GPU with four Xe cores and support for OpenVINO, ONNX Runtime, and DirectML. It also lists an Intel AI Boost NPU. [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/237328/intel-core-ultra-5-processor-135u-12m-cache-up-to-4-40-ghz/specifications.html).

OpenVINO explicitly supports inference on Intel integrated GPUs. Its guidance includes float16 inference and batching requests. [GPU documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html).

The proposed deployment is CPU rules/search workers plus a local inference worker that can select CPU or Intel GPU. Benchmark both while Unity is running. Small individual requests can be dominated by dispatch overhead; batching candidates or several search leaves may improve GPU utilization. The GPU shares laptop power, memory resources, and rendering work, so hardware support does not establish a speedup. Retain CPU fallback. NPU optimization can wait until the model and latency profile justify another backend.

The earlier tests measured the stock rules and stock AI on CPU. They did not test neural inference or the GPU driver/runtime. A read-only CIM query for adapter/driver details was denied by the local environment; no driver or inference software was changed.

**Existing code and pretrained models checked**

| Source | Verified useful material | Limits for this project |
| --- | --- | --- |
| [Yuiffy/brass-birmingham-daiyu-ai](https://github.com/Yuiffy/brass-birmingham-daiyu-ai) | Rust game engine, PyTorch policy/value model, self-play and search code. Its `training/model.py` contains separate state/action encoders and a candidate move scorer. | Its README explicitly excludes model checkpoints from the repository and does not claim strong human-level play. Multiplayer value handling includes an approximation. No releases were listed. Its engine/schema differ from the Steam game. |
| [VyqreL/brass-birmingham-rl-model](https://github.com/VyqreL/brass-birmingham-rl-model) | MaskablePPO training code, with 512/512/256 policy and value layers, and a separate Python game implementation. | `brass_env.py` computes the opponent as `1 - current_player`, retaining a two-player reward assumption despite accepting four players. No weights appeared in the checked tree and no releases were listed. It is a reference, not a validated four-player training base. |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | General multiplayer and imperfect-information game framework; MCTS and information-set search implementations. | Requires a game adapter. Its standard Python AlphaZero training example uses two bots and is not a drop-in four-player solution. Its algorithm list marks information-set MCTS as lightly tested. |
| [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | Established implementation of PPO with legal-action masking; useful for a training baseline. | Requires a correct four-player wrapper or per-player trajectories. Simply alternating four players' rewards inside a normal single-agent trajectory is not the intended training setup. |

I did not find a verified, downloadable, strong four-player Brass checkpoint that can be plugged into this Steam installation. This is a bounded search result, not a claim that none exists anywhere. Reuse architecture/algorithm ideas and train weights against the verified engine. Even an available Brass checkpoint would need matching state and action encodings, compatible rules, and independent strength tests. The promising repositories were read, not installed or executed.

Inspected repository tree revisions: Daiyu `387840dd7a784f2c05fe561cb381e4f7d7bc590b`; VyqreL `7ec8b5fff9f60445a00b2384b81952b942118b43`. Sources for specific findings: [Daiyu model](https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/training/model.py), [Daiyu status](https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/README.md), [PPO environment](https://github.com/VyqreL/brass-birmingham-rl-model/blob/7ec8b5fff9f60445a00b2384b81952b942118b43/brass_env.py).

**Thinking time and difficulty**

Spend extra time examining alternative plans, possible opponent responses, and plausible hidden hands. Use the real rules to simulate actions. Each opponent acts for its own outcome; search must not treat the other three seats as a single opposing team. Use a hidden-information-aware search design instead of giving simulated decisions access to cards the decision-maker cannot see. Search cloning and static state isolation remain engineering work identified in the original review.

| Difficulty | Proposed maximum thinking budget per AI's whole turn | Intended behavior |
| --- | ---: | --- |
| Easy | 1–2 seconds | Coherent economy and plans, less extensive tactical checking, wider choice among credible moves |
| Medium | 3–5 seconds | Better two-action combinations, resource timing, and response to competition |
| Hard | 8–12 seconds | More alternatives and uncertainty samples; stronger tactical checking while retaining personality |

These are configurable design budgets, not measured speed or established strength levels. Return early for clear choices and provide a Play Now action. Three opponents at the Hard limit imply 24–36 seconds of thinking between human turns, plus animations. Difficulty should depend on calibrated policy/checkpoint, candidate selection, and search effort; thinking time alone is not a skill rating. A separate speed setting can cap waiting time.

**Personality and enjoyable play**

| Personality | Typical preference |
| --- | --- |
| Industrialist | Industry development and planned sequences of profitable sales |
| Networker | Expansion, contested connections, and link scoring |
| Supplier | Coal, iron, and beer timed around demand from the table |
| Investor | Cash flexibility, income stability, and longer-term investments |

Personality is a soft preference, not an instruction to repeat one strategy regardless of the board. Keep it stable during a game, let the agent pursue plans over several actions, and select among several reasonably valued moves. The acceptable estimated quality loss can be calibrated separately for each difficulty. Maintain a match-strength check because neural move values are imperfect estimates.

Use a shared model with a style embedding or small per-style adapters. Each seat gets its own style, random seed, and allowed observation. Calibrate every style into the same strength band for a given difficulty using mixed tables and rotated seats. Benchmark actual style diversity, repeated openings, failed plans, score margins, and qualitative human feedback. Shared model weights do not require identical decisions or cooperative objectives. Broadly human-like preferences can be designed; reliable imitation of human decisions would benefit from human move data and playtesting.

**Four-player training and acceptance**

Train on four-player games only, including early prototypes. Sample opponents from previous checkpoints, different styles, and the stock bot as a baseline. Rotate seats; vary legal game setup and random seeds; keep validation games separate from training. Attribute rewards and trajectories to the correct player. Train per-seat outcome estimates and use final standings plus score-related signals carefully enough that players still pursue good outcomes when victory becomes unlikely. Preserve observation boundaries for all seats.

New Easy exceeding stock Hard must be tested rather than assumed. A basic test is one new Easy seat against three stock Hard seats across many held-out seeds and rotated seats; 25% is the equal-strength first-place baseline. Require a meaningful improvement with uncertainty bounds, plus corroborating rank/score and mixed-table results. Repeat for every personality, and then check the Medium/Easy and Hard/Medium ordering. Exact equal skill across personalities cannot be guaranteed for every matchup.

Human playtesting decides whether the resulting games are enjoyable. Success includes varied credible strategies, coherent recovery from setbacks, fair information access, believable competition for resources, and manageable waiting time. Model size and stock-bot win rate alone cannot establish that outcome.
