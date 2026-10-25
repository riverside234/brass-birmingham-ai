# Rust Simulation and Search Backend Migration Plan

Updated: **2026-10-24**. Status: **implementation plan**; all implementation and qualification gates below start pending.

The migration has two portions, in this order:

| Portion | Deliverable | Required evidence before proceeding |
| --- | --- | --- |
| **Part 1 — Local game** | Rust game engine, our Rust neural search, exact native features and same-process CPU ONNX inference, integrated with Unity here | Local rules/model parity, shadow testing, complete live games, failure/rollback checks and measured two-session performance |
| **Part 2 — Lab server training** | Reuse the locally qualified core for ordinary self-play, branches and explicitly enabled search consumers; retain the existing Python/PyTorch trainer | The Part 1 release receipt first; then Linux/GPU, replay, PPO/PPG and resume qualification before a bounded lab canary |

**Execution order: local backend → local debugging and live testing → recorded local acceptance → lab backend → isolated lab validation → controlled training rollout.** A Python test adapter used locally is diagnostic infrastructure; it does not require migrating or running lab training first. Local defects get a reproducible regression fixture and a local fix before promotion. Linux/GPU-only defects receive server tests as well, and changes to the shared core rerun the affected local gates.

The requested local timing target is **four settings: 4s / 8s / 10s / 12s**, all with **two independent search sessions (one thread each)**. Keep the matched complex-state comparison at **one 12-second whole-turn allowance**. Four-setting game/catalog work is being updated separately in this workspace; the migration must integrate and qualify its admitted version. This document records planned backend work, not a completed Rust release or lab deployment.

This revision incorporates the inspected self-play implementation through repository HEAD `5b49722d7bc328bbad7866c08671967fa505d4ad`, including the 24 October g850 recipe. It also preserves the earlier native-design, same-process inference, profiling, batching and performance requirements. Pin actual source/configuration/checkpoint hashes again at implementation time; current uncommitted work and a lab run's adopted recipe can differ from HEAD or a template config.

## Part 1 — Local game: Rust search and simulation first

### A1. Outcome and starting points

Run **our neural Masked ISMCTS** and every simulated transition in Rust, starting from the rules in [brass-birmingham-daiyu-ai](https://github.com/riverside234/brass-birmingham-daiyu-ai). Unity keeps the live game, display, saves, input, final validation and application of complete actions.

The production CPU worker contains the rules, tree, belief sampling, exact Features-4/derived inputs, bounded batch scheduler and ONNX Runtime. It performs no per-move or per-leaf Python/C# RPC. Reuse admitted neural architectures/weights after exact input/output checks. Python/C# remain local parity oracles; the existing qualified C# V4 AI remains the fallback.

A faithful native port is the baseline. Upstream's determinized PUCT has different leaf/backup semantics; using it would change the experiment. Our search samples permitted worlds, preserves staged information sets, uses frozen opponent policies and evaluates at a new future normal root-player information frontier. Later visits can expand farther. The separate earlier-leaf experiment must keep its own search identity and quality evidence.

| Inspected component | Decision for this migration |
| --- | --- |
| C# host and Python `BrassSpielGame` / `BrassSpielState` | Keep as parity oracle; implement compatible import, staged choices, observations, chance and returns in Rust |
| Python `MaskedISMCTS`, `SearchSession`, newer deadline admission | Port faithfully, including private trees, four-seat values, complete plans and caller timing |
| Upstream atomic-action enumeration and native saved states | Add direct staged choice generation and external import; avoid constructing every complete action at each stage |
| v11.2 Features-4 model and critic descriptors | Preserve the selected model's exact interface; unknown descriptors for deployment/search by default |
| `LocalTurnBridge` and `IAiExecutionBackend` | Integrate with the existing scheduler, freshness guards, isolated validation and single main-thread commit |
| Installed patch 1.4.32, V4 6/10/15-second settings | Historical compatibility baseline; implement the requested four 4/8/10/12-second settings with versioned save handling |
| Ordinary collection, branch scheduler, replay, PPO/PPG and ratings | Part 2 deliverables; these are not prerequisites for local integration |

#### Source and historical evidence

- Rust clone: `C:/Users/BobXu/Desktop/brass`. Last inspected upstream revision: `f8f0246718d6d3fb3f301742d4154811c90d5813`. Recheck its current revision and local patches rather than treating the earlier inspection as a new source audit.
- Our inspected source HEAD is recorded above. Existing uncommitted personality documentation and four-setting game/installer changes are separate work. Pin selected bytes, runtime receipts and configuration; preserve historical upstream locks and notices.
- The earlier experiment under `C:/Users/BobXu/Desktop/brass/output/cpu-capacity-2workers-20261016-deadline/` used upstream PUCT, initialized networks and a folded feature adapter. It established neither exact model compatibility nor our engine/search speedup. Its clock was per root, not per whole turn.
- The [historical four-worker batching report](../artifacts/feature-engineering/rust-v9-v11-w448x6-4workers-20261016/REPORT.md) favored cap 2 over cap 4 in 13/13 preliminary uncapped pairs (median completion ratio 1.0602). It used four worker processes, a shared CPU service and folded inputs. Retest with two independent one-thread sessions, exact features and same-process inference.
- `artifacts/patch-installation.json` records installed patch 1.4.32; `docs/V4_SEARCH_TIME_1.4.32.md` describes its timing/save behavior. The older `src/BrassLocalAI/README.md` is useful for lifecycle history but is not the current timing authority.
- Latest self-play behavior comes from executable configuration/collector/trainer code and the admitted run signature. `configs/step5/joint7-latest.json` contains older explicit values; `latest_model_config()` applies subsequent migrations including g850. Do not infer the current recipe from the filename “latest.”
- Frozen research and installed Unity observations have separate contracts. Resolve any rules difference explicitly; do not label a deployment observation as admitted training data automatically.

#### Performance work retained

Use compact state, direct staged choices, arena-owned trees, reusable tensors, measured copies/allocations and native encoding. Co-locate inference with CPU search, then retune batching. The earlier estimates of roughly 50 steps per trajectory, 25 ms per host step and 3–8 ms per Python encoding are hypotheses to remeasure, not constants or additive speedup promises. Value-network bootstrapping already exists in the baseline; changing where it occurs is a separately measured algorithm experiment.


### A2. Native architecture and engineering standards

```mermaid
flowchart LR
    U[Unity live game and main-thread validation] -->|Permitted root, recall, identity, remaining budget| W[Rust AI worker]
    subgraph R[One native Windows AI process]
        W --> S[Two independent search sessions: one thread each]
        S <--> G[Rust rules, sampled worlds and explicit chance]
        S --> E[Exact Features-4 and admitted model inputs]
        E --> Q[Per-session cooperative batching]
        Q --> N[Per-session single-threaded ONNX]
        N -->|Policy and four-seat values| S
    end
    W -->|Complete plan and completed-work evidence| U
    H[Local C# and Python parity harness] -.->|Reference fixtures and comparisons| W
```

#### A2.1 Process and ownership boundaries

Use an isolated Rust core crate and a long-lived Windows AI process containing **two independent search sessions, each executed by one dedicated thread**. Each session owns its private tree, sampled states, pending trajectories, scratch buffers and single-threaded inference path. Both receive the same detached permitted root and remaining whole-turn budget. Merge completed root/plan evidence through the coordinator; do not share mutable trees.

The local CPU execution contract is **2 sessions × 1 thread**. Search, encoding, batch preparation, ONNX execution and backup run on that session's thread. Set ORT intra-op threads to 1, inter-op threads to 1 and sequential execution; create no extra inference worker pool or shared cross-session inference dispatcher. Persistent model handles, including frozen opponents when needed, run serially on their owning session thread. Record cached model/session counts and memory separately from the two computational threads; model caching must not add parallel execution. Transport/UI coordination is outside this compute count and performs no simulations or NN forwards.

A session may advance several bounded pending trajectories cooperatively and form a local batch before a synchronous inference call. This is concurrency in the session's state machine, not more threads. The other session can continue independently. Batch caps/flush/pending limits may be tuned within this fixed execution layout; increasing thread count or pooling batches across sessions is outside the requested configuration.

Here, **in-process means engine, search, encoder and CPU inference share the Rust AI process**; Unity remains a separate process. In Part 1 the core owns rules, sampling, search and feature encoding. Part 2 adds ordinary game lanes to the same qualified core. Keep transport, Python bindings, runtime calls and Unity integration behind typed adapters.

Use a framed protocol only at the Unity/diagnostic orchestration boundary and for the Part 2 GPU evaluator. JSON is acceptable for infrequent controls/debugging. Production CPU trajectories perform zero Python/C# rules callbacks, board-to-JSON round trips or per-leaf process RPC. Share immutable board definitions and metadata; measure state/tensor copies and duplicated model memory.

An optional local diagnostic Python binding releases the GIL during native CPU work. Keep mutable instances with their owner; do not force an upstream unsendable wrapper across threads. Verify both sessions run concurrently while each stays single-threaded.

#### A2.2 Native data layout and hot paths

| Area | Required design | Evidence |
| --- | --- | --- |
| Static and shared data | Immutable indexed topology, card/tile definitions, lookup tables and permitted history prefixes | State copies share these immutable inputs; mutable history deltas remain owned by each world |
| Mutable game state | Compact typed IDs, enums, arrays and bitsets where the domain is bounded; explicit phase/pending-action data | Record state size and bytes copied per sampled world/transition; no nested JSON or string-keyed state dictionaries |
| Legal choices | Generate the current stage directly into reusable worker-owned storage; retain every legal semantic choice in reference order | Dense Rail and multi-sale fixtures do not first enumerate all complete plans; report generated choices and allocations |
| Rules application | Apply an already validated typed choice to owner-local state; keep external validation at ingress and live commit | Repeated fixed-capacity apply operations target zero heap allocations; no repeated parsing or global lock |
| Search tree | Bounded arenas of nodes/edges with stable integer handles; keep frequently read visit/value/prior fields compact | Measure bytes per node/edge, arena growth, traversal cost and cache misses where available |
| In-flight trajectories | Explicit resumable states holding path handles, sampled state and an evaluation ticket | No live tree borrows across inference waits; arena generations prevent expired tickets from accessing reused handles |
| Information/evaluation keys | Typed canonical representations of the existing permitted information and context | No per-visit JSON/digest rebuild; equality resolves hash collisions; hidden state cannot enter information-set keys |
| Feature encoding | Write directly into reusable contiguous tensors with candidate offsets; share only immutable, semantically identical prepared inputs | Report encoding time, allocations and bytes transferred; preserve exact masks, order and declared precision |

Map external semantic strings to compact IDs once, and reconstruct strings at protocol/audit boundaries. Internal ID ordering must preserve the reference's ordering and tie-breaking. Keep observation, recall, context, legal-choice and reward identities complete when optimizing keys. A board hash alone cannot identify an information set or evaluation. Cached legality and features need explicit invalidation rules for each state mutation, seat and model contract.

Start with compact state copies and reusable sampled-world storage. Benchmark copy/restore against a mutation journal only after rule parity passes. Adopt apply/unapply if it improves the full trajectory workload enough to justify its complexity. Its journal must restore phase data, resources, derived caches and relevant RNG state exactly, including exceptional/canceled paths. Compare restored states and incremental caches against independently recomputed results. Preserve a simple reference path for diagnosis.

Keep legal-choice, path, tensor and batch buffers across calls. Grow capacity deliberately and record high-water marks; large legal sets must never be truncated to fit a buffer. Separate temporary scratch from retained nodes, keys and replay records. Arena growth and retained records may allocate; repeated scratch use should not. Rust's `Vec` supports capacity reuse, but the allocation budget must be demonstrated by measurement. [Rust Vec documentation](https://doc.rust-lang.org/std/vec/struct.Vec.html)

Build sampled worlds from a prepared permitted root without replaying the entire game history or serializing checkpoints per trajectory. Preserve fresh sampling and the reference's actual conditioning. Share immutable public facts only where the contract allows it; cache reuse must not collapse distinct hidden worlds, recall, seats or utility budgets.

Build views on demand for the current policy consumer and any required root reward/recall consumer, reusing a common public projection. Preserve all required observation updates without materializing four complete JSON views after every move. Validate contracts and source/model identities at their real boundaries; maintain cheap native freshness, legality and information-set checks within the loop.

#### A2.3 Rust engineering and build standards

Prefer safe Rust, explicit ownership and small modules organized around rules, state, observation, belief, features, search and transport. Use typed errors for invalid requests and unsupported states. Avoid unchecked indexing, `unwrap()` on external data, global mutable state and broad `Arc<Mutex<...>>` ownership. Any measured need for `unsafe` requires a narrow implementation, documented invariants and focused verification; it is not a default optimization technique.

Contain failures at the process/binding boundary. A worker panic retires that request through the existing failure lifecycle; an in-process FFI adapter must translate failures without unwinding into foreign code. Document ownership and release rules for every borrowed tensor or external buffer.

Preserve the reference's numeric semantics, including search accumulation precision, normalization and tie-breaking. A faster implementation must not silently narrow values, reorder probability operations or use approximate keys. Keep production checks for protocol validity, finite tensors, legal actions and request identity; expensive whole-state consistency checks belong in debug/differential builds.

Require formatting, lint, native unit/property tests and the relevant cross-language parity suite for each change. Run both debug and optimized tests: unchecked release overflow or build-specific behavior can otherwise escape qualification. Use `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` for the supported feature matrix; lint exceptions must have a local reason. [Clippy usage](https://doc.rust-lang.org/clippy/usage.html)

Maintain a profiling build with release optimization and symbols, plus a portable deployment build with recorded CPU requirements. Compare optimization levels, ThinLTO and codegen-unit settings on representative workloads; compiler flags are experiments, not assumed speedups. Keep machine-specific CPU flags out of the portable artifact. Treat profile-guided optimization as a later experiment using training fixtures distinct from the held-out benchmark. Pin and record the effective profile and toolchain. [Cargo build profiles](https://doc.rust-lang.org/cargo/reference/profiles.html)

Every performance change includes its motivation, affected invariants, before/after raw measurements and relevant regression results. Prefer the simpler design when the speed difference is within noise. Keep verbose per-node tracing disabled in performance builds and measure instrumentation overhead using unprofiled confirmation runs.

#### A2.4 Same-process ONNX Runtime

Implement a narrow Rust inference adapter over a pinned binding to ONNX Runtime's C API. Pin the binding, runtime/library hash, execution provider, graph settings and export/opset contract. Create persistent sessions by admitted model/export identity and warm them before timed search. The C API supports native sessions and tensors backed by caller-provided memory; enforce buffer lifetimes through completion and measure output allocation/copy costs. [ONNX Runtime C API](https://onnxruntime.ai/docs/get-started/with-c.html)

Keep the encoder and all policy postprocessing native on this path: candidate masks/order, softmax or log-probabilities, temperature/exploration, seat remapping, personality adjustments and utility outputs must follow the selected contract. Compare Python/PyTorch, Python/ORT and Rust/ORT on the same exact tensors, then compare full policy responses. Require singleton/mixed-batch parity for variable legal counts and every supported output head.

Use the two session-owned inference paths specified in A2.1. Pin intra-op/inter-op to 1 and sequential execution; verify runtime settings and observed concurrency rather than relying on defaults. Preserve the two-session/one-thread configuration during batch retuning in A5.1. [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)

The existing external evaluator remains the parity oracle; its GPU-training adapter is qualified in Part 2. CPU deployment must pass with that evaluator unavailable. Freeze model versions for each local session and, in Part 2, each collection round, invalidate caches when versions change and keep the existing Python/PyTorch optimizer and checkpoint lifecycle. Do not require a new ONNX export after every PPO minibatch; any CPU-collection export occurs at an explicit frozen snapshot boundary and its cost is reported.

#### A2.5 Proposed implementation locations

These are proposed locations, not files supplied by this plan. Create local core/worker/inference/Unity and diagnostic-harness code in Part 1; add collection and lab orchestration adapters only in Part 2:

| Proposed path | Responsibility |
| --- | --- |
| `native/brass-ai-core/` | Adapted engine, state/action contracts, belief sampling, Masked ISMCTS, ordinary collection loops and exact feature encoder |
| `native/brass-ai-worker/` | Worker/game-lane lifecycle, cancellation, framed protocol, inference transport and bounded decision buffers |
| `native/brass-ai-inference/` | Same-process ONNX adapter, model sessions, native postprocessing and CPU batch scheduling |
| `src/brass_ai/native/` | Research and ordinary-collection adapters and backend selection, preserving the current public training interfaces |
| `src/BrassLocalAI/Native/` | Unity request/result adapter using the existing execution lifecycle |
| `tests/RustBackend/` | Golden fixtures, differential tests and cross-language search traces |
| `native/brass-ai-core/benches/` | Fixed-work rules, restoration, encoding and scripted-search microbenchmarks |
| `configs/references/rust-engine.lock.json` | Upstream revision, imported source hashes, local patches and attribution references |
| `artifacts/rust-backend/<run-id>/` | Versioned parity, benchmark and deployment evidence |

Keep imported code and our adapters distinguishable. Carry forward applicable notices and record modifications. Preserve historical upstream locks and frozen source inventories; add a new backend identity rather than rewriting old evidence to appear compatible.

### A3. Rules, request and neural contracts

#### A3.1 Rules and state import

Define a versioned semantic state DTO and a canonical comparison projection independent of C# object identities and Rust memory layout. Import all information needed to reproduce a position:

- Game configuration, stable seat/color mapping, era, round, turn owner, acting player, actions remaining, first-turn rules and phase.
- Cash, income-track position and derived income, spending, points, industry mats, tile identities/levels, board slots, links, resource quantities, markets, merchants and consumed bonuses.
- Public discard/history, known wildcards, permitted recall and outstanding action/payment/continuation/bonus/debt state.
- For trusted differential fixtures only: complete hidden hands, deck order, setup discards, pending chance and any state needed for deterministic replay. Keep this format separate from production policy requests.

Rebuild Rust connectivity masks, resource indexes and other derived caches after import. Validate card/resource conservation and state invariants. Fail explicitly on an unsupported phase or configuration and use the qualified backend; never manufacture a turn-start approximation for a mid-action state.

**Production search starts from a detached permitted view and allowed recall.** Rust builds sampled hidden worlds from this information. It must not receive the actual opponent hands, deck order or live RNG through a convenient full-state export. Trusted full-state imports are for parity/debugging, not belief construction.

Preserve the current sampler's exact conditioning: `sample_world_setup(view, composition, search_seed)` uses the permitted view; `BrassSpielGame.from_observation` attaches root recall afterward for information-set identity. Carry recall across the Rust interface without silently adding history-conditioned beliefs. Such a sampler change would need separate algorithm qualification.

Initially qualify four-player standard games, matching our neural research contract. Before exposing this backend in Unity, either qualify every selectable configuration it supports or explicitly route unsupported two-/three-player and introductory configurations to the existing backend.

#### A3.2 Staged choices and complete live actions

Preserve our semantic choice stages: ordinary action, payment, continuation, bonus, debt and explicit chance. Upstream's enumeration of every complete atomic action is not a drop-in replacement for this interface.

Expose legal staged choices directly from the Rust phase machinery. Maintain stable semantic IDs and ordering for locations, slots, links, card types, industries, resources and merchants. Detect ID collisions and version any internal compact-ID mapping. Preserve the full legal choice set and policy denominator even when search progressively admits only some edges.

Accumulate the selected current-turn trace into complete actions compatible with Unity's validator. Preserve ordered resource payments, multi-sale order, merchant beer/bonus use, development choices, optional second Rail link, Scout card combinations and liquidation ownership. A first action and a supported second action are separate validated commits on the existing presentation schedule.

#### A3.3 Request and result identity

Every request carries protocol, rules, observation, feature, search and reward identities; model/export and profile hashes; game instance/generation, revision, actor, turn key, request ID; allowed recall; worker count; and remaining budget. Results echo these identities and identify the backend build.

The coordinator owns the turn deadline. Convert it to the worker's clock through an explicitly verified clock mapping, or send a remaining-duration budget with transport/queue allowance and coordinator cancellation. Receiving or retrying a request must never grant a fresh allowance. The coordinator independently rejects expired results. Inference replies carry generation, batch and model IDs as well.

Return complete action plans, continuation preconditions, completed root/plan evidence, interruption reasons and timing counters. Return errors as explicit states that leave the live game untouched.

Specify tensor dtype, shape, layout, candidate offsets/masks, seat mapping and buffer lifetime in the protocol. Reject malformed, non-finite or mismatched responses before cache insertion or backup. Bound queued batches and retained game/tree memory; memory pressure must reduce pending work or trigger an explicit failure, not truncate the legal choice set.

#### A3.4 Neural interface and checkpoint reuse

Freeze the active architecture, feature schema, action schema, context/reward contract, relative-seat mapping, checkpoint hash, model export and preprocessing. The inspected v11.2 model still uses Features-4 state/action widths 3,128/327, but also needs derived/local inputs and critic descriptors. Read dimensions from the admitted export; the earlier width/depth benchmark is not an input manifest.

Port the existing feature meanings, ordering, normalization, known/applicable flags, candidate masks, entity ordering where applicable, personality context and remaining utility budget. The bridge must accommodate the selected model's actual inputs/outputs and frozen opponents' admitted schemas. A model factory name or equal dimensions is insufficient.

The inspected base schema is **Features-4** (`permitted-features/4.0.0`, `features-4.0.0.json`). Port that exact encoder and all additional grouped/local/context inputs required by each admitted checkpoint. V9/v11 model labels do not by themselves identify a feature schema, input width or parameter count. Production CPU evaluation must use the Rust encoder directly, without falling back to Python at new positions.

During development, compare Rust tensors with the Python encoder on identical permitted fixtures. Require exact equality for discrete IDs/masks/order and declared numerical tolerances for floating-point features and inference outputs. Compare policy ranking/ties as well as values. **No folding, hashing or invented projection between upstream features and our model is acceptable.**

Keep weights and architecture unchanged when these checks pass. If rules or feature semantics intentionally change, create a new contract and explicit checkpoint/data admission decision; re-encode or retrain as needed. Do not silently reinterpret old replay or teacher labels.

The inspected current architecture is `brass-four-seat-mlp-resnet/11.2.0`: 5,910,916 base parameters and 5,928,616 with the utility head. It retains the v11.1 actor and adds `seat-descriptors/1.0.0`, a `[B,4,34]` critic-only input with a 34 → 32 → 32 encoder. Verify relative-seat mapping, the all-unknown default, explicit known/unknown inputs, dynamic legal counts and policy independence from descriptor values. Search and deployment use **unknown descriptors by default**, matching the current contract. Do not import privileged training table roles into live search or choose an own-copy descriptor experiment silently.

Select an existing, admitted trained checkpoint whose utility/reward readiness supports local search. Its model, reward and export hashes govern the local profile; g850's new reward settings do not automatically make an older utility head calibrated. If only uncalibrated IL weights are available, use them for tensor/mechanical diagnostics and keep the live neural-search gate pending until a suitable checkpoint is admitted. **Part 1 does not require a new self-play run or a repeated IL run.**

Keep the completed v11.1 IL source and its explicit v11.2 upgrade receipt intact. The supported zero-prior-self-play handoff, fresh optimizer and utility calibration are tested in Part 2. Resuming an established v11.2 self-play checkpoint preserves its architecture, weights, optimizers and progress; changing the simulator language is not another architecture migration.

### A4. Port our search, including its less visible behavior

Use `src/brass_ai/search/ismcts.py`, `search/common.py`, `searchlab/spiel.py`, `personality/search/engine.py`, `recovery/parallel.py` and `recovery/teacher_schedule.py` as behavior references. Freeze both the common search configuration and the selected caller. Part 1 targets the local whole-turn caller; training teacher scheduling is added only in Part 2. Personality sessions and training teachers share search machinery but have distinct scheduling, work-budget and continuation rules.

1. **Belief and chance.** Port permitted-view sampling, known-card conservation, own-hand recall, setup discards and Canal/Rail discard transitions. Sample a fresh valid world per trajectory. Opponents act from their own masked observations. Represent card draws and deals as explicit lawful chance events.
2. **Information-set identity.** Preserve keys derived from allowed view, legal choices, recall and context, including remaining reward budget when enabled. Keep evaluation-cache identities sensitive to model, policy, feature/search contract, context and relevant reward state.
3. **Selection and expansion.** Preserve PUCT/FPU, personality priors, protected action families, progressive expansion, reopening and tail admission. Top-k expansion parameters must not become permanent legal-action pruning. Preserve replica warmup/diversification behavior.
4. **Opponent policy and leaf evaluation.** Sample frozen opponent assignments according to the existing contract. Use their predictions for actions, not as root leaf values. Bootstrap with the root model at the first new future normal root-player information set; allow later completed visits to expand it. Terminal settlement includes required debt/liquidation before utility comparison.
5. **Four-seat backup.** Preserve all four values in stable seat order, win-share or the selected utility contract, and per-seat reward/cost ledgers. Do not adopt upstream's scalar-to-opponent approximation or a two-player negamax conversion.
6. **Pending work.** Preserve selection-only pending reservations/virtual loss. Pending paths do not count as completed visits. Cancellation, exceptions and late NN responses release reservations exactly once. Only fully completed, timely trajectories contribute visits, returns, frontier support, complete-plan evidence or teacher labels.
7. **Turn session.** Preserve turn keys, profile/context lifetime, recall updates, tree/cache reuse and minimum completed-plan support. Reuse a continuation only when its permitted state and legality preconditions still match. Do not extend search by resetting the clock on action two, payment, continuation or replanning.

For deterministic parity, use recorded sampled worlds, chance draws, policy responses and tie-breaking decisions. Equal Python/Rust numeric seeds do not imply equal PRNG streams. Begin with one deterministic lane; then test the two-session asynchronous scheduler by its completed-work and privacy invariants. Timing-dependent interleavings need not produce bit-identical visit distributions.

Implement a trajectory as a native state machine that advances until completion, cancellation or an inference request, then resumes from an owned ticket. The scheduler can advance other admitted trajectories while that request waits. Keep node handles stable and prepare a bounded backup transaction before publication. A full-tree copy, global lock or per-edge cross-language call is not an acceptable way to make backups transactional. Preserve completed-only visibility and test deadline crossings during commit.

#### A4.1 Value leaves and the earlier-leaf experiment

The matched migration already replaces continuation beyond its defined frontier with a value-network prediction. The current `new_root_information_frontier` rule can still require many staged choices before reaching that frontier. Measure choices/actions per completed trajectory, time before the first leaf, NN evaluations and terminal-leaf frequency to establish where shorter trajectories could help.

After baseline parity, implement a separately identified **first-new-compatible-information-set** leaf experiment. Bootstrap at the first newly expanded nonterminal information set whose observation/phase and reward context are supported by the value model, instead of continuing to the baseline's later frontier. Start with root-player observations covered by the existing value training. Evaluating at arbitrary opponent turns needs explicit evaluator, actor-view, seat-remapping and reward-context qualification; do not substitute a frozen opponent's value output for the root model's value.

Keep exact terminal returns, four-seat backup, hidden-information restrictions and the same turn budget. An earlier leaf can end before a complete live action; partial traces must not acquire complete-plan support or become deployable actions/teacher plans. Preserve a legal completion path and measure its cost. Compare this variant with the faithful native baseline using identical trained checkpoints, two single-threaded sessions, 12-second whole turns, complex roots and seat-balanced strength tests. Retune batching for it only as a separately reported optimization.

Adopt the new leaf contract only if its measured decision quality and latency pass the predeclared criteria. Version its search/teacher provenance separately. Higher visit counts caused by shorter trajectories cannot be reported as a Rust engine speedup or as evidence of stronger play.

Preserve the newer launch-admission rule in `search/deadline.py`: timed entry points estimate a path's cost from completed local trajectories (the last eight samples, no estimate before four, then the lower-quartile order statistic) and stop launching when the remaining budget is too short. This does not change the leaf or permit early partial returns. Count-primary teachers consume their configured quota with an exceptional watchdog; Part 2 must preserve this distinction. Record attempted/completed/interrupted paths, admission stops, remaining time and the estimate.

### A5. Whole-turn timing, inference and cancellation

- Give a searched ordinary turn one 12-second allowance and both sessions the same remaining budget. Preserve the selected reference's preparation/admission boundary, measure excluded preparation separately and report total observed turn latency too.
- Pin the finalization reserve to the reference for the matched comparison. Stop admitting work at its work deadline; plan selection, completion and delivery use the remaining allowance. Tune that reserve only in a separately identified optimization run. Action two, payment, continuation, bonus and replanning do not replenish the ordinary-turn budget.
- Poll cancellation/deadlines inside expensive legal-choice generation, recursive resource/sale exploration, traversal, feature building and transport waits. Upstream's optional benchmark deadline patch is evidence of a problem, not the final production cancellation API.
- Measure polling overhead and the longest interval of unchecked native work. Use bounded chunks with cheap cancellation checks and appropriately spaced clock reads; an expensive recursive operation cannot hide behind one outer-loop check. Preserve the same cutoff contract while tuning check frequency.
- Keep completed-work accounting transactional. A timeout during a simulation or NN wait must not publish a partial backup. Record last completed-work time, work-stop time, response-ready time and live-apply time separately.
- Use session-owned pending queues and persistent single-threaded ONNX handles. Count encoding, queueing and inference inside search work. The two independent sessions each run search and inference on their one thread; there is no extra inference pool. Record actual computation/transport threads and CPU use separately.
- Maintain separate latency policies for interactive search and ordinary collection. Search batches flush on a small bounded delay or the approaching deadline; collection batches can favor throughput across independent games. Group only compatible model/export/schema requests, preserve candidate offsets and never wait indefinitely to fill a batch. Measure batch size, queue age, padding and worker idle time before tuning pending-path limits or NN thread counts.
- Put a deadline on each batch and ignore results for expired requests/generations. Bound pending work and buffer ownership. Cancellation does not mean an in-flight native/device operation has already stopped; the execution backend remains busy until all its CPU/device work has actually completed.
- On timeout, worker failure or an unsupported state, use the bounded legal fallback only while the same live request is still valid. Cancellation caused by game replacement, history navigation, menu exit or a superseding request retires the job without applying any action. Never start a fresh 12-second fallback search after the allowance has elapsed.

A hard search cutoff and zero wall-clock delivery overhead are different claims. Qualification must state the permitted delivery tolerance and observed worst case; do not describe a 12-second search followed by unbounded draining as a 12-second turn.

**Reference timing details to preserve:** personality `SearchSession` keys sessions by `(turn_key, seat, is_debt)` and disables search at debt roots. The training teacher wrapper also routes debt roots through its policy/completion path. Forced single-choice decisions remain recorded without launching search. Direct-policy calls have their own inference timeout in the research code; they do not establish a 12-second Unity latency guarantee. Give deployment policy/fallback work an explicit bounded lifecycle and qualify it separately.

The target local menu has exactly four search allowances: **4s / 8s / 10s / 12s**, each using two independent search sessions (one thread each) and one shared ordinary-turn budget. Keep the four-setting catalog's existing default (Hard, 10s) and use 12s for the matched benchmark; verify deadlines and lifecycle at all four settings. The installed 1.4.32 6/10/15-second catalog is an input compatibility baseline, not the target. Section A9 defines its explicit save migration. Production teacher recipes may instead use phase-specific time allowances or simulation quotas with a watchdog. Preserve the selected recipe's budget mode, counters and admission behavior; do not convert it to time-based search as part of the port.

#### A5.1 Retest batching within two single-threaded sessions

The historical shared-service cap-2 result is only a hypothesis. The required local layout is **two independent sessions, one thread each**, with separate inference execution and no shared batch dispatcher. Both sessions keep ORT intra-op/inter-op at 1 with sequential execution throughout qualification.

1. Start with one pending trajectory per session and immediate singleton inference. Establish latency, completed support and CPU/memory use.
2. Compare two, then four cooperative pending trajectories per session if useful. Test local caps 1/2/4 only where compatible pending requests can fill them; batching never combines the two sessions or creates an inference thread. Multiple frozen models may reduce occupancy.
3. Compare immediate flush with short bounded policies such as 0.5/1/2 ms where the session can advance other trajectories. Do not block waiting for requests only that same thread could produce. Near the deadline, flush ready work or retire it under the pinned completion rules.
4. Use NN-heavy Canal, dense Rail, variable legal counts and mixed frozen models. Measure actual batch size, padding, time to leaf, idle/queue time, allocation, completed plans and deadline misses. Test singleton/mixed-batch output parity.
5. Select using paired whole-turn results and held-out cases; keep the session/thread configuration fixed. Record cap, pending capacity, flush policy, per-session ORT settings and binary/model hashes.

The single deterministic session used for trace parity is a correctness fixture, not a request for another performance configuration. Part 2 retains the lab recipe's separate ordinary-lane/GPU batching settings; the local two-session limit is not a new cap on all training actors.

### A6. Local implementation sequence and exit gates

All new paths and test/benchmark drivers below are proposed deliverables. Keep step status, source/model/runtime hashes, commands actually run, result paths and unresolved differences in `artifacts/rust-backend/<run-id>/local/progress.json`. Start as pending; a gate passes only when its evidence exists. These are technical gates, not repeated permission requests.

| Steps | Dependency | Result |
| --- | --- | --- |
| L01–L07 | Prior local step | Pinned reference, Windows Rust core and qualified simulation |
| L08–L09 | L07, then L08 | Exact current model inputs and same-process CPU inference |
| L10–L12 | L09, then prior step | Faithful native search, whole-turn sessions and two-session completion |
| L13 | L12 | Unity shadow integration and four-setting/save compatibility checks |
| L14 | L13 | Matched local performance, batching and quality evidence |
| L15 | L14 | Opt-in local live games, reproducible bug fixes and rollback; **Local acceptance gate** |

Recommended order: **L01 → … → L15 → local acceptance receipt → Part 2**. No self-play collection, PPO update or positive teacher-admission test is a predecessor of L13/L15. Use an already admitted, value-ready checkpoint for local play; an IL-only mechanical test cannot certify search quality.

Performance work starts with the harness and continues at rules, features, inference and search milestones. A Python simulator adapter is useful for local differential testing and benchmark variant B; final trajectories remain native. Preserve a simple implementation for optimization ablations and keep unrelated workspace changes and active training runs intact.


#### L01 — Freeze the local rules, model and installed-game baseline

**Targets:** `configs/references/rust-engine.lock.json` and `artifacts/rust-backend/<run-id>/baseline.json`; existing model/checkpoint manifests, installed 1.4.32 receipt and relevant completed IL lineage.

1. Identify the exact current C#/Python sources, configuration bytes, upstream Rust revision and local patches. Select the four-player rules baseline and record unsupported configurations.
2. Pin an admitted existing trained checkpoint/export and its lineage, architecture, descriptor/feature/action/context/reward contracts, value readiness, inference runtime and frozen opponent policies. Preserve IL upgrade receipts where applicable. Freeze the four target allowances (4/8/10/12 seconds), two single-threaded sessions and the legacy-save mapping in A9.
3. Record the reference environment, search counters, timing boundaries, CPU/device/thread settings and local deployment policy. Measure baseline A and freeze the section A8 performance targets and regression limits before inspecting B/C results. Include a held-out complex-state set for final evaluation.

**Exit:** the manifest resolves every source/model/configuration identity and the selected checkpoint passes its existing hash/architecture/source/readiness checks. The installed fallback and original saves have a recoverable release baseline. No weights or training artifacts have been rewritten.

#### L02 — Build the independent fixture and comparison harness

**Targets:** proposed `tests/RustBackend/fixtures/`, `test_contracts.py`, `test_state_parity.py` and `scripts/qualify-rust-backend.py`; existing authored C# host and runtime contracts.

1. Define versioned permitted-request and trusted-debug state schemas, semantic choices and a canonical state comparison projection. Keep hidden-state fixture imports separate from production policy requests.
2. Capture reference roots, staged transitions, all four masked views, explicit chance outcomes, terminal scores and the selected utility/reward returns. Include independent expected outcomes for known rule edge cases.
3. Implement fixture loading and a comparison report with the first differing field/choice and its source identities. Add recorded random draws and scripted NN replies for reproducible cross-language traces.

**Exit:** the harness validates its reference fixtures, detects deliberately changed state/legal/view/return data and never exposes trusted hidden fields through the permitted-request serializer. Fixture quantities and coverage follow section A7.

#### L03 — Isolate and build the Rust core and worker skeleton

**Targets:** proposed `native/Cargo.toml`, `native/brass-ai-core/`, `native/brass-ai-worker/` and `src/brass_ai/native/`.

1. Create the Rust workspace, pin dependencies/toolchain/build identity and isolate imported rule code from upstream web, training and application concerns. Preserve attribution and the local patch inventory.
2. Introduce typed state/choice/error interfaces and a worker handshake reporting supported contracts and build identity. Implement the ownership/layout requirements in section A2, starting with compact state copies and a simple framed local protocol.
3. Add explicit backend selection defaulting to the existing implementation. Establish separate instances for mutable game state; do not switch an active run or Unity deployment during scaffolding.
4. Add optimized microbenchmarks and allocation/copy counters for rules, state restoration, feature preparation and scripted search as each becomes available. Establish debug, profiling and portable release builds plus formatting/lint checks.

**Exit:** core and worker compile natively for Windows x64 on this machine; native tests, lint and handshake/error checks pass. The benchmark skeleton records build identity and raw measurements. Unsupported contracts are rejected explicitly, and the reference path still works.

#### L04 — Implement complete state import, export and restoration

**Targets:** proposed core `state.rs` and state DTO modules; `tests/RustBackend/test_state_parity.py`.

1. Import the selected semantic state DTO, including actions remaining, phase, pending payments/continuations/bonuses/debt, public history and stable tile/card/seat identities.
2. Rebuild derived connectivity/resource indexes, check conservation/invariants and implement export plus clone/restore without shared mutable state.
3. Compare canonical roots and round-trip exports against fixtures at every supported phase; test malformed inputs and unsupported configurations.

**Exit:** exact discrete state agreement, valid round trips and clone isolation. A mid-action snapshot is reconstructed as that state, not approximated as a new turn.

#### L05 — Port staged legal choices, transitions and chance

**Targets:** proposed core `choices.rs`, `phases.rs`, `chance.rs` and rule adapters; `tests/RustBackend/test_rules_parity.py`.

1. Implement ordinary action, payment, continuation, bonus and debt choices with the reference semantic IDs, ordering and full legal sets. Generate staged choices without enumerating every complete atomic plan first.
2. Apply staged transitions and explicit chance outcomes; handle turn/era boundaries, forced liquidation, final settlement and the declared target rules' tie-breaking.
3. Replay the same legal choices/chance events through C# and Rust and compare the next actor/phase, resources, public consequences and terminal outcomes. Retain a regression case for each rule fix.

**Exit:** the representative transition suite has zero unexplained legal/state/reward differences; invalid choices are rejected without partial mutation.

#### L06 — Port permitted observations, recall and world sampling

**Targets:** proposed core `observation.rs` and `belief.rs`; existing `searchlab/spiel.py` and its sampling/recall helpers; proposed `test_privacy.py`.

1. Produce each seat's permitted observation and preserve own-hand recall, public history, wildcards and hidden-card conservation through both eras.
2. Reconstruct sampled worlds from the detached permitted view and carry recall separately into information-set identity, matching the reference's actual conditioning. Test distribution/support and explicit chance semantics; use recorded draws for deterministic fixtures.
3. Test all four seats, sampled opponent observations and counterfactual live hidden hands/deck orders. Keep policy-facing views distinct from the environment's internally complete state.

**Exit:** masking, conservation, recall and sampler checks pass; changing unavailable live hidden information cannot change root policy inputs or recorded-draw search behavior.

#### L07 — Expose the reference simulator adapter and qualify full games

**Targets:** proposed `src/brass_ai/native/simulation.py`; the `BrassSpielGame` / `BrassSpielState` boundary; rules/privacy fixtures.

1. Expose Rust equivalents for initialization, permitted-root construction, current player, legal choices, apply, chance, terminal returns and clone/restore through an explicit local diagnostic adapter.
2. Replay complete four-player games and import/restore at arbitrary staged checkpoints. Expand to the section A7 corpus and classify every unsupported case or mismatch.
3. Verify that this adapter can be selected without changing the Python search algorithm. Retain its per-move overhead as an intermediate diagnostic, not the finished native search design.
4. Profile legal generation, sampled-root construction, apply and clone/restore on simple and dense states. Remove repeated parsing/history reconstruction and unnecessary copies. Compare compact copying with a journal only if copying remains a significant cost, and rerun restoration/cancellation parity for the chosen design.

**Exit:** the selected full-game/state/observation/return suite passes with zero unexplained differences, variant B is available and a rules/allocation profile identifies the remaining dominant costs. The native hot path meets the applicable section A2 requirements. This is the first usable simulator milestone.

#### L08 — Port exact features and verify the admitted v11.2 checkpoint

**Targets:** proposed core `features.rs` and `tests/RustBackend/test_features.py`; existing Features-4/derived-context encoders, model factory and checkpoint loader.

1. Implement native Features-4 and the checkpoint's additional inputs, preserving pinned feature meanings, ordering, masks, known/applicable flags, legal candidate encoding, seat mapping, personality context and remaining utility budgets.
2. Compare Rust/Python inputs on roots, simulated future states and all staged/forced decisions. Reuse the same admitted checkpoint in the existing inference runtime and compare policy/value outputs.
3. Include the v11.2 [B,4,34] critic descriptors: unknown deployment defaults, relative-seat mapping, policy independence and explicit known/unknown output parity. Exercise admitted frozen-opponent schemas and cache identities. Derive dimensions from the chosen manifest; do not add folded features or retrain the model to conceal an encoding mismatch.
4. Encode into reusable typed buffers, prepare common permitted facts once per matching decision and retain candidate offsets. Measure full legal-set encoding, cache hits and misses; avoid an intermediate Rust-object-to-JSON-to-Python encoding path.

**Exit:** exact IDs/masks/order and declared floating-point/output tolerances pass, with evidence that the pinned checkpoint remains usable. Any semantic incompatibility is resolved or explicitly versioned before proceeding.

#### L09 — Qualify same-process ONNX, batching and worker lifecycle

**Targets:** proposed `native/brass-ai-inference/`, worker `protocol.rs` / `server.rs`, `src/brass_ai/native/protocol.py` and `test_inference_protocol.py`; existing inference/export code as the oracle.

1. Define one typed evaluator interface with a reference-service adapter and the required local CPU ONNX adapter. Load pinned sessions in the Rust AI process, preserving model/seat/request/batch/generation identities, input/output schemas and candidate ordering.
2. Implement native policy postprocessing on each session's one thread, buffer ownership, bounded session-local queues, response matching, timeouts, cancellation and explicit worker failure. Keep model loading and warm-up separate from timed work; retain ownership until an in-flight runtime call actually finishes.
3. Verify Python/PyTorch, Python/ORT and Rust/ORT parity, including full policy/value responses, frozen-model routing and singleton/mixed batches. Test CPU operation with the external evaluator unavailable, plus stale replies and binding/buffer lifetime failures.
4. Profile queueing, packing, copying and inference separately. Establish initial bounded flush/backpressure settings, fixed one-thread-per-session settings and the section A5.1 batch-sweep harness. Final tuning waits for native search; preserve the external adapter for matched attribution and GPU collection.

**Exit:** exact native inputs and the admitted ONNX export pass all output tolerances; CPU evaluation executes locally without Python feature encoding or inference RPC. Cancellation/error tests leave no orphaned buffers or accepted stale replies. The local evaluator is ready for native search, and the reference adapter supports parity; complete trajectories are verified in L12 and Part 2 B02.

#### L10 — Port deterministic single-lane Masked ISMCTS

**Targets:** proposed core `search.rs` and `tests/RustBackend/test_search_trace.py`; existing `search/ismcts.py` and `search/common.py`.

1. Implement information-set keys, evaluation caching, PUCT/FPU, progressive expansion, protected families, reopening, tail admission and frozen opponent action sampling with the typed arenas, compact IDs and resumable trajectories specified in section A2. Preserve semantic equality and collision checks.
2. Preserve the new-future-normal-root-information-frontier leaf rule, terminal evaluation, stable four-seat backups and per-seat utility/cost ledgers. Keep upstream PUCT outside the matched implementation.
3. Feed recorded worlds/chance/ties and scripted policy/value outputs to both implementations; compare chosen edges, frontier decisions, backups and completed counters after each trajectory.
4. Instrument staged choices/actions and time before leaf evaluation so the proposed rollout savings can be measured. Keep the section A4.1 earlier-leaf experiment behind a separate search identity; its adoption waits for the L14 quality comparison.

**Exit:** deterministic traces agree under declared numerical tolerances. Single-lane execution is a correctness tool; additional worker-count performance comparisons are not required.

#### L11 — Port whole-turn sessions and complete-plan selection

**Targets:** proposed core `session.rs`; existing `personality/search/engine.py` and its contracts; proposed `test_sessions.py`.

1. Port turn keys, profile/context lifetime, own recall updates, root/tree/cache reuse and minimum completed-plan support.
2. Convert the selected staged trace into validated complete actions and continuation preconditions. Retain separate commits/presentation for the first and second action.
3. Test changed legality, failed continuation, stale revisions and replanning against the same remaining ordinary-turn budget. Test debt as the reference's separate non-search path. Verify context/utility identities survive every sub-choice.

**Exit:** reference and Rust produce compatible supported plans; invalid continuations are discarded, ordinary-turn sub-choices grant no extra search time and debt/forced decisions follow the declared caller contract.

#### L12 — Add two-session scheduling, deadlines and transactional completion

**Targets:** native search scheduler and worker cancellation paths; proposed `tests/RustBackend/test_deadlines.py`.

1. Run two independent one-thread sessions with private trees, shared immutable definitions and separate session-owned inference paths. Aggregate only completed root/plan evidence using the existing replica semantics.
2. Implement pending reservations, exactly-once release and cooperative checks inside expensive operations. Use the pinned reference finalization reserve for the initial matched 12-second whole-turn comparison. Define the completion point at which an entire backup becomes visible; interrupted backups must never publish partial counters.
3. Inject cancellation, exceptions, queue delays and late NN replies during generation, traversal, encoding and backup. Verify that resource ownership continues until outstanding native/device work actually ends.
4. Profile both active workers with scripted and real inference. Measure allocations per completed trajectory, key construction, tree traversal, synchronization, idle time and peak memory. Remove avoidable per-move allocations, serialization and shared locks before integrated qualification; test the optimized path against the simple native baseline.
5. Exercise the complete same-process engine → Features-4 → ONNX → backup path with two independent sessions, one thread each, then run the staged batch sweep in section A5.1. Keep the frozen baseline and tuned scheduler results separately identified for L14.

**Exit:** adversarial timing tests produce no partial visits/teacher labels, leaked reservations, private-node merging or extra fallback search allowance. Report search cutoff and delivery latency separately.

#### L13 — Package the native backend and integrate Unity shadow mode

**Targets:** proposed `src/BrassLocalAI/Native/`; existing `Execution/AiExecutionScheduler.cs`, `Unity/LocalTurnBridge.cs`, native-player tests and build manifests.

1. Build pinned Windows x64 artifacts containing the Rust engine/search/encoder and loading the qualified ONNX Runtime library in the same AI process. Package admitted exports, runtime dependencies, startup/health checks and retirement; verify operation without a Python service. Keep research and deployment artifacts separately identified.
2. Implement an `IAiExecutionBackend` adapter through the existing scheduler with detached permitted requests, full identity checks, cancellation/resource ownership and isolated complete-action validation.
3. Run shadow requests without applying Rust actions. Exercise both actions, staged/debt states, new game, menu/history/save/load transitions, stale replies, worker failures and all supported configurations. Exercise 4/8/10/12-second settings and the versioned old-save mappings in A9.

**Exit:** native-player lifecycle and validation tests pass with a distributable runtime; Unity remains responsive and no shadow result mutates live state. Unsupported configurations route to the qualified backend.

#### L14 — Qualify local search performance and decision quality

**Targets:** proposed `scripts/benchmark-rust-backend.py`; versioned evidence under `artifacts/rust-backend/<run-id>/local/`; section A8.

1. Run A/B/C_ref/C fixed-work controls and actual NN search on this Windows machine, with two single-threaded sessions and one 12-second ordinary-turn allowance. Use the primary admitted v11.2 export; retain the requested width/depth comparisons only for exact compatible exports.
2. Retune per-session batch caps, pending paths and flush delays with ORT threads fixed at one using A5.1. Separate the frozen baseline from scheduler optimizations; isolate inference runtime/placement with C_ref_ORT when needed.
3. Collect component profiles, state/node sizes, allocation/copy statistics, latency distributions, supported-plan/fallback rates and held-out complex-state results. Improve measured bottlenecks one at a time and confirm without diagnostic instrumentation.
4. Run seat/setup-balanced decision-quality matches with trained weights. The optional earlier-leaf experiment has a separate identity and measured retain/reject result; baseline admission cannot silently adopt it.
5. Verify all four local time settings for timing, complete actions and cancellation. The 12-second matched study remains the performance comparison; another worker-count sweep is not required.

**Exit:** the predeclared local parity/performance/latency/memory/quality gates pass with raw evidence and no unresolved supported-state failure. An unmet target remains open with its cause and next measurement; do not lower it after inspecting candidate results. Missing trained checkpoints or representative Windows evidence keep this gate pending.

#### L15 — Test complete local games, fix failures and freeze the release candidate

**Targets:** opt-in backend selection, native Unity integration, local failure bundles and section A11.

1. Apply plans through existing complete-action validation and single main-thread commits. Revalidate action two, preserve presentation/events/saves and use only the remaining turn budget.
2. Run complete local matches through Canal, Rail, debt and final settlement, including human-versus-AI play and reproducible AI games. Test the declared supported configurations, all four allowances, old saves and fallback.
3. For every rules mismatch, invalid plan, hang, late action or crash, retain the request identity, permitted root/recall, semantic trace, independent search draws, model/build manifest, error/timing data and expected outcome. Keep any trusted full-state debug fixture separate from policy inputs. Reproduce in the local harness, fix, rerun affected checks and add a regression before promoting a new candidate.
4. Exercise worker failure, model mismatch, history navigation, game replacement, save/load and disabling Rust. A canceled/superseded request commits nothing; a current failed request uses only its bounded validated fallback.
5. Freeze a local acceptance receipt tying together the exact source/config/model/runtime hashes, Windows binaries, support matrix, parity reports, performance/quality evidence, full-game results, issue status and tested rollback.

**Exit — Local acceptance gate:** all applicable A11 checks pass, no unexplained supported-state correctness failure remains and the locally tested candidate is reproducible. Only this candidate may enter Part 2. A subsequent shared-core rules/features/search fix invalidates affected evidence and must be verified locally before the lab consumes it.

### A7. Local correctness corpus

Use real allowed observations and trusted full-state fixtures from the selected reference. Include the six earlier benchmark positions only as starting examples; they do not cover our rule/phase contract. Start with at least 100 varied roots and 20 complete four-player games across multiple seeds, retaining every discovered regression. Publish coverage by era, phase, action family and configuration so an aggregate count cannot conceal a missing category. These are initial coverage targets, not proof for all possible games.

| Area | Required cases and comparison |
| --- | --- |
| Turn structure | Opening one-action turn, ordinary two-action turns, income/spending/turn order, empty deck, era transition and final settlement |
| Build and network | Own/opponent connectivity, nearest-coal ties, external-market access, coal/iron exhaustion and overbuilding, stable slot/link IDs |
| Multi-stage actions | Ordered two-tile development/payment, ordered multi-sales, merchant beer used once, brewery ownership/distance rules, merchant/free bonuses, optional second Rail link, Scout combinations and Pass |
| Debt and terminal rewards | Loan floor, negative cash/income, forced liquidation by the correct owner, all mandatory final debt completion and tie-breaking |
| Hidden information | All four seats' views, known wildcards, card-pool conservation, own-hand recall, hidden setup discards and Rail reshuffle without double-subtracting old discards |
| Large states | Mid/late Canal and Rail, dense networks, resource ties, many sale orders, low cash and very large legal-action expansions |
| Lifecycle | Import at every supported phase; clone/restore independence; save/reload; malformed/unsupported DTO rejection; cancellation during generation, sampling, encoding, NN wait and backup; canceled requests apply no fallback action |
| Model/search | Exact semantic masks and IDs, input/cache identities, four-seat outputs, reward ledgers, deterministic selection/frontier/backup traces, complete-plan support and two-session aggregation |
| Local timing/settings | Four selectable allowances (4/8/10/12 seconds), two independent one-thread sessions, shared turn clock, versioned old-save mapping and no extra inference threads |
| Native implementation | Arena-handle lifetime, deliberate key collisions, incremental-cache invalidation, buffer growth/reuse, apply/unapply equivalence if used, numeric overflow and optimized-build behavior |

At each staged transition compare legal semantic choices, actor/phase, permitted views, public consequences and eventual complete-action outcome. At terminal states compare scores, tie-breaking and training rewards, not only board layouts. Normalize representational differences; never normalize away a rule difference.

For information-leak tests, change only actual hidden opponent cards/deck order while keeping the root permitted information and recorded sampling inputs fixed. The search-facing tensors and behavior must remain unchanged. Separately verify that simulated opponents receive their own sampled private information, not the root's view or live hidden information.

Training-only critic eligibility, branch families and optimizer checks are added in B3–B4 after local acceptance. Local reward-ledger tests already cover the admitted model's causal insolvency budget, including partial final charges and exhausted caps.

### A8. Matched local benchmarks

#### Search comparison: two one-thread CPU sessions, 12 seconds per searched ordinary turn

| Variant | Engine | Search | Inference | Purpose |
| --- | --- | --- | --- | --- |
| A: reference | Current admitted C# host | Our Python Masked ISMCTS | Pinned reference evaluator | Actual baseline |
| B: simulator replacement | Qualified Rust through reference adapter | Same Python Masked ISMCTS | Same evaluator as A | Engine replacement, including adapter costs |
| C_ref: native core control | Qualified Rust | Faithful Rust Masked ISMCTS | Same evaluator as A | Isolates native tree/encoding/trajectory execution before changing inference placement |
| C: production CPU backend | Qualified Rust | Same faithful Rust Masked ISMCTS | ONNX Runtime in the AI process | Intended complete backend, with native Features-4 and local batching |
| D: optional upstream baseline | Upstream Rust | Upstream PUCT | Explicitly recorded | Separate algorithm experiment, excluded from equivalent-search speedup claims |

A→B includes the changed host/adapter boundary. B→C_ref includes native search, encoding and state ownership. C_ref→C measures the inference-path change; A→C is the deployment comparison. If the reference evaluator is PyTorch, add a `C_ref_ORT` run using the same external ONNX export/runtime settings as C to isolate placement/IPC savings from runtime differences. Do not attribute all differences to programming language alone.

Keep the earlier-leaf experiment under a separate identity, `E`, compared with C at the same scheduler settings. Then report any additional retuning separately. A/B/C_ref/C use the same baseline leaf contract, so a reduction in work per trajectory cannot masquerade as a faster implementation.

#### Performance targets and regression limits

The following are initial engineering targets, not measured results or promised speedups. Measure A, then freeze the final thresholds and target machine in the baseline manifest before evaluating B/C. Use fixed-trace critical-path timings to check feasibility: if an unaffected component occupies fraction `f` of that path, accelerating everything else has an idealized ceiling of `1/f`. Account for overlapping workers/inference rather than adding overlapping phase times. Any target adjustment must be justified by baseline evidence and recorded before candidate results are inspected.

| Gate | Initial target | Measurement and interpretation |
| --- | --- | --- |
| Native rules and search work | At least **2×** A→C throughput on the fixed-work rules/scripted-search suite | Same sampled worlds, staged transitions and scripted NN results; both engines execute equivalent work. Report rules and tree-management components separately. |
| Real NN search | At least **1.25×** A→C completed trajectories per searched ordinary turn for the primary admitted model | Geometric mean of paired fixture ratios, with a confidence interval supporting a benefit; two single-threaded sessions and the same 12-second session contract. Report plan coverage and strength separately. |
| Complex-state regressions | No reproducible median slowdown greater than **5%** in a declared critical era/phase group | Keep dense Rail, resource ties, multi-sale and near-terminal results visible. Expand inconclusive samples; an aggregate gain cannot hide a severe case. |
| Turn latency | No counted search work after the reference work deadline; warm response-ready p95 at most **12.05 s**, maximum observed at most **12.15 s** from the ordinary-turn clock start | Targets for controlled target-hardware qualification, including finalization and transport. Gather enough turns for tail claims; report preparation, inference retirement, Unity dispatch and animation separately. |
| Memory and lifecycle | Stay within the declared tree, pending-path, tensor and replay budgets; no unexplained growth across repeated turns/games | Report peak/high-water and post-retirement memory, allocations per useful unit, canceled-job draining and retained shared model caches. |

The fixed-work gate is the clearest test of native engine/search efficiency. Real NN speedup can still be limited by model computation or by different explored paths under a time budget. Use C_ref/C and the batch sweep to quantify inference placement and scheduling. After co-location, use profiles to decide whether remaining packing, padding, thread contention or model computation dominates. An unmet target remains an unmet objective; do not replace it with a favorable microbenchmark or lower it after seeing the candidate.

Freeze concrete memory budgets from the selected model, node/edge limits, maximum pending paths, legal-choice tensors and game-lane count. Measure the largest supported cases. Arena or buffer exhaustion must take the declared failure/fallback path, preserve full legal sets and appear in coverage statistics.

#### Procedure

1. Use one fixed fixture set spanning early/middle/late Canal and Rail, plus payment, continuation, bonus and near-terminal cases. Include dense late-Rail roots and high-branching cases where the earlier benchmark needed native deadline interruption. Debt and forced-choice fixtures check the reference's non-search behavior and are reported separately from search throughput.
2. First run fixed-work microbenchmarks: same legal generation, apply, clone/restore, sampling, encoding and deterministic search trace with scripted NN responses. Measure equivalent units of work, with no NN path variation.
3. Run actual NN-guided search with identical architecture/weights, feature/context/reward contracts, frozen bank, search/leaf parameters, finalization reserve and worker count across A/B/C_ref/C. Hold evaluator/runtime settings fixed for each attribution pair and identify deliberate differences. First use matched batching/pending/flush settings; report the tuned C configuration separately. Validate outputs before interpreting throughput.
4. Use two independent sessions, one thread each, and **one 12-second whole-turn allowance**, shared by first-action choice, staged completion, second action and replanning. A resumed mid-turn fixture carries its original remaining budget; it does not receive another 12 seconds. If a separate microbenchmark gives a staged root a fresh budget, label it as a per-root diagnostic and exclude it from whole-turn results. Reset session/cache state consistently between paired trials.
5. Reuse recorded worlds/chance/policy responses for deterministic comparisons. Actual NN runs use paired fixture/setup schedules and at least five measured paired repetitions initially. Record setup fingerprints and RNG versions; equal numeric seeds across languages are insufficient. Counterbalance implementation order and run trials sequentially on the same machine.
6. Record warm-up, cold startup/model load, excluded preparation and timed search separately. Keep power mode/AC state, process load, temperatures where available, fixed one-thread-per-session settings, actual ORT/compute-thread counts and runtime/build versions in the evidence. The earlier battery-powered WSL result is not a Windows deployment qualification.
7. Use the admitted v11.2 checkpoint as the current model baseline (5,928,616 parameters for the medium utility model), recording its exact export and descriptors. Retain the requested W352×6, W448×6 and W512×8 size comparisons only after their exact exports are admitted to the same semantic contract. Pair each size across A/B/C. Initialized weights may measure mechanical cost; trained matching checkpoints are needed for representative search behavior and strength claims.
   The cited v9 / W448×6-with-v11-features / full-v11 report uses different exact architectures and parameter counts. If those are the selected deployment candidates, benchmark their pinned exports too; do not identify models only by trunk width/depth or combine their results with the older folded-feature experiment.
8. Use optimized builds with identical instrumentation, runtime-loaded fixtures and consumed/checksummed results so the compiler cannot eliminate benchmark work. Collect sampled CPU profiles and allocation/copy counts in diagnostic runs; confirm timings without those diagnostics. Tune on the development fixtures, then evaluate the frozen candidate on held-out complex states.

#### Reported measurements

- Legal choices generated, applied staged transitions, completed trajectories, frontier evaluations, completed root-edge/complete-plan visits and unique information sets. Define each counter once across implementations.
- Choices/actions per trajectory, time to the first value leaf and terminal versus NN-leaf frequency. These establish whether the reported long-trajectory cost exists and distinguish implementation speed from the earlier-leaf experiment.
- Time in generation, apply/restore, sampling, encoding, inference queue/compute and transport; NN positions/batches, actual batch sizes, cache hits, memory and CPU consumption.
- Allocations and copied/transferred bytes per completed trajectory/decision, state/node/edge sizes, arena high-water marks, key-construction cost, lock/queue contention and worker idle time. Distinguish useful NN candidate cells from padding and compare cold cache, warm cache and invalidation-heavy cases.
- Completed work per whole-turn second, searched plan coverage/depth, valid-plan rate, fallback rate, deadline interruption reasons and late/stale responses.
- Search cutoff, response-ready and live-apply latency; sample counts, median, observed quantiles and maximum latency. Five repetitions per fixture are an initial screen, not a reliable tail-latency estimate. Collect more evidence before making p95 or rare-failure claims.
- Separate equal-time playing-strength evaluation with seat/seed balancing after correctness. More trajectories is not itself evidence of better play.

Apply the frozen targets above and separately declared strength criteria. Report paired per-fixture ratios and uncertainty, keeping complex-state results visible. Include failures, timeouts and fallback routing in the full-fixture report; native-only speed must not conceal poor coverage. If the evidence cannot distinguish a benefit from noise, report the result as inconclusive. Deployment requires semantic parity, bounded behavior, valid actions and a measured benefit meeting the declared criteria.

For each optimization, retain an ablation against the preceding native implementation: compact copy versus journal, uncached versus cached projection, buffer reuse, batch policy and compiler profile as applicable. Change one factor at a time where practical. Retain the faster, simpler design supported by representative results; revert changes whose advantage disappears in the complete workload. Use the benchmark suite as an ongoing regression check after the migration.

Attribute the requested savings explicitly: engine in-process/restore costs through A/B and component traces; native encoding and tree execution through B/C_ref plus fixed-input encoder tests; local inference through C_ref_ORT/C; batch retuning through matched C scheduler variants; earlier value leaves through C/E quality tests. The rollout and batching changes are algorithm/scheduling choices available in other languages as well.

Use the same **2 sessions × 1 thread** constraint for the native CPU arms. Record reference evaluator placement, thread settings and any unavoidable external service overhead; an unfair extra inference pool must not enter the claimed matched result. Measure the existing C# V4 gameplay backend separately as a deployment/quality control because its heuristic policy is not our Python neural-search baseline.

Local production timing checks cover all four 4/8/10/12-second choices, while A/B/C performance ratios remain at 12 seconds. Use the same 50 ms p95 / 150 ms maximum response-ready engineering tolerances relative to each setting, subject to the declared sample counts and target hardware; animation/preparation are reported separately. Do not omit zero-completion roots from aggregate results: report their absolute completed work, fallback and valid-plan rates, and define treatment of undefined ratios in the frozen protocol.

### A9. Unity deployment, four timing settings and rollback

Integrate through the current `IAiExecutionBackend` / scheduler lifecycle rather than adding a second independent turn coordinator. Keep `LocalTurnBridge`'s detached observations, generation/instance/revision/actor/request/profile guards, isolated complete-action validation and single main-thread commit. Revalidate a planned continuation after the first action; a second action retains its later-frame presentation and remaining turn allowance.

Game replacement, history navigation, menu return and destruction must invalidate the request and cancel work. Hiding the panel alone follows the existing lifecycle. The scheduler must not launch another backend while canceled native/inference work is still using resources. Preserve the existing resource/FPS scope and cleanup only after actual completion.

Build and qualify Windows x64 release artifacts for the game; build the lab platform artifacts only in Part 2. The WSL/PyO3 benchmark binary is not a Unity plugin. Pin ABI/protocol, target CPU requirements, model/runtime dependencies and source/binary hashes in a new manifest. Isolate upstream web/application dependencies from the simulation core and never package private Steam assemblies as part of the Rust engine.

The current shipped game has no installed neural runtime. For L13, package the Rust AI executable, compatible ONNX Runtime native library, admitted model exports and manifests. Verify startup, CPU provider selection, model loading, tensor/output parity, health checks and retirement on Windows with the reference Python service unavailable. Keep warm-up separate from timed turns and keep sessions/model buffers resident across turns. The Unity-to-worker process boundary remains; engine/feature/search/inference calls stay inside the AI process.

Backend selection must be explicit and traceable. Preserve strategy/profile/source identity; the new neural implementation gets its own backend identity. Integrate the admitted four-setting catalog/schema for exactly four options labeled **4s / 8s / 10s / 12s**, as Easy/Mid/Hard/Max respectively, retaining the current Hard (10s) default and two independent one-thread sessions at every setting. Every menu/description/player summary reads the same catalog.

For compatibility, preserve original signed save payloads and use a versioned execution adapter: legacy named Easy/Mid/Hard map to **4s/8s/10s**; an unnamed legacy 10-second record stays **10s**. The new Max option selects 12s. The in-progress catalog uses `local-ai-execution/1.2.0`; verify the admitted source/receipt at implementation time. Validate old 1.0.0/1.1.0 difficulty records, source/profile identities and new saves; never rewrite historical signatures in place or keep a hidden 15s execution path behind a 12s label. Test old-save load, save again and reload at all mapped settings. Existing optimizers, checkpoints and replay remain owned by their current training code.

On an unsupported snapshot, worker failure, incompatible model/contract or invalid plan, discard the result before mutation. Invoke the qualified bounded fallback only if the original game/request is still current. A canceled or superseded job applies nothing. Keep a runtime switch to disable Rust for subsequent requests. Backend changes invalidate incompatible trees/caches and are recorded in teacher provenance; they must not silently blend incompatible labels into an admitted dataset.

Keep the two sessions under **one** existing Unity execution-backend task/turn lease. Both threads and all their runtime calls must retire before scheduler resources are released or another turn starts. The Part 1 package must work with Python unavailable, and remain switchable to the retained qualified backend. A fallback inherits the current request's remaining allowance; timing migration must not grant a new clock.

### A10. Local risks and mitigation

| Risk | Mitigation / evidence needed |
| --- | --- |
| Upstream implements a subtly different rule or history model | Differential state/choice tests plus independently specified edge-case expectations; resolve differences against the declared target rules |
| Atomic enumeration explodes in a dense state | Preserve staged choice generation; add internal cancellation; optimize allocation/caches only after parity |
| Rust reproduces Python-style allocation and serialization costs | Typed native state, staged generation, reusable buffers, arena-owned search and early allocation/copy profiles |
| Incremental state or cache optimization produces stale results | Explicit invalidation dependencies, full recomputation oracle, collision tests and restore/cancellation regressions |
| Two single-threaded sessions wait or duplicate model memory | Session-owned trees/inference, cooperative local batches, thread-count verification and measured model/cache memory; no extra inference pool |
| The previous cap-2 result is treated as universal | Retest actual occupancy, flush delays and pending paths within fixed one-thread sessions after native encoding and same-process inference; qualify with two independent search sessions (one thread each) |
| Earlier value leaves increase visits but weaken plans | Separate search identity, calibrated actor/phase values, complete-plan checks and equal-time strength tests |
| Native ONNX differs from the reference evaluator | Pin exports/runtime settings, compare full native postprocessing and mixed batches, and retain an external ORT placement control |
| Features or model contracts change during the port | Freeze a manifest for each qualification run; version later changes and rerun affected tensor/search checks |
| Reused full states or cache keys leak hidden information | Separate trusted debug imports from permitted requests; counterfactual masking tests and per-owner private trees |
| NN queues or native loops outlive canceled turns | Deadline-aware admission, stale-reply rejection, bounded buffers and scheduler completion that includes all outstanding work |
| Raw counts exaggerate speed or playing strength | Common counter definitions, fixed-work controls, equal-time matched NN runs and separate strength matches |
| Integration disrupts old saves or current local AI | Additive backend selection, isolated native-player tests and retained qualified fallback |

Rules/state/feature parity is the largest correctness risk. State representation, legal generation, feature construction and inference coordination are the main performance questions to resolve with early profiles. Keep the native design simple enough to diagnose, and qualify the integrated Unity backend locally before migrating training. Local acceptance is necessary for the lab rollout, while Linux/GPU and training-data correctness still need their own Part 2 evidence.

### A11. Local acceptance checklist — prerequisite for Part 2

- [ ] Source, rules, model, feature, descriptor, search and reward baselines are pinned, including upstream patches and the installed fallback.
- [ ] Rust imports every supported phase and reproduces legal staged choices, chance, observations and terminal outcomes; every discovered mismatch has a regression.
- [ ] Permitted sampling, all-seat masking and recall pass; trusted debug state never enters production policy requests.
- [ ] Current v11.2 input/output parity passes, including critic-only descriptors, unknown deployment defaults, full masks/order and unchanged policy logits.
- [ ] Both independent search sessions run on one thread each, including native features and single-threaded same-process ONNX; no shared inference dispatcher, additional compute pool or per-leaf Python RPC is present.
- [ ] Faithful Masked ISMCTS, four-seat backups, newer deadline admission, session reuse and complete-plan evidence pass deterministic and asynchronous tests.
- [ ] Native layout, arenas, buffer ownership, cache equality/invalidation and allocation/copy profiles meet A2; formatting/lint and debug/optimized checks pass.
- [ ] Adversarial cancellation and deadline tests show only completed timely work; scheduler ownership lasts until both sessions retire, and stale/canceled requests apply no action.
- [ ] Exact **4s / 8s / 10s / 12s** choices, the current Hard/10s default, shared ordinary-turn clocks and versioned legacy-save mappings pass menu/runtime/save tests.
- [ ] A/B/C_ref/C use matched 12-second whole turns, two one-thread sessions, complex and held-out roots, and trained checkpoints; inference-placement and batching changes have separate evidence.
- [ ] Predeclared local throughput, latency, memory and quality criteria pass; a shorter leaf policy is adopted only under its separately tested search contract.
- [ ] Native Windows package passes shadow testing and complete live games across both eras, both actions, debt/final settlement and the declared support matrix.
- [ ] Validated fallback, backend disable/rollback, old-save load and resource/FPS restoration are reproducible.
- [ ] A release receipt records all tested identities, commands, results and issue disposition. No unresolved supported-state correctness issue remains. **Part 2 is still pending until this receipt exists.**

The first server build must reference this receipt and the same shared-core source. Reproducing a bug locally is the preferred diagnosis path; server-specific behavior additionally needs Linux/GPU evidence. A shared-core fix reopens the affected local acceptance checks before another training canary.

## Part 2 — Lab server: migrate training after local acceptance

### B1. Entry gate and current self-play contract

**Do not start this migration before the Part 1 local acceptance receipt exists.** Reuse its exact shared-core source, rules, feature encoder and faithful search contract. Build a separate Linux artifact and qualify its platform-specific behavior. Local CPU results do not certify Linux/Mono/CUDA behavior or training-data compatibility.

At entry, inspect the actual lab run's committed checkpoint, saved config/signature, generation, pending collections and source receipts. This plan reviewed local code and exported reports; it does **not** establish that the live server has adopted g850. Use `latest_model_config()`, `training_revision_g850.configure()`, the generated run config and the saved signature as appropriate. Older static templates and archived guide sections are not authority for a resumed run.

Freeze backend replacement separately from a recipe change. For an already-g850 run, preserve its exact settings/state. For an older run, either qualify Rust against that admitted recipe or finish the separately reviewed recipe adoption first; do not combine two unexplained changes in a performance comparison.

| Current code/recipe area | Contract to preserve or qualify |
| --- | --- |
| Model | v11.2 architecture, Features-4/derived inputs, critic-only `seat-descriptors/1.0.0`, utility/reward identity and any separately checkpointed auxiliary heads |
| Ordinary policy | Policy-only learner actions. Under g850, temperature 1.5 for each neural seat's first two action decisions, then 1; fresh counters after a branch fork; epsilon 0.05 through the existing family/industry/context distribution |
| Behavior evidence | Actual temperature-plus-epsilon chosen-action probability, raw policy evidence, per-seat action-decision count and versioned opening/branch contracts. Old T2/epsilon-zero records cannot be relabeled |
| Opening PPO objective | `opening-ppo-objective/2.0.0`: retain the raw-T1 target surrogate with the actual detached behavior denominator; Rust supplies correct collection evidence, Python applies the objective |
| Branches | `selfplay-branching/3.0.0`: 0.010 per eligible decision, at most one child per eligible self-play parent, existing early/final-round exclusions, excluded fork action and deck-only reshuffle |
| Entropy | Fixed coefficient 0.005 for all four groups, adaptive controller off. Preserve historical controller/baseline/rollout evidence and report `effective_alpha`; migration does not reset it |
| Rewards | g850 insolvency 0.02/event capped at 0.075/seat/game; fourth charge 0.015, then zero. Keep causal remaining budgets, the selected combined-return/VP-cap contract, and search/learner agreement; frozen opponents retain their own reward identities |
| Critic rows and descriptors | Current fresh generator uses fixed snapshot weight 1.0 (`critic-rows/3.0.0`); a resumed run retains its signed mode/state. Learner/snapshot attribution is by policy identity, not equal model weights; IL/v4 rows remain excluded from critic fitting |
| PPO/GAE | Preserve current learner nonforced actor/GAE eligibility, full legal denominators, strict behavior reconstruction, DCPG preservation, LR/IL-KL controls and the exact 2,048-row categorical KL control. Sampled KL stays diagnostic |
| PPG | `ppg-value-phase/15.0.0`: every iteration, current plus previous committed critic age, uniform row weights, shuffle each epoch, maximum two epochs |
| PPG acceptance | Both ages' complete held-out game/branch families are excluded from fresh critic fitting; accept strictly non-increasing pooled row-mean held-out loss, with no SE allowance. Cloning/KL and GAE/snapshot evidence remain current-only |
| PPG LR/guards/state | Peak 0.65 × last actually applied PPO LR; attempted-step cosine warmup from 0.1× at step 1 to peak at 16, then decay to 0.3× at the full cap. Loss and KL checks every 16 attempted steps plus tails/final verification; hard KL 0.002; beta/trips preserved with bounds 0.5–32; matching weight/Adam rollback |
| Auxiliary labels | Preserve enabled `local-scoring-targets/1.0.0` current-era ownership/scoring records, topology/order and causal fork origin. These are targets, not permitted actor features; companion head/optimizer remain in Python |
| Search enablement | Current generated recipe explicitly sets `teacher_enabled=False`, `v4_search_enabled=False` and `search_wall_fraction=None`. Preserve these switches and disabled Monitor/Broader Evaluation settings; porting search does not enable it |
| Job/population schedule | Python retains frozen model/policy assignments, quality/PFSP pool, learner seats (4 in self-play, 2 in mixed tables), rotations, profile cycle, table-share/anchor state, setup splits and collection/update barrier |
| Inference/preparation | Preserve streaming frozen-model queues, candidate-cell/padding limits, urgent/aged-request fairness, reply envelopes, shared-slot ownership and per-collection telemetry. Preserve bounded Python input caches/prefetch and recompute predictions after optimizer changes |
| Ratings | Python retains the statistical PL contract, slots, priors, seed/bootstrap draws and promotions. g850 compute uses warm starts and 200 refits, four single-threaded CPU processes by default (4–8 within the CPU-reader budget), with failure/convergence safeguards |

These are compatibility requirements, not requests to rewrite PPO, PPG or ratings in Rust. Consult the current code again at B01; any newer accepted recipe needs its own manifest and tests.

### B2. Lab ownership and transport

```mermaid
flowchart LR
    P[Python frozen job planner and admission] -->|Jobs, policies, contexts and provenance| C[Rust ordinary game and branch lanes]
    C <--> G[Locally qualified Rust rules and exact encoder]
    C -->|Typed compatible evaluation requests| I[Existing frozen-model GPU inference service]
    I -->|Policy responses and identities| C
    C -->|Complete sealed game families and scoring receipts| R[Python replay validation and preparation]
    R --> T[Existing PyTorch PPO, PPG, auxiliary fitting and ratings]
    T -->|Committed frozen checkpoints for next collection| P
    P -.->|Only when explicitly enabled| S[Qualified Rust search adapter]
    S <--> G
    S <--> I
```

Python keeps training orchestration, population/recipe decisions, GPU inference ownership, dataset admission, reward/advantage targets, backpropagation, optimizer state, checkpoints, ratings and recovery. Rust takes environment progression, staged legal generation, observation/encoding and branch-state execution; the same search core serves enabled search consumers.

Use the existing GPU evaluator first. Do not add a Rust training runtime, require ONNX after every PPO step, or copy the CPU-local inference layout onto CUDA. Retain batch grouping by model/schema, observer mapping, exact candidate order, bounded backpressure, cancellation/late-result rules and frozen-model boundaries. Shared memory or packed tensor transport is an optimization after correctness; no per-transition C# rules RPC in native collection lanes.

The local **two sessions × one thread** contract stays fixed for local gameplay/search qualification. Ordinary lab game lanes, GPU service and Python preparation/rating pools retain the actual signed lab resource recipe; they are not all limited to two threads. Existing lab search worker/budget settings, if explicitly enabled, are a separate qualified profile. Do not silently change them or start a scaling study as part of this plan.

Keep immutable source/model hashes identical across platforms where applicable, while recording Windows/Linux binaries, toolchains, CPU features, runtime providers and platform-specific patches separately. Windows acceptance cannot be replaced by a Linux-only fix.

### B3. Collection, branch, replay and optimizer boundaries

#### Complete ordinary records and exact behavior

Run complete games in independent Rust lanes from Python's frozen jobs. Preserve every predecision view and ordered full legal set, including action two, payment, continuation, bonus, debt and forced choices. Record semantic ID/index, style/intent, observer/seat mapping, behavior policy/model IDs, opening count, temperature, exploration, actual sampled probability, descriptor metadata, rule/backend/feature/source identities and auxiliary receipts when enabled.

A forced choice has probability one and does not need a policy forward. It stays available to the critic under the current admission rules; it does not become an actor/GAE row. Learner classification uses policy identity even when a frozen snapshot has the same weights. Keep actor, learner critic, snapshot critic, IL/rule-bot exclusions and complete-game holdout selection unchanged.

The ordinary environment owns true hidden state, but every policy and feature encoder receives only its permitted view. Training seat descriptors come from sealed frozen job metadata and feed critic heads only. Root search/deployment continues to use unknown descriptors. Training-only future local-scoring outcomes must never appear in Features-4, policies, belief inputs or search keys.

Use scripted responses/recorded draws for cross-language collector parity. Equal numeric seeds alone do not imply identical Python/Rust random streams. Existing replay validators reconstruct specific branch seed/draw choices; implement compatible RNG/selection for those receipts or retain that discrete decision in an explicit trusted adapter. A merely statistically similar fork choice must not bypass deterministic admission.

Support the current search-free V4 and frozen historical opponents through exact admitted policy adapters, keeping profiles/behavior unchanged. If a table is unsupported, route the whole frozen job through the reference collector and report that coverage/cost. Do not change opponent sampling to increase the native percentage. A claim of complete native coverage requires every job type in the selected recipe to be qualified.

Publish a complete admitted game/family once with stable job/game identity, content hash and commit acknowledgment. Reuse a previously validated complete result on retry; reject stale, partial and duplicate publication. A fallback heuristic action is not an NN learner sample.

#### Fork continuations and whole-family publication

1. Capture the exact trusted environment checkpoint at an eligible branch point, including pending stage, hands, deck, public history, contexts, per-seat opening counters and prior insolvency events. Keep this internal checkpoint out of replay and policy inputs.
2. Preserve the excluded alternative-action mixture (0.70 T1, 0.25 T2, 0.05 exploration-q), per-component exclusion/renormalization, zero-support handling, chosen component and marginal probability. This mixture stays unchanged under g850's T1.5 ordinary opening.
3. Continue the parent while scheduling the ready child on an available ordinary lane. Preserve deck-only reshuffle and the child's new per-seat opening counters after the excluded fork move. Payment/bonus/continuation choices do not consume action-decision counters; no recursive extra branch is introduced.
4. Keep the fork move in authoritative causal replay but exclude it from **both actor and critic**. Preserve prior cost consumption and reset any local-scoring recorder at the exact continuation origin.
5. Preserve current ready-work scheduling: branches first, longer remaining branches first, branch-producing parents early, stable tie-breaking and optional model cohorts without consuming game RNG or redrawing opponents.
6. Join the family asynchronously; neither a free game lane nor a replay writer waits on its sibling. Publish only after parent and child complete and existing family validation passes. Bound queued checkpoints, feature buffers and unfinished families.
7. Use whole-game/branch-family grouping for critic holdouts and caches. Restart reuses complete validated families and recollects unpublished ones through the original frozen job; it must not join a new child to a stale parent or count either twice.

#### Targets, critic inputs and trainer reuse

Keep Python terminal/utility target construction, GAE, strict behavior verification, descriptor normalization/relative ordering, critic-only dropout, fixed snapshot weight or signed legacy gate, and cached source admission intact. Add explicit Rust producer provenance to replay and local-scoring receipts; do not impersonate an old C# `host_sha256` or weaken its validator to accept arbitrary producers. A versioned compatible producer route must prove equivalent labels, topology, seat ordering, era boundaries and branch origin.

Preserve the current state-only critic optimization without removing full candidate checks from policy/BC batches. Caches may reuse admitted immutable inputs; they must not retain stale network predictions across PPO updates. Retain exact per-step categorical KL, bounded packed/probe caches, ordered CPU preparation and cleanup after cancellation.

The g850 PPG held-out acceptance pool contains two ages, while progress/gate evidence remains current-only. Repeated validation of an older age must not count as another healthy iteration. Preserve the row-derived full step cap, attempted-step LR curve, beta/trip history, matching core/auxiliary weights and Adam state on rejection, and both-age cache ownership/expiry after commit.

A completed IL checkpoint with zero self-play remains supported: use its admitted v11.2 handoff or explicit completed-v11.1 upgrade, preserve original IL files/lineage, create a separate self-play run/new PPO optimizer and calibrate utility from fresh admitted ordinary data as required. Do not repeat IL or re-upgrade an established v11.2 PPO run. A normal backend migration resumes existing weights, all optimizer states and progress through an exact reviewed source/backend compatibility route.

### B4. Lab implementation sequence and exit gates

Keep lab statuses and evidence in `artifacts/rust-backend/<run-id>/lab/`, linked to the immutable local acceptance receipt. The first five steps use isolated validation runs. Production adoption is B07.

| Step | Prerequisite | Deliverable and exit evidence |
| --- | --- | --- |
| B01 | Local acceptance | Reproduce the shared core on Linux; pin the actual lab recipe/checkpoint/runtime; cross-platform fixture parity and GPU adapter smoke |
| B02 | B01 | Rust ordinary collection and branch continuations; complete family and behavior/feature/target parity |
| B03 | B02 | Strict replay/critic/auxiliary admission, checkpoint handoff and source identity compatibility |
| B04 | B03 | Disabled search stays disabled; enabled teacher profiles pass separate search/admission tests where in scope |
| B05 | B03–B04 | Bounded PPO/PPG iterations, two-age acceptance, exact resume/rollback and failure recovery |
| B06 | B05 | Matched collection and full-iteration evidence on actual lab hardware |
| B07 | B06 | Bounded isolated server canary, explicit production adoption and rollback qualification |

#### B01 — Pin lab state and qualify Linux/GPU adapters

1. Verify the local receipt, shared-core hash and unresolved-issue list. Build a Linux release from that source; record compiler/runtime/provider identities and any platform patch.
2. Snapshot the actual committed config/checkpoint, generation, source reviews, model/optimizer/RNG hashes, population, runtime manifests, pending-work status and device/CPU budgets. Resolve an old pending collection through its existing recovery path before testing a new backend.
3. Run the portable Part 1 rules/observations/features/search fixtures on Linux. Add GPU inference tensor/output checks, mixed frozen models, variable candidates, explicit/unknown descriptors, canceled/expired replies and device/service failure.
4. Keep the current streaming GPU batching/fairness and shared-slot ownership. Verify every response reaches its original model/observer/request and that stopped frames can retire before encoding.

**Exit:** Linux core agrees with the accepted local contract, GPU tolerance checks pass and the signed reference recipe is frozen. A platform failure blocks lab rollout; shared-core fixes return to local regression first.

#### B02 — Implement ordinary lanes and branch continuations

1. Consume frozen jobs without changing table mix, profile cycle, rotations, context, learner assignments or random scheduling.
2. Move staged progression, exact encoding and complete decision buffering to Rust; preserve the g850 behavior metadata and all forced/intermediate states.
3. Implement the branch contract and scheduling/publication rules in B3, including portable trusted fork import, deterministic alternative receipts, post-fork counters, causal costs and bounded asynchronous family joins.
4. Produce enabled current-era local-scoring targets/receipts from native rule events and compare them with the authored recorder at every scoring/era boundary.
5. Exercise neural, snapshot/IL and search-free V4 tables. Track any explicitly unsupported reference-routed tables. Measure component cost and memory before tuning transport.

**Exit:** complete main-game and branch-family records agree on legal order, features, probabilities, attribution, rewards and scoring targets. No lost/duplicate family, incomplete replay, checkpoint leak or mislabeled fallback is admitted.

#### B03 — Admit replay, critic data and both checkpoint entry paths

1. Add a strict backend/producer schema and reviewed source compatibility path. Keep original provenance immutable and reject changed source/config/feature/reward identities.
2. Compare Python preparation of fixed native/reference records: actor/GAE eligibility; forced learner rows; fixed-weight snapshot critic rows; IL/v4 exclusions; same-weight/different-policy attribution; descriptors; whole-family holdouts; utility and auxiliary targets.
3. Qualify an established v11.2 self-play checkpoint with all optimizer/gate/history state preserved. Separately test the optional zero-self-play completed-IL handoff, without making it a requirement to restart the existing run.
4. Verify owned replay/cache reconstruction, source seals, duplicate/stale rejection and exact checkpoint loading. Preserve the Python input/probe/preparation optimizations.

**Exit:** native records pass the existing strict training checks under explicit provenance. Both supported entry paths work; model equality cannot misclassify snapshot rows as learner actions.

#### B04 — Keep disabled search inert; qualify enabled consumers separately

1. With the current generated recipe, prove `teacher_enabled=False` and `v4_search_enabled=False` produce no Rust search requests, teacher labels or distillation updates. Ordinary NN actions remain policy-only; the retained V4 opponent uses its admitted search-free behavior.
2. For an explicitly enabled profile, route its search to the same native core and preserve that caller's phase clocks or simulation quota/watchdog, replica allocation, minimum plan support, admission/readiness and completed-only labels.
3. Test unready/incompatible utility deferral before any positive teacher case. Run a positive bounded teacher case only with an admitted ready model; do not turn teachers on in the production recipe to satisfy the migration.
4. If a historical enabled recipe uses wall-time admission, retain and measure its allowance. The current generated recipe's null wall fraction stays null.

**Exit:** the selected production switches remain exact. Enabled profiles are either separately qualified with evidence or explicitly unsupported/routed; they do not block the current disabled-teacher ordinary migration. Local search remains available regardless of training teacher enablement.

#### B05 — Qualify PPO/PPG, interruption and resume

1. Feed a fixed admitted dataset and identical incoming checkpoint/optimizer/RNG state through the existing updater. Compare behavior checks, losses, parameters, all Adam states, categorical KL, GAE/snapshot/entropy/LR state and checkpoint receipts.
2. Run at least three bounded isolated iterations so current-only startup, two-age data, both-age held-out acceptance, cache expiration and continuation are exercised. Preserve production semantics; small smoke batches are labeled diagnostic.
3. Add targeted fixtures long enough to cross step 16 and a later interval, plus rejection and retained-state checks. A two-step smoke cannot validate the g850 cosine peak, two-epoch cap, KL rollback or pooled acceptance.
4. Interrupt during collection/fork join, replay publication, preparation, PPO/PPG and commit boundaries. Resume after a committed iteration and compare the next fixed-data update with uninterrupted execution. Retire only owned caches after commit; partial old-source work is never adopted as new.
5. Verify auxiliary head/optimizer persistence, current-only gate evidence, beta/trips, population/rating slots, game/seed counters, IL lineage and original input checkpoint hashes.

**Exit:** fixed-data update/resume agrees within declared device tolerances, committed progress is not duplicated, incomplete work is rejected and source artifacts remain intact. This proves compatibility, not improved playing strength.

#### B06 — Measure the actual training workload

Run section B5's protocol on actual Linux/GPU hardware with the frozen reference and candidate. Report ordinary main-game/branch-family collection separately from replay preparation, PPO, PPG, auxiliary fitting, ratings and persistence. Tune only a measured bottleneck after baseline parity; retain production job/lane/model schedules and CPU limits for the matched comparison.

**Exit:** predeclared lab throughput/resource/regression criteria pass, with full-workload evidence and unchanged admitted work/recipe. A fast rules microbenchmark or local CPU result cannot stand in for this gate.

#### B07 — Bounded server canary, then explicit adoption

1. Start from a verified copy/fork of a committed lab state in a new isolated canary directory. Pin the accepted local/Linux candidate and current recipe; preserve source run/checkpoints.
2. Choose and record a finite iteration/game limit before starting. Compare collection/replay/optimizer health, rows/probabilities, native/reference routing, branch completion, memory, GPU queues and checkpoint/resume. Ordinary ratings follow the existing recipe; migration does not launch an extra evaluation campaign.
3. Abort and preserve evidence for any rules/encoding/probability mismatch, lost/duplicate family, unexpected optimizer/gate reset, non-finite output, invalid provenance or unrecoverable cancellation. Debug locally where reproducible, then rerun affected local and lab gates.
4. Promote only after the canary passes, at a clean committed boundary through the existing run lock and reviewed source/backend transition. Record the reference checkpoint/source/config and rollback method before adoption.
5. Verify a subsequent commit/resume and bounded rollback/reselection to the qualified reference. Do not merge canary sample histories into production or edit source signatures manually.

**Exit:** the lab uses the locally accepted Rust core under a tested training adapter, with reproducible adoption and rollback. Repeated upstream/core changes use the same local-first gates.


### B5. Matched lab benchmarks and acceptance

Compare the reference collector with Rust using identical frozen learner/opponent policies, source/rules/features/reward, opening/branch settings, game/setup schedule, table mix, lane count and inference/device/thread settings. Include all current staged decisions, dense Canal/Rail games, scored snapshots, search-free V4 tables and branch continuations. First verify deterministic records; then measure paired counterbalanced real-inference collections.

The most recently documented collection baseline is 24 actor processes × 2 lanes, GPU batch cap 64, 1 ms flush, 131072 candidate cells, padding ratio 2.0, model cohorts off and a 72-model cache. **Read the actual adopted execution config at B01**; these historical values are not instructions to overwrite it. Include Python preparation, native worker threads, GPU service, Mono fallback and PL bootstrap processes in the resource inventory.

| Measurement | Evidence to retain |
| --- | --- |
| Fixed work | Identical frozen main-game jobs, eligible/selected branches, admitted rows, optimizer work and rating jobs; timed setup excluded/reported consistently |
| Ordinary throughput | Main games/s, completed families/s, branch continuations/s, decisions/s and stage/table coverage; learner/actor/forced/snapshot counts and native/reference routing |
| Data path | Rules, encoding, branch capture/restore, queue wait, feature packing/transfer, GPU forward, publication, replay validation and preparation; bytes/copies, padding, queue age and memory |
| Update path | Behavior reconstruction, utility calibration, GAE, PPO compute and exact KL probes, PPG two-age preparation/fitting/acceptance/KL, auxiliary fitting and cache/persistence cost |
| Whole iteration | Wall time from identical isolated committed state to next commit, including scheduled fingerprints/ratings/cleanup and enabled teacher work; overlapping sub-times are not additive |
| Correctness and recovery | Row/probability/target parity, no stale/duplicate family, reproducible fixed-data update and resume, bounded failures and unchanged source checkpoint hashes |
| Scheduling effects | Actual game/branch/job/row/update counts, any teacher admission, rating refits and fallback routing; a shorter iteration that performs less work is not an equivalent-work speedup |

Initial lab engineering target: at least **1.25× complete ordinary-game throughput**, including branch-family completion/publication, at the actual reference lane/device recipe; no reproducible median regression above 5% in a declared critical table/phase/branch group. Pin numerical thresholds, memory budgets and measurement protocol after measuring the reference and before seeing candidate results. Report uncertainty and expand inconclusive samples. Whole-iteration benefit needs its own paired evidence; unchanged PPO/PPG/ratings may dominate and no multiplicative end-to-end promise follows from engine speed.

Use the following two modes where scheduling depends on measured time:

| Mode | Fixed inputs | Interpretation |
| --- | --- | --- |
| Fixed-work attribution | Manifest of ordinary/branch jobs, exact admitted training rows, optimizer work, any teacher/evaluation/rating jobs and identical incoming state | Cost of equivalent work; any harness-only scheduling override is recorded |
| Unchanged production recipe | Actual signed scheduler/config with each backend's measured costs | Operational iteration time and work produced; report every admitted/deferred job and update |

For a separately enabled teacher recipe with wall fraction `f`, preserve `max(0, ordinary_seconds * f / (1 - f) - search_seconds)`; faster ordinary collection may admit less teacher work. The current generated recipe uses no such allowance and disables teacher search, so report zero teacher work explicitly. Do not enable it to create a benchmark.

Use separate evidence directories for local CPU and lab GPU measurements. The g850 PL-only timing improvement is historical non-Rust work; keep its warm-start/parallel fitter fixed in both arms and never attribute that speedup to the backend migration. Likewise, preserve current CPU preparation/input caches rather than comparing Rust against an obsolete Python path.

Keep a permanent dense-state/branch/mixed-model regression suite. Changing job/row sampling, branch frequency, PPG cadence, reward, teacher enablement or worker recipe is a separate experiment with its own identity.

### B6. Source admission, cache handling and rollback

A language/backend change needs new immutable producer/source/build identities plus an exact compatible admission route. Preserve historical source approvals, frozen banks, runtime inventories, dataset manifests, IL upgrades and completed reports. No generic “compatible” override, edited saved signature or copied C# hash is sufficient.

For an already-g850 run, preserve weights, PPO/value/auxiliary Adam moments, RNG, generation/game/seed counters, population, ratings/anchors, economic baseline, LR/entropy controllers, GAE/snapshot state, beta/trip history and cache origin. **Do not rerun the g850 adoption or reset beta/cache origin merely because Rust is selected.** A backend-only equivalent target path may rebuild source-bound tensor caches through existing ownership rules, but cannot invent a reward transition.

If a separate reviewed reward change is still required, follow its existing locked adoption path: commit the reviewed signature, retire only identified owned stale reward caches, retain unknown files and reconstruct the appropriate new critic ages. The g851-one-age/g852-two-age schedule in the g850 report describes that original reward adoption; it is not a schedule to restart at an arbitrary later backend cutover.

On failure, preserve evidence and stop admission. Discard incomplete/unpublished native jobs; do not repair their probabilities or mix their partial shards with reference output. Resume/recollect from the last committed state through a validated source transition. The saved reference executable/config/checkpoint gives a reproducible rollback point. Any return to the reference collector must itself satisfy source/replay compatibility; switching a flag does not authorize relabeling native data.

Keep canary directories, reference artifacts and production state separate. Cleanup is limited to owned caches after actual writer/device completion and the relevant commit. A hard kill or lost device is a recovery case, not a reason to delete unknown files.

### B7. Lab risks and completion checklist

| Risk | Required protection |
| --- | --- |
| Local tests miss Linux/GPU behavior | Cross-platform fixtures, real GPU batch/descriptor checks and a bounded isolated server canary |
| Branching changes behavior or doubles samples | Exact alternative receipts, fresh per-seat counters, causal costs and exactly-once whole-family admission |
| Equal weights conceal different policy roles | Sealed policy identities and explicit learner/snapshot/IL/rule-bot admission checks |
| Critic targets become actor inputs | Separate native scoring receipts/training metadata from permitted features and live search descriptors |
| Old PPG behavior survives in the plan or cache | Freeze effective g850/saved recipe, two-age acceptance, current-only gate evidence and exact reward/cache identity |
| Rust adoption resets training or replays a migration | Committed-state transition, all optimizer/RNG/gate preservation, no repeated g850/IL upgrade |
| Higher collection throughput masks changed work | Fixed-job and unchanged-scheduler measurements with branch/row/update/teacher/rating counts |
| A lab fix regresses the accepted local game | Link shared-core changes to a local regression and renew affected local acceptance evidence before promotion |

- [ ] Part 1 acceptance receipt exists and identifies the shared source used on the server.
- [ ] Linux rules/features/search and actual GPU inference agree with the admitted contracts; local two-session settings and lab resource settings are separately recorded.
- [ ] Ordinary policy-only collection preserves all staged/forced records, identities, actual probabilities, opening counters and strict likelihood checks.
- [ ] Branch checkpoint/reshuffle/alternative distribution, post-fork counters, causal budgets, ready-work scheduling and complete-family publication/retry pass.
- [ ] v11.2 descriptors, critic roles, whole-family holdouts, fixed snapshot mode or signed legacy state, and enabled local-scoring receipts pass exact admission.
- [ ] Current teacher/V4-search switches remain disabled where configured; any enabled profile has its own readiness, budget, completed-label and provenance evidence.
- [ ] g850 rewards, raw-T1 opening objective, fixed entropy, PPO KL controls and two-age PPG acceptance/LR/beta/rollback behavior remain exact.
- [ ] Fixed-data updates and multi-iteration resume preserve every optimizer/RNG/gate/counter/population state; partial work and duplicate families are rejected.
- [ ] Completed IL remains unchanged; optional fresh handoff and established checkpoint continuation are separately qualified.
- [ ] Lab collection and whole-iteration comparisons use the actual recipe and hardware, with branch/row/work counts, memory and regression evidence.
- [ ] Isolated server canary passes before production adoption; clean-boundary source transition, failure recovery and rollback are demonstrated.

### B8. Implementation references and existing regression anchors

All paths below are relative to this repository. They are behavior references, not claims that the proposed Rust modules exist. Recheck effective manifests at implementation time.

| Reference | Contract/evidence |
| --- | --- |
| `src/brass_ai/search/ismcts.py`, `src/brass_ai/search/common.py`, `src/brass_ai/search/deadline.py` | Faithful search, leaf/backup/candidates, pending work and newer launch admission |
| `src/brass_ai/searchlab/spiel.py`, `src/brass_ai/beliefs.py`, `src/brass_ai/personality/search/engine.py`, `src/brass_ai/personality/search/contracts.py` | Rules adapter, permitted sampling/recall, whole-turn sessions and plans |
| `src/brass_ai/learning/features.py`, `src/brass_ai/learning/state_layout.py`, `src/brass_ai/learning/model.py`, `src/brass_ai/learning/utility_model.py`, `src/brass_ai/learning/seat_descriptors.py` | Features-4, v11.2 inputs, critic-only descriptors and model identities |
| `src/brass_ai/learning/export.py`, `src/brass_ai/learning/onnx_tools.py`, `src/brass_ai/personality/laptop/inference.py` | Export and inference parity |
| `src/BrassTrainingHost/FastHost.cs`, `docs/R1_HOST.md`, `src/brass_ai/recovery/host_runtime.py` | Authored rules host and runtime provenance |
| `src/BrassLocalAI/Unity/LocalTurnBridge.cs`, `src/BrassLocalAI/Execution/AiExecutionScheduler.cs`, `src/BrassLocalAI/Policy/DifficultyCatalog.cs` | Live validation, resource lifetime and timing/save schemas |
| `docs/AI_EXECUTION_RESOURCES_1.4.7.md`, `docs/V4_SEARCH_TIME_1.4.32.md`, `artifacts/patch-installation.json` | Local lifecycle and installed historical timing baseline; inspect current four-setting work before building |
| `src/brass_ai/recovery/config.py`, `src/brass_ai/recovery/training_revision_g850.py`, `scripts/configure-training-g850.py`, `ai-research/TRAINING_G850_REVISION.md` | Effective generator/reviewed continuation and g850 changes |
| `src/brass_ai/recovery/ordinary.py`, `src/brass_ai/recovery/execution.py`, `src/brass_ai/recovery/exploration.py`, `src/brass_ai/recovery/opening_objective.py` | Frozen jobs, current behavior and objective attribution |
| `src/brass_ai/recovery/branching.py`, `src/brass_ai/recovery/branch_scheduler.py`, `ai-research/SELFPLAY_G210_REVIEW.md` | Trusted forks, fresh counters, asynchronous continuations and family validation |
| `src/brass_ai/recovery/encoding.py`, `src/brass_ai/recovery/inference.py`, `src/brass_ai/recovery/inference_batches.py`, `src/brass_ai/recovery/batch_cache.py`, `src/brass_ai/recovery/ppo_preparation.py` | Native encoder oracle, streaming GPU queues and retained Python performance work |
| `src/brass_ai/recovery/critic_data.py`, `src/brass_ai/recovery/ingress.py`, `src/brass_ai/recovery/replay.py`, `src/brass_ai/learning/reward.py` | Row admission, descriptors, forced/branch exclusions and causal returns |
| `src/brass_ai/recovery/local_scoring_targets.py`, `src/brass_ai/recovery/local_scoring_auxiliary.py` | Authoritative current-era scoring receipts, auxiliary weights and optimizer |
| `src/brass_ai/recovery/updates.py`, `src/brass_ai/recovery/ppo_kl_control.py`, `src/brass_ai/recovery/gae.py`, `src/brass_ai/recovery/value_phase.py`, `src/brass_ai/recovery/ppg_g210.py`, `src/brass_ai/recovery/ppg_replay.py` | PPO/DCPG, exact categorical control, PPG 15, heldouts, schedule, state and cache retirement |
| `src/brass_ai/recovery/plackett_luce.py`, `src/brass_ai/recovery/ratings.py`, `src/brass_ai/recovery/policy_anchor.py`, `src/brass_ai/recovery/v4_share.py` | Unchanged statistical rating/population gates and g850 compute settings |
| `src/brass_ai/recovery/teacher_schedule.py`, `src/brass_ai/recovery/teacher_collection.py`, `src/brass_ai/recovery/parallel.py` | Enabled-only teacher budgets, work admission and replica semantics |
| `src/brass_ai/recovery/resume.py`, `src/brass_ai/recovery/demonstrations.py`, `src/brass_ai/learning/checkpoint.py`, `configs/learning/selfplay-resume-sources.json` | Exact continuation, IL handoff and historical source preservation |
| `docs/MLP_V11_LAB.md`, `docs/TRAINING_RECOVERY_LAB.md`, `ai-research/OPPONENT_CRITIC_DESCRIPTORS_REPORT.md` | Current guide entry points and historical descriptor evidence; later recipes supersede earlier settings |
| `ai-research/SELFPLAY_INPUT_CACHE_REVIEW.md`, `ai-research/SELFPLAY_G412_PPO_REVIEW.md`, `ai-research/SELFPLAY_G470_THROUGHPUT.md` | Existing batching/input/preparation optimizations to retain and measure |
| `docs/UI_PATCH_RESEARCH_COMPATIBILITY.md`, `docs/R0_REFERENCE_NOTICES.md`, `configs/references/upstream.lock.json` | Separate deployed/research contracts and preserved reference provenance |

Extend the relevant current regressions with native backend variants, rather than replacing their assertions: `tests/Recovery/test_training_g850.py`, `test_branch_scheduler.py`, `test_critic_export.py`, `test_opening_ppo.py`, `test_categorical_kl_guard.py`, `test_ppg_g210.py`, `test_selfplay_input_cache_resume.py`, `test_inference_deadline_race.py`, `test_pl_parallel.py` and the existing local Unity/rules suites. Run only relevant suites for each implementation change, then the declared combined release qualification. This document update itself does not execute those implementation tests.

