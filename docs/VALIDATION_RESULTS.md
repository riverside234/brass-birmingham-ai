# Validation results and benchmarks

[Back to the project overview](../README.md)

Updated **26 October 2026**. This page summarizes recorded checks and measurements for the training pipeline and local Rust backend. Each result belongs to its documented source, model, runtime, workload, and machine. Test counts from overlapping suites are not added into a project-wide total.

The current local implementation is release **1.4.36**, using the intermediate **generation-890 (g890)** checkpoint. The final neural model remains in training. Qualification here concerns engineering correctness, compatibility, and bounded local performance; statistical playing-strength superiority and broad hardware support remain unestablished.

## Recorded correctness and integration checks

| Area | Recorded coverage | Source report |
| --- | --- | --- |
| Native rules parity | **546 accepted roots**, **20 complete games / 8,344 transitions**, and **1,406 alternative moves** checked against the frozen C# reference; no unexplained legal/state/reward differences | [Staged Rust rules](../ai-research/RUST_BACKEND_L05_REPORT.md) |
| Hidden-information boundary | All-seat permitted views, own-hand recall, sampled-world conservation, counterfactual hidden-state independence, and failed-operation atomicity | [Observation and sampling parity](../ai-research/RUST_BACKEND_L06_REPORT.md) |
| Native feature encoding | Exact Features-4 layout, complete legal masks/order, stable-seat mapping, staged contexts, cancellation, and unchanged checkpoint inputs | [Native feature parity](../ai-research/RUST_BACKEND_L08_REPORT.md) |
| Neural inference | **11 golden positions**, all five output heads, and mixed batches pass the existing CPU tolerances; maximum absolute output error approximately **4.58 × 10⁻⁵** in the L15 rerun | [CPU inference contract](../ai-research/RUST_BACKEND_L09_REPORT.md), [L15 checks](../ai-research/RUST_BACKEND_L15_REPORT.md) |
| Search semantics | Executable toy reference, full Brass fixed-count comparisons, recorded random draws, first-new information-set leaves, per-seat backups, completed-round snapshots, and pooled recommendation | [Algorithm reference](../ai-research/GUMBEL_IS_MCTS_REFERENCE.md), [Rust search](../ai-research/RUST_BACKEND_L10_REPORT.md), [two-session coordination](../ai-research/RUST_BACKEND_L12_REPORT.md) |
| Unity lifecycle | **818 checks** in the retained smoke suite, plus **45 validation/event** and **65 checkpoint** checks; cancellation, stale identities, action two, save/load, failures, and resource retirement | [Validation optimization](../ai-research/RUST_BACKEND_L14_REPORT.md), [L15 checks](../ai-research/RUST_BACKEND_L15_REPORT.md) |
| Local release and recovery | Actual-search complete games, package tamper rejection, exact rollback, interrupted native/managed publication recovery, installed startup, and **106 existing save files unchanged** | [Local release report](../ai-research/RUST_BACKEND_LOCAL_RELEASE.md) |

Earlier milestones retain their original source identities and qualification status. The final release reuses the corrected L15 runtime binaries; packaging and additional release checks are identified separately. See the [migration plan](../ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md) for the sequence and contracts.

## Training architecture and recovery

The current v11.2 medium network has grouped state projections, a **512-wide / eight-block** residual trunk, local entity conditioning, and separate action and four-seat outcome heads.

| Configuration | Parameters |
| --- | ---: |
| Base / imitation model | 5,910,916 |
| Utility / self-play model, including the local g890 test checkpoint | 5,928,616 |

Opponent descriptors enter critic heads only; deployment defaults to unknown descriptors. Descriptor changes leave policy logits independent of those inputs. Optional local-scoring heads are a separate companion and are not included in these base/utility counts. [Architecture and descriptor report](../ai-research/OPPONENT_CRITIC_DESCRIPTORS_REPORT.md).

Recorded training checks include:

- **149 distinct tests** for critic descriptors, row admission, holdouts, gradients, configuration, ONNX parity, cache recovery, and optimizer restart. One CUDA-only check was skipped on the CPU host; no GPU bitwise-equivalence result is inferred.
- The current opening/PPG revision has **32 passing follow-up checks** and a **24.1-second CPU smoke** over three small authored iterations. Restart reproduced the next update's weights, PPO/value/ownership optimizer state, RNG, beta/trip history, and gate state.
- Focused PPG regressions exercise separate loss and KL cadences, provisional acceptance, retained-checkpoint verification, and rollback to the matching model/optimizer state. Strict replay checks reject relabeled behavior and incompatible partial collection.

The current recipe uses three action decisions of opening exploration per neural seat, fresh opening windows after branch forks, learner-specific PPO admission, and PPG over two critic ages. The detailed [current revision report](../ai-research/TRAINING_G1120_REVISION.md) and [critic report](../ai-research/OPPONENT_CRITIC_DESCRIPTORS_REPORT.md) state the executed configurations. These small recovery tests establish implementation behavior within their fixtures; they do not measure production training throughput or policy improvement.

## Local performance measurements

The measured host is **Windows x64, Intel Core Ultra 5 135U**. Comparisons retain the g890 export, **two independent sessions / one compute thread each**, and **one four-second ordinary-turn allowance** where timing is the control. The deployed provider is **ONNX Runtime 1.22.1 CPUExecutionProvider**, through `ort 2.0.0-rc.10`.

### Unity validation responsiveness

The original path performed three full-action validations per ordinary action. The candidate performs one detached full validation in the background, followed by final identity/state/legality checks and a single main-thread commit. Action two receives a fresh snapshot.

| Live-turn metric | L13 baseline | Background validation |
| --- | ---: | ---: |
| Median worst-frame interval | 238.620 ms | 122.187 ms |
| Median total delivery | 4,028.563 ms | 3,978.806 ms |
| Observed p95 total delivery | 4,176.359 ms | 4,198.471 ms |
| Maximum total delivery | 4,195.984 ms | 4,224.116 ms |

This study contains **80 turns**, 40 per arm, across eight Canal/Rail roots with five paired repetitions and counterbalanced order. Median worst-frame time improves **48.8%**; aggregate process CPU changes by approximately **+0.3%**. Observed tail delivery and some per-fixture CPU/delivery screens regress. The optimization was retained on its measured responsiveness benefit with passing correctness checks; the failed screens remain recorded. This is an initial local sample, not a population-tail guarantee. [Full methods and fixed-reply breakdown](../ai-research/RUST_BACKEND_L14_REPORT.md).

### Allocation and native decision-data reuse

| Measurement | Recorded change | Scope |
| --- | --- | --- |
| Managed validation object bytes | 742,149,300 → 383,083,234; **48.4% fewer** | Eight identical actions per arm, Canal/Rail, two measured repetitions after warm-up, main/worker allocation self-tests, exact shipped Mono and candidate assemblies in an isolated host |
| Native fixed-work time | Median **6.9% faster** | Six roots × two paired repetitions, same completed work and plans |
| Rust allocation traffic | **28.8% fewer bytes**, **36.4% fewer allocation calls** | Separate instrumented run over six cases |

The managed probe counts snapshot construction, detached validation, and guarded commit. It excludes Unity presentation/engine allocations, scheduler bookkeeping, and ORT/native allocation. It enables Mono profiling before managed startup; the earlier callback inside the live player failed its self-test and remains recorded as unavailable. Allocation traffic is not retained memory or a latency measurement. [Managed comparison](../ai-research/RUST_BACKEND_LOCAL_RELEASE.md), [native optimization](../ai-research/RUST_BACKEND_L14_REPORT.md).

One of ten searched fixed-work pairs was 0.2% slower; the other nine improved. Native allocation instrumentation is kept separate from ordinary timing. A separate 20-turn memory screen observed a **313.68 MiB native peak working set** and **17.56 MiB retained growth after retirement**; it is a short screen rather than an uninterrupted-game leak guarantee.

### Interpreting implementation speed

The Rust migration changes both implementation and, for neural deployment, the search algorithm. Trajectory counts from the old Masked ISMCTS and new Gumbel IS-MCTS measure different work and cannot establish a language speedup by themselves.

The reports instead retain controls with matched search semantics, recorded fixed-work traces, C# versus Rust simulator adapters, and external versus in-process inference. These support component attribution within their workloads. The headline percentages above concern specific optimizations; none is a claim that the entire AI is that much faster or stronger. [Comparison design and executed controls](../ai-research/RUST_BACKEND_L14_REPORT.md).

OpenVINO candidates failed correctness admission, and evaluation was stopped with the CPU provider retained. There is no qualified OpenVINO speedup. [Provider decision](../ai-research/RUST_BACKEND_L14_5_REPORT.md).

## Complete local games

The final release uses actual production search and normal **four-second whole-turn allowances**, with no shortened-clock injection. The two families are tested on separate tables; human actions are scripted Pass.

| Game | Turns / actions | Actual AI search turns | Human-End checkpoints | Short execution groups |
| --- | --- | ---: | ---: | --- |
| Human + three NN seats | 64 / 124 | 48 | 16 | Eight segments, 23.8–27.9 seconds each |
| Human + V4 | 40 / 78 | 20 | 20 | Five segments, 17.6–18.1 seconds each |

Both games reach Canal, Rail, and final scoring. All **48 native replies** are retired, within their native deadline, and search-supported. NN plans cover 35 builds, 32 networks, 13 loans, eight developments, and five sales. Neither game entered debt; earlier tests on the unchanged runtime cover 14 combined debt phases and focused liquidation cases.

Trusted test event journals reconstruct successive segments at ordinary-turn boundaries. They are separate from gameplay saves and permitted policy inputs. These runs cover complete-game behavior while keeping each invocation short; they do not establish uninterrupted-game memory behavior or playing strength against competitive human input.

Caller delivery includes initialization, dispatch, validation, and commit. NN median delivery was **4,085 ms**, observed p95 **5,884 ms**, maximum **6,157 ms**, including repeated cold initialization across segments. V4 median was **4,263 ms**, maximum **4,506 ms**. The four-second setting governs the turn's search allowance, not a guaranteed four-second user-visible completion time. [Release methods and limitations](../ai-research/RUST_BACKEND_LOCAL_RELEASE.md).

The g890 utility-calibration receipt is null. Its use for local live search was explicitly accepted for that exact model identity, independently of measured calibration or strength. The final model requires its own admission and relevant parity checks.

## Reproducibility and availability

- **Available in a source clone:** algorithm reference, contracts, regression sources, versioned configurations, Rust sources and lockfiles, selected training artifacts, and written reports. The [README quick start](../README.md#getting-started) exercises the standalone algorithm and contract checks.
- **Require a compatible game runtime:** full C# reference replay, engine-backed collection, Unity integration, and installation verification. Proprietary game assemblies are external dependencies.
- **Generated evidence:** detailed logs, private state fixtures, binaries, profiler output, and sealed run receipts generally live under ignored `artifacts/`. Historical reports link to those local records; such links are not downloadable public benchmark bundles. Written summaries and tracked test sources remain available for review.

Broader CPU/platform support, Linux/GPU collection, statistical strength/noninferiority, calibrated neural difficulty, and admission of the final model require their own results. The current local release does not imply those stages have passed.
