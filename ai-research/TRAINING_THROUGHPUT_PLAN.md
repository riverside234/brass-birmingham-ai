# CPU and GPU throughput recovery

30 September 2026. The core pipeline is now implemented for Step 5 training, evaluation and R4/R5: phase telemetry, cached preparation, vectorized updates, persistent private CPU actors, shared batched inference and R4 prior/value reuse. Read alongside the [training recovery plan](TRAINING_RECOVERY_PLAN.md) and use [the recovery lab guide](../docs/TRAINING_RECOVERY_LAB.md) for current versioned configurations and commands. This document retains the design and measurement criteria. RTX throughput improvements have not been measured locally.

**Recommendation:** profile the existing phases, remove repeated Python preparation and GPU synchronization, then use persistent CPU game workers feeding a shared batched GPU evaluator. Raising memory limits or switching the search algorithm alone will not resolve the current pipeline.

The target is the user's lab allocation of 64 logical CPUs and an approximately 90 GB RTX GPU. Keep conda environment `ai-training`. All lab artifacts and proposed run directories are under `/data/not_backed_up/yxu209/ai-training/`. Optimize completed legal games, useful learning updates, and search quality per second; GPU memory occupancy is not the success metric.

## What the supplied measurements establish

Source: [results/1.0/report.json](../results/1.0/report.json), with implementation interpretation from [league.py](../src/brass_ai/step5/league.py) and [batching.py](../src/brass_ai/learning/batching.py).

| Measurement | Observed value | Interpretation |
| --- | ---: | --- |
| Reported iteration elapsed | 18,390 s / 5.11 hours | 320 ordinary games and three PPO optimizer steps; measured before the final checkpoint save |
| Collection | 14,577 s / 4.05 hours | 79.27% of iteration time |
| PPO phase | 3,813 s / 63.55 minutes | 20.73%; includes preparation and baseline calculation |
| Search/distillation | Approximately zero | Schedule not due; not the bottleneck in this report |
| Collection throughput | 79.03 games/hour | About 62.64 games/hour including PPO |
| Learner evaluator | 3,677 calls for 77,095 positions | Mean 20.97 positions/call; maximum 32 |
| Learner evaluator time | 3,001 s | Nested within collection, not additional elapsed time |
| Inference padding peak | 13,280 candidate cells | Below the configured limit of 131,072 |
| PPO padding peak | 131,064 candidate cells | Reaches the current memory microbatch limit |
| CUDA peak allocated | 1,310.4 MiB / 1.28 GiB | Large memory headroom |
| CUDA reserved at reporting | 1,720 MiB | Allocator reservation, not active computation |

The evaluator timer averages approximately 0.816 seconds per call, but includes digest/schema checks, feature encoding, collation, transfer, forward computation, and result handling. It is **not CUDA kernel time**. It also excludes some caller preparation and action selection. Learner evaluator time is about 20.59% of collection; the remaining time cannot be assigned entirely to the rules engine without finer timers.

The previously supplied `top` snapshot showed very low overall CPU use, but it is not a synchronized trace of this report. Neither file supplies a GPU SM-utilization timeline, per-core activity, or per-policy timing. The evidence is consistent with a pipeline that fails to feed the hardware; it does not prove a GPU driver problem or establish the exact dominant Python function.

The earlier recovered report achieved about 51.16 games/hour end to end, compared with 62.64 here. That is only about a 22% increase despite much larger collection groups, but different populations and runs make it an indicative comparison rather than a controlled speed benchmark.

## Would the OpenSpiel adapter use the GPU better

**Not automatically, and the current adapter is not a replacement for this performance work.**

The existing [OpenSpiel adapter](../src/brass_ai/searchlab/spiel.py) uses the same C# rules host. Its state operations include serialization/restoration, chance handling, and permitted observations; refreshing a state obtains four seat views inside the trusted harness. Tree traversal remains Python/CPU work. The [R4 evaluator](../src/brass_ai/searchlab/ismcts.py) can also request prior and value predictions separately for a node.

The [R4/R5 executor](../src/brass_ai/searchlab/execution.py) already uses spawned processes, which is a useful pattern for parallel CPU work. Each worker currently loads its own model and CUDA context, and many tree evaluations are individual requests. More search could produce more inference work, but could also add host and synchronization overhead. Increased GPU activity would not itself demonstrate better games per hour or stronger equal-time play.

Reuse the process-isolation and batching ideas across Step 5, S2, R4, and S3. Do not port Step 5 into OpenSpiel merely to obtain GPU utilization. A native replacement simulator would require a separate rules-equivalence project and is outside this first recovery.

## B0 Measure the missing phase boundaries

Add inexpensive aggregate timers and counters before changing concurrency. Use a short sampled profiler window, plus an unprofiled throughput run; profiler overhead must not become the benchmark.

| Area | Required measurements |
| --- | --- |
| Rules host | Startup/hash verification, reset, observation/legal choices, apply, restore, chance, IPC/JSON wait and parsing |
| Python preparation | Schema validation, copying/digests, feature encoding, candidate counts, policy selection |
| GPU evaluation | Queue wait, packing, host-to-device transfer, CUDA forward, result transfer; real and padded candidate cells |
| Collection | Per-table-kind time, current/historical/v3 policy time, active/idle workers, straggler tail, replay writing |
| Learning | Replay read/validation, encoding, baseline/advantage calculation, forward/backward, optimizer, metric synchronization, checkpoint write |
| Search | Host calls, neural requests, expansions, completed plans/replies/confirmations, timeout/fallback causes |

Record mean and p50/p95 latency, peak host/GPU memory, model identity, policy identity, work counts, and worker configuration. Use CPU and CUDA activities with labeled scopes in the [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html); CUDA kernel duration and Python elapsed duration answer different questions.

Verify the training process's CPU affinity and any scheduler/cgroup allocation, not just the machine's `nproc`. Record PyTorch/BLAS thread settings and the actual GPU/PyTorch/CUDA versions without changing the functioning conda environment.

For a short external GPU sample while the existing job is running:

```bash
conda activate ai-training
mkdir -p /data/not_backed_up/yxu209/ai-training/artifacts/performance/recovery
timeout 120s nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 > /data/not_backed_up/yxu209/ai-training/artifacts/performance/recovery/gpu.csv
```

These are monitoring commands, not a proposed new training CLI. Capture CPU/process activity over the same interval using `pidstat` if available. Collect one window during collection and one during PPO; an idle GPU during CPU simulation is different from an idle GPU while a training batch is supposedly executing.

**Exit:** distinguish CPU preparation, host simulation/IPC, GPU computation, and waiting. Preserve a baseline trace and an unprofiled run before selecting optimizations.

## B1 Prepare each decision once

The current [collector](../src/brass_ai/step5/collector.py), [policy](../src/brass_ai/step5/policy.py), [feature encoder](../src/brass_ai/learning/features.py), and [replay code](../src/brass_ai/learning/replay.py) repeat validation, copying, and encoding at several boundaries. [prepared](../src/brass_ai/learning/training.py) also encodes all seats before PPO filters learner rows; terminal targets are derived repeatedly for records from the same game.

Implement the following in profiling priority order:

1. Validate external observations and complete choices once at ingestion into an immutable prepared decision. Carry schema/rules/feature identity, permitted-view digest, and ordered semantic choice IDs with it.
2. Retain cheap freshness, identity, and legality checks when accepting an inference result. Never remove the hidden-information boundary, stale-response check, replay admission, or final live-action validation to gain speed.
3. Cache invariant lookup tables and encode the prepared decision once. Share that immutable encoding between inference and replay preparation when the model context matches.
4. Calculate terminal targets once per game. Filter eligible PPO actor rows before encoding; an explicit separate value-learning pass may still use other admissible seats.
5. Add a bounded encoded-data cache for repeated BC/PPO epochs and search distillation. Key it by dataset hash, feature contract, permitted context, and ordered choices. Store compact arrays with offsets for variable-length choices instead of repeatedly parsing and encoding JSON.
6. Move replay compression/writing behind a bounded queue if profiling shows it matters. Finish and verify replay writes before committing a checkpoint/state transition.

Original replay data remains the audit source. Cache invalidation must reject changed rules, choices, style/intent, or encoders. An optimization must not expose a combined private state to the policy or turn a mutable dictionary into an unsafe shared cache.

**Exit:** parity on representative permitted observations and all choice kinds, unchanged legal decisions under fixed deterministic inference, and a measured reduction in preparation time.

## B2 Make learning batches efficient

The current [PPO update](../src/brass_ai/step5/updates.py) has several concrete optimization candidates: per-row conversion of CUDA values to Python floats while preparing advantages, per-row exploration tensors, and conversion of losses to Python floats in each microbatch. Measure their contribution, then:

- Gather own-seat baseline values and advantages with tensor operations. Copy/log summaries in batches, not once per decision.
- Precompute chosen indices, old behavior probabilities, exploration distributions, and terminal targets. Reuse them through the frozen update.
- Group similar legal-choice counts for memory packing, while shuffling effective optimizer batches and preserving example weights. Do not truncate legal actions to fit a batch.
- Use contiguous buffers and bulk transfers. Evaluate pinned host memory and nonblocking copies where overlap is possible.
- Accumulate loss metrics on device and read them at explicit reporting points. KL guards still require timely measurements, but not one host synchronization for every scalar.

These synchronization and transfer recommendations follow the [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization). Their payoff in this project must be established by the profiler.

Separate the three controls:

| Control | Purpose | Initial experiment |
| --- | --- | --- |
| Effective optimizer batch | Learning statistics and step frequency | BC 1,024; PPO 2,048, as in Plan A |
| Memory microbatch | Fit variable action tensors | 256–512 positions |
| Padded candidate-cell limit | Bound the actual action-scoring tensor | 131,072, then compare 262,144 |

The current cell cap genuinely limits PPO batch packing, but raising it is useful only if backward/packing throughput improves. Estimate the packed dataset's actual memory footprint before considering a GPU-resident cache; 90 GB of VRAM does not imply every full legal-choice replay will fit. Use the large host RAM for a bounded cache first.

After preparation/synchronization fixes, compare FP32 with supported BF16 autocast, keeping sensitive probability/ratio/loss calculations in FP32. Require finite outputs, legal masks, and behavior-probability compatibility. Test compiled/bucketed execution only if kernels or launch overhead remain important; account for compilation warm-up. Do not change model size simply to consume memory.

The same machinery must serve BC, PPO, S2 distillation, and S3 distillation. Algorithm changes in Plan A and pure performance changes need separately labeled comparisons so a faster pipeline is not confused with a better training objective.

**Exit:** higher useful learner decisions per second, correct effective-batch gradients and PPO likelihoods, and no padding-related loss of legal choices or samples.

## B3 Keep CPU game workers and the GPU supplied

### Current scheduling constraints

The Step 5 collector runs host RPCs in threads, but performs substantial Python policy work on the coordinator. Collection proceeds through observation and action barriers. Its policy groups use object identity, which can split requests for identical checkpoint weights into separate small batches.

With 32 live games and one outstanding decision per game, the observed maximum neural batch of 32 is unsurprising despite `batch_size=64`. Raising the configured cap to 128 cannot create requests that are not ready. Historical policy/model instances are recreated across groups, and hosts are restarted/verified for new chunks.

Legacy selection loops through batch requests and maintains timed plans. Increasing v3 tables from 5% to 40% without isolating those plans risks both serialization and elapsed deadlines while other games run. This must be addressed before scaling the curriculum.

### Proposed execution model

Use persistent, game-owning CPU workers feeding a GPU service:

```text
CPU worker processes
  private game hosts, permitted views, contexts, RNG, encoding
                    |
       bounded inference request queues
                    |
GPU evaluator
  immutable model cache, dynamic same-model batches, deadline-aware replies
                    |
CPU workers apply validated choices and continue their games
                    |
completed replay queue -> training preparation -> optimizer updates
```

Start ordinary collection with **8 Python worker processes owning up to 4 live game hosts each**, a single GPU evaluator, and 1 CPU tensor/BLAS thread per actor process. Benchmark 16, 32, then 48 total live games within the actual allocation. Actor processes multiplex independent host waits; each host still has one owner and ordered commands.

Refill a slot when its game finishes instead of waiting for the slowest game in a chunk. Freeze all participating model snapshots for the collection round and update only after that round's trajectories are complete. This improves scheduling without introducing asynchronous stale-policy learning.

For the inference service:

- Group by immutable checkpoint/model and feature contract, not Python object identity. Different style/intent inputs may share a batch when weights match.
- Cache each frozen model once; evict only when no outstanding request needs it. Never mix logits from different snapshots.
- Begin with a maximum of 64 positions and the current candidate cap. Flush on fullness, a short measured queue delay initially around 1–3 ms, or an approaching deadline. Do not wait for a full batch when only a few games are ready.
- Use request IDs, game/revision, seat, context, model identity, and choice digests to route responses. Preserve private per-seat state, independent RNG streams, and the exact probability of the action actually sampled.
- Sample per game using recorded deterministic streams rather than arrival order. Reject stale or mismatched responses; failed training games must not quietly record a fallback under the original policy likelihood.

Search and v3 workers should each own their planner, simulator, and turn budget. Start with eight independent processes for these heavier games. Do not share a mutable legacy planner across games or count a scheduler's long queue wait as if the planner had performed useful search.

Frozen legacy-bank admission hashes cover some shared runner/adapter files. Prefer new versioned orchestration modules or a documented compatibility path that leaves the old snapshot reproducible. If a pinned implementation must change, create and requalify a new identity; never bypass its hashes or rewrite old qualification reports.

Persistent host verification may reuse a launch manifest only while the admitted runtime remains immutable. Check hashes at the trust boundary and reject a changed binary; do not replace integrity checks with a blind cache.

**Exit:** workers remain occupied without shared-game races, batching is no longer fragmented by duplicate model objects, and early high-v3 collections preserve the frozen opponent's intended behavior.

## B4 Apply the changes to evaluation

The current [Step 5 evaluator](../src/brass_ai/step5/evaluation.py) uses batched ordinary neural games, but legacy/search games use separate threaded lanes. Reuse the persistent process workers and GPU service for all three cases, with frozen models, no gradients, no exploration for the candidate, and exact seat/setup quotas.

Use the smaller diagnostic panel in Plan A while tuning; reserve the existing 64-setup full panel for shortlisted checkpoints. Increasing evaluator workers should shorten the same experiment, not silently change opponent counts, partitions, thinking budgets, or promotion requirements.

Record per-lineup throughput and wall time. Ordinary S0 evaluation can batch freely subject to determinism and fairness. A timed search comparison must include queue wait, IPC, encoding, inference, restoration, and final validation inside its whole-turn cap.

Run concurrent timed games as a screen. Preserve serial, controlled equal-budget qualification where required, and do not benchmark a candidate while an unrelated training job saturates the same GPU. Shared batching may improve aggregate tournament throughput while increasing a single game's latency; report both.

**Exit:** identical evaluation population and attribution, unchanged privacy/legal gates, and a measured improvement in evaluation completion time.

## B5 Apply the changes to S2 and R4/R5

The [experiment configuration](../configs/searchlab/experiment.json) currently requests 64 setups, four rotations, and three variants: **768 games**. It uses eight processes, a 3-second turn budget, up to 128 simulations, and four CPU tensor threads per process. R5 distillation requests 64 teacher games with eight workers.

First instrument a small screen. There is no R4/R5 timing evidence in `results/1.0`, so do not attribute an observed speedup or bottleneck to these variants yet.

| Path | Planned improvement | Correctness constraint |
| --- | --- | --- |
| Step 5 S2 teachers | Replace Python-heavy threaded search lanes with private process lanes and shared batched inference | Complete-turn identity, masked opponents, fresh confirmation, and continuation labels |
| R4 OpenSpiel | Cache combined prior/value output for the same permitted model input; batch independent node evaluations when possible | Four-seat backups, explicit chance, restoration, and observer-specific inputs |
| R5 S3 | Batch independent rollout/leaf requests across games or legal parallel work in a tree | Root plan shared across hidden worlds; future branches depend only on allowed observations |
| S3 distillation | Reuse encoded-data and optimized supervised batches | Training-only confirmed teachers, separate dataset/checkpoint lineage |

R4's host state refresh and restore calls may remain expensive even after inference improves. If profiling confirms that, add a versioned host RPC that returns the required permitted observation and complete legal choices together, avoiding redundant round trips. Preserve event ordering, chance semantics, and restoration checks. Do not eliminate observer-specific masking or merge opponents' private views into one neural input.

Within search, cached inference keys must contain the full permitted observation/context, model version, and ordered legal choices. A public-board-only key is insufficient because the acting player's own cards matter. Keep any trusted full-state simulation caches separate from policy inputs.

The service must support search deadlines and cancellation. Stop scheduling work that cannot finish before the reserved validation time. Preserve the current experiment's distinction between parallel screening and serial promotion; faster search can explore more under a fixed deadline, so use fixed-work comparisons for functional regression and equal-wall-time comparisons for playing strength.

After the small screen, retain the existing full comparisons and at least 30 independent setups for promotion. S3 must show a correctness-qualified equal-budget benefit before its additional teacher cost becomes part of routine Step 5 learning. Implementing R4/R5 remains required; promoting either remains evidence-dependent.

## Proposed resource starting points

These starting points are represented by the new strict configuration versions: Step 5 training 2.0.0, evaluation 1.1.0 and searchlab 1.2.0. Use the supplied files and guide rather than adding these fields to an older schema. Compiled execution, pinned-memory overlap and combined host RPCs remain conditional optimizations requiring profiler evidence; no frozen bank/runtime contract is changed for them.

| Workload | CPU scheduling | GPU batching | Learning/search budget |
| --- | --- | --- | --- |
| BC | 4–8 data-preparation workers, bounded cache | 256–512 positions, candidate-aware | Effective batch 1,024; early stopping |
| Ordinary Step 5 collection | 8 processes, initially 32 live hosts total | Maximum 64; prompt partial flush | 320 complete games per frozen round |
| PPO | Cached learner rows, bulk tensor preparation | Microbatch up to 512; compare cell caps | Effective batch 2,048; 2 epochs with KL guard |
| v3 data/curriculum tables | Initially 8 private planner/game lanes | Batch learner seats where present | Freeze bank semantics and account for planner deadlines |
| Step 5 evaluation | Ordinary 32-host pool; initially 8 heavy lanes | Same frozen-model evaluator | Small development panel, then full panel |
| S2/R4/S3 screen or teacher collection | Initially 8 search processes, 1 tensor thread each | Shared same-model request batches | Preserve whole-turn caps and label quality |
| R4/R5 promotion | One controlled match worker | Measure actual deployment/comparison latency | Existing serial equal-budget gate |

Run workload phases separately for the first benchmark; these are not instructions to start all pools simultaneously. Measure process contention before adding workers. The first goal is to use the available 64 logical CPUs productively, not make every process spawn 64 threads.

## Verification and rollout order

1. **Capture B0.** Keep raw results and frozen checkpoints unchanged. Compare equivalent complete-game workloads and warm-up separately; a 32-worker throughput sample must actually supply at least 32 live games.
2. **Implement B1/B2 first.** They are smaller changes with direct relevance to the reported 63.55-minute PPO phase and repeated collection preparation. Check feature/output parity and gradient equivalence on a compact sample, then time an unprofiled pass.
3. **Pilot B3 with a fixed collection population.** Compare 16/32/48 live hosts only while game throughput improves. Match seeds, policies, and work settings; report stragglers and memory, not just an average GPU percentage.
4. **Combine with the Plan A optimizer and curriculum.** Report changed optimizer-step counts separately from speed gains. Validate high-v3 scheduling before the 40% bootstrap phase.
5. **Reuse the pipeline for B4/B5.** Run a short evaluation and search screen, then the existing qualification process for a selected model/variant.

Acceptance requires no information leaks, illegal or stale actions, likelihood mismatch, lost games, or checkpoint/replay corruption. Use the existing focused contract checks and a small real four-player collection; this plan does not require a new large correctness tournament for every performance edit.

Measure games/hour, useful nonpadding candidate evaluations/second, learner updates/second, economic/playing-strength progress per wall-clock hour, and search decisions completed within budget. A twofold end-to-end speedup is a useful engineering target, **not a forecast supported by the current logs**. Keep only optimizations with measured benefit and acceptable latency.

Update [Step 5 lab instructions](../docs/STEP5_LAB_TRAINING.md) and [R4/R5 lab instructions](../docs/R4_R5_SEARCH_LAB.md) when the new code/configs exist, including conda commands, migration/fresh-run rules, telemetry, and measured resource presets. Do not publish runnable commands for configuration fields that have only been proposed.
