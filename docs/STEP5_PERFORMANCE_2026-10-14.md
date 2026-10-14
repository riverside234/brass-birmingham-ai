# Step 5 CPU input performance and measurement coverage — 14 October 2026

The [explicit container-check revert](STEP5_CONTAINER_CHECK_REVERT_2026-10-14.md) records the current implementation, accepted list/tuple cache gap, measurements and source approvals. It supersedes the recursive checks from the [subsequent correctness audit](STEP5_PERFORMANCE_AUDIT_2026-10-14.md). The measurements below describe the original optimization.

The implemented changes reduce repeated Python input work. They do **not yet demonstrate faster complete single-GPU training**. Local measurements used Windows, Python 3.12.14 and Torch 2.8.0+cpu, with no CUDA device or lab access. Search clocks and training configurations are unchanged.

The baseline is commit `615210d`, implementation `6446834c236ef81c720c1f166a5f281798c64a2a4c7cba9419382e18531dc4ad`, including the inference deadline-race fix. The new implementation is `c45e7b5338ad4ce4367ec42cbb2723135e1a0f23c302b2ec0aa4e0d4c9717c2c`. Learning compatibility remains `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`.

## Findings and changes

Persistent actors/hosts, shared encoded input transport, frozen-model caching, grouping requests by model, packed inference outputs, overlapped replay ingress, candidate bucketing, private search workers and shared CPU admission already exist. Multi-GPU inference support also exists; this work does not enable it. Raising the batch limit above 64 would not explain or repair the reported mean batch of 2.41. Arrival rate, different model IDs and CPU/native work must be measured first.

A two-game actor profile identified repeated request `deepcopy` and hashing each previously seen history event as avoidable Python work. One instrumented actor spent about 2.43 seconds constructing requests (2.29 seconds under `deepcopy`) and 3.38 seconds in input validation. Native RPC reads were a larger cost, about 8.5 seconds. These are instrumented cumulative timings, not independent CPU percentages or production iteration time.

Implemented:

- **Detached request snapshots:** copy exact built-in wire dictionaries/lists directly, retaining memoized aliases/cycles and `deepcopy` for other types. Immutable JSON scalars avoid generic copy dispatch. Request digests, later-mutation rejection, model/seat/context routing, expiry and cancellation checks remain intact.
- **History validation:** hash public history in chunks of 32. An unchanged chunk reuses its prior validation; changed content anywhere goes through the existing exact per-event validator. The complete frame digest, outer constraints, ordering, privacy checks and bounded caches remain in force.
- **Teacher timing visibility:** each collection reports setup, collection, replay preparation, distillation, ordinary-data recalibration and total time. `search.timings` sums the collections. Console output now includes `teacher=` and `total=`. `teacher=` includes the existing enclosing drift diagnostics; the component report isolates actual collection/update costs. Existing `timings.iteration_seconds` includes evaluation and cleanup, but excludes invocation startup and the final report write itself.

No actor count, batch size, flush delay, CPU budget/reserve, four-worker search allocation, precision, architecture, reward, PPO setting, opponent distribution, cadence, split or search clock was changed. Teacher whole-turn clocks remain **12/12/16/24 seconds** and v4 searched opponents retain **one second per whole turn**.

## Local measurements

Evidence lives under `artifacts/step5-performance-20261014/`. `before.json` and `before/` preserve the starting source/configuration and protected-file hashes. The benchmark driver is `scripts/benchmark-step5-performance.py`; it only accepts a new output directory under `artifacts` and does not resume a training run.

The final CPU microbenchmark warms both implementations, alternates six measurements in before/after/after/before/before/after order, and asserts equal outputs on real native-game observations. It measures the final production functions, not the earlier prototype.

| Workload | Before median | After median | Local wall reduction |
| --- | ---: | ---: | ---: |
| Request input copying, 278 nonforced positions | 0.735 s | 0.280 s | 61.9% |
| Complete input-boundary checks, 619 positions | 1.963 s | 1.777 s | 9.5% |

CPU-time medians agree: copying 0.734 → 0.281 seconds; boundary checks 1.969 → 1.750 seconds. This supports retaining the two CPU optimizations. It does not estimate overall training improvement. See `final-cpu-microbench.json` and `final_microbench.py`.

Complete ordinary-game comparisons use the same seeded small fixture model, two setup seeds, 619 decisions, four tensor threads, one actor/lane and one warm-up game. The benchmark overrides only the two saved input modules for the baseline, in both parent and spawned actors; it never swaps live checkout files. Its `source` field identifies the installed checkout; `baseline_hashes` identify the executed old modules. These are diagnostic runs, not training provenance or trained-lab strength evidence.

| Trial, in execution order | Collection wall | Python actor CPU | Native host CPU |
| --- | ---: | ---: | ---: |
| Baseline pilot | 22.409 s | 6.703 s | 9.203 s |
| Optimized pilot 1 | 23.713 s | 6.609 s | 10.250 s |
| Optimized pilot 2 | 22.646 s | 6.328 s | 9.672 s |
| Optimized, cold replay prepare | 25.100 s | 6.844 s | 10.234 s |
| Baseline, cold replay prepare | 34.375 s | 10.375 s | 14.953 s |

The first pair is slower overall despite lower actor CPU; the reverse-order cold pair is faster, with a large change in native CPU time too. This hybrid Windows machine does not provide a stable overall speed estimate. The large cold-pair difference cannot be attributed entirely to the Python changes.

| Cold component workload | Before | After |
| --- | ---: | ---: |
| Two complete games | 34.375 s | 25.100 s |
| Replay preparation | 3.303 s | 3.564 s |
| Disposable PPO update | 2.807 s | 2.832 s |
| Checkpoint write | 0.041 s | 0.044 s |
| Component total | 40.526 s | 31.540 s |
| Collection games/second | 0.0582 | 0.0797 |
| Component-workload games/second | 0.0494 | 0.0634 |

These totals exclude startup, diagnostics, teacher games/distillation, production opponent distribution, evaluation and cleanup. Preparation uses one reader after collection, unlike production's overlapped ingress. The first three pilot preparations were inadvertently cache-warmed by a parity read; only their collection/CPU results are used above. The final driver measures preparation before parity inspection. Every ordinary trial has the same trajectory digest, `19c79e753f944ab977ada007ea88815e9d981c70b1da466119dd50ced4ad514f`; both cold trials have the same post-PPO model ID, `545407671c76bdb6740e423141f754d43745dcbfc946250caf302cdde57d6171`.

The four-worker teacher probes use the same frozen, calibrated checkpoint and the same permitted replay views. They invoke the normal schedule, CPU admission, search, retirement and label validator. Each row is one root decision; startup, replay validation and host creation are outside its search duration.

| Comparison / phase | Clock | Before wall / completed | After wall / completed | Qualified labels before / after |
| --- | ---: | ---: | ---: | ---: |
| First pair, early Canal | 12 s | 11.875 s / 15 | 12.156 s / 18 | 1 / 0 |
| First pair, late Rail | 24 s | 23.968 s / 24 | 23.953 s / 24 | 1 / 1 |
| Reverse-order repeat, early Canal | 12 s | 12.000 s / 19 | 11.859 s / 21 | 1 / 1 |
| Reverse-order repeat, late Canal | 12 s | 11.906 s / 8 | 11.859 s / 9 | 1 / 1 |
| Reverse-order repeat, early Rail | 16 s | 15.875 s / 8 | 15.844 s / 13 | 1 / 1 |
| Reverse-order repeat, late Rail | 24 s | 23.875 s / 24 | 23.891 s / 32 | 1 / 1 |

The first optimized early-Canal call overran by 0.156 seconds and **correctly exported no teacher label**. The remaining optimized roots had zero reported overrun; 7–8 interrupted paths per root remained distinct from completed backups. Expired inference requests were discarded: six across the baseline probes and ten across the optimized probes. Overall, six of six baseline roots and five of six optimized roots qualified. Increased work in the repeat supports the CPU-path improvements but does not establish a reliable qualification-rate or strength gain. The repeat completed-only rates were 1.58→1.77, 0.67→0.76, 0.50→0.82 and 1.01→1.34 simulations/second by phase; these few timed samples are hardware-sensitive.

The selected frozen balanced v4 opponent was also probed twice at each of the same early-Canal/late-Rail roots, retaining its one-second clock and four workers. Call walls before/after were 1.375/1.343 seconds cold, then approximately 1.015–1.032 seconds. Completion counts were `[4,12,4,4]` before and `[4,10,1,4]` after, with no failures or neural requests. These are native-root controls: neither Python optimization accelerates the frozen engine, and the variable completion counts reinforce the need for lab measurements. Call duration includes startup/finalization outside the native search clock; no clock was extended. Opponent mode exports no teacher labels by design.

The ordinary one-lane fixture produced 278 single-position batches per measured round and zero candidate padding. Its pilot queue-wait totals were 3.28/3.48 seconds (before/after), with host inference-forward totals 0.79/0.82 seconds. Teacher mean batches were roughly 2.1–2.4 positions. This workload does not test the lab's many-actor contention or historical-ID fragmentation. `measurements.json` consolidates phase counts, expiry and throughput; the full reports retain per-model histograms and cumulative service snapshots. Aggregate actor/native/IPC durations overlap and must not be added to infer wall time.

The archived generation-149 lab report separately recorded 275.8 seconds ordinary collection, 5.3 replay tail, 4.3 PPO, 488.2 teacher/search and distillation, 17.7 cleanup and 792.0 total. **It used the older count-based teacher recipe**, so it is only evidence that the omitted teacher component can matter, not a baseline for current clocks. Its aggregate worker phase totals overlap wall time and one another. Generation 170's supplied console fields alone cannot establish its total or teacher cost.

## Correctness and continuation

The focused regression run passed **131 tests** in 35.3 seconds (`final-regressions.log`), covering detached snapshots and mutation rejection, history edits across chunk boundaries, eviction, privacy and seat routing, exact feature/inference results, cancellation/deadline races, completed-only search retirement, teacher clocks/qualification/calibration, timing accounting and compact-history preservation, replay integrity, single-GPU behavior, and guarded source continuation. Timing mocks were updated for the additional observations; training behavior was not changed to satisfy them. This was a targeted run, not a full repository test sweep.

All ten measured ordinary games have exact trajectory parity. The cold before/after PPO checkpoints also have **bit-for-bit equal tensors, Adam state, CPU RNG, calibration and model metadata**, beyond just matching model hashes. `verify_final.py` checks this plus all 1,783 snapshotted files. Only the five intended Python source files and two approval registries differ; protected training assets and installed managed files are unchanged. Reconstructing both original approval prefixes reproduces their starting byte hashes (`verification.json`).

A separate one-game profiling smoke completed and generated parseable CPU inference and PPO traces (`profile-driver-smoke`). Torch emitted `External init callback must run in same thread as registerClient`; the inference trace nevertheless contains four measured forward batches. This short PPO smoke only reached policy measurement in the scheduled trace window, so it is not a backward/optimizer profile. No profiler time is included in the A/B claims, and CUDA capture remains unverified locally. The final driver additionally separates initialization, probe lifetime and time from main entry; earlier local `total_command_seconds` fields start after model initialization, so use the component/collection fields above for comparisons.

All 1,137 prior self-play source approvals and 26 IL handoff approvals are preserved. The 102 explicit transitions into the baseline are carried forward with their exact configuration restrictions; the direct baseline-to-new transition is also recorded, for 1,240 total self-play approvals. The baseline is added to IL handoff eligibility, for 27 entries. Neither an unknown implementation nor an arbitrary configuration change is admitted.

Changing the validator source invalidates old **derived** feature-cache receipts. Readers fall back to authoritative replay validation/encoding; they do not rewrite the source dataset automatically. Repeated reads of old immutable caches can therefore cost extra preparation time. Fresh generated shards use the new receipt. The existing reviewed-source requirement for fresh ordinary utility calibration remains unchanged.

Stop the old trainer after a committed iteration and update source plus both review registries together. Preserve the same run, config and IL origin. If an interrupted iteration remains, follow [the explicit recovery procedure](INFERENCE_DEADLINE_RECOVERY_2026-10-14.md) using the **current committed iteration from `state.json`**, rather than assuming it is still 170. Partial old-source games and teacher labels must not be reused in the upgraded iteration.

The bounded continuation command is:

```bash
conda activate ai-training
brass_root=/data/not_backed_up/yxu209/ai-training
cd "$brass_root"
CUDA_VISIBLE_DEVICES=2 python scripts/step5.py \
  --root "$brass_root" --config configs/step5/joint7-g150-gate72.json --device cuda \
  train --run "$brass_root/artifacts/step5/v4-selfplay-joint" \
  --imitation-run "$brass_root/artifacts/step5/v4-batch7-joint" \
  --resume-reviewed-source --iterations 1
```

This is a real training iteration, for the operator to run after updating; it was **not launched locally**. It restores the committed learner, Adam/RNG, counters, population, champion and IL lineage through normal guarded continuation. The original `--iterations 90` can be used for a later training campaign.

## Missing single-GPU measurements

The highest-value next measurement is a matched lab comparison with actual frozen learner/historical weights and simultaneous per-process CPU plus CUDA traces. Keep one allocated GPU and the current execution settings. Use a fixed checkpoint retained for the entire experiment, with training stopped so retention cannot remove it mid-comparison.

Prepare an isolated baseline copy and select the committed model without editing the run:

```bash
conda activate ai-training
brass_root=/data/not_backed_up/yxu209/ai-training
cd "$brass_root"
probe="artifacts/step5-perf-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$probe/before/learning" "$probe/before/recovery"
git show 615210d:src/brass_ai/learning/batching.py > "$probe/before/learning/batching.py"
git show 615210d:src/brass_ai/recovery/inputs.py > "$probe/before/recovery/inputs.py"
git show 615210d:src/brass_ai/recovery/encoding.py > "$probe/before/recovery/encoding.py"
git show 615210d:src/brass_ai/recovery/replay.py > "$probe/before/recovery/replay.py"
checkpoint=$(python -c 'import json; s=json.load(open("artifacts/step5/v4-selfplay-joint/state.json")); print(s["history"][-1]["path"])')
sha256sum "$checkpoint" > "$probe/checkpoint.sha256"
common=(--root "$brass_root" --config configs/step5/joint7-g150-gate72.json \
  --device cuda --checkpoint "$checkpoint" --actors 24 --lanes 2 --games 48 --rounds 2 --seed 936211)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${common[@]}" \
  --baseline-inputs "$probe/before" --output "$probe/ordinary-before"
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${common[@]}" \
  --output "$probe/ordinary-after"
# Reverse the order into two further new output directories before drawing conclusions.
sha256sum --check "$probe/checkpoint.sha256"
```

The current driver also restores the saved encoder and replay modules so the later correctness repair cannot leak into the baseline. If that Git baseline is unavailable on the lab checkout, copy these four files from this task's `artifacts/step5-performance-20261014/before/src/brass_ai/` into the same `before` layout and verify their hashes against `before.json`. Do not approximate a baseline by disabling validation. See the [subsequent correctness audit](STEP5_PERFORMANCE_AUDIT_2026-10-14.md) for the final implementation and continuation receipts.

For fragmentation, repeat matched comparisons with the same explicit historical checkpoints passed as repeated `--opponent-checkpoint <path>` arguments. The probe cycles the supplied identities across seats; it diagnoses model grouping but does not reproduce production's population distribution. Reports include per-model request/candidate totals and forward batch histograms, queue-wait telemetry, padding, expiry, CPU admission, native CPU/RPC phases and games/second. Ordinary round `before`/`inference` snapshots support per-round deltas. Native child CPU is separate from actor CPU; multi-lane actor process CPU is unavailable in the existing report and needs external sampling.

Use the **same** generated replay and frozen checkpoint for search comparisons:

```bash
replay=$(find "$probe/ordinary-before/round-0" -maxdepth 1 -name '*.jsonl.gz' -print -quit)
search_common=(--root "$brass_root" --config configs/step5/joint7-g150-gate72.json \
  --device cuda --checkpoint "$checkpoint" --replay "$replay" --rounds 2 \
  --phases canal_early canal_late rail_early rail_late --seed 936211)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${search_common[@]}" \
  --mode teacher --baseline-inputs "$probe/before" --output "$probe/teacher-before"
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${search_common[@]}" \
  --mode teacher --output "$probe/teacher-after"
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${search_common[@]}" \
  --mode opponent --baseline-inputs "$probe/before" --output "$probe/opponent-before"
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${search_common[@]}" \
  --mode opponent --output "$probe/opponent-after"
```

These are isolated root decisions, not complete teacher collections. Check completed versus interrupted simulations, qualified labels, fallback and overrun counts at unchanged clocks. Do not demand identical visit counts or accept expired results to improve throughput. Full teacher collection/distillation, evaluation, persistence and cleanup costs require the bounded real iteration above and its new `search.timings` plus enclosing `timings` fields.

Collect instrumentation separately from uninstrumented wall-time comparisons:

```bash
# In another terminal, for this experiment's lifetime (sysstat/pidstat required):
pidstat -h -u -r -d -p ALL 1 > "$probe/process-cpu-io.txt"
# Stop that sampler after the bounded probes; inspect Python actors and Mono children separately.

CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${common[@]}" \
  --profile --output "$probe/profile-ordinary"
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${search_common[@]}" \
  --mode teacher --profile --output "$probe/profile-teacher"
```

Pass the same `probe` directory value to the second terminal. `--profile` produces short CPU/CUDA traces under `profile-inference` and, for ordinary mode, `profile-ppo-*`; profiler overhead disqualifies these runs as timing comparisons. Inspect CUDA kernel and transfer events and CPU queue/packing activity, not asynchronous host-forward timing alone. The short inference window is early in the invocation, so a steady-state CUDA capture may also be needed if startup dominates it. No CUDA timing was obtained locally. Actual medium-model, many-actor contention, historical-model fragmentation, teacher distillation and complete-iteration GPU gains remain unmeasured.
