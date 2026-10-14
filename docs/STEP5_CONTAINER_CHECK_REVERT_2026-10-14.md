# Step 5 recursive container-check revert — 14 October 2026

At the user's explicit request, remove only the recursive Python container checks introduced by the [latest correctness audit](STEP5_PERFORMANCE_AUDIT_2026-10-14.md). `InputBoundary.check`, both encoder ingress paths, replay row keys and replay seals again use the previous canonical-content hash paths. `_wire_containers` and its `input_digest` wrapper are removed. No replacement full-tree scan is added. The previous constant-time history-list check is restored at its original schema-validation location.

The only production source changes are [inputs.py](../src/brass_ai/recovery/inputs.py), [encoding.py](../src/brass_ai/recovery/encoding.py) and [replay.py](../src/brass_ai/recovery/replay.py). Their executable behavior matches the pre-audit optimized implementation; the input-boundary module also documents the accepted limitation. Detached snapshots, history chunks, subtree caches, shared inference transport, model caching, batching, replay preparation and component timing remain unchanged.

## Accepted limitation

Canonical JSON hashes intentionally do not distinguish a list from an equal tuple. For example, after `boundary.check(view, choices)`, replacing `view['own_hand']` with `tuple(view['own_hand'])` can reuse the cached identity. Equivalent nested mutations can reuse part, history-chunk or event validations, encoder features and replay input seals. A frame miss alone does not guarantee rejection: a matching subtree/event can already be cached, including from earlier items in that same frame. This is an explicitly accepted validation gap, not a claim that all malformed Python containers are rejected.

The pinned schema validators are unchanged and reject these tuples when their corresponding checks actually run. Cache-disabled regression tests verify that rejection. Mutations that change canonical JSON content still change the hash. Existing tests retain coverage of illegal/unknown fields, private hands, stale content, model/seat/action attribution, replay labels/context/membership, finite values, cancellation and deadlines. Only tuple-strictness expectations and the hash-wrapper instrumentation were adjusted; the accepted gap is now exercised explicitly.

No model architecture, precision, reward, PPO parameter, data format, opponent mixture, cadence, process count or training configuration changed. Teacher clocks remain **12/12/16/24 seconds**, with their existing absolute deadline starting after atomic CPU admission. Opponent search remains **one second per whole turn with four workers**. Admission, late-result retirement and teacher qualification are unchanged.

## Verification and bounded measurements

Results are recorded in `artifacts/step5-container-performance-20261014/`. The before snapshot is the actual audited working tree, not Git HEAD. Benchmarks use saved baseline modules within isolated processes; the checkout is never swapped. Outputs and disposable checkpoints are confined to new artifact directories.

Focused regressions: **149 passed, three CUDA-only tests skipped** (152 unique tests). Coverage includes the accepted container gap and uncached schema rejection; changed-content/feature parity and snapshot detachment; replay attribution/membership and parallel PPO preparation; deadline races, cancellation, worker retirement and shared clocks; four/five-head CPU inference; exact source continuation and preserved approval prefixes; R2 training/export and shared personality-search callers. Other correctness tests were retained unchanged. Logs: `fast-path-tests.log`, `regressions-core.log`, `regressions-shared.log`.

Local environment: Windows, Python 3.12.14, Torch 2.8.0+cpu. Component probes reuse 619 decisions from the same two saved native games, with one unmeasured trial per version followed by five interleaved samples each. The before/after order is A/B/B/A/A/B/B/A/A/B. Every trial checks identical hashes or feature bytes. Median wall seconds:

| Workload | Audited recursive checks | Restored fast path | Time reduction |
| --- | ---: | ---: | ---: |
| Input boundary, initially empty caches | 2.151 | 1.814 | 15.7% |
| Prevalidated encoding, default 256-entry feature cache | 1.205 | 0.912 | 24.3% |
| Cached replay seal verification and feature reuse | 0.741 | 0.443 | 40.2% |

The encoding probe iterates more positions than its default cache holds, so its measurement includes feature-cache eviction and recomputation; it is not a pure cache-hit benchmark. Median process CPU seconds respectively changed from 2.125 to 1.781, 1.188 to 0.875, and 0.719 to 0.422. The script and all samples are in `microbench.py` / `microbench.json`.

The native A/B/B/A comparison used two measured games per invocation, one warmup game, a seeded small fixture model, one actor/host lane, four Torch threads and matching learning/search settings in both versions. Production config files were left unchanged. It ran eight measured games plus four warmups and wrote only disposable PPO checkpoints. Two samples per version:

| Workload | Before median seconds | After median seconds | Time reduction |
| --- | ---: | ---: | ---: |
| Two-game collection | 22.901 | 20.606 | 10.0% |
| Cold replay preparation | 2.748 | 2.076 | 24.4% |
| Collection + prepare + PPO + checkpoint | 27.771 | 24.732 | 10.9% |

Collection ranges were 22.889–22.914 seconds before and 20.390–20.822 after. Component-total ranges were 27.632–27.909 and 24.497–24.966 seconds. PPO medians were 2.082 and 2.020 seconds, while checkpoint medians were 0.040 and 0.029 seconds; those small differences are not attributed to the reverted input checks. All four probes produced the same 619-decision trajectory digest, `19c79e753f944ab977ada007ea88815e9d981c70b1da466119dd50ced4ad514f`.

These are bounded CPU fixture measurements, not evidence of complete production-iteration or lab-GPU speedups. The unchanged public driver is [benchmark-step5-performance.py](../scripts/benchmark-step5-performance.py); `run_comparison.py` and the saved `*-command.json` files record the exact invocation sequence. Before reports record the installed source identity plus the four executed baseline module hashes; the overridden-baseline teacher labels are diagnostic, not production attestations.

Teacher probes used the same frozen fixture checkpoint and one saved root in each phase, with four workers and the unchanged clocks:

| Phase / allocated seconds | Before wall seconds | After wall seconds | Completed simulations, before / after |
| --- | ---: | ---: | ---: |
| Canal early / 12 | 11.860 | 11.844 | 21 / 21 |
| Canal late / 12 | 11.875 | 11.859 | 10 / 10 |
| Rail early / 16 | 15.860 | 15.844 | 16 / 16 |
| Rail late / 24 | 23.875 | 24.031 | 33 / 33 |

All eight decisions were legal and inference reported no errors. Seven produced qualified labels with visit totals equal to completed simulations and the correct frozen-model attribution. The restored rail-late probe retired **31 ms after its deadline and correctly produced no teacher label**. Attempted simulations equaled completed plus interrupted in every probe; expired inference requests were four before and six after. This confirms the retained rejection of late evidence, not a hard bound on native-call retirement or an improvement in teacher search strength. The clock starts after CPU admission, as before.

[Final verification](../artifacts/step5-container-performance-20261014/verification.json) confirms byte-identical weights, Adam state, CPU RNG, calibration and checkpoint metadata across all four disposable PPO outputs, as well as exact trajectories. It compares **1,928 baseline files**, finding only the intended three source files, two registries, two existing tests and three existing documentation files changed. All 1,365 frozen training files, 170 managed binaries, other configs/source/tests and the benchmark driver remain byte-identical. The new report, continuation test and diagnostic artifacts are additive. The three restored modules match the pre-audit executable syntax exactly, excluding module docstrings; approval prefixes reconstruct the original registry bytes exactly, and all 104 carried routes preserve their restrictions. A Windows path-separator mismatch in the diagnostic verifier was corrected before its successful final run; the initial verifier log is retained.

The task did not modify production data or checkpoints, commit, push or launch long training. CUDA tests and performance remain unverified on this CPU-only host.

## Exact continuation receipts

Previous audited source: `c3e45029ffbbfab1d3a0410d4cd5eb597ff7c3275ede907882132393cdcadcc4`.

Current source: `9b7dcf9e1ea59602bca328a2c094a12944995265d248d9856760293cef4d4099`.

Unchanged learning compatibility: `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`.

The [self-play registry](../configs/learning/selfplay-resume-sources.json) retains all **1,344** previous approvals, carries forward the **104** routes into the audited source with their exact restrictions, and adds the direct audited-source transition: **1,449** explicit approvals total. The [IL registry](../configs/learning/imitation-handoff-sources.json) retains all **28** earlier entries and adds the audited source: **29** eligible historical sources. No implicit approval, unknown source, configuration bypass or incompatible partial collection is admitted.

Update these three source files and both registries together with the rest of the existing performance work. Same-run continuation uses the original run/config/IL origin plus `--resume-reviewed-source`; it preserves committed weights, Adam/RNG, population, champion, counters and lineage. Source changes invalidate derived feature-cache receipts and require the existing ordinary-data utility recalibration. Checkpoint formats and learning compatibility are unchanged; no production checkpoint or dataset was rewritten by this task.

For a bounded lab continuation after updating the source, use the original command with one iteration for verification (this was **not run** locally):

```bash
conda activate ai-training
brass_root=/data/not_backed_up/yxu209/ai-training
cd "$brass_root"
CUDA_VISIBLE_DEVICES=2 python scripts/step5.py \
  --root "$brass_root" \
  --config configs/step5/joint7-g150-gate72.json \
  --device cuda train \
  --run "$brass_root/artifacts/step5/v4-selfplay-joint" \
  --imitation-run "$brass_root/artifacts/step5/v4-batch7-joint" \
  --resume-reviewed-source --iterations 1
```

An unfinished old-source iteration remains subject to the existing [interrupted-iteration recovery procedure](INFERENCE_DEADLINE_RECOVERY_2026-10-14.md); the new receipts do not silently reuse it. No commit, push or long training campaign was performed.
