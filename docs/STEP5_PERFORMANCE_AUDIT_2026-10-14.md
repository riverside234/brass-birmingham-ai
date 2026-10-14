# Step 5 performance correctness audit — 14 October 2026

**Superseded container-type policy:** the user explicitly accepted the JSON-equivalent list/tuple cache-validation gap and requested removal of this audit's recursive container checks. See the [revert report](STEP5_CONTAINER_CHECK_REVERT_2026-10-14.md) for current behavior, performance and exact source receipts. The evidence below describes the earlier audit; its other correctness regressions remain in place.

One confirmed P2 defect was repaired across validation, feature caching and replay attestation. This audit preserved the pre-existing performance work. It does not establish that the pipeline is bug-free or that production GPU training is faster.

| Finding | Trigger and consequence | Repair |
| --- | --- | --- |
| P2: JSON-equivalent Python containers bypass cached validation | Warm a valid frame, then replace a list with an equal tuple: `digest` produces the same JSON hash although the pinned schema rejects tuples. Frame hits, cached subtrees/history events, feature hits and attested replay reuse could accept that malformed input. The demonstrated consequence is a validation/proof bypass; equivalent tuple values did not produce different numerical features. | `recovery/inputs.py:input_digest` checks container types before canonical hashing. Both paths of `PreparedEncoder.encode` and replay input seals use it. Public hashes and features for valid wire inputs are unchanged. |

The defect existed in the cache path before this audit; it was not caused by the latest detached-snapshot routine. Changed nested contents still require fresh hashes and original schema checks. History chunking, bounded eviction and fast request snapshots remain enabled. No training setting, architecture, precision, reward, PPO parameter, opponent distribution, cadence, worker count or search clock was changed.

Implementation files: [input boundary](../src/brass_ai/recovery/inputs.py), [encoder](../src/brass_ai/recovery/encoding.py), [replay seals](../src/brass_ai/recovery/replay.py). Regression tests: [cache types](../tests/Recovery/test_cache_type_review.py), [device/transport parity](../tests/Recovery/test_performance_audit_devices.py), [exact continuation](../tests/Recovery/test_performance_audit_resume.py). The diagnostic benchmark now loads all four saved input/encoding/replay modules for an isolated baseline; restoring only two would leave part of this repair active in the comparison.

The minimal trigger is `boundary.check(view, choices); view['own_hand'] = tuple(view['own_hand']); boundary.check(view, choices)`. The last call previously returned the cached valid identity and now raises. Four regression methods failed before the fix, with 11 failing assertions/subcases. The repaired tests cover frame hits, frame misses, part/chunk/event caches, nested Scout cards and mat tiles, both encoder ingress paths, and replay seals. See `artifacts/step5-performance-audit-20261014/cache-tests-before.log` and `cache-tests-after.log`.

Across the targeted final test results, **246 tests passed and three CUDA tests were skipped**. These include R2 training/export, personality search/training and P3–P6 shared callers; four/five-head predictions, padding, oversized candidates, mixed models and reply-slot reuse; deterministic deadline races, cancellation, partial submission and worker failures; CPU admission/abort, completed-only backup retirement, teacher qualification/provenance; replay mutation/membership, parallel preparation/PPO parity, checkpoint/resume, and retention dependency/quarantine recovery. Multi-device routing tests use CPU-backed replicas and are not GPU validation. An early source test ran before the new receipt was written and passed on rerun. Six R2 sandbox temporary-directory errors also passed after moving their temporary files into the writable audit directory. No product validation was relaxed for these reruns.

The existing clock boundary is explicit: **atomic CPU admission precedes the search clock**. Admission queue time is measured separately; it is not included in 12/12/16/24 seconds. Once admitted, observation/encoding, inference queueing, all private workers, both actions/subchoices and teacher finalization use the same absolute deadline and existing reserve. This audit preserves that established contract. It does not claim an admission-inclusive latency bound. Native calls and retirement may finish late; late results cannot qualify as teacher evidence. The prior `Request no longer live` repair remains covered by deterministic tests.

Local measurements used Windows, Python 3.12.14, Torch 2.8.0+cpu, four tensor threads and disposable artifacts. The completed optimization was `c45e7b5338ad4ce4367ec42cbb2723135e1a0f23c302b2ec0aa4e0d4c9717c2c`; the earlier pre-optimization reference is commit `615210d`, implementation `6446834c236ef81c720c1f166a5f281798c64a2a4c7cba9419382e18531dc4ad`.

| CPU workload | Completed optimization | Repaired implementation |
| --- | ---: | ---: |
| 619 boundary validations, median of three samples | 1.849 s | 2.273 s |
| 619 cached replay-proof checks, median of three samples | 0.435 s | 0.746 s |
| Two native games, collection | 25.310 s | 28.178 s |
| Replay preparation | 3.497 s | 2.735 s |
| Disposable PPO update | 2.868 s | 2.366 s |
| Checkpoint write | 0.055 s | 0.039 s |
| Sum of those four native-run components | 31.730 s | 33.318 s |

The container check adds real CPU work: boundary validation is about 23% slower than the optimized cache and 10% slower than the pre-optimization boundary (2.063 s). This is an integrity tradeoff, not a speedup. The unchanged fast snapshot still takes 0.548 s versus 0.814 s for `deepcopy` on 278 requests in this run. Microbenchmarks alternate variant order and compare outputs exactly. The native numbers are one matched pair with one warm-up game, one actor/lane, the same small fixture model and seeds; the approximately 5% component-total increase is not a production throughput estimate. Native execution and preparation varied, so individual phase differences should not be attributed entirely to this repair.

Teacher probes used the same frozen checkpoint and replay roots, with four private workers and unchanged clocks:

| Phase / allowance | Before wall / completed | After wall / completed | Qualified before / after |
| --- | ---: | ---: | ---: |
| Early Canal / 12 s | 11.891 s / 15 | 12.234 s / 18 | 1 / 0 |
| Late Canal / 12 s | 11.906 s / 8 | 11.891 s / 8 | 1 / 1 |
| Early Rail / 16 s | 15.875 s / 9 | 15.906 s / 9 | 1 / 1 |
| Late Rail / 24 s | 23.891 s / 32 | 23.906 s / 32 | 1 / 1 |

The repaired early-Canal call overran by 0.234 s and correctly exported no teacher label. Seven or eight interrupted paths per root remained separate from completed simulations; every qualified label's visits sum to its completed count. Both variants discarded two expired inference requests and reported no service error. All qualified labels retain the frozen model and source identities. This small sample demonstrates the rejection gate and bounded search invariants, not a qualification-rate or strength improvement.

Both native variants produced identical 619-decision trajectories. Their resulting checkpoints match byte-for-byte in tensor contents, Adam state, CPU RNG, utility calibration and model metadata. No CUDA RNG restoration was exercised on hardware. `verification.json` records the comparison and checks all 1,921 snapshotted source/config/test/training/managed files: only the three repaired source files, two approval registries and one existing test differ. All frozen training assets and installed managed binaries are unchanged. The original 1,240 self-play approvals and 27 IL approvals reconstruct to their exact starting byte hashes.

Final source identity is `c3e45029ffbbfab1d3a0410d4cd5eb597ff7c3275ede907882132393cdcadcc4`. Learning compatibility remains `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`. There are now 1,344 explicit self-play transitions: the 103 routes into the preceding implementation retain their exact restrictions, and one direct transition is added. IL handoff has 28 eligible historical sources. Unknown transitions, changed learning/configuration contracts and old partial collections/teacher targets remain rejected. Source changes invalidate derived feature-cache receipts; readers revalidate authoritative replay without rewriting frozen datasets.

Stop the old trainer after a committed iteration, update source and both `configs/learning/*-sources.json` registries together, and retain the same run/config/IL origin. The bounded continuation is:

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

This restores the committed learner, Adam/RNG, counters, population, champion and original IL lineage. It retains the existing requirement for fresh ordinary utility calibration after a reviewed source transition. If the old process failed inside an iteration, use the recovery tool first; do not assume the committed generation is still 170:

```bash
generation=$(python -c 'import json; print(json.load(open("artifacts/step5/v4-selfplay-joint/state.json"))["iteration"])')
python scripts/recover-step5-iteration.py --root "$brass_root" \
  --run artifacts/step5/v4-selfplay-joint --expected-iteration "$generation"
# Inspect that plan, then archive the same uncommitted evidence:
python scripts/recover-step5-iteration.py --root "$brass_root" \
  --run artifacts/step5/v4-selfplay-joint --expected-iteration "$generation" --apply
```

The recovery command refuses active or ambiguous publication states and preserves committed checkpoints and progress. See the [recovery procedure](INFERENCE_DEADLINE_RECOVERY_2026-10-14.md) for those guards. These commands were not run against the lab or production run. Later use of `--iterations 90` means 90 additional iterations from that point, not the original campaign's remaining count.

CUDA hardware, Linux multiprocessing, pinned-memory/asynchronous transfers under the real workload, medium-model/many-actor contention, full teacher collection/distillation and production continuation remain unverified. No lab connection was available. Run these exact bounded hardware checks first:

```bash
CUDA_VISIBLE_DEVICES=2 python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:tests/R2:tests/Recovery:tests/Personality \
  python -m unittest -v test_cache_type_review test_performance_audit_devices \
  test_inference_deadline_race test_performance_audit_resume \
  test_selfplay_performance.SelfplayPerformanceTests.test_cuda_ppo_with_prefetched_actor_critic_and_rehearsal
```

The device tests compare four/five-head FP32 predictions, legal padding, seat mapping, oversized candidate chunks, mixed models and shared-slot reuse. They use `rtol=1e-5, atol=1e-6` for device arithmetic, with exact routing/order and unchanged delivered buffers. The same test bodies pass on CPU.

For the actual GPU performance comparison, copy the four pre-audit files (`learning/batching.py`, `recovery/inputs.py`, `recovery/encoding.py`, `recovery/replay.py`) from `artifacts/step5-performance-audit-20261014/before/src/brass_ai/` to that same layout on the lab. Their original hashes are in the audit's `before.json`; the driver records executed baseline hashes in each report. Keep the trainer stopped so the selected checkpoint cannot be removed by retention. Then:

```bash
probe="artifacts/step5-audit-$(date +%Y%m%dT%H%M%S)"
checkpoint=$(python -c 'import json; print(json.load(open("artifacts/step5/v4-selfplay-joint/state.json"))["history"][-1]["path"])')
mkdir -p "$probe"
sha256sum "$checkpoint" > "$probe/checkpoint.sha256"
baseline=artifacts/step5-performance-audit-20261014/before/src/brass_ai
common=(--root "$brass_root" --config configs/step5/joint7-g150-gate72.json \
  --device cuda --checkpoint "$checkpoint" --actors 24 --lanes 2 --games 48 --rounds 2 --seed 936211)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${common[@]}" \
  --baseline-inputs "$baseline" --output "$probe/ordinary-before"
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${common[@]}" \
  --output "$probe/ordinary-after"
replay=$(find "$probe/ordinary-before/round-0" -maxdepth 1 -name '*.jsonl.gz' -print -quit)
search=(--root "$brass_root" --config configs/step5/joint7-g150-gate72.json --device cuda \
  --checkpoint "$checkpoint" --replay "$replay" --rounds 2 --seed 936211 \
  --phases canal_early canal_late rail_early rail_late)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${search[@]}" \
  --mode teacher --baseline-inputs "$baseline" --output "$probe/teacher-before"
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark-step5-performance.py "${search[@]}" \
  --mode teacher --output "$probe/teacher-after"
sha256sum --check "$probe/checkpoint.sha256"
```

Repeat in reverse order into new directories before drawing performance conclusions. These diagnostic games do not reproduce the production opponent distribution; do not adopt them as training data. Teacher simulation counts and moves may differ under unchanged clocks; inspect completed/interrupted counts, legality, qualified labels, expired requests and overruns. Only the bounded real continuation above exercises the complete production iteration. No commit, push, deployment or long training campaign was performed.
