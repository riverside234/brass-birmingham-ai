# Encoder reuse and teacher clocks

Implemented 15 October 2026, on the existing compact grouped v9 target. The six projections remain **32/64/48/192/224/48**, fusion **620 -> 304**, six width-304 residual blocks, and base/utility/personality parameter counts **2,005,917 / 2,015,937 / 2,081,770**. No model, feature, loss or policy distribution change was made.

Validated implementation: `e794a86e13b891d56fcb21a8a7580a8af1c754bc54065a8b636dd5b04f6792f8`. The learning compatibility dictionary is exactly equal to the preceding v9 revision. Evidence is under [encoder-reuse-v9-20261015](../artifacts/feature-engineering/encoder-reuse-v9-20261015/).

## Teacher budgets

| Phase | Existing reservation | Shared seconds |
| --- | --- | ---: |
| Early Canal | Rounds 1-4 | 9 |
| Late Canal | Rounds 5-8 | 12 |
| Early Rail | Rounds 1-6 | 12 |
| Late Rail | Rounds 7-8 | 12 |

Python defaults and configs/step5/joint7-latest.json now agree. Archived clock constants and explicitly configured recipes stay intact; the current config generator applies these new defaults. All workers, both actions and continuations share one absolute turn deadline. The maximum-phase initialization/timeout resolver now returns 12 seconds. Finalization reserve, cancellation, label coverage/quality, model/source freshness and utility calibration guards are unchanged.

Eight workers per teacher game and five games per collection remain **40 private workers** maximum. Other opponent/evaluation settings, cadence 160, eight distillation steps, minibatches/reuse limits and search_wall_fraction=None/null remain unchanged. Four seats times four phase reservations gives **180 summed seconds per game, 900 per collection**, previously 184/920. The 2.17% allocation reduction is not an elapsed-time prediction: games run concurrently and add ordinary play, initialization, encoding/inference and training overhead.

## What was wasting encoding work

The historical [grouped report](GROUPED_PROJECTION_REPORT.md) recorded median new_encoder/persistent_miss/warm_hit times of 8.603/4.722/0.860 ms on its v7 feature contract. Those values demonstrate a reuse opportunity; they are not a current-v9 speed comparison.

RemoteEvaluator constructed a PreparedEncoder for each frozen model on each private reply lane. Policy-cache replacement discarded those encoders. The parallel forest's coordinator and eight replicas separately encoded the same root. Teacher replay looked only for policy.evaluator, while search stores that evaluator in policy.base; replay therefore missed the already encoded root. run_game also constructed a fallback encoder even when a neural evaluator already supplied one.

The revision keeps one encoder per persistent reply endpoint, across model replacements. Each actor's coordinator and private workers share a content-keyed LRU of completed, read-only feature arrays. Mutable categorical/derived scratch state stays private to each lane, and cache locks cover only lookup/publication. Different misses can compute concurrently; simultaneous identical misses can still duplicate computation. There is no global encoding lock or single-flight wait that extends a turn deadline.

The shared bound is 256 entries per reply lane: 256 for an ordinary actor, 2,304 for a nine-channel teacher actor. Frozen models no longer multiply retained copies. All policies still retain their own model identity, exploration/context and request deadlines. The cache contains input features only; it never shares neural predictions across models or remaining utility budgets.

Replay now resolves the search policy's base evaluator, and actor games reuse the lane encoder. Per-game execution records report search.encoding_cache.warm_hit/miss/evictions/retained/capacity once for the whole actor lane. Historical cached collections do not invent missing counters. These detailed execution records are not an additional compact-history field.

## Focused correctness and continuation checks

**41 distinct focused tests passed.** Coverage includes shared-cache identity and lifetime; exact state/action/seat equality against the reference encoder; changed cash, revision, history, seat, candidate order, style and intent; illegal fields and duplicate candidates; concurrent independent misses, LRU eviction and disabled caches; stopped encoding and warm-hit deadlines; teacher replay reuse; real CPU inference with two distinct frozen utility models and different remaining budgets; eight-worker phase deadlines; collection estimates/null cumulative admission; config generation and unchanged evaluation workers; and compatible heuristic dataset loading.

Exact preceding-v9 source receipts admit completed IL handoff and committed self-play continuation with unchanged configuration or only the requested clock change. A regression restores exact weights, Adam moments/steps and RNG, then checks counters, population, anchor, curriculum and checkpoint lineage across the committed transition and restart. Unknown sources, unrelated config changes and pending old-source collections remain rejected. The existing source-change requirement for fresh utility calibration remains active.

The real configure CLI also accepted the previously completed v9 smoke IL checkpoint and generated an isolated config with 9/12/12/12 clocks and 8/40 teacher workers. It did not retrain IL or modify its checkpoint/state. The reference encoder, feature manifest, derived code and learning compatibility are unchanged. The prepared-encoder source hash changes its derived-cache receipt version, so stale disk caches are safely re-encoded from raw data through normal loading. No native data conversion/re-audit is required.

Initial test runs exposed only new test-harness mistakes (a misspelled suite selector, wrong mock argument index, and an empty pending directory that is correctly allowed). They were corrected; the failed cases were rerun successfully. Logs retain those attempts rather than erasing them.

## Small benchmark

Local Windows CPU, Python 3.12.14, PyTorch 2.8.0+cpu, two tensor threads. The same 14 permitted positions from the earlier report span both eras, 1-387 legal choices and action/payment/continuation/debt contexts. Four blocks alternate measurement order. Every result is checked bit-for-bit against the unchanged reference encoder, including candidate ordering and observer mapping. Total benchmark elapsed: **21.33 seconds**.

| Current-v9 encoding path | Samples | Median ms | Empirical p95 ms |
| --- | ---: | ---: | ---: |
| new_encoder | 168 | 18.549 | 44.167 |
| persistent_miss | 168 | 7.567 | 27.655 |
| warm_hit | 168 | 1.570 | 2.823 |

These isolated paths include input hashing/validation and feature building or lookup; no model forward is included. Differences from historical v7 absolute timings are not a regression/speedup estimate because feature contracts and runtime conditions differ.

The controlled ownership comparison runs coordinator encoding, eight concurrent worker-root encodings, then replay encoding. Both arms use the same current feature implementation, inputs, threads and results. The former arm emulates evaluator-local encoders and a separate replay encoder; the current arm uses the production shared-cache/endpoint lifecycle.

| Encoding workload | Former local ownership | Shared actor cache |
| --- | ---: | ---: |
| Root + 8 workers + replay, median ms (56 samples) | 69.223 | **21.426** |
| Same workload, p95 ms | 325.222 | **51.035** |
| Initial encoders for two models over nine lanes, median ms (4 samples) | 221.314 | 86.413 |
| Replace current model on nine lanes, median ms (4 samples) | 50.231 | 3.319 |
| Hits / calls per 14-position block | 0 / 140 | **126 / 140** |
| Misses / retained input entries per block | 140 / 140 | **14 / 14** |

This controlled component workload has **3.23x lower median encoding latency (69.0% reduction)** and 90% warm hits. It intentionally reuses each root across workers and replay; it does not estimate the hit rate of all hypothetical states in real training. All four blocks show the same cache counts. Small construction samples and CPU scheduling variance limit tail-latency interpretation. Model forwards, IPC, native simulations, label yield, aggregate game throughput and playing strength were not benchmarked. No long training or evaluation campaign ran.

Reproduce from the project root in the lab environment:

~~~bash
export PYTHONPATH=src:tests/R2:tests/Recovery:tests/FeatureEngineering:tests/Personality
python -m unittest test_encoder_reuse test_encoder_reuse_resume.EncoderReuseResumeTests \
  test_fe_review.EncodingBudgetTests test_teacher_phase_revision.PhaseClockTests \
  test_teacher_phase_revision.PhaseConfigTests test_time_budgets.TimeTeacherTests \
  test_compact_widths test_latest_model_recipe.RecipeTests.test_defaults_generation_and_migration_are_idempotent -v
python tests/FeatureEngineering/benchmark_encoder_reuse.py \
  --output artifacts/feature-engineering/encoder-reuse-benchmark-lab
~~~

The benchmark rebuilds the same small native fixture corpus when --inputs is omitted. The recorded run reused the existing raw permitted corpus under grouped-v7-20261015/profile/inputs.json. Use a new output directory. See [lab instructions](../docs/FEATURE_ENGINEERING_LAB.md) for fresh config generation and the exact existing-v9 continuation command.

## Changed files

- recovery/config.py and configs/step5/joint7-latest.json: current phase defaults, with archived values retained.
- recovery/encoding.py, inference.py and execution.py: atomic bounded cache, endpoint lifetime, actor sharing/replay selection and per-game counters; feature arithmetic unchanged.
- configs/learning/selfplay-resume-sources.json and imitation-handoff-sources.json: exact preceding-v9 receipts; all prior entries/definitions preserved.
- New test_encoder_reuse.py, test_encoder_reuse_resume.py and benchmark_encoder_reuse.py; current expectations in test_compact_widths.py, test_teacher_phase_revision.py and test_latest_model_recipe.py.
- This report, the active model contract and lab guide; historical reports link to this follow-up without rewriting their measured evidence.
