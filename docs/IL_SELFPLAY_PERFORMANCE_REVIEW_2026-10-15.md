# IL and self-play debugging and performance review — 15 October 2026

The current v7 IL and self-play paths were reviewed for P0/P1 defects. No P0/P1 defect was reproduced in the reviewed paths. P2 repairs were excluded as requested. Three performance improvements were implemented, with numerical regression coverage and bounded CPU measurements. These measurements do not establish complete-iteration throughput or playing strength.

The review started from the user's existing, uncommitted grouped-projection implementation. That work was preserved. All evidence below is relative to the captured working tree, rather than Git HEAD.

## Changes

- **IL validation:** `recovery/demonstrations.py` and `recovery/updates.py` calculate ordinary and balanced validation reports from one forward pass. Balanced loss uses the same weights and per-microbatch reduction order as the previous second pass. The ordinary report still controls checkpoint selection. Report diagnostics are independent copies; timing and memory fields describe the shared pass.
- **PPO critic and utility calibration:** `recovery/updates.py` and `recovery/utility.py` collate one real legal candidate for stock `FourSeatNet`/`UtilityNet` value-only passes. Their value heads depend on state, so unused candidate encoding, transfer and scoring are avoided. Original encoded shape/version checks run before slicing. Microbatch membership/order, loss scaling, clipping and optimizer steps are unchanged. Actor, BC, rehearsal, teacher and behavior-verification passes retain every legal candidate. Subclasses retain the original full-candidate path because their values may depend on actions.
- **Inference replies:** `recovery/inference.py` retains prediction arrays until shared-slot packing, avoiding an intermediate array-to-list-to-array conversion. The float64 reply wire format and detached lists delivered to consumers remain unchanged. Inline/overflow replies convert once. Existing evaluator overrides and instrumentation retain their original one-argument/list interface. A compatibility problem with that interface was caught and corrected during implementation, with real-service tests for subclasses and instance wrappers.

The same critic transfer reduction applies to CUDA, and IL validation removes a redundant GPU forward pass when running on CUDA. Actual CUDA throughput and numerical parity were not measured: this machine has PyTorch `2.8.0+cpu`. No mixed precision, model architecture, compilation, training setting, or search-clock changes were made.

## Bounded performance comparison

All figures are median elapsed time for the named component on this Windows CPU environment. Smaller is better. Each comparison uses the same fixture, inputs and settings on both sides.

| Component/workload | Before | After | Time reduction |
| --- | ---: | ---: | ---: |
| Paired ordinary + balanced IL validation, 96 rows | 53.792 ms | 27.823 ms | 48.3% |
| Critic update, 16 rows, 2–16 legal choices | 47.963 ms | 41.845 ms | 12.8% |
| Critic update, 16 rows, 17–128 legal choices | 39.321 ms | 33.617 ms | 14.5% |
| Critic update, 16 rows, 130–380 legal choices | 55.669 ms | 43.581 ms | 21.7% |
| Reply construction/transport, 16 × 128, four heads | 0.357 ms | 0.278 ms | 22.2% |
| Reply construction/transport, 16 × 128, five heads | 0.431 ms | 0.344 ms | 20.2% |
| Reply construction/transport, 64 × 512, five heads | 3.384 ms | 1.862 ms | 45.0% |
| Reply construction/transport, 1 × 16, five heads | 0.02327 ms | 0.02389 ms | −2.7% |

There is no demonstrated benefit for the tiny single-position reply case; its absolute difference was 0.000625 ms. The batching gains above must not be applied directly to total training time.

IL used the actual captured pre-change update implementation versus the new implementation, a medium v7 model, two CPU threads, and four ABBA sequences (eight samples per version). All non-telemetry report fields matched exactly.

Critic measurements used re-encoded real native observations, the medium v7 utility model (2,762,529 parameters), four CPU threads, and identical eight-row microbatches. Each workload had two warmup pairs and five measured pairs. Full-candidate versus narrowed-candidate batching preserved exact value outputs, loss, used gradients, updated weights, Adam state and RNG in all 21 pairs. Timing includes collation, forward/backward, gradient clipping and AdamW; it excludes actor, BC, teacher and collection work.

Reply measurements used seven alternating samples with the actual shared-slot pack/unpack implementation and exact received reply checks. They cover host reply construction and transport only, excluding model inference. The final quiet run and the earlier noisier run are both retained in the evidence directory.

## Correctness and regression results

The final focused logs contain **115 distinct tests: 111 passed and four CUDA-only tests skipped**, with no failures. Repeated test IDs across logs are counted once. Coverage includes:

- IL validation metrics, two-epoch checkpoint selection, weights, Adam and RNG; current grouped checkpoint metadata and layout rejection.
- Complete PPO updates and calibration parity; policy candidate preservation; action-dependent subclass fallback; original encoded shape/version validation.
- Four/five-head inference, shared/inline/overflow transport, detached results, finite-value rejection, deadlines, cancellation, routing failures and evaluator overrides.
- Replay mutation/attribution, partial collection recovery, source freshness, teacher readiness, ordinary learner exploration, current recipe and grouped model contracts.
- Exact current-v7 source continuation, completed IL handoff, required review flag, weights/Adam/RNG/population/counters, unchanged checkpoint bytes, retained partial evidence, and rejection of unknown sources, altered settings and incompatible architectures.

Seven test logs and the deduplicated IDs are listed in [test-summary.json](../artifacts/il-selfplay-review-20261015/test-summary.json). The new test modules are `test_il_validation_performance.py`, `test_value_batch_performance.py`, `test_reply_array_performance.py`, and `test_training_performance_resume.py` under `tests/Recovery`.

This was a focused run, not a claim that every historical test passes. Initial exploratory selections encountered an obsolete `test_imitation` assertion for the former 1,840,109-parameter model and three historical resume tests that assumed pre-v7 sources could continue into the current architecture. Those fixtures were left unchanged. Current-v7 regression tests exercise the intended contracts; old architectures were not approved to satisfy obsolete expectations.

A separate native integration smoke used one actor/lane, one warmup game and two measured ordinary games, with an authored small v7 fixture. It completed 632 learner decisions across eight seat trajectories, two PPO steps and three calibration steps, then wrote and loaded a disposable checkpoint. Calibration was ready; model/optimizer tensors were finite; inference reported no errors, expired requests or cancellations. Total command time was 60.25 seconds. This is functional evidence only, with no baseline throughput comparison. Its instrumentation overrides the base evaluator, so the new reply fast path is covered by service regressions rather than this smoke.

## Preserved contracts and accepted limitation

The model, feature encoder, grouping layout and every training/search configuration remain byte-identical to the pre-review snapshot. This includes architecture v7, the 32/80/64/256/256/64 projections, 764→304 fusion, six residual blocks, feature version 2, 5,489/548 input dimensions, FP32, exploration 0.08, entropy/value weights 0.005/0.5, teacher cadence, and phase clocks 12/12/16/24 seconds. Schema validation, content hashes, model/seat/action attribution, finite checks, deadlines, cancellation and previous performance optimizations remain active.

**The known JSON-equivalent list/tuple cache-validation gap remains explicitly accepted.** JSON-equivalent nested list/tuple substitutions can reuse an already validated cache entry because the content identity does not encode that distinction. Uncached schema checks remain active. No recursive container-type scans or replacement full-tree scans were added to close this gap. Its existing acceptance and malformed-uncached-input regressions still pass.

## Exact source and continuation receipts

| Receipt | Digest |
| --- | --- |
| Captured previous implementation | `34a62056d352342da5d33662b34e4c35eeb11146c5d1a5be4e6ee3cc6a7141f7` |
| Reviewed implementation | `a882234d3003a052ccb12e9a489469e772d78ec054f24a71adfeaaea9bcc6cfc` |
| Unchanged learning compatibility | `565dff0e8ff0973214ee7c85b43c81dabc9ef6b6409aaa2e0becdd2c122c4ecf` |

`selfplay-resume-sources.json` preserves all 2,138 previous transitions and appends exactly one previous-v7→current transition, with no configuration migration. `imitation-handoff-sources.json` preserves all 36 previous origins and appends the completed previous-v7 IL origin. Existing definitions and approval prefixes were checked exactly. Older feature/model architectures remain incompatible.

A self-play run from that exact previous source can use the existing `--resume-reviewed-source` mechanism with its original run/config/IL lineage. It preserves committed training state; normal freshness, partial-evidence and utility-calibration rules still apply. Completed compatible IL can be handed off under the added receipt. This does not add cross-source resumption for an unfinished IL run: its existing exact training-signature requirement remains in force.

The final scope check found only four production source changes and two registry changes among 1,919 captured existing files; the other 1,913 were identical and none were missing. Existing tests were not edited. New tests, this report and disposable review artifacts were added. No production data/checkpoints were changed, no long training was launched, and no commit or push was made.

Evidence: [scope verification](../artifacts/il-selfplay-review-20261015/scope-verification.json), [source receipts](../artifacts/il-selfplay-review-20261015/source-review.json), [IL measurements](../artifacts/il-selfplay-review-20261015/il-validation-performance.json), [critic measurements](../artifacts/il-selfplay-review-20261015/value-batch-performance.json), [reply measurements](../artifacts/il-selfplay-review-20261015/reply-array-performance.json), and [native smoke](../artifacts/il-selfplay-review-20261015/native-current/report.json). Reproduction scripts and the pre-change source snapshot are in the same artifact directory.
