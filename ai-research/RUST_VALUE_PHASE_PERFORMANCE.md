# PPG and auxiliary value-phase performance

Updated 2026-11-08. Review uses results commit
`ebe6084eb2e29486b8e31f9c35fdc6f78529bb49`, the g1280 export. These measurements
predate the revised A10/B20/C10 head schedule. No local benchmark, native game
or full training session was run for this optimization.

## What the lab measured

| Collection G | Entire value phase | PPG fit reported seconds | Critic cache write | Remaining time outside reported PPG fit/cache write | Retained / attempted steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1278 | 44.021s | 20.230s | 7.170s | 16.621s | 0 / 16 |
| 1279 | 67.905s | 35.386s | 16.233s | 16.285s | 16 / 32 |

The old PPG `seconds` ends before the auxiliary callback. The remainder can
include auxiliary work, replay opening/indexing and pool cleanup; the old
reports cannot separate those costs accurately. They do show 4.367/8.743 GB of
CPU-prepared critic outputs, 589/1,178 batches, and roughly 3.6 GB of two-age
dense critic cache. Worker wait is only 0.17/0.33s, so more preparation workers
are not the first fix.

Both phases stop on increased held-out loss. Much of their wall time therefore
does not become retained learning. The auxiliary head itself remained zero:
its old post-PPO policy-staleness filter excluded even current-age rows. That
starvation is fixed by the separate [unconditional-stage revision](RUST_TEACHER_STAGE_SCHEDULE_REVISION.md).
Do not compare an old zero-head-update phase to a new 32-head-step phase as a
matched speed benchmark.

## Implemented changes

1. **Defer full-training diagnostics until PPG retains updates.** The current
   v24/v25 recipe formerly scanned all 235k training rows before finding out
   whether its first interval would be rejected. Loss acceptance uses the
   held-out rows, not that full-training diagnostic. A zero-retained phase
   already issued no utility readiness; it now omits this scan and explicitly
   records training metrics as **unmeasured**, rather than fabricating values.
   For G1278, this removes **235,728 row evaluations**: **461 of 589** prepared
   batches, approximately **78.3%** of its reported critic-collation batches.
   This is a counted work reduction, not a measured wall-clock percentage.

   An accepted phase still measures every training row at both exact weight
   states. Reuse the already-owned frozen post-PPO snapshot for its baseline,
   restore accepted weights in a `finally` block, then measure final weights.
   Neither Adam nor RNG is changed by these measurements. Full readiness,
   held-out acceptance, nonfinite checks and the exact KL cap remain.
2. **Compute only the outputs the call consumes.** Explicit critic-only calls
   skip candidate preparation/scoring and return the unchanged value heads.
   Raw cloning/reference/probe calls skip descriptor/outcome/utility graphs.
   The shared `value_output` default still returns real policy logits because
   PPO likelihood verification also uses it. Custom action-dependent model
   subclasses retain their original full-forward fallback.
3. **Write the same dense critic bytes in bounded blocks.** Reuse the current
   mmap cache, receipts, source checks, layout, publication and recovery. Replace
   individual field writes and five finite-array checks per row with 256-row
   contiguous blocks. Every row still gets source, role, holdout, prefix, seat,
   encoded-layout and descriptor admission. Finite checks run on the destination
   FP32 values, retaining rejection of FP64-to-FP32 overflow. At 124,242 rows,
   five-per-row finite-array calls fall from **621,210 to 2,430**. Output bytes,
   row order, contexts and game IDs are unchanged. Disk volume is unchanged.
4. **Reduce auxiliary preparation and synchronization.** Reuse the existing
   validated `copy_row` helper for numeric target rows instead of recursively
   copying every scalar in the full state vector. Keep exact lambda recurrence
   and independent numeric/nested containers. Stage C packs immutable inputs
   and targets once per pass; sampling consumes the same RNG/index sequence.
   It still recomputes trunk encodings after every optimizer step. Stage A/B
   already cache frozen encoder outputs once per pass; retain that mechanism.
   Return the 32 observed loss scalars with one stacked host read instead of
   32 individual reads; required per-step finite-gradient checks remain.

No optimization changes the losses, sampling probabilities, PPO/PPG acceptance,
beta values, KL guards, optimizer ownership, stage counters, validation cadence,
search_v1 behavior, reward or checkpoint format. The requested policy teacher,
A10/B20/C10 schedule, four-game diagnostic cadence/qualification override,
joint KL cap .003 and PPG cloning-beta floor .25 remain.

## Timing and verification

New reports expose:

- `value_phase.value_work`: cache staging, replay opening/indexing, fit including
  auxiliary callback, and whole value work.
- `value_phase.timings`: clone selection/reference, train-before/after scans,
  acceptance/current/descriptor validation, training/retention, policy diagnostics
  and post-PPG auxiliary callback. Training/retention includes acceptance and KL
  work; these timings overlap and must not be blindly added together.
- `value_phase.wall_seconds_with_auxiliary`: includes the callback; the existing
  `seconds` retains its earlier PPG-only scope.
- `value_phase.train_metrics_measured`, `train_metrics_deferred` and
  `full_train_metric_rows_avoided`: distinguish omitted diagnostics from real
  zero-valued measurements. A zero-retained phase keeps readiness null.
- `value_phase.buffer.serialization`: rows, block/finite-check counts, bounded
  scratch bytes and serialization time.
- `search_value.timings` and `search_value.update.timings`: value reading/target
  preparation, frozen-encoder preparation, head updates, diagnostics and total.

Coarse wall timers reuse existing synchronous boundaries; no extra CUDA
synchronization is introduced just to time a kernel. They do not constitute a
GPU kernel profile.

**30 distinct affected debug regressions passed; one CUDA-only case is skipped
because this local environment has no CUDA.** The new checks compare full/fast
predictions, gradients, Adam and RNG; accepted and rejected PPG transactions;
exact pre/final readiness; scalar/block cache bytes and malformed inputs;
lambda targets and mutation isolation. Existing checks cover full PPO likelihood
verification/update equivalence, stage-C KL rollback, strict checkpoints,
two short synthetic PPG commits/restarts and pending-work/source recovery.
An early regression caught the shared likelihood caller and was corrected by
making the heads-only path explicit; the full PPO equivalence check now passes.

## Next lab run

Sync the code and use a single normal continuation to inspect the new timings,
if g1280 is still the committed boundary:

```bash
python scripts/continue-rust-teacher.py --expected-generation 1280 --iterations 1 --share-target 0.25
```

If the run has advanced, use its actual committed generation. The earlier
`--expected-generation 1280 --iterations 10` request remains the complete A10
continuation; after one commit, repeating it runs only the remaining nine.
No Rust rebuild, archive overlay or repeated successful qualification is needed.
The launcher exports evidence under `results/rust-lab-qualification/` as before.

Actual GPU speedup remains **Pending on lab**. Retain accepted-work counts and
compare cache/diagnostic/optimizer/head costs separately. Accepted phases still
need two full training scans and fresh trunk gradients; dense disk volume,
actual model forward/backward work, cloning and rollback remain possible costs.
Do not relax correctness gates or reduce training data to claim a speedup.
See the [source/debug receipt](../distribution/rust-collector-performance/teacher-value-performance.json).
