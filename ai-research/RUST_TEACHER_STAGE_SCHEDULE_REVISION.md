# Unconditional auxiliary-head stages and sparse validation

> **After server transfer:** apply the user's latest exact reward: win 1.0, **placement 0.05**, VP .001/cap180 paid at canal end and at final end by difference, insolvency charge 0. Source training keeps its old reward. Prepare/reuse the separate zero-cost Rust teacher adapter; preserve Adam/RNG/progress/beta, invalidate old reward targets/readiness and rebuild buffers before auxiliary progression. [Commands and details](../docs/STEP5_MACHINE_TRANSFER.md).


> **g1284 revision (8 November 2026):** Stage C now reserves eight whole buffered games for interval loss retention in addition to its hard 0.003 KL cap. Machine transfer omits replay buffers and original IL data/caches, retains the fixed 2048-position probe, and pauses auxiliary training/stage counts until eight fresh return ages are rebuilt. See [the revision](RUST_STAGE_C_TRANSFER_REVISION.md) and [transfer commands](../docs/STEP5_MACHINE_TRANSFER.md). This supersedes older Stage-C no-loss-stopping and transfer-buffer statements below.


The later [value-phase performance follow-up](RUST_VALUE_PHASE_PERFORMANCE.md)
preserves this schedule and its learning behavior. It reduces diagnostic/cache/
copy work and adds phase attribution; a single normal lab iteration can record
the new timings without repeating qualification.

Updated 2026-11-08 following the user's stage/cadence revision. This supersedes
the gated auxiliary lifecycle in the original B04.5/B05 plans for the reviewed
production recipe. Historical qualification evidence and legacy configs remain
unchanged.

## Review of results ebe6084

Pulled inside `results/` to commit
`ebe6084eb2e29486b8e31f9c35fdc6f78529bb49`. The
[reviewed g1280 export](../results/joint7-v11-2-g1290-trained-restart-g1280-20261108T173525364834Z/state.json)
committed **g1280**, checkpoint SHA-256
`0eeb499da77bebe32508731add936eb9f286ded8643547a922fc52625b5e7fcd`, model
`c37d4c61160739d7414ea284ea35c17ed8565c0e3c62c4acf5e37d0126148068`.
Teacher policy mode is on and its qualified beta remains **0.16110139609854138**.

The auxiliary head remains Stage A, blend zero and trunk scale zero. Its 19
recorded gate observations have zero correction improvement. Both latest reports
(collection G1278 and G1279) show **no auxiliary update**, zero correction output
and **all eight value ages rejected**, including current age zero. At G1279,
current-label drift 0.010939 exceeds search improvement 0.003861. Applying a
policy-label staleness gate after PPO therefore starved return-head training.
The revised head path removes that gate and starts fitting admitted return rows
directly. The policy distillation term retains its own existing staleness rule.

The policy teacher is active: weighted gradient-share medians are 0.154631 and
0.148541, with no outside-share warning; encoder cosine medians are 0.056439 and
0.103163. All 28+4 tables were fresh in both latest iterations. The g1279 root
label yield is 511/512. Held-out teacher KL rises during both updates; monitor
future scheduled observations and ordinary PPO health, without interpreting the
fixed-beta qualification as a strength result. Beta remains fixed.

Latest ordinary collection takes 206.46/204.44 seconds; teacher collection
122.80/123.72 seconds; checkpoint publication 21.29/20.92 seconds. These are
existing lab measurements, not a matched benchmark. Removing unused duplicate
head policy-label history and its repeated staleness evaluations reduces work;
actual new end-to-end timing remains to be measured during normal lab training.

## Revised stages

| Stage | Duration | Head training | Search blend | Trunk gradient | Advancement |
| --- | --- | --- | --- | --- | --- |
| A | 10 committed on-mode iterations | 32 head-only Adam steps each iteration with admitted rows | 0 | 0 | Directly to B after iteration 10 |
| B | 20 committed on-mode iterations | Continue head-only training | 0.05, 0.10, …, 1.00 for its 20 collections | 0 | Directly to C after iteration 20 |
| C | Warm up over 10 iterations, then stay at full scale | Head plus shared trunk through existing PPG optimizer | 1 | 0.1, 0.2, …, 1.0 | Unconditional warm-up; no validation qualification or stage rollback |

Stage A/B have **no auxiliary validation gates**. There is no eight-consecutive-validation
requirement or ordinary-health qualification for stage transitions, and
**no 256-game head-to-head stage**. Stage A/B fit return targets from the eight
ages of independently admitted, reward-bound value shards. Old policy-label
views are no longer copied into the auxiliary checkpoint; the original teacher
policy buffer and its validation remain separate. Existing lambda is preserved;
if it has not been selected, use final-return lambda 1 without a validation gate.

Stage C retains the **hard joint KL cap 0.003**, measured against the original
pre-PPG policy using the existing ordinary probe. As explicitly confirmed by the
user, an over-cap 16-step trunk interval restores its weights and both Adam
states, retaining any preceding accepted interval. The g1284 revision also reserves
eight whole buffered games and rejects an interval whose held-out MSE increases.
This restores only that interval, without rolling back the stage. A spent KL budget or missing PPG
probe defers the trunk update; it does not hold the stage warm-up counter.
Exceptions/cancellation retain transactional restoration for recovery.

The normal PPG recipe is versioned to `ppg-value-phase/25.0.0`, with beta floor
**0.25**. Adoption raises a saved beta only if it is below that floor; it preserves
larger values, trip history, both Adams, replay, cadence and RNG. This cloning
beta is separate from the fixed policy-teacher beta 0.16110139609854138.

The old Stage-A gate history does not count as ten iterations of the new
schedule. At the g1280 adoption boundary, the revised A counter initialized to zero
while preserving all existing head weights/Adam and value shards:

- **g1280→g1290:** ten A iterations, then enter B.
- **g1290→g1310:** twenty B collections, blend 0.05→1.0, then enter C.
- **g1310→g1320:** ten C iterations, trunk scale 0.1→1.0.

Counters commit with the checkpoint. A crash/retry does not double-advance them.
Legacy B progress is preserved on adoption; an old head-to-head state goes
directly to C. Nonfinite values, changed shards/checkpoints, invalid identities,
stale replies and unknown source/config edits continue to fail their existing
correctness checks.

## Four-game diagnostic cadence

The 28 train tables remain every iteration. **Create and run four held-out
tables only when the next committed generation is a multiple of 10**, so from
g1280 the first diagnostic is collected at G1289 and committed at g1290.
Other iterations create no held-out games or empty held-out shards. Preserve the
independent reserved setup-seed stream so train table seeds do not shift.

Scheduled diagnostics record current-generation teacher KL/top-1 and auxiliary
main/corrected MSE on the four complete games. They never decide stage progress.
Old validation rows are never reported as a new measurement. Ordinary PPO/PPG
critic holdouts and KL probes retain their existing schedules and purpose.

The new config exposes:

```json
"teacher_phase": {
  "search_value": {
    "schedule": {
      "contract": "search-value-stage-schedule/2.0.0",
      "stage_a_iterations": 10,
      "stage_b_iterations": 20,
      "stage_c_iterations": 10,
      "validation_interval": 10,
      "validation_during_beta_qualification": true
    }
  }
}
```

With this switch true, a shadow run still collecting its required beta
qualification observations runs all four held-out games **every iteration**.
Once its sealed qualification passes, the ten-iteration cadence resumes.
Changing the switch to false explicitly applies sparse validation even during
qualification; the existing requirement for four full 28+4 observations is
unchanged, so incomplete rounds do not qualify beta. The current run is already
qualified and does not repeat those measurements.

## Lab continuation and verification

The [reviewed config](../configs/step5/joint7-v11-2-g1290-trained-restart-rust-teacher-on-stages-v2-share025.json)
keeps teacher on, branching zero, search_v1, V205/DOMiNO/AZ_DB off and the current
optimized Rust collection/inference/CPU limits. No Rust source or binary changes
are required. Sync Git, exit the older trainer and run from the lab code root:

```bash
python scripts/continue-rust-teacher.py --expected-generation 1280 --iterations 10 --share-target 0.25
```

The helper defaults to the revised recipe and writes its config under the run
lock. It verifies the existing checkpoint/source/qualification, uses the normal
reviewed transition and preserves Adam/RNG/population/PPG state. Because future
collection changes, incompatible unfinished work is preserved in the existing
owned `configuration-history/` archive; unfinished next checkpoints or published
teacher shards still require normal recovery first. Repeating the same command
only completes the remaining requested iterations. After g1290, use the actual
current generation to request the next continuation.

No adapter rebuild, new ZIP or repeat of passed qualification is required. The
helper exports success/failure evidence to `results/rust-lab-qualification/`.
Observe actual auxiliary `update.steps`, committed A/B counters, collection
blend, training scale, scheduled-only diagnostics, retained joint KL and beta
floor in the new reports. The first real revised iteration remains Pending on
the lab. Focused local checks use tiny models/authored data only; no benchmarks,
native games or full training sessions are run here. See the
[debug/source receipt](../distribution/rust-collector-performance/teacher-stage-schedule.json).

**47 distinct affected debug regressions passed**: 19 revised-schedule checks
and 28 existing activation/PPG/checkpoint/inference checks. Coverage includes
actual direct head updates, A10/B20/C10 counters and first warm-up steps, sparse
validation and qualification override, strict checkpoint/Adam/RNG recovery,
lossless pending-work archive, PPG floor adoption without resetting larger beta,
the public launcher, and rejection/restoration of only an over-cap trunk interval.
