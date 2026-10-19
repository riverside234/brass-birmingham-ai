# PPG safe progress and v4 variant rotation

Date: 2026-10-18. Evidence: [generation-one report](../results/joint7-v11-2-grouped-lr-critic-g0001-20261018T215854542444Z/iteration-0000/report.json).

## Why value fitting accepted zero steps

The first phase attempted 16 value steps, with base LR **6e-6**, clone beta **1**, 512 clone presentations per step, and 82,410 eligible critic rows. Its raw categorical policy KL, relative to the frozen post-PPO policy on 2,048 positions, reached **0.00214711**: 7.36% over **0.002**. The separate value optimizer has no PPO warmup; the reported final PPO LR was 2.618e-6. Value gradients update the shared trunk, so they can move the policy despite the cloning penalty. A finite cloning penalty is not a hard bound on that movement.

The old transaction discarded the **whole epoch**, restoring both weights and value Adam. This is why 16 attempted steps became zero accepted steps. The attempted held-out loss improved from **1.48559 to 1.44860**, but that measurement was taken at the over-cap weights. The exported log does not contain the earlier per-step held-out losses or weights, so it cannot establish how many of those steps would have passed the new acceptance check.

## Implemented PPG adjustment

- Keep the **0.002** cap, value LR, beta schedule, clone samples, three-epoch limit, and two-iteration buffer unchanged.
- Save one pre-step model/value-Adam snapshot. If a step exceeds the cumulative cap, restore that step and stop further value updates.
- Evaluate the preceding safe prefix on the original held-out games. Retain it only when loss does not increase relative to the last accepted state. Otherwise restore the epoch origin. A first-step violation accepts nothing; previously accepted epochs remain.
- The KL reference stays fixed at post-PPO for the entire phase. It is not reset after each accepted step.
- Rejected-step value/clone presentations are attempted work only. Reports now include `acceptance=kl-safe-prefix/1.0.0`, `kl_cap`, `partial_epochs`, `completed_epochs`, per-epoch safe/accepted steps and attempted/retained KL. `last_attempted_heldout` now measures the safe prefix; an explicit scope field records this.
- PPO Adam, warmup/LR controller and actor samples are untouched. The value optimizer's moments and step counters match the retained weights. Sampling RNG remains deterministically consumed as before. KL trips still feed clone-beta adjustment and block the existing healthy-iteration gates, including when a prefix is retained.

The change adds an in-memory snapshot per attempted value step, bounded to one step plus the existing epoch snapshot. With the 5,928,616-parameter utility model, a full FP32 model plus two initialized Adam moments is approximately **68 MiB** of additional snapshot storage; actual allocation depends on initialized state. No extra training data or snapshot files persist. An accepted prefix also requires the existing final train/probe measurements that a fully rolled-back phase could reuse, so successful fitting can take longer than rejecting everything. Production GPU timing is unmeasured.

## v4 variant bug and fix

At 50/45/5, legacy tables occur at absolute ordinary-game indices **17, 37, 57, ...**. Consequently `(index*3+s)%5` repeats slots 1 and 2 of the sorted five-profile list: cotton and goods-beer.

Selection now uses `(legacy_table_ordinal * opponent_count + slot) % profile_count`. The ordinal counts earlier legacy tables from committed ordinary progress, using exact 20-game quota cycles and curriculum/boost boundaries. It survives restarts and uneven collection chunks without another mutable counter. Five profiles cycle over the two opponent seats; both remain distinct when the pool has at least two profiles. The 320-game 50/45/5 plan produces **32 v4 opponent seats**, distributed **7/7/6/6/6** across balanced/cotton/goods-beer/rail-tempo/supplier in its first iteration.

`pottery-flex` remains evaluation-only and is explicitly rejected from ordinary training profile lists. Existing learner-pair rotation, 50/45/5 quotas, 4/2/2 learner counts, exploration and disabled v4 search remain unchanged. `provenance.legacy_profiles` binds each table's ordinal and physical seats. `sampling.legacy_profiles` counts actual completed seats, including zero-count profiles, and compact history retains it.

The existing shared funded-industry/v5 work in this checkout was preserved, including its source-bound runtime identities. This fix does not promote a candidate bank or rewrite the admitted training/evaluation pool.

## Validation

**75 focused tests passed**, covering value-step and epoch rollback, exact retained Adam state, zero safe steps, later-epoch failure, held-out rejection, accepted versus attempted usage, two-iteration disk buffers, critic/GAE gates, checkpoint/RNG continuation, all five profiles, physical seats, irregular chunks, quota/boost boundaries, evaluation separation, search disablement and source freshness. See the [validation inventory](../artifacts/ppg-v4-fix-20261018/validation.json).

The [bounded compute smoke](../artifacts/ppg-v4-fix-20261018/guard-smoke/report.json) used the current medium v11.2 utility model, authored Features-4 tensors (8 games / 128 rows, 1–4 legal candidates), CPU/two threads. No native games, evaluation campaign or production checkpoints were used.

| Check | Accepted / attempted steps | Retained policy KL | Held-out loss |
| --- | ---: | ---: | ---: |
| Normal cap 0.002, unchanged value LR/beta | 7 / 7 | 0.000072412 | 1.64324 → 1.62349 |
| Stricter diagnostic cap 0.000020665 | 1 / 2 | 0.000010946 | 1.64324 → 1.62428 |

The diagnostic cap was chosen between the first two **real measured** step KLs, then the same initial weights, Adam and RNG were restored. The second step reached 0.000030384 and was rejected; its predecessor passed validation and remained. Normal fitting took 2.04 seconds; the strict case took 0.69 seconds; the entire smoke took 5.83 seconds. These small compute measurements do not predict lab throughput, native-game stability or playing strength.

## Continue the existing run

Keep the existing config, IL folder, validation probe, population and checkpoint. No cap edit, config regeneration or restart is needed. The exact reviewed source transitions cover the reported generation-one source, the preceding rehearsal fix, and the checkout at the start of this task. Unrelated config changes and stale pending collections still fail admission.

```bash
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 1
```

This continues from the latest committed generation. Add `--expected-generation 1` only if generation 1 is still the intended entry; use `--iterations 20` for twenty additional iterations. Review the next normal report's accepted/attempted value steps, retained KL, held-out loss and variant counts before drawing conclusions about training stability.

Changed implementation files: `recovery/value_phase.py`, `curriculum.py`, `ordinary.py`, `league.py`; exact source/handoff review receipts; lab instructions; focused regressions and the isolated compute smoke. The model architecture, existing config, IL, probe and archived report files were not changed. Tested implementation: `bee655014a2a238555afc936e2e5c341f8b7e630ddf63a60a77704ee344c41c2`; unchanged learning compatibility: `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
