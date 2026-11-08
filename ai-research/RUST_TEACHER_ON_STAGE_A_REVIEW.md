# Qualified teacher activation and search-value Stage A

Reviewed 2026-11-08. Pulled inside `results/` through commits
`92dbef3e5f76ecfe71f384101ebcd1f91200ffa6` and
`a966c7583590577393bbf687a14b894cd6c657b3`. The four full-volume shadow
iterations **passed**. The user explicitly requests teacher on-mode and auxiliary
search-head training, starting at Stage A. The implementation and launch config
are ready; the first real on-mode iteration remains to run on the lab.

## Four-iteration evidence

The committed run is `joint7-v11-2-g1290-trained-restart`, **g1261**, 403,520 games.
The [g1261 export](../results/joint7-v11-2-g1290-trained-restart-g1261-20261108T070035419646Z/state.json)
binds checkpoint `generation-1261.pt`, SHA-256
`45ba3988d4a7cd7fa8d6b78fc2edd738268aa179e33023c5b7046439ebd2e578`, model
`abed90dda89fd6bf384401c88b9364750eb724f29e9ffb8af34308903d4a8864`.
The [qualification export](../results/rust-lab-qualification/20261108T063358240043Z-20261108T060920000950Z-shadow-training-05484c69-61fd605f/evidence/qualification.json)
reports successful training and gradient qualification. The later run export
retains the complete four-observation receipt.

| Collection G → checkpoint G+1 | Tables train/held-out | Policy labels, all splits | Valid gradient steps | Median completed trajectories | Root-label yield | Teacher collection seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1257 → 1258 | 28/4 | 512 | 46 | 51 | 99.805% | 116.612 |
| 1258 → 1259 | 28/4 | 516 | 46 | 54 | 99.805% | 12.666, all 32 tables recovered from cache |
| 1259 → 1260 | 28/4 | 513 | 46 | 52 | 99.805% | 119.100 |
| 1260 → 1261 | 28/4 | 512 | 46 | 50 | 100% | 120.216 |

Each observation covers all four Canal/Rail phases. The pooled median of all
**184** finite positive raw teacher/ordinary-actor gradient ratios is
**1.551817712660179**. At share target **0.25**, the qualified fixed weight is
**β = 0.16110139609854138**, without clamping. Receipt SHA-256:
`e3c986df434df8f4b1a6a26e83486e1784277c596de406e0ccd4cf646e012eab`.
Do not estimate a fresh-collection speedup from the cached g1258 measurement.

The g1261 readiness receipt is a real current-reward PPG fit: 236,598 decisions,
608 games and 16 steps; held-out MSE improves from 0.155740 to 0.154656. This
activation does not use a null-readiness exception. The entropy schedule is
complete. Search-value state remains A, `w=0`, trunk scale zero, with no trained
companion yet: shadow collected value rows without applying head gradients.

## Review findings and repairs

1. **First-head model identity.** Eagerly installing the new residual before
   collection changed the model ID while frozen jobs still referred to the
   committed checkpoint. For this reviewed activation, first installation now
   happens at the post-PPG head-training boundary. Existing heads and their Adam
   state still restore eagerly from full checkpoints. The first collection uses
   the exact g1261 model; the new head belongs to the g1262 checkpoint.
2. **Completed entropy schedule.** The static schedule switch remains `on` after
   its ramp finishes. Testing only that switch would defer the head forever.
   Activation and head training now check committed schedule progress and any
   waiting reward stage. An incomplete ramp still blocks activation.
3. **Qualified on-mode wiring.** Reuse the production Rust ordinary collector,
   the full 32-table native teacher barrier, current binary receipts, label/value
   admission and PPO term. Bind activation to the four measured observations,
   their actual checkpoint/model lineage and recomputed pooled median. Missing,
   altered or incomplete qualification is rejected, including a resealed but
   incorrect median. Record a separate activation receipt in backend state.
4. **Recovery and measurement.** Preserve pending work only when the full
   projected frozen collection signature is identical; changed targets/actors,
   unknown source edits and unfinished next checkpoints retain their guards.
   Reuse validated completed tables and normal cold admission. Report generic
   `teacher_collection_seconds` and `teacher_collect=` for on-mode; retain the
   legacy shadow timing key for shadow reports.

The evidence did not justify changing staleness or search rules. Old label ages
are correctly skipped when policy drift exceeds their original search
improvement; current-age train labels remain available. Root-label coverage is
high despite roughly 43–50% deadline-hit fractions, which reflect bounded
searches. Encoder cosine medians are positive but small, with negative tails;
monitor the applied gradient share and ordinary PPO health after activation.
Held-out teacher KL growth under shadow measures ordinary policy drift, so it
does not establish that teacher-on improves playing strength.

Inference-channel observations in the last three reports return to **zero**
active channels, shared slots and leases, with created/released counts matching
(48/48, 129/129, 210/210). No repeated descriptor leak is visible. Evaluation
and retention work explain the large g1259 total; retain their separate timings.
Checkpoint publication grows to 14.87 seconds in the final report and is a
remaining cost to monitor as value-label history grows. No timing or strength
claim is made for on-mode before its lab run.

## Activated recipe

Use [the on/Stage A config](../configs/step5/joint7-v11-2-g1290-trained-restart-rust-teacher-on-stage-a-share025.json).
It preserves branching **0%**, full **28 train + 4 held-out** teacher tables,
search_v1, V205 off, DOMiNO/AZ_DB absent, and the existing ordinary actor,
publisher, inference and CPU limits. Full and fast search budgets stay 4s/K16
and 0.8s/K4; independent verification remains uncapped under the current
user-requested verification policy. No Rust source or executable changes are
required for this Python training activation.

Teacher policy gradients now enter the existing guarded PPO update with the
qualified **fixed β**, using only the actor parameter allowlist and independent
sampling RNG. Critic/outcome/search-head parameters remain excluded from that
term. Existing KL limits, teacher staleness and held-out isolation remain.

The residual starts **Stage A**, `w=0`, trunk scale **0**, and trains after the
ordinary PPG boundary with its own Adam: **32 steps per iteration**, batch size
512, learning rate 0.0001. Main weights and ordinary Adam are untouched by this
head-only pass. The residual does not contribute to search while `w=0`.
Later blend/trunk stages retain the existing held-out/window and ordinary-health
gates; enabling Stage A does not manually advance them.

Monitoring does not change β. Explicit future target edits to 0.4 or 0.5 use
the same frozen raw calibration, producing fixed β values 0.2577622337576662 or
0.32220279219708275, respectively. Do not remeasure or silently adapt β.

## Lab continuation

Sync the code checkout, stop any older trainer and use the existing training
environment from its repository root:

```bash
python scripts/continue-rust-teacher.py --expected-generation 1261 --iterations 1 --share-target 0.25
```

This advances **g1261 → g1262** using the full checkpoint, ordinary/companion
Adam, RNG, PPG state, population and counters. The helper validates source,
checkpoint and qualification, writes the exact on-mode config, and exports
success/failure evidence under `results/rust-lab-qualification/`. Repeating the
same command only completes remaining requested work. Once g1262 commits,
subsequent continuation uses its actual current generation. No adapter rebuild,
new ZIP or repeat of successful qualifications is needed.

Review the resulting `teacher_phase.mode`, applied fixed beta and weighted
share/cosine, held-out KL, ordinary PPO health, resource counters and separate
collection/update/checkpoint timings. `search_value.update.steps` should show
the head-only update when valid train rows are present, with Stage A `w=0` and
scale zero until its normal gates pass. Missing task readiness still defers
teacher collection; it never fabricates labels or bypasses admission.

**Validation:** 38 distinct affected debug regressions passed, including strict
checkpoint/Adam/RNG restoration, the real transition commit and retry, public
launch preparation, frozen collection identity, full 32-table authored
barrier/cold recovery, the actual teacher policy gradient, first-head 32-step
update and interrupted rollback, strict head reload, zero-weight inference and
chunking, and immutable checkpoint publication. Tiny CPU fixtures exercise
updates; no native games, benchmarks, lab-sized training or unrelated successful
qualification were run locally. See the [machine-readable receipt](../distribution/rust-collector-performance/teacher-on-stage-a.json).
