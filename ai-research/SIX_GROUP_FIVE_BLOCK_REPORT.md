# Six-group rollback and five residual blocks

15 October 2026. Hand coverage has been removed from the active pipeline. The
six existing feature groups A–F, their semantic checks and unrelated bug fixes
are retained. See [the current lab guide](../docs/FEATURE_ENGINEERING_LAB.md) for
raw-data conversion, fresh IL, self-play initialization and later resume commands.

## Active model and features

The immutable `permitted-features/2.0.0` schema is active again: **5,489 state /
548 action inputs**, with style[4]/intent[8] still last. The hand-coverage context
pass and its 28 state/four action facts are removed, including the production
`hand.py` module. Existing raw hand counts and card/action categorical features
remain. The archived schema 3 is retained for interpreting explicitly re-audited
old heuristic recordings; its coverage features are not computed or fed to models.

Default, small and medium model presets now use **five residual blocks**.
Medium width remains **304**. The architecture is separately versioned as
`brass-four-seat-mlp-resnet/4.0.0`, while the original feature schema is reused
without modification.

| Component | Parameters |
| --- | ---: |
| State encoder | 2,599,808 |
| Base, including policy/official heads | 2,766,477 |
| With utility head | 2,776,497 |
| With personality/goals | 2,842,330 |

Existing 3M base/utility and 3.1M personality caps remain sufficient. The fifth
block adds 186,048 parameters to the earlier four-block A–F model. Removing the
hand inputs saves 63,168 input weights, giving a net increase of 122,880 parameters
over the preceding hand-coverage/four-block model. Removing the coverage pass
does not establish a total training speedup: the extra residual block adds neural
forward/backward work, and no matched throughput campaign was run.

## Preserved fixes and settings

The `configure` command still binds rehearsal to the completed IL run's audited
dataset, preserving rehearsal weight/duration. Retention still pins and retires
all known unversioned/v2/v3 feature sidecars with their raw replay and supports
restore. The three native/encoded semantic regressions remain: affected-player
attribution; action-specific resource/merchant/beer access; and development unlock
versus practical realization. Deadline/cancellation, legal masking, strict PPO
behavior probabilities, raw audit and source/checkpoint admission safeguards remain.

The checked-in latest-model recipe changes only architecture from 3 to 4.
Exploration 0.08, PPO, rewards, insolvency cap 0.075, curriculum, four-seat learner
collection, disabled champion gate, opponent settings, teacher five games/cadence
160/eight distillation steps, explicit phase clocks 12/12/16/24 and
`search_wall_fraction=null` are unchanged. Review/source-approval registries and
the immutable feature schema files are unchanged.

## Validation

All **208 focused tests passed in 90.658 seconds**. They include actual five-block
module/parameter counts, reference/prepared parity, dynamic ONNX output/mask checks,
incompatible old models/hand tensors, shared inference, checkpoint Adam/RNG,
config binding, cache retention/restore, semantic checks and teacher deadlines.

Two existing heuristic games were re-audited from the archived Features-3 raw
manifest into a separate Features-2 output. Full native audit took **29.812 seconds**.
All records, choices/labels and setup/splits are preserved. Every state/action,
offset and seat array matches the original six-group cache exactly across
**691 positions / 27,866 candidates**. No new demonstration games were required.

The final CPU smoke completed in **244.484 seconds**. It exercised one fresh IL
epoch, two ordinary games across two PPO iterations with save/resume, four
diagnostic evaluation games (**zero failed**) and four reserved-turn neural teacher
probes. PPO used **136 and 37 distinct eligible actor samples**, with none unused;
critic usage was 192 and 64. Resume preserved all 50 Adam parameter states and the
saved RNG exactly. The earlier committed checkpoint was unchanged.

| Teacher phase | Attempted/completed rollouts | Qualified / insufficient coverage | Whole-turn elapsed s |
| --- | ---: | ---: | ---: |
| Early Canal | 16 / 8 | 1 / 0 | 12.109 |
| Late Canal | 16 / 8 | 1 / 0 | 12.140 |
| Early Rail | 28 / 20 | 1 / 0 | 16.016 |
| Late Rail | 56 / 48 | 1 / 0 | 24.031 |

There were zero teacher timeouts or reported deadline-guard overruns. Whole-turn
elapsed includes finalization. Four qualified labels were diagnostic: **zero
distinct labels trained, four unused, zero teacher policy drift**. This is not a
full scheduled five-game collection or a distillation update. The production
cadence and gates were unchanged.

Evidence root: `artifacts/feature-engineering/rollback-five-blocks-20261015/`.
`review.json` verifies final source/compatibility, preserved files/configuration,
raw records/splits and exact six-group arrays. `tests.log`, `numeric-parity.json`,
`converted/conversion.json`, `smoke/report.json`, `inspect.json` and
`lab-config.json` contain the underlying receipts.

This is bounded CPU functional validation, with reduced actors/PPO batch settings.
Production CUDA/concurrency, a full scheduled teacher collection/distillation and
playing-strength improvement are not established. Existing datasets and production
checkpoints were not overwritten.

## Checkpoint use

Use fresh **five-block** IL and a new self-play output. Previous four-block and
hand-coverage checkpoints remain incompatible, even with `--resume-reviewed-source`;
there is no silent padding, fifth-block insertion, optimizer migration or reset of
an old run. New five-block checkpoints resume their own model, optimizer, RNG,
counters, curriculum, population and lineage through the normal continuation path.
The updated lab guide uses distinct `features2-five-block-*` paths.

The earlier [hand report](HAND_MATCHING_REPORT.md), [contract](HAND_MATCHING_CONTRACT.md)
and [addition plan](FEATURE_ENGINEERING_ADDITION_PLAN.md) are archived experiment
records and explicitly marked as superseded.
