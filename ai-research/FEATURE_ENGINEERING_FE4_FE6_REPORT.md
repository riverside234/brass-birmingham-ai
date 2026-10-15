# FE4–FE6 implementation and bounded validation

14 October 2026. All six feature groups are enabled in both encoders. Width 304
is retained. No full lab training or playing-strength campaign was launched.

## Frozen contract and model

`configs/learning/features-2.0.0.json` is the ordered field/source/scale/mask
contract for `permitted-features/2.0.0`: **5,489 state / 548 action** inputs,
including 326 state facts and 52 action facts, each with seven components.
State summaries precede the final style[4]/intent[8] tail and reach the shared
policy/value/utility embedding. Candidate consequences enter the action encoder.

| Model portion | Parameters |
| --- | ---: |
| State encoder, width 304 / four residual blocks | 2,413,760 |
| Base, including action/policy/official outcome heads | 2,580,429 |
| With utility | 2,590,449 |
| With utility and existing goal extension | 2,656,282 |

Architecture `brass-four-seat-mlp-resnet/2.0.0` explicitly permits **3,000,000**
base/utility parameters. The personality model has a versioned **3,100,000** cap
and retains the 100,000 extension allowance. These checks remain active.

Finance, provisional order, opportunities, realization, development and candidate
effects are all mandatory; missing source facts fail admission. Reference and
optimized tensors agree across all four seat rotations, phases and changing legal
candidate counts. Original choice IDs/order and label alignment are retained.
Static categorical caching remains separate from dynamic consequences. Inputs
remain detached, outputs immutable, caches bounded, and histories are not serialized
again for the additional derived-context identity. The existing exact-type snapshot
helper replaces repeated scalar deepcopy dispatch; immutable exact facts and the
layout identity have bounded caches. No validation or feature group was removed.

Learning/model/source identities now include the new schema and derivations.
Mixed-version arrays/batches, old checkpoints and stale teacher identities are
rejected. `.features-v2.*` uses separate cache paths and receipts. Oversized shared
memory frames retain the queue fallback without dropping actions. Dynamic batching,
candidate chunking, masks, stable-seat outputs and existing deadline/cancellation
handling remain active. Relevant personality handoff/package metadata includes
the new schema; old rosters/packages are not automatically requalified.

## Preservation and data preparation

The before snapshot and old benchmark runtime are under
`artifacts/feature-engineering/fe4-fe6/`. The latest-model preset changed only
`architecture` and `parameter_limit`. Its 0.08 exploration, temperature/PPO/reward,
curriculum, four-seat collection, no-gate recipe, population/retention, opponent
search and worker controls are unchanged. Its explicitly pinned teacher clocks
remain **12/12/16/24 seconds**, five games, cadence 160, eight distillation steps
and `search_wall_fraction=null`. Latest-model continuation preserves these explicit
clocks rather than applying archived gated-run migrations.

`scripts/feature-engineering.py convert` is an explicit heuristic-only raw
re-encoding workflow. It checks old contract/rules/source hashes, reconstructs
every native transition, validates full permitted observations/legal lists and
terminal results, then writes a new manifest/cache directory. Two archived games
passed: **345 training and 346 validation decisions**. Raw game objects, choices,
chosen labels, setup/splits and source bytes were preserved. Neural/search labels
and missing required facts are rejected; failed conversions commit no manifest.
Normal training retains qualified-bank/admission checks. Old encoded caches,
weights and optimizer states are not migrated into the new schema.

`scripts/feature-engineering.py configure` binds a completed new IL checkpoint
and its lineage to a separate current-recipe config. Production source-review
registries were not expanded to admit v1 weights. Historical migration unit tests
use explicit test-only receipts for synthetic same-contract states.

## Verification

**129 focused tests passed in 83.594 seconds**. Command, with project dependencies
and `src`, `tests/R2`, `tests/Recovery`, `tests/FeatureEngineering` on PYTHONPATH:

```text
python -m unittest test_fe1 test_fe2 test_fe3 test_fe4 test_recovery test_exploration08 test_latest_model_recipe test_teacher_phase_revision test_time_budgets test_inference_deadline_race test_teacher_readiness_review -q
```

Coverage includes all A–F arithmetic/masks/native effects, parity, observer/private
input isolation, stale contexts, checkpoint/Adam/RNG restoration, neutral BC label
alignment, incompatible arrays/caches/checkpoints, shared-memory overflow, full
legal masks, behavior-probability reconstruction, teacher quality/freshness/reuse,
null cumulative admission and shared turn clocks. Native preflight parity, device
backward and the optimized ordinary/search workers passed. Its receipt was saved
separately, preserving the existing preflight artifact.

Official-head ONNX parity passes. Additional utility and personality/goal named
exports pass shapes `(batch,candidates)` **(1,1), (2,7), (4,31), (1,257)**, including
masked logits and all output heads. Largest absolute error is below **0.000035**
on ORT CPU. These are numerical/compatibility checks, not release qualification.

The final isolated native smoke used the optimized final source: real random
width-304 IL for one epoch on the two audited games, exact completed-IL resume,
fresh IL-to-PPO handoff, **two ordinary games over two iterations**, exact
Adam/RNG/curriculum/lineage resume, four diagnostic seat-rotated evaluation games
(zero failures), and four teacher turns. It took **274.531 seconds** on local CPU
with reduced actors/minibatches/PPO epochs. PPO used 296 and 78 eligible learner
decisions from four and one learner trajectories, with three and one optimizer
steps. An earlier six-game mixed-table smoke also passed before the allocation
optimization; it remains separate evidence and was not imported into the final run.
Existing production runs/checkpoints and the installed game were not modified.

## Teacher measurements

Final-source probes used the pinned production phase clocks and calibrated utility
model, four search workers, real observed roots and unchanged quality gates.
Each row is one reserved whole-turn probe, not a complete teacher collection.

| Phase | Budget s | Search attempts / completions | Rollout attempts / completions | Visited / legal actions (revisited) | Qualified labels | Insufficient coverage | Timeouts | Total elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Early Canal | 12 | 1 / 1 | 15 / 8 | 4 / 387 (2) | 1 | 0 | 0 | 12.125 |
| Late Canal | 12 | 1 / 1 | 16 / 8 | 4 / 101 (2) | 1 | 0 | 0 | 12.282 |
| Early Rail | 16 | 1 / 1 | 16 / 8 | 4 / 198 (2) | 1 | 0 | 0 | 16.141 |
| Late Rail | 24 | 1 / 1 | 56 / 48 | 4 / 4 (4) | 1 | 0 | 0 | 24.047 |

Total: **4 generated qualified labels**, **72 completed / 103 attempted rollouts**,
64.595 elapsed seconds, about **1.115 completed rollouts/s** and **0.0619 qualified
labels/s** for these four probes. Search deadline-overrun counters are zero; total
elapsed includes completion/bookkeeping. **Distinct native labels actually used:
0; unused: 4; presentations: 0; teacher policy drift: 0** because these partial
probes never enter training. The ordinary run had not reached its first scheduled
collection threshold. This does not measure full five-game collection throughput.

A separate explicitly **authored** five-game/80-row-identity fixture checks real
distillation mechanics at width 304: 80 qualified rows, **80 distinct used**, zero
unused, 128 presentations, eight steps of 16, maximum reuse two. Policy KL before
versus after that fixture update is **0.000392383** over 80 rows. These are synthetic
label-accounting results, not native position/label yield. Earlier prototypes used
different weights/positions; their yields cannot establish a before/after gain.

## Paired performance and remaining overhead

The unchanged corpus has 14 positions / 1,014 candidates, SHA-256
`0a6923e6fcd8b0497f0674c3bb6f4b728cfe213b314265989544f3b02e4ce28b`.
Both arms use width 304, the same seeds/actor/thread counts and old-weight
predictions preserved by zeroing only added input weights in the **test control**.
No production weight importer is provided. All three native choice sequences match.
The larger dimensions/model work are included, so this is not a pure encoder-only
end-to-end comparison. Timings are local Windows CPU, 70 corpus calls per mode.

| Encoding mode | Before median / p95 ms | Final median / p95 ms |
| --- | ---: | ---: |
| Reference, full validation | 38.589 / 181.302 | 44.536 / 198.284 |
| Prepared cold, including new encoder construction | 2.702 / 10.880 | 8.739 / 32.008 |
| Persistent encoder, result-cache misses; categorical/admission caches warm | 1.162 / 3.172 | 3.225 / 13.729 |
| Prepared warm result-cache hits | 0.895 / 1.541 | 0.875 / 1.674 |

The new facts/wider tensors add about **2.06 ms median** per persistent-encoder
miss. Warm hits are essentially unchanged. Cold construction also parses/checks
the larger contract. Profiling identified redundant snapshot/fact allocation;
the final implementation reduces it without dropping fields or weakening gates.
Before that optimization, prepared-cold median was 10.484 ms and the persistent
miss probe was 10.220 ms. Remaining work is the new context and wider tensor/model
processing. Cold-path overhead remains material in relative terms and must be
included in lab search-throughput expectations.

Separate Python-allocation passes peak at **1.896 → 4.861 MB** for this warm-cache
corpus. Cold peaks are 1.082 → 2.979 MB. These are not total peak RSS or GPU memory.
RSS was unavailable in this runtime; GPU utilization/peak memory are unmeasured
because the installed Torch runtime is CPU-only. The existing candidate/worker/
cache bounds remain active, but each candidate tensor is wider.

One warmup game per arm took 24.219 → 25.984 s. The two timed games took
**17.937/15.860 → 19.890/11.657 s**, or 0.0592 → 0.0634 games/s in aggregate.
The per-game variance prevents a reliable end-to-end speedup/regression claim
from this small sample. Final actor Python CPU averages were 0.331/0.350 cores;
this excludes Mono children/GPU and is not machine utilization. Search budgets
were not reduced. Production GPU/concurrency/teacher throughput needs lab measurement.

## Evidence and next action

Machine-readable evidence is in `artifacts/feature-engineering/fe4-fe6/`: final
tests, converted-final, native-final, authored-distillation, named-exports,
preflight, before snapshot, paired before/after-optimized, persistent-miss probes,
parameter/preservation/review reports. Exploratory failures/prototypes are retained
separately; the selected final receipts identify their source/contracts.

Use [the lab guide](../docs/FEATURE_ENGINEERING_LAB.md) for exact conversion,
fresh IL, evaluation, new config/start and same-version continuation commands.
No improvement in playing strength, economic stability or learning quality is
demonstrated by these bounded checks.
