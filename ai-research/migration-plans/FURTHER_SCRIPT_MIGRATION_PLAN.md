# Further script migration plan and results

Prepared on 2026-11-17 against checkout `306423a2`, following the completed
[local-game migration](LOCAL_GAME_SCRIPT_MIGRATION_PLAN.md) and
[training migration](TRAINING_SCRIPT_MIGRATION_PLAN.md).

All **76 script moves** listed below are implemented: the 36 first-pass moves
and all 40 follow-ups. The 44 retained scripts remain at their original paths.
The original review and disposition lists below describe the pre-migration
layout; the workflow indexes link to current runnable paths. Compatibility
checks and implementation details are recorded at the end.

## Review scope and current workflow

The review covers all **120 scripts remaining directly under `scripts/`**, plus
the indexes and cross-folder callers of the **293 scripts already categorized**.
Counts include Python, PowerShell, Bash, and the C# weaver; they exclude
READMEs and generated `__pycache__` files.

| Location | Scripts reviewed | Recommendation |
| --- | ---: | --- |
| `scripts/` directly | 120 | Apply the disposition lists below |
| `scripts/training/` | 54 | Keep the completed historical migration grouping |
| `scripts/local-game/` | 145 | Keep feature groups and their shared callers |
| `scripts/legacy/` | 43 | Keep archival workflows, including 40 Rust lab tools |
| `scripts/experiments/` | 23 | Keep one folder per experiment and its helpers |
| `scripts/reports/` | 17 | Keep the existing single-level workflow |
| `scripts/benchmarks/` | 11 | Keep the existing single-level workflow |
| **Total** | **413** | |

The newest available local export is
`joint7-v11-2-g1190-restart-3-g3170-20261117T205147266837Z`, rather than the
g2990 reference used for the previous plan. Its manifest and state both record
generation **3170** for `artifacts/step5/joint7-v11-2-g1190-restart-3`.
The recipe retains 32 training teacher tables and four held-out teacher tables.
The ordinary table quota is 70% self-play and 30% historical opponents, with
zero legacy tables in that quota. These are observations of the export, not
configuration changes proposed here.
[Export manifest](../../results/joint7-v11-2-g1190-restart-3-g3170-20261117T205147266837Z/export-manifest.json) ·
[Committed state](../../results/joint7-v11-2-g1190-restart-3-g3170-20261117T205147266837Z/state.json)

“Occasional” below means setup, manual diagnostics, historical reconstruction,
or release work outside the normal continuation call chain. Invocation counts
are not recorded centrally, so this is a dependency and workflow assessment,
not measured usage frequency. References were checked by full script path,
basename, import name, dynamic loaders, test fixtures, CI, source inventories,
and archive builders. A textual occurrence of `r2` in an R-squared metric is
not a caller of `r2.py`.

Current training still uses IL lineage, the fixed IL probe, model checkpoints,
historical opponents, and shared learning/bank implementations. Relocating a
CLI does not retire its artifacts or the `src/brass_ai/` modules it imports.
[Training source inventory](../../src/brass_ai/learning/source_identity.py) ·
[Bank loading and admission](../../src/brass_ai/bank.py) ·
[Native transfer dependencies](../../scripts/step5_transfer_runtime.py)

## Recommended scope

| Disposition | Scripts | When |
| --- | ---: | --- |
| First pass: imitation/model tools, bank tools, checks, reference hosts | 36 | Move with ordinary path, caller, test, and documentation revisions |
| Earlier recovery commands | 2 | Follow with disposable restore/retry coverage |
| Checkpoint export/calibration pair | 2 | Follow after resolving the frozen reference lookup |
| Rust reference, qualification, report, benchmark, and oracle tools | 36 | Follow in dependency groups with archive/receipt coverage |
| Retain current paths | 44 | Live entry points, source-bound builders, and required shared dependencies |
| **All remaining top-level scripts** | **120** | **76 proposed moves; 44 retained** |

The first pass reduces the top level to 84 scripts. Completing all proposed
follow-up groups reduces it to 44. A conditional group stays in place until its
listed callers and evidence readers work at the new locations. Existing
documentation references and fixed parent-depth calculations are ordinary
migration work, rather than permanent exceptions.

Keep CLI flags, defaults, exit status, outputs, run roots, model selection,
generation checks, and restore behavior. Update scripts, callers, tests, and
current command examples as needed. Do not add compatibility wrappers, change
training objectives, or regenerate old evidence to accommodate a folder move.

## First pass: 36 moves

Each list records source basenames formerly under `scripts/`. They now reside
at the destination shown, preserving the basename.

### Imitation learning and model training: 6

Destination: `scripts/training/imitation/`.

```text
r2.py
feature-engineering.py
il_replay.py
smoke-s1-training.py
verify-s1-training.py
verify-r2.py
```

These are fresh-model, conversion, inspection/export, rehearsal preparation,
and earlier scaffold checks. The g3170 continuation does not invoke these six
entry points to initialize the learner on every iteration. Preserve their
availability for fresh IL and model tooling; `feature-engineering.py` also
supports current model inspection and export, so it belongs with maintained
training tools rather than retired bundles.

Update `tests/Recovery/test_il_replay.py` and its importing fixtures, the
FeatureEngineering smoke/benchmark launchers, and the benchmark that invokes
`feature-engineering.py`. Keep `il_replay.py` calls to the retained
`scripts/step5.py`. Compare its generated continuation argv and its own script
provenance field. Search R2 and R3 command examples separately from R-squared
statistics.
[IL rehearsal driver](../../scripts/training/imitation/il_replay.py) ·
[Model tooling](../../scripts/training/imitation/feature-engineering.py) ·
[IL fixture](../../tests/Recovery/test_il_replay.py) ·
[Feature benchmark caller](../../scripts/benchmarks/benchmark-feature-engineering.py)

### Historical opponents and legacy banks: 12

Destination: `scripts/training/opponent-banks/`.

```text
prepare-legacy-bank.py
freeze-legacy-bank.py
collect-legacy-bank.py
check-frozen-bank.py
check-legacy-admission.py
sample-legacy-pool.py
qualify-legacy-bank.py
amend-v3-qualification.py
promote-v3-bank.py
verify-v3-bank.py
verify-r3.py
capture-h0.ps1
```

These prepare, freeze, collect, qualify, or inspect opponents and old stock-AI
baselines. Current training consumes admitted bank/profile/checkpoint state;
it does not run these preparation CLIs for every historical table. Keep the
whole maintenance workflow together even when a particular bank is historical.

`verify-v3-bank.py` records four bank-tool paths in new provenance;
`verify-r3.py` records its own path. Revise those new-record source locators
together. Existing qualification reports and frozen source snapshots retain
their original paths and hashes. Bank loading verifies sources inside the
bank snapshot; it is not a reason to keep every bank CLI at the top level.

Update `scripts/local-game/mixed-ai/verify-mixed-legacy.py`, which hashes the
collector script, and the R1 guide's stock-capture command. Preserve pool
membership, training/evaluation separation, frozen IDs, transferred runtime
support, and `training/opponents/` contents.
[Frozen bank reader](../../src/brass_ai/bank.py) ·
[V3 provenance producer](../../scripts/training/opponent-banks/verify-v3-bank.py) ·
[Mixed-AI caller](../../scripts/local-game/mixed-ai/verify-mixed-legacy.py)

### Training checks: 9

Destination: `scripts/training/checks/`.

```text
check-step5-distribution.py
check-step5-evaluation.py
check-step5-scaling.py
smoke-dcpg.py
smoke-local-scoring.py
smoke-ppg-throughput.py
smoke-ppg-phasic.py
verify-recovery.py
verify-step5.py
```

These remain useful manual regression commands but are outside the routine
continuation chain. Move the four related smoke commands together:
`smoke-dcpg.py`, `smoke-ppg-phasic.py`, and `smoke-ppg-throughput.py` load
`smoke-local-scoring.py` as a helper. The distribution check must retain its
import from `scripts/reports/check_qualification_reports.py`. Update the
dynamic `verify-step5.py` loader in `tests/Recovery/test_lab_review.py`.
[Shared smoke fixture](../../scripts/training/checks/smoke-local-scoring.py) ·
[Distribution check](../../scripts/training/checks/check-step5-distribution.py) ·
[Lab review test](../../tests/Recovery/test_lab_review.py)

### Reference rules host and contracts: 9

Destination: `scripts/training/rules-host/`.

```text
build-host.ps1
check-r1-runtime.ps1
check-r1-traces.py
define-r1-contracts.py
export-catalog.py
smoke-host.py
test-r1.py
verify-r0.py
verify-r1.py
```

These set up or check the earlier reference host and contract milestones.
Keep their generated runtime and contract paths unchanged. The separate three
source-bound training-host builders listed under retained paths stay at the
top level.

Update the compiler-setup diagnostic in
`scripts/local-game/legacy-bridge/build-legacy.ps1` to name the moved
`build-host.ps1`, and the schema/reference tests that open a script directly.
`check-r1-runtime.ps1` must continue to load the already relocated legacy-bridge
helper. Preserve `-GameRoot` defaults: the checkout root remains the default
after the script directory changes.
[Legacy builder](../../scripts/local-game/legacy-bridge/build-legacy.ps1) ·
[R1 runtime check](../../scripts/training/rules-host/check-r1-runtime.ps1) ·
[R0 reference test](../../tests/R0/test_references.py)

## Follow-up groups: 40 moves

### Earlier recovery commands: 2

Destination: `scripts/training/migrations/`.

```text
continue-rust-revised.py
joint7_restart160.py
```

These are not the current g3170 continuation driver. They were deferred from
the previous pass because they preserve and replace run directories, rather
than merely print configuration. Move them in a separate batch with their
existing restore/retry fixtures. Update the direct `joint7_restart160` import
and patch targets in its test. Keep calls from the revised-reward restorer to
the retained transfer, reward-preparation, and teacher-continuation scripts.
[In-place restore test](../../tests/Transfer/test_inplace_revised_restore.py) ·
[g160 reconstruction test](../../tests/Recovery/test_joint7_restart160.py)

### Checkpoint export and calibration: 2

Destination: `scripts/training/checkpoints/`.

```text
export-test-nn.py
test_nn_calibration.py
```

Keep this pair together: each dynamically loads the other by sibling filename.
The NN beta2 builder loads the exporter, as does
`src/brass_ai/rust_backend/baseline.py`. The latter also includes its old path
in the frozen L01 reference inventory. Its existing archived-verifier fallback
is reached only after hashing the current file; moving it without revising
that lookup would raise a missing-file error before the fallback.

This group therefore needs explicit frozen-archive handling and model/export
fixtures, not just a root-depth edit. Preserve verification of the exact
archived member and digest, and keep old receipts unchanged. A change to the
shared baseline module belongs in this separate batch and must be assessed
against the current training source identity.
[Frozen baseline verifier](../../src/brass_ai/rust_backend/baseline.py) ·
[NN beta2 caller](../../scripts/local-game/nn-beta2/build-nn-beta2.py) ·
[Exporter tests](../../tests/Reports/test_export_test_nn.py) ·
[Calibration tests](../../tests/Reports/test_test_nn_calibration.py)

### Rust reference capture: 6

Destination: `scripts/training/rust-lab/reference-capture/`.

```text
capture-rust-privacy-edges.py
capture-rust-privacy.py
capture-rust-rule-branches.py
capture-rust-state-regressions.py
capture-rust-states.py
complete-rust-state-capture.py
```

These are occasional reference-corpus producers. Move with the qualifier
callers and explicit archive lists below. Keep imports of the retained
`capture-rust-rules.py` at its existing path. The state capture/completion
tools depend on each other, so move them together.

### Rust fixture preparation: 4

Destination: `scripts/training/rust-lab/fixtures/`.

```text
prepare-rust-feature-layout.py
prepare-rust-inference-fixtures.py
prepare-rust-onnx.py
prepare-rust-state-catalog.py
```

These produce layouts, categorical inputs, ONNX fixtures, and state catalogs
for the earlier Rust milestones. Update qualifiers and generated provenance
locators, while preserving feature offsets, exported model selection, fixture
data, and output locations. Do not use a directory reorganization to refresh
the frozen model or regenerate native feature tables.

### Rust validation: 14

Destination: `scripts/training/rust-lab/validation/`.

```text
qualify-rust-backend.py
qualify-rust-coordinator.py
qualify-rust-features.py
qualify-rust-inference.py
qualify-rust-privacy.py
qualify-rust-rules.py
qualify-rust-search.py
qualify-rust-sessions.py
qualify-rust-simulation.py
qualify-rust-state.py
verify-rust-coordinator.py
verify-rust-features.py
verify-rust-inference.py
verify-rust-search.py
```

These run milestone parity/qualification rather than each normal training
iteration. The dependencies form a chain through state, rules, features,
inference, search, sessions, and coordinator checks. The scripts also build
`implementation.zip` and source receipts from explicit root-level filename
lists. Update execution paths and newly captured archive member paths together;
otherwise some old list comprehensions silently omit missing files.

The retained scaffold and C# replay helpers still supply shared functions.
Keep their paths intact. Preserve sealed L01/L02 reference extraction, every
executed oracle digest, and the distinction between a new qualification and
old milestone evidence.
[L04 archive list](../../scripts/training/rust-lab/validation/qualify-rust-state.py) ·
[L05 archive list](../../scripts/training/rust-lab/validation/qualify-rust-rules.py) ·
[Rust workflow index](../../scripts/training/rust-lab/README.md)

### Rust measurements and study helpers: 8

Destination: `scripts/benchmarks/`, retaining the existing single level.

```text
benchmark-rust-backend.py
benchmark-rust-coordinator.py
benchmark-rust-inference.py
benchmark-rust-providers.py
profile-rust-simulation.py
l14-search-controls.py
l14-smoke.py
seal-l14-smoke.py
```

These were kept at the original paths in the earlier organization because
qualification studies capture them. Plan a complete study-tool batch rather
than treating every historical capture as a permanent top-level requirement.
Update new source archives, command generation, and dynamic loaders. Important
external callers are `scripts/benchmarks/benchmark-search-v2.py`, which imports
the coordinator's `Memory` structure, and
`scripts/local-game/host-delivery/smoke-host-delivery.py`, which loads backend
profiling functions. Preserve recorded study directories and stopping rules.
[Search v2 caller](../../scripts/benchmarks/benchmark-search-v2.py) ·
[Host-delivery caller](../../scripts/local-game/host-delivery/smoke-host-delivery.py)

### Rust study reports: 2

Destination: `scripts/reports/`, retaining the existing single level.

```text
report-l14-smoke.py
report-rust-backend.py
```

Move with the measurement group: the backend report loads its benchmark driver,
and the smoke sealer includes the report source. Update
`tests/RustBackendLocal/test_l14_report.py`. Preserve missing/open/failed gate
reporting and existing assessment data; producing a report does not requalify
an old release.

### Independent Gumbel follow-up oracles: 2

Destination: `scripts/experiments/gumbel-is-mcts/`.

```text
reproduce-gumbel-sessions.py
reproduce-gumbel-coordinator.py
```

Move these independent session/coordinator oracles with their test and
qualification callers. The coordinator loads the session oracle. The original
`reproduce-gumbel-is-mcts.py` stays top-level because its path and bytes are
pinned by `configs/references/gumbel-is-mcts.lock.json`; do not move that locked
reference as a side effect.
[Reference lock](../../configs/references/gumbel-is-mcts.lock.json) ·
[Session tests](../../tests/GumbelISMCTS/test_sessions.py) ·
[Coordinator tests](../../tests/GumbelISMCTS/test_coordinator.py)

## Retain 44 current paths

These lists account for every top-level script not included in the 76 proposed
moves. “Retain” applies to this migration, not a claim that each CLI is invoked
frequently.

### Current continuation, recovery, and transfer: 13

```text
step5.py
continue-rust-g1190-restart-2.py
continue-rust-g1491.py
continue-rust-shadow.py
continue-rust-teacher.py
recover-step5-iteration.py
resolve_checkpoint.py
joint7_preflight.py
step5-transfer.py
step5_transfer_frozen.py
step5_transfer_restore.py
step5_transfer_runtime.py
check-transferred-legacy.py
```

The generation-named launcher still loads the g1491 driver, which loads the
shared continuation. The teacher driver shares that code. Transfer is an
explicit CI entry point and its sibling imports remain together. Preserve the
documented current continuation and recovery commands.
[Current launcher](../../scripts/continue-rust-g1190-restart-2.py) ·
[Transfer workflow](../../.github/workflows/step5-transfer-release.yml)

### Training runtime builders: 4

```text
build-training-host.py
build-search-host.py
build-local-scoring-host.py
build-shared-search.py
```

The first three paths and contents are in the Step 5 implementation inventory
and runtime receipts. The shared-search builder remains a supported runtime
preparation command named by the current shared-heuristic loader. Keep it with
the retained builders for this pass; moving a builder is not necessary to
remove the occasional model/bank/check CLIs.
[Source identity](../../src/brass_ai/learning/source_identity.py) ·
[Shared-search runtime](../../src/brass_ai/search/legacy_runtime.py)

### Current CPU and teacher runtime preparation: 5

```text
prepare-rust-cpu-inference.py
qualify-rust-cpu-inference.py
prepare-rust-collector-performance.py
prepare-rust-teacher-verification.py
prepare-rust-post-transfer-reward.py
```

Current runtime resolvers name or dynamically load these files. The
post-transfer reward command is still printed by `step5-transfer.py` and used
by the reward runtime; the older word “transfer” does not make it obsolete.
[CPU resolver](../../src/brass_ai/rust_backend/lab_cpu_runtime.py) ·
[Reward resolver](../../src/brass_ai/rust_backend/lab_reward_runtime.py)

### Shared lab helpers: 6

```text
qualify-rust-lab.py
qualify-rust-lab-b02.py
rust_lab_results.py
rust_lab_b06_metrics.py
rust_lab_b07_retry.py
rust_lab_b07_completion.py
```

Retain the dependency findings from the previous migration: CPU preparation
uses the first two, current continuation exports use `rust_lab_results.py`,
and runtime rebuilding/profiling use the remaining helpers. They are not
retired merely because their filenames contain old milestone numbers.
[Shared helper index](../../scripts/training/rust-lab/README.md#shared-build-qualification-and-export-helpers)

### Local-game runtime and installation dependencies: 11

```text
build-rust-local-package.py
install-rust-local-release.ps1
install-rust-alpha.ps1
qualify-rust-local.py
qualify-rust-release.py
verify-rust-local.py
finalize-rust-local.py
test-rust-local-upgrade.ps1
test-rust-release-deployment.ps1
measure-mono-allocation.py
patch-install-common.ps1
```

Retain the exact-path release/upgrade sources and their dependent installation
helpers. The earlier local-game plan records the L15/final-release pins and
why this complete group stayed. This plan does not reopen live installation
or rewrite accepted release receipts.
[Retained local-game paths](../../scripts/local-game/README.md#paths-retained-at-the-top-level) ·
[Original pin assessment](LOCAL_GAME_SCRIPT_MIGRATION_PLAN.md#rust-source-pins-and-other-exceptions)

### Shared or locked Rust reference tools: 5

```text
qualify-rust-scaffold.py
capture-rust-rules.py
verify-rust-sessions.py
reproduce-gumbel-is-mcts.py
replay-native-failure.py
```

`qualify-rust-scaffold.py` supplies toolchain/build/receipt functions to many
current local-game builders. `verify-rust-local.py` loads
`verify-rust-sessions.py` and `capture-rust-rules.py`; moving them would require
editing the retained local-game group. The Gumbel oracle has an explicit
reference lock. `replay-native-failure.py` is captured by the L15 source pins.
Keeping these five avoids a release-source transition in the proposed batches.
[Local replay caller](../../scripts/verify-rust-local.py) ·
[L15 source pins](../../scripts/qualify-rust-release.py) ·
[Gumbel lock](../../configs/references/gumbel-is-mcts.lock.json)

## Implementation sequence and validation

1. **Refresh the baseline.** Recheck the newest committed export and current
   continuation dependency chain at implementation time. Save the actual local
   Step 5 implementation ID, CPU overlay fingerprint, relevant CLI outputs,
   and existing test results before editing. At the start of this review, local hashes were
   `95a8a6d17061d4c62810eeee906ac5a47e3f1b55008f97fac507f6b1bf55c1c3`
   and `d9719f5192a81a37b6c4498e69e8938dd27041ac9261f5b344a31b117baead00`.
   The export records implementation
   `9efcf34f853d75ab844366b6819da4b0921be38d3e2dfb806cf282ace76e7a30`;
   this pre-existing difference is not caused or resolved by a folder move.
   Recalculate the local values before implementation; other work may change
   the checkout after this point-in-time baseline.
2. **Move the first 36 in complete groups.** Start with imitation tools, then
   bank maintenance, shared training checks, and the reference-host commands.
   Compare identical fixture outputs, generated commands, and admitted bank
   identities. Update each group with its callers and docs in the same batch.
3. **Move the two recovery commands separately.** Exercise preview, preservation
   of the later run, restore, retry, interruption, and stale-generation checks
   in disposable directories. Do not operate on the real lab run.
4. **Resolve exporter reference handling, then move the pair.** Verify old
   bundle reads, calibration behavior, strict model identities, archive pins,
   and the local NN beta2 caller. A changed filename must not bypass a hash
   check or require rewriting a historical manifest.
5. **Move the 36 Rust follow-ups by dependency group.** Start with the two
   independent oracles, then reference capture/fixture/validator groups, then
   measurement/report tools together. Update explicit archive inventories and
   every retained caller. If an actual live receipt requires old source
   bytes, retain the affected file and record that precise reader rather than
   weakening it. Reconcile the final counts if a batch must be revised.
6. **Finish indexes and evidence.** Update `scripts/README.md`, training and
   Rust lab indexes, report/benchmark/experiment indexes, and runnable guides.
   Record actual moved/retained counts, tests, unavailable runtimes, and any
   additional exceptions here. Keep completed earlier plans as history.

Root lookups must reflect the chosen destination:

| Destination | Repository lookup for a Python script |
| --- | --- |
| `scripts/training/<workflow>/` | `Path(__file__).resolve().parents[3]` |
| `scripts/training/rust-lab/<workflow>/` | `Path(__file__).resolve().parents[4]` |
| `scripts/reports/` or `scripts/benchmarks/` | `Path(__file__).resolve().parents[2]` |
| `scripts/experiments/gumbel-is-mcts/` | `Path(__file__).resolve().parents[3]` |

Keep `--root`, `--project-root`, `--run`, `-GameRoot`, explicit output paths,
and caller-supplied working directories governed by their existing semantics.
Use the script directory for siblings that move together and the resolved
checkout for retained dependencies. Preserve the expected archive layout when
executing copied scripts, not just the layout of the live checkout.

Parse Python/PowerShell without executing setup at import time. Several
scripts, including `verify-r2.py`, the distribution check, stock catalog/contract
generators, and some bank commands, execute at module scope. Do not blindly
import every script or assume every `--help` path is read-only. Use existing
fixture tests and stub external processes where appropriate. Exercise the
new commands from a different working directory and a path containing spaces.

Use focused existing tests for IL replay, feature tools, banks/admission,
checkpoint export/calibration, restore/transfer, Gumbel oracles, and Rust report
parsing. Compare pre-existing failures before/after; record missing native,
private model/runtime, Mono, or CUDA prerequisites. A directory-only move does
not call for a new full training run or a long tournament.

Reconcile all 120 source basenames exactly once across the disposition lists.
Confirm no old script copies/wrappers remain for successful moves. For the
first pass, current continuation, transfer/CI paths, and the two training
fingerprints must remain unchanged. Follow-up code changes that alter a
source-bound implementation require their existing reviewed transition;
do not invent a new admission rule for this reorganization.

## Migration-plan documents grouped now

The documentation organization accompanying this plan places all three
previous migration-plan Markdown files alongside this one:

```text
ai-research/migration-plans/
|-- README.md
|-- FURTHER_SCRIPT_MIGRATION_PLAN.md
|-- LOCAL_GAME_SCRIPT_MIGRATION_PLAN.md
|-- TRAINING_SCRIPT_MIGRATION_PLAN.md
`-- RUST_AI_BACKEND_MIGRATION_PLAN.md
```

The Rust backend plan is an architectural migration history; the other three
concern script organization. Relative links inside moved plans and incoming
documentation links follow the new locations. Retired bundle publishers and
the L14 smoke source-archive list use the new backend-plan source locator for
new archives. Recorded manifests, hashes, old archives, and result exports
keep their historical paths. This document grouping moves no scripts and
changes no training configuration.

The documentation work was checked against the inventories above: all 120
top-level basenames appear exactly once, and all 413 script paths remain in
place. The only script-content revisions are 11 replacements of the backend
plan's document locator. Their Python syntax passed. Existing relative-link
targets were preserved across the document moves, and all links in this new
plan and its index resolve. None of these changed documents or script locators
is in the Step 5 implementation source inventory. Concurrent training-source
edits were left untouched; refresh the baseline before implementing the plan.

## Migration results

Completed on 2026-11-17. All **76 moves** were implemented without wrappers or
deleting scripts. The 44 retained entry points remain at their original paths.
The script inventory still contains 413 files, now with 44 directly under
`scripts/`. Current continuation, transfer/CI, model selection, training
configuration, and installation behavior were preserved.

| Destination | Moved scripts |
| --- | ---: |
| `scripts/training/imitation/` | 6 |
| `scripts/training/opponent-banks/` | 12 |
| `scripts/training/checks/` | 9 |
| `scripts/training/rules-host/` | 9 |
| `scripts/training/migrations/` | 2 |
| `scripts/training/checkpoints/` | 2 |
| `scripts/training/rust-lab/reference-capture/` | 6 |
| `scripts/training/rust-lab/fixtures/` | 4 |
| `scripts/training/rust-lab/validation/` | 14 |
| `scripts/benchmarks/` | 8 |
| `scripts/reports/` | 2 |
| `scripts/experiments/gumbel-is-mcts/` | 2 |
| **Total** | **76** |

Repository-root discovery, explicit and dynamic callers, test imports,
command examples, Markdown links, and new archive/source inventories use the
new locations. The retained `capture-rust-rules.py` needed two loader-path
revisions for the moved capture helpers; its function implementations and
top-level path remain the same. The other 43 retained scripts kept their
implementation-start bytes, including pre-existing working changes.

The exporter and calibration helper remain siblings. The Rust baseline loader
and new freeze inventories use their new paths. Existing baselines recording
`scripts/export-test-nn.py` resolve the current helper at its new location;
their original archive member and SHA-256 are still verified by the existing
provenance fallback. The fallback remains specific to the freeze-time exporter.
Executed oracle sources, checkpoints, and bundle metadata retain their exact
checks. No historical manifests or archive contents were rewritten.

Three Rust wildcard source lists preserve all 54 of their original script
members at the new paths, and the two explicit milestone lists preserve their
six and nine members. The native feature-table generator retains its original
generated header text so a location-only change does not alter the checked-in
table or the native fingerprint. Git's global `checkpoints/` ignore rule needed
a narrow exception for authored Python tools in `scripts/training/checkpoints/`;
generated model and metadata files there remain ignored.

| Verification | Result |
| --- | --- |
| Move inventory | All 76 old paths removed, all destinations present, all 44 retained paths present |
| Python syntax | All 73 relocated Python scripts and 18 Python callers compiled |
| Repository lookup | All 71 explicit Python root lookups resolve to the same checkout |
| Function/class comparison | 203 definitions match after translating relocated path strings; five archive-inventory functions were checked separately |
| PowerShell | Three relocated scripts and the related legacy builder parsed; repository and default game-root calculations verified |
| CLI help | All 64 outcomes and normalized text matched; 62 commands passed and two reproduced existing errors |
| Existing focused tests | 131 cases in 16 modules retained their outcomes: 21 existing errors and three existing failures |
| Frozen exporter compatibility | Eight new tests passed for original/relocated inventories, unchanged data, archive/member tampering, and retained oracle checks |
| Actual frozen reference archive | All 462 pinned members verified, including the original exporter |
| Rust source archives | All three wildcard sets and both explicit path lists preserved their original membership |
| Workflow indexes and links | Every migrated script indexed; previously present Markdown targets preserved |
| Git inclusion | All destination scripts and the new test are eligible for tracking; generated checkpoint data remains ignored |
| Step 5 implementation and CPU overlay | Both matched their implementation-start values |

The existing test errors include obsolete V11 fixture topology, older
configuration/cache expectations, missing g179 exports, and a missing legacy
profile in a disposable transfer fixture. Before/after case IDs, statuses, and
exception types match. One pre-existing feature-recipe assertion reports a
different first mismatching key because it iterates an unordered set; the
feature command's `main` implementation is identical. The two CLI errors are
the older scaling check's removed `collect_search_games` import and the
simulation profiler's caller-supplied Python import path. The moved profiler
also passed help with the normal qualifier-provided `PYTHONPATH`.

The implementation-start and post-migration fingerprints were
`a6e6eba62e4f6bc45c6c2170b223e09e6aa9240efb6b99f4f86f3561d9fc18f5`
for Step 5 and
`d9719f5192a81a37b6c4498e69e8938dd27041ac9261f5b344a31b117baead00`
for the CPU overlay. These use the working checkout at implementation time,
including unrelated training edits that were left untouched. Full native/CUDA
qualification, a real lab continuation, and game installation were not run for
this directory reorganization. Sealed historical workflows continue to use
their matching checkout and recorded source archives.
