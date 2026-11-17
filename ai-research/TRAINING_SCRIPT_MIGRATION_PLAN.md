# Training script migration plan and results

Prepared on 2026-11-17 against checkout `0a39492`.

The migration moved all 94 listed scripts, with 54 in
`scripts/training/migrations/` and 40 in `scripts/legacy/rust-lab/`.
The legacy inventory retains all 40 files. Callers, test imports, source lists,
shell launchers, and workflow indexes use the relocated paths.

The scope below records the approved plan, including its deletion option.
Revised to include retirement of the deleted lab bundle workflows. The proposed
pass moves **54 historical training scripts** into `scripts/training/migrations/`
and **40 retired lab scripts** into `scripts/legacy/rust-lab/`, for **94 proposed
moves**. Deletion is authorized for an obsolete bundle script when preserving
it would require substantial work or reconstruction of the removed bundle.
Record each deletion against the proposed move list.

## Evidence and selection

The local report
`joint7-v11-2-g1190-restart-3-g2990-20261117T053650589389Z` records generation
2990 for `artifacts/step5/joint7-v11-2-g1190-restart-3`. Its state and export
manifest agree on the committed generation and implementation. The continuation
scripts, their dynamic imports, test callers, source inventory, and transfer
workflow were inspected for this plan.
[g2990 export](../results/joint7-v11-2-g1190-restart-3-g2990-20261117T053650589389Z/export-manifest.json) ·
[g2990 state](../results/joint7-v11-2-g1190-restart-3-g2990-20261117T053650589389Z/state.json)

The existing migration index lists 60 scripts. This plan selects 47 older
configuration commands and seven earlier restart commands for the training
folder, and three old bundle builders for the legacy folder. Three transfer or
in-place rollback commands are deferred. The three lab sections identified for
cleanup contain another 43 files, of which 37 serve retired workflows and
six still supply shared functions to current commands.
[Current migration index](../scripts/training/migrations/README.md)

A script's filename or generation number alone does not establish that it is
inactive. For example, `continue-rust-g1491.py` is loaded by the current
`continue-rust-g1190-restart-2.py`, and its shared continuation loads
`rust_lab_results.py`. Moving an old CLI also leaves its shared recovery
implementation in the training source inventory.
[Current launcher](../scripts/continue-rust-g1190-restart-2.py) ·
[Shared continuation](../scripts/continue-rust-shadow.py) ·
[Training source inventory](../src/brass_ai/learning/source_identity.py)

## Proposed layout and exact move list

The following five groups originally lived directly under `scripts/`.
Their destination is `scripts/training/migrations/<same-filename>`.
The retired bundle list later in this plan goes to `scripts/legacy/rust-lab/`.

| Group | Files | Migration work |
| --- | ---: | --- |
| Saved-generation recipes | 15 | Root lookup, test imports, command examples |
| Joint-7 continuation recipes | 16 | Root lookup, dynamic test imports, help references |
| PPO, PPG, and self-play configuration | 12 | Root lookup and focused fixture callers |
| Teacher, scoring, and evaluation configuration | 4 | Root lookup and one explicit bundle source list |
| Earlier restart commands | 7 | Root lookup, shared-driver lookup, restart fixtures |
| **Training migration subtotal** | **54** | |

### Saved-generation recipes

```text
configure-training-g370.py
configure-training-g570.py
configure-training-g611.py
configure-training-g760.py
configure-training-g850.py
configure-training-g950.py
configure-training-g1041.py
configure-training-g1050.py
configure-training-g1120.py
configure-training-g1241.py
configure-training-g1247.py
configure-training-g1320.py
configure-training-g1322.py
configure-training-g1324.py
configure-training-g1340.py
```

### Joint-7 continuation recipes

```text
joint7-entropy-groups.py
joint7-entropy95.py
joint7-exploration.py
joint7-g110-hotfix.py
joint7-g503.py
joint7-g503-ppg.py
joint7-g540-ppg.py
joint7-panel-gates.py
joint7-ppg-eight.py
joint7-ppg-window.py
joint7-ppo-only.py
joint7-qualified-anchor.py
joint7-snapshot-r2.py
joint7-throughput.py
joint7-v4-share.py
joint7-vp-cap.py
```

### PPO, PPG, and self-play configuration

```text
configure-dcpg.py
configure-entropy-schedule.py
configure-gae-floor.py
configure-pfsp.py
configure-ppg-eight-fast.py
configure-ppg-g210.py
configure-ppg-phasic.py
configure-ppg-sixteen.py
configure-ppg-throughput.py
configure-ppo-lr-threshold.py
configure-ppo-probes.py
configure-selfplay-throughput.py
```

### Teacher, scoring, and evaluation configuration

```text
configure-domino.py
configure-domino-context-evaluation.py
configure-local-scoring-aux.py
configure-teacher-phase.py
```

### Earlier restart commands

```text
continue-rust-g1190-restart.py
continue-rust-g1490-restart.py
prepare-neutral-selfplay.py
prepare-rust-branch-restart.py
prepare-rust-g1190-restart.py
prepare-rust-g1190-restart-2.py
prepare-rust-g1490-restart.py
```

The seven restart commands are launch or preparation interfaces around shared
code. The two older continuation wrappers can move while continuing to load
the top-level `scripts/continue-rust-g1491.py`. The original
`prepare-rust-g1190-restart-2.py` created the seed for this run and has a
separate preparation role from the current continuation command.

## Retired lab bundle workflows

The three named lab sections are included in this pass. Move the 37 retired
entries below, together with the three older generation-specific bundle
builders, into **`scripts/legacy/rust-lab/<same-filename>`**. This keeps the old
bundle commands and their helpers in one legacy folder.

| Original group | Move to legacy or delete if impractical | Shared files retained |
| --- | ---: | ---: |
| Lab bundle preparation, transfer, and recovery | 12 | 0 |
| Lab launchers and qualifications | 16 | 3 |
| B06 and B07 qualification helpers | 9 | 3 |
| Generation-specific bundle builders from the migrations index | 3 | 0 |
| **Total** | **40** | **6** |

### Bundle preparation, transfer, and recovery

```text
prepare-rust-lab-b01.py
prepare-rust-lab-b01-transfer.py
prepare-rust-lab-b02.py
prepare-rust-lab-b025.py
prepare-rust-lab-b03.py
prepare-rust-lab-b04.py
prepare-rust-lab-b05.py
prepare-rust-lab-b06.py
prepare-rust-lab-b07.py
prepare-rust-lab-results.py
restore-rust-lab-b03.py
rust_lab_toolchain.py
```

### Retired launchers and qualifications

```text
continue-rust-lab-b07.py
manage-rust-lab-b07.py
qualify-rust-lab-b025.py
qualify-rust-lab-b03.py
qualify-rust-lab-b04.py
qualify-rust-lab-b045.py
qualify-rust-lab-b05.py
qualify-rust-lab-b06.py
qualify-rust-lab-b07.py
run-rust-lab-qualification.py
run-rust-lab-b01-b02.sh
run-rust-lab-b01.sh
run-rust-lab-b02.sh
run-rust-lab-b025.sh
run-rust-lab-b03.sh
run-rust-lab-b04.sh
```

### Retired B06 and B07 helpers

```text
rust_lab_b06_ordinary.py
rust_lab_b06_teacher.py
rust_lab_b07_publication.py
rust_lab_b07_r3_publication.py
rust_lab_b07_r4_publication.py
rust_lab_b07_sealing.py
rust_lab_b07_smoke.py
rust_lab_b07_transport.py
rust_lab_b07_v4.py
```

### Older generation-specific bundle builders

```text
prepare-rust-lab-g370.py
prepare-rust-lab-g570.py
prepare-rust-lab-g1250.py
```

These three builders leave the deferred list. Their historical bundle
publication requirements are grounds for retiring a script when relocation is
impractical.

### Shared functions still used by current commands

The following six files from the named sections stay at their existing paths
for this pass because live commands load their functions.

| File under `scripts/` | Current caller or dependency |
| --- | --- |
| `rust_lab_results.py` | `continue-rust-shadow.py` calls `export()` after continuation. CPU preparation, qualification, and performance tools also use it. |
| `qualify-rust-lab.py` | CPU preparation uses `python_environment()`, `prepare_test_inputs()`, and `command()`. |
| `qualify-rust-lab-b02.py` | CPU preparation calls `check_shared_dependencies()`. Its old B02 qualification entry point shares this file. |
| `rust_lab_b06_metrics.py` | CPU inference qualification imports `Meter`. Collector profiling also uses its comparison functions. |
| `rust_lab_b07_retry.py` | Collector, teacher-verification, and post-transfer reward preparation call its runtime builders. |
| `rust_lab_b07_completion.py` | The retry helper imports `reuse_build()` when verifying runtime builds. |

`rust_lab_toolchain.py` is included in the legacy moves. Its compiler restore
is called by the historical B02/B04.5 bootstrap-build paths. The current CPU
builder uses the dependency check from the B02 module and retained bootstrap
pins.
[CPU build calls](../scripts/prepare-rust-cpu-inference.py) ·
[Evidence export](../scripts/continue-rust-shadow.py) ·
[Runtime build and verification](../scripts/rust_lab_b07_retry.py)

The old transfer ZIPs have been removed from this checkout. Two payloads still
exist locally and have distinct uses. CPU rebuilds extract
`distribution/rust-lab-g1250/inputs.zip`, which carries the sealed native
source and fixtures. The historical compiler restore opens the Roslyn package
under `distribution/rust-lab-toolchain/`. Move or retire the obsolete
publication scripts independently of these payloads. If the native input ZIP
is also missing on the lab, CPU rebuilding needs a verified source input, even
when an already prepared worker can continue training.
[Native input extraction](../scripts/prepare-rust-cpu-inference.py) ·
[Compiler restore](../scripts/legacy/rust-lab/rust_lab_toolchain.py)

### Caller updates and authorized deletion

- Update root discovery in moved Python files to `parents[3]`, and fix the
  shell launchers' root discovery and command paths. Preserve sibling imports
  within the legacy group and explicit calls into retained shared helpers.
- Update `qualify-rust-lab-b02.py`'s bootstrap helper to load the relocated
  `rust_lab_toolchain.py` when retaining the historical bootstrap workflow.
  Its CPU dependency-check function keeps its existing behavior.
- Update the retired command paths used by the B07 workflow identity function
  in `rust_lab_b07_completion.py` if that workflow is retained. Its live
  `reuse_build()` function keeps its receipt verification.
- Update dynamic loaders and imports in affected tests and source inventory
  assembly. The collector benchmark imports `freeze_jobs` and `prepared_rows`
  from `rust_lab_b06_ordinary.py`, and teacher-verification tests import
  `replay_stream` and `summarize_turns` from `rust_lab_b06_teacher.py`.
  These callers must load the relocated helpers. Preserve these functions if
  a surrounding retired workflow is deleted.
  [Collector benchmark](../scripts/benchmarks/profile-rust-collector-performance.py) ·
  [Teacher verification tests](../tests/RustBackendLab/test_shadow_verification.py)
- When a script serves only a deleted bundle and restoring its old workflow
  would require substantial work, delete it and update its obsolete callers,
  tests, and command index. Record the reason beside its filename. Existing
  tests of functions used by current preparation or profiling remain relevant.
- Treat the surviving source receipts as recorded evidence. Recreating deleted
  transfer bundles or replaying the full B01–B07 qualification sequence is
  outside the work needed to retire those workflows.

The target is 94 moves before any authorized deletions. Report actual moves
and deletions separately, with every one of the 40 retired files accounted for.
The six shared helpers can be considered for a later coordinated move when
their current callers are included.

## Retained entry points and dependencies

| Paths retained at their current locations | Reason for retention in this pass |
| --- | --- |
| `continue-rust-g1190-restart-2.py`, `continue-rust-g1491.py`, `continue-rust-shadow.py`, `step5.py` | Current continuation chain and trainer entry point |
| `continue-rust-teacher.py`, `recover-step5-iteration.py`, `resolve_checkpoint.py`, `joint7_preflight.py` | Supported continuation, recovery, and preflight entry points |
| `step5-transfer.py` and `step5_transfer_*.py` | Machine transfer, sibling imports, and the GitHub release workflow |
| `build-training-host.py`, `build-search-host.py`, `build-local-scoring-host.py` | Explicit paths in the source inventory and runtime build workflows |
| CPU preparation and qualification commands and the six shared files listed above | CPU worker rebuilds, continuation exports, qualification, and runtime preparation still use these functions |
| Other training, opponent-bank, IL, and host commands | Outside the initial historical migration selection |

Three entries from the existing migration index are deferred.

| Script or group | Concrete dependency or additional verification needed |
| --- | --- |
| `prepare-rust-post-transfer-reward.py` | `step5-transfer.py` emits this command for reward-revision restores. The reward runtime also names it, and it imports the B07 retry helper. |
| `continue-rust-revised.py` | Performs an in-place restore, preserves the later run by rename, and invokes transfer repair and teacher continuation. A later move needs its complete restore and retry fixtures. |
| `joint7_restart160.py` | Reconstructs an audited older boundary and publishes restored run data. Its direct Python import and rollback fixtures warrant a separate batch. |

The 40 obsolete bundle scripts are covered by the legacy migration and deletion
list above. Deleting their former transfer bundles is sufficient reason to
retire publication-only tools. Source receipt paths do not require keeping
those obsolete tools at the top level.
[Transfer commands](../scripts/step5-transfer.py) ·
[CPU worker preparation](../scripts/prepare-rust-cpu-inference.py) ·
[Rust lab index](../scripts/training/rust-lab/README.md)

## Required path and caller revisions

1. Resolve the checkout from the new directory. For a file in
   `scripts/training/migrations/`, use
   `Path(__file__).resolve().parents[3]` wherever the original
   `parents[1]` resolved the repository. Keep existing `--root` semantics
   and repository-relative configuration, artifact, and run paths.
2. In the two moved continuation wrappers, replace the sibling
   `Path(__file__).with_name('continue-rust-g1491.py')` lookup with an explicit
   path to `scripts/continue-rust-g1491.py` from the resolved checkout.
   Preserve the run defaults, required share target, teacher defaults, and
   arguments forwarded to the shared driver.
3. Update test imports that open a CLI by its old path. This includes the
   generation-specific tests and the DCPG, entropy, PPG, sparse PPO,
   DOMiNO, throughput, and policy-anchor fixtures. Check dynamic path builders
   as well as literal filenames.
4. Update `helper(name)` in `test_entropy_group_revision.py`. In
   `test_skip_utility_qualification.py`, adjust only the preparation-script
   lookup, because the same helper also loads retained continuation scripts.
   Preserve its check of the original g1190 weight selection and g1491 seed.
   [Entropy test helper](../tests/Recovery/test_entropy_group_revision.py) ·
   [Restart and continuation fixture](../tests/Recovery/test_skip_utility_qualification.py)
5. If retaining `prepare-rust-lab-b025.py` in the legacy folder, update its
   source list, which explicitly includes `scripts/configure-teacher-phase.py`.
   If retiring that publisher, remove its obsolete command references.
   Existing receipt hashes retain the source paths from the recorded checkout.
   [B02.5 source list](../scripts/legacy/rust-lab/prepare-rust-lab-b025.py)
6. Update runnable examples, relative Markdown links, and commands printed in
   help or errors. This includes the entropy95 handoff to entropy-groups and
   the restart preparation/continuation references. Update
   `scripts/README.md`, `scripts/training/README.md`, the migrations index,
   `scripts/training/rust-lab/README.md`, and `scripts/legacy/README.md`.
   The three named lab sections should link to the shared functions and the
   legacy inventory with each retired or deleted command identified.
   Preserve original paths inside historical evidence.

Keep the shared `src/brass_ai/recovery/` modules at their current paths. They
implement behavior used by later training as well as by the older CLIs.
The planned edits concern script location and callers. Existing CLI flags,
defaults, output schema, generation checks, locks, preview/apply behavior,
and restart weight selection remain the behavior to verify.

## Implementation order and verification

1. **Record the baseline.** Recheck the g2990 continuation dependency chain
   against the checkout used for implementation. Record the current Step 5
   implementation ID, CPU overlay fingerprint, and relevant test results.
   Compare these identities before and after the move within that checkout.
   Record any pre-existing difference from the g2990 export separately.
2. **Pilot two configuration scripts.** Move
   `configure-training-g1241.py` and `configure-ppg-phasic.py`, fix their
   imports and focused tests, and compare generated configuration JSON using
   identical disposable input fixtures.
3. **Complete the 47 configuration commands.** Move each listed group with
   its callers. Test the `--apply` hotfix commands against disposable run
   fixtures, retaining their exact changes, repeat-call behavior, and stale
   generation rejection. The 47 count includes the two pilot files.
4. **Move the seven restart commands.** Check shared-driver loading and
   forwarded arguments without launching training. Use the existing
   `test_fresh_neutral_selfplay.py`, `test_branch_restart.py`,
   `test_g1190_restart.py`, `test_g1490_restart.py`, and
   `test_skip_utility_qualification.py` coverage for preparation behavior.
   Exercise CLI preparation through mocked or disposable fixtures.
5. **Retire the 40 bundle scripts.** Move each retained legacy file with its
   current test and profiling callers. Exercise shared functions and the
   relevant existing fixtures. Use the authorized deletion option for
   obsolete bundle-only workflows that are impractical to migrate. Verify
   dynamic imports from current CPU preparation, qualification, and
   continuation independently of old bundle publication.
6. **Finish the inventory and docs.** Account for all 94 candidates as moved
   or deleted, with every moved path indexed. Search by full path, filename,
   module name, and dynamically constructed names. Check the updated indexes
   and source lists for missing files.

For each batch, parse the moved Python files and run supported `--help`
commands from the checkout root and another working directory, including a
path with spaces. Run the affected existing tests and compare outputs against
the same fixtures before the move. Existing historical test failures should
be documented with their original-path reproduction. For retained legacy
shell scripts, check syntax and command resolution. Obsolete bundle
qualification fixtures can be retired with their deleted workflow, while
tests of shared CPU, transport, export, and profiling behavior follow their
current callers.

Use fixture data for configuration application and restart preparation.
The local results export supplies reference metadata, while a real training
resume requires the lab's full checkpoint and admitted runtime. Local checks
can verify script discovery and behavior without treating the export as a
resumable checkpoint.

Before completing implementation, verify that the Step 5 implementation ID
and CPU overlay fingerprint match their baseline values, and that the active
continuation, transfer commands, and CI paths still resolve. A layout-only
move of these scripts should not require adopting a training-source or
recipe transition.

Each batch can revise its file selection or destination when a concrete
dependency warrants it. Update this plan with the actual moved count, retained
paths and reasons, test outcomes, and any verification requiring the lab.

## Migration results

Completed on 2026-11-17. All 94 files were moved, with 54 under
`scripts/training/migrations/` and 40 under `scripts/legacy/rust-lab/`.
The six shared lab helpers retain their callable top-level paths. The two
shared helper edits update the legacy compiler locator and B07 workflow source
paths. Their CPU dependency checks and runtime receipt verification functions
retain their original implementation.

| Verification | Result |
| --- | --- |
| Focused configuration and restart tests | 43 cases matched the original run, including 34 existing errors |
| Lab helpers, exports, transport, CPU, and teacher startup tests | 317 cases matched the original run, including seven existing errors and 45 skips |
| CLI help | All 75 command outcomes and normalized help text matched the baseline |
| Legacy diagnostic environment | The two diagnostics that require the caller's Python import path loaded with that environment |
| Bash launchers | All six passed syntax checks and stubbed repository-root and command-path checks |
| Configuration and preparation implementations | All 95 function/class definitions in 52 relocated scripts matched the original AST, with the same resolved checkout |
| Historical continuation wrappers | Both forwarded their original defaults and arguments to the retained g1491 driver |
| Script inventories, Markdown links, and Python syntax | Passed |
| Step 5 implementation ID and CPU overlay fingerprint | Matched their pre-migration values |

The 360-test comparison used the same fixture selection before and after the
moves. Error case IDs and exception messages matched, allowing for generated
temporary directory names. The existing errors include old source-transition
checks, missing historical inputs, and older API expectations. The skipped
cases require native workers, CUDA, a hosted game/bootstrap, or Windows symlink
support. Full native/CUDA training validation requires the corresponding lab
runtimes.

The recorded local fingerprints are
`734c709512a95d5860dcc2294635737a4bf1931f0d1d96c322d3c496cea7a6a0`
for Step 5 and
`08d7dc7a998db1336570dbb7bdb620f4ec97deb143b5a367f1c3419e1ad3e16e`
for the CPU overlay.

