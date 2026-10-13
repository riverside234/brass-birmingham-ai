# P0–P6 bug audit — 13 October 2026

Reviewed the active post–Step 5 implementation at starting HEAD `72bd43ff01a10a8d917705617b3805f6df41e924`, including the preceding P5/P6 fixes. **Twelve confirmed defects were fixed**, with 17 additional regression test methods. No additional independent P3 or P5 defect was confirmed. P5 was updated to enforce the corrected P4 evidence contract.

All 138 Personality tests pass. Fresh native, export, service, save and isolated installer checks pass. These results establish bounded interface and failure-path behavior using authored fixtures; they do not establish trained-model quality, qualified personalities, calibrated difficulty or release readiness.

## Scope and preservation

Parallel reviews covered P0/P1 handoff and continuation; P2 search, labels, readiness and P6 runtime; P3/P4 repertoire and qualification; and P5/P6 export, provenance, native bridge and persistence. Findings were reproduced before repair and the integrated source was frozen before final native staging and export. The root review checked the execution paths, integrated changes and independently validated the final evidence chain.

The working tree already contained unrelated joint7/recovery/configuration/report work. This audit did not edit or revert it. `preservation-before.json` records the starting status and 2,718 protected file hashes. The final comparison reports:

- The authorized `configs/personality/source-migrations.json` change.
- Concurrent changes to `docs/EXPORT_TRAINING_REPORTS.md` and `tests/Recovery/test_joint7_150_review.py`, plus new joint7 review evidence. These were not written by this audit and remain intact.
- No unexpected differences; all 1,766 recorded installed/guarded-host/training files retain their starting hashes.

No installed game binaries, frozen opponent banks, real datasets or checkpoints were changed. No commit, push, real installation, training campaign or strength tournament was performed. Installer tests wrote only a newly marked fixture sandbox under `artifacts/p6b/isolated-game`, then rolled it back. Preexisting upstream source changes remain present; the Step 5 fingerprint during the final smokes is `3de301188b2eb777a7929da14a3e8d3362f106e75e54129eef8776864809f8e6`.

Evidence below is local under [`artifacts/p0p6-audit-20261013`](../artifacts/p0p6-audit-20261013), except the short-path installer/RPC sandbox [`artifacts/p6b`](../artifacts/p6b). The shorter path avoids Windows runtime path limits. These private fixture artifacts include game dependencies and are not public redistribution material.

## Confirmed findings and repairs

Severity describes the consequence if the affected path is exercised. Integrity failures below concern mismatched or edited artifacts; content hashes are not publisher authentication.

| ID / severity | Confirmed defect and consequence | Repair and affected implementation | Before-fix evidence / regression |
| --- | --- | --- | --- |
| 01 / High | P0 checked copied file hashes but did not bind duplicated manifest claims to those snapshots. Altering training settings, fixture origin, parent generation or selection claims could survive load. | `personality/handoff.py` shares selection validation between inspect/load and checks required copied evidence, authoritative state/best/report/spec, parent configuration/provenance, references and declared trace receipts. Loading stays bundle-local. | `p01-before.log`; `test_p01_major_audit.py::test_handoff_claims_are_bound_to_copied_source_evidence` reproduces accepted altered claims. |
| 02 / Medium | P1 discarded saved RNG when changing from warm-up to joint mode, because RNG restore was coupled to same-phase Adam restore. Identical committed input and data could yield different updates depending on ambient process RNG. | `personality/checkpoint.py` exposes independent RNG restoration; `training.py` restores it on phase transition while retaining fresh Adam groups for the enlarged parameter set. | `p01-before.log`; `test_warmup_to_joint_uses_saved_rng_with_a_fresh_optimizer` checks deterministic weights and fresh optimizer behavior. |
| 03 / Medium | P2 registration/channel setup occurred outside cleanup; failures could leave pipes/shared inference allocations behind. A lane acquisition failure could also bypass session cleanup. | `personality/search/collection.py` retires the service after preparation errors and includes opponent endpoint acquisition inside session cleanup. | `p2-before.log`; real pipe fault injection and session-acquisition failure tests in `test_p2_major_audit.py`. |
| 04 / High | P2 could accept individually valid, rehashed teacher labels whose context, goal, profile or search recipe differed from the decision/collection they claimed to teach. This could silently train on misattributed targets. | `search/collection.py::iter_pairs` requires exact agreement with the executed sidecar and frozen collection specification. | `p2-attribution-before-head.log` uses the actual HEAD implementation and demonstrates the mismatch; context/goal/profile/search mutation regression. |
| 05 / High | A changed P2 readiness calibration could be rehashed and accepted without matching the checkpoint's saved calibration. Runtime goal bias could then use unsupported settings. | `search/readiness.py` binds calibration to checkpoint training metadata and validates complete finite metrics, allowed temperatures, support counts and readiness thresholds. | `p2-before.log`; saved-model receipt mutation and malformed numeric/support regressions. |
| 06 / Medium | P2 counted all setup groups for confidence calibration, including groups with every confidence label masked. One observed setup plus a masked setup could incorrectly pass the two-setup bias gate. | `search/readiness.py::confidence_calibration` counts only groups contributing observed confidence labels. | `p2-before.log`; 72 observed labels from one setup plus one fully masked setup must remain unready. |
| 07 / High | P4 roster selection dropped fixture provenance, and its checker had no real-artifact gate. A full authored fixture panel could receive `p4_qualified=True` when all other gates passed. P5 already independently rejected fixture parents, limiting propagation. | `deployment/roster.py`, `evaluation.py`, `contracts.py` preserve/recheck provenance, fail closed when it is absent and require `real_artifacts`. P5 `difficulty/panels.py` validates the new gate and report provenance. | `p34-before.log`; `test_p34_major_audit.py` isolates this gate with all previous gates passing, tests both origin/scope provenance and rejects cleared/missing fixture metadata. |
| 08 / High | P6 validated ONNX reward metadata against the package's reward, but both could agree on a different reward from the qualified roster's search. | `laptop/package.py` requires the exported reward to equal the roster search reward. | `p6-package-before.log`; `test_p6_major_package.py` changes both package/model metadata while preserving the roster. |
| 09 / Medium | P6 semantically validated the copied full roster without recomputing its content ID or checking its exact source contract. Edited full and compact profiles could retain an old roster identity. | `laptop/package.py` validates format, nested P4 contract and full roster digest before matching the compact roster. | `p6-roster-before-verified.log` runs the actual HEAD validator with a real profile change. The earlier exploratory `p6-roster-before.log` used a no-op mutation and is not defect evidence. |
| 10 / High | P6 did not carry/bind the readiness calibration receipt, so a rehashed package could use a different goal temperature while retaining its qualified roster identity. | `laptop/package.py` copies sealed `readiness.json` and validates its ID, current search contract, actor, reward, utility readiness and complete calibration against the roster/runtime recipe. | `p6-recipe-before.log`; changed temperature and substituted readiness receipt subcases in `test_p6_major_package.py`. |
| 11 / High | P6 did not require exported resource/hint profiles to equal the profiles in its P5 report. A different supported profile could retain the old difficulty evidence. | `laptop/package.py` requires exact profile/report equality, including fixture packages that carry a report. | `p6-recipe-before.log`; changed hint profile with an unchanged report is rejected. |
| 12 / Medium | A benchmark timer callback could run after `Runtime.plan` cleared its events but before timer retirement, leaving Play Now/cancel set for the next ordinary case. This could contaminate results and certification evidence. | `laptop/benchmark.py` clears controls before each case, after the previous timer has been cancelled and joined. | `p6-runtime-before.log`; `test_p6_major_runtime.py` deterministically schedules this race for both controls. |

The new package format is `personality-laptop-package/1.2.0`; `BrassPersonalityGame/Package.cs` accepts the matching version. P4 deployment, roster and roster-check formats are 1.1.0. The native save/RPC schemas, reward design, learning settings, budgets, workers and opponent schedules are unchanged.

One existing test harness also needed repair: the partial-channel-allocation test closed a Windows pipe before the multiprocessing feeder duplicated its queued handle. It produced a delayed `WinError 6` despite passing assertions. `test_p6_runtime_audit.py` now drains/closes the first queued duplicate before injecting the second allocation failure. All real retirement assertions remain; exceptions are not suppressed. The final full run is free of this diagnostic.

## Review coverage without an additional confirmed defect

- **P3:** masked temporal representation inputs, train-only grouped fitting, public observation boundaries, causal behavior effects, encoder refits/re-embedding, archive geometry, quality filtering and selection of actual tested controllers. Existing regressions passed; no new production change was justified.
- **P5:** panel job ownership and replay, derived report checks, source/evidence closure, stock-Hard baseline, profile/hint qualification and public intent handling. The P4 gate integration above is the only new production edit in this stage.
- **Native P5/P6:** state/save cloning, pending-action restoration, revision/history rewind, foreign-game rejection, exact current legal path validation before live mutation, public history reconstruction, own permitted information, cancellation and main-thread resource lifecycle. Three complete history games and focused native checks passed. Full interactive Unity acceptance remains untested.

## Verification

The baseline suite passed 121 tests. After repairs, the final clean suite passed **138 tests in 159.381 seconds**. An earlier integrated run also passed but exposed the test harness feeder race described above. The clean log contains no traceback, `WinError`, resource warning or failure summary. Existing ONNX tracer/deprecation warnings remain; actual exported fixture parity and dynamic-shape checks passed.

| Verification | Result / local evidence |
| --- | --- |
| Focused P0/P1 | 24 passed, 13.606 s; `p01-focused.log`. |
| Focused P2/runtime | 25 passed, 7.861 s; `search-runtime-after.log`; the seven new search/runtime methods also passed separately. |
| Focused P3/P4 | 32 passed, 16.721 s; `p34-final.log`. |
| Focused P5/P6 | 34 passed, 3.688 s; `p56-after.log`. |
| Final Python suite | 138 passed; `full-suite-clean.log`. |
| Historical checkpoint import/continuation | Actual frozen Git-source fixtures from both supported P0/P1 versions loaded with exact weights, Adam and saved RNG payloads. Copied pilots continued warm-up count 1→2; original fixture trees remained unchanged. `p01-migration-report.json`. This was CPU execution, not a CUDA continuation measurement. |
| P2/P3 native smoke | One complete reserved-teacher game, one complete direct discovery game, four paired initial-root probes, all four permitted trajectories, 120.609 s; `search-smoke/report.json`. |
| P4 native smoke | One complete both-era direct preview game, 1 of 104 planned jobs, 36.922 s. `real_artifacts=false`, P4/release qualification false; `deployment-smoke/checked/report.json`. |
| P5 native smoke | Two complete both-era direct fixture games including 93 stock-Hard actions, 44.765 s. No failures or outstanding admission slots; P5/release qualification false. `difficulty-smoke/smoke-report.json`. |
| Fresh staged helper | Current source/build validation passed. Initial native checks: 37 assertions. Final check using the actual packaged Python save: **44 assertions** (includes the initial 37), `native-save-python/report.json`. |
| Native save/history | Separate six save/clone/pending/rewind/foreign-game checks passed. Three complete four-player games: 993 decisions, 369 exact action-boundary history comparisons; `native-save-final/report.json`, `native-history-final/report.json`. |
| P6 export/runtime | Real float32 ONNX export and ORT CPU whole-turn plans at 2/5/12 s, legal paths, no deadline resets/overruns; `laptop-smoke/benchmark.json`. |
| Standalone RPC and sandbox lifecycle | Play Now, cancellation, exact save restoration, mandatory debt, seven native observation reconstructions, packaged interpreter preflight, dry run, marked-sandbox install and exact rollback passed; `artifacts/p6b/report.json`. |
| Final evidence closure | Current readiness, P4, P5, package and staged helper revalidated together; `final-validation.json`. Preservation comparison: `preservation-after.json`. |

Primary command for the complete Python suite, using the existing dependency environment on this Windows checkout:

```powershell
$env:PYTHONPATH="$PWD/src;$PWD/tests/R2;$PWD/tests/Recovery;$PWD/tests/Personality"
$env:TEMP="$PWD/artifacts/p0p6-audit-20261013/tmp"
$env:TMP=$env:TEMP
& artifacts/p6e/Scripts/python.exe -m unittest discover -s tests/Personality -p 'test_*.py' -v
```

No dependencies were installed for this review. The local environment used CPU PyTorch 2.8.0, ORT 1.22.1 and the existing native compiler/hosts. Commands for the isolated checks are the existing `smoke_search.py`, `smoke_deployment.py`, `smoke_difficulty.py`, `smoke_laptop.py`, `check_laptop_pipeline.py` and `check_laptop_native.py` scripts under `tests/Personality`; each used fresh output directories.

### Measured timing and its limits

The paired P2 12-second/four-worker probes completed 17 simulations without bias and 18 with authored bias: 0.777 versus 0.825 completed simulations/second. Final publication took 10.875–10.984 seconds with zero overruns; prior preparation took 2.2–4.6 ms. This four-root sample is noisy and establishes neither a speedup nor equal playing strength.

The separate P5 two-second turn produced a legal fallback and **zero completed simulations**, with no reported turn overrun. In the P6 ORT fixture, Easy/Medium/Hard completed **0/0/4** simulations; Easy and Medium used legal fallbacks. Admitted time was 1.813/4.547/10.922 seconds with zero deadline resets or turn overruns. End-to-end case time was 5.547/4.625/11.093 seconds; cold runtime startup was 3.453 seconds. These tiny authored models/positions and concurrent local checks are unsuitable for a backend or laptop throughput recommendation.

Standalone cold Play Now took **8.406 seconds end to end**, while its recorded admitted work was 0.250 seconds and planned work 0.390 seconds. The control was observed and returned a complete legal plan, but startup/preparation latency is material and is not covered by the admitted search clock. Cancellation took 0.375 seconds and preserved the save exactly; debt took 0.172 seconds without search or budget mutation. Representative warm/cold UI responsiveness still needs actual laptop acceptance. No production responsiveness claim follows from passing this fixture.

## Compatibility and continuation

### Existing Step 5 and P1 work

No Step 5/IL model or dataset migration is introduced by this audit. P0/P1 source identity is now:

`7857312a03491be8102a621826f1bcfc58018eaa4416f9dca26de5049ae54176`

The migration registry keeps the historical persistence receipt and adds two **direct, exact** `p0-p1-audit-continuation/1.0.0` receipts from:

- `3bb0eab8a753a5fb29e33b595e19e6ac0ab6d2e2d09875c059e93c9387bee068`
- `31b8d44872bbfcd9b89c8e3cc591ef38337fe39fbdebf64d4dc8a76ba46464ff`

Import checks the original model identity, requires the same model/goal/feature/base contracts and revalidates copied handoff evidence. It preserves compatible weights, saved optimizer/RNG and counters. No broad source whitelist or arbitrary migration chaining was added. JSON newline normalization does not alter parsed receipt digests.

To continue a completed supported pilot:

1. Stop all commands using the pilot. Preserve its handoff and run directory, including `state.json` and the immutable checkpoint referenced by `state.json.model_path`.
2. Update the audited source and `configs/personality/source-migrations.json` together. Keep the existing handoff/run paths and repeat the last successful P1 command with the same options. Do not rerun Step 5, erase state or manufacture a new handoff to hide a mismatch.
3. A same-phase continuation restores Adam/RNG. The first warm-up→joint transition restores RNG and intentionally initializes fresh Adam for the new parameter set. A successful update writes the new source/model identity and migration record; read-only import does not rewrite old artifact files.
4. Unsupported source versions or inconsistent copied evidence must still fail. Retain those artifacts and use their matching old checkout for recovery/review; do not edit source IDs, hashes or receipts to force admission.

For the documented lab joint pilot, when these are its existing paths/options, the continuation command remains:

```bash
conda activate ai-training
brass_root=/data/not_backed_up/yxu209/ai-training
cd "$brass_root"
python "$brass_root/scripts/personality.py" --device cuda joint-pilot \
  --handoff "$brass_root/artifacts/personality/handoff-v1" \
  --run "$brass_root/artifacts/personality/goal-pilot-v1" \
  --games 20 --workers 4 --epochs 2
```

Use the actual existing command's paths/options if different. For a continuing warm-up, repeat its `warmup` command instead; do not revert a joint pilot to frozen preparation. Interrupted uncommitted phases follow the existing [P0/P1 recovery procedure](PERSONALITY_P0_P1_LAB.md#5-checkpoints-traces-and-scope): preserve the orphan operation directory, keep the authoritative state/checkpoint pair and rerun after stopping all writers. This audit does not reconstruct bytes already lost by older persistence implementations.

### P2–P6 evidence and saved games

The P1 migration does **not** migrate old teacher labels, readiness, discovery archives, descriptors or qualification results. Regenerate downstream evidence under current source IDs. P4 1.1.0 requires the new fixture gate; dependent P5 contracts therefore change even though the P5 version remains 1.2.0. Re-export P6 as 1.2.0 and restage the helper after upstream qualification. Do not merely retag old bundles.

Existing immutable P6 bundles and saves remain untouched. Use an old bundle's matching bundled runtime/helper to validate or resume its pinned save. The current helper intentionally rejects older package formats. The save schema is still 1.0.0; this does not authorize transferring an old save to a new package identity or resampling its member.

Final local fixture package ID: `40e30e11e975091bb2145c12036435bb28504a100aeb4c39ddf517f1082533f5`. Staged helper ID: `19e6ca43020a992346ee274cd9b28383df35c1295b70002417643183f483114f`. Both remain unqualified fixtures.

## Remaining issues and limits

- **Unresolved budget specification:** active P2 `search/teachers.py` has `WALL_FRACTION=.9` and the lab guide describes 90%, while master plan §4.4 specifies 50%, matching Step 5. The audit preserves all current budgets and explicitly records the conflict in both documents. Resolve and version the intended P2 accounting policy before a real campaign; no guessed change was made.
- **Real artifacts unavailable locally:** no accepted lab parent was imported, no real 16×8 repertoire/difficulty panel was qualified, and no human recognition/tactical evidence was created. Authored fixtures must not become training, qualification or release evidence.
- **Hardware/platform scope:** Windows CPU and native .NET execution only. Linux/Mono, CUDA continuation execution, multi-GPU, Intel/OpenVINO and long-running resource behavior were not measured. Saved CUDA RNG payload preservation is not proof of CUDA execution parity.
- **Unity acceptance remains open:** interactive menu layout, full mixed/neural play, pending-action save/load, history rewind, cancellation and presentation/resource restoration need review in an isolated running Unity copy. No interactive game was launched here.
- **Performance:** cold preparation latency and short-budget fallback rates above need representative laptop testing with real models. No performance optimization or hyperparameter change is justified by these fixtures alone.
- **Crash durability:** ordinary error cleanup and existing atomic pilot publication are tested. Forced termination at every installation/publication instruction, power loss and recovery from prior already-corrupted artifacts are not covered.

The review found and repaired specific remaining defects. It does not establish that P0–P6 is bug-free or ready for release.
