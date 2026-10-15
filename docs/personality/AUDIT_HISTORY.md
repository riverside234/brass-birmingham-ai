# P0-P6 audit history

[Documentation index and required follow-up](README.md) | [Design](DESIGN.md#design) | [Audit history](AUDIT_HISTORY.md)

Historical evidence, preserved by date. These reports describe the source revisions and fixtures they reviewed; their test counts are not new results from this documentation reorganization.

Reviews: [11 October: P0-P4](#audit-2026-10-11-p0-p4) | [12 October: P5/P6](#audit-2026-10-12-p5-p6) | [13 October: P0-P6](#audit-2026-10-13-p0-p6).

<a id="audit-2026-10-11-p0-p4"></a>

## P0–P4 correctness review — 11 October 2026

Reviewed the handoff, auxiliary goal training, search/teacher integration, learned repertoire and unified deployment workflow. Confirmed defects were fixed in the personality packages and covered by focused regressions. This was a code/data-integrity review, not a strength or 16-style qualification campaign.

<a id="audit-2026-10-11-p0-p4-findings-and-fixes"></a>

### Findings and fixes

| Stage | Issue and consequence | Correction |
| --- | --- | --- |
| P0 | No confirmed new defect in stopped-run admission, accepted-parent selection, pinned evidence or compatible weight import. | Existing rejection/restore tests passed. The P1 source fix changes the shared personality fingerprint, so create a fresh handoff. |
| P1 | Matching model IDs alone could let externally selected actions or a different sampling temperature be treated as on-policy PPO data. | Require recorded model sampling at temperature 1 with no teacher before assigning personality collection rows. Legacy auxiliary-only input remains separate. |
| P2 | `prepare --distilled` could attribute an older teacher branch to a different pilot or newer generation sharing the same Step 5 parent. This could pair incompatible weights and PPO state. | Verify the pinned originating readiness bundle, exact pilot path and pilot model hash before importing the distilled weights. |
| P2 | A forced first action could defer goal generation to a payment subchoice or the unsupervised second action. | Only the first action root proposes a goal. A forced root retains any surviving intent and defers fresh proposals until the next turn. |
| P3 | Public effects were measured until the next own turn, misattributing opponent consumption/flips and crossing the Canal clear. | Aggregate adjacent observations after own choices; censor unavailable terminal/era-clear effects with explicit coverage/boundary fields. |
| P3 | Merchant beer was compared with bare merchant IDs, although native choices use `merchant:`-prefixed targets. | Count the actual native target format. |
| P3 | Archive refits could silently replace retained source receipts before re-encoding. | Verify the previous archive's pinned manifest and receipt hashes before reusing its evidence. |
| P3 | Large proposals were allowed, but an entire proposal had to fit within the 4,096-game collection limit. A 2,048-candidate proposal could not be collected even with one setup. | Add bounded candidate slices preserving the full proposal, controller IDs and matched setup/seat/opponent/sampling streams. Only the slice containing candidate zero collects the neutral control. |
| P4 | Neighborhood selection compared the nearest distance with the largest unrelated pair's noise, and could stop before trying another valid candidate. | Check every candidate against each selected member's own pairwise noise threshold, then choose among remaining compatible neighbors. |
| P4 | Blind examples all came from the first setup and seat; partial chunks could change the chosen examples. | Freeze outcome-independent cases from the planned jobs, cover every member and balance setups/seats. Missing planned cases stay missing. |

No reward coefficients, PPO settings, search budgets, worker counts, release thresholds or installed game code changed. P3 also avoids extracting unused opponent trajectories when only active candidate records are requested.

<a id="audit-2026-10-11-p0-p4-compatibility-and-lab-use"></a>

### Compatibility and lab use

The Step 5 implementation identity remains `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`. Existing Step 5 checkpoints, main self-play, IL handoff and base generated-data formats are unchanged. The personality additions remain separate post-Step-5 commands.

Personality contracts intentionally change when their source changes. After updating, finish Step 5 and create a **new P0 handoff → P1 pilot → P2 readiness/teachers → P3 discovery/descriptor/archive → P4 roster/panel** in new directories. Do not rewrite old hashes or reuse qualification receipts for changed code. The new P0/P1 source identity is `3bb0eab8a753a5fb29e33b595e19e6ac0ab6d2e2d09875c059e93c9387bee068`.

P3's derived schema is **`public-behavior-trace/1.1.0`, 32 turns × 95 fields**, previously 93 fields. The additional `effects_observed` and `era_boundary` fields distinguish missing/censored consequences from observed zero effects. Old descriptor weights, latent coordinates and downstream rosters require regeneration; no converter pretends they are equivalent. This does not change the underlying game's replay format.

Instructions: [P0/P1](TRAINING_P0_P3.md#p0-p1), [P2/P3, including candidate slicing](TRAINING_P0_P3.md#p2-p3), [P4](QUALIFICATION_P4_P6.md#p4). For 2,048 candidates and two discovery setups, use four 512-candidate slices with the same seed and distinct output directories. For eight validation setups, use 128-candidate slices. Supply all matching slices to fitting/archive construction; duplicates are rejected.

<a id="audit-2026-10-11-p0-p4-verification"></a>

### Verification

- **59 focused tests passed**: 17 P0/P1, 14 P2, 12 P3 and 16 P4. This includes 12 additional regression methods plus expanded existing coverage. The combined run took 84.5 seconds.
- A **single native four-player direct-preview game** completed both eras through the actual rules host, collection, updated 95-field extraction and P4 report generation. It correctly remained incomplete and unqualified. The fixture report is `artifacts/personality/p0-p4-review-20261011/smoke-report.json`.
- That smoke took **72.5 seconds while the test suite ran concurrently**. This is not a search/GPU throughput comparison or evidence of a performance regression.
- Independent cross-review checked P4's pairwise filtering and frozen blind selection, and the P1/P2/P3 interactions. The new P4 regressions also passed independently.

Reproduce the focused tests with the project's dependencies available:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
export PYTHONPATH="$PWD/src:$PWD/tests/R2:$PWD/tests/Recovery:$PWD/tests/Personality${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest test_personality test_review_p01 test_search test_search_training test_repertoire test_deployment -v
```

<a id="audit-2026-10-11-p0-p4-remaining-limits"></a>

### Remaining limits

The actual trained parent is still on the lab. This review cannot establish goal calibration, preserved strength, human-recognizable styles, stable 16-by-8 neighborhoods or CUDA utilization. Small authored fixtures must not supply those qualifications. P4 remains an intermediate evidence gate; P5 difficulty calibration and P6 laptop deployment are later work. Era-clear and final-action consequences without an observable successor remain explicitly censored rather than reconstructed from hidden state.

<a id="audit-2026-10-12-p5-p6"></a>

## P5/P6 implementation audit — 12 October 2026

Base: `05def96` (`Add personality difficulty and laptop runtime`). The checkout was clean at the start. This follow-up reviewed the implemented P5 calibration/evidence pipeline and P6 export, persistent service, native game bridge, saves and deployment checks. It does not repeat the earlier whole-training audit or qualify a model for release.

Eight confirmed defects were repaired. Separate reviewers owned P5, P6 runtime and native code; the integrating reviewer checked their execution paths and evidence. All experiments used authored fixtures or existing pinned native dependencies. No training campaign, real qualification study or live installation ran.

<a id="audit-2026-10-12-p5-p6-confirmed-findings-and-repairs"></a>

### Confirmed findings and repairs

| Severity | Defect and execution path | Repair and regression evidence |
| --- | --- | --- |
| High | **A new neural game could fail at its first save.** `PersonalityState.Attach` sealed an envelope without outer `game_id`. The existing `SaveExtensions.Merge` then added it. `LocalMenu.StartGame` immediately saved and called `LocalGameState.Read`, which rejected the changed seal. | Include the stable metadata game ID before sealing in `src/BrassPersonalityGame/State.cs`. The actual Attach → Merge → Read sequence reproduced `Neural save integrity differs` before repair. Native regressions cover roundtrip, unchanged legacy/envelope data, pending-plan clone/load, rewind and foreign-game rejection. |
| High | **Failed/crashed P5 jobs could be retried under another output directory.** `collection.execute` reserved only its chunk; the verifier saw only supplied chunks. Omitting the original attempt allowed replacement evidence for the same planned job. | A sealed panel-owned `reservations.json`, updated under the existing OS-backed run lock, assigns every job to its immutable chunk before collection. Loading checks exact ownership and reserved job coverage. Tests cover crash retry, overlap, concurrent claims, disjoint continuation and copied-chunk rejection. |
| High | **P5 reports accepted contradictory derived summaries.** `evaluation.load_report` recomputed metrics/gates but accepted resealed changes to failures, stock failures, timing and environment. Release planning trusted the editable failure list. | Recompute all those summaries and source receipts from pinned records. Release planning/loading uses the verified `no_failed_games` gate. Tests first accepted the damaged reports, then rejected them while retaining a valid roundtrip. |
| High | **A P6 package could claim the qualified rollout pool while using different ordered reference model IDs.** Runtime uses `models[1:]`; package validation previously checked only the actor against the roster. | Bind the digest of the ordered reference IDs to both manifest and roster pool IDs. Matching exports remain valid; reordered, substituted and mismatched pool fixtures fail. Existing graph metadata/hash checks remain intact. |
| High | **Certification could credit controls that never occurred, and overlook Play Now clock violations.** A benchmark timer cancelled after normal completion still left `interrupt='play-now'`. The certificate checked its label/status but not observed metrics; clock gates covered ordinary rows only. | Require observed cancellation/Play Now booleans and the corresponding status. Apply the no-reset/50 ms overrun gate to successful Play Now turns too. Positive synthetic evidence passes; unobserved controls, reset deadlines and excessive overruns fail without producing a certificate. |
| Medium | **P5 inference preparation failures bypassed cleanup.** Registration/channel allocation preceded the service context; an exception could leave owned queues, pipes and shared buffers open. | Protect preparation and close the service on failure in `difficulty/collection.py`. Injected channel-allocation failure reproduced missing cleanup before repair. |
| Medium | **Failed P6 forest creation or partial channel allocation leaked persistent inference resources.** Closing acquired worker endpoints alone left service-owned lanes alive, and the server retained the failed runtime for retry. | Retire the entire runtime on either failure, close native/inference resources with idempotent cleanup, and create a fresh runtime for the next request. Tests use real portable channel allocation with injected native/allocation failures; native workers/model execution are mocked in these specific failure tests. |
| Medium | **Non-object JSON killed the persistent service.** Handling `[]`, `null`, a number or a string called `.get` both in dispatch and its error handler. | Validate the decoded envelope type before assigning it to the error-response variable. The regression sends all four malformed values followed by a successful `hello`; diagnostics remain redacted. |

P5 admission/verification is explicitly versioned as `personality-difficulty/1.2.0`; reservations use `personality-difficulty-reservations/1.0.0`. P6 package/save schemas remain unchanged; exact source identities change and force fresh packages/staged helpers. No reward, objective, architecture, training schedule, worker count or search budget changed. P5/P6 still use two workers and 2/5/12-second whole-turn allowances.

<a id="audit-2026-10-12-p5-p6-verification"></a>

### Verification

Evidence is under [`artifacts/p56-audit-20261012`](../../artifacts/p56-audit-20261012), except the short Windows-path pipeline sandbox `artifacts/p6a`.

| Check | Result / evidence |
| --- | --- |
| Initial P5/P6 Python baseline | 41 tests passed; `baseline.log`. |
| Focused P5 repair checks | 27 passed; `p5-after.log`. Before-fix evidence: `p5-before.log`, `p5-preparation-before.log`. |
| Focused runtime and save checks | 8 passed; `p6-runtime-final.log`. Before-fix evidence: `p6-runtime-before.log`, `p6-channel-before.log`. |
| Package/certificate and existing installation regressions | 6 passed; `package-after.log`. Before-fix evidence: `package-before.log`, `certification-clock-before.log`. |
| Complete personality Python suite | **121 passed in 97.494 seconds**, including P0 checkpoint/optimizer/counter restoration and P1–P4 contracts; `personality-suite.log`. |
| Fresh isolated native helper | Compiled and passed current source identity/file checks; `native-build-final.log`, `native-verification-final.json`. |
| Actual native save hooks | Six checks passed on the final helper; `native-save-final/report.json`. Before-fix product failure: `native-save-repro.log`. Initialization RPC is stubbed in this test. |
| Existing native bridge checks | 37 assertions passed without optional Python save input; seven observation/choice cases in `native-final/report.json`. Repeating with the actual Python pipeline's saved state passed **44 assertions**; `native-save-python/report.json`. |
| Complete native public-history parity games | Three games, 993 legal decisions, 369 checked completed-action boundaries, including both eras and sale/bonus/debt paths; `native-history-repro/report.json`. |
| Original Hard P5 native bridge | Three complete games, 377 stock actions including five liquidations. Strict observations/pre-settlement views and official results passed; `stock-history-report.json`. No search/neural strength claim. |
| Fresh P6 export and three-tier CPU smoke | ONNX export/parity and legal complete plans passed; `laptop-smoke/benchmark.json`. Fixture remained unqualified. |
| Fresh P5 collection | Two full authored-fixture games completed, including a table against three Original Hard seats, plus a real two-worker searched turn. No game failures or remaining CPU leases; `difficulty-smoke/smoke-report.json`. Pinned report reload passed. |
| Standalone bundled-runtime pipeline | Actual RPC, observed Play Now/cancellation, exact save restoration, late-control isolation, policy-only mandatory liquidation and seven native reconstructions passed. Interpreter/backend preflight, dry run, installation and byte-exact rollback passed **only in the marked isolated sandbox**; `../p6a/report.json`. |
| Preservation | All **1,743** snapshotted upstream source/config/bank/installed-DLL files match. Step 5, learning and P0–P4 identities match; `preservation-before.json`, `preservation-after.json`. |
| Final evidence validation | P5 report recomputation, P6 package validation, staged helper hashes/current identities and actual control flags passed; `final-validation.json`. |

The one-position CPU smoke measured admitted planning times of 1.843 / 4.547 / 10.922 seconds for Easy / Medium / Hard, with 0 / 0 / 4 completed simulations and 0 / 0 / 1 quality samples. There were no deadline resets, measured overruns or remaining CPU leases. Peak process-tree RSS was about 674 MiB. Other audit checks ran concurrently, so these are execution observations, not a speed comparison. No performance improvement is claimed.

P5 collection took 35.719 seconds; its two-second searched turn returned a legal fallback with no completed simulations and retired both leases. In the standalone P6 smoke, cold Play Now took 8.000 seconds including initialization (0.437 seconds inside planning), cancellation took 0.422 seconds, and warm mandatory liquidation took 0.172 seconds. Both controls were actually observed; none reset the admitted deadline or reported an overrun. These fixtures remained unqualified.

The final package is `6c53d7108c18f2aeab8816df2ed4d29e75cbfed3a2b57733af84384316b50166`; the final staged build is `10ab2fb003b13478dad5f10b691c35bdc07fdace4ef10cff7166fef852965c7e`. Final P5/P6 source identities are pinned in `final-validation.json`. Test-driver setup errors retained in `native-save-before.log` / `native-history-before.log` were compiler mistakes, not additional product findings; `native-save-repro.log` contains the actual pre-fix native defect.

<a id="audit-2026-10-12-p5-p6-compatibility-and-continuation"></a>

### Compatibility and continuation

Existing Step 5/IL checkpoints, optimizer/RNG state, counters, datasets, frozen banks and P0–P4 artifacts retain their contracts. Continue those runs with their existing commands; do not reset, retrain or migrate them for this audit.

For P5, retain old evidence under its original identity. The new verifier intentionally refuses old panels/reports: they lack the new job-ownership guarantee. Once a real P4 roster is qualified, use a **new output root** (for example `P5="$ROOT/artifacts/personality/p5-audit-1.2"`) and the plan/collect/check commands in [the P5 guide](QUALIFICATION_P4_P6.md#p5). Obtain new calibration and disjoint release evidence before claiming qualification. Do not delete or rewrite `reservations.json` to retry a failed game. A reserved crashed job requires an explicitly amended study; a different chunk name is no longer a retry mechanism.

For P6, re-export a new immutable candidate from compatible qualified P4/P5 evidence, restage its helper, and rerun the backend/Unity acceptance procedure in [the P6 guide](QUALIFICATION_P4_P6.md#p6). Never replace an old package's runtime in place or edit an old neural save's package ID. Existing valid neural saves remain tied to their original immutable package/runtime. Envelopes already invalidated by the first-save bug remain rejected; no speculative save repair was added. Legacy-only saves retain their format.

Reproduce the Python regression suite from a dependency-provisioned environment:

```powershell
$env:PYTHONPATH="$PWD/src;$PWD/tests/R2;$PWD/tests/Recovery;$PWD/tests/Personality"
python -m unittest discover -s tests/Personality -p 'test_*.py' -v
```

Use fresh output directories for `smoke_difficulty.py`, `smoke_laptop.py`, `check_p56_native_audit.py`, `check_laptop_native.py` and `check_laptop_pipeline.py`. The latter requires a short Windows sandbox path and an interpreter whose own environment contains the package dependencies. Development `PYTHONPATH` additions are deliberately not inherited by installation preflight.

<a id="audit-2026-10-12-p5-p6-remaining-limits"></a>

### Remaining limits

- There is no qualified real parent/roster or P5/P6 release evidence locally. Synthetic positive certificate tests exercise validator logic only; they are not release certificates for a real package.
- Windows CPU and isolated native hooks were exercised. Linux, lab CUDA, Intel graphics, multi-GPU behavior and representative real-model laptop performance were not measured.
- Actual Unity interaction, menu rendering, user-driven rewind/cancellation and post-game resource restoration still require the documented acceptance session. A stubbed native initialization plus a separate real Python pipeline is not a full interactive game test.
- Cold Play Now still waits for lazy model/native initialization. Native cooperative operations can outlast their soft cancellation point; the existing measured overrun gate remains in force.
- Abrupt OS/process termination during installation/rollback remains a manual journal/backup recovery case. Ordinary replacement failure recovery is covered by existing tests. No live deployment was attempted.
- This was a bounded implementation audit, not proof that the code is bug-free. No long qualification campaign or throughput benchmark was run.

<a id="audit-2026-10-13-p0-p6"></a>

## P0–P6 bug audit — 13 October 2026

Reviewed the active post–Step 5 implementation at starting HEAD `72bd43ff01a10a8d917705617b3805f6df41e924`, including the preceding P5/P6 fixes. **Twelve confirmed defects were fixed**, with 17 additional regression test methods. No additional independent P3 or P5 defect was confirmed. P5 was updated to enforce the corrected P4 evidence contract.

All 138 Personality tests pass. Fresh native, export, service, save and isolated installer checks pass. These results establish bounded interface and failure-path behavior using authored fixtures; they do not establish trained-model quality, qualified personalities, calibrated difficulty or release readiness.

<a id="audit-2026-10-13-p0-p6-scope-and-preservation"></a>

### Scope and preservation

Parallel reviews covered P0/P1 handoff and continuation; P2 search, labels, readiness and P6 runtime; P3/P4 repertoire and qualification; and P5/P6 export, provenance, native bridge and persistence. Findings were reproduced before repair and the integrated source was frozen before final native staging and export. The root review checked the execution paths, integrated changes and independently validated the final evidence chain.

The working tree already contained unrelated joint7/recovery/configuration/report work. This audit did not edit or revert it. `preservation-before.json` records the starting status and 2,718 protected file hashes. The final comparison reports:

- The authorized `configs/personality/source-migrations.json` change.
- Concurrent changes to `docs/EXPORT_TRAINING_REPORTS.md` and `tests/Recovery/test_joint7_150_review.py`, plus new joint7 review evidence. These were not written by this audit and remain intact.
- No unexpected differences; all 1,766 recorded installed/guarded-host/training files retain their starting hashes.

No installed game binaries, frozen opponent banks, real datasets or checkpoints were changed. No commit, push, real installation, training campaign or strength tournament was performed. Installer tests wrote only a newly marked fixture sandbox under `artifacts/p6b/isolated-game`, then rolled it back. Preexisting upstream source changes remain present; the Step 5 fingerprint during the final smokes is `3de301188b2eb777a7929da14a3e8d3362f106e75e54129eef8776864809f8e6`.

Evidence below is local under [`artifacts/p0p6-audit-20261013`](../../artifacts/p0p6-audit-20261013), except the short-path installer/RPC sandbox [`artifacts/p6b`](../../artifacts/p6b). The shorter path avoids Windows runtime path limits. These private fixture artifacts include game dependencies and are not public redistribution material.

<a id="audit-2026-10-13-p0-p6-confirmed-findings-and-repairs"></a>

### Confirmed findings and repairs

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

<a id="audit-2026-10-13-p0-p6-review-coverage-without-an-additional-confirmed-defect"></a>

### Review coverage without an additional confirmed defect

- **P3:** masked temporal representation inputs, train-only grouped fitting, public observation boundaries, causal behavior effects, encoder refits/re-embedding, archive geometry, quality filtering and selection of actual tested controllers. Existing regressions passed; no new production change was justified.
- **P5:** panel job ownership and replay, derived report checks, source/evidence closure, stock-Hard baseline, profile/hint qualification and public intent handling. The P4 gate integration above is the only new production edit in this stage.
- **Native P5/P6:** state/save cloning, pending-action restoration, revision/history rewind, foreign-game rejection, exact current legal path validation before live mutation, public history reconstruction, own permitted information, cancellation and main-thread resource lifecycle. Three complete history games and focused native checks passed. Full interactive Unity acceptance remains untested.

<a id="audit-2026-10-13-p0-p6-verification"></a>

### Verification

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

<a id="audit-2026-10-13-p0-p6-measured-timing-and-its-limits"></a>

#### Measured timing and its limits

The paired P2 12-second/four-worker probes completed 17 simulations without bias and 18 with authored bias: 0.777 versus 0.825 completed simulations/second. Final publication took 10.875–10.984 seconds with zero overruns; prior preparation took 2.2–4.6 ms. This four-root sample is noisy and establishes neither a speedup nor equal playing strength.

The separate P5 two-second turn produced a legal fallback and **zero completed simulations**, with no reported turn overrun. In the P6 ORT fixture, Easy/Medium/Hard completed **0/0/4** simulations; Easy and Medium used legal fallbacks. Admitted time was 1.813/4.547/10.922 seconds with zero deadline resets or turn overruns. End-to-end case time was 5.547/4.625/11.093 seconds; cold runtime startup was 3.453 seconds. These tiny authored models/positions and concurrent local checks are unsuitable for a backend or laptop throughput recommendation.

Standalone cold Play Now took **8.406 seconds end to end**, while its recorded admitted work was 0.250 seconds and planned work 0.390 seconds. The control was observed and returned a complete legal plan, but startup/preparation latency is material and is not covered by the admitted search clock. Cancellation took 0.375 seconds and preserved the save exactly; debt took 0.172 seconds without search or budget mutation. Representative warm/cold UI responsiveness still needs actual laptop acceptance. No production responsiveness claim follows from passing this fixture.

<a id="audit-2026-10-13-p0-p6-compatibility-and-continuation"></a>

### Compatibility and continuation

<a id="audit-2026-10-13-p0-p6-existing-step-5-and-p1-work"></a>

#### Existing Step 5 and P1 work

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

Use the actual existing command's paths/options if different. For a continuing warm-up, repeat its `warmup` command instead; do not revert a joint pilot to frozen preparation. Interrupted uncommitted phases follow the existing [P0/P1 recovery procedure](TRAINING_P0_P3.md#p0-p1-5-checkpoints-traces-and-scope): preserve the orphan operation directory, keep the authoritative state/checkpoint pair and rerun after stopping all writers. This audit does not reconstruct bytes already lost by older persistence implementations.

<a id="audit-2026-10-13-p0-p6-p2p6-evidence-and-saved-games"></a>

#### P2–P6 evidence and saved games

The P1 migration does **not** migrate old teacher labels, readiness, discovery archives, descriptors or qualification results. Regenerate downstream evidence under current source IDs. P4 1.1.0 requires the new fixture gate; dependent P5 contracts therefore change even though the P5 version remains 1.2.0. Re-export P6 as 1.2.0 and restage the helper after upstream qualification. Do not merely retag old bundles.

Existing immutable P6 bundles and saves remain untouched. Use an old bundle's matching bundled runtime/helper to validate or resume its pinned save. The current helper intentionally rejects older package formats. The save schema is still 1.0.0; this does not authorize transferring an old save to a new package identity or resampling its member.

Final local fixture package ID: `40e30e11e975091bb2145c12036435bb28504a100aeb4c39ddf517f1082533f5`. Staged helper ID: `19e6ca43020a992346ee274cd9b28383df35c1295b70002417643183f483114f`. Both remain unqualified fixtures.

<a id="audit-2026-10-13-p0-p6-remaining-issues-and-limits"></a>

### Remaining issues and limits

- **Unresolved budget specification:** active P2 `search/teachers.py` has `WALL_FRACTION=.9` and the lab guide describes 90%, while master plan §4.4 specifies 50%, matching Step 5. The audit preserves all current budgets and explicitly records the conflict in both documents. Resolve and version the intended P2 accounting policy before a real campaign; no guessed change was made.
- **Real artifacts unavailable locally:** no accepted lab parent was imported, no real 16×8 repertoire/difficulty panel was qualified, and no human recognition/tactical evidence was created. Authored fixtures must not become training, qualification or release evidence.
- **Hardware/platform scope:** Windows CPU and native .NET execution only. Linux/Mono, CUDA continuation execution, multi-GPU, Intel/OpenVINO and long-running resource behavior were not measured. Saved CUDA RNG payload preservation is not proof of CUDA execution parity.
- **Unity acceptance remains open:** interactive menu layout, full mixed/neural play, pending-action save/load, history rewind, cancellation and presentation/resource restoration need review in an isolated running Unity copy. No interactive game was launched here.
- **Performance:** cold preparation latency and short-budget fallback rates above need representative laptop testing with real models. No performance optimization or hyperparameter change is justified by these fixtures alone.
- **Crash durability:** ordinary error cleanup and existing atomic pilot publication are tested. Forced termination at every installation/publication instruction, power loss and recovery from prior already-corrupted artifacts are not covered.

The review found and repaired specific remaining defects. It does not establish that P0–P6 is bug-free or ready for release.
