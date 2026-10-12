# Comprehensive implementation audit — 12 October 2026

Confirmed defects were fixed in replay attribution, P1 checkpoint publication, run ownership, retention recovery, host failure handling and search construction. P0 now stores its dependency pins in the handoff instead of changing the completed training run's pins. The fixes preserve the approved training objective, architecture, hyperparameters, budgets, worker counts and opponent schedules. This report does not establish that the repository is bug-free or that a trained model is qualified.

## Scope and source identities

The audited base revision was `a873a78830096d92c567afac4e0170859b89ca1d` (“Improve step5 transfer and training budgets”). Repository instructions, Git status, current entry points, approved configurations, current lab guides, existing reports and the [personality plan](../ai-research/PERSONALITIES_AND_DIFFICULTY_PLAN.md) were inspected. Parallel reviewers owned host/contracts, search/runtime and P0–P4; the coordinating reviewer owned retention/continuation and integrated the changes. Confirmed paths and regressions were cross-reviewed.

Active training uses `scripts/step5.py`, dispatching to `recovery/` for the current recipe. The independent post-training entry points are `scripts/personality.py`, `scripts/personality_search.py`, `scripts/personality_repertoire.py` and `scripts/personality_deployment.py`. Historical dense models, old experiment recipes and frozen native banks were treated as references, not current specifications.

| Identity | Value |
|---|---|
| Original Step 5 implementation | `aeef0917b3645bdbdfe8dcb9d2bfd34ab05593c8b5fb9433dbc1e988b4ef131f` |
| Audited Step 5 implementation | `8d19aff88a93edd9b734561a73c83a440f5d3517429be007d9913aceab8cffd1` |
| Unchanged learning compatibility | `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2` |
| Original P0/P1 source | `3bb0eab8a753a5fb29e33b595e19e6ac0ab6d2e2d09875c059e93c9387bee068` |
| Audited P0/P1 source | `31b8d44872bbfcd9b89c8e3cc591ef38337fe39fbdebf64d4dc8a76ba46464ff` |

Source fingerprints use the existing canonical text hashing; no Git commit was created. All test data, generated runtime fixtures and smoke checkpoints stayed in isolated audit/temp directories. Installed game binaries, frozen opponent banks, existing datasets and real checkpoints were not modified. No patch installation, long training campaign or strength tournament was run.

## Confirmed findings and fixes

“High” means a plausible loss of committed progress, conflicting writers or incorrect training labels. “Medium” covers failure recovery, resource leaks and integrity admission. Boundary/telemetry defects with downstream defenses are lower severity. Evidence paths below are under `artifacts/bug-audit-20261012/` unless stated otherwise.

| Severity | Confirmed failure and evidence | Fix and affected files | Regression |
|---|---|---|---|
| High | Recovery replay accepted an ISMCTS distribution with reversed legal action IDs or a different observer. The distillation loss consumes probabilities positionally. Eight raw/cached cases passed validation before the fix (`teacher-attribution-before.log`). Fresh live search already checked attribution; no existing dataset corruption was demonstrated. | Bind teacher `legal_ids` to the exact ordered row choices and `observer` to its seat, including attested-cache ingress. [recovery/replay.py](../src/brass_ai/recovery/replay.py). | `test_teacher_attribution_audit.py`: versions 1.1/1.2, raw/cached rejection, valid legacy finalist/current targets, no-teacher records and seal integrity. |
| High | P1 overwrote `run/model.pt` before publishing its new state. An exception publishing the report/state left the prior committed state pointing to changed bytes and unable to resume. | Write the new checkpoint in its immutable phase directory, then atomically publish `state.json.model_path`; update the convenience `model.pt` afterward. P1 and P2 resolve the committed pointer. Version the pilot layout as `personality-pilot/1.1.0`, retaining the original layout reader. [personality/training.py](../src/brass_ai/personality/training.py), [checkpoint.py](../src/brass_ai/personality/checkpoint.py), [search/readiness.py](../src/brass_ai/personality/search/readiness.py), [search/continuation.py](../src/brass_ai/personality/search/continuation.py). | Injected state-publication and final-copy failures preserve a loadable committed model/optimizer/RNG/counters; continuation advances afterward. Old-layout/migration fixtures also pass. |
| High | `run_lock` checked activity, then removed a stale marker without serializing that gap. A contender could remove a newly acquired owner's marker and enter concurrently (`lock-before.log`). CLI cleanup/pinning also relied on a check before mutation. | Hold a stable OS lock in sibling `.run-locks/` through marker replacement and the entire operation. Lock restore, CLI cleanup/purge and pin writes. Windows byte locks and POSIX `flock` release on process exit; the guard inode is retained. [recovery/retention.py](../src/brass_ai/recovery/retention.py), [step5/cli.py](../src/brass_ai/step5/cli.py). | Deterministic competing-owner test using actual Windows locking; stale-marker recovery, exception release and existing active-run checks. |
| Medium | A native worker writing enough stderr could block before its stdout RPC reply because Python never drained the pipe. A worker already exited also left stdin open; malformed replies and broken pipes escaped the advertised error type (`host-failures-before.log`). | Drain stderr continuously into a bounded 16 × 2,048-byte tail. Close every pipe, join the drain thread, wrap transport/JSON/envelope failures in `HostError`, and clear prior native telemetry before each call. [host.py](../src/brass_ai/host.py). | Real subprocess floods stderr beyond pipe capacity and returns a reply; exited-worker cleanup, malformed replies and transport failures. The existing correlation check was also inspected. |
| Medium | Runtime manifest admission accepted missing/duplicate `BrassHost.exe` entries. A missing entry bypassed executable hashing; duplicate entries were still hashed but their ambiguous inventory was accepted. | Require exactly one executable entry and unique dependency names before process launch; retain existing dependency/hash checks. [host.py](../src/brass_ai/host.py). | Missing and duplicate pin sets rejected before launch; native runtime tests still pass. |
| Medium | Failure after opening a `BrassSpielGame` host leaked that process; failure constructing a parallel replica closed earlier replicas but left the coordinator's hosts open. | Close the started host on all subsequent adapter-init exceptions; close the coordinator as well as constructed replicas on forest-init failure. [searchlab/spiel.py](../src/brass_ai/searchlab/spiel.py), [recovery/parallel.py](../src/brass_ai/recovery/parallel.py). | Injected adapter/replica failures check exact closure; `test_search_audit.py` failed on the original source and passes after repair. |
| Medium | Restore moved a quarantined file before recording the move. Interruption at that point made the next invocation reject the already restored path (`retention-before.log`). | Journal `restoring` intent before moving, then recognize and hash-check that exact interrupted move on retry. Existing/changed destinations still fail; no overwrite is allowed. [recovery/retention.py](../src/brass_ai/recovery/retention.py). | Failure between rename and receipt commit, successful retry, changed destination and pre-existing destination rejection. |
| Medium | Another run retiring a control file after global dependency enumeration could abort cleanup following an otherwise valid training commit (`retention-scan-before.log`). | Retry the complete graph from scratch up to three times on `FileNotFoundError`. Never delete from a partial graph or ignore corrupt controls. [recovery/retention.py](../src/brass_ai/recovery/retention.py). | Vanishing external control fixture confirms retry retains the other run's checkpoint pin. Sustained churn can still exhaust the bound and fail closed. |
| Medium | P0 wrote `pins.json` inside the source training run, including before copies could fail. | Write source dependency pins in the separate handoff directory, then add copied dependency pins there. Global retention already discovers these pins. [personality/handoff.py](../src/brass_ai/personality/handoff.py). | Full source-file hashes remain equal after success and injected copy failure; retention still protects the original dependencies. A transient `.active-run.json` ownership marker is still created/removed while locking the source; persistent source contents are unchanged. |
| Low | Fixed-count search could report one interrupted trajectory without launching any, or omit pending trajectories retired at loop-boundary expiry. | In both modes set interruptions to attempted minus completed after retiring generators. Search policy, budgets and completed backups are unchanged. [search/ismcts.py](../src/brass_ai/search/ismcts.py). | Cancellation before first launch and deadline expiry with multiple pending paths. |
| Low | Runtime result validation accepted NaN utilities or ranks/winner utilities inconsistent with official VP, income and cash. Learning's terminal-target validator already rejected these, so training corruption was not demonstrated. | Check finite utilities, competition ranks and shared winner utility at the runtime boundary. [runtime_contracts.py](../src/brass_ai/runtime_contracts.py). | NaN, wrong rank/winner, competition ties and both tiebreakers; native terminal games still validate. |

Test maintenance is separate from these defects. Several tests still asserted retired dense-model sizes, 90%/7:9 Step 5 admission, an unchanged old win-only search contract, or omitted the added per-seat diagnostic metrics. They now assert the active residual presets, approved 50% admission and explicit combined-utility migration while preserving all unchanged fields. A private teacher mock now supplies the actual owner's cancellation fields. No production recipe was changed to satisfy those tests.

## Coverage and current semantics

| Area | Audit and checks | Limits |
|---|---|---|
| Host/rules/privacy | Four-player terminal scoring, official ties, legal multi-step decisions, deterministic restore/undo, revision/cache boundaries, explicit chance, hidden-state sampling and observation privacy; actual native fast-path parity/fault tests. | No change to C# game rules. Tests sample scenarios rather than exhaust all Brass states. |
| Demonstrations/IL | Native replay auditing, group-disjoint partitions, duplicate/corrupt receipts, idempotent combined datasets, continuation on additional manifests, old IL handoff, variable legal choices, supervised batching and feature seals. | No lab-only dataset was available. Existing raw data was not regenerated. |
| PPO/rewards | Exact behavior/model/seat attribution; forced-choice actor exclusion with critic targets retained; full legal-action probability denominators; undiscounted returns and causal capped insolvency; model/Adam/RNG restore, current/initial/history sampling, incumbent gates and separate official metrics. | CPU tensors and small fixtures; no learned-strength conclusion. |
| Search/teachers | Shared whole-turn clocks across actions/subchoices/replicas, CPU admission, cancellation/retirement, legal incumbents, incomplete-evidence rejection, multiplayer utility, hidden information, quality/reservation schedules and stale-target rejection. | Bounded native turns had no completed NN rollouts under concurrent CPU load; successful teacher qualification is covered by focused fixtures, not this smoke. |
| Runtime/performance | Host failure paths, queue/replica cleanup, batching bounds, shared-memory ownership, inference routing, actor/reader lifecycle, per-run locks and bounded histories/caches. | CUDA and multi-GPU routing were inspected/mocked; no Linux/GPU scaling or utilization claim. |
| Persistence/evaluation | Same-run source migrations, immutable committed models, population/incumbent preservation, seat/composition accounting, holdout separation, pin/quarantine/restore containment and Windows/Linux path handling. | POSIX lock branch and Linux unlink check were not exercised on Windows. Global external pin creation is not a filesystem-wide transaction. |
| P0/P1 | Accepted immutable champion selection, rejection of IL-only/unqualified parents, reward/utility compatibility, neutral output transfer, masked public-outcome goal labels, gradient freezing, fresh learner-only PPO and exact migration. | Authored source fixtures stand in for the lab parent. No future/private features are added; realized future outcomes are masked supervision only. |
| P2 | Bounded root-only style prior, zero-bias equivalence, saved intent, legal masks, model/seat/style identities, readiness, fresh teachers, policy-only distillation and compatible successor pilots. | Trained utility/goal calibration and equal-budget strength remain unmeasured. |
| P3 | Public temporal trajectories, grouped splits, AE/PCA interfaces, noisy-candidate rejection, train-only fitting, immutable archive refit/re-embedding, normalization/source IDs and tested-member selection. | No qualified learned repertoire was produced. |
| P4 | Exact 16 × 8 membership, unique stable IDs, shared weights, seeded seat-specific once-per-game sampling in calibration jobs, changed/missing artifact rejection, manifest serialization/loading, `resolve()` inference recipes and preview/full evidence gates. | Qualification, P5 difficulty calibration and P6 laptop export/parity, product runtime sampling/save restoration and installation remain separate unfinished work. |

The current Step 5 recipe remains time-primary: **12 seconds per NN whole turn**, **one second per searched-v4 whole turn**, **four private workers**, clocks starting after shared CPU admission, and **50% cumulative teacher admission**. The lab preset retains 24 actor processes, each with two overlapping ordinary hosts, 24 queued heavy tables, 60 CPU slots under 64 allocated cores and 16 replay readers. There is no independent full budget per action or replica. Archived count-mode contracts remain supported and tested.

The objective remains `W + 0.10*placement + 0.001*VP - causal capped insolvency`, gamma 1; official win/place/VP heads and strength metrics stay separate. The personality plan's Step 5 reference table still says 90%, whereas the active Step 5 code/config/guide says 50%. P2 also explicitly retains its own 90% accounting contract. These are reported distinctions; this audit did not guess a new cross-stage budget or alter P2's recipe.

## Verification and measurements

Environment: Windows, Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5; CUDA unavailable. Tests used isolated fixture paths. Evidence is in `artifacts/bug-audit-20261012/`; overlapping suite totals must not be summed.

| Check | Result | Evidence |
|---|---|---|
| Full distinct Recovery suite on final source | 279 run: **275 passed, four skipped**, zero failures/errors; 948.040 s | `recovery-final.log`, `recovery-final.json` |
| Consolidated P0–P4 suite | 67 passed, 147.887 s | `personality-tests-consolidated.log` |
| Host/R1 plus native runtime and telemetry | 32 passed, 234.806 s | `host-native-final.log` |
| Learning/R2 | 24 passed, 35.813 s | `r2-final.log` |
| Existing Step 5 tests | 35 passed, 68.839 s | `step5-final.log` |
| Transfer/continuation tests | 27 passed, 199.196 s | `transfer-final.log` |
| Final replay/search/review focused checks | 30 passed, 51.147 s | `search-final-coverage.log` |
| Retention fixes and existing pin/quarantine/path tests | 9 passed, 75.327 s | `retention-complete.log` |
| Actual old-source P0/P1 artifact migration | Exact tensor/Adam/RNG import; warm-up counter 1 → 2; source files unchanged | `personality-migration-report.json` |
| Native four-game PPO continuation | Passed: two iterations, two optimizer steps each; 242.156 s | `ppo-resume-native/report.json` |
| Final native search | Three legal two-action turns; start/end source matched | `search-native-final/report.json` |
| Step 5 migration receipt independent verifier | Original 268 routes and 15 IL receipts unchanged; 41 exact route clones + one direct source transition + one IL source approval | `migration-independent-final.json` |

Intermediate broad runs overlapped source edits and fixture cleanup; one private runner was also corrected to use a Windows multiprocessing main guard. Those runs are not final passing evidence. Early failing logs were retained, including pre-fix fault reproductions. The consolidated final results supersede them.

The final Recovery run's source fingerprint matched `8d19aff8…` before and after execution and its process exited successfully. Its four skips were two CUDA-only checks, the POSIX unlinked-working-directory check and the opt-in four-game IL head-to-head native test. Separate native PPO/search/host smokes described here did run; they do not turn that skipped IL head-to-head check into a pass. Final `git diff --check` passed; Git only reported expected CRLF-to-LF normalization notices.

The four-game PPO smoke uses current native code with a small ordinary-only, old-source-format fixture (`826b6beb…`), then exercises the real reviewed loader into the final source. It does not run the historical implementation or a lab checkpoint. It advanced two generations/four games with two optimizer steps per generation, preserved the earlier checkpoint and incumbent bytes/opponent window, and refreshed utility calibration from new ordinary returns. Separate exact-source tests cover the direct `aeef0917…` transition.

The relevant repository verification commands, with the dependencies and native runtimes already installed, are:

```sh
PYTHONPATH=src:tests/R2:tests/Recovery python scripts/verify-recovery.py
PYTHONPATH=src:tests/R2:tests/Recovery:tests/Personality \
  python -m unittest discover -s tests/Personality -p 'test_*.py' -v
PYTHONPATH=src:tests/R2:tests/Recovery \
  python tests/Recovery/smoke_selfplay_review.py --output artifacts/audit-resume-smoke-new
```

The audit's full Recovery runner deduplicates imported test IDs and records its source before/after; the standard script may report seven additional duplicate test executions. Use a new smoke directory, preserve existing runtimes and do not overlap independent fixture-deleting suites when interpreting retention timing.

The final search smoke measured NN evaluation **12.140 s** full turn / **11.906 s** search and teaching **12.063 s** full turn / **11.829 s** search. Each attempted eight trajectories, completed zero and retired all eight, correctly admitting zero labels. Recorded search publication overrun was zero; applying the complete legal turn adds engine time. Frozen v4 took **3.359 s** full turn / **3.156 s** native search RPC against its one-second budget, completed zero rollouts and used its legal incumbent. CPU admission ended with zero active/queued leases (peak four), inference had no error, and v4 batching had no pending requests.

These timings were collected with other CPU checks running and are not before/after throughput benchmarks. No performance speedup is claimed. The stderr regression demonstrates removal of a specific deadlock, not higher steady-state games/second. Frozen v4's cooperative native incumbent/retirement path remains an observed deadline limitation; its bank was not modified.

The two native PPO iterations reported cleanup phases of **11.8 s and 11.2 s** in this workspace, versus collection phases of 58.4 s and 53.2 s. Global dependency scans remain noticeable I/O work, especially in a fixture-heavy artifacts tree. These measurements are not directly comparable to the earlier throughput review's artifact inventory or background load, and no further optimization was justified by this audit alone.

A separate P4 development smoke completed one actual four-player game through both eras in **104.297 s** with an untrained fixture. It collected only one of 104 planned preview games; the full-size, searched, quality, variety and blind-review gates correctly remained false. Its Step 5 identity was the intermediate `494f11…`, before the final teacher-attribution guard, and its P1 identity was the final `31b8d4…`. That final guard only executes for non-null teachers and does not run on this smoke's direct-policy rows. This is development/interface evidence, not final-source qualification or a trained roster.

## Checkpoint and data compatibility

### Existing Step 5 self-play

The explicit review is `step5-comprehensive-bug-audit/1.0.0` in [selfplay-resume-sources.json](../configs/learning/selfplay-resume-sources.json). The current source-only transition is exact `aeef0917…` → `8d19aff8…`. Forty-one previously approved routes ending at the original source are extended to the final source with their **existing config changes/migrations preserved exactly**. All original entries remain unchanged; temporary audit source identities are not approved.

The continuation tests check compatible model weights, Adam moments/steps, RNG, game/iteration counters, incumbent/population, opponent-window progress and cumulative search accounting. New utility calibration comes from fresh ordinary rollouts before teaching; prior-source search labels remain rejected. The source transition does not silently reset the optimizer or treat the latest learner as an accepted incumbent.

On the lab, finish a committed iteration and stop all old-source writers before syncing the source and both receipt files. Keep the existing run directory, configuration and initialization-origin arguments. Repeat the existing command with `--resume-reviewed-source`. For example, substitute the actual existing paths into this template:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ \
  --config <same-config-path> --device cuda train \
  --run <same-selfplay-run-path> --imitation-run <same-completed-il-run-path> \
  --iterations 1 --resume-reviewed-source
```

Omit `--imitation-run` only if the original run did not use it; preserve any original `--demonstrations` override instead. Global `--root`, `--config` and `--device` precede `train`. Do not substitute `--initial`, `--from-run`, a new run directory or changed recipe. Keep normal allocated-GPU visibility; this audit adds no GPU override. Once the first transition commits, the same run continues normally.

An **unfinished old-source iteration must be finished/recovered with its original source before migration**. Preserve its work and use its existing recovery procedure; do not delete receipts or edit source IDs to bypass validation. Old processes do not honor the new OS guard, so they must be stopped before using this update.

### IL and datasets

[imitation-handoff-sources.json](../configs/learning/imitation-handoff-sources.json) adds the exact original source while retaining every earlier approval. Completed compatible IL still supports `imitate --from-imitation-run <completed-run>` on additional data in a new run, and `train --imitation-run <completed-run>` for a fresh PPO handoff. Existing manifest, checkpoint, lineage and split checks remain mandatory.

Unfinished same-run IL retains its exact-source rule. Finish/recover it with the original source before updating; this audit does not introduce an unreviewed mid-epoch optimizer migration. Completed IL continuation on additional manifests intentionally starts its documented new optimizer; same-run PPO continuation preserves its optimizer.

The authoritative replay/feature/model schemas and learning compatibility digest are unchanged. The recovery replay and runtime validator source bytes changed, so **stale derived caches are ignored, authoritative replay is revalidated and features are recomputed in memory**. The reader does not automatically rewrite the old cache files. Existing audited heuristic source data remains eligible; no blanket data regeneration, relabeling or weakening of hashes is authorized. Newly detected malformed teacher attribution must be rejected, not repaired by permuting labels without original evidence.

### P0/P1 and downstream artifacts

[personality/source-migrations.json](../configs/personality/source-migrations.json) approves only the exact P0/P1 persistence transition `3bb0eab8…` → `31b8d448…`, with identical non-source contracts. Old weights are verified against their original model identity before assigning the new identity. Loading does not rewrite source artifacts. A fixture produced by the actual exported original Git source was imported with identical weights/Adam/RNG and continued successfully, preserving its original source tree.

Continue a completed compatible P1 pilot with the **same command, handoff and run path**. Pilot `1.0.0` still reads its hash-bound `model.pt`; new commits use `1.1.0` and a hash-bound relative `model_path`. For a phase run with the audited code that failed before state publication, first stop pilot writers, preserve that uncommitted phase under `<run>/interrupted/`, then repeat the command from the committed state. Never move the phase referenced by `state.json.model_path`. If only the final convenience copy failed, state is already committed; the next invocation follows its immutable pointer. A pre-fix interruption may already have overwritten the old checkpoint; moving that orphan phase cannot reconstruct lost bytes. Recover that case from a known-good matching checkpoint/state backup or the original-source recovery procedure, preserving all available files. See [the P0/P1 guide](PERSONALITY_P0_P1_LAB.md) for the concrete phase example.

This migration does **not** approve old P2 teacher labels, readiness, P3 archives or P4 qualification under a changed model/source identity. Recreate downstream readiness, collections and qualification through their normal gates. The original Step 5 model remains a separate checkpoint format; personality checkpoints cannot be supplied to its reader.

## Remaining risks and untested areas

- `RulesHost.call` remains synchronous without a watchdog for an engine that silently stops replying. Draining stderr fixes the reproduced pipe deadlock but cannot detect every native hang.
- Frozen v4 can exceed the requested time while finishing its native incumbent/cooperative retirement. The smoke measured this directly. Candidate-incumbent truncation was inspected as a possible concern, but no reachable incorrect-plan reproduction was established; no speculative frozen-code change was made.
- Retention retries only three times and fails closed if external controls keep disappearing. Run locks serialize participating writers for one run; they do not make simultaneous creation of pins in other directories globally atomic. Direct library callers of prune/purge must retain their existing caller-held-lock discipline. Cross-host/network-filesystem lock behavior needs lab verification.
- Actual Linux/POSIX locking, CUDA prefetch/updates, VRAM, multi-GPU routing and the full 64-core allocation were not measured here. Mock/device-selection checks do not establish hardware performance.
- Lab-only checkpoints/datasets were unavailable. Exact committed-state fixtures and real old-source P1 artifacts establish compatibility behavior, not successful recovery of an unseen lab run or its filesystem.
- Successful trained-parent P0 handoff, useful goal calibration, equal-budget style skill, learned 16/128 repertoire qualification, per-tier difficulty and laptop release remain pending. P5/P6 and the later full personality league were not implemented merely to complete the audit.

Before resuming production, use the bounded same-run continuation on the allocated lab hardware and inspect its source transition, preserved counters, ordinary-rollout calibration and hardware diagnostics. This is a remaining environment check, not permission to reset or replace the existing training run.
