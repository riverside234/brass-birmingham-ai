# B07 g1250: Git checkout continuation after passed qualification

## Ordinary collector performance update

PI00–PI05 now use the production ordinary path, including combined completion export and immutable replay reuse.
Before the shadow command below, build/verify the upgraded **ordinary** adapter once:

```bash
python scripts/prepare-rust-teacher-verification.py --expected-generation 1258 --offline && \
python scripts/prepare-rust-collector-performance.py --expected-generation 1258 --offline
```

The commands prepare source-pinned adapters for the bonus transition repair; the qualified search core and checkpoint/Adam/RNG are preserved.
Its success/failure report exports under results. Then use the existing shadow continuation
with `--share-target 0.25`; do not repeat B01–B07 qualification or extract old archives.
The shadow training JSON now saves `execution.collector_validation="production_fast"`
and `execution.replay_bindings=true`; no shell exports are needed. The launcher
preserves explicit JSON overrides and passes them to workers without changing frozen jobs.
Generation must match the actual committed boundary. See [the performance supplement](../rust-collector-performance/README.md)
for current mode/coverage, short smoke commands and measured component limitations.

## Full shadow gradient qualification

The attached log confirms the latest committed checkpoint is **g1258**. Retain that first measurement; see [the bonus transition repair](../../ai-research/RUST_SHADOW_BONUS_AND_FOUR_ITERATION_FIX.md). The current revision fixes the
descriptor leak below and implements the user's additional settings:

- Branching **0%**; full **28 train + 4 held-out** shadow teacher tables per iteration.
- Rust search_v1; existing full/fast search profiles and the current uncapped independent-verification policy.
- Measure actual teacher/PPO actor gradient ratios on shared encoders at beta=1,
  with zero teacher gradient applied to ordinary updates.
- After four qualifying consecutive full shadow iterations, persist a **fixed**
  `beta=clip(target/median(raw_ratio),0.01,10)`. Default target **0.25**; later
  explicit target options **0.4** and **0.5** reuse the frozen measurements.
- Keep 24 ordinary actor processes / 48 lanes, up to 24 publishers and the global
  60-slot CPU allowance. The leak fix adds no inference requests or record scans.
  Full shadow collection/gradient diagnostics add their separately reported cost.

After syncing the code to the lab and fully exiting the failed trainer, run in
the existing training environment:

```bash
git pull --ff-only
python scripts/continue-rust-shadow.py --expected-generation 1257 --iterations 4 --share-target 0.25
```

This resumes the same full checkpoint/Adam/RNG/PPG/population state and advances
**g1257→g1261**, running only three remaining iterations from g1258. It verifies original native/runtime pins and writes the exact reviewed
shadow config and invokes the normal trainer. It retains completed new shadow
tables for validated recovery. The 2→4 measurement upgrade preserves pending
ordinary/teacher work and the first completed observation; only initial adoption
from a different collection recipe needs its original archive path. Repeating this command after any commits
does only the remaining work; it never starts an extra generation. A pending
checkpoint or teacher publication must first recover on its current config.
Prepare both affected adapters above first; no new ZIP or repeat of passed qualifications is needed.

The config is [the default shadow recipe](../../configs/step5/joint7-v11-2-g1290-trained-restart-rust-shadow32-no-branch-share025.json).
Inspect each iteration's `teacher_phase.term` for raw share/cosine distributions,
phase support, projected weighted share and gradient qualification. The fixed
receipt/beta persists at `teacher_phase.weight.gradient_qualification` in state
and checkpoint. Missing/invalid data stays Pending. **It remains shadow after
qualification**; teacher on-mode requires a separate explicit step.

Later, at a clean committed boundary, choose `--share-target 0.4` or
`--share-target 0.5` with the actual current `--expected-generation`. For example,
this runs one further shadow iteration while explicitly selecting target 0.4:

```bash
python scripts/continue-rust-shadow.py --expected-generation 1261 --iterations 1 --share-target 0.4
```

Changing the target updates the fixed beta from the saved raw calibration; it
does not enable gradients or adapt beta automatically. The helper prints and
exports a fresh `results/rust-lab-qualification/` receipt after success, pending
qualification or failure. Push that folder from the results repository as usual.
Old qualification files and native binaries must remain in their original lab
locations. The exact g1257 null-readiness case carries the actual hash-bound g1254
fit into teacher state, following the plan's last-non-null rule; it preserves the
current checkpoint's calibration field and explicit-false precedence.

**81 distinct focused debug checks passed.** This includes unchanged shadow PPO
weights/Adam/RNG, fixed-beta/restart/target edits, 32-table authored orchestration,
cache/staged-shard recovery and resource lifecycle tests. No local benchmarks,
native games or live training sessions ran. Actual full-volume timing and the
four-iteration gradient qualification is Pending on this lab continuation.

## Current g1257 open-file repair

Results pulled inside `results/` to `d6ed70075232997c322231e034cda81bf83e3cb9`
confirm committed **g1257**, checkpoint
`cad965da88338e6a424cd1e615ffb61da014c52e8ff3237baeeae8d94e429d40`.
The next collection failed with `Too many open files`: private inference pipes,
lease queues and shared slots survived each retired Rust/evaluation pool until
the whole training process exited. Joined process sentinels also needed explicit
closure. The Git-only fix retires these at existing pool shutdown boundaries,
drains unclaimed handle transfers, protects in-flight shared views and cleans up
partial startup. See [design and checks](../../ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md#g1257-descriptor-exhaustion-collection-boundary-resource-retirement).

**Speed/behavior constraints:** keep 24 actor processes / 48 lanes, up to 24
publishers, existing batching/model caches and the same 60-slot CPU allowance.
No extra NN requests/forwards, feature encoding or record validation is added.
Small per-envelope lifetime accounting protects shared views; no lifetime lock
spans inference computation or IPC send. Resource cleanup stays at round/pool
boundaries. This avoids reducing workers or restarting every generation; exact
timing impact awaits the next normal lab report, rather than a local benchmark.

For the original teacher-off recipe only, after syncing this fix through Git,
**fully exit the failed trainer** and resume
the same run in a fresh process. Preserve `iteration-1257` and every original
model/cache/qualification/build dependency. The existing strict recovery checks
reuse valid published families. No override, rebuild, new archive or repeat of
passed qualification is required:

```bash
git pull --ff-only
python scripts/step5.py --root "$PWD" --config configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json --device cuda --mono mono train --run artifacts/step5/joint7-v11-2-g1290-trained-restart --imitation-run artifacts/step5/features4-v11-2-batch7-il --expected-generation 1257 --iterations 1
```

This older command advances **g1257→g1258** with teaching off; use the full-shadow
command above for the newly requested settings. The tracked g1247 config is exactly equal to the
config committed in g1257. Increase `--iterations` for sustained continuation
after verifying that the resource counts return to their stable baseline. The
generation guard rejects a stale command; inspect state if training advanced.
Check the next iteration's `inference.reply_channels` active/retiring/shared-slot
and lease counts. Persistent search pools may retain their own live channels.

**41 distinct focused debug checks passed**, including 624 real lane allocations
across repeated rounds with a stable OS handle count, spawned/unused transfers,
cancel/stale/partial-startup cases, both replica routes, unchanged predictions,
request/forward counts and strict predecessor checkpoint identity. No benchmark,
simulator game or training update ran locally. For only the affected lab debug
checks, if desired:

```bash
PYTHONPATH=src:scripts:tests/R2:tests/Recovery:tests/FeatureEngineering:tests/RustBackendLab python -m unittest test_inference_channel_lifecycle -v
```

Check the process limits with `ulimit -Sn` and `ulimit -Hn`. If permitted and the
soft limit is small, `ulimit -n 4096` before starting the trainer adds descriptor
headroom for parallel workers; it must stay within the hard limit. Raising a
limit alone cannot cure a leak. The fix preserves the exact g1257 checkpoint
execution as a reviewed predecessor and records the new execution on the next
commit. Old checkpoints, frozen jobs and passing qualification are not rewritten.
The next normal lab iteration owns verification of this operational repair.

## Historical g1253 collection repair

Results were pulled inside `results/` to `097327ef4fab7190f8d91ce4996dd1d4db4c5ce2`.
New 320-family Rust collections took 501.5s and 496.2s. The repair addresses the
two-writer bottleneck, serial cache revalidation, unnecessary actor startup and
repeated byte/history/feature checks. Current full-size jobs use **24 publishers**
under the same 60-slot CPU allowance; small batches retain at most two. Cold
validation remains complete; current independent receipts authorize warm reuse.
See [design, C# comparison and measurement limits](../../ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md#full-size-collection-repair-from-results-097327e).

After syncing this Git update, continue the **existing g1253 → g1254** run in the
same lab environment. Original binaries, checkpoint/Adam/RNG, recipe, caches,
backend selection and teacher-off state are preserved. No source override,
new archive or repeat of passed qualifications is needed:

```bash
python scripts/step5.py --root "$PWD" --config configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json --device cuda --mono mono train --run artifacts/step5/joint7-v11-2-g1290-trained-restart --imitation-run artifacts/step5/features4-v11-2-batch7-il --expected-generation 1253 --iterations 1
```

The generation guard rejects stale commands. If the run advanced, inspect its
actual committed generation before changing the guard. The old adoption wrapper
below applies only at the historical g1250/g1251 boundary.

For a bounded CPU component profile first (no simulator, NN or training):

```bash
python tests/RustBackendLab/profile_b07_publication.py --out artifacts/b07-publication-profile-lab --families 8 --decisions 128 --workers 4
python tests/RustBackendLab/profile_b07_publication.py --out artifacts/b07-binding-profile-lab --read-existing artifacts/b07-publication-profile-lab
```

Use new output paths when repeating. These compare authored frozen records and
feature values, startup, publication, cold validation and replay reading; the
forced component worker count does not change production config. Raw Windows
profiles are `profile-097327e-small.json`, `profile-097327e-medium.json`,
`profile-097327e-binding.json` and `profile-097327e-warm-checks.json` beside this README.
The last file measures the final hot-check repair: 1.562→1.263s median warm read,
with equal training games and unchanged feature bytes. They include both positive
phase improvements and negative startup totals. Later results `d6ed700` report
fresh 320-family collection at 367.4–395.9s with 24 publishers and 48 lanes;
successive jobs are not a matched speed benchmark. The affected debug suites passed
38 distinct checks, including the 24-lane publication scheduler; historical
qualification archives remain byte-for-byte intact. The CPU timing probes use
two/four workers; the requested 24-process full-size throughput awaits the lab.

The rest of this document preserves the original qualification/adoption history.

**Latest lab status:** R8 qualification, both canaries, diagnostic continuation,
Rust adoption, complete rollback restoration and the **real g1250→g1251 training
commit/full restart passed**. Results were pulled inside `results/` to `c51c7d9`:
[successful production receipt](../../results/rust-lab-qualification/20261107T134344148855Z-20261107T133341969715Z-b07-continue-cf42c992-3ed81ebb/evidence/qualification.json).
Rust collected the training data; g1251 and its full optimizer/RNG/PPG state
restored successfully. All eight exported files passed byte/hash verification.
This closes B07's teacher-off backend adoption gate. Continue normally from
g1251 using the command below; keep the original lab evidence and native binaries.
No new source override, bundle or native build is needed.

The earlier scheduler/PPG/imitation failures are resolved. If verifying the
already completed adoption boundary, the repeat-safe command is:

```bash
git pull --ff-only
python scripts/continue-rust-lab-b07.py --expected-generation 1250
```

At the observed g1251 boundary, it verifies only and never starts g1252. If
recovering the earlier adopted g1250 boundary, it skips passed qualifications,
preserves collected families and runs only that first real iteration. The exact
execution-source review retains frozen collection signatures and records actual
execution separately; unreviewed code or recipe changes fail. Do not extract R8
or an older ZIP over this Git update. The
tracked manifest still checks every listed file, including the reviewed repair
and its compatibility record. Historical ZIPs retain their original bytes.

The PPG path/retry repair has **39 passing focused debug checks**; no local
benchmark or real training iteration was run.
The startup follow-up passed **11 affected debug checks in 28.291s**, including
the actual launcher → CLI → scheduler path, stopping before collection and
checking that retries preserve pending work. No benchmarks ran.

**R4 ordinary qualification passed.** The result was pulled inside `results/`
at `9e219a2`. Same 3 families / 1,161 decisions / 675 forced rows, with equivalent
trajectories/features/targets: accepted work **30.632s → 26.820s (12.4% less time)**.
Sampled peak host memory was 4.917GB → 7.479GB. This is one bounded matched batch,
not a statistical or whole-training speed claim.

R8 fixes the latest qualification failure in the checkpoint regression fixture.
The test placed `shadow.pt` inside the directory reserved for value shards; strict
recovery correctly rejected the foreign file. Checkpoints and value shards now
use separate directories, with a negative test retaining that rejection.

Search-value publication prepares **one detached plain-label snapshot per commit**.
Checkpoint saving and commit verification use that exact snapshot. Repeated calls
reuse it; a stored integrity binding detects caller mutations. Failed saves or
failed state verification retain it for retry; successful commit releases it.
Further admission or PPG changes are blocked while publication is pending. Strict
`weights_only=True`, nonfinite rejection, live immutable labels, optimizer state,
receipts, recovery bindings and search behavior remain enforced.

**34 distinct focused debugging checks passed**, following the user's clarification.
No benchmarks, new games, training sessions or unrelated qualifications are run.
The exact successful R6 native teacher build and R4 ordinary qualification are
reused. Canaries/diagnostic continuation/adoption subsequently **passed on the lab**.
The fast 0.8s/K4 + <=0.5s and full 4s/K16 + <=1s budgets are unchanged.

## One command for real g1250 → g1251 continuation

From the lab code checkout, in the existing training environment, with the run
paused at the qualified g1250 checkpoint:

```bash
git pull --ff-only
python scripts/continue-rust-lab-b07.py --expected-generation 1250
```

This command **changes the production backend to Rust and runs one real training
iteration after the remaining gates pass**. It automatically:

1. Verifies the sealed R4 result, original ordinary build, source, committed
   state, checkpoint, config, population and runtime pins. It does not repeat
   passing B01–B06 or R4 ordinary qualification.
2. Runs the two pending shadow passes (two tables each, w=0, 4s/K16 +
   <=1s post-work and 0.8s/K4 + <=0.5s on fast turns) and the bounded continuation/update/reload check. It reuses the
   pinned R6 teacher build after checking its source, binary and regression logs.
3. Selects Rust ordinary collection in the same run with **teaching off**, under
   the existing run lock. It restores a complete rollback copy before adoption
   and verifies that copy again. No optimizer, seed, progress or recipe reset.
4. Uses the config and `imitation_source.bc_run` saved in `state.json`, writes the
   config to the evidence directory, and calls the existing trainer with
   `--iterations 1 --expected-generation 1250 --imitation-run SAVED_IL_RUN`.
   The checkpoint, Adam/RNG, PPG ages, local-scoring state, replay and worker cap
   are preserved. Rust owns ordinary game state, including V4 tables; the C# V4
   decision worker is retained. Established evaluation routing is unchanged.
5. Verifies g1251's actual native collection, report/qualification identity,
   checkpoint/backend persistence and full isolated restart. Exports evidence
   to the non-ignored `results/rust-lab-qualification/` folder.

The real iteration takes its normal training time; it is not a shortened smoke.
Follow its output using the `training.log` path printed by the launcher, for example
`tail -f artifacts/rust-backend/REPLACE_WITH_B07_CONTINUE_ATTEMPT/training.log`.
The launcher uses the first visible GPU, as the normal trainer does. To select one:

```bash
python scripts/continue-rust-lab-b07.py --expected-generation 1250 --cuda-visible-devices 0
```

**Repeat safety:** the same command resumes after adoption if still at g1250.
If g1251 already committed, it only verifies rollback/restart and never trains
g1252. It rejects another generation or an explicit reference-backend switch.
Ordinary interrupted work uses the existing trainer recovery rules; nothing is
deleted or rewound automatically. Keep native binaries and all referenced
qualification/rollback directories on the lab machine.

## Run only the remaining qualification first

For a run not yet adopted, this command performs no production adoption or real
training. The current lab run has already passed it; do not rerun it:

```bash
python scripts/qualify-rust-lab-b07.py --device cuda:0 --complete-remaining
```

Then use its **original lab evidence** receipt to adopt and train:

```bash
python scripts/continue-rust-lab-b07.py --expected-generation 1250 --qualification artifacts/rust-backend/REPLACE_WITH_COMPLETED_ATTEMPT/qualification.json
```

If a later phase fails, preserve the original attempt and reuse successful phases:

```bash
python scripts/continue-rust-lab-b07.py --expected-generation 1250 --resume-qualification artifacts/rust-backend/REPLACE_WITH_PARTIAL_ATTEMPT/qualification.json
```

The saved phases must match the exact R8 workflow, source, committed state and
teacher build; their bytes are rechecked. Failed phases are never reused. The
ordinary result and build are always the pinned successful R4 ones. An existing
current-source R6 teacher build can be supplied with `--teacher-build PATH/build.json`;
add `--offline` for Cargo-cache-only builds. Original C# binaries and Rust 1.99.1
remain required. Preserve the existing environment; if its measurement dependency
is missing, use `python -m pip install -r distribution/rust-lab-b06-g1250/requirements.txt`.

The qualified run is `artifacts/step5/joint7-v11-2-g1290-trained-restart`, g1250
checkpoint SHA256 `de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`.
Do not rewind a newer run. A different committed checkpoint needs refreshed gates.
The g1250 null-utility receipt exception applies only to the shadow canaries;
this workflow does not enable production teaching or change checkpoint metadata.

## Continue from g1251 after qualification/adoption finishes

The latest lab receipt confirms g1251 is committed. Run from the lab code checkout
in the same training environment:

```bash
python scripts/step5.py --root "$PWD" --config configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json --device cuda --mono mono train --run artifacts/step5/joint7-v11-2-g1290-trained-restart --imitation-run artifacts/step5/features4-v11-2-batch7-il --expected-generation 1251 --iterations 1
```

This resumes **g1251 → g1252** with the persisted Rust backend, optimizer, RNG,
replay and teacher-off recipe. The g1247 config file was checked against the
actual committed g1250 config: it is exactly equal, SHA256
`895d866502425dad30f2277b822dba2a15247f9a701d2dd5dd10b7e396de3ca8`.
The adoption wrapper carries that same config into g1251. The generation guard
rejects a stale command. No source override or fresh-run flag is needed.
`--iterations 1` deliberately advances one real generation; change it only when
you intend to run more iterations. The adoption wrapper verifies only its first
adopted boundary, so do not invoke it with `--expected-generation 1251`.

For an explicit return to reference collection at the current clean boundary:

```bash
python scripts/manage-rust-lab-b07.py reference --expected-generation 1251
```

This keeps all committed g1251 progress. Use the actual current generation if
it advanced. Pending work must first finish/recover on its existing backend.
A broken Rust executable does not prevent the explicit reference switch.

All successes/failures export under `results/rust-lab-qualification/`. Use the
printed `git -C results add/commit/push` commands; pull inside `results/` on the
other computer. Exported review copies do not relocate original lab binaries.

Local checks cover exact reuse, stale/tampered evidence, phase selection,
failed adoption/training, full-state restore, rollback integrity, committed-config
forwarding and no duplicate iteration. No full training session or unrelated
successful qualification is run locally. See [implementation and evidence](../../ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md#r8-one-snapshot-per-checkpoint-commit).
Code updates travel through Git. Archives are immutable historical inputs; create
new supplemental archives only for required assets not already supplied by Git.
