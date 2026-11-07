# B07 g1250 R8: reuse checkpoint snapshots and repair the fixture

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
reused. Remaining canaries/continuation/adoption are still **Pending on the lab**.
The fast 0.8s/K4 + <=0.5s and full 4s/K16 + <=1s budgets are unchanged.

## One command for real g1250 → g1251 continuation

From the lab code checkout, in the existing training environment, with the run
paused at the qualified g1250 checkpoint:

```bash
unzip -o distribution/rust-lab-b07-g1250/rust-lab-b07-g1250-r8.zip
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
4. Uses the config saved in `state.json`, writes it to the evidence directory,
   and calls the existing trainer with `--iterations 1 --expected-generation 1250`.
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

This command performs no production adoption or real training:

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

After the command above succeeds and g1251 is committed, run from the lab code
checkout in the same training environment:

```bash
python scripts/step5.py --root "$PWD" --config configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json --device cuda --mono mono train --run artifacts/step5/joint7-v11-2-g1290-trained-restart --expected-generation 1251 --iterations 1
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
Archives are immutable; future changes require a new revision.
