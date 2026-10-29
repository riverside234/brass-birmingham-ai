# B02 — Rust ordinary collection and branch families

Implemented 2026-10-28; search_v1 bundles refreshed 2026-10-29. **B01 lab qualification: Pending. B02 lab qualification: Pending.** The user will run both on the lab server. B02's qualification runner requires a passing search_v1 B01 v2 receipt for the same unchanged committed run and requested CUDA device. These changes do not select a new training backend or install anything into the game.

## Implementation

`lab-native/` is an additive Rust crate using the exact B01 `brass-ai-core`, plus locked `serde`, `serde_json` and `sha2` crates. The lab runner stages the core from the passing B01 checkout, not the mutable local `native/` directory. The shared source remains:

`fa0b3f7b74e9ed638c8b6f96d3f23b0910b4e5df9387ff8df58d5472a8014dd2`

Rust owns ordinary game progression, permitted observations, full ordered legal sets, exact Features-4 encoding, transactional row buffering, portable trusted fork state and passive current-era scoring receipts. Ordinary draws preserve the original ordered hidden deck; explicit search chance resolutions do not add ordinary decision revisions. Each process owns one lane. Frames, candidate pages, decision counts and retained rows have explicit bounds; incomplete lanes cannot publish rows.

The Python adapter calls the existing frozen ordinary planner and keeps its seeds, table composition, profile cycle, rotation, learner assignments, policy identities and contexts. It reuses `NeuralPolicy` sampling, versioned opening rules, branch selection/alternative draws and the existing GPU `InferenceService` with private reply channels/shared slots. Every forced and intermediate decision remains a row. Publication reconstructs the actual temperature/exploration denominator and replays the frozen per-seat random streams, including forced choices. Native encoding carries unknown actor descriptors; training-only descriptors remain outside the policy input.

A small authored C# bootstrap supplies the existing seeded setup and reference RNG draws. Native lanes do not call C# for individual moves. Qualification separately runs the authored C# host as an independent per-step oracle. Branch restore imports a job-bound sealed native checkpoint, shuffles only the unseen deck with the original System.Random shuffle indices, excludes the alternate fork action from learning through the existing branch convention, resets the versioned opening counters and retains prior causal insolvency costs. The original checkpoint stays internal to scheduling.

The bounded scheduler gives ready branches priority without blocking a lane on its sibling. At most twice the lane count of unfinished families can exist. A separate writer validates and atomically publishes a complete parent/child family once. A retry loads a validated existing family; changed duplicates, foreign jobs, incomplete families, stale replies and canceled publication fail explicitly.

Native output uses **`brass-lab-family/1.0.0`**, with **`training_admitted=false`**, immutable frozen-job metadata and an explicit native producer. Native scoring uses `rust-local-scoring/1.0.0`; it does not impersonate the C# recorder. The legacy replay validator still accepts its original C# envelope only. **B03 must implement strict producer/replay/source admission before these families can enter training.**

Neural and frozen historical/IL-model tables use the native lane when all seats are supported. Tables containing retained legacy V4 adapters are routed as complete reference jobs, preserving their exact policy assignment. Search-bearing or unsupported jobs are never replaced with a different policy. The B02 qualification accepts only the existing search-free ordinary recipe; V205 stays disabled and no search or teacher work is enabled.

## Short lab coverage

The default collection watchdog is **240 seconds**, with two diagnostic lanes at most. This is a bounded correctness smoke, not a long self-play session or a throughput qualification:

- Check the 13 pinned B01 roots, including dense and staged positions, against the reference projection and exact Python tensors. Candidate pages preserve the entire legal set.
- Select at most one self-play, one historical and one legacy table from the actual next frozen plan. Selection does not consume or modify saved scheduling counters. Missing table kinds fail the coverage gate rather than being synthesized.
- Complete the three tables and one branch continuation. For this diagnostic only, trigger the first eligible self-play fork so branch coverage does not depend on a rare random event. The evidence labels this override; the frozen job and alternative-move RNG remain unchanged. A branch-disabled recipe cannot pass this gate.
- Compare every native decision's view, legal order and features with the C# / Python oracles, then compare final results, every era's scoring maps/audits and prepared terminal/causal reward targets. The excluded fork action also receives feature parity checks.
- Validate complete-family joins, actual behavior denominators/RNG streams, policy attribution, immutable job identities and retry without recollection. The short development suite additionally covers cancellation, failed work, child-first completion, duplicate messages and stale/foreign data.

The report distinguishes reference-routed tables from native ones; it never claims full native V4 coverage. Component counters include Rust projection, encoding and transitions; adapter policy/inference envelopes, step IPC and drain costs; GPU queue/batch/service counters; serialized lane-buffer bytes; and Linux native peak RSS. Some timings are nested and the total includes qualification-oracle overhead. These are diagnostic costs, not an equivalent-work speedup claim. Transport tuning and matched throughput measurements remain B06.

## Transfer and prerequisites

Update the lab project checkout, then copy the B01 and B02 transfer ZIPs into the matching `distribution/` subdirectories. Extract B01 first and B02 second, before running either qualification, so both see the same current adapter sources:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-b01/b01-search-v1-update.zip
unzip -o distribution/rust-lab-b02/b02-search-v1-update.zip
```

Use the existing project environment with Python 3.11+, CUDA-enabled PyTorch, Linux linker, Rust **1.99.1** and Mono. The existing private rules dependencies, authored local-scoring host package and `artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe` must already be available on the lab. B02 builds its own bootstrap beside copied verified runtime dependencies; the source-only transfer ZIP contains no proprietary game DLLs, model weights or private run data. The compiler, bootstrap and runtime files are hashed in the build receipt.

If the committed recipe uses critic descriptors, its existing `il-kl/manifest.json`, sealed probe arrays and saved probe receipt must be present. The runner reads them; it does not construct a new probe or edit the run signature. Keep training stopped at a clean committed iteration while running B01 and B02. Both use the existing run lock and check that checkpoint/state/population/runtime pins remain unchanged. If training advances between the two qualifications, generate a new B01 receipt for the new committed boundary.

The current B02 transfer records search_v1 and the exact required B01 transfer/input hashes. B02 rejects older cache-only B01 receipts even if they report success. Its build, family producer and qualification receipt carry search_v1 provenance, while `search_enabled=false` remains explicit. All current bundles are under `distribution/`; keep older ZIPs as history, not extraction inputs.

## Qualification commands

Run from the lab project root. Replace `YOUR_RUN` with the existing run directory. Because B01 has not run yet, the simplest command runs B01 and then B02 only if B01 succeeds:

```bash
conda activate ai-training
bash scripts/run-rust-lab-b01-b02.sh artifacts/step5/YOUR_RUN cuda:0
```

It prints the new evidence directory and both receipt paths. The equivalent separate commands are:

```bash
conda activate ai-training
RUN=artifacts/step5/YOUR_RUN
B01_OUT="artifacts/rust-backend/$(date -u +%Y%m%d-%H%M%S)-b01/lab"

python scripts/qualify-rust-lab.py \
  --run "$RUN" \
  --device cuda:0 \
  --profile configs/search/lab-search-v1.json \
  --bundle distribution/rust-lab-b01/b01-search-v1-inputs.zip \
  --bundle-sha256 935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76 \
  --out "$B01_OUT"

# Run after B01 reports passed: true.
bash scripts/run-rust-lab-b02.sh "$RUN" "$B01_OUT/qualification.json" cuda:0
```

For an already passing **search_v1 B01 v2** receipt, B02 alone is:

```bash
bash scripts/run-rust-lab-b02.sh \
  artifacts/step5/YOUR_RUN \
  artifacts/rust-backend/YOUR_B01_OUTPUT/lab/qualification.json \
  cuda:0
```

The equivalent direct B02 command exposes the smoke watchdog and optional populated Cargo cache:

```bash
python scripts/qualify-rust-lab-b02.py \
  --run artifacts/step5/YOUR_RUN \
  --b01 artifacts/rust-backend/YOUR_B01_OUTPUT/lab/qualification.json \
  --device cuda:0 \
  --smoke-timeout 240 \
  --out "artifacts/rust-backend/$(date -u +%Y%m%d-%H%M%S)-b02/lab"
```

Add `--offline` only when the pinned Cargo dependencies are cached. Linux compilation has a separate 900-second watchdog per build/test command; this does not lengthen the collection smoke. There is no CPU-only qualification bypass and no skip-B01 switch.

## Evidence and development checks

Only a new **`qualification.json` with `passed: true`, `contract: brass-lab-b02/2.0.0`, and `method: search_v1`** qualifies refreshed B02. Keep its linked `build.json`, `commands.json`, `frozen-jobs.json`, `collection-smoke.json`, complete family files and logs. Failed or timed-out checks retain failure evidence and never produce a successful receipt. The whole diagnostic process group is terminated on watchdog expiry before the run lock is released. Collection artifacts are private trusted evidence, outside production replay and run directories.

Original B02 checks were development evidence only: the 38 B01/B02 Python tests, the existing scoring-receipt contract test, five Rust lane unit tests and C# bootstrap compilation. No full local game or Linux/CUDA qualification was run for this task. Original development logs are under `artifacts/rust-backend/20261028-b02/development/`; the refreshed search_v1 admission/package checks are under `artifacts/rust-backend/20261029-search-v1-bundles/`. The refresh passed **42 Python adapter tests**, Bash syntax checks, mocked wrapper success/failure checks and shared dependency/source-pin checks. No Rust/C# behavior change or new local native qualification was required.

To rerun the short Python adapter checks after editing:

```bash
PYTHONPATH=src python -m unittest discover -s tests/RustBackendLab -v
```

Both lab exit gates remain **Pending** until the user supplies passing server results. B03–B07, trainer adoption and throughput claims remain separate work.
