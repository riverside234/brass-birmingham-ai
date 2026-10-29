# B03 replay admission and joint7 continuation

2026-10-29. **Implementation ready; lab qualification Pending.** B01, B02 and B02.5 lab qualification remain Pending. Default search remains **search_v1**, with V205 disabled. Production adoption remains B07.

`b03-search-v1-admission.zip` is a source supplement for the updated project checkout and existing private training runtime. It includes admission/continuation code, reviewed source transitions, bounded tests, runners and documentation. It contains no model weights or game binaries. `bundle.json` identifies the archive; `transfer-manifest.json` pins its files and trainer implementation. Earlier bundles remain immutable.

## Install before qualifying B01/B02

From the updated lab project root, using the existing Python 3.11+ CUDA environment and B01/B02 Rust/Mono dependencies:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-b01/b01-search-v1-update.zip
unzip -o distribution/rust-lab-b02/b02-search-v1-update.zip
unzip -o distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip
unzip -o distribution/rust-lab-b03/b03-search-v1-admission.zip
RUN=artifacts/step5/joint7-v11-2-grouped-lr-critic
```

Use the actual existing run directory if it differs. Pause training at a clean committed boundary and keep it paused throughout qualification. Extract **all four updates before running B01/B02**, because B03 needs B02 evidence sealed with this trainer and the frozen jobs/families. Extracting an earlier archive afterward would replace newer source files. A stale B02 receipt must be refreshed; it is not reinterpreted as a B03-compatible pass.

Since B01/B02 are still Pending, run their bounded qualification first:

```bash
bash scripts/run-rust-lab-b01-b02.sh "$RUN" cuda:0
```

Use the **B02 receipt path printed by that command** below. B01/B02 keep their existing short coverage: pinned roots/transitions, then at most three tables and one branch; B02 collection has a 240-second watchdog. See the [earlier prerequisites](../RUST_LAB_SEARCH_V1.md). B03 itself reuses those completed games.

## Qualify B03

```bash
B02=artifacts/rust-backend/YOUR_B01_B02_OUTPUT/b02/lab/qualification.json
G=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["iteration"])' "$RUN/state.json")
SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["history"][-1]["sha256"])' "$RUN/state.json")
bash scripts/run-rust-lab-b03.sh "$RUN" "$B02" "$G" "$SHA" cuda:0
```

The runner verifies the transfer, runs 19 CPU fixture tests with CUDA hidden, then holds the existing run lock while checking the exact committed generation, checkpoint hash, saved recipe, reviewed source path and B02 evidence. It copies the committed dependencies to an isolated diagnostic directory, restores learner weights, PPO/value and auxiliary Adam states, RNG, retained PPG ages, probe and persisted controllers, and compares native/reference Python preparation on the sealed B02 families. It performs **zero new games and zero optimizer steps**, and does not rewrite production state/checkpoints or enable teacher collection.

Output is printed as `artifacts/rust-backend/<timestamp>-b03/qualification.json`. Keep this file and its `continuation-preflight.json`, `restoration.json`, `replay-admission.json`, `continuation/diagnostic-fork.json`, `fixtures.log` and `lab-checks.log`. Only a full Linux/CUDA `passed: true` qualifies this B03 scope for the exact unchanged checkpoint. CPU fixtures alone leave lab qualification Pending. The CPU suite has a 180-second watchdog; the lab restore/admission child has a 300-second watchdog and runs under the parent-held production lock. Timeout/failure keeps evidence and releases the lock after child cleanup.

If the saved CUDA RNG topology differs, inspect the failure first. The existing explicit `--allow-cuda-topology-change` route is available through the Python command and records the topology adjustment; strict matching is the default. Do not edit checkpoint metadata or source hashes.

For CPU fixtures only:

```bash
python scripts/qualify-rust-lab-b03.py --fixtures-only
```

For an additional **separate** completed-IL handoff check, use the direct runner and supply the completed IL directory:

```bash
python scripts/qualify-rust-lab-b03.py \
  --run "$RUN" --b02 "$B02" --expected-generation "$G" \
  --checkpoint-sha256 "$SHA" --device cuda:0 \
  --il-run artifacts/step5/YOUR_COMPLETED_IL_RUN
```

That optional result does not replace established joint7 continuation and does not start a new PPO run.

The original B02.5 transfer remains sealed to its earlier implementation. To qualify its separate T1/device scope after installing B03, explicitly select the supplement. This checks both manifests and the unchanged prior archive:

```bash
python scripts/qualify-rust-lab-b025.py \
  --transfer-manifest distribution/rust-lab-b03/transfer-manifest.json \
  --run "$RUN" --expected-generation "$G" --device cuda:0
```

This optional B02.5 command includes its one small authored CUDA policy step. It does not update the production learner; B03 itself still performs zero optimizer steps.

## Recheck the diagnostic restoration

```bash
python scripts/restore-rust-lab-b03.py \
  --fork artifacts/rust-backend/YOUR_B03_OUTPUT/continuation \
  --expected-generation "$G" --checkpoint-sha256 "$SHA" --device cuda:0
```

The fork stores the original state as `payload/source-state.json` and original checkpoint bytes unchanged. The explicit path map verifies the original `training.fresh_run` binding; copying the folder alone is not a resume path. Normal training refuses this diagnostic folder. B05 will add the qualified next-update/commit path; B07 owns production promotion/rollback. Keep the original referenced external IL/runtime dependencies available and unchanged while rechecking the fork.

The B02 branch probe remains diagnostic data and cannot enter production PPO/PPG. Teacher labels use their separate B02.5 contract. See [B03 implementation and tested scope](../../ai-research/RUST_BACKEND_B03_REPORT.md) and [the migration milestone](../../ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md#b03--admit-replay-critic-data-and-existing-joint7-continuation).
