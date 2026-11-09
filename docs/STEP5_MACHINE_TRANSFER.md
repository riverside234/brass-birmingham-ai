# Step 5: transfer the current Rust training run

Updated for `joint7-v11-2-g1290-trained-restart`, committed g1284 in results commit `a21a13d` (8 November 2026). Use the actual committed generation if training has advanced.

Transfer a stopped run with `scripts/step5-transfer.py`. Code comes from the matching Git checkout; large immutable state and private runtime files travel as split `tar.zst` GitHub Release assets, or by your usual file-copy tool. Nothing in these commands uploads data until you explicitly upload the archive parts. Do not put checkpoint archives in Git history.

## What travels

The current native-run exporter requires compact mode and records `fresh-buffers/1.0.0` in its manifest. It preserves every included file byte for byte and checks SHA-256 before publication/restoration.

| Included | Purpose |
| --- | --- |
| `state.json`, latest full learner checkpoint, reviewed destination resume config | Weights, PPO/value/head Adam moments, RNG, generation, seeds, schedules and progress |
| Live historical/opponent models, incumbent, evaluation anchor, pool fingerprints | Preserve opponent identities and future sampling |
| Committed rating outcome summaries (`ratings/slots/.../games.jsonl`), slot spec/report/panels/bootstrap and compact rating history | Preserve immutable rating evidence and future PL refits; no action replays |
| Completed IL source/ancestry metadata, selected IL checkpoint and dataset manifests | Preserve strict IL provenance without copying demonstrations |
| `<run>/il-kl/probe.npz` and `<run>/il-kl/manifest.json` | The existing fixed 2048 **positions**, including encoded features and original reference probabilities |
| `<run>/seed/manifest.json`, when present | Small startup provenance for later report export; does not add unreferenced original seed tensors |
| Original and active Rust ordinary/teacher binaries, bootstrap DLLs, build/qualification receipts, calibration fixtures | Continue the adopted backend without losing its integrity bindings |
| Rules DLLs and runtime manifests from `training/runtime/distribution.json` | Destination preflight and the existing C# V4 worker |
| Frozen rating/pending-evaluation `search_runtime` receipts, runtime manifests and all pinned EXE/DLL files | Keep the original rated opponent identity; package as private runtime even in repo mode |
| Source-transition backups and transfer instructions/evidence | Recovery and subsequent transfers |

The destination config queues `win=1.0`, **`placement=0.05`**, `vp=0.001` with the existing 180-VP cap, and **zero insolvency reward charge**. Untied non-VP bonuses become `1.05 / 0.033333 / 0.016667 / 0`. Exact ties retain the existing pairwise placement and shared-win rules. The positive `.025` budget scale remains for compatible NN inputs; every event charge and resulting deduction is exactly zero. Current/source-server training keeps its old reward until the destination instruction authorizes the change.

The new reward contract `brass-combined-return/3.0.0` pays VP after canal scoring, then pays the difference at game end. With `f(v) = .001 × min(v, 180)`, the payments are `f(canal VP)` and `f(final VP) - f(canal VP)`. For 60 canal VP and 140 final VP, pay `.060` then `.080`; for 60 then 200, pay `.060` then `.120`. Apply the cap once to the cumulative total, not separately to each era increment. Win and placement pay at game end. Read the first authoritative rail observation after canal scoring, including automatic/forced transitions; retries never add another payment. Raw official VP labels and tie rules stay unchanged.

The existing NN/PPG/search values retain their full-game coordinate. Ordinary future-return/GAE calculations subtract the already-paid canal credit from both returns and the corresponding value coordinate. This preserves the tensor layout and Adam state, avoids requiring the NN to infer an earlier score, and keeps undiscounted search_v1 selection/backup equations intact. Replay preparation preserves the credit through compact IPC. Teacher value admission reconstructs the same public integer scores from the existing bound first rail root tensors, records both payments and future utilities, and keeps head-fitting targets in that same full-game coordinate. No extra game, search, feature encoding or transport request is added.

The original full IL replay, IL preparation/feature caches, ordinary replay, `.ppg-eight-cache`, `.ppg-value-cache`, `teacher-labels`, `search-values` and old diagnostic rollback replay payloads are omitted. Original IL rehearsal must be disabled, as it is in the current run. The exporter rejects an active rehearsal recipe rather than exporting its full data. It also rejects native `--mode full` and replay-buffer pins.

Committed rating `games.jsonl` files are compact **outcomes**, containing agent/seat identities, final ranks/VP/win shares and the slot-spec binding. They are required by `ratings.validate_state`, future PL history/refits and compaction, so they travel byte for byte. The exporter admits only the exact state-pinned `records` of a canonical committed rating slot, verifies the outcome-only schema and complete slot/spec/lineup bindings, and checks the original SHA-256. Fields containing views, choices or full records, unrelated JSONL, ordinary/IL replay and preparation caches remain rejected. Completed rating evidence and state receipts are never rewritten to avoid this check.

The existing IL probe is reused exactly. A committed probe is checked against its original manifest receipt, dataset-manifest/source/model bindings, array SHA-256 and semantic tensor hash. Loading it needs no original IL shards. Building a new/uncommitted probe still requires and verifies the source shards. Keep the small original dataset manifests and completed-IL checkpoint/metadata; copying only `probe.npz` is insufficient.

The current `features4-v11-2-batch7-il` origin is a **critic-upgraded IL directory**: its completion evidence is `upgrade.json` plus `candidate.pt`. It intentionally has no `state.json` or `report.json`. The exporter follows the exact hash-bound upgrade receipt to `features4-v11-1-batch7-il`, then recursively retains the parent's completed-BC state/report, selected checkpoint and dataset manifests. It checks the upgrade signature, compatibility, candidate ownership/hash/model identity and dataset lineage. Ordinary completed-BC directories still require their state/report. No missing metadata is fabricated, and no original replay/preparation cache is added.

The old B07 rollback manifest travels as audit evidence. Its omitted diagnostic payload is not a destination backup; use the retained transfer archive to restore the transferred checkpoint if necessary.

## Fresh buffers and Stage A

Restoration does not rewrite the archived checkpoint or state. A new destination receives `machine-transfer-resume.json` plus a hash-bound copy of the original state under `<run>/machine-transfer/`. At the first continuation, the loader authorizes only a buffer-reference reset at that exact checkpoint. Weights, all Adam moments, RNG, fixed teacher beta, PPG beta/trip history and ordinary progress remain intact. The normal reviewed reward migration updates utility metadata, invalidates old-reward readiness and retires old targets. It does not relabel old returns or reset the head/optimizer. A restart in the config-transition/checkpoint gap still binds the original checkpoint reward correctly.

Fresh ordinary games build the two-iteration PPG buffer. Fresh teacher games build the four-iteration policy-label and eight-iteration return buffers. PPG can use the first current age and then both ages; teacher policy training uses the available fresh label ages. No old targets are fabricated. The existing utility-readiness and collection checks still apply.

Auxiliary training and its stage counter pause while the return buffer fills. Eight consecutive fresh collection ages are required; the eighth age makes the buffer ready and allows that phase's head update. At g1284, saved Stage A is 4/10. It stays at 4/10 until eight consecutive fresh **new-reward teacher collection ages** are available, then resumes at 5/10. The old-reward utility-readiness receipt is invalidated; ordinary PPG must first issue a new-reward receipt before teacher search resumes. Thus rebuilding can take more than eight ordinary iterations. No new manual beta qualification is introduced, and the fixed teacher beta is preserved with its original provenance.

The one-time transfer receipt is saved in the next checkpoint/state. Restarting after a partial first iteration repeats the same authorized reset and recovers fresh work normally. Restarting after a committed iteration keeps the newly built buffers. Keep the instruction and original-state evidence with the run.

Stage B still warms up unconditionally for 20 iterations. Stage C keeps its 0.1-per-iteration trunk warm-up and hard joint KL cap of 0.003. Each Stage-C phase additionally draws eight whole games from the live eight-age training return buffer with an independent reproducible RNG. Those games are excluded from fitting in that phase. Loss is measured before the pass and after each 16-step interval; an increase stops the pass and restores that interval's weights, both Adam states and companion step count. Already accepted intervals remain. This does not roll back the stage schedule. The separate four newly collected held-out games remain on the existing ten-iteration diagnostic cadence (or each beta-qualification iteration).

## Source machine: stop and export

### Apply the complete revision on the old machine, restoring g1284

Keep `ROOT=/data/not_backed_up/yxu209/ai-training`. If the old command reached g1286, stop it first. The same-project driver below preserves the entire g1286 run (including incomplete next-iteration work) under the ignored `.training-rollback/` directory, restores the chosen g1284 archive into the original run path, explicitly installs the checkpoint-bound fresh-buffer/reward instruction even at the original absolute root, and launches the current teacher continuation. Project folder names and the machine do not change.

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=0

# The driver discovers the source export; no DOWNLOAD shell variable is needed.
python scripts/continue-rust-revised.py \
  --expected-generation 1284 --preserve-current-generation 1286 \
  --iterations 5 --share-target 0.25 --device cuda
```

The driver discovers archives under `$ROOT/artifacts/step5-transfer/`, including manual `g1284/...` exports and workflow `step5-transfer-<run-id>-<attempt>/...` folders. It reads small `export-result.json` receipts to match the run/generation, checks that all indexed parts exist with the recorded sizes, and groups duplicate copies by archive SHA-256. It prints the selected archive path, then performs the existing complete archive validation before retiring any data. Multiple distinct eligible exports produce a list and require `--bundle /absolute/path/to/chosen.parts.json`. `--bundle-directory /export/folder` changes the search directory. An explicit missing path explains that `DOWNLOAD` may be unset instead of raising a bare file-not-found traceback. This discovery step does not scan large part payloads or require a download variable, new export or network access.

Unsplit ZIP/zstd candidates must contain a valid `transfer.json` with the Step 5 transfer format/digest and matching run/generation. Runtime-only archives such as `private-rules-runtime.zip` and `private-rules-runtime-v4.zip` are excluded; their presence must not make the migration archive ambiguous. Discovery reads the small manifest, not model payloads. All file/part checksums are still verified by the normal restore before a run can move. The reported source export can also be selected directly without another code update:

```bash
python scripts/continue-rust-revised.py \
  --bundle "$ROOT/artifacts/step5-transfer/retry-20261108T225938Z/selfplay-migration.tar.zst.parts.json" \
  --expected-generation 1284 --preserve-current-generation 1286 \
  --iterations 5 --share-target 0.25 --device cuda
```

The driver verifies the archive generation and complete source identity before retiring the old run; it refuses a different current committed generation instead of silently discarding more work. It journals the reversible move and verifies the g1286 state/checkpoint backup on retries. A partially completed restore resumes, and an existing completed adoption is never restored/reset a second time. It preserves all archived g1284 weights, optimizers, RNG, fixed beta and residual-stage progress. Old replay is absent from the restored working run; the g1286 backup remains available. This uses the existing revised reward/transfer/teacher code, without a new learner-source transition.

The printed `revision` report checks **all** requested settings: placement `.05`, zero insolvency, canal then final-difference VP; Rust `search_v1`, V205 off, 4s teacher search plus up to 1s delivery, fast delivery `.5s`; branching 0%, teacher on with its existing fixed beta; A10/B20/C10 warmups, no 256-game gate, Stage-C hard joint KL `.003`, PPG beta floor `.25`; four diagnostic held-out games every tenth iteration with the beta-qualification override; and the Stage-C eight-game holdout from the eight-age return buffer with loss-increase stopping. The latter restores only the rejected 16-step interval (weights/both Adams/companion step count), retaining earlier accepted updates and the stage clock. Auxiliary fitting/counters pause until eight fresh new-reward teacher-return ages are ready. Existing Rust adapters are reused when matching.

`--prepare-only` performs the reversible restore/instruction preparation without starting training. The driver leaves the original checkpoint/state bytes intact until the normal reviewed training loader commits the reward transition. It never substitutes a g1284 checkpoint into a g1286 state. Keep `.training-rollback/` local; its full weights/replays do not belong in Git. Focused regressions for the server (not run locally): `PYTHONPATH=src:scripts:tests/Transfer python -m unittest -v test_inplace_revised_restore`.

### Export the stopped source run

Finish the active command normally. An uncommitted iteration/checkpoint must be recovered on the source before exporting. Publish these code changes to your repository and use the matching commit on both machines. The exporter supports the current reviewed qualified-source alias while recording the actual complete execution-source fingerprint (recipe 4, including Rust). Unknown source versions remain rejected. Exporting itself never runs an extra training iteration.

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
RUN=artifacts/step5/joint7-v11-2-g1290-trained-restart
BUNDLE="$ROOT/artifacts/step5-transfer/g1284/selfplay-migration.tar.zst"
cd "$ROOT"
python -m pip install -r requirements-transfer.txt

# Recommended: prepare the separate teacher variant now so it travels in the archive.
# This changes no source run/config/checkpoint and runs numeric/wire tests, not games.
python scripts/prepare-rust-post-transfer-reward.py \
  --run "$RUN" --expected-generation 1284

# Review included files and sizes; creates no archive and uploads nothing.
# Stop this command sequence immediately if validation/export fails.
set -e
python scripts/step5-transfer.py export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" --dry-run

python scripts/step5-transfer.py export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" \
  --mode compact --dependencies repo --format zstd \
  --zstd-level 3 --threads 8 --part-mb 1900
```

The preparation tool reuses an existing matching variant. A first build needs the existing Rust 1.99.1 toolchain and tiny sealed core source input; it preserves search_v1 and all old reward behavior while admitting the version-3 payout contract and an exact zero charge. It writes a separate selector, leaving the current source teacher active. If you skip source preparation, the destination preparation command needs Rust 1.99.1 and builds there; the archive carries the small sealed source input when available. No compiler/target cache or full IL data is transferred.

If an earlier export failed asking for `features4-v11-2-batch7-il/state.json`, update the exporter and retry the dry run/export above. A successfully prepared reward adapter remains valid: this transport-only fix changes no training/source fingerprint or native build, so do not repeat its tests/build or any completed training qualification. The trailing `ls ...part*` error only means the failed export produced no archive parts. List/copy parts only after export succeeds.

If the error instead rejects `ratings/slots/generation-.../games.jsonl` as replay data, sync **both** `scripts/step5-transfer.py` and `scripts/step5_transfer_runtime.py` and retry the same dry run/export. This transport-only repair retains the necessary rating outcomes under the narrow validation above; reuse the existing reward adapter and completed qualifications.

Output identifies the checkpoint generation, Git commit, archive size, parts and fresh-buffer/reward policy. Each part is bounded to 1900 decimal MB by default; the `.parts.json` index binds all parts. Keep every part and the index. The compressor streams the TAR without creating a second full uncompressed archive. Python 3.11+ is required; `requirements-transfer.txt` supplies zstd for versions that need it. `--format zip` is available for compatibility. Use a new bundle path for every export.

`--dependencies repo` omits only unchanged tracked code whose recorded bytes are proved reproducible from the export's Git tree: an exact blob or its uniform CRLF form. Normalized Git status alone is insufficient. Mixed line endings, clean-filter differences and any other working-tree bytes that cannot be proved reproducible are bundled automatically. The proof is bound to each record's byte count and full SHA-256; later source changes abort export. It bundles all required private rules/native runtime files even when their paths are tracked. `--dependencies bundled` also includes code. Neither mode includes original IL data or replay buffers for a native run. Never use `--dependencies external` unless you deliberately supply the exact pinned runtime files separately.

## Move the parts

Copy `selfplay-migration.tar.zst.part*`, including `.parts.json`, to the destination. For GitHub Releases, use the existing **Step 5 transfer release** workflow in `.github/workflows/step5-transfer-release.yml` on the source machine's self-hosted runner. Select the source checkout's exact published commit and use run path `artifacts/step5/joint7-v11-2-g1290-trained-restart`. The workflow uses an isolated transfer environment, verifies the stopped run, uploads parts and then publishes the Release. It never stops/starts training or updates the source checkout.

For a manual upload, create a new Release/tag targeting the export's `git_commit`, upload every part, then upload `.parts.json`. Download all assets on the destination. Do not upload the original IL dataset or its caches separately. Once uploaded, the assets no longer depend on the source machine or runner.

## Destination: restore and verify

Use a full Git clone containing the export's exact Git commit/Release tag. Keep the training code at its recorded source and update the transfer tools to the latest repair: `scripts/step5-transfer.py`, `scripts/step5_transfer_runtime.py`, `scripts/step5_transfer_restore.py` and `scripts/step5_transfer_frozen.py`. Newer transfer tools can restore an older archive without replacing themselves. Install the OS/Python/PyTorch/CUDA/Mono dependencies from [the lab guide](TRAINING_RECOVERY_LAB.md); the archive is not a conda/system image. Select your allocated GPU using `CUDA_VISIBLE_DEVICES`. Retained native binaries require a compatible Linux CPU/ABI; use preflight to check the destination before training.

```bash
# ROOT is the NEW checkout; DOWNLOAD contains all archive parts and the index.
ROOT=/scratch/bob/ai-training
DOWNLOAD=/scratch/bob/transfer-g1284
cd "$ROOT"
python -m pip install -r requirements-transfer.txt

set -e
python scripts/step5-transfer.py restore --root "$ROOT" --relocate \
  --bundle "$DOWNLOAD/selfplay-migration.tar.zst.parts.json" \
  > "$DOWNLOAD/restore.json" &&
python scripts/step5-transfer.py verify --root "$ROOT" \
  --bundle "$DOWNLOAD/selfplay-migration.tar.zst.parts.json"
```

No manual concatenation is needed. Restoration verifies all archive payloads and preflights every destination file before writing. Identical files already present in a clone are reused. Differing **Git-tracked text/config/source files** in the declared repository trees are backed up first, then replaced with the exact archive-pinned bytes. All backups finish before any replacement. This handles source-machine edits to tracked launch configs such as `configs/step5/joint7-v11-2-g1290-trained-restart-rust-shadow32-no-branch-share025.json`, and LF/CRLF differences. Git HEAD and index are unchanged; restored source/config overrides can appear as working-tree modifications.

If a repository dependency was omitted from the archive in `repo` mode, the restore tool can recover it from the archive's exact recorded Git commit, including an LF/CRLF variant only when its complete SHA-256 matches. Older exporters could incorrectly omit mixed-line-ending files because Git reported them unchanged. Legacy recovery now also handles a single LF/CRLF transition: the archived byte count determines the number of CR bytes, and both possible prefix/suffix forms are checked against the full archived SHA-256. This covers the reported `src/BrassLocalAI/Unity/manifest.json` and `src/BrassPatch/Persistence/manifest.json`; no new archive export is needed. It never enumerates arbitrary line-ending permutations or accepts normalized-only matches. Future exports bundle non-reproducible bytes instead.

The tool performs no fetch/reset/checkout. A missing commit or unresolved dependency remains a blocking error with all affected paths reported together. Keep using the **new** destination root, for example `ROOT=/data/not_backed_up/yxu209/brass-birmingham-ai`. If restore fails, verification cannot yet read the relocation mapping: its error now explains that restore must complete first. Do not switch `ROOT` back to the archive's historical `/data/not_backed_up/yxu209/ai-training` path. The `&&` above prevents the follow-on verification after a failed restore.

Backups and the interruption-safe receipt live under `artifacts/step5-transfer/restore-<archive-id-prefix>/`; `restore.json` lists their exact paths in `repository_reconciliation.backups`. The first destination-state identity is retained so retries cannot lose the fresh-buffer instruction after a partial restore. Later edits to an already backed-up file, changed backups, or a different journal/archive binding fail closed. Existing **checkpoints, run state, data, private binaries and untracked files** are never overwritten on a mismatch. `--strict-repo-files` retains the original policy of refusing any differing repository file. `verify` only checks; it never repairs/replaces files.

Current transfer tools stay in the checkout. Their older archive versions are retained as hash-verified snapshots in that receipt directory; only the listed non-training transport modules use this mapping. Other archived files remain at their original relative paths, and the full training-source fingerprint is still required. This keeps fixes available for retries and verification without weakening the archived model/config/data/source bindings. The archive itself is unchanged.

`--relocate` supports a different absolute project directory. `.step5-relocation.json` maps historical root components at access time, including Rust receipt/binary paths. Signed historical JSON is unchanged. Restored native workers retain their Linux executable bit. Keep the relocation mapping and restore receipt/backups with the checkout; do not edit path strings inside saved checkpoints or receipts.

Run archive `verify` before any subsequent training changes. Full verification against the old archive is expected to fail once new training commits change its state. Keep the archive so the original checkpoint can still be restored into a clean directory.

## Destination: preflight and continue

Mono is a separate Linux runtime dependency, even with the Rust training backend: preflight and C# opponent decisions still launch managed workers. The transfer archive and Python requirements do not install it. In the active training environment, check/install it before launching preflight:

```bash
conda activate ai-training
conda install -y --freeze-installed conda-forge::mono
command -v mono
mono --version
```

The [conda-forge Mono package](https://anaconda.org/conda-forge/mono) supplies the Linux executable without administrator privileges; [`--freeze-installed`](https://docs.conda.io/projects/conda/en/stable/commands/install.html) asks Conda to retain existing installed package versions. If a site-provided Mono is already available, load its module or use its absolute executable path instead of installing another copy. A `FileNotFoundError: ... 'mono'` means the worker launcher could not find this executable; fix its installation/PATH and retry the same preflight. No archive re-export/restore, adapter rebuild or training-source migration is needed for that error. Keep the activated environment or loaded module in every training/SLURM job, not just the interactive setup shell.

For a Mono executable outside PATH, use the existing global CLI option **before** the subcommand: `python scripts/step5.py --root "$ROOT" --config <restored-config> --device cuda --mono /absolute/path/to/mono preflight`. Pass the same `--mono /absolute/path/to/mono` before `train` in the continuation command. The generated commands use `mono` on PATH by default.

The restore JSON contains ready-to-run `preflight` and `resume` commands using the reviewed destination config, IL source, original run and expected generation. Run `prepare_reward` from the restore JSON first, then the generated preflight. A shipped, compatible reward adapter is reused without rebuilding. The source machine's old preflight does not substitute for a destination check. Ordinary native collection, Rust search_v1 and the existing C# V4 worker keep their adopted profiles. Do not rebuild/requalify an unrelated successful milestone just to transfer compatible, pinned binaries.

```bash
export CUDA_VISIBLE_DEVICES=0
# Fail before worker launch if Mono was not installed/loaded in this job.
command -v mono >/dev/null || { echo 'Mono is missing: install/load it as described above.' >&2; exit 1; }
python - "$DOWNLOAD/restore.json" <<'PY'
import json, shlex, subprocess, sys
r = json.load(open(sys.argv[1]))
if r.get('prepare_reward'):
    print(r['prepare_reward'], flush=True)
    subprocess.run(shlex.split(r['prepare_reward']), check=True)
# Execute only the reviewable preflight command printed by restore.
print(r['preflight'], flush=True)
subprocess.run(shlex.split(r['preflight']), check=True)
print('Continuation command:', r['resume'])
PY
```

Then run the printed `resume` command. It defaults to 20 additional iterations and includes `--resume-reviewed-source` for the existing reviewed native execution alias. Change its iteration count to suit your allocation. Do not pass `--initial`, create a new run, regenerate the probe or reset optimizers. The continuation reports fresh-buffer progress in `search_value.buffer_rebuild`; Stage A resumes automatically when ready.

After this revision has been adopted, the normal helper can also continue the same run:

```bash
python scripts/continue-rust-teacher.py \
  --expected-generation 1284 --iterations 8 --share-target 0.25
```

Use the actual committed generation for later commands. The first destination continuation adopts the queued reward only with its exact restoration instruction. Running the generated reward config against the untouched source run is rejected. It does not repeat completed beta qualification. Teacher mode stays on, branching stays 0%, teacher beta remains fixed and search_v1/V205 settings are unchanged.

### Missing frozen rating runtime in an older archive

If continuation reports `Frozen rating runtime missing: ... search-76a2f9ad7529c0e5e2117a36`, update the checkout with this repair and retain the stopped restored run. Older exports walked `path`/`sha256` receipts but missed the rating opponent's typed `id`/`recipe` reference. Future exports now explicitly retain enabled rating and pending-evaluation runtimes, including their original receipts/manifests and every pinned binary; export refuses a missing live dependency.

The Git supplement under `training/runtime/shared-search/bank-3dc65253a610191d66a5/frozen/search-76a2f9ad7529c0e5e2117a36/` contains the exact original `search.json`, `runtime.json` and authored executable. Its hashes are sealed in `supplement.json`; proprietary DLLs are reused by their exact hashes from the restored distribution. No private DLLs are added to Git. The frozen recipe, executable and runtime-manifest identities remain unchanged; old policy sources are not rebuilt or relabelled.

Run on the destination, with training stopped:

```bash
python "$ROOT/scripts/step5-transfer.py" repair-runtimes --root "$ROOT" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart
```

Then repeat the same five-iteration continuation command from `restore.json`. This is an additive repair: it does not rewrite state/checkpoints, reset rewards/ratings/Adam/RNG, change the archive or require the old machine. Keep the completed preflight and adapter qualifications. All source hashes and destination conflicts are checked before writing; mismatched existing files are preserved and block repair. `search.json` is published last, and interruption/retry reuses identical files. `--dry-run` previews the repair without writes. The repair tool requires no Mono worker/game execution.

Focused regressions for the lab, without games or benchmarks (not run locally): `PYTHONPATH=src:scripts:tests/Transfer python -m unittest -v test_frozen_runtime_transfer`.

The code and `results/` are separate Git repositories. For Git-visible training exports on the new machine, clone the results repository into `$ROOT/results` before continuation if it is not present:

```bash
git clone https://github.com/riverside234/ai-training-results.git "$ROOT/results"
```

## Older archives and debug checks

Recipe 1/2/3 archives still use their original fingerprint and byte-preserving restoration. Restore each older archive into matching old code, verify it, then use its explicitly reviewed source-upgrade workflow. Older archives do not acquire missing dependencies or a fresh-buffer reset instruction retroactively. Current native transfers use shared fingerprint recipe 4; the source and transfer tool call the same implementation inventory.

Focused checks for this revision use tiny authored tensors/archives, no gameplay benchmark or training campaign:

```bash
PYTHONPATH=src:scripts:tests/Transfer:tests/RustBackendLab:tests/Recovery:tests/R2:tests/FeatureEngineering \
python -m unittest test_native_transfer test_search_value_retention test_auto_il_kl.ProbeTests test_post_transfer_reward test_era_vp_reward -v
```

The lab transfer/continuation itself remains a destination check. Debug results and the Stage-C policy are recorded in [the implementation note](../ai-research/RUST_STAGE_C_TRANSFER_REVISION.md).
