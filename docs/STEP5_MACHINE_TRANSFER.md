# Step 5: Transfer to Another Lab Machine

Copying only the last `iteration-*` directory or `best.pt` is insufficient. Resuming training also requires the latest learner's full checkpoint (including Adam and RNG state), `state.json`, the current historical opponent pool, the incumbent, the evaluation anchor, and imitation learning (IL) provenance. `best.pt` may be older than the latest learner.

The standalone `scripts/step5-transfer.py` defaults to **compact + repo + multithreaded tar.zst + 1900 MB parts**, transferred through **GitHub Release assets**. The destination machine checks out the code version associated with the Release, then restores the game rules runtime/DLLs and training state from its assets. Once the upload succeeds, restoring that snapshot no longer requires access to the source machine. **`restore --relocate --root <new-directory>` is now supported**, without administrator privileges, bind mounts, or recreating the original absolute directory. Restoration preserves historical training files; upgrade older training code as described below after verifying the original bytes.

The workflow supports both public and private repositories. Assets in a public repository are publicly downloadable; private repositories require read access. Here, "GitHub alone" means that project code, runtimes, and training continuation data all come from GitHub. The operating system, Python/conda, PyTorch, CUDA drivers, and Mono/OpenSpiel still need to be installed according to the guide; the archive is not an offline system image.

## 12 October: zstd and GitHub Releases

This revision changes the default dependency mode from `external` to `repo`. It includes every rules runtime listed in `training/runtime/distribution.json`, plus the files of the host selected by the saved config, in the Release assets, with SHA-256 verification for each file. Code and tracked workers still come from the matching Git commit; other restoration dependencies are packaged. Identical DLLs are deduplicated by content rather than compressed again for every runtime directory. Missing or modified DLLs cause export to fail, and the workflow confirms that all runtimes are packaged before uploading.

The defaults are `--format zstd --zstd-level 3 --threads 0 --part-mb 1900`. `--threads 0` automatically uses the available CPU count, accounting for process affinity and `SLURM_CPUS_PER_TASK`. On shared machines, set an explicit limit such as `--threads 8`; `--threads 1` uses a single thread. Small files may not provide enough work to keep every thread busy. Compression speed and size depend on the actual checkpoints/data; no fixed speedup is guaranteed.

Packaging streams TAR contents directly into zstd without first creating a complete TAR file. Restoration reads the parts directly without writing a complete TAR to disk. Content deduplication, SHA-256 verification, external dependency checks, and byte-preserving restoration remain in place. Single-file ZIP archives, split ZIP archives, and older manifests are still detected automatically by content, with no extra restore flags.

The script requires **Python 3.11+**. Python 3.14+ normally includes `compression.zstd`; on earlier versions, install the separate transfer dependency in the environments on both machines:

```sh
ROOT=/data/not_backed_up/yxu209/ai-training
python -m pip install -r "$ROOT/requirements-transfer.txt"
```

If a Python 3.14+ distribution was built without native zstd, install `python -m pip install backports.zstd==1.8.0`. Missing zstd produces installation instructions rather than silently falling back to ZIP. `--dry-run` and explicit `--format zip` do not require this dependency. It is separate from the training requirements and does not change the training signature.

## Current Self-Training Compatibility

The audited relocation revision is `step5-project-relocation/1.1.0`, with source fingerprint `f6f8028538c13fdaa5cbcd40ffde5e81fe5d9cb58b649875d8a5cd1ea8487191`. The shared path layer is included in this fingerprint. New archives record `source_recipe: 3` and `relocation_version: step5-project-relocation/1.1.0`. Recipe 3 adds `bank.py` and `legacy.py` to the existing Python and host/build inputs; `learning/paths.py` is also source-bound. Older archives are still verified using their original recipe 1/2/3 and do not need to be exported again. New code cannot substitute for old code to pass verification.

`configs/learning/selfplay-resume-sources.json` explicitly registers reviewed transitions from the original relocation version (`c9c9bb914477feea94b61b1698db93fd5e767d0f2f88b0c715f0cfea12074140`) and the **12 October comprehensive bug audit** (`8d19aff88a93edd9b734561a73c83a440f5d3517429be007d9913aceab8cffd1`) to this revision, along with previously supported transitions subject to their original config restrictions. Learning compatibility remains `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`. Completed IL sources are reviewed separately in `imitation-handoff-sources.json`; their original manifest, checkpoint, and ancestry signatures are still checked. Unknown source versions, changed learning contracts, and unregistered config changes remain rejected. This does not authorize resuming unfinished IL training from older source versions across a source upgrade.

**Restoration and code upgrades are separate steps.** First restore an existing older archive into a checkout of its matching old source and run `verify`. Then upgrade to this revision on the destination machine and continue the same run with `--resume-reviewed-source`; the source machine is not needed. Archives already using this revision can resume directly under matching source. The upgrade preserves the model, Adam/RNG state, counters, schedules, and provenance, and records the source transition as usual. The existing upgrade mechanism requires fresh ordinary games to recalibrate utility and rejects teacher targets from older source versions. Directory mapping does not bypass these checks.

When restoring an older recipe 3 archive into a different directory, also check `requires_code_update`. Earlier relocation versions without this `relocation_version` require an upgrade; recipe 3 alone does not mean that the training code includes the audit fixes. The sidecar format remains `step5-project-relocation/1.0.0`, so existing mappings do not need to be rewritten. Project maintainers must publish the new code to GitHub first; local uncommitted changes do not automatically become available on the destination machine.

## Directory Mapping and Subsequent Transfers

For example, the source may be `/data/not_backed_up/yxu209/ai-training` and the destination `/scratch/bob/ai-training`. The source directory is read from the archive's `source_root`; `--root` specifies the destination. A different destination requires explicit `--relocate`. Restoration at the original path continues to support the original command. Subsequent `verify` calls on a restored project recognize the mapping associated with that archive.

**`.step5-relocation.json`** at the project root is separate machine-local metadata. It records the destination root, historical source roots, archive ID, and its own digest, and is excluded from Git. The shared resolver maps historical directory components to the destination when accessing checkpoints, state/config references, IL ancestry, datasets/calibration, evaluations, retention cleanup, and runtime dependencies. `/old/project-other` does not match `/old/project`. Relative paths retain their meaning; IL filenames remain relative to their owning IL run. Newly generated model/evaluation/IL references prefer project-relative paths, while signed historical text is preserved.

Keep the mapping file with the project. Subprocesses and later sessions read it automatically without temporary environment variables. Restoration checks each file's SHA-256 and preserves checkpoint/report/config/state bytes. Conflicting destination files, conflicting or corrupted mappings, overlapping roots, `..`, symlink/junction escapes, and dependencies outside the project are rejected. External data must be brought into the project **before export** through its existing data/signature workflow; the tool does not automatically map arbitrary external directories. Relocation cannot supply an external dependency omitted from an older archive.

Mappings support at most 64 roots, including the current directory, and the serialized file must be smaller than 64 KiB. Exceeding either limit produces an error before restoration writes files, rather than reporting success after creating a sidecar that the training program cannot read.

To move again, first complete the supported source continuation on the current machine, stop the run, and run `export` from the **current root**. The new manifest carries all historical roots, and the next `restore --relocate` maps them directly to a third directory without needing either previous machine. Do not edit the sidecar manually or move the entire directory and reuse its old sidecar: it validates the destination root. The destination and historical roots must be disjoint, with none nested inside another.

`export` generates an exact resume config for the new state inside the archive, without writing it back to the source project. `verify` checks a **restored directory**, so run it after `restore` into the third directory. To verify the new archive in place on the exporting machine, first run `restore` there (without `--relocate`; matching files are preserved) to materialize the new config, then run `verify`. Do not ignore "Missing restored file" errors.

## Files Included by Default

| Content | Default treatment |
| --- | --- |
| `state.json`, exact resume config, best metadata, explicit pins | Included |
| Latest full checkpoint and resume checkpoints specified by state | Original bytes preserved, including optimizer/RNG state |
| Current historical opponent pool, incumbent, evaluation anchor | Included; the training opponent pool is not reduced to save space |
| Historical candidate models still on disk and not represented by the current pool | Included because the next population update may still use them |
| Latest iteration report, evaluation report/spec, selected champion evidence | Included |
| Unfinished evaluation caches, referenced models, separately selected hosts | Included in compact/full mode, including models and runtime dependencies outside the run but inside the project |
| State/report, selected checkpoints, and dataset manifests for the IL source and its ancestors | Included; unused epoch checkpoints are omitted |
| IL training data shards | Training shards included only while rehearsal is enabled and before `rehearsal_until`; omitted afterward |
| Consumed PPO/teacher/completed-evaluation replays and feature caches | Omitted unless explicitly pinned |
| Unreferenced old models, retention history, quarantine | Omitted |
| Code, repository configs, frozen rules AI material | Obtained from Git; only dependency hashes are recorded in the parts |
| Accelerated ordinary/search worker EXEs and manifests | Obtained from Git at `training/runtime/throughput/`, not added to training archive parts |
| Generated `artifacts/training-host/` | Under standard configs, destination preflight deploys Git workers using rules DLLs from the assets; if the config directly selects one of these hosts, all of that host's files are included |
| Game rules runtimes/DLLs and runtime manifests | **Included in default Release assets**; verified and deduplicated using the distribution and training/unfinished-evaluation host manifests, including source pins checked at runtime startup |
| Conda environment, CUDA drivers, other runs, source-machine preflight results and process locks | Omitted |

Identical content is compressed and stored only once. For example, identical model files at multiple paths do not consume duplicate space in the parts. Restoration recreates the original file layout. All source files are preserved, and the tool does not clean up the source machine. Actual size depends on the current pool and rehearsal requirements. `--dry-run` reports logical size, deduplicated size, category sizes, and the ten largest files.

## 1. Stop Training and Export on the Source Machine

Wait for the current training command to finish normally before proceeding. The script rejects active runs, uncommitted iterations/checkpoints, and damaged or missing dependencies. If an interruption left a partial iteration, first resume and finish it on the source machine using the original command.

Set `RUN` to your actual self-play directory and place `BUNDLE` outside that run and the IL run:

```sh
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
RUN=$ROOT/artifacts/step5/selfplay-best-v1
BUNDLE=$ROOT/artifacts/step5-transfer/selfplay-migration.tar.zst
python -m pip install -r "$ROOT/requirements-transfer.txt"

# Inspect the size first without creating an archive.
python "$ROOT/scripts/step5-transfer.py" export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" --dry-run

# These are the defaults, shown explicitly for review.
python "$ROOT/scripts/step5-transfer.py" export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" \
  --mode compact --dependencies repo \
  --format zstd --zstd-level 3 --threads 0 --part-mb 1900
```

This creates parts such as `selfplay-migration.tar.zst.part0001` and `.part0002`, plus `selfplay-migration.tar.zst.parts.json`. Each part contains at most **1,900,000,000 bytes (about 1.77 GiB)** by default. For the same archive, this reduces the part count to approximately 1/19 of the previous 100 MB scheme; archives smaller than 1900 MB need only one part. No additional complete archive is retained. The index is written last and records sizes and SHA-256 hashes for the parts and the complete archive. Output also records the iteration, game count, Git commit, `compressed_bytes`, part count, format, compression level, and configured thread count. Progress goes to stderr; stdout is parseable JSON. Use a new filename for each export; existing parts are never overwritten.

## 2. Transfer Through GitHub Release Assets

**Release assets are separate files uploaded to a release page. They do not enter Git commit history and are not downloaded automatically by `git clone/pull`.** Both lab machines only need access to GitHub; they do not need SSH connectivity to each other. The assets now include the rules runtimes, so a separate runtime ZIP transfer is unnecessary.

**The self-hosted runner only reads, packages, and uploads data from the source machine. Once every asset has uploaded successfully, the data is stored on GitHub; losing the source machine or runner afterward does not remove those assets.** The destination can download and restore independently. It restores the last successfully uploaded snapshot; later training progress that was not uploaded does not automatically enter the Release. Keep the repository, Release/tag, and every part. Release assets do not use the time-based expiration mechanism of Actions artifacts.

| Repository visibility | Release asset access | File size limit |
| --- | --- | --- |
| Private | Authentication and repository read access required | Each asset must be smaller than 2 GiB |
| Public | Public downloads; individual published assets cannot be made private | Each asset must be smaller than 2 GiB |

Repository privacy controls access, not the size limit. A Release supports at most 1000 assets, including the index. The default 1900 MB parts stay below 2 GiB. The usual Git limits of 100 MiB per file and 2 GiB per push do not apply to these assets. [GitHub Release documentation and limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

### Recommended: Package and Upload with GitHub Actions

The workflow is [`.github/workflows/step5-transfer-release.yml`](../.github/workflows/step5-transfer-release.yml), named **Step 5 transfer release**. It is triggered manually through **Run workflow**. Once triggered, it packages, verifies, uploads, and publishes the Release automatically. It does not stop training or start a resumed training run.

One-time setup:

1. Commit and publish the workflow, transfer script, `requirements-transfer.txt`, and matching training code to GitHub. The workflow must exist on the default branch before GitHub displays its manual trigger.
2. Register a Linux runner on the **source lab machine that holds the original run**, following **Settings → Actions → Runners → New self-hosted runner** in the repository. Give the machine a unique custom label, such as `step5-transfer-lab1`, and enter it as the `runner_label` input. The service account needs read/write access to the project directory. Do not use the active training project as the runner's working directory. [GitHub self-hosted runner labels](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow)
3. The runner's PATH must include `git`, `gh`, and Python 3.11+ with `venv` support. Use `python_executable` to specify a full Python path if needed. Packaging does not require PyTorch/CUDA; the workflow creates a separate environment in the runner's temporary directory and installs the transfer dependency there.

For each transfer, wait for the current iteration to finish and stop training normally. Confirm that the code associated with the saved run on the source machine has been committed and pushed to GitHub. Then open **Actions → Step 5 transfer release → Run workflow**, select the branch/tag matching the source machine's `HEAD`, and enter:

| Input | Default / purpose |
| --- | --- |
| `runner_label` | `step5-transfer`; replace with the source machine's unique label |
| `source_root` | `/data/not_backed_up/yxu209/ai-training` |
| `run_path` | `artifacts/step5/selfplay-best-v1`, relative to the project root |
| `python_executable` | `python3`; requires 3.11+ and venv |
| `threads` | `0` for automatic selection; set a count such as `8` on a shared machine |
| `zstd_level` | `3`, range 1–22 |
| `part_mb` | `1900`, range 1–2000 MB |

GitHub-hosted `ubuntu-latest` cannot access the source lab's `/data/...`, so this workflow requires a self-hosted runner. Jobs remain queued if the runner is not registered or its label does not match. The workflow does not perform checkout, pull, or reset. It fails with a clear error if the source version differs, code/config changes are uncommitted, the run is active, or an iteration is uncommitted.

The workflow uploads to the **current repository** using its built-in `GITHUB_TOKEN` with `contents: write`; no separate personal token is required. Release tags use `step5-transfer-<run_id>-<run_attempt>`, and local files are stored under `artifacts/step5-transfer/<tag>/` on the source machine. It creates a draft, uploads every part, uploads `.parts.json` last, then publishes a prerelease without marking it as Latest. On failure, the draft and local archive remain, and the original training data is preserved. A rerun uses a new attempt/tag. When complete, the Actions summary provides the Release link and destination download commands.

The workflow supports public and private repositories without requiring a private repository. If the code repository is public, the rules DLL and training data assets are public too; a private repository restricts access to users with read permission. The word `private` in a filename, or changing a draft into a prerelease, does not change repository access permissions.

### Manual Upload of an Archive Already Created on the Source Machine

After installing GitHub CLI, run `gh auth login` on the source machine with an account that has repository write access. The following example uses the current code repository. Choose a new `TAG` each time; the code commit must already have been pushed:

```sh
set -euo pipefail
REPO=riverside234/ai-training
TAG=step5-transfer-20261012-01
gh release create "$TAG" --draft \
  --repo "$REPO" --target "$(git -C "$ROOT" rev-parse HEAD)" \
  --title "Step 5 transfer $TAG" --notes "Stopped Step 5 training state; restore using the parts index." \
  --prerelease --latest=false
for part in "$BUNDLE".part[0-9]*; do
  gh release upload "$TAG" "$part" --repo "$REPO"
done
gh release upload "$TAG" "$BUNDLE.parts.json" --repo "$REPO"
gh release edit "$TAG" --repo "$REPO" --draft=false --prerelease --latest=false
```

Do not run `git add/commit/push` on these assets or remove `artifacts/` from the ignore rules. The default `--dependencies repo` already includes the rules DLLs, so the old runtime ZIP does not need to be uploaded separately. [GitHub CLI upload command](https://cli.github.com/manual/gh_release_create)

## 3. Check Out Code, Download Assets, and Restore on the Destination

Choose any new directory where you have write access, preserving the run's relative layout. This Bash example restores the old `/data/not_backed_up/yxu209/ai-training` into `/scratch/bob/ai-training`; adjust `ROOT` for your storage. `TOOLS` is a **separate checkout of the updated tools**, allowing restoration of Releases whose older code does not yet support `--relocate`. Both checkouts come from GitHub, and the source machine can be offline.

```bash
set -euo pipefail
ROOT=/scratch/bob/ai-training
TOOLS=/scratch/bob/step5-transfer-tools
REPO=riverside234/ai-training
TAG=step5-transfer-20261012-01  # Replace with the actual Release tag from the Actions summary.
CODE_REF=main                 # Use a published commit/branch containing this relocation revision; prefer a fixed SHA.
# Run gh auth login when first using GitHub CLI; private repositories also require read access.
# For private HTTPS clones, gh auth setup-git can configure Git credentials.
mkdir -p "$(dirname "$ROOT")"
git clone --branch "$TAG" --single-branch "https://github.com/$REPO.git" "$ROOT"
git -C "$ROOT" rev-parse HEAD  # Must match the Source commit in the Release notes.
git clone "https://github.com/$REPO.git" "$TOOLS"
git -C "$TOOLS" switch --detach "$CODE_REF"
CODE_COMMIT=$(git -C "$TOOLS" rev-parse HEAD)
python -m pip install -r "$TOOLS/requirements-transfer.txt"
DEST=$ROOT/artifacts/step5-transfer/$TAG
mkdir -p "$DEST"
gh release download "$TAG" --repo "$REPO" \
  --pattern 'selfplay-migration.tar.zst.part*' --dir "$DEST"
BUNDLE=$DEST/selfplay-migration.tar.zst.parts.json

RESTORE_JSON=$DEST/restored.json
python "$TOOLS/scripts/step5-transfer.py" restore \
  --root "$ROOT" --bundle "$BUNDLE" --relocate > "$RESTORE_JSON"
python "$TOOLS/scripts/step5-transfer.py" verify \
  --root "$ROOT" --bundle "$BUNDLE" --relocate

# Older relocation code without this relocation_version also needs an upgrade; verify original files first.
NEEDS_UPDATE=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["requires_code_update"]))' "$RESTORE_JSON")
if [ "$NEEDS_UPDATE" = 1 ]; then
  git -C "$ROOT" fetch origin "$CODE_COMMIT"
  git -C "$ROOT" switch --detach "$CODE_COMMIT"
fi
```

The `part*` pattern downloads both numbered parts and the `.parts.json` index. The Release tag points to the commit used at export time, preventing the destination from accidentally using later training code. For an existing destination repository, first check for changes you need to preserve, then run `git fetch origin tag "$TAG"` and `git switch --detach "$TAG"`. You can create a branch from there if you need to change code after restoration. Do not substitute the Release page's automatically generated **Source code (zip/tar.gz)** downloads for the training assets. [Asset download command](https://cli.github.com/manual/gh_release_download)

Restoration reads the parts directly, without manual concatenation or an additional large TAR/ZIP file. It checks every part, the archive contents, and Git code dependency hashes before restoring the rules runtimes and training files. Running `import-runtime` is unnecessary. Matching destination files can be reused; missing external dependencies or different contents stop restoration, and existing files are not overwritten.

The initial checkout of the old tag satisfies the archive's **original source/external file hashes**. Restoring an old archive directly into a newer checkout produces conflicts; use the two-step process above rather than removing hash checks or editing state. The tool depends on `src/brass_ai/learning/paths.py` from the same checkout, so do not copy only the updated script. If Git reports local changes or conflicts, preserve and inspect those files instead of forcing an overwrite. Updated training source must contain the corresponding explicit review receipts; choosing a newer commit alone does not establish compatibility.

Older exporters may have omitted a host selected separately by an unfinished evaluation, or runtime source pins outside the code directories. Such archives can continue only when **exactly matching** dependencies are supplied separately; the transfer tool cannot recover files that were never uploaded. The updated exporter rejects missing dependencies early. `verify` establishes the integrity of the archive and its listed files; it does not replace actual preflight, data/IL checks, or resume signature verification.

Run `verify` after restoration and **before** upgrading code or resuming training. Upgraded source differs from the old archive, and subsequent training updates state/checkpoints, so either change causes full verification against the old archive to fail as expected. Keep the original archive indefinitely if desired; it can be restored and verified again in another clean directory.

## 4. Run Destination Preflight, Then Resume the Original Run

Install and activate the `ai-training` conda environment, then install PyTorch/CUDA, Python dependencies, and Mono/OpenSpiel according to the [training guide](TRAINING_RECOVERY_LAB.md#transfer-and-preflight). The transfer script requires Python 3.11+. Install the zstd dependency as described at the start of this document; legacy ZIP mode still needs only the standard library.

`restore` and `verify` output two complete commands:

1. `preflight`: validate the device and rules runtimes on the destination; the source machine's successful preflight record cannot be reused.
2. `resume`: continue for 20 iterations using the saved config, original `--run`, and required `--imitation-run`. You can change the number of additional iterations; do not pass `--initial` again.

Run `preflight` first, then `resume` after it succeeds. Existing self-play state restores the PPO checkpoint. `--imitation-run` only validates the IL source; it does not repeat IL. Schedules, the historical pool, teacher budgets, and counters are preserved.

Continue with the following Bash commands, reusing the variables from the previous section. Paths are read from the restore output, supporting different run names and archives with or without IL provenance:

```bash
set -euo pipefail
CONFIG=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"])' "$RESTORE_JSON")
RUN=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"])' "$RESTORE_JSON")
IL_RUN=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["imitation_run"] or "")' "$RESTORE_JSON")
RESUME_ARGS=()
if [ -n "$IL_RUN" ]; then RESUME_ARGS+=(--imitation-run "$IL_RUN"); fi
if [ "$NEEDS_UPDATE" = 1 ]; then RESUME_ARGS+=(--resume-reviewed-source); fi

python "$ROOT/scripts/step5.py" --root "$ROOT" --config "$CONFIG" --device cuda preflight
python "$ROOT/scripts/step5.py" --root "$ROOT" --config "$CONFIG" --device cuda \
  train --run "$RUN" --iterations 20 "${RESUME_ARGS[@]}"
```

For CPU validation, replace both occurrences of `cuda` with `cpu`. In an older archive's restore JSON, `requires_code_update: true` means that you must install mapping-aware training source first and use `--resume-reviewed-source` for the first source continuation. The ordinary `resume` string in the output does not itself approve the upgrade. A source SHA absent from the review list produces an explicit error. After the upgrade, later sessions can resume with the same config/run, and the mapping remains effective.

Updated preflight also installs and checks both accelerated workers. Its `throughput_workers` output should show `optimized: true` for both ordinary and search. The required rules DLLs have already been restored from the Release assets. Current self-training defaults to one GPU; GPU allocation is a destination runtime option and is not copied automatically from the source machine. If you actually use multiple GPUs for inference, set `CUDA_VISIBLE_DEVICES` for the destination allocation and add `--inference-devices auto` before the `train` subcommand in the generated resume command. Do not add this option to `preflight`.

If the source machine is still available, validate a destination restoration before retiring it. Later restoration of that snapshot does not require the source machine. Different GPUs, library versions, and search thread scheduling can produce different subsequent results. The tool preserves saved-file bytes and restoration dependencies; it does not guarantee identical game results across machines.

## Other Modes

- `--mode full`: also retains historical files in the selected self-play/IL directories for complete backups that need old replays/traces. It is not required for the current transfer workflow.
- `--dependencies repo`: **default**. Clean tracked code comes from Git; all rules runtimes/DLLs, selected hosts, and other untracked dependencies are included in Release assets.
- `--dependencies bundled`: includes both code and rules runtimes, producing a larger archive. The restore script and Python/zstd must still be available first. The automated workflow always uses `repo`.
- `--dependencies external`: retains the old behavior, requiring code and runtimes to be supplied separately. It does not meet the goal of restoring solely from GitHub code and the current assets, so the automated workflow does not use it. Older external archives do not automatically gain missing DLLs.
- `--format zip`: uses legacy ZIP/DEFLATE export; use `.zip` for the corresponding bundle filename. Restoration detects ZIP/zstd automatically by content rather than guessing from the extension.
- `--part-mb 0`: disables splitting and creates one `.tar.zst` (or explicit ZIP). For manual Release uploads, ensure the whole file is smaller than 2 GiB. The workflow uses parts so index verification is always available.
- `--part-mb 100`: preserves compatibility with ordinary Git transfers. 100 MB is below GitHub's 100 MiB per-file limit, but each push is still limited to 2 GiB total and requires batched commits/pushes. Default 1900 MB parts are intended for Releases and cannot be added directly with `git add` and pushed. [Large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [push limits](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)

## Local Validation

### Directory Relocation Audit (1.1.0)

The following issues were reproduced and then fixed, without changing real training data or uploading/pushing anything:

| Issue | Cause and fix |
| --- | --- |
| Separate hosts for unfinished evaluations and some runtime source pins were omitted from archives | The original collector only traversed the training host and fixed code trees. Compact/full modes now collect evaluation models, host manifests, and source files checked at startup. Missing dependencies fail before archive publication. |
| Existing evaluations could not be reused after spelling the same host with a new absolute or relative path | The original spec compared host text exactly. That field now checks path identity after component mapping, while all other config fields still require exact equality. Original spec/provenance bytes and digests are preserved. |
| Some malformed archives incorrectly returned `verified: true` | File/directory or temporary run-lease conflicts caused restoration to treat temporarily created paths as restored files; oversized mappings could also create unreadable sidecars. These manifests, active markers, non-directory parents, and oversized mappings are now rejected before writing. Case variants of private configuration paths and non-portable filenames are also rejected. Inert OS guard files in older full archives are still verified and restored byte-for-byte; new exports omit them. |
| Runtime installation could write through links, or a cached return could bypass checks | Ordinary/search installers now validate source, destination, and destination files before copying or returning cached results. Preflight cannot write through a linked directory before reporting an error. Windows junction regression checks were actually executed. |

Path tracing covered the following call boundaries. Low-level checkpoint/replay/inference APIs receive actual file paths resolved by their callers; old strings in historical JSON are not replaced globally:

| Stage | Path and integrity boundary |
| --- | --- |
| export | `inventory → below/resolve_project_path`: state/live models, IL ancestry, dataset/calibration, pins, pending evaluations, runtime/source pins; hashes of original bytes. |
| restore / verify | Manifest, parts, and payload verification → layout/destination conflict checks → atomic writes without overwrite → source verification using the original recipe → separate sidecar. |
| preflight | Distribution/assets, frozen bank, LegacyWorker, rules/accelerated workers; installation inside the new directory, with DLL/source pins and actual rules parity still checked. |
| resume | `load_resume` checks checkpoint/state/model/generation/signature and restores Adam/RNG; upgrades require an explicit source receipt. Historical references resolve through `project_path`. |
| IL / rehearsal | Original manifest/parent/checkpoint bindings for completed imitation; `load_dataset` resolves shards/calibration in the parent before passing them to spawned readers. |
| workers / evaluation | Actor subprocesses read the sidecar from the destination root; the parent creates jobs under the destination, and inference loads resolved checkpoints. Existing evaluations retain the original spec digest. |
| retention / re-export | Live/pinned/cleanup references continue through the mapping; re-export uses current source/state and carries historical roots, so the next destination needs neither previous directory. |

The final unit-test commands are below. First install the transfer/training dependencies and add `src`, `tests/Transfer`, `tests/Recovery`, and `tests/R2` to `PYTHONPATH`:

```sh
python -m unittest test_relocation_audit test_relocation test_step5_transfer -v
python -m unittest test_audit_resume test_time_resume test_resume_evaluations test_imitation \
  test_audit_retention test_revision test_training_runtime_review test_step5_working_directory -v
python -m unittest test_native_runtime_review.NativeRuntimeReviewTests.test_search_transformation_is_pinned_and_changed_helper_is_rejected -v
```

On Windows / Python 3.12 / PyTorch 2.8 CPU, **all 57 transfer/archive tests passed**. The related recovery group had **36 passes and 2 skips out of 38 tests** (actual CUDA hardware and POSIX removal of the current working directory). The runtime source-pin check had **1 pass**, for **96 distinct tests: 94 passed and 2 skipped**. Junction checks initially skipped inside the sandbox were later executed successfully in a normal user context without administrator privileges. Relevant existing tests had no failures. Source compatibility tests also reject unknown source versions, unreviewed config changes, and incorrect checkpoint/state bindings. Actual CPU checkpoints were compared field by field for weights, Adam, RNG/next random output, and complete state.

Original failure logs, final unit results, and native integration logs are under `artifacts/step5-relocation-audit/`. The first native exercise passed destination preflight and PPO continuation, but nested test directories produced a 267-character Windows retention rename path and hit the system path limit. The exercise was rerun using shorter temporary directories. On Windows, project and generated-file paths must still fit the system's supported lengths; this revision does not change system long-path settings. The exercise uses a separate new evaluation config that retains replays to test actual cache reuse in the third directory, without rewriting the archived training config or an existing spec.

**The final exercise with real runtimes passed.** `tests/Transfer/smoke_migration_audit.py` uses the actual historical exporter from Git commit `05def96720df7635626b7a3ea0de1a4ab2888da3` and constructs a bounded fixture with absolute host/checkpoint references before archiving. After deleting the source directory, it restores into a second directory, verifies **2,799 files**, upgrades to the reviewed source above, and completes real DLL preflight, one two-game PPO iteration, and four evaluation games. It then exports 45 test parts of 1 MB, deletes the second directory, restores into a third directory, and verifies **2,819 files**. The third directory passes preflight again, **reuses the original replay cache for all four evaluation games with unchanged spec bytes and results**, and completes another training iteration. Training counters advance through **iteration 2 → 3 → 4 and games 4 → 6 → 8**. The second move adds no source transition. The 1 MB size is only for split-archive stress testing; the actual default remains 1900 MB.

The exercise blocks Python filesystem access to the first two directories in independent Python launches and spawned workers. Both directories remain absent at completion, and every file hash in the input test run is unchanged. The execution context's administrator flag is false. Elapsed time was approximately **864 seconds**; see `native-result.json`, the `second-*`/`third-*` logs, and `user-context.json`. `scope-check.json` confirms that unrelated uncommitted files and prior review receipts were preserved. Unit results are in `final-transfer-tests.log`, `final-tests.log`, and `native-source-pin.log`.

Offline simulations of the public/private Release workflow, actionlint, and Bash syntax checks for this document's six shell blocks also passed. No actual GitHub uploads/downloads were performed. Actual Linux/CUDA machines, step-by-step consistency across devices, archives larger than 2 GB, and full playing-strength evaluation were not tested. CPU restoration of the model/Adam/RNG and these bounded exercises do not establish identical subsequent training across different devices and clock budgets. Dependencies omitted from older archives still need to be supplied separately, and system/Python/Mono/CUDA/OpenSpiel environments must still be prepared according to the guide.

### Previous Validation Record (1.0.0)

The relocation revision was validated on **Windows / Python 3.12 / PyTorch 2.8 CPU**. The final combined run had **46 tests** (30 existing transfer tests, 11 new relocation tests, and 5 audit/time-budget continuation tests): 45 passed, and one junction test was skipped because the sandbox could not create links. Junction escape rejection, atomic publication without overwrite, invalid mappings, and old-archive restoration at the original path were then tested separately in a normal user context; **all 4 passed**. Another **38 related recovery/IL/evaluation/retention/runtime/launch-directory tests** ran: 36 passed, with one CUDA and one POSIX skip. Five overlapped with the combined run and are not counted twice.

The new cases export a temporary project, **delete its source root**, restore into another directory, then re-export and move into a third directory. Coverage includes absolute/relative paths; ZIP/zstd/parts and legacy recipe 1/2 manifests; original bytes for every archived file; mapping access from later independent processes; IL ancestry and real checkpoints; parallel rehearsal readers and calibration checks; evaluation model dependencies and original spec bindings; corrupted archives; destination conflicts; invalid mappings/out-of-project references; and retention dependency protection. A real utility-head checkpoint is restored through the existing `load_resume`/`resume_optimizer`, comparing all weights, Adam state, RNG (including the next random output), and complete state. Tampered checkpoint/state bindings remain rejected. Evaluation cases test real model loading and reuse of existing summaries/specs, without a full playing-strength evaluation campaign.

**The historical-archive exercise with real runtimes passed.** `tests/Transfer/smoke_relocation.py` extracts historical code from commit `05def96720df7635626b7a3ea0de1a4ab2888da3` and uses that version's actual exporter to package a committed two-game-per-iteration fixture. It deletes the temporary source project, restores at a different root, and verifies **2,799 files** individually. Original state/checkpoint bytes remain intact during restoration. It then updates destination source, runs actual two-game rules parity and preflight for both accelerated workers, and resumes one iteration with `--resume-reviewed-source`. The result is **iteration 2 → 3 and games 4 → 6**. Original checkpoint bytes, the original state backup, and the source transition record all pass verification. The input fixture was unchanged; total elapsed time was approximately 777 seconds.

Evidence for this revision is under `artifacts/step5-relocation-validation/`: `final-tests.log`, `final-security.log`, `recovery-tests.log`, `native-smoke-result.json`, `native-preflight.log`, and `native-resume.log`. The exercise shares an installed third-party Python environment; system dependencies are not packaged. No actual network downloads/uploads or transfers between two Linux lab machines or actual CUDA devices were performed. The destination must still run preflight.

The workflow also passed actionlint 1.7.12, Bash/embedded Python syntax checks, and simulated public/private repository uploads in `tests/Transfer/check_release_workflow.py`. Release download behavior is unchanged, and a failed part or index upload still prevents publication. An actual GitHub self-hosted runner was not exercised.
