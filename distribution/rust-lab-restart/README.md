# B01–B04: latest restart iteration, task-only search_v1

Updated 2026-11-02. **Lab qualification: Pending. Production teacher: off.**

Use this version for the **latest committed iteration** of `artifacts/step5/joint7-v11-2-g1290-trained-restart`, for example **g390**. The launcher captures its generation, checkpoint hash and state identity once, then rejects any change during qualification. It does not select the old g370 checkpoint. The default recipe is [joint7-v11-2-g1290-trained-restart-g370.json](../../configs/step5/joint7-v11-2-g1290-trained-restart-g370.json); the config's g370 suffix identifies its revision, not the required training iteration.

DOMiNO is off. AZ_DB code and its intrinsic teacher objective are removed from the current sources. `search_v1` remains the default; search_v2 and V205 are disabled. Teacher searches use **4 seconds for search**, with **at most 1 additional second** for finalization, collection, verification and delivery. Both actions/replans retain the original deadlines. No search is admitted during the extra second. Two diagnostic tables run in separate sessions, each with one thread.

## Run on the lab

Pull the current project checkout first, including deletions. Do not extract an old B01/B02/g1330 source update over it. The six original stage archives inside this bundle remain sealed build/reference inputs.

With training paused at a clean committed boundary:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-restart/rust-lab-b01-b04-restart-r2.zip
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0
```

The launcher verifies the archive, source, removed-file inventory, full checkpoint identity and config before running B01, B02, B02.5, B03 and B04 in order. It stops at the first failed/deferred stage. It never trains, adopts a config, changes the production run, or enables its teacher. B03 restores a complete isolated diagnostic fork; B04 has exactly two shadow tables. No long strength campaign runs.

**Recipe reference:** the supplied g0370 report predates the g370 config revision. It is retained only as source evidence. Qualification is intended for a later iteration with the g370 recipe already adopted: direct GAE 0.95, 0.5% T2-only branches and PPG beta floor 1.0. The launcher compares that iteration's committed config to `--config`. B03 restores that iteration's full checkpoint and requires exact existing phase state; it does not repeat an already completed g370 adoption. The narrow pending-transition check remains only for a signature/checkpoint adoption boundary. No extra training iteration or config mutation is performed by qualification.

No generation/hash arguments are needed for a later iteration. To assert that the latest committed state must specifically be g390:

```bash
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0 --expected-generation 390
```

For an explicit checkpoint pin or a later reviewed config:

```bash
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0 \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation YOUR_GENERATION \
  --checkpoint-sha256 YOUR_FULL_CHECKPOINT_SHA256 \
  --config configs/step5/YOUR_CURRENT_CONFIG.json
```

A newer unreviewed source/config requires a new reviewed bundle. Explicit pins are assertions against the latest state; they never select or rewind to an older checkpoint. The resolved generation/hash/config are recorded in `evidence/runner.json`.

## Individual stages

```bash
python scripts/run-rust-lab-qualification.py --stage b01 --device cuda:0
python scripts/run-rust-lab-qualification.py --stage b02 --device cuda:0 --b01 ORIGINAL_B01_QUALIFICATION_JSON
python scripts/run-rust-lab-qualification.py --stage b025 --device cuda:0
python scripts/run-rust-lab-qualification.py --stage b03 --device cuda:0 --b02 ORIGINAL_B02_QUALIFICATION_JSON
python scripts/run-rust-lab-qualification.py --stage b04 --device cuda:0 \
  --b02 ORIGINAL_B02_QUALIFICATION_JSON --b03 ORIGINAL_B03_QUALIFICATION_JSON
```

Every attempt, including failures and deferrals, writes a fresh Git-visible folder under `results/rust-lab-qualification/`. Open its `evidence/runner.json` first. Push that new folder and pull it on the local computer. Retain original `artifacts/rust-backend/...` receipts on the lab for later dependent gates; exported review copies do not replace those receipts.

Prerequisites remain the existing CUDA Python environment, Rust 1.99.1, Linux linker, Mono, pinned Roslyn/private game runtimes, full run/checkpoints, retained PPG caches, seed/IL and opponent references. The ZIP contains source and small reports, not model tensors or proprietary binaries. A missing current-reward utility readiness receipt defers B04; there is no intrinsic-fit gate and no local-game readiness waiver.

`bundle.json` records the ZIP hash. `training-reference.json` distinguishes the historical report, requested recipe and latest-state qualification policy. `local-development.json` records short local checks and their limits. Historical archives remain unchanged; this bundle supersedes their launch instructions for current qualification. B05–B07 remain unimplemented and Pending.
