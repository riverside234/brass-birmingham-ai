# B01–B04 qualification at g570

Updated 2026-11-03. **Linux/CUDA qualification: Pending. Production teacher: off.**

This bundle supersedes `rust-lab-restart-r2` for the current checkout. The previous transfer pinned an older implementation. Added Python modules participate in the implementation hash even if every file listed in that older transfer still matches. This transfer pins the complete fingerprint input inventory and names added, changed or missing dependencies when it rejects source drift. Integrity checks remain mandatory.

## Run on the lab

Pull the matching current checkout, including added files and deletions. Keep training paused at a clean committed boundary. From the repository root:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-g570/rust-lab-b01-b04-g570.zip
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0 \
  --expected-generation 570 \
  --checkpoint-sha256 06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9
```

The target run is `artifacts/step5/joint7-v11-2-g1290-trained-restart`. The expected full checkpoint is `generation-0570.pt`, model `18f42a48933c18095d96f1cc64b836d3f9aac92361a4e4ee7c423830d9a11a91`, at 182,400 ordinary games. The config defaults to [the exact recorded g570 config](../../results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z/recorded-config.json), which matches the prior g370 recipe. Qualification performs no recipe adoption, source-run write or training update. B03 restores all checkpoint/optimizer/RNG/PPG state into an isolated diagnostic fork.

Do not extract an older source ZIP after this one. The six original B01–B04 stage archives inside this bundle are sealed build/reference inputs; the current launcher selects them itself. The new source snapshot contains the current optional g570/entropy modules to keep the checkout consistent, but the recorded config has no entropy schedule or new branch/V4-history adoption. If the lab has already adopted a different config, use its exact reviewed config with `--config`; an unqualified recipe transition cannot be hidden by changing a hash.

If training has advanced, qualify the actual latest committed checkpoint without rewinding:

```bash
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0
```

The launcher captures its generation/hash/state identity once and stops if that state changes. The default config must still match. Explicit generation/hash arguments assert the latest state; they do not select an older checkpoint.

## Individual stages and results

```bash
python scripts/run-rust-lab-qualification.py --stage b01 --device cuda:0 --expected-generation 570
python scripts/run-rust-lab-qualification.py --stage b02 --device cuda:0 --expected-generation 570 --b01 ORIGINAL_B01_QUALIFICATION_JSON
python scripts/run-rust-lab-qualification.py --stage b025 --device cuda:0 --expected-generation 570
python scripts/run-rust-lab-qualification.py --stage b03 --device cuda:0 --expected-generation 570 --b02 ORIGINAL_B02_QUALIFICATION_JSON
python scripts/run-rust-lab-qualification.py --stage b04 --device cuda:0 --expected-generation 570 \
  --b02 ORIGINAL_B02_QUALIFICATION_JSON --b03 ORIGINAL_B03_QUALIFICATION_JSON
```

Every attempt, including failure or deferral, exports a fresh Git-visible folder under `results/rust-lab-qualification/`. Open `evidence/runner.json` first. Push the new result folder and pull it on the local computer. Keep original `artifacts/rust-backend/...` receipts on the lab for dependent stages. The reported old producer failure is not a passing qualification.

Search remains task-only `search_v1`, K16, four seconds of search plus at most one second total post-search work. B04 uses two independent one-thread sessions and exactly two shadow tables, with bounded typical/complex root probes. DOMiNO/AZ_DB and search_v2/V205 remain disabled. Required current-reward utility readiness still gates B04.

Prerequisites remain the existing CUDA Python environment, Rust 1.99.1, Linux linker, Mono, pinned Roslyn/private game runtimes, full checkpoint, retained PPG caches and opponent/seed references. The bundle includes source and reports, not checkpoint tensors or proprietary runtimes. `bundle.json`, `transfer-manifest.json` and `local-development.json` record the exact archive, sources and bounded local checks. Full lab qualification and B05–B07 remain Pending.
