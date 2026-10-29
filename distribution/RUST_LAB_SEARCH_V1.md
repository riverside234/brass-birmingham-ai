# B01/B02 search_v1 qualification bundles

Refreshed 2026-10-29. **B01 lab qualification: Pending. B02 lab qualification: Pending.** All current ZIPs are under `distribution/` and explicitly allowed by `.gitignore` so they can be committed and pushed with the project.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| [B01 transfer](rust-lab-b01/b01-search-v1-update.zip) | Updated profile, adapter tests, Linux/CUDA runner and embedded input ZIP | `ae634e746dd1864bee281b36f2415112510aeb66a8c5de7faf8d7e4c75dac685` |
| [B01 inputs](rust-lab-b01/b01-search-v1-inputs.zip) | Frozen installed source/local receipts, 13 roots, 9 transitions; also embedded in B01 transfer | `935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76` |
| [B02 transfer](rust-lab-b02/b02-search-v1-update.zip) | Collector source/adapters, new B01 entry gate, combined runner and updated migration plan | `62a9d022bb3daa41f8ad874dfbbf40cb24e3b526c478d559db9db115e4725664` |

Use these new `search-v1` filenames. Older `b01-lab-update.zip` and `b02-lab-update.zip` remain historical cache-only packages; do not extract them over the refreshed files. The ZIPs contain no model weights, proprietary game assemblies or private lab state. They supplement the current project checkout and its existing private training runtime.

## Run both gates

From the updated lab project root, with training paused at a clean committed iteration, replace `YOUR_RUN` with the existing directory containing `state.json`:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-b01/b01-search-v1-update.zip
unzip -o distribution/rust-lab-b02/b02-search-v1-update.zip
bash scripts/run-rust-lab-b01-b02.sh artifacts/step5/YOUR_RUN cuda:0
```

Extract both packages before starting B01. Keep training stopped through B02. The runner prints its fresh evidence directory, runs B01 first, and starts B02 only after B01 succeeds. Results go to:

```text
artifacts/rust-backend/<UTC timestamp>-b01-b02-search-v1/b01/lab/qualification.json
artifacts/rust-backend/<UTC timestamp>-b01-b02-search-v1/b02/lab/qualification.json
```

Only `passed: true` with the new version-2 qualification contract and `method: search_v1` counts as a pass. B02 also requires the same unchanged run and qualified CUDA device. Failed output is retained; rerun into a new output directory after resolving the issue.

Use the existing Python 3.11+ CUDA-enabled environment, Linux linker, Rust 1.99.1 and Mono. B02 also needs the project's authored local-scoring runtime/private dependencies and existing Roslyn compiler at `artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe`. Existing committed population/anchor checkpoints and any required IL-KL probe files must be present. See the [B01 report](../ai-research/RUST_BACKEND_B01_REPORT.md) and [B02 report](../ai-research/RUST_BACKEND_B02_REPORT.md) for prerequisites, separate commands and optional offline builds.

## Run separately

```bash
bash scripts/run-rust-lab-b01.sh artifacts/step5/YOUR_RUN cuda:0
# After the new search_v1 B01 passes, use its printed receipt path:
bash scripts/run-rust-lab-b02.sh \
  artifacts/step5/YOUR_RUN \
  artifacts/rust-backend/YOUR_B01_OUTPUT/lab/qualification.json \
  cuda:0
```

B01 runs bounded rules/features/GPU diagnostics, without full games. B02 uses at most three tables and one branch, with a 240-second collection watchdog. Both preserve the production recipe; search/teacher training is not enabled and collected families are not admitted to training. B04 search integration/teacher readiness and B07 deployment remain separate.

The refresh passed 42 local Python adapter tests, shell syntax and mocked runner success/failure checks. Input/transfer hashes, installed-source identity and collector dependency compatibility were checked. These are development/package checks, not Linux/CUDA qualification.
