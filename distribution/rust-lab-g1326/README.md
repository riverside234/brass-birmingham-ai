# B01–B04 qualification with portable results

Use this combined bundle for `joint7-v11-2-grouped-lr-critic`. The latest supplied report is generation **1326**, with **424,320 games**:
`results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z`.

The checkpoint is `artifacts/step5/joint7-v11-2-grouped-lr-critic/generation-1326.pt`, SHA-256 `ada8d72195837cb399c13effb975cce19840e847080a7a0d4e03091235d85cbe`. The launcher checks the real lab run's committed identity; it does not resume from the exported log directory. The existing reviewed source transition covers this run. No saved signature or training state needs editing.

This update contains the original sealed B01/B02/B02.5/B03/B04 bundles plus a new launcher and result exporter. Their original hashes, source identities and admission checks are preserved. Use the new launcher below in place of the older direct commands to get automatic result export.

## Install and run on the lab

Keep training paused at a committed boundary for the entire qualification. Use the updated project checkout, existing private training runtime/checkpoints and the prerequisites in the B01 instructions (Rust 1.99.1, Mono, CUDA/PyTorch).

From the project root:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-g1326/rust-lab-b01-b04-g1326.zip
unzip -o distribution/rust-lab-b01/b01-search-v1-update.zip
unzip -o distribution/rust-lab-b02/b02-search-v1-update.zip
unzip -o distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip
unzip -o distribution/rust-lab-b03/b03-search-v1-admission.zip
unzip -o distribution/rust-lab-b04/b04-search-v1-teacher.zip

python scripts/run-rust-lab-qualification.py --stage all --device cuda:0
```

The defaults bind the named run, generation 1326 and the exact checkpoint hash above. If training advances, explicitly supply its new committed identity:

```bash
RUN=artifacts/step5/joint7-v11-2-grouped-lr-critic
G=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["iteration"])' "$RUN/state.json")
SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["history"][-1]["sha256"])' "$RUN/state.json")
python scripts/run-rust-lab-qualification.py --stage all --run "$RUN" \
  --expected-generation "$G" --checkpoint-sha256 "$SHA" --device cuda:0
```

Stages run sequentially and stop at the first failure or deferred qualification. B03 reuses B02's evidence; B04 still runs only two shadow tables at 4s/K16 after readiness passes. This launcher adds no games, optimizer updates or teacher enablement. All Linux/CUDA statuses remain Pending until qualified on the lab.

To run one stage, use `--stage b01`, `b02`, `b025`, `b03` or `b04`. B02 needs `--b01 PATH`; B03 needs `--b02 PATH`; B04 needs both `--b02 PATH --b03 PATH`. These must be original lab receipt paths printed by earlier runs, with matching source and committed training state. B02.5/B03 automatically receive the B04 transfer supplement.

## Push results back

Every invocation creates a new directory under:

```text
results/rust-lab-qualification/<UTC timestamp>-g1326-<stage>-<unique suffix>/
```

That folder is **not ignored by Git**. It is exported on success, failure, deferred readiness and handled interruption. Open `evidence/runner.json` for the overall status, then stage `qualification.json` files and logs. Reports of 1 MiB or more are gzip compressed; the export manifest records original and exported hashes and portable relative file paths. Failed launches without a stage receipt still include the runner error and console log.

Original evidence remains under `artifacts/rust-backend/` for subsequent gates. Build trees, executables, dependencies, model weights and replay/cache payloads are excluded from the Git copy. Original receipt bytes retain lab-local paths and hashes; the export is for review here, not authorization to reuse a lab qualification on another machine.

```bash
# On the lab, after the command finishes (including a nonzero exit):
git add results/rust-lab-qualification/
git commit -m "Add Rust lab qualification results for joint7 g1326"
git push

# On the current computer:
git pull
```

No commit or push is performed automatically. To export an older result, or recover after an unhandled machine/process termination:

```bash
python scripts/rust_lab_results.py artifacts/rust-backend/YOUR_EVIDENCE_DIRECTORY
```

Optional short local check of this export flow, without a lab qualification claim:

```bash
python scripts/run-rust-lab-qualification.py --stage b04 --fixtures-only --device cpu
```
