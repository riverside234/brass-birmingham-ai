# Export reports for analysis

Run from the lab project. This uses Python's standard library; it does not load
models or launch training.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python scripts/export-training-reports.py --output results/joint7-review-0060
```

The default source is `artifacts/step5/v4-selfplay-joint`. Choose any new folder
under `results/`; existing output folders are never overwritten. Omit `--output`
to generate a folder named with the run, committed generation, and UTC timestamp:

```sh
python scripts/export-training-reports.py
```

The script copies:

- `state.json`, `retention-history.json`, and available run metadata such as `best.json`.
- The latest **two committed iteration** reports and associated JSON reports/receipts.
- All available reports and specifications under the run's `evaluations/`, including champion, monitor, and broader panels.
- The originating IL run's state/report and available ancestor IL states/reports.

It also writes `recorded-config.json` from the saved run state and an
`export-manifest.json` containing source paths, SHA-256 hashes, file sizes, missing
files, and report/receipt mismatches. Missing or pruned reports are listed rather
than reconstructed. Iteration/evaluation folder names are preserved, for example:

```text
results/joint7-review-0060/
  state.json
  retention-history.json
  recorded-config.json
  export-manifest.json
  iteration-0058/report.json
  iteration-0059/report.json
  evaluations/champion-00019200/report.json
  evaluations/champion-00019200/spec.json
  evaluations/monitor-0003/report.json
  lineage/artifacts/step5/v4-batch7-joint/report.json
  ...
```

Upload/copy the **whole output folder**, including its subfolders. It is an
analysis bundle, not a resume backup: checkpoints, replay shards, feature caches,
and profiler traces are excluded. Original files are copied byte-for-byte and
never edited. The script is outside source-bound training packages, so adding or
running it does not change a run's learning implementation identity.

Use `--run` for another self-play run, `--latest` for more full iteration reports,
and repeat `--extra` to include independent comparisons outside the run directory:

```sh
python scripts/export-training-reports.py \
  --run artifacts/step5/v4-selfplay-joint \
  --output results/joint7-review-next \
  --latest 4 \
  --extra artifacts/joint7-review/evaluation-g0060-fixed16
```

Use `--root /data/not_backed_up/yxu209/ai-training/` when invoking the script
against an explicit project root. Export after an iteration finishes for its
final timings. Read-only exports during training are allowed; changes to selected
files during copying abort publication and request a retry. A run-lock marker and
missing final iteration timing are recorded in the manifest. No training lock is
created or removed by this script.

For the generation-150 comparison, include the independent panels with
`--extra artifacts/joint7-g150-panel` and the paired comparison with
`--extra artifacts/joint7-g150-comparison` in the project where each resides.
`comparison.json`, evaluation specifications/results, and new teaching/calibration
fields in full iteration reports are already included. See the
[generation-150 review](JOINT7_GENERATION150_REVIEW.md) for exact continuation and
comparison commands. Exports contain reports; complete game replays and checkpoint
tensors remain separate when an investigation needs them.

The [teacher phase support update](TEACHER_PHASE_SUPPORT.md) adds per-phase work,
coverage and timing, distinct used/unused labels, and distillation policy drift.
These fields are included automatically in iteration reports and compact history;
`null` cumulative allowance means the wall-fraction restriction is disabled.

Verification: `python -m unittest discover -s tests/Reports -v` covers report
selection, original bytes, lineage, missing evidence, output protection, concurrent
changes, and a smoke export using the supplied generation-60 JSON evidence.
