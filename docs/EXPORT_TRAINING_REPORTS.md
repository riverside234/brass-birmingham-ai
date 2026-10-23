# Export reports for analysis

Run from the lab project. This uses Python's standard library; it does not load
models or launch training.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python scripts/export-training-reports.py
```

The default source is **`artifacts/step5/joint7-v11-1-grouped-lr`**, matching the
[current fresh-start and continuation commands](MLP_V11_LAB.md#fresh-self-play-from-completed-v111-il).
The destination is automatically named with the run, committed generation, and
UTC timestamp, for example `results/joint7-v11-1-grouped-lr-g0021-<timestamp>/`.
Choose another new folder under `results/` with `--output`; existing output
folders are never overwritten:

```sh
python scripts/export-training-reports.py --output results/joint7-v11-1-grouped-lr-review
```

For the current v11.2 run, export with incremental reuse:

```sh
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --continue
```

`--continue` still creates a **new** generation/timestamp folder. It finds prior
exports of the same `--run` anywhere under `results/`, including version-1.0/1.1
manifests, and skips reports whose exported bytes are already present.
It checks the actual old file, not just its manifest hash. New or changed reports
are copied normally; a missing or altered old copy is copied again. Exports from
another run or a later generation are not used. With no prior export, this flag
performs a normal first export.

The version-1.2 manifest lists newly copied reports in `files`, and skipped
reports in `reused_files`, with `previous_export`, `previous_path`, and SHA-256.
References point directly to a physical file in an older bundle. Keep and share
the referenced older bundles along with the new folder; an incremental folder
alone is not a complete analysis bundle. `recorded-config.json` and the new
manifest are always generated. Omit `--continue` for an independent export of
the selected reports. Existing folders are never modified or deleted.

The exported root `retention-history.json` is now compact JSON. Only whitespace
outside strings and an optional UTF-8 BOM are removed; a final newline is retained.
All report fields, iterations, strings, escapes and exact numeric tokens stay intact.
The original file in the training run is never edited. Other report files still
preserve their original bytes, including evidence referenced by state receipts.

For compact history, the manifest records `transform="json-whitespace-only/1.0.0"`:

- `bytes` and `sha256` describe the **exported compact file**.
- `source_bytes` and `source_sha256` describe the **original run file**.

Publication rechecks the original source hash, so even a formatting-only source
change during export aborts the operation. Incremental reuse checks the compact
file's actual bytes and rechecks dependencies before publication. Source formatting
changes can reuse the same compact bytes while recording the new source hash.
Old pretty-printed history is copied once into compact form; other unchanged files
remain reusable from older exports. Existing export folders are never rewritten.

Validation on the g620 report in an isolated fixture reduced history from
54,854,166 to 34,576,705 bytes (52.31 to 32.97 MiB, 37% smaller), preserving all
128 entries and source bytes. A second incremental export reused the compact
history with matching hashes. All 37 exporter regressions pass, including exact
numeric/string tokens, legacy exports, source races and dependency tampering.

The script copies:

- `state.json`, `retention-history.json`, and available run metadata such as `best.json`.
- The latest **two committed iteration** reports and associated JSON reports/receipts.
- Any additional committed iteration report not yet present in compact history,
  so interrupted cleanup does not leave gaps outside the latest-two selection.
  These reports are listed in the export manifest's `uncompacted_iterations`.
- All available reports and specifications under the run's `evaluations/`, including champion, monitor, and broader panels.
- The originating IL run's state/report and available ancestor IL states/reports.
- The fixed IL-KL probe's `il-kl/manifest.json` and `il-kl/initial-report.json`, when present.
- The **latest committed** PL rating slot folder: its specifications/reports,
  separate panels, compact game outcomes (`games.jsonl`), and progress curve
  (`progress.svg`). Older slots needed by panel-retirement or competitive-coverage
  gate receipts are also included until archived. Other older and pending slot folders are excluded, with or without
  `--continue`. The selection is recorded as `rating_slot_generation`; unchanged
  latest-slot files can be reused with `--continue` when no new rating has run.
- The committed cumulative rating archive, when present, plus separate temperature
  comparison reports and their compact evidence.
- Qualified policy-reference transition reports, its active probability-cache
  manifest, and the fit/spec supporting its paired promotion interval. These
  remain available even when the active reference outlives ordinary slot retention.

For the current v11.1 recipe, full iteration reports and compact history retain
the PPO LR controller, exact categorical KL, overall and per-group IL-anchor
distance/counts/status, warmup, optimizer steps, actual sample usage, fixed
50/45/5 table counts and learner-seat attribution. Disabled teacher search and
an absent champion gate need no substitute reports. The IL source is read from
the saved run lineage; its existing directory/file names are preserved. No IL
retraining or renaming is needed.

The calibrated-limit revision also retains `il_kl_limit` (economic health baseline,
overall/group limits, recovery latch and hold/raise/tighten
decision), `rehearsal_schedule`, and zero-VP/insolvency counts by table type.
The probe manifest carries selected-position identities and source/cache hashes;
its dense `probe.npz` and `.prepared/` validation cache are excluded from this
report-only bundle.

For qualified references, full/compact reports retain `policy_anchor`,
`anchor_transition` and `rating_coverage`, along with KL to the active reference,
original IL KL, promotion confidence interval and whether anchor/economic checks
permit LR increases. Original IL and fixed probe identities are unchanged. The
active checkpoint, dense reference-probability array and bootstrap samples stay
in the training run and are not copied; the helper verifies them on the lab host.

The PFSP revision retains per-member S/C and raw/decayed outranks, frozen IL share
and planning probabilities, seat/comparison counts, concentration summaries,
review-only regression alarms and admission/eviction decisions. The existing
`quality_pool` report/state key is retained; its contract identifies PFSP versus
archived quality scores. Compact history and exports preserve the entire block.
The staged 17/24-member total pool and independently retained rated agents are visible in
state. The latest slot's compact held-out JSONL contains outcomes, not PPO
trajectories. Reproducing a full historical refit can require earlier exports or
the original run's other slot outcomes as well as its archive. Committed outcome
and panel hashes are verified before export. Dense opponent-probe caches and
`bootstrap.npz` are excluded. Rating reports retain uncertainty, paired differences,
v4/regression warnings and the pending manual deployment-review status.

It also writes `recorded-config.json` from the saved run state and an
`export-manifest.json` containing source paths, SHA-256 hashes, file sizes, missing
files, and report/receipt mismatches. Missing or pruned reports are listed rather
than reconstructed. Iteration/evaluation folder names are preserved, for example:

```text
results/joint7-v11-1-grouped-lr-g0021-<timestamp>/
  state.json
  retention-history.json
  recorded-config.json
  export-manifest.json
  iteration-0019/report.json
  iteration-0020/report.json
  evaluations/monitor-<games>/report.json
  evaluations/monitor-<games>/spec.json
  ratings/slots/generation-000020/games.jsonl
  ratings/slots/generation-000020/panels.json
  ratings/slots/generation-000020/report.json
  ratings/slots/generation-000020/spec.json
  ratings/slots/generation-000020/progress.svg
  lineage/artifacts/step5/features4-v11-1-batch7-il/state.json
  lineage/artifacts/step5/features4-v11-1-batch7-il/report.json
  ...
```

Upload/copy the **whole output folder**, including its subfolders, and any older
bundles referenced by `reused_files` when using `--continue`. It is an
analysis bundle, not a resume backup: checkpoints, replay shards, feature caches,
and profiler traces are excluded. Run files are never edited; exported retention
history alone is whitespace-minified as described above. The script is outside
source-bound training packages, so adding or
running it does not change a run's learning implementation identity.

Use `--run` for another self-play run, `--latest` for more full iteration reports,
and repeat `--extra` to include independent comparisons outside the run directory:

```sh
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-1-grouped-lr \
  --output results/joint7-v11-1-grouped-lr-review-next \
  --latest 4 \
  --extra artifacts/joint7-v11-1-grouped-lr-comparison
```

Use `--root /data/not_backed_up/yxu209/ai-training/` when invoking the script
against an explicit project root. Export after an iteration finishes for its
final timings. Read-only exports during training are allowed; changes to selected
files, including reused files in older bundles, during copying abort publication
and request a retry. A run-lock marker and
missing final iteration timing are recorded in the manifest. No training lock is
created or removed by this script. Publication retries transient Windows handle
errors for at most 0.75 seconds, checking that the destination is still unused
before each attempt.

Before any reports have been archived, `retention-history.json` may not exist;
the manifest lists it as missing and still exports the committed reports. No
evaluation files are expected until an evaluation has actually run. For an older
run, select it explicitly, for example
`python scripts/export-training-reports.py --run artifacts/step5/v4-selfplay-joint`.

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

Rating retention keeps full slot folders for the most recent 100 training
iterations, normally ten slots. Older compact outcomes and diagnostic panels
are consolidated into the checksummed `ratings/history-<digest>.json` named by
`state.json` → `ratings.archive`. Export verifies and copies that archive along
with **only the latest committed slot**, rather than all ten recent folders.
The latest rating report/curve still includes the cumulative fit. Historical games
in the original run and earlier exports still support cumulative PL
fits and paired bootstrap refits; older per-slot bootstrap tensors and full
reports/curves can be cleaned. Existing bundles already under `results/` are
not automatically removed.

Incremental export regressions cover legacy-bundle reuse, first export, changed
or corrupted reports, same-run/generation isolation, flat references, concurrent
changes to reused evidence, CLI invocation, and latest committed rating selection
without copying older or pending slots. They run without models or training.

October 20 validation: **39 report tests passed**. An isolated report-only fixture
reconstructed the original g185 source bytes from the saved bundle (normalizing
Git checkout line endings only inside the disposable fixture). A regular export
selected 20 source files and only the g180 rating slot; an unchanged incremental
export copied zero source files and reused all 20. After simulating generation
186 in that fixture, the incremental export copied two changed/new files and
reused 19 files (45,359,904 bytes), with no missing reports or warnings. Generated
configuration/manifest files are additional. No real training state or model
checkpoints were modified.

Verification: `python -m unittest discover -s tests/Reports -v` covers report
selection, original bytes, lineage, missing evidence, output protection, concurrent
changes, the new default run and grouped LR/anchor metrics, and a smoke export
using the supplied generation-60 JSON evidence.

October 16 validation: **26 report tests passed**, and a read-only export of the
existing v11.1 three-game smoke copied four source reports with matching bytes
and checksums, preserving grouped anchor/LR fields and the original IL path.
Its first-generation archive was absent as expected. Evidence is in
`artifacts/report-export-grouped-lr-20261016/`.

October 18 quality-pool/rating validation: the expanded focused suite passes,
including immutable rating outcome/panel/spec/report checks, original JSONL/SVG
bytes and dense-cache exclusion. Exporting the final three-game native smoke
copied six files with matching hashes and no warnings; the first-generation
compact archive was correctly reported missing. Evidence is in
`artifacts/quality-rating-20261018/validation.json` and `report-bundle/` there.

October 17 validation: the 67-test focused controller/report suite passed. An
actual export of the final three-game calibrated-limit smoke copied six source
JSON files with matching bytes/checksums, including `il-kl/manifest.json` and
`il-kl/initial-report.json`. Dense probe tensors and `.prepared/` data were
excluded. The missing first-generation archive was recorded; no warnings were
reported. Evidence: `artifacts/auto-il-kl-20261017/validation.json` and
`artifacts/auto-il-kl-20261017/report-bundle/`.
