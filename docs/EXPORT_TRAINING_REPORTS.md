# Export reports for analysis

Run from the lab project. This uses Python's standard library; it does not load
models or launch training.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python scripts/export-training-reports.py
```

The default source is **`artifacts/step5/joint7-v11-2-g1290-trained-restart`**, matching the
[current g1290 restart command](../ai-research/NEUTRAL_SELFPLAY_G1290_RESTART.md#start-on-the-lab-machine).
The destination is automatically named with the run, committed generation, and
UTC timestamp, for example `results/joint7-v11-2-g1290-trained-restart-g0021-<timestamp>/`.
Choose another new folder under `results/` with `--output`; existing output
folders are never overwritten:

```sh
python scripts/export-training-reports.py --output results/joint7-v11-2-g1290-trained-restart-review
```

For the current g1290 restart, export with incremental reuse:

```sh
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --continue
```

### Performance reports and VP-history repair (2026-11-09)

The same command now copies **small performance and profiling documents by file
name**, so a newly added report needs no script change. A file is selected when
its name contains one of `profile`, `profiling`, `profiler`, `perf`,
`performance`, `timing`, `timings`, `startup`, `throughput`, `telemetry`,
`benchmark`, `latency` or `bottleneck` as a whole word (`timing-report.json`,
`lane-startup.json`, `perf_summary.json`, `PROFILE.md`), and it is a `.json`,
`.jsonl`, `.md`, `.txt` or `.csv` document. They are looked for in the latest
`--latest` iterations (including `teacher-shadow/` and `profile-*` folders),
`evaluations/`, the exported rating slots and run-level `profile*` folders.

- Raw profiler traces (`*.trace.json`, TensorBoard event files), generic names such as
  `report.json` inside a profiler folder, logs and anything over **16 MiB** are not copied.
  An oversized file is listed in the manifest's `missing` entries with its size limit.
- A partly written or unreadable performance file adds a warning and is skipped; it
  does not stop the export.
- The per-lane startup breakdown and Mono bootstrap timings are fields inside
  the iteration reports and `teacher-shadow/collection.json`, which were already exported.

**VP history.** A compact restore (here at g1284) can leave the run's
`ratings/vp-history.json` holding only the generations since the restore, which
drew a chart with no complete groups. The exporter now compares that cache with
earlier exports of the same run under `results/` that share its initial
checkpoint. Generations that those exports still hold are added, and the
exported `ratings/vp-history.json` and the latest slot's `progress_vp.svg` are
rebuilt from the combined history. The run's own rows are authoritative: an
earlier export whose rows disagree is skipped whole with a warning, and a run
whose history is complete keeps its exact original bytes. Only each earlier
export's small cache, iteration reports and state are read, not its retention
history. Drawing the chart needs matplotlib; without it the run's own files are
exported with a warning.

The manifest marks both rebuilt files with `transform` =
`selfplay-vp-history-merged/1.0.0` plus the run file's `source_bytes` and
`source_sha256`, and `vp_history_merge` lists the added generation ranges and
the earlier exports used. The manifest format is now `selfplay-report-export/1.4.0`;
1.0–1.3 exports remain valid `--continue` sources. The run itself is not edited, so
its own cache stays short; every export repairs its copy. To repair the run too, stop
training at a committed boundary and import the repaired history:

```bash
python scripts/backfill-vp-history.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --cache results/YOUR-NEW-BUNDLE/ratings/vp-history.json
```

Generations that no earlier export holds (for this run g1–70 and g371–440) stay
missing; the chart shows them as gaps. Against the real g1290 evidence, the
combined history has 1,150 generations (g71–g1290) and 115 complete groups,
where the original export had 7 generations and none.

### Rust teacher diagnostics (2026-11-08)

The same command now also selects the available `teacher-shadow/collection.json`
and `teacher-shadow/calibration.json` for the latest **two committed iterations**.
`--latest N` controls that detail window; `--continue` keeps the limit and reuses
verified earlier exports. It also fills these newly selected files when an older
export of the same iteration omitted them. Historical trends continue to use
`retention-history.json.gz`, with the existing interrupted-cleanup gap handling.

`collection.json` contains teacher collection/inference timing and table metrics.
`calibration.json` contains the search throughput/quota estimate and warm-up
samples. It is included by explicit user choice; beta gradient calibration is
separate. The saved beta/observations remain in
`state.json` → `teacher_phase.weight.gradient_qualification`; iteration reports
retain `teacher_phase.term` measurements and the `search_value` phase/update
metrics. These documents preserve their original bytes and source hashes.

External `artifacts/rust-backend/*/training.log` and launcher failure/qualification
records are **not automatically selected or uploaded**. Bulk teacher trajectories,
value rows, replay shards, checkpoints and feature arrays remain excluded.
Pruned or never-recorded details cannot be reconstructed. Local validation uses
disposable report files and `--no-push`, with no training or benchmark sessions.
The affected exporter suite completed 47 tests: 46 passed, and the optional
archived-generation-60 fixture was skipped because its source files are absent.

The CLI fetches and publishes through the independent `results/` repository.
Its default destination is `https://github.com/riverside234/ai-training-results.git`,
configured in `scripts/results_git.py`. Every fetch/push uses that URL directly;
an older or absent `results/origin` does not select the destination or block
export. Each operation pins both URLs in command-scoped Git configuration;
`pushInsteadOf` cannot redirect the upload. Existing origin URLs and upstream
settings are preserved. The private
project-root `.env.results-git` file (`RESULTS_GIT_URL`) supplies authentication;
it does not replace the scripted destination. Validation accepts equivalent
HTTPS/SSH addresses and authenticated URL rewrites for the selected repository,
and rejects rewrites to another host/repository without printing credentials.
Selected-target history must still be compatible; divergent history is never
force-pushed. For an intentional alternate results repository, supply
`--results-remote URL`. Use `--no-push` for a local export with no Git operations.

`--continue` still creates a **new** generation/timestamp folder. It finds prior
exports of the same `--run` anywhere under `results/`, including version-1.0/1.1
manifests, and skips reports whose exported bytes are already present.
It checks the actual old file, not just its manifest hash. New or changed reports
are copied normally; a missing or altered old copy is copied again. Exports from
another run or a later generation are not used. With no prior export, this flag
performs a normal first export.

The new run uses its own committed generation, starting at zero. Its seed receipt
is exported as `fresh-start.json`; startup overrides and auxiliary initialization
receipts remain in `state.json`, and fitting/warmup metrics remain in iteration
reports and compact history. Manual IL/v4 retirement and competitive-PL overrides
need no measured panel files; they are preserved as manual permissions. Older
`joint7-v11-2-grouped-lr-critic` exports are not reused for this separate run.

### Reports after compact restore or the old-machine reward revision

The same `--run ... --continue` command works after restoring g1284 and rebuilding
buffers. A compact archive may omit the original generation-zero-only
`seed/manifest.json`; that absence must not prevent exporting a completed run.
The exporter first uses the working seed file when present, then tries the
checkpoint-bound `.training-rollback/` backup and physical earlier same-run
`fresh-start.json` reports. Recovery requires the **exact full SHA-256** saved in
`state.signature.config.fresh_selfplay_seed`; JSON is never synthesized or
re-serialized. A recovered entry records `recovered_from` and `recorded_source`.
If no exact copy exists, `export-manifest.json` lists the missing seed provenance
and a warning while exporting the available analysis. A present but changed
working seed file remains an integrity error. Matching rollback journals/states
and selected source documents are checked again before publication.

The new checkpoint-bound reward/buffer instruction and original-state receipts
under `machine-transfer/` are exported as verified JSON. Rating slot folders
with the supported `-source-<16 hex>` suffix are accepted after source/reward
changes, with their existing state-pinned file hashes still required. Original
IL/critic-upgrade provenance is selected from the signature even when the run's
origin is a weights-seeded restart. Full iteration/state reports retain revised
rewards, buffer-rebuild/stage progress, Stage-C held-out loss/interval retention,
joint KL and fixed-beta fields without dropping or rewriting them.

This is report-only recovery: it does not restore/delete buffers, restart
training, alter optimizers/RNG or copy seed weights, checkpoints, full IL data
or feature arrays. Future compact transfers retain an available seed JSON
manifest, without adding unreferenced seed tensors. No re-export/retraining is
needed for an existing run. Focused regression additions are authored for the
server; local execution was not requested. Server debug checks without Git,
training or benchmarks: `python -m unittest discover -s tests/Reports -p test_export_training_reports.py`.

The version-1.3 manifest lists newly copied reports in `files`, and skipped
reports in `reused_files`, with `previous_export`, `previous_path`, and SHA-256.
References point directly to a physical file in an older bundle. Keep and share
the referenced older bundles along with the new folder; an incremental folder
alone is not a complete analysis bundle. `recorded-config.json` and the new
manifest are always generated. Omit `--continue` for an independent export of
the selected reports. Existing folders are never modified or deleted.

The exported root history is now **`retention-history.json.gz`**, using deterministic
gzip compression (level 9) after minification. Only whitespace outside strings
and an optional UTF-8 BOM are removed; a final newline is retained.
All report fields, iterations, strings, escapes and exact numeric tokens stay intact.
The original file in the training run is never edited. Other report files still
preserve their original bytes, including evidence referenced by state receipts.

For compressed history, the manifest records `transform="gzip-json-whitespace-only/1.0.0"`:

- `bytes` and `sha256` describe the **exported gzip file**.
- `source_bytes` and `source_sha256` describe the **original run file**.
- `uncompressed_bytes` and `uncompressed_sha256` describe the minified JSON inside it.

Publication rechecks the original source hash, so even a formatting-only source
change during export aborts the operation. Incremental reuse checks the compressed
file's actual bytes and rechecks dependencies before publication. Source formatting
changes can reuse the same compact bytes while recording the new source hash.
Old plain history is copied once into compressed form; other unchanged files
remain reusable from older exports. Existing export folders are never rewritten.

The exporter's `load_json(path)` function reads both `.json` and `.json.gz`.
Passing a missing `retention-history.json` automatically checks the adjacent
`.json.gz` file. Export also accepts an already-compressed source history, though
normal training continues using its original plain JSON file. Other report files
retain their original bytes. For standalone analysis, use Python's standard library:

```python
import gzip
import json

with gzip.open("results/YOUR-BUNDLE/retention-history.json.gz", "rt", encoding="utf-8") as stream:
    history = json.load(stream)
```

With `--continue`, use the physical `previous_export` / `previous_path` location
from the manifest if history is listed under `reused_files`. A new export does
not replace an earlier bundle's large plain file; avoid adding that obsolete
plain copy to the same Git commit if the goal is to eliminate its size warning.

Validation: 39 exporter tests pass, covering gzip/plain loading, deterministic
compression, exact JSON tokens, legacy manifests, incremental reuse, corruption,
source races and dependency tampering. On the existing g620 history, minification
plus gzip reduced 54,854,166 bytes to 10,723,545 bytes (80.45% smaller), with
decompression matching the minified bytes exactly. The source was unchanged.

The script copies:

- `state.json`, `retention-history.json.gz`, and available run metadata such as `best.json`.
- The latest **two committed iteration** reports and associated JSON reports/receipts.
- Any additional committed iteration report not yet present in compact history,
  so interrupted cleanup does not leave gaps outside the latest-two selection.
  These reports are listed in the export manifest's `uncompacted_iterations`.
- All available reports and specifications under the run's `evaluations/`, including champion, monitor, and broader panels.
- The originating IL run's state/report and available ancestor IL states/reports.
- The fixed IL-KL probe's `il-kl/manifest.json` and `il-kl/initial-report.json`, when present.
- The **latest committed** PL rating slot folder: its specifications/reports,
  separate panels, compact game outcomes (`games.jsonl`), and progress curves
  (`progress.svg` for PL strength and `progress_vp.svg` for self-play learner VP).
  The small cumulative `ratings/vp-history.json` is also included when present.
  Older slots needed by panel-retirement or competitive-coverage
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
  retention-history.json.gz
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
  ratings/slots/generation-000020/progress_vp.svg
  ratings/vp-history.json
  lineage/artifacts/step5/features4-v11-1-batch7-il/state.json
  lineage/artifacts/step5/features4-v11-1-batch7-il/report.json
  ...
```

Both charts are generated at the existing main-rating cadence. The VP chart
uses ordinary **self-play tables' learner seats only**, including zero-VP seats;
history/v4 tables, branches and evaluation games do not contribute. Each point
is pooled `vp_sum / seat_games` across a fixed group of ten committed iterations:
g1–10, g11–20, and so on. An `iteration-0009/report.json` belongs to g10.
The two panels show the full available history and the latest 200 generations,
in the same visual style as the PL chart. Missing groups remain gaps; unfinished
groups are omitted. These training scores are not controlled strength ratings.

Automatic generation retains the small per-iteration totals in
`ratings/vp-history.json`, so later report cleanup does not shorten the curve.
Already-pruned evidence cannot be invented. To redraw both charts from a run or
a complete export bundle without training, inference, or a PL refit:

```bash
python scripts/render-pl-progress.py \
  --report results/YOUR-BUNDLE/ratings/slots/generation-000370/report.json
```

This writes sibling `progress.svg` and `progress_vp.svg`. It leaves the report,
training state and source history unchanged. `--run PATH` explicitly selects
the run/bundle when it is not an ancestor of the report. Optional repeated
`--history-source results/OLDER-BUNDLE` arguments fill gaps using complete older
bundles with the same initial checkpoint lineage; conflicting totals are rejected.
A PL-only report with no training history produces an explicitly empty VP chart.
The normal `python scripts/export-training-reports.py` command copies the new
chart and history automatically, including verified reuse with `--continue`.

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
