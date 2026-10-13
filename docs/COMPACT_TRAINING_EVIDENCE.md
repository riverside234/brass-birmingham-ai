# Compact training evidence

The storage revision `training-evidence-compaction/1.0.0` preserves the complete
historical approval registry and generation-150 audit. It changes storage and
readers only; training presets and current search time settings are unchanged.

## Resume approvals

[`selfplay-resume-sources.json`](../configs/learning/selfplay-resume-sources.json)
uses `selfplay-resume-sources/2.0.0`. Repeated reasons, configuration changes,
migrations and learning compatibility hashes are stored once in `definitions`.
Each transition explicitly references those values with `*_ref` fields.
Transition order, source hashes and all other metadata remain intact. No
approval is inferred transitively and no historical approval is deleted.

Use `expand_source_reviews(receipt)` from
[`source_reviews.py`](../src/brass_ai/recovery/source_reviews.py) before inspecting
or editing individual approvals. It accepts v1 and v2 and returns independent
v1 records. Use `compact_source_reviews(receipt)` before writing an edited
registry. Missing, unknown or ambiguous references are rejected. To convert a
legacy file or normalize an existing v2 file:

```sh
python scripts/compact_selfplay_reviews.py
```

The reader change has a new source identity. Exact source-only continuation
from the preceding implementation is recorded, together with the existing
incoming routes and their original configuration changes/migrations. Completed
IL handoff from that implementation remains explicitly approved. The normal
`--resume-reviewed-source`, checkpoint, lineage and configuration checks still
apply. Existing runs and checkpoint files are not rewritten by compaction.

## Generation-150 audit

[`metrics.json`](joint7-150-evidence/metrics.json) uses
`joint7-generation150-audit/2.0.0`. Aggregates, provenance, missingness and input
hashes remain readable in that file. Its `iterations_ref` names an adjacent
`iterations-<sha256>.json.gz` file containing the full JSON array of all 150
iteration records, plus its compressed-byte SHA-256 and record count.

The filename is content-addressed. The generator publishes the sidecar before
atomically replacing the summary, so interruption preserves the previous
readable report. Gzip timestamps and platform header differences are removed;
unchanged output is not rewritten. Keep the referenced sidecar with the summary
when copying evidence. A changed report can produce another sidecar; any old
one can be removed once no retained summary references it.

Both writers use a separate temporary file for each write and clean it up on
failure. Overlapping exports or compactions cannot overwrite each other's
temporary files; the last completed summary publication selects its matching
sidecar. Invalid report types are rejected before any output is written.

Regenerate both files with `python scripts/review_joint7_150.py`. To load the
complete original v1 data structure, including checksum/count validation:

```python
import sys
sys.path.insert(0, 'scripts')
from review_joint7_150 import read_metrics

metrics = read_metrics('docs/joint7-150-evidence/metrics.json')
assert len(metrics['iterations']) == 150
```

`read_metrics` also accepts legacy v1 reports. Regression tests pin the semantic
hash of all original audit data and the first 934 historical approvals, and
exercise existing checkpoint/Adam/RNG continuation across this storage revision.
