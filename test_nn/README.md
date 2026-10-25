# Migration test checkpoints

This folder transfers selected committed lab checkpoints to the local migration
tests through ordinary Git. The exporter uses only Python's standard library;
it never imports the trainer, changes a training run, or starts training.

## On the lab

From the repository root, export the latest committed model:

```bash
python scripts/export-test-nn.py
```

The default run is `artifacts/step5/joint7-v11-2-grouped-lr-critic`. If it is
absent, the script accepts a single discoverable run under `artifacts/step5/`;
multiple candidates require `--run`.

To choose another run or selection:

```bash
python scripts/export-test-nn.py --run artifacts/step5/joint7-v11-2-grouped-lr-critic --select latest
python scripts/export-test-nn.py --run artifacts/step5/joint7-v11-2-grouped-lr-critic --select best
python scripts/export-test-nn.py --run artifacts/step5/joint7-v11-2-grouped-lr-critic --select both
```

`latest` means the checkpoint referenced by the captured committed state, never
the newest filename or a `.tmp` file. `best` uses the recorded `best` entry,
otherwise the accepted `champion`, otherwise the qualified policy anchor. The
manifest records this choice; an anchor is not a claim of globally best strength.
If no such selection exists, `best` fails rather than substituting latest.

The script copies the checkpoint byte-for-byte into parts of at most 48 MiB,
verifies their combined SHA-256 against the training receipt, and snapshots:

- `state.json`: exact captured source state;
- `config.json`: effective config from that state;
- `source.json`: exporting checkout commit and tracked-change status;
- `manifest.json`: original checkpoint entry, metadata/part sizes and hashes.

The source run's config may be newer than an older best/anchor checkpoint. The
checkpoint's embedded `model_config`, reward and calibration remain authoritative
for that model. This is an inference-test bundle, not a full training-resume
archive or an ONNX export.

Training can continue during export. An atomic committed-state snapshot and the
recorded hash define the model being copied. If retention removes it before it
can be opened, retry the export; the script does not acquire the trainer's lock.

Inspect and publish the prepared folder on the lab:

```bash
git add test_nn
git commit -m "Export NN checkpoints for migration tests"
git push
```

The script does not stage, commit or push automatically. Pull the script and its
Git ignore/attribute rules onto the lab before exporting.

## After pulling locally

```bash
git pull
python scripts/export-test-nn.py --restore --select latest
```

For an export containing both selections:

```bash
python scripts/export-test-nn.py --restore --select both
```

The command verifies every part and metadata receipt, reconstructs the original
`checkpoint.pt`, and prints its absolute path. Use that path for checkpoint
inspection, exact-feature/NN checks and subsequent ONNX export. `latest.json`
and `best.json` identify their immutable bundle directories. If both select the
same checkpoint/state, they share one bundle.

Verify an export without reconstructing model files:

```bash
python scripts/export-test-nn.py --verify --select latest
```

Checkpoint parts and metadata are tracked; reconstructed `checkpoint.pt` and
temporary staging files are ignored. This avoids requiring Git LFS and preserves
binary/JSON receipt bytes across Linux and Windows. An existing restored file
with the wrong hash is rejected rather than overwritten.

Repeated exports reuse an identical bundle, update only the requested selection
pointers, and retain older bundles. No historical exports or training files are
deleted. No production checkpoint is included until an export is run on the lab.
