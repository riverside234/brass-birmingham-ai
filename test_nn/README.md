# Migration test checkpoints

This folder transfers selected committed lab checkpoints to the local migration
tests through ordinary Git. Exports perform read-only utility validation
by default, using the lab's existing PyTorch environment and prepared rules host.
The active training run and original checkpoints remain unchanged. Restoration,
verification and `--no-utility-validation` byte-copy exports use only the standard library.

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

The script first preserves a byte-for-byte source bundle, verifies its SHA-256
against the training receipt, and then exports a copy with a validation receipt.
Checkpoint parts are at most 48 MiB. Source metadata includes:

- `state.json`: exact captured source state;
- `config.json`: effective config from that state;
- `source.json`: exporting checkout commit and tracked-change status;
- `manifest.json`: original checkpoint entry, metadata/part sizes and hashes.

The source run's config may be newer than an older best/anchor checkpoint. The
checkpoint's embedded `model_config`, reward and calibration remain authoritative
for that model. This is an inference-test bundle, not a full training-resume
archive or an ONNX export. The validated copy preserves every weight, model ID,
optimizer/RNG value and other training metadata. Only `training.utility_calibration`
is replaced, so the file hash changes. Its manifest retains the exact original
checkpoint entry and source-bundle link.

## Read-only utility validation

The default export now does the following before updating `latest.json` or `best.json`:

1. Load a private copy of the selected checkpoint with the strict loader.
2. Generate **4 independent complete validation games**, using the selected
   frozen policy, a separate host process and distinct seeds. Existing training
   replay is not used. Policy collection uses T1,
   epsilon .05, neutral context and unknown critic descriptors. Targets use the
   selected checkpoint's own reward/remaining-cost contract, including for an older best.
3. Select at most 128 rows per game across seats/eras/phases/forced states to bound
   memory. Compute labels from complete outcomes before selecting those rows.
4. Evaluate the unchanged model in inference mode. Report **MSE, RMSE, MAE,
   signed bias, prediction/target means and sample counts**, overall, separately
   for **CanalEra and RailEra**, and for each game. Bias means prediction minus
   realized utility: positive values overestimate; negative values underestimate.
   Metrics weight sampled decisions equally and include all four seat predictions.
   Era metrics compare predictions made in that era to the same existing utility
   return contract; they are not separate era scores.
5. Require completed games, finite predictions/targets/metrics, both eras in every
   game's sample and exact unchanged weights/config. **There is no MSE or bias
   performance threshold, optimizer, fitting, backward pass or parameter update.**
6. On completion, save/reopen a copy with
   `training.utility_calibration.ready is True` with the matching reward ID.

`utility_calibration` remains a structured receipt, not a bare boolean. The
validated bundle also includes `calibration.json`, `model-config.json`, and the
fresh game shards under `validation-games/`, all with hashes. The receipt records
`mode: read-only-validation`, `steps: 0`, the source model ID and metrics.
**`ready: true` means the requested read-only validation completed with valid data;
it does not certify a numerical accuracy or playing-strength threshold.** The
selection pointer identifies the same model weights with new validation evidence.

Validation defaults to **CPU, one Torch thread**, so it does not occupy a GPU
used by training. Run in the existing lab training environment; the configured
rules host must already be prepared. Options include:

```bash
python scripts/export-test-nn.py --select best --validation-games 4
python scripts/export-test-nn.py --select best --validation-device cuda:0
```

`--validation-host` selects an already prepared host; `--validation-seed`,
`--validation-rows-per-game` and `--validation-threads` set the bounded diagnostic
recipe. Matching verified validation bundles are reused; changing inputs/code/settings
invalidates reuse. With `--select both`, identical source selections share one evaluation.
The corresponding `--calibration-*` spellings remain aliases except for
`--calibration-heldout-games`, which was removed along with the fitting/heldout split.

If a game is incomplete, an era is missing, a value is non-finite or validation
fails to run, no new readiness claim is published and the requested selection
pointers remain unchanged. Finite high MSE or bias is recorded without rejecting
the export. The active run and original checkpoints are never modified.

For a raw copy **without** a readiness claim, explicitly select:

```bash
python scripts/export-test-nn.py --select best --no-utility-validation
```

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

The command verifies every part and metadata receipt, reconstructs the exported
`checkpoint.pt`, and prints its absolute path. Use that path for checkpoint
inspection, exact-feature/NN checks and subsequent ONNX export. `latest.json`
and `best.json` identify their immutable bundle directories. If both select the
same checkpoint/state, they share one bundle.

Verify an export without reconstructing model files:

```bash
python scripts/export-test-nn.py --verify --select latest
```

Restoration never reruns validation or needs the lab host/PyTorch; the saved receipt is
already inside the exported `.pt`. Previously exported raw bundles remain readable.

Checkpoint parts, fresh validation game shards and metadata are tracked; reconstructed `checkpoint.pt` and
temporary staging files are ignored. This avoids requiring Git LFS and preserves
binary/JSON receipt bytes across Linux and Windows. An existing restored file
with the wrong hash is rejected rather than overwritten.

Repeated exports reuse an identical bundle, update only the requested selection
pointers, and retain older bundles. No historical exports or training files are
deleted. No production checkpoint is included until an export is run on the lab.
