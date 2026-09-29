# Curated model artifacts

`bootstrap/` contains three loadable, approximately 1.9M-parameter R2 checkpoints:

- `r2-fresh-medium.pt`: fresh weights; use this for a new substantial self-play run.
- `r2-value-scaffold.pt`: the existing R2 value-learning smoke checkpoint.
- `r2-onpolicy-scaffold.pt`: the existing R2 single-update policy-learning smoke checkpoint.

The latter two are retained for reproducibility, not recommended as strong
opponents or expert teachers. Their exact weight identities and file hashes are
in [`../runtime/distribution.json`](../runtime/distribution.json). Loading uses
R2's four-seat compatibility checks and PyTorch `weights_only=True`.

`releases/` holds deliberately selected weights and accompanying manifests.
Any Step 5 CPU smoke checkpoint here is a workflow artifact with **unmeasured
playing strength**, not a promoted laptop model. Unselected checkpoints, optimizer
state and bulk replay stay under ignored `artifacts/step5/`.

R3 v3 policies are C# code plus JSON profiles, under the frozen
[`../opponents/`](../opponents/) bank. They are not `.pt` neural models. The
original stock Hard policy lives in the private rules DLL; use the verified
runtime transfer described in the [lab guide](../../docs/STEP5_LAB_TRAINING.md).
