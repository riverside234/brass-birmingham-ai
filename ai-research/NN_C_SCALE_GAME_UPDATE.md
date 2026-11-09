# NN search c_scale 1 — local release 1.4.54

Date: 2026-11-09.

All NN seats now search with Gumbel `c_scale` 1.0 instead of 0.1, in every NN package (`alpha-g890`, `beta-g1290`, `il-v11-2`) and every search setting (K8/K16/K32, temperature 1.0/1.5).

## What changed

Release `1.4.54-rust-local` edits `config.c2` from `0.1` to `1.0` in the 18 `policy*.json` files, and updates the three `build.json` lists that pin them. Nothing else changes: the managed assemblies, ONNX models, ONNX Runtime and the Rust workers are the same files as 1.4.53.

## Effect

The search adds `c2 × (50 + max visits) × Q` to each action's prior when ranking actions. With c_scale 1 that term is ten times larger, so the search follows its value estimates much more strongly and the final choice leans less on the raw policy.

## Verification

- Each of the 18 policies was started on its real worker. Each loaded, passed its warm-up search and reported the same search settings as before.
- The installer was run forward, rolled back and run forward again, checking every native file against the qualified list.
- A full turn in the running game (Unity) has **not** been played on this release. The receipt records `nnCScaleInstalledSmoke: pending`.

## Rollback

With the game closed:

```powershell
& ./scripts/install-nn-c-scale.ps1 -Action Rollback
```

After an interrupted installation, use `-Action Recover`. Saves are not touched.
