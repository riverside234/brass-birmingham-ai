# g1330 AZ_DB teacher qualification supplement

This updates B02.5, B03 and B04 for the versioned DOMiNO search teacher. It includes current trainer sources, the g1330 DOMiNO review fixes and the six unchanged B01–B04 prerequisite archives. Historical source ZIPs are retained as evidence/build inputs. **Do not extract them over this update.** Use the updated project checkout and its existing private lab runtimes, IL data, run and full checkpoints.

Reference: `joint7-v11-2-grouped-lr-critic`, generation **1330**, **425,600 games**. Full checkpoint SHA-256:

```text
d7e4e9473472251289b20991f423908c30a55a548a66e7be6afbc0c1f0c236b8
```

The report directory is `results/joint7-v11-2-grouped-lr-critic-g1330-20261031T024940047516Z`. It is evidence, not a resumable training directory. The original DOMiNO adoption at g1326, current PPG v19 recipe and fixed group entropy 0.1 are preserved. `teacher_phase` is absent in the reference; optional adoption adds v2 as off and disables dormant teacher value. The installed game is unchanged. Lab qualification and B05/B07 production activation remain **Pending**.

## Install and qualify

Pause training at a clean committed boundary for the whole check. Use the existing `ai-training` environment, Rust 1.99.1, Mono and CUDA/PyTorch prerequisites from B01. From the lab project root:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-g1330/rust-lab-az-db-g1330.zip
python scripts/run-rust-lab-qualification.py --stage all --device cuda:0
```

Copy `bundle.json` alongside the ZIP when transferring manually; Git carries both. The launcher verifies the latest source and immutable archive hashes, runs B01 → B02 → B02.5 → B03 → B04, and exports results even on failure/deferred readiness. It automatically passes the g1330 supplement and `--teacher-profile domino` to the revised stages. Do not use old extraction commands or edit hash/signature files to admit mismatched code.

**Expected g1330 outcome:** B04 defers with `missing_retained_intrinsic_fit`. The supplied g1327–g1330 reports show no retained intrinsic fit. Old checkpoints still restore all training state; they cannot authorize diversity search from unsupported metadata. Continue ordinary training normally. The new receipt is written only after accepted, nonzero ordinary PPG intrinsic fitting with actual support and phase normalization. No additional training is launched by qualification.

After ordinary training advances, use its new committed identity explicitly:

```bash
RUN=artifacts/step5/joint7-v11-2-grouped-lr-critic
G=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["iteration"])' "$RUN/state.json")
SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["history"][-1]["sha256"])' "$RUN/state.json")
python scripts/run-rust-lab-qualification.py --stage all --run "$RUN" \
  --expected-generation "$G" --checkpoint-sha256 "$SHA" --device cuda:0
```

Single-stage commands use `--stage b025`, `b03` or `b04`. B03 needs `--b02 PATH_TO_ORIGINAL_LAB_QUALIFICATION_JSON`; B04 needs `--b02 PATH --b03 PATH`. B02 needs `--b01 PATH`. Every prerequisite must match the same source, run checkpoint and device; exported Git copies are for review, not reusable authorization.

B02.5 runs bounded authored CPU fixtures, one authored CUDA policy step/wrapper parity and a read-only adoption check. B03 restores the full real state, checks both intrinsic Adams/controller/fingerprints and reuses B02 complete families to prove normalization, with zero new games or optimizer steps. Once readiness passes, B04 runs:

- Two root calibrations × 8 completed trajectories, plus two K1 continuation probes × 8. Continuation must actually occur; failure does not expand the corpus automatically.
- Neutral/context-only/diversity controls on the same two roots, counterbalanced, **4 seconds/K16** each. Neutral and context-only have separate two-root/eight-trajectory calibration.
- Exactly two full isolated shadow tables, one train and one held-out, one session/thread per table and two concurrent lanes. Calibration/control probes never enter the buffer. Missing cases are covered by fixtures, not more games.

This is bounded correctness/cost qualification, not playing-strength or seven-policy separation evidence.

## Optional config-only adoption and ordinary continuation

The following helper is implemented and generation/hash checked. It creates a new config; it does not modify the run or start training:

```bash
python scripts/configure-teacher-phase.py --run "$RUN" \
  --expected-generation "$G" --checkpoint-sha256 "$SHA" \
  --profile domino --mode off \
  --output "configs/step5/joint7-az-db-off-g${G}.json"
```

Use the variables from the prior block, read from the actual paused run. Repeating the same command is idempotent. Keep the existing IL directory `artifacts/step5/features4-v11-2-batch7-il`. On the next separately requested ordinary training continuation, select that config through the existing `scripts/step5.py --config ... train --run ... --imitation-run ... --resume-reviewed-source --expected-generation ...` interface. Weights, every Adam/RNG state, PPG history, DOMiNO state, PFSP and progress continue in the same run; do not rerun the DOMiNO adoption helper or initialize a new learner.

The config helper also validates later exact `--mode shadow`, `on` or `off` changes from an already adopted profile. Production `shadow`/`on` execution remains intentionally rejected until B05/B07 qualify update/adoption. Use the isolated B04 command above for the currently implemented shadow path. No unimplemented production activation flag is advertised.

## Return results through Git

Every launcher invocation writes a fresh, Git-visible directory:

```text
results/rust-lab-qualification/<UTC>-g<generation>-<stage>-<unique>/
```

Review `evidence/runner.json`, stage qualification receipts, readiness, restore/normalization checks, calibration, `matched-4s-controls.json` and teacher/inference reports. Large text reports are gzip compressed. Models, build trees and replay/cache payloads are excluded. Keep original evidence under `artifacts/rust-backend/` for subsequent qualification gates.

```bash
git add results/rust-lab-qualification/
git commit -m "Add joint7 AZ_DB lab qualification results"
git push
```

Pull on the current computer to inspect those results. No commit/push is automatic. To export an older or interrupted lab directory:

```bash
python scripts/rust_lab_results.py artifacts/rust-backend/YOUR_EVIDENCE_DIRECTORY
```

Local evidence and limitations are in [the report](../../ai-research/RUST_BACKEND_AZ_DB_REPORT.md) and `local-development.json`. The reference model is still a training checkpoint; no final model or game release is bundled.
