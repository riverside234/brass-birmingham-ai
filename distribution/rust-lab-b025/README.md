# B02.5 teacher-phase transfer

2026-10-29. **Python T1 implemented; lab qualification Pending.** B01 and B02 lab qualification remain Pending. Native teacher collection/label exports remain B04; PPO/PPG continuation and production adoption remain B05/B07.

`b025-search-v1-teacher-phase.zip` contains the Python changes, tests, reviewed source transitions, helper, qualification runner and plans. It supplements the current project checkout and existing private training runtime. It contains no checkpoint weights or game binaries. `bundle.json` identifies the ZIP; `transfer-manifest.json` pins each text file (LF-normalized SHA-256) and the complete trainer implementation. Existing B01/B02 ZIPs are unchanged.

## Lab qualification

Use the existing Python 3.11+ `ai-training` environment and updated project checkout. If installing the earlier search_v1 B01/B02 updates, extract those **before** B02.5: their archives contain older documentation/source files. B02.5's fixture qualification can run before B01/B02 qualification and does not mark either gate passed.

Pause your joint7 run at a clean committed iteration boundary. From the lab project root:

```bash
conda activate ai-training
unzip -o distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip
RUN=artifacts/step5/joint7-v11-2-grouped-lr-critic
G=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["iteration"])' "$RUN/state.json")
bash scripts/run-rust-lab-b025.sh "$RUN" "$G" cuda:0
```

The runner checks the transferred files and reviewed implementation, runs 24 bounded CPU fixture tests, optionally runs **one** four-row CUDA policy step, then locks and reads the actual committed run. The run check verifies checkpoint/signature identity, restores the full model and Adam on CPU, compares Adam state, and validates any existing teacher buffer receipts. It does not write a config or commit/advance training. The run lock uses the existing scheduler's ownership mechanism.

Results are printed at `artifacts/rust-backend/<timestamp>-b025/qualification.json`, with `fixtures.log` and an optional `device.log`. `passed: true` qualifies this Python scope only. A missing/mismatched checkout, stale generation, unfinished iteration, wrong checkpoint, tampered shard or failing test produces a failed receipt. Keep that evidence and rerun into a new output directory after fixing the cause.

Use `cpu` instead of `cuda:0` to omit the device step. CPU tests hide CUDA devices so checkpoint fixture RNG capture does not initialize the lab GPUs. No full games or training iterations run. The CPU suite has a 180-second watchdog; the optional CUDA fixture has a 60-second watchdog.

For fixtures alone, without inspecting a real run:

```bash
python scripts/qualify-rust-lab-b025.py --device cpu
```

## Configuration helper

The helper writes a proposed config from the **actual saved recipe**, preserving newer joint7 settings. It checks the run lock, expected generation, committed checkpoint hash and reviewed transition; repeated identical output is allowed. First adoption must be `off` and disables teacher-value fitting:

```bash
python scripts/configure-teacher-phase.py \
  --run "$RUN" --expected-generation "$G" --mode off \
  --output "configs/step5/joint7-teacher-off-g${G}.json"
```

Future mode/share toggles use this helper and the existing `--resume-reviewed-source --expected-generation G` training route. **B02.5 deliberately refuses live `shadow`/`on` training until B04 supplies native collection and exports.** It cannot silently fall back to legacy Masked distillation. Do not start the long late-training rollout as part of this qualification.

The future teacher profile remains search_v1, 4 seconds/K16, one session × one thread, one pending path, V205 disabled. It is independent of the installed game's selectable profiles.

See [implementation and scope](../../ai-research/RUST_BACKEND_B025_REPORT.md), [migration milestone](../../ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md#b025--revise-the-teacher-distillation-phase), and the [detailed teacher specification](../../ai-research/TEACHER_SEARCH_PHASE_REVISION_PLAN.md).
