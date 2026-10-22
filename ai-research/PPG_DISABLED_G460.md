# PPO-only continuation from g460

2026-10-21. The supplied export is **generation 460, 147,200 ordinary games**. The “iteration 418” text in the request is stale. No production checkpoint has been edited and no production training or evaluation was launched locally.

## Effective changes

| Setting | g460 saved recipe | New continuation / fresh default |
| --- | ---: | ---: |
| `value_phase.enabled` | `true` | `false` |
| `update.microbatch_size` | 512 | 2048 |
| `update.max_candidate_cells` | 131072 | 524288 |
| PPO effective batch | 2048 | 2048, unchanged |
| Critic batch | 2048 | 2048, unchanged |

The generated continuation changes exactly those three leaves. The model, completed IL identity, PPO objective, rewards, LR/warmup/controller state, epsilon/temperature/entropy, branching, opponent pool and ratings remain unchanged. Collection/inference candidate limits are unchanged. Larger physical PPO batches retain all legal actions and existing strict behavior verification, masking and exact KL guards; accumulation still normalizes by the same effective optimizer batch. Floating-point accumulation order can differ from 512-row microbatches. GPU speed and peak memory require measurement on the training host.

PPG is disabled in fresh config generation as well as the explicit g460 config. The optional eight-age recipe remains available through an explicit enablement helper. Completed IL files are unchanged.

## Checkpoint and cache handling

The v5 recipe previously rejected `enabled=false`, and the old disable helper recognized only the earlier per-iteration PPG checkpoint format. The reviewed transition now accepts the eight-age checkpoint, preserves both saved optimizers (including a decayed value-Adam LR), weights, RNG, progress and lineage, and does not open or rebuild the old auxiliary cache. Dormant value Adam performs no steps or phase resets.

Eight-age PPG already supplied **PPO** work to GAE evidence. Disabling it preserves those evidence windows, attained lambdas and table activation. The current 8 self-play / 8 history / 2 v4 whole-game holdouts, ordinary PPO critic training, economic health and PPO KL safeguards remain active. Snapshot R² preserves its weight and frozen reference; its existing training-regime rule starts new stage-local observations on the next committed PPO-only iteration. No old PPG steps become PPO evidence.

On startup, under the run lock, restore/validate the checkpoint and persist the exact reviewed PPO-only source/config transition first. Then remove known generation arrays, manifests and partial writes from **both** `.ppg-eight-cache` and `.ppg-value-cache`, before collecting the next iteration. This also handles owned stale auxiliary caches at fresh PPO-only startup. Audit every target in both directories before unlinking any file; refuse unknown artifacts, invalid owners and symlinks/junctions. There is no recursive deletion. Small `owner.json` markers remain.

Cleanup reports deleted files/bytes in the console, the next iteration report, and `ppg-disabled-cache-cleanup.json`; the report exporter already includes root JSON metadata. Ordinary replays, checkpoints, populations, IL datasets and fixed policy/validation probes are preserved. No new PPG replay is collected. A restart immediately after cleanup can still migrate the original g460 checkpoint without the deleted cache. Resuming the old PPG-enabled recipe instead would require its matching replay backup.

Source review preserves all **42,308** existing expanded receipts and appends explicit routes. The final source is `afcb66986a870567dbbd22ea45c5f43467ca0f7829948f04b561a9f8019e556b`; model/learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. Unknown configuration changes and pending old collections still fail closed.

## Continue after Ctrl+C

After syncing the revision, run from the Linux repository:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 460 --apply && \
python scripts/joint7-ppo-only.py --expected-generation 460 \
  --microbatch-size 2048 --max-candidate-cells 524288 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only-g460.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only-g460.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This trains g461–g480. Recovery archives an unfinished next collection and refuses an unexpected committed generation or uncommitted checkpoint requiring separate recovery. The config helper locks and reads the actual run, refuses conflicting output, and never edits a checkpoint. On later clean continuations, repeat only the training command. Do not use `--fresh` on this existing run or rerun old g418/entropy/restart-160 hotfixes.

## Log observations and validation

At g460 the eight caches contain **15,797,004,753 critic bytes + 3,712,573,288 policy-input bytes = 19.51 GB**, excluding manifests. Cleanup frees the actual present bytes, including any known partial next cache. g460 spent **7.665 s** writing cache data and **8.334 s** in the full value-phase path despite being a non-auxiliary iteration. The last scheduled phase, g458, attempted 96 value steps, retained 32 and stopped on validation patience, with **151.90 s** reported for auxiliary fitting. Disabling PPG removes that auxiliary fitting/cache-writing work; current-iteration holdout diagnostics and ordinary utility-head calibration remain.

The existing g460 v4 EV warning and anchor-budget/recovery-latch LR blockers are preserved. This request does not relax health thresholds or demonstrate stronger play.

Validation: **129 focused regressions passed in 189.13 s**, plus the separately run fresh-start cache-cleanup regression passed (**130 total**). Coverage includes exact g460 configuration adoption, strict source/pending-collection checks, value-Adam LR preservation, nonempty PPO GAE/R² state, zero PPG/cloning/RNG work, pre-iteration cleanup of both formats and partial files, unknown-file/owner/link refusal, cleanup restart idempotency, helper generation guards, physical microbatch/candidate-cell boundaries and complete row coverage, existing auxiliary opt-in behavior, PPO probability/KL guards and exact checkpoint resume. The suite initially exposed a fresh-generator path that had retained the older disabled v4 recipe; it now generates the current disabled v5 recipe consistently and the final defaults checks pass. Final log: `artifacts/ppg-off-g460-review/tests-final.log`.

The final isolated CPU smoke used the current **5,928,616-parameter v11.2 model**, authored 18-game fixtures and a synthetic g460 checkpoint with nonempty optimizers. It performed **8 PPO steps and zero PPG steps**, retired both old cache formats, verified unchanged incoming checkpoint hashes, and verified an exactly matching next PPO update after checkpoint restoration. It exercised the final configured 2048/524288 limits on small authored data; it is not a full-sized GPU benchmark. Wall time was **11.83 s**, first PPO update **2.18 s**, while focused tests shared this host. Evidence: `artifacts/ppg-off-g460-review/smoke-final/report.json`.

Changed files: recovery `ppg_config.py`, `config.py`, `actor_value.py`, `value_phase.py`, `value_buffer.py`, `league.py`, `resume.py`; `scripts/joint7-ppo-only.py`; the new `configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only-g460.json`; source/handoff review registries; PPO/PPG tests and smoke; this report and `docs/MLP_V11_LAB.md`.
