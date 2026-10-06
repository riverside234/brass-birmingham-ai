# Fresh training in conda: selected v4 teachers and Masked ISMCTS NN

Recipe `step5-training/2.2.0` uses a fresh **1,840,109-parameter MLP-ResNet** and **Masked ISMCTS + NN** for search/distillation. The source-based six-style revision and comparison requirements are documented in [the opening-book note](../ai-research/TOURNAMENT_STYLE_V3.md). The user selected the previous advanced bank, `bank-3dc65253a610191d66a5`, as **v4**, with productive-development stall reset and opening-biased candidates. Further strength tests were stopped; this is an explicit selection rather than a tournament promotion. The rejected network/Scout additions are excluded. `training/opponents/current.json` and the distribution both identify that exact bank. The main lab preset is **24 queued heavy tables, four private search workers per decision**, with one shared 60-slot CPU admission gate under your 64-core allocation. Waiting for admission does not consume the search clock.

Imitation and heuristic evaluation use searched Masked **v4** at **one second/four workers**. Early RL uses its distinct search-free incumbent; at 10,000 games, when heuristic opponents occupy 5% of tables, it switches to searched v4. Internal `v3-*` profile IDs and `training/runtime/v3*` paths remain compatibility aliases for the selected bank. Start the model from random weights. A completed, fresh standalone imitation run can now explicitly hand its selected weights to self-play via `--imitation-run`; its optimizer, population and neural replay are not imported. Audited entirely heuristic datasets may be reused with their original receipts. Resume within the same run and unchanged source/config/dataset; the explicitly reviewed v2 source transition below is a narrow exception that preserves existing PPO progress.

## Transfer and preflight

**Pre-lab review (2 October):** the qualification checker now understands the selected v4 receipt and accepts the same LF/CRLF-only metadata conversion as training. Dataset audits share one serial/parallel validation path, exclude damaged shards before admission, and reject neural rows before creating a heuristic manifest. Evaluation attributes each replay through its job/game receipt, so shard order cannot change the candidate seat. BC and rehearsal use CPU legality metadata for eligibility counts instead of repeatedly reading individual CUDA flags; losses and effective-batch denominators are preserved.

Sync these fixes **before starting the new run**. Completed, audited heuristic datasets remain usable. An unfinished demonstration collection or neural run remains bound to its original source/config signature: keep it intact and use a new output directory after updating the code, except for the specifically reviewed same-run v2 continuation documented below. Do not edit its signature to bypass the check. The v4 bank, private bundle, 24-table/four-worker lab preset and fresh 1.84M-parameter model remain unchanged.

**Windows/Linux metadata fix (2 October):** if `demo-generate` reports `Selected v4 bank differs`, update `src/brass_ai/bank.py` and `.gitattributes` from this revision. Git previously changed the v4 manifest and `selection.json` from CRLF to LF; the stored pins referred to the Windows bytes. Admission now accepts only byte-exact or LF/CRLF-equivalent JSON metadata. Content changes, reformatting, frozen-source edits and executable changes remain rejected. The selected v4 identity and all historical pins remain unchanged; do not replace the hashes manually or regenerate the bank/selection receipt.

After syncing, run this quick admission check in conda before retrying the same `demo-generate` command:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src python -c "from brass_ai.recovery.era_bank import current; print(current('/data/not_backed_up/yxu209/ai-training/')['bank'])"
```

It must print `bank-3dc65253a610191d66a5`. This checks all five training teachers and the evaluation-only Pottery identity without launching games. A failure before teacher admission creates no dataset spec, so the originally empty output directory can be reused. Existing dataset specs remain subject to their original resume checks.

Keep conda `ai-training`. Use the project's pinned requirements; no new environment is needed if those dependencies already work.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python -m pip install -r /data/not_backed_up/yxu209/ai-training/requirements-searchlab.txt
```

Sync this revision's source/configs and tracked `training/` directory first. Read the selected bank from `training/opponents/current.json`; `training/runtime/distribution.json` must identify the same bank. The game uses **two workers**, lab heuristic and NN search **four**. Privately copy the freshly exported `artifacts/step5-transfer/private-rules-runtime-v4.zip` to `/data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime-v4.zip`. The distribution manifest records every member and SHA-256; import verifies the exact inventory before writing. The bundle includes proprietary game dependencies and remains outside Git. Authored workers and frozen source/profiles are tracked; archival bootstrap weights are never loaded.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py import-runtime --bundle /data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime-v4.zip
python /data/not_backed_up/yxu209/ai-training/scripts/check_qualification_reports.py --root /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
```

The bundle importer verifies the full member inventory and hashes before writing. It refuses to replace different runtime files. If an old clone has conflicting privately generated runtime files, use a clean clone/checkout and retain the old run separately. Preflight checks full-game rules parity, the selected bank and a device backward pass. It is not a v3 clock calibration or strength tournament.

Historical pre-lab review validation passed **82 focused Python tests**, the 71-file distribution check, and the two-game rules-parity preflight. The real-engine smoke generated two v4 demonstrations at 1s/four workers, audited them, ran one fresh BC epoch followed by four PPO games across a same-run restart, and completed seat-rotated evaluation. The final audit and evaluation changes were also checked against these actual replays/checkpoints. The four-worker NN probe completed 12 simulations across four alternatives on its warm repetition and produced a valid teacher; the cold repetition exhausted its 20-second test allowance without fabricating labels. That historical probe does not establish coverage at the current training allowance. Check completed simulations/teacher rows on the lab; CUDA/Linux execution and throughput were not measured locally.

Use the bundle exported with this source revision: **10 authored workers and 904 private runtime files**, including the selected bank and immutable S1, early Masked, prior Masked, industry, economic and advanced controls. Bundle SHA-256: `49103db526cf01d43b6374d45f124da2a0c6ac309543bc6a640e295c28764be8`. Current authored executables and frozen bank source/profiles are tracked under `training/`; vendor DLLs stay private. An old bundle is incompatible even if its filename matches. On a conflicting lab checkout, retain the old runs and import into a clean checkout.

The NN config declares `evaluator_kind: neural`, `q_transform: none` and `leaf_policy: new_root_information_frontier`. `future_turns` is removed. Search finishes the current turn, then evaluates a new ordinary acting-root information set; later supported visits can expand it further. Choice/tree/clock limits still interrupt safely. Heuristic reply limits and Q normalization are rejected by the NN schema. CPU batching, shared CUDA inference, self-play curriculum and retention remain enabled. Old search teachers are incompatible; generate new labels after this revision.

## Generate and audit heuristic demonstrations

Read the active bank identity from `training/opponents/current.json`. Its five training styles use **1,000 ms per complete turn and four private root workers**, with K 4/2, guarded sibling Q and zero voluntary reply turns. The new worker implements the forest internally, sharing a CPU heuristic batching/cache queue instead of starting four separate policy processes. Advanced Pottery Variants's opponent identity remains evaluation-only; learner Pottery choices retain full legality and exploration support. Imitation supplies foundations; the mostly self-play curriculum establishes strength beyond the teachers.

For a first dataset, 500–1,000 complete games gives independent train/validation/test setups and broad style coverage. Start smaller to verify throughput. New generation rotates the five admitted teachers. Existing audited heuristic manifests can be loaded with their immutable archived provenance; do not pass old neural replay to `demo-audit`.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-generate --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/masked-24x4-tournament-v4 --games 1000
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-audit --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/masked-24x4-tournament-v4/*.jsonl.gz' --output /data/not_backed_up/yxu209/ai-training/training/datasets/masked-24x4-tournament-v4.json
```

Use a new manifest name: auditing never overwrites an existing receipt. Audit replays every transition against the rules engine, verifies complete legal lists, deduplicates games, partitions by setup and rejects held-out/unadmitted teachers. BC balances teacher/era/phase/action labels without forcing a Pottery build rate. Validation is held out by setup. Early stopping selects the best validation weights after at least three epochs, at most ten, with patience two.

## Fresh imitation, then self-play

You can run these three stages separately, inspect the evaluation, then start self-play. Keep conda `ai-training`. Use new output directories after this code update; existing audited demonstration data does **not** need regeneration. `--initial` and `--from-run` still reject arbitrary older neural checkpoints/populations.

**1. Imitation only.** This initializes the 1.84M model randomly, learns the audited heuristic choices and selects `best.pt` using held-out validation. For the 64-core / 90-GB lab machine, start with 16 CPU replay readers, effective batches of 2,048, and microbatches up to 1,024 positions / 262,144 padded candidate cells. The larger microbatch changes memory/throughput, not the effective-batch loss denominator. BF16 is optional and requires PyTorch support on the selected CUDA GPU; use `--precision fp32` if unavailable.

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda imitate \
  --manifest /data/not_backed_up/yxu209/ai-training/training/datasets/masked-24x4-tournament-v4.json \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-check \
  --epochs 10 --min-epochs 3 --patience 2 \
  --workers 16 --batch-size 2048 --microbatch-size 1024 \
  --max-candidate-cells 262144 --precision bf16
```

Without the optional performance flags, `imitate` uses the selected config (main defaults: 8 replay readers, effective batch 2,048, microbatch 512, FP32). `--device cuda` applies to the learner; parsing gzip/JSON and validating/encoding replays are CPU work. The initial loading stage is expected to have little GPU activity. Each reader has one tensor thread, and the number of readers is capped by the number of selected replay shards. Increasing Torch's tensor threads alone does not parallelize Python replay processing.

Logs print **20%, 40%, 60%, 80%, 100%** separately for train/validation shard loading, tensor preparation, and each epoch's training, natural validation and balanced validation. Loading percentages count completed shards; epoch percentages count decisions. Percentages describe their named phase, not the entire command. Early stopping can finish before epoch 10. The command prints the requested device, precision, worker count and batch settings at startup.

The loader preserves manifest order, validates hashes once per shard, and reuses the existing audited feature receipts. Worker results contain encoded arrays and compact training metadata instead of duplicated board histories. All legal choices and targets survive; train/validation partition checks are retained. During CUDA updates, two bounded background batch-preparation threads pin CPU buffers and submit nonblocking transfers. Diagnostic predictions/losses come back once per epoch instead of after every microbatch. This follows PyTorch's [data-loading performance guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide) and [multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html). Loading uses `spawn`; GPU model execution stays in the learner process.

Monitor actual compute utilization in another terminal; a small network need not consume 90 GB:

```sh
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv --loop=2
```

**2. Evaluate the selected imitation model.** The same imitation run saves `initial.pt`, its random starting model, as a stable comparison. This command runs 16 games: two setups × four candidate-seat rotations, against the random anchor and searched v4 in separate panels. Treat this as a development screen; beating random is not a strength qualification. Check VP, rank, sales, stranded tiles and Pottery opportunities against the v4 panel. Evaluation games never become imitation or PPO training data.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-check/best.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-check/initial.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-evaluation \
  --setups 2 --lineups recent-neural mixed-v3
```

**3. Start self-play from that imitation result.** Use a separate empty self-play directory. The handoff verifies the completed report, source signature, architecture, manifest and immutable selected checkpoint. It loads the selected weights once, resets the PPO optimizer and game counters, and automatically uses the same audited manifest for the existing decaying rehearsal schedule. It does **not** repeat imitation. There is no automatic promotion based on the small evaluation above.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-selfplay \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-check \
  --iterations 20
```

Repeat the identical command, including `--imitation-run`, to resume self-play. The completed imitation run must remain available and unchanged. Retention records its selected checkpoint as a dependency. Do not combine `--imitation-run` with `--demonstrations`, `--initial` or `--from-run`.

### Train again with a different demonstration dataset

After inspecting evaluation, generate additional **training-purpose** games with a different seed and audit a new manifest. Do not recycle the evaluation replays as demonstrations. You can audit just the new collection, or combine old and new collections to keep the earlier coverage and reduce forgetting:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-generate \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/v4-round2 \
  --games 1000 --seed 19100000
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-audit \
  --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/masked-24x4-tournament-v4/*.jsonl.gz' \
            '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/v4-round2/*.jsonl.gz' \
  --output /data/not_backed_up/yxu209/ai-training/training/datasets/v4-round2-combined.json
```

To **continue the selected imitation weights** on that dataset, use `--from-imitation-run` with a new destination. This creates a new optimizer, early-stopping history and validation comparison for the new manifest. A lower learning rate is shown for this continuation. The parent model, dataset receipts and evaluation stay unchanged.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda imitate \
  --manifest /data/not_backed_up/yxu209/ai-training/training/datasets/v4-round2-combined.json \
  --from-imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-check \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-round2 \
  --epochs 10 --min-epochs 3 --patience 2 --learning-rate 5e-5 \
  --workers 16 --batch-size 2048 --microbatch-size 1024 \
  --max-candidate-cells 262144 --precision bf16
```

To **start over randomly** on the new dataset, omit `--from-imitation-run` and choose another empty `--run`. Ordinary same-run resume still requires the same manifest; replacing it in place is rejected. Continuation accepts only completed imitation runs in the current fresh lineage, not arbitrary archived neural checkpoints or PPO runs. Keep the parent run and manifests available; hashes and lineage are checked again at handoff/resume.

For a continuation, its `initial.pt` contains the **previous selected imitation model**, so use `v4-bc-round2/best.pt` versus `v4-bc-round2/initial.pt` in the evaluation command above and a new evaluation output directory. This directly measures whether the new training helped against its starting model. Then select `--imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-bc-round2` with a **new** self-play directory. It cannot replace the weights inside an already-running self-play population.

### Combined command

The original combined command remains available when you do not want a separate evaluation between BC and PPO. It initializes random residual weights, performs imitation inside that new run, resets optimizer moments, then starts self-play:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh --demonstrations /data/not_backed_up/yxu209/ai-training/training/datasets/masked-24x4-tournament-v4.json --iterations 20
```

Repeat the identical command to resume that run. The source/config signature, current checkpoint hash and every population path are checked. After a crash before the first league commit, an identity-matching imitation progress receipt permits BC to restart from the original random seed; it does not import partial weights. A config/code change requires a new directory unless covered by the explicit v2 source review below. Run without either `--demonstrations` or `--imitation-run` only if deliberately testing random self-play; then there is no BC or rehearsal.

Twenty iterations is a pipeline/early-learning run (6,400 ordinary games), not a promise of strong play. Continue based on held-out economics and opponent-panel progress, rather than loss alone. Monitor unflipped tiles, cash distress, sales/merchant beer, legal opportunities versus chosen industries, VP and seat-specific outcomes. Later self-play should establish strength beyond the heuristic teachers.

| Setting | Main lab preset |
| --- | --- |
| Ordinary collection | 320 games/iteration; 48 single-host actors on the 64-core preset |
| Heavy collection | 24 queued tables × 4 private workers; searched decisions share 60 active CPU slots |
| CPU allowance | Shared FIFO gate: 60 active work slots + 4 reserved cores; ordinary turns cost one slot, searched turns four; not OS affinity |
| v4 teacher clock | v4 + Masked ISMCTS; 1-second shared turn deadline after admission; 4 private roots; K 4/2 and guarded heuristic Q. Completion/selection may retire late and is measured |
| NN search | 12-second whole-turn teacher allowance after CPU admission, shared by both actions, subchoices and 4 private root workers; K 16/8, raw Q, new-root-information frontier, selection-only virtual loss 0.02, 2 pending paths / batch per replica |
| Central CPU/GPU inference | Batch up to 64 positions, 131,072 candidate cells, 1 ms partial flush; CPU feature preparation |
| PPO | Effective batch 2,048; microbatch 512; independent critic batch 2,048; 2 epochs; LR 1e-5; KL stop 0.01 with a proportional era/phase probe of up to 2,048 nonforced decisions |
| Learner preparation | 8 persistent CPU readers (`audit_workers`), checked compact labels/features; two queued pinned CPU batches for CUDA PPO; 20/40/60/80/100% replay-preparation logs |
| BC | LR 1e-4; main effective/microbatch settings; complete legal policy denominator |
| Curriculum | v3 tables 40% → 20% → 10% → 5% at 2,000 / 5,000 / 10,000 games |
| Search teachers | Warm-up 640 ordinary games; due every 320 games; 4 teacher games, at most 8 policy-only distillation steps; 50% collection/update wall-time admission limit |
| Distillation evidence | At least 16 qualified decisions, 2 distinct setups, and 2 decisions in each phase; batch 16, at most 2 presentations per sample |
| Routine evaluation | 4-game monitor every 6,400 games; 8-game broader screen every 25,600 (20 / 80 main-preset iterations) |
| Retention | 8 recent + 4 anchor weight-only opponents; 2 full resume checkpoints; initial evaluation anchor; 2 replay rounds; immediate deletion of retired unpinned files |

The first search collection is due after the unchanged 640-game warm-up. Later due points advance by **320 ordinary games**. The active lab presets use **`search_every_games: 320` and `search_wall_fraction: 0.50`** (50%, increased from 25% at the user's request). Custom lower limits, including `0.20` and `0.25`, remain supported when set before starting a new run. `search_games: 4`, the `distill_steps: 8` ceiling, BC and rehearsal settings are unchanged. The 6 October review below changes PPO's learning rate/KL probe and teacher search/update safeguards. Search actions remain external behavior and never become PPO neural likelihoods.

Each NN seat in a **training teacher game** reserves exactly one seeded random round from each window:

| Phase in `search.teaching.phases` | Reserved round |
| --- | --- |
| `canal_early` | Canal 1–4 |
| `canal_late` | Canal 5–8 |
| `rail_early` | Rail 1–6 |
| `rail_late` | Rail 7–8 |

Sampling uses the setup seed, sampling seed, absolute seat and independent phase streams. Policy RNG consumption, worker completion order and failed searches cannot change later reservations. The old `turn_interval` / `max_searches_per_seat` fields remain readable for compatibility and generic search/evaluation; training teacher games use these four reservations instead. An unproductive reservation is not redrawn or moved to another phase. A normal four-game collection therefore schedules 64 whole turns: 16 per phase across four seats per game.

The **12-second clock starts immediately after CPU admission**, before observing the turn. Both actions, resource payments and all other subchoices retain that deadline; four workers receive the same remaining time. Tree work stops 150 ms early to leave time for merging and label validation. Native calls already in progress retire before private channels/hosts are reused; any decision returned after the hard deadline loses its teaching label and reports the overrun. This is a cooperative deadline, not process termination. Workers, actions and forced choices never get another 12 seconds. A timeout, fallback or insufficient-coverage result disables further searches in that turn; legal completion can use the supported continuation or the unlabelled fallback without restarting search. Training replicas reuse the coordinator's fallback rather than redundantly simulating their own discarded fallback plans; their private tree statistics still merge normally.

Teaching requires a nonforced choice, completed visits on at least two distinct alternatives, and a supported complete-turn continuation for the chosen visit-leading action. The target still mixes 90% normalized visits with 10% full-support prior. Forced choices, missing evidence, cancelled/late results and prior fallbacks produce no label. Incomplete in-flight rollouts are counted separately; only completed evidence can qualify.

The optional `distillation` config block now declares `min_samples: 16`, `min_games: 2` (distinct setup groups), `min_samples_per_phase: 2`, `batch_size: 16`, and `max_reuse: 2`; omitted blocks receive these defaults. All four phase minima must be met before **any** distillation optimizer step. An insufficient collection is reported and skipped without retries. Qualified rows are shuffled without replacement per pass and presented at most twice, with at most eight optimizer steps: 16 rows permit two steps, 32 permit four, and 64 permit eight at the default batch size. These are initial safeguards, not empirically optimal thresholds. `search` reports the support counts, skip reason, actual steps and sample reuse.

Distillation trains the **policy only**, with the existing weight 0.25. Its terminal game-outcome losses remain diagnostics, not critic targets. PPO and BC retain their four-seat value losses. The value heads receive no distillation gradients, Adam-moment updates or weight decay; the shared trunk can still change their predictions. This uses the teacher action distribution as in [Policy Distillation](https://arxiv.org/abs/1511.06295), while avoiding fitting the critic repeatedly to a tiny searched-game sample.

In each round's `report.json`, inspect `search.teaching.phases`: `scheduled_turns` are planned reservations, `reached_turns` are reservations actually reached, `searched_turns` count turns with a call, `actual_searches` count decision calls, and `qualified_samples` count accepted decision labels. `forced_choices`, `failures`, `timeouts`, `fallbacks`, `insufficient_coverage` and `cancelled` do not count as qualified samples. `successful_turns` means at least one qualified sample; `failed_turns` means reached without one. `unreached_turns` stays explicit. `elapsed_seconds` measures search/merge/qualification calls; `turn_elapsed_seconds` also includes host execution and unlabelled subchoices. `deadline_overrun_seconds`, `completed_rollouts` and `interrupted_rollouts` expose late retirement and evidence coverage. Per-seat reserved rounds and counters also survive cached collection/resume in shard provenance.

The wall-time gate can postpone a due collection. Its available allowance is `ordinary_seconds * fraction / (1 - fraction) - prior_search_seconds`; it admits a collection only when that covers the conservative estimate for four seats × four turns × 12 seconds per game plus ordinary game overhead. At **50%**, the available allowance is `ordinary_seconds - prior_search_seconds`: search may receive as much accounted time as ordinary collection/preparation/PPO. This is a share of their combined time, not a GPU-utilization target or a per-turn clock. Four teacher games require an estimated **768 seconds plus overhead** of available allowance, even though parallel tables may finish faster. Workers do not multiply this estimate. Actual collection/distillation time is charged afterward, so unexpected overhead reduces future allowance rather than triggering catch-up retries. At most one collection runs per ordinary iteration; no unlimited retry loop is added. `search.budget` records the limit, estimate and admission result. At the supplied log's approximately 819 seconds of ordinary time after 640 games, the new limit would admit that first due collection; the former 25% limit would postpone it. Actual timing can still delay admission. The 12-second whole-turn clock and four workers are unchanged by this limit increase.

Generated demonstrations, replay/teacher schemas and model/checkpoint formats are unchanged. Existing completed fresh IL checkpoints from the immediately preceding, explicitly reviewed compatible source still work with `--imitation-run` or `--from-imitation-run`; model/data compatibility, completion, lineage and hash checks remain required. Do not regenerate your audited demonstrations for this scheduler change. Use a **new self-play run directory** with the existing completed IL run: source-bound PPO runs cannot silently change recipes during resume. IL training itself, generic search/evaluation scheduling and installed game AI are unchanged.

The 5 October scheduler check passed 15 focused tests, including seeded/late-Rail reservations, after-admission shared deadlines, failed-label rejection, an eight-step distillation update, PPO guards and compatible IL handoff. One complete native CPU teacher game reached/searched all 16 reservations (four in late Rail), and cached replay loading preserved its receipts. The fresh random network produced **zero qualified labels**: five searches lacked completed coverage and eleven timed out, so no distillation update ran. Native retirement exceeded individual deadlines by a measured aggregate 1.378 seconds; all late outputs were excluded. This verifies failure handling and scheduling, not useful teacher coverage on the GPU server. The supported-label fixture separately verified the positive eight-step update path.

To repeat only that one-game plumbing smoke on the lab (a fresh network, simulation cap 16 per worker, no saved learner or campaign), use a new output directory:

```sh
PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src python /data/not_backed_up/yxu209/ai-training/tests/Recovery/smoke_teacher_schedule.py \
  --device cuda --output /data/not_backed_up/yxu209/ai-training/artifacts/recovery/teacher-phase-smoke
```

For fewer resources, select `--config /data/not_backed_up/yxu209/ai-training/configs/step5/lab-small.json` before creating a new run. All active lab presets retain four NN search workers and now use the 12-second teacher allowance. The historical filename `lab-16x2.json` currently aliases the 24-table/four-worker preset. All v3/v4 heuristic presets still use one second. `recovery-smoke.json` is a compact local scaffold check with periodic teachers disabled by default. None of these is permission to migrate an old source-bound run in place.

The smaller 32-core preset retains 28 ordinary actors. Heavy tables stay at the selected 24 × 4 in the main preset, sharing the existing admission gate with ordinary games. Increasing ordinary actors supplies more concurrent neural requests; it does not change the v4 or NN search clocks. Keep the effective PPO batch and learning rate unchanged when comparing throughput. More GPU memory allocation alone is not a training improvement.

## Review of the first post-IL self-play run (6 October)

Evidence: [`state.json`](../results/selfplay-v1/state.json) and the supplied extensionless JSON reports [`iter0000`](../results/selfplay-v1/iter0000) / [`iter0001`](../results/selfplay-v1/iter0001). This is 640 ordinary games after the completed `v4-batch6` imitation run, not a long learning curve or a post-update evaluation. No automatic evaluation was due yet.

| Measurement | Iteration 0000 | Iteration 0001 |
| --- | ---: | ---: |
| Ordinary games / meaningful learner decisions | 320 / 27,433 | 320 / 27,380 |
| PPO learning rate | 3e-5 | 3e-5 |
| PPO actor loss | 0.00404 | 0.00345 |
| PPO value loss | 1.22567 | 1.01799 |
| Win loss / VP loss | 1.20732 / 0.06252 | 1.00144 / 0.05049 |
| Entropy / clip fraction | 1.9476 / 14.42% | 1.9626 / 10.76% |
| Full post-update KL / in-loop probe KL | 0.02946 / 0.01024 | 0.01277 / 0.00952 |
| Optimizer steps before stopping | 6 | 14 |
| Incoming NN vs three search-free v4: win rate | 1.56% | 3.91% |
| Same lineup: mean VP / mean place | 74.78 / 3.734 | 74.62 / 3.680 |
| Pure self-play mean VP | 66.30 | 68.39 |
| Ordinary collection / replay loading / PPO seconds | 359.0 / 38.4 / 21.2 | 348.6 / 38.2 / 13.5 |

Value prediction improved on these training samples, and entropy did not collapse. Near-zero PPO actor loss is not a quality score: it depends on the clipped surrogate and normalized advantages. Strength remains weak against even search-free v4. These lineup statistics describe the incoming collection model, not the checkpoint produced at the end of that iteration; seeds/populations differ, and pure self-play's overall win rate is necessarily 25%. They cannot establish that either update caused a strength improvement or regression.

**PPO change:** lower LR from **3e-5 to 1e-5**; keep clip 0.2, target KL 0.01, two-epoch ceiling, entropy/value weights and rehearsal. The old 256-row probe allotted similar counts to very different phases: in iteration 0000, action choices represented 21,741 of 27,433 meaningful decisions (79%), but were underrepresented in that probe. This can hide drift in the common action phase; the logged full KL reached 0.02946 while the probe read 0.01024. The reports do not contain per-phase KL to attribute the entire gap to sampling. The new **2,048-row, proportional era/phase probe** represents the actual actor distribution, with the full final KL check retained. The rate is a conservative starting adjustment, not an optimized value. The reason to keep the guard is that PPO clipping does not itself bound the policy change; [Spinning Up's PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html) describes KL-based early stopping.

**Search/distillation change:** the four teacher games at iteration 0001 reached all **64 reserved turns** but produced only **one qualified label**, in early Canal. There were 14 completed rollouts against 790 interrupted rollouts; 14 turns timed out and 49 lacked sufficient coverage. The former update nevertheless sampled that one label eight times and trained its game-outcome critic each time: win loss fell **0.9903 → 0.0004**, while VP loss worsened **0.1671 → 0.2558**. The falling combined loss was one-point fitting, not evidence of better search or play. The revised policy-only update, diversity gate and reuse bound prevent this case. The longer 12-second shared allowance, two pending paths per replica and removal of duplicate replica fallback work aim to yield more completed evidence; they do not relax legality, hidden-information or continuation checks. Useful lab teacher coverage still needs measurement.

**Runtime:** CUDA was active. Mean ordinary inference batches were only 1.34 and 1.27 positions; collection dominated, while PPO took 13–21 seconds. Aggregate host/validation/legacy timings overlap, and host-reported neural forward time is not GPU utilization. The 1.84M model need not fill 90 GB of memory. Keep the existing actor/batch sizes for this comparison; do not increase them merely to allocate memory. Teacher collection plus update cost 114.3 seconds, or 12.24% of accounted ordinary-plus-search time. That recorded run used a 25% admission limit; active presets now use **50%** at the user's request.

Start a **new PPO run from the same completed IL run** after syncing the changed source/configs. Its source identity is explicitly included in the reviewed IL handoff allowlist; no demonstration regeneration or repeat imitation is needed. The older PPO run remains source/config-bound and cannot resume with changed settings. This command requests six short main-preset iterations (1,920 ordinary games), rather than a long campaign:

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-selfplay-v2 \
  --iterations 6
```

Use an empty destination and retain `v4-batch6` and its dataset manifest. Compare full KL versus probe KL, critic losses, actual PPO steps, and per-phase teacher support before extending the run. An admitted teacher collection with inadequate coverage should now report a skipped update; do not weaken the evidence guard simply to produce optimizer steps. Each generation prints LR, full KL, value loss and teacher rows/steps, and `state.json` retains compact `recent_training` summaries for the latest two rounds. Full details remain in retained reports. Installed game AI, v4 opponents, evaluation schedules, model/replay formats and IL losses are unchanged.

**Verification:** 23 targeted tests passed, covering representative seeded KL sampling, sparse/missing-phase teacher rejection, policy-only updates with existing critic Adam moments, bounded sample reuse, 12-second four-worker deadlines, unchanged phase reservations, microbatch equivalence, config validation and compatible IL handoff. The [one-game native CPU smoke](../artifacts/recovery/selfplay-v1-review-smoke-20261006/report.json) reached and searched all 16 reservations, completed 116 rollouts and qualified **7 labels** (early Canal 1, late Canal 1, early Rail 3, late Rail 2). It rejected four timed-out and five insufficient-coverage turns; aggregate search-finalization overrun was 0.171 seconds, with late outputs excluded. Cached replay loading preserved the phase receipts. The diversity gate correctly skipped this single-game update; the supported 64-label test verified the positive eight-step path. This fresh random-network smoke checks execution, not playing strength or lab CUDA throughput. No long training campaign or game patch was run.

## Review of selfplay-v2 (6 October)

Evidence: [`results/selfplay-v2/state.json`](../results/selfplay-v2/state.json) retains both iteration summaries; [`iter0001.json`](../results/selfplay-v2/iter0001.json) supplies the second iteration's full report. The first iteration's full runtime/lineup report was not supplied. Both runs started from the same completed `v4-batch6` IL model. This remains an early **640-game** run, with no post-update evaluation yet.

| Measurement | v2 iteration 0000 | v2 iteration 0001 |
| --- | ---: | ---: |
| PPO LR / optimizer steps | 1e-5 / 6 | 1e-5 / 8 |
| Actor loss | -0.01085 | 0.00547 |
| Value loss | 1.23595 | 1.07764 |
| Win loss / scaled VP loss | 1.21737 / 0.06437 | 1.06064 / 0.05558 |
| Entropy / clip fraction | 1.9552 / 4.52% | 1.9500 / 5.81% |
| Full KL / probe KL | 0.01113 / 0.01095 | 0.01046 / 0.01035 |
| Qualified search teachers | Not due | 37 from four setups |

**Keep the learning/search settings.** LR **1e-5**, target KL **0.01**, 2,048-row proportional probe, **12 seconds per shared turn**, four workers, two pending paths per worker, 640-game warm-up, 320-game due interval and the **50%** admission limit remain appropriate for the next measurement. The KL probe now closely tracks the full result; the earlier v1 overshoot to 0.02946 is absent. Stopping after 6/8 minibatches is the guard doing its job, not a requirement to lower LR again until every epoch finishes. PPO checks after a step, so a small overshoot is expected. Actor-loss sign alone is not a strength metric; [PPO's documented KL stopping rule](https://spinningup.openai.com/en/latest/algorithms/ppo.html) supports keeping that guard.

Search coverage improved from **1/64** qualified turns in v1 to **37/64 (57.8%)** in v2: early Canal 8, late Canal 9, early Rail 7 and late Rail 13. Completed rollouts rose from 14 to **710**; 425 were interrupted. Nine turns timed out, five fell back and thirteen had insufficient coverage. All failures remained excluded. Aggregate deadline overrun was 0.255 seconds. The update used six bounded steps, 74 sample presentations, at most two per sample, and no value-head loss. Search plus distillation took **240.7 seconds**, or **23.14%** of cumulative accounted ordinary-plus-search time, below the 50% admission limit. Do not extend the search clock merely because a budget remains available.

The six minibatch policy losses ranged from 1.437 to 1.812 and used different samples. Decision-weighted pass means were 1.641 and 1.623; this is compatible with learning, but it is not a fixed-model before/after comparison. The old report's `root_action_top1: 0` was an uncomputed placeholder for teacher updates, not measured zero accuracy. Teacher terminal win/value losses are diagnostics only and should not be interpreted as optimized losses.

**Implemented performance and reporting fixes:**

- Validate each distinct player/board/choice subtree once against its exact pinned schema, then reuse a bounded cache keyed by its full content. Every frame still checks outer structure, array bounds, seat/order/hand consistency and all changed/new content. Choice dispatch uses the schema's disjoint required `kind` branches. Private/unknown fields and mutations remain rejected; no wire or generated-data format changed.
- Prepare the immutable imitation rehearsal tensors **once per training invocation**, reuse them across PPO iterations, and release them when rehearsal expires. Previously the 2,816,690-row rehearsal dataset was rebuilt inside every update. The one-time preparation cost now prints at startup; it is not included in each subsequent PPO timing. Sampling, loss weights, device checks and resulting PPO updates are preserved.
- Calculate teacher root-action/family accuracy and per-phase/action diagnostics. Report `fit_before` and `fit_after` on the **same qualified samples**, including teacher cross-entropy, target entropy and KL. These measure fit to training labels, not held-out playing strength.
- Record `actor_python_cpu_seconds`, `actor_python_average_cores` and `actor_cpu_recorded_games` in ordinary-collection utilization. These measure actor-process CPU, including its threads; they exclude Mono child processes, the learner and GPU. Cached games or multiple simultaneous host lanes report unavailable totals rather than misleading zeros or double counting.

**CPU/GPU assessment:** iteration 0001 spent **348.1 s collecting**, **37.5 s loading replay**, **12.5 s in PPO**, **240.7 s searching/distilling**, and **3.0 s cleaning up** (641.9 s total). PPO preparation alone was 8.81 s; forward/backward was 1.26 s. Ordinary inference averaged **1.28 positions per batch**, almost unchanged from v1. The actors accumulated 2,134 s in input validation across concurrent games; this is overlapping elapsed time, not 2,134 s of serial runtime. CPU admission waits were negligible. The supplied logs do not contain system CPU-busy percentages, and a peak of 48 admission slots is not 48 fully utilized cores. Keep 48 ordinary actors / 24 heavy tables and current GPU batch limits while measuring the code fixes; increasing processes, GPU memory allocation or queue waits is not justified by these logs alone. Four teacher games can use at most four simultaneous searched tables (16 private workers), even though 24 heavy tables are available for larger workloads.

The learner remains weak: in the second iteration it won **0 of 128** games against three search-free v4 opponents, averaging **73.55 VP / 3.828 place**. Pure self-play averaged 69.84 VP. Those are incoming-policy collection diagnostics, before this iteration's PPO/distillation, with different trajectories from v1. They do not prove that the newly saved checkpoint regressed or improved. Pottery had 694 builds and no diversity warning. Keep the curriculum, exploration, rewards, model size and evaluation frequency unchanged; measure more than two early iterations before making strategic or strength claims.

**Verification:** 35 distinct focused tests passed (37 executions, including two repeated checks). They cover exact schema/mutation rejection, bounded caches, cached-versus-rebuilt rehearsal producing identical PPO updates, teacher metrics, value-head protection, IL handoff and search scheduling. The [local validation benchmark](../artifacts/recovery/selfplay-v2-review-20261006/validation-benchmark.json) used the same 128 saved permitted positions with fresh caches over three repetitions: median **4.307 s → 1.093 s (3.94×)**. This is validation speed only, not a measured whole-training or lab speedup. A [two-game native CPU smoke](../artifacts/recovery/selfplay-v2-review-20261006/native/report.json) completed, verified exact R2 feature parity at 32 positions, reused replay caches, recorded process CPU, and ran five PPO updates. No GPU benchmark, strength tournament or long training campaign was launched.

**Continue the existing `selfplay-v2` run.** This review changed performance and reporting, while preserving the learning/search settings. The explicit `--resume-reviewed-source` path accepts only the pinned transition from the supplied v2 source (`f85a8c99…`, commit `2ee1bf9`) to this reviewed implementation. Every other signature field must match exactly: config, data/IL receipts, rules/features, teacher bank and budgets. The approval is recorded in `configs/learning/selfplay-resume-sources.json`; this is not a general source-check bypass.

Sync the revision to the lab, let any active training invocation finish, then run:

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 \
  --resume-reviewed-source \
  --iterations 6
```

The command loads the latest checkpoint named by the **lab run's** `state.json`. In the supplied snapshot that is `generation-0002.pt` at **640 games**. It restores the model, Adam moments and saved RNG, keeps the opponent population and all curriculum/seed/search/evaluation counters, and performs **six additional iterations** (1,920 games, reaching 2,560 if starting at the supplied snapshot). The next teacher collection remains due at **960 games**; the 640-game warm-up is not repeated. Keep `--imitation-run` to verify the original IL lineage and rehearsal manifest; it does **not** overwrite the resumed PPO weights or reset Adam. Do not use `--initial`, create a `selfplay-v3` directory, or copy just a checkpoint into an empty directory for this continuation.

Before updating the source signature, the trainer verifies the latest checkpoint's hash, model identity, generation and original signature, restores Adam/RNG, and checks live population/evaluation files. It writes a `state-before-source-*.json` metadata backup and an atomic transition receipt in `state.json`. A crash before the next generation commits can resume the same old checkpoint through that exact receipt. Subsequent invocations can repeat the command or omit `--resume-reviewed-source` once the transition is recorded. The original checkpoint bytes and generated-data formats are untouched; normal retention still applies, so the metadata backup is not an indefinite checkpoint archive.

The transition requires a committed iteration boundary. An unfinished next `iteration-*` directory or uncommitted next-generation checkpoint is rejected, with its exact path reported, so old-source partial replays cannot silently enter a new-source collection. Finish/recover that iteration with the previous source before upgrading; do not edit signatures or delete evidence to force the check through. Unreviewed source changes, configuration changes and changed IL/data/opponent/checkpoint files remain rejected.

**Continuation verification:** 24 focused checks passed, including exact weight/Adam/RNG restoration, preserved counters and settings, rejected config/source/hash/lineage changes, partial-iteration protection, interrupted atomic transition recovery, existing IL handoff and teacher scheduling. A [two-game native CPU smoke](../artifacts/recovery/reviewed-resume-20261006/report.json) used a synthetic old-source small-model checkpoint fixture, completed one upgraded iteration and one ordinary same-source restart, and performed three PPO steps in each. It verified unchanged original checkpoint bytes and preserved Adam moments at the first resumed update. This tests continuation machinery, not the actual lab checkpoint's strength or CUDA performance; the lab run has not been launched locally.

Compare collection speed, input-validation time, PPO preparation after startup, fixed-sample teacher KL, phase coverage and full PPO KL in the next reports. The automatic four-game monitor still begins at 6,400 ordinary games; six additional iterations are a pipeline/learning diagnostic rather than a strength qualification. Game AI, rule-based opponents, generated-data formats and installed binaries are unchanged.

## Evaluation and selected-backend probe

Scheduled evaluations use frozen snapshots from the same new run. Evaluation games never enter PPO or BC. For a manual 8-game development screen, choose two distinct generations from that run:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh/population/weights-REFERENCE_MODEL_ID.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-screen --setups 2
```

Replace `REFERENCE_MODEL_ID` with a path recorded in the run's `state.json`. Broader and formal panels require two distinct same-run references. The full shortlist preset includes held-out Advanced Pottery Variants. Add `--search` to evaluate the selected search backend; ordinary monitor panels stay fast. Search/evaluation reports do not promote a model without their separate evidence gates.

The automatic four-game monitor is one setup with all four seat rotations; the broader screen is one setup per lineup with all four rotations (eight games total). At 25,600 games both panels run: 12 evaluation games, versus the old 24 at that boundary. Over 25,600 training games the schedule uses 24 evaluation games instead of 192, an **87.5% reduction in game count**. Wall-time savings depend on opponent/search costs. These small screens are noisy learning alarms, not strength qualifications. Manual evaluation sizes and formal promotion gates remain unchanged.

A quick decision probe compares only the selected backend against its own NN prior and, optionally, the same search with top-K fully opened. It is a coverage/latency check, not a tournament:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/masked-search.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/search/lab-gpu.json --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh/generation-0020.pt --compare-unpruned --output /data/not_backed_up/yxu209/ai-training/artifacts/masked-search/probe.json
```

The CPU path is mandatory and uses `configs/search/lab-cpu.json`; laptop CPU uses `configs/search/laptop-cpu.json` (two private search workers, two tensor threads, eight pending paths per root). Standalone probes can override the NN clock with `--seconds`; this never changes the v3 bank's fixed one-second clock. The frozen S1 comparison remains a historical diagnostic control. S2/S3 and alternative neural-search comparisons remain retired.

## Storage and diagnostics

Cleanup runs after every committed iteration, including the last iteration requested on the command line. Main training presets now use `quarantine_hours: 0`: retired unreferenced files are journaled and deleted in the same cleanup, so they do not occupy another 24 hours of disk space. It retains two full resume checkpoints, up to twelve slim opponent checkpoints, the initial evaluation anchor, two replay/report rounds, and two recent completed development screens per panel. Extra explicitly pinned files and dataset/formal-evaluation dependencies are retained.

Older round and development-screen metrics are consolidated into `retention-history.json`; their per-game execution receipts and expired raw replays are removed. Two completed cleanup receipts remain; unfinished receipts are preserved for recovery. Empty retired directories are removed. Historical lineage entries in `state.json` remain small audit metadata and may refer to retired full checkpoints: load opponents from the live `population` entries, and resume from the latest checkpoint. Weight-only opponents no longer carry optimizer state or full report copies, and full checkpoints contain a small update receipt instead of another copy of all game telemetry. The in-memory report list is also limited to the latest two rounds.

First inspect a manual plan:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh --apply
```

Use `pin --run ... --paths ...` before a long independent evaluation when retaining an older generation. For reversible delayed cleanup, set `quarantine_hours: 24` in a custom config **before** starting that run; `restore --manifest` then accepts an unpurged quarantine receipt. Zero-hour cleanup permanently deletes the retired bytes. Purge rechecks dependencies and restores newly pinned files; journals also recover an interrupted deletion. Active runs owned by another process and outside-workspace paths are rejected.

The local six-game/three-iteration smoke verified resume, removal of the oldest full checkpoint and replay round, and an empty quarantine after the final iteration. Focused retention checks also cross iteration 10,000, preserve indirect dataset pins and formal evidence, and recover interrupted deletion. Directory metadata reuse and skipping an unnecessary second scan reduced a measured local no-op cleanup from roughly seven seconds to 1.81 seconds. Actual deletion time also depends on retained/pinned data and storage speed.

Report inference batch positions/cells, padding, expired requests, queue/encoding/inference timing, host RPC/IPC timing, completed simulations, distinct alternatives and turn retirement. Warm private lanes avoid per-batch host swapping; obsolete search generations release their lanes instead of filling the lightweight policy cache. Parallel roots use private trees/channels/RNG and await actual completion before reuse. Device utilization is diagnostic: a 1.84M model does not need to fill 90 GB, and slower rule simulation may dominate. GPU/lab throughput and playing strength require fresh measurements on the server.

Self-play now validates and prepares separate replay shards on up to eight CPU cores. Readers verify replay hashes, membership, split, labels, strict observations and feature receipts before returning compact training data. Full repeated board/history dictionaries stay in those readers. PPO stages its action/value/rehearsal batches in pinned CPU buffers while CUDA works, and likelihood/KL checks accumulate on-device before reading their result. Forced-action handling, the full legal denominator, four-seat targets and KL guards are preserved. Audited rehearsal tensors are prepared once per invocation, reused across iterations, released after decay reaches zero and not reloaded on a later resume after expiry. Actor replay writers hash each shard once for its game receipts.

Each round prints collection, preparation, PPO, evaluation and cleanup elapsed time, mean neural inference batch size, PPO LR/full KL/value loss, and qualified teacher rows/update steps. `state.json` keeps the latest two compact training summaries. `report.json` adds `utilization` with **per-collection** inference deltas, aggregate actor phase times, actor-process CPU seconds/average cores, padding and games/second; the existing `inference` section remains cumulative within one invocation. Phase sums overlap and are not CPU/GPU utilization percentages. The actor-process CPU fields cover Python processes only, not native Mono children or the whole machine. To record actual GPU activity in another terminal during a lab run:

```sh
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 5
```

Use `top` to inspect the Python reader/actor and Mono host processes together, not only the learner PID. In the older `results/1.1` two-game smoke, CPU preparation took about 95–96% of PPO wall time and neural batches averaged about one position. That does not measure the present 48-actor preset. A local two-game replay check measured 3.69 s for serial read/preparation/diagnostics versus 1.97 s with two warmed readers; process startup made the first parallel pass 5.22 s. Readers persist across rounds to amortize startup. CUDA throughput must be measured on the lab; local checks use CPU.

The selected **v4 / patch 1.4.20** uses C# heuristic Masked ISMCTS, six advanced-opening styles, **two game workers** and the existing **2/5/12-second** options. Lab search remains at four workers. Its previous `smoke-7` screen tied 6–6 on wins and averaged 88.79 versus 84.71 VP against the economic control; the user then explicitly selected this bank and stopped further strength tests. This selection does not certify tournament superiority. Productive development resets intent age, and opening goals influence build/setup candidates. The [1.4.7 resource lifecycle](AI_EXECUTION_RESOURCES_1.4.7.md) remains in force. Trained NN deployment is a later qualification step.

## Historical rule-based comparison commands

These commands document earlier S1/Masked diagnostics. They are not required for the v4 transfer or fresh training, and no further strength comparison was run for this selection.

Run eight matched four-player games (two setups × four seat rotations). Each table has one Balanced serial 1s/240-work opponent, one Balanced 1s/two-worker/720-total-work opponent, and two rotating serial S1 styles.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-s1-training.py --tables 16 --setups 2 --output /data/not_backed_up/yxu209/ai-training/artifacts/s1-training-smoke/lab
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-s1-training.py --compare-masked --tables 16 --setups 2 --output /data/not_backed_up/yxu209/ai-training/artifacts/s1-vs-masked-smoke/lab
```

The second command compares S1 with Masked ISMCTS, both at one second and two private workers, alongside the same serial S1 anchors. Selection uses official win rate, then lower mean rank, then higher mean VP; a complete tie keeps S1. The previous threefold work multiplier applies only to the S1 control's 240-to-720 node allowance; it does not multiply Masked simulations. Active presets keep this compatibility field at one.

That historical preset supported 16 concurrent heavy tables; eight smoke games can occupy at most eight. A local laptop smoke used `--tables 2`. Reports provide VP, official split win rate, first-place frequency, mean rank, work counts and actual retirement times. Those smoke games never enter training. The **current** main preset queues 24 heavy tables with four workers per decision. [The recorded local results](V3_S1_TRAINING.md) retain the earlier selection evidence.


## Parallel execution and measurements

The shared admission gate applies to imitation generation, ordinary RL, later searched v3 opponents, NN distillation and evaluation. Search clocks start after admission and all workers retire before a turn lease releases. NN trees own their nodes privately; pending paths use virtual loss only during selection. The central inference queue locks private reply channels, groups frozen-model requests and flushes by the earliest deadline. Heuristic v3 uses a CPU queue, not the GPU.

Ordinary encoding caches semantic action vectors with exact R2 parity; replay gzip uses compression level one. Host persistence, audited seals, derived feature receipts and asynchronous writes remain. Track `cpu_admission`, `cpu.admission.wait`, queue/encoding/RPC times, valid/padded candidate cells and completed search support. These changes were tested locally on CPU; measure Linux/Mono/CUDA throughput on the lab before claiming a speedup.

Historical era-comparison command (eight games, both one second/four workers; not required for this release):

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-era-v3.py --tables 8 --seed 842400 --output /data/not_backed_up/yxu209/ai-training/artifacts/rail-era/lab-confirmation
```

This small standalone smoke runs at most eight tables (32 search slots); the main dataset/training preset queues 24 tables under the shared 60-slot admission gate. It is diagnostic and never enters BC/PPO. Preserve each output directory.



The six **v4** styles are Adaptive Advanced Book, Coventry Goods Opening, External-Beer Three Cotton, Worcester Iron-Cotton, Zero-Loan Industry and Advanced Pottery Variants. Opening/expansion evaluation accounts for funded access, current public beer and real resource-market receipts. The selected bank is `bank-3dc65253a610191d66a5`; the rejected network/Scout revision is not active. Resolve the admitted bank from `training/opponents/current.json`; see [the source and comparison note](../ai-research/TOURNAMENT_STYLE_V3.md).
