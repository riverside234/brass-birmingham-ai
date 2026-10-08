# Fresh training in conda: selected v4 teachers and Masked ISMCTS NN

Recipe `step5-training/2.2.0` uses a fresh **1,840,109-parameter MLP-ResNet** and **Masked ISMCTS + NN** for search/distillation. The source-based six-style revision and comparison requirements are documented in [the opening-book note](../ai-research/TOURNAMENT_STYLE_V3.md). The user selected the previous advanced bank, `bank-3dc65253a610191d66a5`, as **v4**, with productive-development stall reset and opening-biased candidates. Further strength tests were stopped; this is an explicit selection rather than a tournament promotion. The rejected network/Scout additions are excluded. `training/opponents/current.json` and the distribution both identify that exact bank. The main lab preset is **24 queued heavy tables, four private search workers per decision**, with one shared 60-slot CPU admission gate under your 64-core allocation. Waiting for admission does not consume the search clock.

Imitation and heuristic evaluation use searched Masked **v4** at **one second/four workers**. Active lab presets now apply a one-time recovery schedule: **60% searched-v4 tables for 20,000 ordinary games, then 30% for 10,000 games**. Historical opponents stay at 25%; current self-play occupies 15% then 45%. PPO also uses a small capped insolvency cost. The schedule begins when a new run is created or when an existing run adopts this reviewed revision, and persists across restarts. Afterward the original mature curriculum resumes at 70% current self-play / 25% historical / 5% searched v4. Internal `v3-*` profile IDs and `training/runtime/v3*` paths remain compatibility aliases for the selected bank. Start a new model from random weights. A completed, fresh standalone imitation run can explicitly hand its selected weights to self-play via `--imitation-run`; its optimizer, population and neural replay are not imported. Audited entirely heuristic datasets may be reused with their original receipts. Same-run continuation from reviewed versions preserves existing PPO progress; see the insolvency recovery section below.

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

## Recover evaluation startup after a missing working directory

If evaluation fails in `multiprocessing.spawn.get_preparation_data()` at `os.getcwd()` with `FileNotFoundError`, the parent process has lost its working directory. This does not identify which directory disappeared, and does not mean a checkpoint or replay is missing. A nonempty `iteration-0039` folder does not establish the process's actual working directory. Cleanup of a retired launch folder is one possible cause, not a confirmed diagnosis of this lab failure.

Update `scripts/step5.py` from this revision. The launcher now changes to its own project root before importing training dependencies, so new workers inherit a stable directory outside iteration cleanup. Config/artifact paths still resolve through `--root`. The learning/search implementation signature, hyperparameters, checkpoint format and generated data are unchanged; existing same-run and partial-evaluation receipts remain valid. No new source migration is required for this launcher-only fix.

**For the reported generation-40 failure:** iteration folders are zero-based. `iteration-0039` produces `generation-0040.pt`. At the reported `schedule_evaluations` call, the trainer has already saved the new checkpoint and committed `state.json`. Keep both generations and let the saved state select the latest checkpoint; do not delete iteration 39 or restart from imitation. After syncing the fix, inspect the state from the stable project root:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python -c "import json; from pathlib import Path; s=json.loads(Path('/data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/state.json').read_text()); print('Next iteration:', s['iteration']); print('Resume checkpoint:', s['history'][-1]['path'])"
```

For this failure, expect next iteration `40` and a path ending in `generation-0040.pt`. Checkpoint hashes, optimizer state and lineage remain verified on resume. Retry the original command with the same run, config and IL origin; this example retries any due evaluation and then trains **one additional iteration**:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 1
```

Use the original run/IL paths if yours differ, and retain any original `--config` option. Increase `--iterations` for the desired additional training, not the total since imitation. The launcher fix does not require `--resume-reviewed-source`; it is retained here from the existing continuation command. The pending evaluation keeps its original specification and reuses valid completed games. Evaluation games remain outside training. The local checks exercise a removed launch folder, simulated missing `getcwd()`, real spawned children and unchanged continuation guards; the real POSIX unlink check is provided for Linux and skipped on Windows.

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

Without the optional performance flags, `imitate` uses the selected config (main defaults: 16 replay readers, effective batch 2,048, microbatch 512, FP32). `--device cuda` applies to the learner; parsing gzip/JSON and validating/encoding replays are CPU work. The initial loading stage is expected to have little GPU activity. Each reader has one tensor thread, and the number of readers is capped by the number of selected replay shards. Increasing Torch's tensor threads alone does not parallelize Python replay processing.

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
| Learner preparation | 16 persistent CPU readers (`audit_workers`), checked compact labels/features; two queued pinned CPU batches for CUDA PPO; 20/40/60/80/100% replay-preparation logs |
| BC | LR 1e-4; main effective/microbatch settings; complete legal policy denominator |
| Curriculum | Next 20,000 games: 15% self-play / 25% historical / 60% searched v4; following 10,000: 45/25/30; afterward 70/25/5 |
| Insolvency cost | PPO actor only: 0.02 per debt event, capped at 0.10 per seat/game; no Pass or per-tile liquidation penalty |
| Search teachers | Warm-up 640 ordinary games; due every 160 games; at most 2 collections per iteration, each 4 games and at most 8 policy-only updates; 77.78% accounted wall-time limit (3.5× ordinary time) |
| Distillation evidence | At least 16 qualified decisions, 2 distinct setups, and 2 decisions in each phase; batch 16, at most 2 presentations per sample |
| Routine evaluation | 4-game monitor every 6,400 games; 8-game broader screen every 25,600 (20 / 80 main-preset iterations) |
| Retention | Up to 64 weight-only opponents: 16 recent + initial/IL anchor + 47 older reservoir entries; 2 full resume checkpoints; 2 replay rounds; immediate deletion of retired unpinned files |

The first search collection remains due after the 640-game warm-up. Later due points advance by **160 ordinary games**. Active lab presets use `search_every_games: 160`, `search_collections_per_iteration: 2`, and `search_wall_fraction: 0.7777777777777778`. With 320 ordinary games per iteration, two independently collected and qualified teacher batches can run after the PPO update. The second batch searches with the model after the first batch's accepted updates. Each collection keeps `search_games: 4` and `distill_steps: 8`; it never reuses a small first collection to fill the second. BC, rehearsal, PPO and per-turn search settings are unchanged. The 70-iteration review below documents the exact same-run migration. Search actions remain external behavior and never become PPO neural likelihoods.

Each NN seat in a **training teacher game** reserves exactly one seeded random round from each window:

| Phase in `search.teaching.phases` | Reserved round |
| --- | --- |
| `canal_early` | Canal 1–4 |
| `canal_late` | Canal 5–8 |
| `rail_early` | Rail 1–6 |
| `rail_late` | Rail 7–8 |

Sampling uses the setup seed, sampling seed, absolute seat and independent phase streams. Policy RNG consumption, worker completion order and failed searches cannot change later reservations. The old `turn_interval` / `max_searches_per_seat` fields remain readable for compatibility and generic search/evaluation; training teacher games use these four reservations instead. An unproductive reservation is not redrawn or moved to another phase. A normal four-game collection therefore schedules 64 whole turns: 16 per phase across four seats per game.

The **12-second clock starts immediately after CPU admission**, before observing the turn. Both actions, resource payments and all other subchoices retain that deadline; four workers receive the same remaining time. Tree work stops 1.2 seconds early to leave time for merging and label validation. Native calls already in progress retire before private channels/hosts are reused; any decision returned after the hard deadline loses its teaching label and reports the overrun. This is a cooperative deadline, not process termination. Workers, actions and forced choices never get another 12 seconds. A timeout, fallback or insufficient-coverage result disables further searches in that turn; legal completion can use the supported continuation or the unlabelled fallback without restarting search. Training replicas reuse the coordinator's fallback rather than redundantly simulating their own discarded fallback plans; their private tree statistics still merge normally.

Teaching requires a nonforced choice, completed visits on at least two distinct alternatives, and a supported complete-turn continuation for the chosen visit-leading action. The target still mixes 90% normalized visits with 10% full-support prior. Forced choices, missing evidence, cancelled/late results and prior fallbacks produce no label. Incomplete in-flight rollouts are counted separately; only completed evidence can qualify.

The optional `distillation` config block now declares `min_samples: 16`, `min_games: 2` (distinct setup groups), `min_samples_per_phase: 2`, `batch_size: 16`, and `max_reuse: 2`; omitted blocks receive these defaults. All four phase minima must be met before **any** distillation optimizer step. An insufficient collection is reported and skipped without retries. Qualified rows are shuffled without replacement per pass and presented at most twice, with at most eight optimizer steps: 16 rows permit two steps, 32 permit four, and 64 permit eight at the default batch size. These are initial safeguards, not empirically optimal thresholds. `search` reports the support counts, skip reason, actual steps and sample reuse.

Distillation trains the **policy only**, with the existing weight 0.25. Its terminal game-outcome losses remain diagnostics, not critic targets. PPO and BC retain their four-seat value losses. The value heads receive no distillation gradients, Adam-moment updates or weight decay; the shared trunk can still change their predictions. This uses the teacher action distribution as in [Policy Distillation](https://arxiv.org/abs/1511.06295), while avoiding fitting the critic repeatedly to a tiny searched-game sample.

In each round's `report.json`, inspect `search.teaching.phases`: `scheduled_turns` are planned reservations, `reached_turns` are reservations actually reached, `searched_turns` count turns with a call, `actual_searches` count decision calls, and `qualified_samples` count accepted decision labels. `forced_choices`, `failures`, `timeouts`, `fallbacks`, `insufficient_coverage` and `cancelled` do not count as qualified samples. `successful_turns` means at least one qualified sample; `failed_turns` means reached without one. `unreached_turns` stays explicit. `elapsed_seconds` measures search/merge/qualification calls; `turn_elapsed_seconds` also includes host execution and unlabelled subchoices. `deadline_overrun_seconds`, `completed_rollouts` and `interrupted_rollouts` expose late retirement and evidence coverage. Per-seat reserved rounds and counters also survive cached collection/resume in shard provenance.

The wall-time gate can postpone a due collection. Its available allowance is `ordinary_seconds * fraction / (1 - fraction) - prior_search_seconds`, now **`3.5 * ordinary_seconds - prior_search_seconds`**. This is the user's clarified **150% of the previous allowance**: the old 70% total-time cap allowed a ratio of 2.3333, increased by 50% to 3.5 (7/9, about 77.8%, of combined accounted time). It is not 150% of total wall time or a 1.5× ordinary-time cap. Each four-game collection needs a conservative estimate of **768 seconds plus overhead** of allowance, although parallel tables can finish sooner. The gate is checked separately before each collection, and its actual elapsed collection/distillation time is charged before checking the next. At most two collections run per main-preset iteration; missed or inadequate teachers never trigger extra games or unlimited catch-up. `search.budget` records admitted/deferred work, and `search.collections` retains each batch's evidence, fit, clock and source model. The 12-second turn deadline and four workers remain unchanged. In the supplied run, the cap was not restricting teacher collections; increasing the frequency therefore required the bounded multi-collection coordinator as well as a config change.

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

**Runtime:** CUDA was active. Mean ordinary inference batches were only 1.34 and 1.27 positions; collection dominated, while PPO took 13–21 seconds. Aggregate host/validation/legacy timings overlap, and host-reported neural forward time is not GPU utilization. The 1.84M model need not fill 90 GB of memory. Keep the existing actor/batch sizes for this comparison; do not increase them merely to allocate memory. Teacher collection plus update cost 114.3 seconds, or 12.24% of accounted ordinary-plus-search time. That recorded run used a 25% admission limit; the 6 October update raised it to 50%, and the current 7 October presets use **70%** at the user's request.

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

**6 October decision: keep the learning/search settings.** LR **1e-5**, target KL **0.01**, 2,048-row proportional probe, **12 seconds per shared turn**, four workers, two pending paths per worker, 640-game warm-up and 320-game due interval remain unchanged. The admission limit was 50% for this measurement; the later 7 October request raises it to **70%**. The KL probe now closely tracks the full result; the earlier v1 overshoot to 0.02946 is absent. Stopping after 6/8 minibatches is the guard doing its job, not a requirement to lower LR again until every epoch finishes. PPO checks after a step, so a small overshoot is expected. Actor-loss sign alone is not a strength metric; [PPO's documented KL stopping rule](https://spinningup.openai.com/en/latest/algorithms/ppo.html) supports keeping that guard.

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

**Continue the existing `selfplay-v2` run.** This review changed performance and reporting, while preserving the learning/search settings. The explicit `--resume-reviewed-source` path accepts only pinned source transitions, including the supplied original v2 source (`f85a8c99…`, commit `2ee1bf9`). Signature fields must match exactly: config, data/IL receipts, rules/features, teacher bank and budgets, except for the subsequently requested **50% → 70% search-wall limit** explicitly recorded in the review receipt. The approval is recorded in `configs/learning/selfplay-resume-sources.json`; this is not a general source/config-check bypass.

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

The transition requires a committed iteration boundary. An unfinished next `iteration-*` directory or uncommitted next-generation checkpoint is rejected, with its exact path reported, so old-source partial replays cannot silently enter a new-source collection. Finish/recover that iteration with the previous source before upgrading; do not edit signatures or delete evidence to force the check through. Unreviewed source/config changes and changed IL/data/opponent/checkpoint files remain rejected.

**Continuation verification:** 24 focused checks passed, including exact weight/Adam/RNG restoration, preserved counters and settings, rejected config/source/hash/lineage changes, partial-iteration protection, interrupted atomic transition recovery, existing IL handoff and teacher scheduling. A [two-game native CPU smoke](../artifacts/recovery/reviewed-resume-20261006/report.json) used a synthetic old-source small-model checkpoint fixture, completed one upgraded iteration and one ordinary same-source restart, and performed three PPO steps in each. It verified unchanged original checkpoint bytes and preserved Adam moments at the first resumed update. This tests continuation machinery, not the actual lab checkpoint's strength or CUDA performance; the lab run has not been launched locally.

Compare collection speed, input-validation time, PPO preparation after startup, fixed-sample teacher KL, phase coverage and full PPO KL in the next reports. The automatic four-game monitor still begins at 6,400 ordinary games; six additional iterations are a pipeline/learning diagnostic rather than a strength qualification. Game AI, rule-based opponents, generated-data formats and installed binaries are unchanged.

## Review of selfplay-v2-31 (7 October)

Evidence: [`state.json`](../results/selfplay-v2-31/state.json), [`retention-history.json`](../results/selfplay-v2-31/retention-history.json), and [`31.json`](../results/selfplay-v2-31/31.json). The saved state contains **32 committed iterations / 10,240 ordinary games**, ending at `generation-0032.pt`. Starting from the previous two iterations, that is **30 additional committed iterations**. `31.json` is zero-based iteration 31; the supplied evidence does not contain generation 33. Iteration 30 has compact training/search summaries in state, but its complete runtime/match report was not supplied. Preserve this trained checkpoint and its Adam state.

### Learning and playing behaviour

| Measurement | First two iterations | Iterations 24–31 |
| --- | ---: | ---: |
| Mean final-epoch value loss | 1.1568 | 0.9263 |
| Mean final-epoch win loss | 1.1390 | 0.9130 |
| Mean full-policy KL | 0.01080 | 0.00479 |
| Mean clip fraction | 5.16% | 5.65% |
| Mean policy entropy | 1.9526 | 1.7933 |
| Mean PPO optimizer steps per iteration | 7 | 38.25 |

**Keep LR 1e-5, two PPO epochs, target KL 0.01, clip 0.2, entropy weight 0.005 and the existing batch sizes.** There is no exploding loss or KL overshoot trend in the continuation. Later iterations complete both epochs, with lower loss and a moderate change in policy entropy. Lowering LR again is unsupported; increasing it simply to fill the KL allowance risks accelerating an undesirable strategy. Win cross-entropy is not a playing-strength score, and the changing opponent mixture changes its difficulty.

Against three search-free v4 opponents, the incoming learner improved from **77.30 VP / 3.703 average place / 2.73% win share** over iterations 2–5 (512 games) to **87.07 VP / 3.396 place / 7.81% win share** over iterations 24–29 (192 games). These are different training setups with a seat-balance defect described below, not a controlled head-to-head result. The last report's **95.3 VP / 2.85 place / 15% wins** covers only 20 games, mixing eight fast-v4 and twelve searched-v4 tables as the run crosses 10,000 games; it cannot establish a new win rate.

There is a late economic warning in pure self-play: mean VP falls from roughly 73–75 in iterations 16–23 to 66.34 in iteration 31. Comparing iterations 2 and 31, coal builds increase from **2.68 to 4.06 per seat**, sales decrease from **2.61 to 2.11**, development decreases from **5.57 to 3.73**, and liquidation decisions increase from **0.43 to 0.92**. Pottery remains near **one build per seat**, with no missing-Pottery warning. Rehearsal simultaneously decays from 0.0936 to 0.0008 and expires at 10,000 games. This correlation does not prove that forgetting IL caused the decline: stronger competition can also lower total VP. Keep the current reward and rehearsal schedule while checking the frozen IL comparison below; do not hard-code a coal penalty or force Pottery from these aggregate counts.

Only a four-game monitor at 6,400 games is recorded, and its outcome report is absent from this upload. It compared the candidate against a recent checkpoint. The supplied logs therefore do not establish post-update strength or retention of IL skill.

### Search teachers and runtime

The 30 continued iterations collected **1,361 qualified teacher decisions from 1,920 reserved turns (70.9%)**, using 194 bounded distillation steps. Mean fixed-sample teacher KL improved **0.4024 → 0.3573** after each update; this is fit to searched training labels, not a test-set result. Phase qualification was early Canal **395/480**, late Canal **369/480**, early Rail **308/480**, late Rail **289/480**. There were **413 timeouts**, 41 fallbacks and 102 insufficient-coverage turns. Failed/forced choices remained excluded. Keep four teacher games per due collection, the 320-game interval, maximum eight distillation steps, two uses per sample and all diversity/coverage guards. The user requested increasing the wall-time admission ceiling from **50% to 70%**; active lab/smoke presets and defaults now apply that one config change. It does not increase the 12-second turn clock or force spending 70% when the existing collection schedule needs less.

The latest iteration took **229.0 s collection + 41.8 s replay loading + 12.6 s PPO + 248.3 s search/distillation + 8.4 s cleanup**, about **540.4 s** overall. Search consumed **42.1%** of cumulative accounted ordinary-plus-search time. PPO preparation was 1.67 s, forward/backward 6.04 s and policy/KL measurement 4.43 s. Peak allocated CUDA memory was about **1.27 GB** (4.05 GB reserved); ordinary inference averaged **1.46 positions per batch**. Enlarging GPU batches or model size to fill VRAM will not fix the dominant simulation/search costs.

The recorded **6.85 average Python actor cores** excludes Mono engine processes, the learner and GPU; it is not whole-server utilization. Actor phase timings overlap across processes. CPU admission wait jumped when searched v4 joined the collection, consistent with 48 ordinary actors sharing 60 admitted slots with four-worker searches. Keep the existing 48 ordinary actors, 24 heavy tables, four search workers and eight replay readers for this correction; the logs do not support increasing concurrency beyond the current CPU budget. Search-free v4 remains the early opponent and searched v4 remains the mature 5% opponent.

### Implemented corrections

1. **Uncouple seat rotation from curriculum quotas.** The 20-game quota cycle and `index % 4` rotation previously put the learner into only seats 1/2 at 10% legacy, then only seat 1 at 5%. Rotation now advances once per quota cycle as well as per game. Every fixed quota position visits all four seats over 80 games, without changing the table mixture or resetting the absolute game counter on resume. Historical tables receive the same correction.
2. **Retain the 12-second whole-turn budget, with 1.2 seconds reserved for retirement/merge/validation.** The old reserve was 0.15 seconds. In the last collection, individual finalization overruns reached 0.89 seconds and caused 14 timeouts despite completed search work. Training teachers now stop tree work at 10.8 seconds into the admitted turn. All workers and actions still share the one 12-second deadline; no extra allowance, retry or late-label acceptance is introduced. This trades a little simulation time for usable evidence; measure qualified Rail coverage and completed rollouts on the lab. Standalone search/evaluation clocks and the installed game AI are unaffected.
3. **Use the fixed initial/IL checkpoint for the scheduled four-game monitor.** It remains every 6,400 games with the same seat rotations and cost. This gives a stable forgetting/progress alarm instead of comparing only neighbouring PPO generations. The broader population/legacy panel remains every 25,600 games. Existing saved monitor results are not rewritten.
4. **Remove duplicate diagnostic board walks and measure replay preparation.** Overall diagnostics are combined from the already computed disjoint table-type summaries. Counts and weighted metrics are preserved; floating-point means can differ only by rounding. `utilization.replay_ingress` now reports reader-task CPU, overlapping worker time, read/validation, diagnostics, feature preparation, compaction and parent wall time. CPU totals explicitly exclude process startup/result serialization. Replay validation, legal choices, targets, features and generated-data formats are unchanged.
5. **Allow the reviewed continuation from the supplied source `760778c9…`.** The exact source transition and optional **`search_wall_fraction: 0.5 → 0.7`** change are pinned in `configs/learning/selfplay-resume-sources.json`. All other config fields, model/data/IL contracts, opponents and checkpoint hashes must still match. The transition prints and records the budget change. Model weights, Adam/RNG, population, warm-up and scheduling counters are retained; this is not a restart from imitation.

### Lab commands

After syncing the revision, take a small **eight-game post-update screen against the fixed IL starting model** before a long extension. This is a noisy development check, not promotion evidence:

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/generation-0032.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/initial.pt \
  --lineups recent-neural --setups 2 --no-search \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2-il-anchor-g0032
```

`recent-neural` is the existing single-reference lineup name; the explicit reference here is `initial.pt`, which contains the original selected IL weights in this self-play run. Evaluation games do not enter PPO or imitation. No monitor result for the supplied generation has been inferred or fabricated.

Continue the **same** run for eight additional iterations, reaching 12,800 ordinary games and the next automatic fixed-anchor monitor from the supplied state:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 8
```

The lab's actual `state.json` chooses the latest committed generation. Keep the original IL run/manifest for lineage verification; imitation is not repeated. An active run or old-source unfinished next iteration remains protected by the existing lock and transition guards. Keep LR/search/batch settings fixed for this comparison. Assess the anchor result, per-seat legacy counts, Rail teacher coverage/timeouts, sales/liquidation trends and replay phase CPU before another tuning change. The fixes do not claim a measured playing-strength improvement.

**Local verification:** 37 focused tests passed, covering the 70% admission calculation, allowed checkpoint continuation, seat balance, teacher deadlines, diagnostic parity and parallel replay ingestion. One real native teacher game reached all 16 reserved turns and qualified **14/16** samples, with **zero timeouts, deadline overruns or fallbacks**; two insufficient-coverage samples were excluded. Its 14 labels and single setup correctly failed the minimum-data gate for an optimizer update. This used a fresh CPU model and a small simulation cap, so it verifies execution and qualification rather than trained strength or lab GPU speed. A five-repeat comparison on two saved native games (641 decisions) preserved diagnostic outputs and reduced median diagnostic-only time from **13.33 ms to 6.53 ms** (2.04×). Replay read/validation, IPC and the rest of training are excluded from that microbenchmark; no end-to-end speedup is claimed. Reports are under `artifacts/recovery/selfplay-v2-31-review-20261007/`.

## Review of selfplay-v2-70 (7 October)

Evidence: [`state.json`](../results/selfplay-v2-70/state.json), [`retention-history.json`](../results/selfplay-v2-70/retention-history.json), and the final [`report.json`](../results/selfplay-v2-70/report.json). This is **70 committed iterations / 22,400 ordinary games**, ending at `generation-0070.pt`. Folder/report indices are zero-based. Full reports cover iterations 0–67 and 69; iteration 68 supplies loss/teacher summaries in state but no full match/runtime report. The aggregates below therefore use nine full reports for the last ten iterations. They describe the incoming collection policy, before that iteration's PPO/distillation update.

### Findings and decisions

| Metric | Iterations 16–23 | Iterations 60–69 |
| --- | ---: | ---: |
| Mean last-epoch value loss | 0.9685 | 0.8756 |
| Mean last-epoch policy KL | 0.00491 | 0.00398 |
| Mean clipped-action fraction | 6.36% | 4.66% |
| Pure self-play VP per seat | 73.55 | 49.01 |
| Industries selected for sale per self-play seat (including grouped-sale continuations) | 2.58 | 1.45 |
| Liquidation decisions per self-play seat | 0.52 | 2.02 |
| Pass decisions per self-play seat | 1.05 | 3.59 |
| Pottery builds per self-play seat | 0.99 | 0.63 |

**Counter interpretation:** `sell` counts each industry selected, including additional industries within one Sell action; selling three tiles together adds three, not one. Fewer complete Sell actions can reflect efficient grouping, but the current counter does not measure that efficiency. `liquidate` counts industries removed during compulsory round-end debt settlement, not ordinary merchant sales. Several removals can belong to one insolvency event. `pass` is an action that discards a card; ending a multi-part action uses the separate `finish` choice. The uploaded aggregate reports cannot distinguish avoidable passes from positions with no useful alternative, or quantify grouped-sale efficiency and liquidation losses.

**Follow-up decision:** The user subsequently requested an insolvency penalty and a temporary increase in searched-v4 opposition. Both are now enabled in active lab presets; implementation and continuation instructions are in the insolvency recovery section below. No Pass penalty was added. The following numerical review describes the earlier unshaped run; its loss values should not be interpreted as measurements of the revised actor objective. Action penalties can change the preferred strategy ([reward-shaping analysis](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)).

PPO is numerically controlled: the latest ten iterations have no KL early stops, their final mean KL is below the 0.01 guard, and their 40 steps/iteration complete at LR `1e-5`. Mean entropy declines from 1.84 to 1.67, but these are different legal-choice distributions, not an isolated entropy experiment. A smaller value loss only means better fitting the current targets; it does not establish stronger play. LR, two epochs, entropy/value coefficients, clipping and batches remain unchanged. The follow-up adds a bounded insolvency cost to the actor's win-share return, preserving official critic labels. Do not increase PPO update size to fill GPU memory.

The worsening self-play economy warrants intervention, although aggregate self-play VP alone cannot distinguish regression from competition for resources. Against the **same searched-v4 opponent mode**, the learner's aggregate training win share changes from 8.13% over 160 seat-games in iterations 40–49 to 4.86% over 144 seat-games in the available final reports; VP changes from 83.44 to 79.71 and mean place from 3.34 to 3.56. Seeds differ and samples are small, so these are warning signs, not a controlled strength result. The latest individual iteration reaches only 42.01 self-play VP. No coal penalty, Pottery quota or reset to IL is introduced. Pottery stays fully legal and exploration-supported. The user-requested follow-up below adds the insolvency return separately.

Teacher reliability improved after the earlier retirement fix: the last ten collections qualify **552/640 reserved turns (86.25%)**, with **zero timeouts or deadline overruns**. Phase counts are 160/160 early Canal, 135/160 late Canal, 129/160 early Rail and 128/160 late Rail. There are 21 fallback and 66 insufficient-coverage events; failures still produce no labels. Mean same-collection teacher KL improves 0.3518 → 0.3209, but the incoming policy already agrees with the teacher's top action about 95.9% of the time. This is training fit, not proof of search strength. More independent teaching and varied opponent models are preferable to repeatedly fitting these same sparse labels. Keep **12 seconds/four workers**, including the **1.2-second finalization reserve**, and keep the existing coverage gates and two-use limit.

The latest iteration spends **237.3 s in collection, 41.3 s in replay preparation, 11.6 s in PPO, 203.1 s in teacher collection/distillation and 8.5 s in cleanup**. Replay readers use about 7.02 CPU cores out of eight allocated; they run after collection. Increase lab `audit_workers` to **16**, preserving all validation and replay formats. Keep 48 ordinary actors, 24 queued heavy tables, four search workers and the 60-slot admission gate. Ordinary inference averages only 1.47 positions/batch. Peak allocated CUDA memory is about 1.07 GB, with 5.53 GB reserved. These are memory/host-phase metrics, not a measured GPU activity percentage; whole-server CPU includes Mono work absent from Python actor counters. No lab end-to-end speedup is claimed until measured.

### Historical population and paper comparison

The uploaded pool contains only generations **0, 1, 61–70**. Its old selection repeatedly spaced an already-pruned list; middle generations gradually disappeared. Merely enlarging that same algorithm would not reliably preserve history.

AlphaGo Zero generated self-play with its best accepted network; AlphaZero used the continually updated latest network. Neither method uniformly sampled all historical opponent checkpoints. This distinction is explicit in the [AlphaZero paper](https://arxiv.org/html/1712.01815v1). A league mixing current and historical opponents is closer to [AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf), which also used performance-based opponent sampling to address forgetting. Here a simple stratified mixture is used; noisy four-player table results are not converted into two-player PFSP ratings.

The revised bounded archive keeps **16 recent versions + the initial/IL anchor + 47 older versions**, at most **64 distinct weight-only models**. Older slots use fixed, deterministic hash priorities: a streaming bottom-k reservoir. Repeated selection over retained entries has the same result as selection over full history at fixed capacity; it does not drift toward the recent end. Existing unavailable checkpoints cannot be reconstructed. This run grows the new archive from its surviving population and any surviving full checkpoints. Sixty-four 1.84M FP32 models represent roughly **470 MB of raw weights**, plus checkpoint metadata, while only two full optimizer/resume checkpoints are kept.

Within historical tables, each opponent slot samples **25% initial/IL, 25% recent, 50% older reservoir**; missing strata fall back to available versions. The current learner is excluded when an older version exists. Initial training necessarily uses its only available network until distinct versions exist. The table curriculum stays **70% current self-play / 25% historical / 5% v4** at maturity. Thus training remains 95% neural games. Search opponent ensembles now include the current learner plus up to three frozen models spanning IL, recent and older versions. `historical_opponents.generation_seats` records actual sampling. Increasing inference cache capacity from **16 to 72 models** accommodates the 64-model archive and transient learner/teacher versions without eviction churn. The archive is for opponents; old PPO trajectories are never replayed as on-policy data.

### More teaching, with a bounded budget

The clarified request is **150% of the previous search-time allowance**, not an impossible 150% share of total time. The cap rises from 70% to **7/9 ≈ 77.78%** of accounted ordinary-plus-search time, equivalent to search using **3.5× ordinary time**. The supplied run actually used about **41.24%**, so the old cap was not the bottleneck.

Active lab presets now schedule teaching every **160 ordinary games**, at most **two collections per iteration**, with **four games and at most eight updates per collection**. With the main 320-game iteration this usually means eight teacher games and at most 16 policy-only updates; samples are independent and each collection passes its own diversity/phase gate. Ordinary PPO still collects 320 games and updates once per iteration. Warm-up remains 640 games. Budget admission is rechecked after charging the first collection's actual time; insufficient/failed evidence never triggers extra collection retries. Aggregate `search.teaching` counts both collections, while `search.collections` keeps their distinct source models, thresholds, fit results and elapsed times. Aggregate fit numbers are sample-weighted per-collection comparisons, not one fixed held-out evaluation.

At the observed timings, a second roughly 203-second teacher collection would increase a roughly 502-second iteration to around **705 seconds before replay-reader savings**. This is a workload estimate, not a benchmark: extra teaching deliberately increases runtime. More search cannot by itself correct a consistently wrong value/opponent model; check the fixed-anchor evaluation and economic trends before extending far beyond the next review.

### Continue generation 70 on the lab

The exact source/config transition is pinned in `configs/learning/selfplay-resume-sources.json`. Checkpoint hashes, Adam/RNG, IL lineage, model/data contracts, population and all ordinary progress remain checked and preserved. The future teacher threshold changes **22,720 → 22,560**, so the first new 320-game iteration can serve reservations at 22,560 and 22,720 without replaying past allocations. A partially collected old-source next iteration still blocks migration until recovered with its original source. Completed IL and generated-data formats stay compatible. Installed game files are unaffected.

After syncing this revision, first take a small fixed-IL screen before the generation-70 full checkpoint can be retired. Its 16 games are a learning check, not promotion evidence. The uploaded monitor receipts name evaluations at generations 40 and 60, but their outcome reports were not supplied here, so their results are not inferred:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/generation-0070.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/initial.pt \
  --lineups recent-neural --setups 4 --no-search \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2-il-anchor-g0070
```

Continue the same run, retaining the original IL path, for ten additional iterations to generation 80 / 25,600 ordinary games:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 10
```

Use the lab's actual original run/IL paths if they differ. `state.json` chooses the latest committed weights; this does not initialize from IL again. Automatic evaluation frequency stays unchanged. At 25,600 games the fixed-IL monitor and broader population/v4 panel provide the next scheduled check. Inspect archive age coverage, actual teacher collection counts, qualified Rail labels, sales/liquidation/Pottery trends and replay-reader wall time before a longer extension. Do not infer strength merely from lower loss or longer training.

**Local verification:** 46 distinct focused tests passed, covering streaming history retention across 1,001 generations, seeded historical sampling, separate teacher jobs and update caps, budget rechecking, phase reservations/deadlines, unchanged PPO/data handling, cleanup pins, and exact source/config continuation with Adam/RNG/checkpoint preservation. A four-game native CPU smoke completed two ordinary games and two independent one-game teacher collections, reaching all 32 reserved turns and committing five PPO steps. With its deliberately reduced one-second/four-simulation test budget, it produced zero qualified labels (10 timeouts and 22 insufficient-coverage turns); both single-setup collections correctly made zero distillation updates. This smoke checks scheduling, accounting and rejection, not qualification at the retained 12-second lab setting. Analysis and smoke receipts are under `artifacts/recovery/selfplay-v2-70-review-20261007/`. No lab GPU benchmark, strength tournament or long training campaign was run.

## Insolvency recovery and temporary searched-v4 curriculum (7 October)

This follow-up is enabled in all five active lab presets. It keeps the same model, IL handoff, replay format, learning rate, PPO batches/epochs, teacher search clocks, search frequency and evaluation schedule.

**Insolvency return:** charge **0.02 once per player per income-settlement round with unpaid debt**, capped at **0.10 per seat/game**. A win has utility 1. Multiple industries liquidated to settle the same round count as one event. Taking a loan, carrying affordable negative income, passing, normal merchant sales and grouped-sale continuations receive no additional penalty. **Final Rail has no income settlement and is exempt**, matching the declared B01 rules repair.

Events come from the first debt-phase public view, which exposes every seat's initial shortfall. When no player has an asset requiring a disposal decision, automatic settlement is recovered at the observed round boundary from the last completed action's public cash/income. These derived labels are calculated in the existing parallel replay readers, after validation. Only small scalar returns cross into the learner; serialized demonstrations, replay shards, feature caches and observation contracts do not change.

For an action at time `t`, the unnormalized actor advantage becomes `official_terminal_win_share - future_capped_insolvency_costs - frozen_win_estimate`. Costs are allocated forward up to the cap, then summed backward to earlier decisions by the same seat. The current debt event is already in the past when liquidation choices are made, so it is excluded from their cost-to-go. Later debt events can still influence those choices. This is an undiscounted Monte Carlo return with an action-independent win baseline; it introduces no extra neural head. The baseline continues to predict official wins, and the rank/VP losses also retain their official labels. PPO normalization, clipping and KL checks follow as before. The actor objective is deliberately changed; playing strength still requires evaluation.

`actor.insolvency` reports learner-only event counts, counts by era, total shortfall, affected trajectories, total/mean capped cost and decisions with future cost. Terminal progress also prints event count and mean cost. `liquidate` remains a separate count of asset-removal decisions in the unchanged economy diagnostics.

**Ordinary training tables after activation:**

| Stage | Duration | Current self-play | Historical NN | Searched v4 |
| --- | ---: | ---: | ---: | ---: |
| First | 20,000 games | 15% | 25% | 60% |
| Second | 10,000 games | 45% | 25% | 30% |
| After recovery | Ongoing | 70% | 25% | 5% |

A v4 table contains one learner and three admitted searched-v4 opponents at **1 second/four workers per complete turn**. The percentages describe table composition, not the fraction of all player seats. The temporary schedule overrides the search-free warm-up while active. It retains the 24-table heavy queue and shared CPU admission gate; it does not raise the core allowance. Main 320-game iterations wholly within stage one contain 48 self-play, 80 historical and 192 v4 tables; stage two uses 144/80/96. Expect more collection time and fewer learner trajectories per iteration than the old 70/25/5 mix.

`state.legacy_boost` persists the activation counter, final counter and configuration digest. `report.legacy_boost.stages` records both boundaries and actual games from each stage, including iterations that cross a boundary. Teacher, evaluation and IL games do not advance this schedule. It never rearms on restart. From the uploaded generation-70 state, boundaries are **22,400 → 42,400 → 52,400 ordinary games**. If the run has advanced, both boundaries move with the actual adoption counter; no absolute counter is hardcoded.

Update the lab checkout, activate the existing conda environment, then continue from the same run and original IL source:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 10
```

The explicit source transition preserves weights, Adam/RNG state, the historical population, retained files and all existing counters, then activates the recovery schedule once. Keep the original IL path if yours differs. A main-preset extension of **94 iterations** covers 30,080 ordinary games; the final 80 already use the restored 70/25/5 mix. The shorter command above is useful for reviewing debt rates and fixed-opponent outcomes before committing to that full extension. Existing incomplete-iteration and altered-checkpoint guards remain enforced.

**Verification:** 25 distinct focused checks passed across the revision, covering debt-event grouping, causal cost-to-go, per-seat caps, automatic no-asset settlement, final-Rail exemption, unchanged official targets/replay/cache behavior, serial/parallel PPO equivalence, exact staged quotas and seat rotation, restart/expiry behavior, reviewed checkpoint/Adam/RNG continuation (including consecutive source transitions), IL handoff and existing distillation limits. Two small native CPU smokes ran six games total. The final three-game smoke used one self-play table, one historical table and one searched-v4 table, committed four PPO steps, and recorded 35 insolvency events over six learner seat-games with costs capped at 0.10 each. It used a fresh small model, not the lab's generation-70 weights; it verifies execution rather than playing strength. The final smoke took approximately 114 seconds. Receipts are under `artifacts/recovery/insolvency-20261007/`, including `native-staged/smoke.json`. All seven active/smoke presets and exact reviewed source receipts validate; `git diff --check` passes. The installed game AI is unchanged.

### Soft-update methods: recommendation and evidence

Lower self-play VP alongside fairly stable results against fixed v4 opponents does not by itself identify unstable optimizer updates. The recent PPO KL and clipping measurements are controlled. However, code review identified a separate place to investigate: `ppo_update` checks policy movement, while the later `distill_update` uses supervised teacher loss and gradient clipping without a comparable policy-change gate. Its `teacher_kl` measures fit to search labels, not movement away from the policy before distillation. Policy-only distillation also changes the shared state encoder, so value predictions can move even though the outcome-head parameters receive no gradients. These are mechanisms to measure, not evidence that distillation caused the observed decline.

| Method | Fit for this pipeline | Recommendation |
| --- | --- | --- |
| KL-anchored distillation | Preserve the policy before the auxiliary updates while learning qualified search labels; monitor the entire legal distribution. The auxiliary policy-cloning term in [Phasic Policy Gradient](https://arxiv.org/pdf/2009.04416) is a relevant reference. | First candidate; adapt its preservation mechanism, without replacing this trainer with PPG. |
| Smaller accepted policy steps | Use a lower distillation learning rate plus bounded backtracking if an update moves the policy too far. [TRPO](https://arxiv.org/abs/1502.05477) motivates controlling policy change; this would be a simpler guard, not a TRPO implementation. | Pair with measured KL limits. Fixed weight interpolation alone does not bound neural-policy changes. |
| Lookahead optimizer | Maintains slow weights and interpolates toward a sequence of fast optimizer updates. See the [original paper](https://arxiv.org/abs/1907.08610). | Lower priority: changes optimizer state and update dynamics across both PPO and distillation; no four-player Brass evidence. |
| Slow reference policy / PPO-EWMA | Separates the behavior policy from a smoother proximal reference; see [Batch size-invariance for policy optimization](https://arxiv.org/abs/2110.00641). | Requires correct behavior-probability corrections and clipping semantics; not a drop-in substitution of old probabilities. |
| EMA evaluation model or EMA regularization | Averaging may smooth fluctuations but can lag improvements. [EMAgnet](https://arxiv.org/html/2606.23995v1) studies regularization toward an EMA policy in two-player zero-sum games. | Useful comparison candidate; those results do not establish a benefit for four-player Brass. |

**Recommended follow-up design:** retain a small, reproducibly sampled probe of fresh ordinary-game positions covering both eras and meaningful decision phases. Freeze its full-legal policy probabilities immediately after PPO and before either teacher collection. Log full-distribution KL and value-prediction changes after each distillation update and after both collections. This separates PPO movement, distillation movement and whole-iteration movement; the current report's PPO KL alone cannot do so.

If the probe shows excessive distillation movement, compare a teacher-only learning-rate factor of **0.5** and a tentative **0.005 mean full-legal KL allowance across the entire iteration's distillation**, not a separate allowance for every collection. Add a small policy-anchor term only if needed. These are proposed starting values, not tuned defaults or a proof of improvement. Also report KL by era/phase so a low overall mean cannot hide Rail-specific movement. Any rejection must restore weights, Adam moments/step counters and RNG consistently, with at most two reduced-step retries; skipped updates must remain skipped rather than requesting more teacher games. A probe-based gate is an empirical bound on sampled states, not a universal trust-region guarantee.

Compare this against unchanged distillation using matched fixed-IL, searched-v4 and older-NN panels. Preserve frozen collection policies and exact stored likelihoods; avoid live weight smoothing during collection. If testing EMA separately, initialize an evaluation copy at adoption and update only after a complete PPO-plus-distillation iteration. A proposed 10-iteration half-life gives decay `2**(-1/10) ≈ 0.933`, with only the restart state and one candidate retained. [PyTorch's averaging utilities](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html) provide the averaging mechanism.

**No additional soft-update algorithm is enabled in this revision.** The active changes are the requested insolvency return and two-stage opponent schedule. KL-anchored, bounded distillation is the first recommended experiment after measuring the missing policy/value movement, rather than assuming that smoother weights imply stronger play.

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

Cleanup runs after every committed iteration, including the last iteration requested on the command line. Main training presets now use `quarantine_hours: 0`: retired unreferenced files are journaled and deleted in the same cleanup, so they do not occupy another 24 hours of disk space. It retains two full resume checkpoints, up to 64 slim opponent checkpoints, the initial evaluation anchor, two replay/report rounds, and two recent completed development screens per panel. Extra explicitly pinned files and dataset/formal-evaluation dependencies are retained.

Older round and development-screen metrics are consolidated into `retention-history.json`; their per-game execution receipts and expired raw replays are removed. Two completed cleanup receipts remain; unfinished receipts are preserved for recovery. Empty retired directories are removed. Historical lineage entries in `state.json` remain small audit metadata and may refer to retired full checkpoints: load opponents from the live `population` entries, and resume from the latest checkpoint. Weight-only opponents no longer carry optimizer state or full report copies, and full checkpoints contain a small update receipt instead of another copy of all game telemetry. The in-memory report list is also limited to the latest two rounds.

First inspect a manual plan:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh --apply
```

Use `pin --run ... --paths ...` before a long independent evaluation when retaining an older generation. For reversible delayed cleanup, set `quarantine_hours: 24` in a custom config **before** starting that run; `restore --manifest` then accepts an unpurged quarantine receipt. Zero-hour cleanup permanently deletes the retired bytes. Purge rechecks dependencies and restores newly pinned files; journals also recover an interrupted deletion. Active runs owned by another process and outside-workspace paths are rejected.

The local six-game/three-iteration smoke verified resume, removal of the oldest full checkpoint and replay round, and an empty quarantine after the final iteration. Focused retention checks also cross iteration 10,000, preserve indirect dataset pins and formal evidence, and recover interrupted deletion. Directory metadata reuse and skipping an unnecessary second scan reduced a measured local no-op cleanup from roughly seven seconds to 1.81 seconds. Actual deletion time also depends on retained/pinned data and storage speed.

Report inference batch positions/cells, padding, expired requests, queue/encoding/inference timing, host RPC/IPC timing, completed simulations, distinct alternatives and turn retirement. Warm private lanes avoid per-batch host swapping; obsolete search generations release their lanes instead of filling the lightweight policy cache. Parallel roots use private trees/channels/RNG and await actual completion before reuse. Device utilization is diagnostic: a 1.84M model does not need to fill 90 GB, and slower rule simulation may dominate. GPU/lab throughput and playing strength require fresh measurements on the server.

Self-play now validates and prepares separate replay shards on up to 16 CPU cores. Readers verify replay hashes, membership, split, labels, strict observations and feature receipts before returning compact training data. Full repeated board/history dictionaries stay in those readers. PPO stages its action/value/rehearsal batches in pinned CPU buffers while CUDA works, and likelihood/KL checks accumulate on-device before reading their result. Forced-action handling, the full legal denominator, four-seat targets and KL guards are preserved. Audited rehearsal tensors are prepared once per invocation, reused across iterations, released after decay reaches zero and not reloaded on a later resume after expiry. Actor replay writers hash each shard once for its game receipts.

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
