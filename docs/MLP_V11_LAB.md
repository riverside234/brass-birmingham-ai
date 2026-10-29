# Grouped MLP v11.2: completed IL upgrade and fresh self-play

**Current g1320 recipe:** use [the g1320 instructions](../ai-research/TRAINING_G1320_REVISION.md)
for continuation and the effective settings. Fresh self-play config generation
also selects this recipe: T1.3 for the first three action decisions per neural
seat (also after a fork), matching decoupled PPO, epsilon .05, fixed entropy .005,
1% branching, .01/.075 insolvency, every-iteration two-age PPG with a two-epoch
cap, both iterations' held-out games for loss acceptance, linear LR warmup to
step 16 followed by cosine decay to .001 times peak, beta minimum
.5, loss checks every 16 steps and periodic KL checks every 32 steps.
Unchecked retained loss checkpoints also require exact KL verification.
Peak remains .65 times actual PPO LR.
The g1320 transition preserves saved adaptive beta and refreshes incompatible
old-reward PPG caches after durable adoption. Its first iteration uses fresh data
only; the second restores the two-age window. Branch points still skip the first
three action decisions. Earlier behavior is never relabeled.
Gate evidence remains current-only. PL bootstrap
execution uses four single-threaded processes and full-fit warm starts.
Completed IL is unchanged; no model architecture migration is part of this update.

**Archived g620 PPO/PPG update:** PPG ran every sixteen committed iterations, using all
sixteen cached ages and all sixteen ages' ordinary holdouts with uniform weights and
a new shuffle each epoch. Its constant rate is **0.35 times the last actual PPO
LR** from that iteration, with no PPG warmup or decay. PPO preserves frozen
pre-update critic outputs through the shared encoder; all fresh value targets
are fitted in PPG. Held-out loss is checked every **16** steps, scheduled KL every
**32**, plus measurements at short tails and of the final retained state.
The throughput caches, fixed five-epoch ceiling and patience 1 remain. The
**1.5-SE rule controls stopping only**. Every new minimum held-out mean game loss
saves weights and matching value-Adam/auxiliary state; each phase returns that
minimum, including step zero if no update improves it. PPG has **no hard KL cap**; policy movement is controlled
only by the cloning KL penalty. PPO's own KL guards remain unchanged. See the
[implementation and official-code comparison](../ai-research/DCPG_G590.md), and the
[archived g620 packed-buffer continuation command](../ai-research/PPG_SIXTEEN_G620.md).
That recipe uses `ppg-value-phase/12.0.0`: cloning beta starts and stays at 1.
Continuation fixes beta at its saved value (the g620 report records **1.0**),
without automatic increases, decreases, KL rejection or KL-triggered retries.
The g620 adoption preserves the current beta and completed
phase counts. It retains available old caches read-only, packs new generations
losslessly, and first runs the sixteen-age phase at g636 on g621–636. Fresh runs
first run PPG at g16. See the report for storage and random-read measurements;
do not repeat `--fresh-ppg-buffer` or the old auxiliary-head adoption command.

V11.2 is the current default: projections **32/64/48/320/352/48 + 12**, fusion **876 → 512**, **eight width-512 residual blocks**, a width-160 action encoder and a 192-wide policy scorer. The trunk and actor are unchanged from v11.1. A shared **34 → 32 → 32** seat-descriptor encoder feeds only the outcome and utility heads. Their new columns start at zero. The base model has **5,910,916 parameters**; the utility model has **5,928,616**. See the [critic revision and validation report](../ai-research/OPPONENT_CRITIC_DESCRIPTORS_REPORT.md).

**Completed v11.1 IL is reused without retraining**, through the explicit upgrade below. Keep `artifacts/step5/features4-v11-1-batch7-il` unchanged. The upgrade writes a new v11.2 handoff directory and proves that every original tensor is copied; policy and original critic outputs match exactly at initialization. It does not convert an existing PPO run or copy its optimizer/progress. Use a newly generated config and a new self-play directory.

## Existing joint7 IL data

The official heuristic strategy remains **v4, `bank-3dc65253a610191d66a5`**. Names such as `balanced-v3-masked-economy` are retained profile aliases; Network/Scout candidates are not enabled. Game and lab now compile the shared search core and the same game policy helpers. New collections have a separate search-runtime fingerprint; old bank/data identities are not rewritten. Reuse completed IL and its dataset unchanged.

The experimental v5 strategy, profiles, candidates and three banks were removed on 19 October at the user's request. The portable v4 bundle is restored to `search-76a2f9ad7529c0e5e2117a36`, the same v4 reference used in the generation-55 run. Pull the restored executable and receipt together with the source before preparing the lab worker. This retains the current search-performance and five-profile sampling fixes.

After pulling search/strategy source changes **and the matching prebuilt files under `training/runtime/shared-search/`**, prepare the versioned lab worker once before **new demonstration generation or self-play** (including search-free v4 opponents). The command prefers the locally compiled portable executable, so the lab does **not** need Roslyn or `csc.exe`. It checks the executable, source/profile reference and private DLL fingerprints, then writes a new directory under `artifacts/legacy-ai/search/`. It does not update a model, configuration, frozen bank or installed game:

```bash
python scripts/build-shared-search.py --root .
```

Linux still uses Mono to run the prepared worker, with the same existing training runtime setup. If the original bank runtime is absent, provide `--managed` pointing to the existing private pinned dependency directory below the project root. Only `BrassLegacyAI.exe` and `prebuilt.json` are distributed; licensed game DLLs remain private. A missing/stale bundle fails explicitly; there is no silent fallback to old search. Git LF/CRLF conversion is accepted only after normalized source fingerprints match; the executable's byte hash remains exact.

On the build machine, regenerate and publish the bundle with `python scripts/build-shared-search.py --root . --compile --publish`, then transfer/commit the updated source **and both bundle files** together. Compiling requires the complete pinned Roslyn 4.12.0 toolset, including its adjacent assemblies; copying `csc.exe` alone is insufficient. `--compiler` selects an existing complete toolset, and `--mono` selects Mono for compilation on Linux. Normal lab preparation requires neither compiler nor downloads.

Teacher and v4 search remain disabled by the generated self-play config. Explicitly enabled searches share deadline admission; fixed simulation quotas retain their watchdog and exact allocation. See [shared-search integration](personality/DEPLOYMENT_PERFORMANCE.md#shared-official-v4-search-for-game-il-and-self-play).

Reuse the already converted, audited **Features-4** dataset as-is. The state/action dimensions remain **3,128 / 327**, and feature, encoder, derived-feature, rules and target contracts are unchanged. The new model checks the static board topology as part of admission. Existing native-board data satisfies those checks, including the three-region Worcester/SouthWest/Kidderminster links.

Keep the manifest, labels, split assignments and raw shards. The normal loader may rebuild source-bound tensor caches in the new IL run's `.imitation-cache/`; do not regenerate demonstrations solely for this upgrade. A path containing v8 is still valid when its contents are Features-4.

Use the explicit completed-v11.1 upgrade, or fresh v11.2 IL if no completed v11.1 model exists. V11 and earlier checkpoints are incompatible. The regular loader also rejects v11.1 weights until the explicit upgrade is performed; source-review flags cannot bypass architecture checks. Old neural PPO trajectories are not current behavior data.

## Training defaults

| Setting | IL | Self-play shared-network updates |
| --- | ---: | ---: |
| Base learning rate | **6e-5** | **6e-6** |
| Previous width-304 rate | 1e-4 | 1e-5 |
| Warmup updates | **500** | **100** |
| First-update multiplier | **0.1** | **0.1** |
| Gradient norm clipping | **1.0** | **1.0** |

Warmup is linear from 10% on the first update to 100% on update 500/100. IL then stays at its base rate. PPO subsequently uses the controller below, bounded to **3e-6–1.8e-5**. These are provisional settings, not a proven optimum. The counter is stored in Adam's parameter group and restored with the checkpoint; restarting does not restart warmup. Self-play counts successful PPO and teacher/distillation shared-network updates. Head-only utility calibration keeps its temporary rate and does not advance warmup. Teacher collection is currently disabled.

The selected JSON config contains:

```json
"learning_rate": 0.000006,
"learning_rate_warmup": {
  "contract": "linear-update-warmup/1.0.0",
  "imitation_steps": 500,
  "selfplay_steps": 100,
  "start_factor": 0.1
}
```

It also sets `imitation.learning_rate=0.00006` and `update.grad_clip=1.0`. To try clipping 0.5, change that value in a new explicitly selected training configuration. `imitate --learning-rate` and `--warmup-steps` override the IL rate and warmup length. Zero warmup steps explicitly disables the ramp. Reported effective rates can be lower than the peak during warmup; full and compact self-play histories retain the schedule counter.

## Optional: fresh v11.2 IL

Skip this section when v11.1 IL is already complete; use the upgrade below. For an entirely new IL run, use the actual converted manifest path if yours differs and a new output directory.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/

python scripts/feature-engineering.py inspect

python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-v11-2-batch7-il \
  --epochs 10 --min-epochs 3 --patience 3 --workers 12
```

The epoch/loader-worker arguments are example lab settings. Inspection should show architecture `brass-four-seat-mlp-resnet/11.2.0`, local layout `grouped-mlp-local-entities/2.0.0` and **5,910,916 base parameters**. The IL CLI now takes its default learning rate from the selected config, so it does not silently retain the old hard-coded 1e-4.

To resume an interrupted v11.2 IL run at its committed epoch, repeat the same command with identical settings and source. Warmup, Adam and RNG restore from its checkpoint. Do not use `--from-imitation-run` to load an older architecture.

To extend this completed 10-epoch run to **20 total epochs**, use the same run and its original settings:

```bash
python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-v11-2-batch7-il \
  --epochs 10 --extend-to-epochs 20 \
  --min-epochs 3 --patience 3 --workers 12
```

`--epochs 10` identifies the unchanged original recipe; `--extend-to-epochs 20` raises its execution cap to 20, **not 20 additional epochs**. Training resumes at epoch 11 after the committed epoch 10. It stops no later than epoch 20 and may stop earlier after three consecutive epochs without an improvement greater than `0.0001` over the recorded best validation loss. Keep the actual original patience/minimum-epoch settings if they differ from this example.

The extension restores the latest checkpoint's weights, Adam moments, CPU/CUDA RNG and warmup counter. It retains all validation history, the previous best checkpoint, the patience counter and the original dataset/checkpoint lineage. Existing prepared shards are reused only through the pinned preparation-equivalence review, with raw-input and cache checksums still verified. No dataset generation or warmup restart is needed.

Repeat the **same extension command** after interruption; the saved total cap is idempotent. It cannot revive a run stopped by patience or silently change the model, dataset, learning rate, batching, loss, precision or other original settings. The extension requires a matching architecture and an explicitly reviewed source; unknown changes remain rejected. The original signature remains immutable, with the increased cap and actual executing source recorded in `state.json`/`report.json` as `epoch_extensions` and in subsequent checkpoints. After this extension commits, self-play handoff requires it to finish (at the cap or by patience); a stale 10-epoch report is rejected. Generate a new self-play config after choosing the final completed IL checkpoint.

`--from-imitation-run` remains a separate weights-only continuation into a **new** run and resets Adam/RNG/warmup. Do not use it for the exact extension above.

## Fresh self-play from completed IL

Use the completed `artifacts/step5/features4-v11-2-batch7-il` directory. Its name and checkpoint remain unchanged. If you still have only v11.1 IL, first run the one-time `upgrade-imitation` command below; skip that command when v11.2 IL is already present. No IL optimization runs in this upgrade.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python -m pip install -r requirements-step5.txt

python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppg-openings-fresh.json

python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-openings-fresh.json train \
  --fresh \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic-ppg-openings \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --iterations 1
```

Only if the v11.2 IL directory does not yet exist:

```bash
python scripts/feature-engineering.py upgrade-imitation \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --output artifacts/step5/features4-v11-2-batch7-il
```

The config/output helpers refuse conflicting existing files; choose unused names for a new run. To continue this **new** run, repeat its train command without `--fresh` and with the desired additional `--iterations`.

The generated config retains q v3 with **epsilon 0.00**, two opening action decisions per neural seat at temperature **2.0**, and **1% per eligible decision** branching after 20 completed iterations and a ready entropy baseline. Action sampling remains probabilistic. Each branch restarts two temperature-2 action decisions per seat after its excluded fork move; intervening subdecisions remain at temperature 1.0. There is still at most one fork per ordinary self-play game. Entropy floors start at **0.95 × original baseline for `canal_action`/`rail_action` and 0.90 × baseline for `canal_other`/`rail_other`**. Each future qualified promotion of the LR gate's KL reference subtracts **0.02 × that same baseline**, down to **0.85 for action groups / 0.80 for other groups**. There is no iteration decay or baseline reset; rejected candidates and repeated rating fits do not reduce floors.

New self-play configs also enable `local_scoring_auxiliary.mode=apply`: training-only ownership/official-VP heads for 49 industry slots and 69 connections. The v11.2 core and completed IL remain unchanged. Auxiliary parameters are separate from the playing model and its PPO optimizer: **91,410** extra parameters, **6,020,026** total during training. The existing two-iteration head-only/five-iteration shared-gradient ramp remains, but the delayed-critic mode fits fresh ownership/VP targets only in PPG (first fresh phase at g16); PPO preserves their pre-update outputs. See the [head implementation](../ai-research/OWNERSHIP_AUXILIARY_HEADS_REPORT.md) and [delayed-critic revision](../ai-research/DCPG_G590.md).

**New configs enable delayed-critic PPG (`ppg-value-phase/12.0.0`) every sixteen committed iterations with sixteen uniform ages and all sixteen ages' ordinary holdouts.** Loss checks run every 16 steps; KL is measured every 32, at epoch tails and on retained weights. Patience remains 1 beyond 1.5 standard errors, with a fixed five-epoch ceiling. The SE rule controls stopping only: each phase restores its strictly lowest mean held-out-loss weights and matching value Adam, including the post-PPO starting state if nothing improves; ties keep the earlier checkpoint. `selection.contract="ppg-minimum-heldout-loss/1.0.0"` replaces coverage-first retention for new configurations. PPG uses a fixed-beta KL penalty only (`kl_cap=null`): no KL-triggered rejection, rollback, retry, stopping or beta adaptation. The g620 continuation preserves beta 1.0; PPO KL guards remain active. Value Adam resets at phase start, then uses a constant **0.35 × actual final PPO LR** for every step. `learning_rate=null`, `learning_rate_source="ppo-last-applied"` and `learning_rate_factor=0.35` bind that rate before the LR controller's next-iteration decision. There is no PPG warmup/decay. The companion's separate configured Adam rate remains unchanged and constant. PPO preserves frozen critic outputs with encoder gradients enabled; no fresh-target value/ownership loss or utility calibration runs during PPO. The actor, entropy, rehearsal, behavior checks and PPO optimizer schedule remain unchanged. Explicit continuation from g620 next runs PPG at **g636, g652**. PPO uses `update.microbatch_size=512`, `update.max_candidate_cells=131072` and `update.preparation_workers=4`; effective actor/critic batches remain 2048. CPU collation remains bounded and ordered, with main-process RNG. Collection retains 1 ms flush, padding ratio 2, model cohorts off, 48 game lanes and the resident model cache. See the [g470 throughput report](../ai-research/SELFPLAY_G470_THROUGHPUT.md) and [g620 command and packing measurements](../ai-research/PPG_SIXTEEN_G620.md).

`update.opening_objective` versions the opening-only PPO correction: its clipping ratio uses frozen pre-PPO raw temperature-1 probabilities, multiplied by detached `min(2, old_T1 / actual_behavior)`. Weights are not renormalized; actual behavior includes temperature and epsilon, and remains strictly verified. Non-opening objectives, value targets, advantages and entropy remain unchanged. `KLT1` is the exact raw-policy probe used by the LR controller; `KLbehavior` independently retains the exact behavior-policy guard at the same PPO limit. The sampled full-data check remains a behavior-policy check. This does not turn recorded returns or GAE into on-policy temperature-1 data. See the [g503 follow-up](../ai-research/SELFPLAY_G503_OPENING_PPO.md).

PPO trains the actor with DCPG value-output preservation every iteration; fresh value fitting occurs in the scheduled PPG phase. Actual retained critic work supports the per-table GAE gates. Snapshot critic loss has **fixed weight 1.00**, with snapshot weight recovery/decay disabled (`critic-rows/3.0.0`, `rollout.mode="fixed"`). Snapshot seats contribute critic rows only; IL/bot exclusions and ordinary whole-game family holdouts remain active. Insolvency costs **0.040 per event**, capped at **0.100 per seat/game**. The additive VP reward is **0.001 × min(VP, 180)**, capped at **0.18** (`reward.vp_cap=180`, combined-return v2). Official VP, winner/tiebreak rules and the VP prediction target remain uncapped. Bounded rehearsal, 50/45/5 tables, the staged PFSP pool, PL ratings and calibrated anchor-KL/LR controls remain active. Qualified policy references and competitive rating coverage await their panel gates in a fresh run. Teaching, all v4 search, Monitor and Broader Evaluation stay disabled. Fresh PPO starts with a new Adam and warmup. The IL upgrade retains the parent checkpoint, dataset, source review and lineage in `upgrade.json`.

For the not-yet-started g620 continuation, the [sixteen-iteration PPG helper and command](../ai-research/PPG_SIXTEEN_G620.md#continue-from-g620) combine the 0.040/0.100 reward with minimum-held-out-loss selection, fixed beta and no PPG hard KL cap. Old-reward PPG caches are deleted at training startup after durable reviewed adoption, before the first new collection. The first phase uses all 16 new-reward collections g621–636. Model weights, both optimizers, RNG and committed g620 progress are preserved; fresh value targets are not fitted inside DCPG PPO.

The first v11.2 run rebuilds its fixed validation probe with recorded chosen moves and industry metadata. A v11.1 probe cannot be silently imported. The optional `--il-kl-probe-from-run` remains available for a stopped, committed **compatible v11.2** donor using the same upgraded IL/dataset and probe settings. Use it only on a new run's first command; it copies the probe, never PPO state or the donor's large prepared cache. Omit it on continuation.

Fresh configs now include `rehearsal_sampling` (`rotating-train-window/1.0.0`): **16,384 sampled rows per iteration**, **32 shard draws**, **4 GiB maximum temporary train-cache bytes**, and `delete_after_update=true`. A lightweight index reads audited per-game decision counts from the train manifest. Each iteration samples shards proportional to their decision count, then samples rows uniformly with replacement inside those draws. Every train decision retains a uniform marginal probability of selection. This is a rotating, correlated sample, not IID full-corpus sampling for every minibatch; repeats are possible and reported. At 10 million decisions, the default window is at most about 0.16% of the corpus, renewed at the next committed iteration.

Only selected source shards are read, using `audit_workers` (**16 in the current config**) and reusing the CPU reader pool already started during ordinary collection. Each worker admits one full source shard, then prepares neutral BC metadata only for distinct selected rows and writes the sampled rows directly under `.rehearsal-cache`. Repeated draws and source order remain exact; dense features do not cross worker IPC. Existing independently attested feature sidecars are checked and reused when present. Missing sidecars are reconstructed in memory without writing full-shard caches. Features are loaded from the temporary cache for training minibatches.

The 4 GiB byte cap is shared across all workers, with reservations acquired before writes. It covers temporary disk output, not decoded source shards, worker processes, PPO data, GPU tensors or the original dataset. Up to 16 source shards may be resident across readers. If a selected window would exceed the cap, preparation fails, joins active writers and cleans up rather than silently dropping large-action examples; reduce `rows_per_iteration` in a new config if necessary. The existing config's `audit_workers` now controls rehearsal loading too; no new config is needed for a run already using 16.

Full train caches deleted after IL are not required. Each source shard's raw hash, audited membership/counts, labels, encodings and legal selectors are checked before its rows can train. The temporary train cache is removed immediately after PPO, including exceptions/Ctrl+C; a hard-kill remnant is removed on the next locked startup. Dataset files and the fixed validation probe are never cleaned by this mechanism. The window selection is deterministic from the manifest hash, run seed and committed iteration, and consumes no PPO RNG. Resume regenerates the same uncommitted window. Existing signed configs without this block keep their previous loading behavior; regenerate the new config to opt in.

Iteration reports and compact history include source/selected/distinct/used row counts, selected shards, selection hash, preparation time, peak cache bytes and cleanup status in `actor.rehearsal_sampling`. Rehearsal weights/floor, PPO losses and behavior checks are unchanged. This reduces eager loading and disk requirements; its training-quality and whole-run throughput effects need ordinary training logs.

The report also records requested/active workers, pool reuse, summed worker preparation/write/CPU seconds and dense IPC bytes. Iteration timing now separates rehearsal preparation, actual PPO, PPG value fitting, teacher work and post-update diagnostics. Console `value_steps=accepted/attempted` and `value_stop=...` distinguish a skipped phase from a transactional rollback. Fully rejected value phases reuse the pre-phase train metrics and post-PPO probes because the exact weights were restored. KL/held-out-loss guards and their thresholds are unchanged. See the [generation-one review and benchmark](../ai-research/REHEARSAL_PARALLEL_REVIEW.md).

Validation on 18 October: **27 focused checks passed**, covering train-only indexing, weighted sampling, deterministic restart, missing feature caches, encoding parity, identical PPO loss/weights/RNG for identical sampled rows, byte caps, interrupt cleanup, probe import/rejection and the existing IL-KL/IL loader regressions. The isolated [smoke report](../artifacts/rehearsal-window-20261018/smoke/report.json) records two one-game CPU iterations (fresh start, then resume), 310.1 seconds total. Each used a 64-row window from a 345-decision diagnostic train game: 55/57 distinct sampled rows, 14/14 distinct rows actually trained, and 21/25 presentations after forced filtering. Temporary caches were 4.70/4.71 MiB and were deleted after each update; preparation took 17.0/7.6 seconds. The 144-position validation probe was reused with identical semantic identity, and donor/IL files and the previous checkpoint stayed unchanged. These small-fixture timings do not measure full joint7 throughput or playing strength.

Use your actual completed v11.1 path for the upgrade, and the new v11.2 handoff path for config generation and training. The shipped `joint7-latest.json` is a template; `configure` binds the selected IL checkpoint and its rehearsal manifest instead of importing the template's historical lineage.

The `latest-model-selfplay/3.0.0` recipe starts with these base quotas. Fresh configs now include the v4 share gate described below:

| Tables | Quota | Learners | Frozen opponents |
| --- | ---: | ---: | --- |
| Self-play | 50% | 4 | None |
| History | 45% | 2 | 2 independently sampled history policies |
| v4 | 5% | 2 | 2 search-free v4 bots |

Once the latest completed **v4 rating panel** reaches **90% learner win share**, the next collection uses **160 self-play / 154 history / 6 v4 games**, or **50% / 48.125% / 1.875%**. The ten released v4 tables go to **history**. This is the existing one-learner/three-v4 evaluation panel (32 games, eight setups at the current cadence), with official ties. The threshold is inclusive: exactly 90% passes. The reduction is persisted before collection and never automatically reversed. No extra evaluation games run, and deployment still requires review.

### Continue from g560: auxiliary heads, 1% branches and 50% LR threshold

The adoption entry point is **committed g560**. If adoption has not yet been attempted, use the full sequence below. If startup or the first iteration failed after adoption, use the recovery instructions below instead. Pull the Python sources and the matching `training/runtime/local-scoring/ordinary/BrassHost.exe` and `runtime.json` together. The lab installs this additive recorder automatically from the prebuilt package and existing private pinned DLLs; no compiler or fresh IL is needed.

```bash
cd /data/not_backed_up/yxu209/ai-training

python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 560 --apply && \
python scripts/configure-local-scoring-aux.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 560 --mode apply --fresh-ppg-buffer \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-local-scoring-lr50-g560.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-local-scoring-lr50-g560.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 560 --iterations 20
```

This continues **g561–580**. The helper copies the locked committed config, changes only the companion block, branching **0.015 → 0.010**, and LR raise threshold **0.45 → 0.50**, checks the full checkpoint hash and exact source review, and writes a config outside the run. It does not edit weights or delete caches. The recorded **0.020/0.080 insolvency**, VP cap 180, epsilon zero, fixed snapshot weight 1, temperature-2 openings, 16-step PPG checks/patience 1, PPO and all existing gates/baselines remain in place.

With `target_kl=0.01`, an LR raise now requires a three-valid-iteration median **strictly below 0.005**, plus the existing health checks and cooldown. The new rule is used for the next post-PPO decision; it does not force a raise or reset the saved LR multiplier, measurement window, cooldown or warmup. Fresh config generation also selects 50%.

Training startup archives old PPG **metadata**, durably records the adoption, then audits and retires the old owned PPG cache payloads. It preserves core PPO/value Adam, RNG, progress, population, the active KL reference, beta 8 and KL-trip history. Only the PPG buffer/clock and epoch-vote window start fresh at g560 (cap 2, bounds 2–6). The first new phase is **g568**, using g561–568; the next is **g576**. Old g553–560 payloads are not backfilled. Returning to that old buffer after retirement needs an external backup; the adopted continuation remains restart-safe.

After a later interruption, recover any incomplete next iteration with its actual committed generation and repeat **only the train command**, updating `--expected-generation`. Do not repeat `--fresh-ppg-buffer`. To disable the heads after a new checkpoint has committed, run the same configuration helper with `--mode off`, the actual generation and a new output name, omitting `--fresh-ppg-buffer`; use its output with `--resume-reviewed-source`. This retains the companion state and current PPO/PPG clocks. `shadow` trains only detached heads. Existing IL/raw replays remain admissible; source-bound derived replay caches may be revalidated/rebuilt.

If startup fails with a missing `training/runtime/local-scoring/ordinary/BrassHost.exe`, sync the executable along with `runtime.json`. The initial package accidentally inherited the repository's `*.exe` ignore rule; the exact local-scoring executable is now explicitly allowed, while private game DLLs remain excluded. Its SHA-256 must be `c36293458ba9bba82f2acbb5525e335d6cb1187df442b8a202b7c34ff17eb20e`. The existing `-text` Git attribute preserves its exact bytes. This is a packaging-only fix: it does not change the reviewed training source or require recompiling on the lab. Retry **only the final train command above**, using the already generated config. Startup may already have committed the adoption and retired the old PPG cache before discovering the missing executable; its saved receipt safely resumes that transition without another fresh-buffer request.

If the first collected iteration fails in `local_scoring.diagnostics` with **`'NoneType' object has no attribute 'rows'`**, sync the diagnostic-lifetime fix and its source-review registry. The old loop released the critic dataset before the auxiliary report read it. Reports now materialize before release, and the datasets are still freed before checkpoint serialization. The traceback occurs before saving g561: archive the incomplete collection, then retry from g560 with the already generated config. The recovery helper checks the actual committed generation and checkpoint; it refuses a different generation or a partial checkpoint instead of rewinding it. Do not regenerate the config or repeat `--fresh-ppg-buffer`.

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 560 --apply && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-local-scoring-lr50-g560.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 560 --iterations 20
```

The failed iteration's shards are preserved under `.interrupted-iterations/`. The retry recollects with the new source identity; it does not adopt unsaved PPO updates. The g560 full checkpoint, Adam/RNG state, progress, population, adoption origin and PPG schedule remain unchanged. Any uncommitted g561 PPG cache is replaced when the retry stages its newly checked collection; committed caches remain receipt-bound.

If that retry instead stops at **`ValueError: Local scoring adoption cannot repeat`**, sync the adoption-restart fix and source-review registry. The collection archive has already completed, and this error occurs before collection starts. The old code rejected the existing adoption receipt while publishing a source fix against the pre-adoption g560 checkpoint. It now accepts only an exactly identical receipt, preserving its cleanup status and original PPG schedule; altered receipts still fail before writing. Reuse the already generated config and run **only**:

```bash
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-local-scoring-lr50-g560.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 560 --iterations 20
```

Keep the existing source-transition backup, adoption receipt and checkpoint files. Do not rerun `--fresh-ppg-buffer` or manually remove the adoption receipt: the restart validates and reuses them, and the first scheduled PPG phase remains g568.

Console `local=apply@scale rows=retained/available head_steps=PPO+PPG` exposes activation and usage. Full and compact reports include ownership/VP diagnostics, missing/masked labels, separate attempted/retained usage, joint rollback counts, cache bytes and timing. `python scripts/export-training-reports.py` includes the small runtime contract and adoption receipt, without copying PPG arrays.

The following **g540 procedure is archived**; it has already been applied to the current run:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 540 --apply && \
python scripts/joint7-g540-ppg.py --expected-generation 540 && \
python scripts/joint7-vp-cap.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-one-se-g540.json \
  --expected-generation 540 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-vp180-g540.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This continues **g541–560**. The helpers write configs from locked committed state without editing checkpoints; the VP helper adds only `reward.vp_cap=180` and its matching search reward to the prepared g540 recipe. Startup preserves weights, both Adams, RNG, beta, PPO/LR/GAE, population, progress and the original entropy/economic baselines. It explicitly resets PPG's replay/schedule origin to g540, hotfixes its epoch cap to **2**, sets the adaptive floor to **2** (maximum 6), and clears old PPG epoch votes and KL-trip history. After durable reviewed config adoption, startup audits and deletes only owned `.ppg-eight-cache` / `.ppg-value-cache` arrays and partial files; ownership markers stay. Unknown files and links are refused before any deletion. The first eight new collections use fresh **VP-180 and insolvency 0.020/0.080** targets; PPG phases are due at **g548 and g556**. Normal later cache expiry still follows checkpoint commit. Repeat only the train command for later clean continuations; fresh reset is not repeated after the first new checkpoint.

If the g540 recipe is already running at a later committed generation, omit the older g540 helper and `--config` input: `python scripts/joint7-vp-cap.py --expected-generation N --output configs/step5/joint7-v11-2-grouped-lr-critic-vp180-gN.json` derives only the VP cap from the current committed config. Use that output for the reviewed train command, and recover an incomplete collection first using its actual committed generation. The reward change starts a fresh eight-collection PPG window at N, preserves the attained epoch cap, and invalidates only reward-dependent utility calibration and PPG cache/evidence. No fresh IL is required. See [VP-180 behavior and validation](../ai-research/VP_REWARD_CAP_180.md).

PPG checks pooled ordinary learner parent games from all eight cached iterations, normally **144 games**. It keeps the minimum mean game-loss reference separately from the selected state. A candidate within **one paired standard error** of that minimum and within both KL caps can be retained for more distinct critic-row coverage. Ties prefer lower loss, then a later budget position. Neutral checks do not consume patience. A lower minimum rechecks the previous selection; accepting a slightly worse state never raises the reference. Final weights and value Adam come from the same selected checkpoint. `value_phase.selection` reports minimum/selected loss, paired difference/SE, coverage and budget positions; epoch-cap votes continue using the minimum-loss position. See [implementation and validation](../ai-research/PPG_ONE_SE_G540.md).

The following **g503 continuation is archived**. It includes the earlier insolvency 0.010/cap 0.075, 1.5% branching and fixed 1.00 snapshot weight changes, plus PPG, epsilon zero and the decoupled opening objective. The earlier g503 command does not need to be run first. After syncing this revision:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 503 --apply && \
python scripts/joint7-g503-ppg.py --expected-generation 503 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-openings-g503.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This continues **g504–523**, with PPG phases at **g511 and g519** after eight fresh iterations each. The recovery command archives an incomplete next collection after Ctrl+C and verifies committed g503. The config helper leaves run/checkpoint files untouched. Training restores weights, both Adams, RNG, LR/warmup, GAE, population and progress, then applies fixed snapshot weight before the first resumed PPO batch. The prior R² controller and its reference/evidence are archived, with no further recovery or decay. The original entropy/economic baselines, entropy coefficients, promotion origin and probe remain unchanged. PPG starts an empty replay window without backfilling old reward/behavior data; expired caches are removed after commit. The LR evidence window starts fresh for the changed objective/collection signature while its multiplier and warmup are retained. Changing the insolvency reward requires fresh targets and utility calibration, not a model reset. See the [g503 review](../ai-research/SELFPLAY_G503_REVIEW.md) and [opening/PPG follow-up](../ai-research/SELFPLAY_G503_OPENING_PPO.md). Repeat only the train command on subsequent clean continuations.

The following **g470 command and zero-floor controller are archived**; use the g503 command above for the current run:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 470 --apply && \
python scripts/joint7-throughput.py --expected-generation 470 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-throughput-g470.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This starts g471 and targets g490. Recovery archives only an incomplete next collection after Ctrl+C; the generation guard refuses a different committed state. The helper derives the config from locked committed state, verifies the exact reviewed transition, and leaves checkpoints untouched. It retains PPG off, current snapshot weight/reference/evidence, both optimizers, warmup/LR multiplier, RNG, population and progress. The snapshot controller may now reduce weight to zero using its existing evidence; the helper does not forcibly set the current weight to zero.

The original saved entropy baselines, alpha values and windows remain unchanged. The signed config records the four existing anchor promotions at g470, which are not counted retroactively. Future promotions reduce ratios by two percentage points of baseline exactly once, including after Ctrl+C between anchor publication and the next checkpoint. First promotion: 0.93 action / 0.88 other; fifth and later: 0.85 / 0.80. The helper refuses conflicting output files. The completed IL folder stays unchanged. Repeat only the training command on later clean continuations. No full-size GPU speedup is claimed from the bounded CPU checks.

Training startup validates/restores the original checkpoint and commits its reviewed source/config transition before deleting old `.ppg-eight-cache` and `.ppg-value-cache` arrays, receipts and partial writes. Cleanup occurs under the run lock before collection. All targets are checked before any deletion; unknown files, links and invalid ownership are refused. Tiny `owner.json` markers remain, and the released bytes are printed and saved in `ppg-disabled-cache-cleanup.json`. No new PPG cache is written. Ordinary replays, checkpoints, IL/probe caches, population and lineage remain intact. The separate temporary `artifacts/brass-ppo-*` preparation transport is retired after PPO, including normal exceptions/Ctrl+C; it contains no resume state.

The original eight-age recipe already used PPO evidence for GAE, so those windows, attained lambdas and table activation survive this disable. The dormant value Adam (including its saved decayed LR) and PPO optimizer/RNG are preserved. Snapshot R² retains its weight and frozen reference; the existing regime-change rule starts new stage-local observations on the next committed PPO-only update. The 8/8/2 current-iteration whole-game holdout, economic checks and PPO KL safeguards remain active. See [g460 disable, validation and runtime notes](../ai-research/PPG_DISABLED_G460.md).

The following **g418 enablement and auxiliary recipe are archived**. PPG was off by default for the g460 continuation; an explicit `joint7-ppg-eight.py` opt-in remains available.

Archived g418 command:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 418 --apply && \
python scripts/joint7-ppg-eight.py --expected-generation 418 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-eight.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This starts g419 and targets g438. The config helper changes only `value_phase`, verifies the reviewed transition, and never edits a checkpoint. The first auxiliary phase is g426, using all eight newly collected iterations; the next is g434. Schedule origin, cache receipts and value Adam live in the generation checkpoint. Restarting never restarts the eight-iteration clock. The existing IL folder, entropy floors, branching, epsilon, PPO optimizer/warmup/LR controller, rewards, PFSP pool, rating and teacher settings stay unchanged. Repeat only the train command on subsequent normal continuations; recover another incomplete collection using its actual committed generation if necessary. Do not rerun the older entropy hotfix. See [eight-iteration PPG design, validation and runtime estimates](../ai-research/PPG_EIGHT_ITERATION_REVIEW.md).

Each phase uses eight cached ages with **uniform weights of 1**, shuffled epochs of at most 4,096 rows per step, and separate raw-T1 cloning/probe sets. Every eligible critic row appears once per planned epoch; the existing snapshot critic loss weight remains active. Start at a **two-epoch cap**. Raise the cap by one, up to six, when the best checkpoint lies in the final quarter of the planned budget in three of the last four phases. Lower it by one, down to one, when the best lies in the first epoch in three of four. Each cap change starts a new four-phase evidence window. Wait iterations do not count. Policy strata use corrected weights without age decay.

Held-out families never enter auxiliary critic or cloning fitting. Paired acceptance uses ordinary learner games from **all eight iterations (normally 144)**, with equal game weighting, a one-standard-error loss tolerance, **checks every 32 steps** (also at each epoch end), and patience of two checks. The best retained weights and matching value Adam are restored together. Every step must satisfy both overall and age-0 exact KL ≤ 0.002. If unsafe weights improve held-out value loss, restore the safe best weights/Adam, double cloning beta and retry the interval using the same cloning draws. Beta persists and is bounded at **32**; the current run keeps its saved beta 8 and can rise to 16 then 32. An unresolved cap violation or absence of value improvement stops the phase at the safe best. KL-limited phases cast no epoch-cap vote; economic/GAE KL safeguards still apply.

Reset **only the value Adam moments and step counters** when a scheduled auxiliary phase begins. Warm its LR linearly from **3e-7 to 3e-6 over the first 32 planned steps**, then decay linearly to **9e-7** at the planned epoch cap. Retries reuse their planned-step rate. Keep-best rollback restores matching weights, moments, counters and LR within the phase. The PPO optimizer, PPO warmup and LR multiplier are unchanged. Both optimizer states still resume exactly from a committed checkpoint; the next scheduled value phase deliberately performs its configured reset.

Reports retain epoch-cap evidence/decisions, beta retries, rate/reset details, age/table/era losses, policy-group support/KL, schedule, actual retained sample use, and disk/RSS/CUDA memory measurements. GAE/snapshot diagnostics still use only this iteration's ordinary learner holdouts; old cached games are not counted again as new evidence. GAE retains its PPO work evidence between phases. Snapshot R² recognizes the changed training regime without lowering its frozen reference.

The `.ppg-eight-cache` directory contains sealed mmap critic rows and complete legal-action inputs for selected learner states. Retain it with the latest checkpoint when backing up or moving a run. It is training state, so the report exporter intentionally omits its bulk arrays. Missing or corrupt committed arrays fail explicitly. Expiry happens only after state/checkpoint commit; staging can temporarily hold nine filesets before returning to eight. An older full checkpoint may require restoring its matching expired cache backup. Normal replay retention stays at two iterations and cannot reconstruct all eight caches. Actual production memory and GPU auxiliary wall time must be read from the first full phase; the small CPU smoke does not measure those.

Archived command for **g412** using its original PPO-only config:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 412 --apply && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-entropy-groups.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The group-floor/branch hotfix is already adopted at g412; do not rerun it or regenerate the config for this performance fix. Recovery archives an incomplete next collection, if present, without changing the committed checkpoint. Training resumes with g413. The PPO input repair removes repeated hashing of unused action lists from state-only value batches, retaining shape/layout and consumed-selector checks, all policy candidates, strict behavior verification and cached-input content checks. Reports now time `ppo.utility_calibration`, `ppo.critic_prepare`, `ppo.gae`, `ppo.probe_prepare` and `ppo.probe_final` separately. The displayed total PPO time includes utility-head calibration; disabling PPG does not disable that existing calibration. See [g412 diagnosis, benchmark and validation](../ai-research/SELFPLAY_G412_PPO_REVIEW.md).

For an older run that has **not adopted the g410 group-floor/branch hotfix**, the archived command below applies that migration and continues for 20 additional iterations, retaining PPO-only training and the **8/8/2 critic holdout**:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 410 --apply && \
python scripts/joint7-entropy-groups.py --apply --expected-generation 410 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-entropy-groups.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

Run this with training stopped. Recovery preserves any incomplete next collection in `.interrupted-iterations` and is a no-op if there is none. An uncommitted checkpoint/report requires commit recovery and is refused. The hotfix locks the run, verifies the exact reviewed source/config transition and checkpoint/Adam/entropy state, writes the config and atomically records adoption. Only `entropy_control` and `branching` change: fixed 95% action / 90% other floors on the **existing** baseline, plus a 0.005 eligible-decision fork rate. Opening and post-fork temperatures remain 2.0 for two action decisions per seat. Weights, both optimizers/RNG, population, progress, entropy baseline/alphas/windows/rollout age, economic/GAE evidence and KL reference/calibration are preserved. No measurement or recalibration is added. Omit the hotfix's `--apply` for a preview. The generation flags guard against a different committed state; use its actual generation if later. Conflicting output configs are refused. `exploration/group-floors-hotfix-report.json` is included by the report exporter. Subsequent continuations need only the train command. See [g410 review and validation](../ai-research/SELFPLAY_G410_REVIEW.md).

The hotfix starts fresh subsequent collections; cached old collections are never relabeled. On the first new update, the existing LR controller clears only its short decision window for the changed collection regime, retaining its multiplier, cooldown and warmup. Snapshot-R² starts stage-local evidence for that regime while retaining its **frozen reference and attained weight**. The hotfix itself preserves the checkpoint and all saved evidence. Epsilon stays 0.02. The g320 entropy95, g213 snapshot-R² and [qualified-reference adoption](../ai-research/QUALIFIED_POLICY_ANCHOR_REVISION.md) migrations are already complete for this run; do not rerun those helpers. The superseded entropy95 helper refuses to undo the new group floors/branch rate.

The input-cache/streaming-batch performance revision needs no new config or additional hotfix. Continue using the current config/run/IL paths and `--resume-reviewed-source`; it preserves the training recipe and existing source/pending-collection safeguards. Fixed PPO probe inputs use at most 256 MiB on device (also at most one eighth of free CUDA VRAM), with a bounded pinned-host fallback; repeated scans cache at most 128 MiB per Dataset. Content checks, full behavior-probability verification and exact KL after every PPO step remain active. Collection now accepts compatible arrivals between model forwards while preserving deadlines, observer/model identities and one reply per envelope. Reports/compact history add `actor.input_cache` and `utilization.inference_batching`. See the [implementation, bounded CPU measurements and GPU benchmark command](../ai-research/SELFPLAY_INPUT_CACHE_REVIEW.md); CPU microbenchmarks do not establish a production GPU speedup.

Both g270 panels have 32 games/eight setups and 100% learner win share. Their gates therefore latch at adoption, and subsequent rating slots run **128 main games with no diagnostic panel games**. Each panel independently retires once its latest complete committed result reaches 90%; IL uses the combined win share of its two learner seats against two IL seats, and v4 uses one learner against three v4 bots. Retirement is one-way and restart-safe. Triggers survive rating compaction with verified receipts. Missing retired results are reported as `null`/retired, never as zero wins. Main PL references, bootstrap fits, GAE rating evidence, and manual deployment qualification remain unchanged.

When the IL panel retires, IL joins the same PFSP normalized distribution and uniform floor as every other member. Its already accumulated ordinary-game S/C statistics are preserved; panel wins never update PFSP scores. IL loses its reserved seat share but remains protected on disk for fixed IL calibration and main ratings. The v4 share gate uses contract v3 (inclusive 90%); existing v1/v2 contracts keep their original signed meanings. The already latched g270 v4 allocation retains its origin and activation. See [g270 review, validation and continuation](../ai-research/SELFPLAY_G270_REVIEW.md). The archived [g213 review](../ai-research/SNAPSHOT_R2_REVISION.md) describes the completed EV-to-R² migration.

After the share gate passes, learner trajectories remain **960**, while PFSP increases to **154 history tables / 308 opponent seats / 616 pairwise comparisons**. Both opponent seats in every added history game use the same frozen PFSP distribution and independent draws as the existing tables, with IL scored after its panel gate. Actual S/C counts include all extra outcomes; there is no reset or artificial rescaling of stored estimates, and the five-iteration half-life and population rules stay unchanged. History learner-pair rotation now uses the actual history ordinal: 25 or 26 games per pair, balanced across iterations. The six remaining v4 games cover all six learner pairs once and all five admitted bot variants across twelve opponent seats. LR-controller evidence resets when the actual composition changes; its multiplier and optimizer state remain intact. Reports and compact history retain `v4_share_gate`, effective table counts and `quality_pool.exposure`.

Before the gate, 320 games means **160 / 144 / 16 ordinary tables**, or **640 / 288 / 32 learner seat trajectories: 960 total** before forced-action filtering. Branch continuations do not advance these counts. Each mixed table type cycles deterministically through the six learner seat pairs, carrying its position across iterations/resumes. Before qualification history uses each pair exactly 24 times per iteration; 16 v4 tables use each pair two or three times, balancing over successive iterations. Exact equality across six pairs is impossible for 16 tables. Ordinary neural learners and frozen historical seats both use **epsilon 0.02, q v3** in the new config. Temperature is **2.0 for each seat's first two action decisions**, then **1.0**; intervening subdecisions stay at 1.0. Evaluation remains epsilon zero with no opening schedule. There is no v4 curriculum/boost window or later fast-to-searched transition. `v4_search_enabled=false` disables MCTS in all ordinary and scheduled self-play evaluation v4 seats, including held-out profiles. The bots use their existing incumbent-only policy. The one-second search budget is stored but inactive; cancellation and failure watchdogs remain active. Existing signed configs retain their previous behavior until explicitly migrated.

Scheduled **Monitor** and **Broader Evaluation** are disabled (`evaluation_schedule.enabled=false`) at startup/resume and after iteration commits. The independent Plackett–Luce ratings remain enabled every ten iterations; each diagnostic panel retires independently at 90% learner win share. Newly generated configs inherit this setting; the older generated config is left unchanged.

The PFSP historical pool holds **pinned IL + at most 16 scored snapshots before 100,000 committed ordinary games**, then **IL + at most 23 snapshots**. The `population` fields remain a compatibility declaration of the final capacity (2 + 22); `opponent_pool` governs selection and eviction. Resume checkpoints and independently rated snapshots remain protected; the pool cap does not limit every checkpoint file. The archived quality-score contract remains readable for explicit old configs.

`teacher_enabled=false` disables teacher games, search, policy distillation and teacher auxiliary values. The stored teacher settings remain **9/12/12/12 seconds**, five games, eight workers per turn, eight distillation steps and `search_wall_fraction=null`, ready for a separately reviewed future change. Ordinary utility calibration is still required. PPO reward/value weights and the disabled champion gate are unchanged. The entropy base remains 0.005; the new controller can raise each group's coefficient to 0.05. Rehearsal follows the floor schedule below.

## Opponent critic rows and seat descriptors

Only nonforced **current learner** decisions train the actor. The critic uses all learner decisions, including forced ones, plus scored snapshot decisions after its gate opens. IL/gen0 and v4 decision rows never train either. Learner rows from history/v4 tables remain eligible. Current whole-game holdouts select **8 self-play, 8 history and 2 v4 games per 320**, before and after the v4-share gate. They exclude every seat and branch in those game families from all critic fitting; learner decisions can still train the actor. A frozen snapshot with the same weights as the learner is distinguished by its behavior-policy identity.

Each recorded table carries four descriptors, reordered into the observing seat's relative order: five role flags, clipped log-scaled generation lag, and 28 style values. The fingerprint contains eight action-family masses and six build-conditional industry masses for each era. Nonempty groups sum to one; absent eras/builds report zero groups. Learner fingerprints use one incoming raw-temperature-1 pass over the fixed IL validation action rows each iteration. Snapshot fingerprints are cached at admission, IL once, and v4 fingerprints use the probe's recorded choices without additional games. Unknown descriptors have only the unknown role set.

Critic training replaces all four descriptors with unknown for 25% of games, chosen deterministically per game/epoch. Advantages, calibration and held-out metrics always use true descriptors. Descriptors cannot directly alter policy logits. The outcome head consumes its own seat embedding; the utility head consumes their mean. ONNX accepts `[B,4,34]` with an overridable all-unknown default, including when the caller omits it. Search defaults to unknown. Choosing unknown versus learner descriptors for own-copy search still requires a later PL experiment; no choice is claimed from these smoke tests.

The signed `critic_data.rollout.mode=auto` starts with learner-only critic fitting. Actual PPO critic work (accepted PPG work when explicitly enabled), nonnegative learner held-out EV gain over unknown, and existing health/KL checks enable snapshot rows **next iteration**, initially weight 1.0. This descriptor admission gate is unchanged. Subsequent weight decisions use the rolling R² controller below. An enabled route with an IL-only pool, merely available rows, and rejected PPG attempts do not count as snapshot training. Reports distinguish `snapshot_rows_enabled` from actual `snapshot_rows_used` and give separate PPO/value distinct-row counts; these counts may overlap.

**Snapshot weight recovery:** pool target/error moments over at most **20 committed iterations**, requiring **10 qualifying iterations** and whole-game support. Only ordinary learner held-out rows enter `R² = 1 − MSE / Var(target)`; MSE includes prediction bias. Branches and snapshot seats are excluded. An initial reference is calibrated once from ten eligible iterations. Five newly committed consecutive healthy, qualifying iterations with rolling R² **at least the frozen reference minus 0.01** permit **0.00 → 0.50 → 0.75 → 1.00**. Before each increase, freeze the pre-increase rolling reference and clear the next stage's evidence/streak. If that estimate is slightly lower within tolerance, retain the higher existing reference to prevent downward ratcheting. Each stage must collect its own ten qualifying iterations. Under the new `critic-rows/2.0.0` contract, a supported drop below `reference − 0.02` after retained snapshot training reduces weight to **0.00**, clears stage evidence and retains the reference. At zero, exclude snapshot rows from fitting; real learner critic work supplies the ordinary holdout evidence needed to recover. Above zero, retained snapshot training is still required. Archived v1 configs retain their 0.5 floor. Changed training/measurement regimes start new evidence without lowering an established reference.

Economic health, PPO warmup/work/early-stop/KL checks and the PPG KL safeguard still gate increases. An empty PPG phase does not erase real PPO snapshot exposure. Unhealthy valid observations remain in the performance estimate; they cannot create a healthy streak. The active-reference KL budget and recovery latch still gate LR increases only. GAE retains its existing EV/health evidence. No IL sale/pass probability check is reintroduced.

The checkpoint now persists `critic_data.weight_recovery` as `snapshot-weight-r2/2.0.0`: committed window, regime, stage origin/weight, frozen reference and healthy streak. Legacy EV is archived and removed from decisions; the rest of the phase payload, both optimizers and RNG remain intact. Ordinary resume restores evidence exactly. The archived `joint7-snapshot-r2.py` helper supplied the one-time g213 baseline migration, which is already applied to this run; do not repeat it for g270. Current/previous PPG rows, when enabled, use the selected current weight. Reports and compact history retain R², learner MSE, signed prediction bias, target variance, supporting generations/games, reference, weights and decisions. Console lines include snapshot R²/support and the five-commit streak. These overlapping online windows are correlated diagnostics, not independent confidence bounds or proof of improved strength.

## Archived g210 continuation with PPO only

This command records the historical g210 switch to PPO only; its checked-in output matches the g210 recipe. For the current run use the g470 command above and its separate output filename. The archived helper also supports v5 eight-age checkpoints, including their decayed value-Adam LR and existing PPO gate evidence.

```bash
python scripts/joint7-ppo-only.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only.json

python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This resumes the committed generation; do not use `--fresh`. The exact reviewed transition preserves model weights, PPO Adam/LR/warmup, RNG, training/game counters, population, lineage, attained GAE lambdas/table activations, snapshot weight/baseline, and staged exploration. Saved PPG Adam moments, beta and trip history remain in the checkpoint but receive no updates.

New PPO-based critic evidence windows start at the transition's committed generation. Old PPG steps are never relabeled as PPO work. GAE observation/support counters and unfinished snapshot-recovery streaks restart; already attained lambda and snapshot states do not reset. PPG arrays are neither loaded nor rebuilt and no new two-iteration cache is created. Known owned PPG arrays are removed only after the first successful PPO-only commit. Existing pending-collection/source freshness checks still apply.

Reports show `value_steps=0/0 value_stop=disabled`, plus `PPO-critic-window=...`. The `value_phase` report remains as a compatibility container for current post-PPO holdout diagnostics and explicit `critic_training.source=ppo`; its optimizer-step counts remain zero. The nonzero diagnostics time is inference, not PPG training. The report exporter and compact history keep these fields. See [PPO-only validation and continuation](../ai-research/PPO_ONLY_REVISION.md).

## Gated GAE and optional PPG-style value fitting

Fresh configs include the signed `advantage` and `value_phase` blocks. GAE starts at **lambda 1**, preserving the original `return - value` advantages exactly. Gamma remains 1. Chains group by game and learner seat, skip forced decisions, and use differences of the existing utility returns as rewards. Forced-decision insolvency costs are already included in those differences. Critic targets remain Monte Carlo returns and terminal outcomes. The actor's normalization, legal masking, behavior-probability verification and exploration mixture are unchanged.

The lighter automatic gate uses committed iterations only:

With current PPG-off training, the evidence source remains **retained ordinary PPO critic work on every iteration**, as it was during eight-age PPG. Existing GAE windows therefore survive disabling PPG. If eight-age PPG is explicitly enabled later, its steps are reported separately and a PPG KL trip fails health. The older accepted-PPG evidence behavior below applies only to historical v2–v4 recipes.

- Move to **0.98** after at least ten iterations with actual critic steps and self-play learner held-out utility explained variance of at least 0.5 for five consecutive iterations. The fallback is thirty iterations with actual critic steps. The evidence source is PPO when PPG is disabled, or accepted PPG work when explicitly enabled. Both paths require current healthy evidence.
- Move from **0.98 to 0.95** after **twenty consecutive healthy iterations**, with at least **16 retained critic optimizer steps across at least 3 iterations** in the rolling 20-iteration window at this lambda. With PPG disabled these are PPO steps with nonempty critic batches and positive `value_weight`; with PPG explicitly enabled they remain accepted PPG steps. A nonnegative rating slope and no existing PL regression/v4-watch alert are still required. The slope uses the last three rated checkpoints from the latest common PL fit. Missing or declining rating evidence delays this transition. No extra rating games or fork runs are launched.
- Self-play lambda-stage health requires final economic checks, completed PPO warmup, a finite nonempty PPO update without early stopping, PPO categorical KL at most 1.5 times target, finite self-play learner held-out EV, and no applicable value-phase KL trip. At lambda 0.98, **zero critic steps alone never reset the healthy counter**: actual work is required across the window. Economic/PPO failures or missing EV still reset it. PPG guards apply only when PPG runs. The calibrated IL distance budget and recovery latch gate LR increases only. Both IL sale- and pass-probability checks are removed from every shared-health consumer. The first lambda transition retains its numerical requirements, counting actual PPO critic-update iterations when PPG is off.

`staged-gae-gate/2.0.0` persists its own bounded `stage_evidence`, separate from mixed-table EV admission. The exact reviewed continuation preserves existing lambda, counters and streaks, but starts this new evidence window at committed progress without backfill. Even an existing long healthy streak cannot skip the 20 new observations. Old accepted steps expire after 20 iterations. Reports and compact history retain observed/healthy iterations, accepted steps, contributing iterations, thresholds and blockers. These initial support thresholds are an experiment, not evidence of better playing strength.

Changes apply on the **next** iteration. Lambda never automatically drops below 0.95 or rolls back. These thresholds are experimental; a nonnegative point-estimate slope is a progression check, not statistically demonstrated strength. Setting `advantage.gate.mode` to `fixed` in a newly generated config permits an explicit fixed lambda instead. Lambda below 1 requires the combined-return recipe.

History and v4 learner advantages remain at **lambda 1** independently until their mixed-table gate qualifies. `advantage.table_gate` now uses contract `table-gae-ev-gate/2.0.0`: a **20-iteration evidence window**, at least **8 distinct held-out games** for both the mixed table type and the self-play comparison, at least **3 contributing iterations**, and at least **3 healthy iterations with actual critic fitting** within that window. With PPG disabled, fitting is the retained PPO critic work. Self-play comparison statistics come from the same contributing iterations. Merge target/error moments to obtain pooled EV; do not average iteration EV values. The EV tolerance remains `mixed EV >= selfplay EV - 0.05`.

Activation also requires **ten consecutive healthy iterations** (final economy/pass checks, PPO warmup/update/KL checks, no PPG KL trip) and a **fresh paired mixed-table sample in the current iteration**. A missing v4 sample does not erase unexpired evidence or reset the health streak. A fully restored held-out-loss rejection still supplies valid predictions; recent accepted value work is required across the window instead of on every sparse sample. Missing/degenerate pooled variance and insufficient game support cannot qualify. Snapshot rows never supply this evidence. The self-play lambda stage and snapshot-critic adoption rules remain unchanged.

These are row-weighted **online EV measurements from successive retained post-PPO models** (post-value when PPG is explicitly enabled), not a re-evaluation of all old positions using the latest model or independent confidence bounds. The checkpoint stores only game IDs and moments, capped at 20 iterations; evidence pooling adds no tensor cache or model inference. Whole-game holdout exclusions remain unchanged. Qualification takes effect next iteration, persists across restart and follows subsequent self-play lambda stages. Qualification can precede the first lambda below 1. Reports distinguish the qualification `origin` from `effective_stage_origin`; the 30-iteration review age counts actual collections at the current lambda stage, beginning at the later of qualification and that stage's origin. A later lambda stage begins its own review age. Review common PL rating evidence and by-table advantage spread; no automatic rollback or additional evaluations are introduced.

The following two-iteration PPG training/cache details describe **historical recipes through v4**. The archived v5 eight-age recipe is documented with the g418 continuation above; current v7 PPG uses noise-aware 16-step checks, one-SE coverage retention and an epoch floor of two as described in the g540 revision. Historically, the value phase ran after every PPO iteration, using current and previous critic-train rows. `value_phase.buffer_iterations=2` retains that archived behavior when explicitly enabled. Archived v2 holdouts used nominal **5%** and a minimum of two games per type. Current v3 holdouts use **8/8/2 per 320 parent games**, scaling for other collection sizes while leaving a training game in every represented stratum. All seats in held-out families are excluded from PPO critic batches, head-only utility calibration and auxiliary fitting. Their nonforced rows remain eligible for ordinary PPO actor losses. This is a critic holdout, not a completely untouched policy evaluation set. Small collections can have no held-out games; missing EV cannot satisfy an adoption gate.

The previous rows live in `<run>/.ppg-value-cache/` as checksummed NumPy memory-mapped arrays. Cache only the critic inputs: state, one real legal candidate, seat mapping, MC/utility/remaining-cost targets, true descriptors, role/table labels, descriptor generation and game/phase identity. Forced rows are included; hash-held-out games are never cached. Previous behavior probabilities never enter PPO, GAE, head-only ordinary calibration, clone references or policy probes. Cloning and holdout validation continue to use current rows only. The batch mixes current and previous critic rows without duplicating either population within an epoch.

Publication is transactional: write the current cache before saving the checkpoint, bind its receipt to `training.actor_value_phases.buffer`, commit `state.json`, then delete the previous cache. After a successful iteration, only that iteration's cache remains for the next update. Interrupted updates retain their committed input; restart removes uncommitted cache output. Missing/corrupt cache files rebuild shard by shard from the checkpoint's verified ordinary replay receipts and must reproduce the original checksum. Missing or changed authoritative replays cause a clear error, never silent loss of previous data. Retention requires at least **two replay rounds**, including manual cleanup; the current fresh template keeps two. Do not manually delete the newest replay rounds if recovery is needed.

At the current 3,128-state / 327-action dimensions, each cached row is **14,503 bytes** plus the array header. At the plan's illustrative 86k learner + 21k snapshot rows, a 95% train split is about **1.47 GB per cache**, with two arrays coexisting until commit (about **2.95 GB**). Phase 1 has fewer eligible rows. At this size, two iterations give about 50 value minibatches per epoch, up to 150 across three epochs; guard stops can reduce work. Descriptors and small targets are resident tensors; full state/action features remain memory mapped. Missing-cache reconstruction uses bounded learner then snapshot passes to reproduce the staged order. Collection and rating game counts are unchanged. Additional costs are snapshot critic work, one incoming fingerprint probe pass and unknown-descriptor holdout evaluation; production GPU/disk throughput remains unmeasured.

Up to **three value epochs**, batches of **4,096**, fit the existing win/rank/VP losses plus utility MSE. A frozen raw post-PPO policy supplies up to **8,192 nonforced clone rows**, sampled in batches of **512**, with beta initially **1**. Value batches retain one legal candidate for the state-only value path; clone batches retain every legal action. The separate **Adam** runs at value LR **3e-6**, without PPO warmup or the LR multiplier; PPO base LR remains **6e-6**. It does not advance PPO counters or modify PPO moments.

In the archived v4 recipe, every value optimizer step checks exact raw-policy KL on the **same position IDs as the PPO probe**, against the fixed post-PPO raw policy. Its cap is **0.002**, cumulative across the whole value phase; accepting an interval never resets the reference. Those v4 configs use **`acceptance_steps=16`**. At each boundary and shorter epoch tail, accept only non-increasing held-out loss. A loss-rejected interval restores its starting weights **and value Adam** while retaining earlier accepted intervals, then stops. A KL-violating step restores its pre-step weights/Adam; the safe interval prefix still needs the held-out-loss check before retention. Draw RNG is consumed deterministically as before; PPO Adam, warmup and LR-controller state are untouched. The archived v5 recipe above replaced these acceptance and cloning details with eight-age independent probes, 32-step checks, keep-best/patience and bounded beta retries.

When the **current iteration trips** the cap and at least three of the last five iterations tripped, double beta up to **8**. Clean iterations cannot raise beta repeatedly using the same older trips. Resume preserves the saved beta and trip history, including an existing beta of 8. The trip remains recorded even when a safe prefix is retained. Existing health/GAE gates do not count that iteration as a clean pass. Reports distinguish attempted/accepted steps and presentations, full/partial accepted epochs, last attempted KL, retained KL, the cap and `beta_reason`. Each epoch's held-out attempt describes its safe prefix after restoring any violating step. A final accepted fit issues a new utility-calibration receipt; a skipped or fully rejected fit clears the stale receipt for the next ordinary calibration. The bounded rollback adds one model/value-Adam snapshot per attempted step; it does not add another optimizer or retain snapshots across iterations. See [the PPG and v4 sampling fix](../ai-research/PPG_V4_PROFILE_FIX.md).

The order is collection → critic-train-only head calibration → advantages → PPO → PPO probes/LR decision → value phase → any teaching → final IL-KL/probes → checkpoint. LR feedback keeps the post-PPO signal. Final calibrated IL-KL limits and quality-pool admission see the final weights. Generation checkpoints include both optimizers, the GAE gate's progress/streaks, clone beta/trip history and RNG. Existing runs without these blocks retain their recipe; use a new config and run for this change.

Each iteration and compact history retain `actor.gae` (lambda, TD-error spread, shadows at 1/0.98/0.95/0.9, sign/correlation and era/phase means), `actor.critic_split`, `actor.final_categorical_probe`, and `value_phase` (held-out/train EV, utility MSE, value loss, ten-bin win calibration, accepted/attempted steps, cloning/probe KL, stop reason and seconds). The `value_phase.buffer` report adds cache generations/bytes, rebuild and cleanup results; value totals distinguish current/previous available rows, distinct accepted rows and presentations. Console lines show used/next lambda, accepted value steps and current+previous rows actually used. The report exporter preserves these fields automatically. See [implementation and validation](../ai-research/GAE_VALUE_PHASE_REPORT.md).

## PFSP history opponents

The PPG refinement uses **3e-6 value LR** and stratifies whole-game critic
holdouts: the archived v2 contract selected **8 self-play / 7 history / 2 v4 games** at 320 games and 50/45/5. The current v3 quota contract selects **8 / 8 / 2** at both 160/144/16 and 160/154/6 ordinary tables; branches inherit their parent's holdout. Its original v2-to-v3 continuation resets EV evidence windows while preserving attained GAE stages, snapshot weight and optimizer/training state. The generation 213 v4-allocation correction already uses this holdout version and does not reset those windows again.
All their seats stay out of every critic fit/cache, while actor training keeps
eligible actions. Old explicit configs retain their hash split and base value
LR until the reviewed migration. That migration preserves both Adam states,
active gates and training counters; it starts fresh selection-dependent EV
evidence/streaks and excludes old cache rows for one iteration. Cache receipts
bind the new whole-game membership for exact rebuilds. See the
[PPG refinement report](../ai-research/PPG_REFINEMENT_REPORT.md).

The generation-110 review adds a one-time, checkpoint-bound IL-KL adjustment and
an explicit reward transition to `event_cost=0.0075`, keeping `max_cost=0.075`.
Fresh configs use this event cost; existing run configs are unchanged unless
explicitly migrated. For this interrupted run, use the
[g110 repair and PPG-refinement continuation command](../ai-research/PPG_REFINEMENT_REPORT.md#continue-from-generation-110).
It resumes the committed g110 checkpoint and completed rating, keeps both Adam
states and RNG, and excludes previous PPG rows with old reward targets for one
iteration. Do not restart the run or regenerate its IL model.

The supplied **PFSP-vs-quality simulation is saved in the repository**. Rerun `python scripts/pfsp_sim.py --games 320 --history-share 0.45 --output results/pfsp-history45.json` when history share, games per iteration or pool capacity changes; choose a new result filename and adjust the share to the proposed setting. The command is an isolated synthetic experiment and does not change training. See [simulation files, options, limitations and reference cross-check](../ai-research/simulations/PFSP_SIMULATION.md).

Ordinary v4 tables cycle through all five admitted training profiles: balanced, cotton, goods-beer, rail-tempo and supplier. Selection advances by the **v4-table ordinal**, derived from committed ordinary progress, rather than the overall game index that aliases the 5% quota. With 320 games at 50/45/5, the 32 v4 opponent seats give each profile 6–7 appearances. Each table uses two distinct variants; learner-pair rotation remains unchanged. `pottery-flex` stays in the evaluation pool. `sampling.legacy_profiles` records actual completed counts by profile and physical seat and is retained in compact history. No extra mutable rollout counter or config regeneration is required.

Each history seat is sampled independently. **Before the IL panel gate**, with `k` total pool models **including IL**, its pinned share is `1/(k+1)`, or 100% while IL is alone. Thus k=2/4/17/24 gives 33.3%/20%/5.56%/4%. `pinned_seat_share.max=null` leaves this uncapped; an explicitly configured `max=0.2` caps the early share, except the IL-only fallback. The remaining seats draw from `p_i = 0.8 * (1-x_i)^2 / sum((1-x)^2) + 0.2/N`, where N=k-1. If all weights are zero, use uniform sampling. Both opponent seats may select the same model. Before the gate, scored probability p is conditional; its absolute seat probability is `(1-il_share)*p`. **After the IL panel reaches 90%**, N=k includes IL, all members use the same PFSP formula, and p is the absolute seat probability. There is no reserved IL seat probability; the 20% uniform floor applies to IL too. Pool membership limits, ordinary-outcome decay and eviction protection of the fixed anchor remain unchanged.

After collection, every learner/snapshot seat pair in a completed history table contributes its official-tiebreak outcome (win 1, residual tie 0.5, loss 0). A repeated snapshot fills two seats and receives four comparisons. For every pool member, including IL, update Python float64 sums once per iteration: `S = d*S + outcomes`, `C = d*C + comparisons`, `x=S/C`, with `d=0.5**(1/5)`. Unsampled members decay without changing their estimated rate. New members start at S=5, C=10. IL's rate is diagnostic until its panel gate passes, then determines its PFSP weight like every other member. Self-play, v4 and rating results never enter this estimator; there is no importance weighting. The signed contract is `pfsp-pool/1.0.0`; q, eta and a fixed numeric IL share are rejected under it. Optional `weighting.function=variance,power=1` implements x*(1-x), but the default is hard/power=2.

Every five committed iterations, admit the current model only after five recorded healthy iterations and when every `KL(member || candidate)` is at least 0.02 on the fixed IL validation probe. Health checks are unchanged, so first admission can occur later than generation 15. When full, protect IL, the two newest snapshots and snapshots younger than 20 iterations; require C>=15 for other eviction candidates. Shortlist members within 0.05 of the highest eligible outrank rate, then evict the one with the smallest KL to its nearest other pool member. With no eligible victim, skip admission. Pool size never exceeds its current cap.

`state.json` stores S/C, admission ages, model/checkpoint identities, cached-probe receipts, alarm streaks, health window, membership events, committed games and the next planning distribution. IL share and p are frozen before collection and preserved in the plan/seat receipts. Reports and compact history retain per-member raw and decayed outranks, C, w, used p, seats/comparisons, effective members, largest/recent share, weighted mean rate, IL statistics and admission/eviction reasons. `planning` describes the completed collection; `next_distribution` and `summary` describe the next collection after admission/eviction. Console includes IL share, effective member count and alarm count. A member below x=0.4 with C>=20 for three consecutive iterations emits a review-only regression warning. It does not change shared health, sampling rules, rehearsal, ratings or gates. Active probe caches, pool checkpoints and rated checkpoints remain protected. See the [PFSP implementation and validation report](../ai-research/PFSP_REVISION_REPORT.md).

## Held-out Plackett–Luce ratings

Every ten committed generations, collect **32 fixed validation setups × 4 candidate-seat rotations = 128 main games**, with exploration 0, neutral focus and no search for either neural or v4 policies. Before qualification, half the setups have an IL reference seat and half v4, with two nearby rated opponents. Once **both panel win shares exceed 80%**, a persistent gate changes the main schedule to **4 candidate + IL + 2 checkpoints, 4 candidate + v4 + 2 checkpoints, and 24 candidate + 3 checkpoints**. Retired 90% panel receipts also qualify. This leaves eight direct reference setups, distributed every four setup positions with the pattern rotating between slots. The game budget stays 128; checkpoint opponents average **2.75 per game** when at least three distinct identities are available.

Use the previous completed fit and latest rated learner to estimate candidate strength. The eight-opponent shortlist explicitly includes the previous rated checkpoint and active KL reference, then fills by rating distance. Deterministic least-used selection balances appearances and excludes duplicate model identities within each table. Limited populations use an explicit reported reference/repetition fallback. Previously evicted training-pool models remain available as rated opponents. Historical outcomes, IL zero, priors, seat correction and paired bootstrap fitting are unchanged.

Until its own 90% gate passes, each diagnostic panel additionally runs **eight setups × four rotations**: one learner versus three v4 bots, and two learners versus two IL opponents. The all-learner health panel comes from recorded ordinary self-play. The total is **192, 160 or 128 held-out games per 3,200 training games**, depending on whether two, one or zero panels remain active (6%, 5% or 4% extra game volume), plus fitting and I/O. Both panels retire at g270 adoption for the current run. Main PL games still include IL/v4 reference opponents. Scheduled Monitor and Broader Evaluation add no games while disabled. Held-out games never train PPO. Rating v4 is the pinned `balanced-v3-masked-economy` profile in the configured reference bank, with search disabled, matching the search-free setting throughout the new self-play recipe.

Keep the runtime under `artifacts/legacy-ai/search/<rated-runtime-id>/` for the lifetime of the rating series. Rating jobs verify its recorded manifest, executable, dependency and profile hashes and use that exact binary even after current source changes. New ordinary collections still require current source freshness. Startup checks the reference before collecting more ordinary games; a missing or changed reference fails explicitly instead of substituting a new opponent under its old identity.

The MAP model fixes IL theta at zero, uses agent priors N(0, 2²), and four zero-sum seat effects with priors N(0, 0.5²). Official residual ties use a seeded shuffle. Every slot refits all main games with L-BFGS and runs 200 setup-cluster bootstrap refits, preserving pairing across slots and rotations. Reports give 173.7×theta points, 95% intervals, seat-neutral outrank probabilities versus IL/v4, and paired difference intervals. Only a difference interval excluding zero supports a directional strength claim within this model. The nominal intervals require adequate real-game evidence; neither ±40 points nor actual 95% coverage is guaranteed.

Committed compact games, slot specifications, fit reports, panel results, bootstrap arrays and `progress.svg` live under `ratings/slots/generation-NNNNNN/`. Rating work finishes before further PPO on resume; an interrupted slot reuses verified games. Reports flag a new upper interval more than 50 points below the best point estimate and a strict decline in v4 win share over three slots. The decline remains a manual-review warning, with no automatic increase or reversal of the v4 share gate. **Deployment qualification requires manual panel review; no model is automatically deployed or published.** The highest lower-bound candidate is reported as a review aid.

Automatic rating slots and offline refits use `src/brass_ai/recovery/rating_plot.py`, shared with `scripts/render-pl-progress.py`. Each newly completed scheduled rating slot automatically writes the updated `progress.svg`; no plotting configuration change is needed. **Full training history has at most 30 dots**, including the latest point: longer histories use the previous adjacent-checkpoint averaging in non-overlapping groups, with first/latest exact and every checkpoint contributing once. The **last 100 generations** remain exact in the recent panel, with an explicitly separate y-axis and a lightly shaded corresponding window in the main plot. Both panels retain the original pointwise 95% setup-bootstrap intervals; the main ribbon is **not a confidence interval for averaged scores**. Smaller summary cards leave more room for the plots and show the latest score and interval, the change versus the previous rated checkpoint using its **paired difference interval from the same fit**, and clearly separated latest-slot and accumulated-fit game counts. Missing paired evidence is labelled unavailable, never inferred from separate rating intervals. IL and v4 remain reference lines; there is no active KL-reference marker. Existing reports, rating fits and training state remain unchanged. To redraw an older plot from its existing fit, run `python scripts/render-pl-progress.py --report <slot-directory>/report.json`; add `--output <new.svg>` to preserve the old chart, `--preview <output.png>` for a PNG preview, or `--max-points 24` for a smaller full-history cap (allowed range 3–30). Redrawing does not refit ratings or collect games.

The recent-panel endpoint score uses a measured right-hand gutter and a short leader to its exact checkpoint, keeping text clear of the curve and uncertainty band. The gutter expands for longer formatted scores. Lighter interval shading, subtle vertical guides and higher-contrast secondary text improve readability in both SVG and PNG output.

With automatic retention enabled, full rating folders cover the **most recent 100 committed training iterations**: normally ten rating folders at the every-ten-iteration cadence. At generation 110, generation 10 is retired and generations 20–110 remain. Before cleanup, older compact outcomes and panel summaries are consolidated into one checksummed `ratings/history-<digest>.json`, referenced by `state.json` under `ratings.archive`. Cumulative PL fits, paired bootstrap refits, alerts and resume still use all outcomes. Older full reports, curves, bootstrap arrays and sampled replays are cleaned through the existing retention/quarantine mechanism; explicit pins are honored. Required rated model checkpoints remain protected. The latest fit is always kept, including with a custom rating interval over 100. This change applies to rating output folders; existing exported `results/` bundles are untouched.

The versioned `table_plan`, `opponent_pool` and `rating` blocks join the run signature. SciPy and Matplotlib are in `requirements-step5.txt`. To refit existing evidence into a new review directory without collecting games:

```bash
python scripts/strength-ratings.py \
  --run artifacts/step5/joint7-v11-1-grouped-lr-quality \
  --output artifacts/joint7-v11-1-grouped-lr-quality-rating-review
```

See the [implementation and bounded validation report](../ai-research/SELFPLAY_QUALITY_RATING_REPORT.md).

## Rehearsal floor

The rehearsal coefficient is `max(floor, 0.1 * max(0, 1 - committed_games / 10000))`.
The floor uses committed iteration progress, so restarting preserves its stage:

| Training iterations (one-based) | Floor |
| --- | ---: |
| 1–60 | 0.03 |
| 61–120 | 0.02 |
| 121–180 | 0.01 |
| 181 onward | 0 |

`rehearsal_floor` is a versioned block with `initial=0.03`, `hold_iterations=60`, `decrement=0.01`, and **`interval_iterations=60`**. Drops occur after 60/120/180 committed iterations, without restarting the clock on continuation. Rehearsal remains loaded past 10,000 games while its floor is positive. Each report records the committed counters, floor, decayed coefficient and effective coefficient. An explicitly zero `rehearsal_weight` still disables rehearsal. Explicit old interval-30 configs retain their old schedule until the reviewed config change below; configs without the floor block retain their original schedule.

## PPO learning-rate controller and calibrated anchor-KL limits

Fresh configs now use **`ppo-categorical-kl-lr/1.2.0`** plus the strict **`auto-il-kl-limit/1.2.0`** block, with no sale-health threshold or IL action-probability guard. Both are part of the run signature. Completed v11.1 IL remains usable through the explicit v11.2 upgrade; its original directory and selected weights stay intact. Existing 1.0/1.1 health configs remain readable for reviewed continuation; retired `sells_ratio` is ignored. The fixed probe is reused unchanged. To enable the new PPG intervals and GAE window, use the [g185 continuation commands](../ai-research/PPG_WINDOW_REVISION.md#continuation); no new IL or g110 hotfix is needed.

The per-update LR signal remains exact **KL(old behavior || post-PPO behavior)** on the ordinary proportional probe, capped at 2,048 nonforced rows. The active PPO trainer now also uses this exact categorical KL after **each optimizer step**, stopping further steps when it exceeds `update.target_kl=0.01`. The old probabilities and probe positions stay fixed throughout all PPO epochs; both distributions use the same legal masks, temperature and 0.08 exploration mixture. The sampled-action estimate on **all nonforced rows** remains a secondary guard at each epoch end. Nonfinite probe or full-dataset KL aborts the update before any further optimizer steps or checkpoint publication. No KL penalty is added to the loss, and stopping does not undo the last step.

The exact and sampled probe values share one forward pass per microbatch. Reports and compact history retain `actor.kl_guard` (contract `ppo-categorical-kl-guard/1.0.0`, checks, maximum exact KL, stop reason and step), plus each epoch's `probe_categorical_kl`. Existing `probe_kl` and `post_update_kl` retain their sampled-action meaning. The console prints `KLstop=categorical_probe`, `sampled_full_dataset` or `none`; the LR report records the same stop reason and actual completed/planned steps.

Raise by 1.25 after a full three-valid-iteration median **below 0.005 (50% of target_kl)** with no stops, no cooldown and passing health/anchor checks. Older recorded configs remain at 45% until the explicit g560 adoption/hotfix above is selected. Cut by 1.5 immediately on an early stop before half the planned steps, two stops in three valid iterations, or current exact KL above 0.015. Cuts start a three-valid-iteration cooldown; multiplier bounds remain 0.5–3. Warmup, empty/nonfinite update handling, shadow/off modes and PPO-only multiplier application remain unchanged. **Protective LR cuts never depend on anchor/health approval.** Full and compact LR reports include the effective raise ratio and absolute threshold.

The categorical guard remains enabled and its preceding guard-only source approvals are preserved. Those approvals do **not** authorize the new quality-pool/rating recipe on an existing PPO run. For this revision, generate a new recipe-3 config and use the earlier `--fresh` command without a source-resume flag. Completed v11.1 IL remains usable; the PPO guard does not alter supervised IL updates. Signed older recipes are not silently upgraded, and pending-collection safeguards remain active.

Validation (2026-10-18): 59 focused regressions passed, covering exact/masked behavior KL, stop boundaries, nonfinite values, sample accounting, LR feedback, reviewed resume and IL extension/cache reuse. A separate CPU smoke with fresh current v11.1 weights and the utility head completed two six-step PPO updates; checkpoint restoration produced bitwise matching next-update weights and Adam state. On a 2,048-position synthetic probe (2–32 candidates, microbatch 128, two CPU threads, three timed repeats), median guard time was 0.990 s for sampled-only versus 1.008 s for exact plus sampled. This measures probe overhead only; CUDA, full collection throughput and playing strength were not measured. The reproducible smoke is `tests/Recovery/smoke_categorical_kl_guard.py` and requires a new `--output` directory under `artifacts/`.

The policy-distance gate uses a separate **fixed validation probe**, built at generation 0 from the completed IL dataset. It selects up to 2,048 distinct nonforced positions with seed 710193, balanced across era × decision-phase strata using deterministic hash priority; exhausted strata redistribute their allocation. The manifest, selected identities, source hashes and cached IL probabilities live under `il-kl/`. No training RNG is used. The generation-zero report verifies KL(IL || IL) = 0. Validation positions never enter PPO or rehearsal updates; rehearsal still uses the training split.

After the saved **IL-panel ≥90%** qualification (combined first-place share of its two learner seats), `policy_anchor` initially chooses the highest-rated available checkpoint with a positive same-fit paired-bootstrap 95% difference lower bound over IL. It stays frozen. Later promotions consider only newer rated checkpoints and require their paired difference lower bound against the active reference to exceed zero in the same current fit and bootstrap draws. Scores or separate confidence intervals from different fits are never compared. Original IL remains the PL zero point and the same PFSP opponent.

Each promotion measures **KL(reference || current learner)** on the unchanged 2,048 positions, resets only reference-dependent calibration to **0.15 overall / 0.30 per group**, clears its previous KL history, and recomputes the budget latch immediately. An already over-budget learner continues ordinary PPO but cannot raise LR. The ten-iteration calibration cadence, growth cap and recovery rules below are retained. Economic baseline and five-iteration rolling counts do not change; neither do the LR multiplier/warmup nor GAE/snapshot evidence. The checkpoint-bound pending receipt makes a restart before the next PPO commit idempotent. Retention protects the active model, probability cache and paired promotion evidence. Original IL KL remains a diagnostic computed alongside the active-reference KL with the same learner forward pass.

Both policies use raw temperature-1 probabilities with **no exploration mixture**. For a finite, nonnegative KL that remains zero for identical policies, both raw distributions receive the same **1e-6 floor and renormalization** for the KL calculation. **IL pass-probability, IL sale-probability and aggregate sales-per-seat checks are removed** from shared health, LR/limit decisions, pool admission and GAE/critic gates. No per-position action-probability pass/fail is computed or reported. Archived behavior flags cannot affect new decisions. Ordinary sales counts are reporting only; ordinary aggregate passes-per-seat remains an economic check alongside VP, zero-VP and insolvency. Old signed probe manifests, bytes, hashes and selection remain unchanged and reusable; other config/source mismatches still fail validation.

The four fixed-probe groups are **Canal action, Canal other, Rail action, Rail other**. This replaces the early/late-round anchor groups for controller 1.2; explicit 1.1 configs retain their previous behavior. Each group needs at least 64 positions, and LR increases require the full 2,048-position probe. Tiny diagnostic datasets can produce smaller probes with explicit counts; they cannot qualify an increase.

Health uses **all-learner self-play tables only**, with seat-weighted five-iteration rolling averages and a frozen baseline from the first ten iterations that start after LR warmup finishes:

| Metric | Healthy when |
| --- | --- |
| VP per seat | At least 0.85 × baseline |
| Passes with a legal non-pass alternative per seat/game | At most 2 × baseline + 0.5 |
| Zero-VP seat fraction | At most max(0.05, 2 × baseline) |
| Fraction of seats with an insolvency event | At most baseline + 0.10 |

Zero-VP seats, insolvency events and distinct affected seats are recorded separately for every table type. Historical/v4 tables cannot influence the calibrated self-play baseline. Missing evidence and failed pass checks block increases.

The overall limit starts at **0.15 nats**. Every group limit is always **2 × the current overall limit**, initially **0.30**. Before the baseline is ready the limits stay at their initial values. Afterward:

- Any economic-health or probe-evidence failure immediately tightens the overall limit to `min(limit, max(0.05, last_healthy_KL))`, updates both limits and restarts the ten-iteration clock.
- After ten further healthy iterations, use their largest overall KL, `K`: `min(1.0, 1.25 * limit, max(limit, 1.25 * K))`. Small distances never lower the limit; growth is capped at 25% per update and 1.0 overall / 2.0 per group.
- A KL budget violation latches the LR-increase gate. Recovery requires healthy evidence, mean KL strictly below 0.8 × its current limit, and every group strictly below 0.8 × its corresponding limit. Mean KL above 2 × the current limit raises a diagnostic alarm.

If warmup finishes after generation 3, the baseline completes at generation 13 and the first healthy-window update is generation 23. Actual timing follows committed progress, not that estimate. The limits only gate LR increases; they do not add an anchor loss or automatic rollback.

The calibrated limit, last healthy KL/generation, update clock, baseline, rolling/healthy windows, latch and probe receipt are stored in `state.json` **and the corresponding generation checkpoint**. Resume requires them to match and verifies the probe/data hashes. The PPO multiplier and its independent three-row window/cooldown remain in Adam's `brass_ppo_lr_controller` parameter group.

Full/compact reports retain the fixed-probe measurements, four group means/counts, economic health metrics, before/after overall/group limits and hold/raise/tighten reason under `il_kl_limit`. They also retain `rehearsal_schedule`, health counts by table type and the existing LR step/median/decision evidence. Console output uses `AnchorKL[gN]`, `OrigILKL`, overall/group limits and `LRraise` (permission from anchor/economic checks; other LR/warmup conditions still apply). Full and compact reports include `policy_anchor`, promotion evidence, original IL KL and `rating_coverage`. The exporter includes the active reference cache manifest, fit/spec and transition reports, as well as the original probe manifest; dense probability/bootstrap arrays are excluded.

Probe preparation/validation happens at startup; normal iterations evaluate only the cached probe. Preparation scans the audited validation partition using disk-backed storage. Resume checks source hashes. These costs and the ongoing rehearsal floor are separate from the per-iteration forward-pass cost; the plan's approximate one-second timing is not a guarantee on every device. The limits and rehearsal schedule are experimental, with no demonstrated playing-strength improvement.

## Continue the current run with staged exploration

Run the helper against the **actual committed training directory**. It reads the saved recipe and writes a continuation config, preserving unrelated settings. It includes the previously requested 16-step PPG acceptance intervals and rolling GAE health window if not yet adopted. It does not edit checkpoints or reset training.

```bash
python scripts/joint7-exploration.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic

python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-exploration.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

The archived exploration-adoption command below retains its signed epsilon 0.03 schedule. Use the current g410 group-floor continuation command above for epsilon 0.02, fixed group floors and temperature-2 openings without restarting attained exploration stages. Under the original migration, the first resumed iteration adopts q v3/epsilon 0.03 for all ordinary neural seats. The second enables the two opening temperatures; the third switches the actor entropy bonus to the raw temperature-1 policy. Ten prospective post-warmup probe measurements establish its four entropy baselines, then healthy iterations may adjust coefficients. Existing economic baselines and automatic IL-KL limits are retained. Branching starts only after 20 completed iterations since adoption and a ready entropy baseline. If starting at committed g185, these collection changes appear in g186, g187, g188 and no earlier than g206 respectively. A 20-iteration command ends at g205; the next continuation can activate branching.

The checkpoint records the origin, baseline, coefficients and windows. Restarting keeps the same stage, weights, PPO/value Adam states, RNG, counters, population and lineage. The LR controller clears its short decision window at exploration regime changes, retaining its multiplier/cooldown; random branch-count fluctuations do not reset it. Pending collections from the old behavior fail closed; recover them with the original source before migration. Once adopted, repeat the train command for more iterations; do not use `--fresh` on this directory. `--expected-generation` is optional. The helper refuses a conflicting existing config rather than overwriting it; use `--output configs/step5/<new-name>.json` if your committed recipe has since changed.

Full and compact reports retain rollout age/settings, ordinary exploration shares, forced-only picks, raw/normalized entropy and coefficients by group, opening diversity, separate branch economy, actual retained branch actor/critic usage, and ordinary/branch held-out EV. **Temperature 1.0 versus 0.7 comparison is disabled by default** (`entropy_control.deployment_temperature_check.enabled=false`) and in the g270 continuation. No deferred/backlogged comparisons run. Existing recorded comparison reports remain readable; old signed configs retain their prior meaning until explicitly upgraded. The g320 revision separately changes ordinary opening temperatures and entropy floors; evaluation policies retain their existing settings.

See [exploration implementation and validation](../ai-research/EXPLORATION_REVISION.md) for the exact scope and limits of the smoke results.

### Continue from g210 with queued branches

Keep the existing exploration config and IL directory. The g210 review changes collection scheduling only; it preserves the saved adoption origin (g185), armed entropy baseline, current lambda, optimizer state and ordinary-game progress. Run the same continuation command above with `--resume-reviewed-source --iterations 20`; do not rerun adoption or use `--fresh`.

Forks are queued immediately and use the next free ordinary lane, with longer ready continuations first. Branch-producing parents start early. The worker cap remains 24 processes × 2 lanes. Families publish only after parent and branch finish, and interrupted incomplete families are recollected with their original deterministic seeds. Full and compact reports retain `exploration_metrics.branching.scheduling`: branch worker-seconds divided by actual worker lanes, queue waits, overlap with the parent, work tail and publication tail. Compare these with collection wall time and **iteration time excluding scheduled evaluations**. The planned +6–8% is a runtime target, not a guaranteed cap.

See [g210 evidence and validation](../ai-research/SELFPLAY_G210_REVIEW.md).

## Archived PFSP adoption from the older quality-score sampler

To adopt PFSP and the slower rehearsal floor in the existing run, switch at a **committed generation divisible by five** (85, 90, etc.). Generate a new config filename from the actual existing config; this helper changes only the recognized default opponent-pool block and rehearsal interval, and never reads or edits model/IL/run files:

```bash
python scripts/build-shared-search.py --root .

python scripts/configure-pfsp.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic.json \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-pfsp.json

python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic-pfsp.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

The `--resume-reviewed-source` flag admits exact reviewed transitions, including health/EV fixes, staged snapshot-weight recovery and this explicitly generated PFSP/rehearsal config. Keep the same IL folder and self-play directory; no IL upgrade, retraining or `--fresh` is needed. Existing sampler members retain their checkpoint/probe/fingerprint identities, admission ages and health window. Each gets C=10 and S=10 times its logged five-iteration outrank rate, or S=5 if no comparisons were logged. An atomic source-transition receipt binds the original checkpoint digest to the new pool, including a restart before the next PPO commit. No q value is reinterpreted as an outrank rate. The g85 export has k=4, so IL's share is initially still 20%; another admission makes it 1/6.

If the run is between admission boundaries, first continue with its **old config** until the next multiple of five; this source retains the old signed sampling/schedule for that purpose. The PFSP switch rejects a non-boundary, an over-cap legacy pool, pending old collections or unknown config changes instead of resetting progress or dropping members. An old table-gate block can still undergo its previously reviewed upgrade. The fixed IL validation probe is reused. Once switched, repeat only the train command with the PFSP config; the config helper refuses overwrite. Omit `--resume-reviewed-source` when source and effective config are unchanged.

Optionally add `--expected-generation 85` only while generation 85 is the intended entry; otherwise omit it or use the actual committed generation. Continuation preserves weights, both Adam optimizers, RNG, active lambdas and enabled table origins, snapshot gate/weight, descriptors/fingerprints, clone coefficient/history, warmup, LR controller, counters, population and lineage. The first snapshot-recovery upgrade adds only an empty recovery streak; later checkpoints resume it exactly. An existing v2 EV window and its streaks are preserved exactly. Only the first upgrade from the old v1 table gate starts an empty window at committed progress and clears its superseded mixed-table streaks; old scalar reports cannot backfill moments or game support. Calibrated IL limits/latch and economic baselines are preserved. See the [generation-85 review](../ai-research/SELFPLAY_G85_REVIEW.md), [health/EV revision](../ai-research/SELFPLAY_HEALTH_EV_REVISION.md) and [generation-55 review](../ai-research/SELFPLAY_G55_REVIEW.md).

Full reports and compact history retain `gate_diagnostics`: pass failures, health and LR blockers, IL-only history fallback, accepted/attempted value steps, beta reason, held-out learner rows and whole-game counts, and mixed-table EV window support/reasons. Its `mixed_ev_status` continues checking the existing pooled tolerance after qualification. An active table with supported evidence below that tolerance emits `mixed_ev_below_tolerance:<kind>` for review; there is no automatic rollback. Console `mixedEV=<kind>:<pooled>vs<paired-selfplay>[<status>]` distinguishes `ok`, `below_tolerance` and `insufficient_support`, alongside `health`, `pass_fail` and game/iteration support. A single-game EV is still reported but does not override the pooled evidence. Removed sale-probability fields are absent from new reports; archived reports remain unchanged. Missing evidence is unavailable, not zero EV.

V11.2 retains v11.1's pre-GELU local action context, fixed-weight action projection and selector-multiply export. Descriptor conditioning affects only critic heads. Existing signed runs/checkpoints stay untouched. The rates remain IL `6e-5` / self-play `6e-6`; config generation does not repeat the earlier width-related rate reduction.

## Export self-play reports

After an iteration finishes, collect the analysis bundle with:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic
```

The explicit `--run` selects the new GAE/value run; the script's older default run name is unchanged. The destination is a new `results/joint7-v11-2-grouped-lr-critic-gNNNN-<UTC timestamp>/` folder. It includes the latest two committed full iteration reports, available compact history and evaluation reports, the saved config, and the originating IL reports. Descriptor/role/EV gate diagnostics, GAE/value diagnostics, exact categorical KL, grouped anchor evidence, LR decisions, actual learner/sample usage, PFSP estimates/plans/alarms, membership and disabled-teacher status are preserved. Recent rating specs/reports, compact outcome JSONL, panels and SVG curves are included together with the active cumulative rating archive; dense probe/bootstrap tensors and full replay shards are excluded. Use your actual `--run` if it differs. The original completed IL stays at `artifacts/step5/features4-v11-1-batch7-il`; the new handoff is `artifacts/step5/features4-v11-2-batch7-il`. See [export options](EXPORT_TRAINING_REPORTS.md).

## Small local checks

With the repository Python dependencies available:

```bash
PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery:tests/Reports \
  python -B -m unittest test_critic_descriptors test_critic_export test_gae_value_phase \
    test_value_buffer test_pfsp_pool test_quality_pool test_auto_il_kl test_rehearsal_window \
    test_export_training_reports -v

PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery \
  python -B tests/Recovery/smoke_critic_descriptors.py \
    --output artifacts/critic-descriptors-local-smoke
```

The compute smoke exercises mixed learner/snapshot rows and an exact two-optimizer restart. The optional `--native --imitation-run <upgraded-IL>` uses two one-game trainer iterations in a new directory and checks cache reconstruction/cleanup. Both are bounded diagnostics, not evidence of playing-strength improvement. See the [revision report](../ai-research/OPPONENT_CRITIC_DESCRIPTORS_REPORT.md) for measured results and scope.
