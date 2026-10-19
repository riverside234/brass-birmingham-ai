# Grouped MLP v11.2: completed IL upgrade and fresh self-play

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

## Fresh self-play from completed v11.1 IL

Install dependencies, upgrade the completed IL into a new directory, and generate a config bound to that upgraded candidate. No IL optimization runs in the upgrade. The original IL directory and Features-4 dataset are not renamed or overwritten. The helper refuses an existing output directory/config; choose another unused suffix consistently if these names already exist.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python -m pip install -r requirements-step5.txt

python scripts/feature-engineering.py upgrade-imitation \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --output artifacts/step5/features4-v11-2-batch7-il

python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --output configs/step5/joint7-v11-2-grouped-lr-critic.json

python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --fresh \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --iterations 1
```

The generated config includes the automatic critic/snapshot and per-table GAE gates, two-iteration disk-backed PPG, bounded rehearsal, 50/45/5 tables, the 24-member quality pool, PL ratings and auto IL-KL/LR controls. Teaching, all v4 search, Monitor and Broader Evaluation stay disabled. PPO starts with a fresh Adam and warmup; the utility extension follows the existing calibration path. The upgrade retains the parent IL checkpoint, dataset, source review and lineage in `upgrade.json`, and verifies the deterministic extension at handoff.

The first v11.2 run rebuilds its fixed validation probe with recorded chosen moves and industry metadata. A v11.1 probe cannot be silently imported. The optional `--il-kl-probe-from-run` remains available for a stopped, committed **compatible v11.2** donor using the same upgraded IL/dataset and probe settings. Use it only on a new run's first command; it copies the probe, never PPO state or the donor's large prepared cache. Omit it on continuation.

Fresh configs now include `rehearsal_sampling` (`rotating-train-window/1.0.0`): **16,384 sampled rows per iteration**, **32 shard draws**, **4 GiB maximum temporary train-cache bytes**, and `delete_after_update=true`. A lightweight index reads audited per-game decision counts from the train manifest. Each iteration samples shards proportional to their decision count, then samples rows uniformly with replacement inside those draws. Every train decision retains a uniform marginal probability of selection. This is a rotating, correlated sample, not IID full-corpus sampling for every minibatch; repeats are possible and reported. At 10 million decisions, the default window is at most about 0.16% of the corpus, renewed at the next committed iteration.

Only selected source shards are read, using `audit_workers` (**16 in the current config**) and reusing the CPU reader pool already started during ordinary collection. Each worker admits one full source shard, then prepares neutral BC metadata only for distinct selected rows and writes the sampled rows directly under `.rehearsal-cache`. Repeated draws and source order remain exact; dense features do not cross worker IPC. Existing independently attested feature sidecars are checked and reused when present. Missing sidecars are reconstructed in memory without writing full-shard caches. Features are loaded from the temporary cache for training minibatches.

The 4 GiB byte cap is shared across all workers, with reservations acquired before writes. It covers temporary disk output, not decoded source shards, worker processes, PPO data, GPU tensors or the original dataset. Up to 16 source shards may be resident across readers. If a selected window would exceed the cap, preparation fails, joins active writers and cleans up rather than silently dropping large-action examples; reduce `rows_per_iteration` in a new config if necessary. The existing config's `audit_workers` now controls rehearsal loading too; no new config is needed for a run already using 16.

Full train caches deleted after IL are not required. Each source shard's raw hash, audited membership/counts, labels, encodings and legal selectors are checked before its rows can train. The temporary train cache is removed immediately after PPO, including exceptions/Ctrl+C; a hard-kill remnant is removed on the next locked startup. Dataset files and the fixed validation probe are never cleaned by this mechanism. The window selection is deterministic from the manifest hash, run seed and committed iteration, and consumes no PPO RNG. Resume regenerates the same uncommitted window. Existing signed configs without this block keep their previous loading behavior; regenerate the new config to opt in.

Iteration reports and compact history include source/selected/distinct/used row counts, selected shards, selection hash, preparation time, peak cache bytes and cleanup status in `actor.rehearsal_sampling`. Rehearsal weights/floor, PPO losses and behavior checks are unchanged. This reduces eager loading and disk requirements; its training-quality and whole-run throughput effects need ordinary training logs.

The report also records requested/active workers, pool reuse, summed worker preparation/write/CPU seconds and dense IPC bytes. Iteration timing now separates rehearsal preparation, actual PPO, PPG value fitting, teacher work and post-update diagnostics. Console `value_steps=accepted/attempted` and `value_stop=...` distinguish a skipped phase from a transactional rollback. Fully rejected value phases reuse the pre-phase train metrics and post-PPO probes because the exact weights were restored. KL/held-out-loss guards and their thresholds are unchanged. See the [generation-one review and benchmark](../ai-research/REHEARSAL_PARALLEL_REVIEW.md).

Validation on 18 October: **27 focused checks passed**, covering train-only indexing, weighted sampling, deterministic restart, missing feature caches, encoding parity, identical PPO loss/weights/RNG for identical sampled rows, byte caps, interrupt cleanup, probe import/rejection and the existing IL-KL/IL loader regressions. The isolated [smoke report](../artifacts/rehearsal-window-20261018/smoke/report.json) records two one-game CPU iterations (fresh start, then resume), 310.1 seconds total. Each used a 64-row window from a 345-decision diagnostic train game: 55/57 distinct sampled rows, 14/14 distinct rows actually trained, and 21/25 presentations after forced filtering. Temporary caches were 4.70/4.71 MiB and were deleted after each update; preparation took 17.0/7.6 seconds. The 144-position validation probe was reused with identical semantic identity, and donor/IL files and the previous checkpoint stayed unchanged. These small-fixture timings do not measure full joint7 throughput or playing strength.

Use your actual completed v11.1 path for the upgrade, and the new v11.2 handoff path for config generation and training. The shipped `joint7-latest.json` is a template; `configure` binds the selected IL checkpoint and its rehearsal manifest instead of importing the template's historical lineage.

The new `latest-model-selfplay/3.0.0` recipe uses the same quotas from game zero onward:

| Tables | Quota | Learners | Frozen opponents |
| --- | ---: | ---: | --- |
| Self-play | 50% | 4 | None |
| History | 45% | 2 | 2 independently sampled history policies |
| v4 | 5% | 2 | 2 search-free v4 bots |

At 320 games this is **160 / 144 / 16 tables**, or **640 / 288 / 32 learner seat trajectories: 960 total** before forced-action filtering. Each mixed table type cycles deterministically through the six learner seat pairs, carrying its position across iterations/resumes. History uses each pair exactly 24 times per iteration; 16 v4 tables use each pair two or three times, balancing over successive iterations. Exact equality across six pairs is impossible for 16 tables. Normal table rotation and contextual family exploration remain intact. The learner uses exploration **0.08**, temperature **1.0**; frozen historical policies retain **0.05**. There is no v4 curriculum/boost window or later fast-to-searched transition. `v4_search_enabled=false` disables MCTS in all ordinary and scheduled self-play evaluation v4 seats, including held-out profiles. The bots use their existing incumbent-only policy. The one-second search budget is stored but inactive; cancellation and failure watchdogs remain active. This signed, strictly boolean setting defaults to false in newly generated configs; old configs without the field keep their historical behavior and cannot silently reuse search-free collection caches.

Scheduled **Monitor** and **Broader Evaluation** are disabled (`evaluation_schedule.enabled=false`) at startup/resume and after iteration commits. The independent Plackett–Luce ratings and their diagnostic panels remain enabled every ten iterations. Newly generated configs inherit this setting; the older generated config is left unchanged.

The historical pool has at most **24 members: pinned IL + 23 scored snapshots**. The `population` fields remain a compatibility capacity declaration (2 + 22); selection and eviction in recipe 3 are governed by `opponent_pool`, not the previous recent/reservoir sampler. Resume checkpoints and independently rated snapshots remain protected; 24 is the training-pool limit, not a cap on every checkpoint file.

`teacher_enabled=false` disables teacher games, search, policy distillation and teacher auxiliary values. The stored teacher settings remain **9/12/12/12 seconds**, five games, eight workers per turn, eight distillation steps and `search_wall_fraction=null`, ready for a separately reviewed future change. Ordinary utility calibration is still required. PPO reward, entropy/value weights, evaluation and the disabled champion gate are unchanged. Rehearsal follows the floor schedule below.

## Opponent critic rows and seat descriptors

Only nonforced **current learner** decisions train the actor. The critic uses all learner decisions, including forced ones, plus scored snapshot decisions after its gate opens. IL/gen0 and v4 decision rows never train either. Learner rows from history/v4 tables remain eligible. A 5% game-id hash holdout excludes every seat in those games from all critic fitting; their learner decisions can still train the actor. A frozen snapshot with the same weights as the learner is distinguished by its behavior-policy identity.

Each recorded table carries four descriptors, reordered into the observing seat's relative order: five role flags, clipped log-scaled generation lag, and 28 style values. The fingerprint contains eight action-family masses and six build-conditional industry masses for each era. Nonempty groups sum to one; absent eras/builds report zero groups. Learner fingerprints use one incoming raw-temperature-1 pass over the fixed IL validation action rows each iteration. Snapshot fingerprints are cached at admission, IL once, and v4 fingerprints use the probe's recorded choices without additional games. Unknown descriptors have only the unknown role set.

Critic training replaces all four descriptors with unknown for 25% of games, chosen deterministically per game/epoch. Advantages, calibration and held-out metrics always use true descriptors. Descriptors cannot directly alter policy logits. The outcome head consumes its own seat embedding; the utility head consumes their mean. ONNX accepts `[B,4,34]` with an overridable all-unknown default, including when the caller omits it. Search defaults to unknown. Choosing unknown versus learner descriptors for own-copy search still requires a later PL experiment; no choice is claimed from these smoke tests.

The signed `critic_data.rollout.mode=auto` starts with learner-only critic fitting. Accepted value fitting, nonnegative learner held-out EV gain over unknown, and existing health/KL checks enable snapshot rows **next iteration**, initially weight 1.0. That iteration's learner EV becomes the Phase-1 baseline. A later drop of more than 0.02 reduces snapshot weight to 0.5 only if that iteration actually retained snapshot training in PPO or PPG. An enabled route with an IL-only pool, merely available rows, and rejected PPG attempts do not count. Reports distinguish `snapshot_rows_enabled` from actual `snapshot_rows_used` and give separate PPO/value distinct-row counts; these counts may overlap.

**Snapshot weight recovery:** five consecutive healthy iterations with actual retained snapshot training and learner held-out EV **at least the fixed Phase-1 baseline** raise the next iteration's weight **0.5 → 0.75**. Five further qualifying iterations at 0.75 raise it to **1.0**, the maximum. Each increase starts a fresh streak. Missing/nonfinite EV, no retained snapshot training, EV below the baseline, or failed economy/pass/PPO/PPG health resets the streak. A rejected PPG fit can still qualify when PPO retained snapshot work and no PPG KL trip occurred. The IL-distance budget and recovery latch continue to gate LR increases only. No sale-probability check is reintroduced. The existing reduction below `baseline - 0.02` remains immediate after actual exposure, including at weight 0.75; it resets the streak. The gap between the reduction and recovery thresholds limits oscillation. Forced snapshot mode without a recorded Phase-1 baseline cannot automatically recover.

Checkpoint contract `actor-value-phases/5.0.0` persists `critic_data.weight_recovery` (`snapshot-weight-recovery/1.0.0`): origin, consecutive healthy count and last weight-change generation. The exact legacy checkpoint migration preserves the saved weight, baseline, activation and fingerprint history and starts an empty recovery streak at committed progress; it never backfills from old reports. Repeated startup before the next commit repeats the same migration. Subsequent resumes retain the streak exactly. Both current PPO rows and current/previous PPG snapshot rows use the selected weight; learner rows retain weight 1. Reports and compact history include `value_phase.critic_rollout.weight_recovery`, weights used/next, decision and reason. Console lines add `snapshot_recovery=<next streak>/5:<decision>`. See [recovery validation](../ai-research/SNAPSHOT_WEIGHT_RECOVERY.md). `learner_only` remains available for an explicitly configured diagnostic run. These gates are experimental diagnostics, not evidence of improved playing strength.

Reports and compact history retain loss/EV by table and row role, true-versus-unknown descriptor gain, learner EV versus the Phase-1 baseline, fingerprint drift, actual critic row usage and gate decisions. The sequential snapshot EV comparison is a diagnostic, not a controlled causal estimate.

## Gated GAE and PPG-style value fitting

Fresh configs include the signed `advantage` and `value_phase` blocks. GAE starts at **lambda 1**, preserving the original `return - value` advantages exactly. Gamma remains 1. Chains group by game and learner seat, skip forced decisions, and use differences of the existing utility returns as rewards. Forced-decision insolvency costs are already included in those differences. Critic targets remain Monte Carlo returns and terminal outcomes. The actor's normalization, legal masking, behavior-probability verification and exploration mixture are unchanged.

The lighter automatic gate uses committed iterations only:

- Move to **0.98** after at least ten iterations with accepted value-phase steps and self-play learner held-out utility explained variance of at least 0.5 for five consecutive iterations. The fallback is thirty iterations with accepted value steps. Both paths require current healthy evidence.
- Move from **0.98 to 0.95** after **twenty consecutive healthy iterations**, with a nonnegative rating slope and no existing PL regression/v4-watch alert. The slope uses the last three rated checkpoints from the latest common PL fit. Missing or declining rating evidence delays this second transition. No extra rating games or fork runs are launched.
- Self-play lambda-stage health requires final IL pass/economic checks, completed PPO warmup, a finite nonempty PPO update without early stopping, accepted value steps with finite held-out EV, and no value-phase KL trip. The calibrated IL distance budget and recovery latch gate LR increases only; they do not independently block GAE or snapshot-critic adoption. The IL sale-probability check has been removed entirely. An unhealthy iteration resets the twenty-iteration streak. A held-out-loss stop after earlier accepted epochs can still be healthy: the rejected epoch has been restored.

Changes apply on the **next** iteration. Lambda never automatically drops below 0.95 or rolls back. These thresholds are experimental; a nonnegative point-estimate slope is a progression check, not statistically demonstrated strength. Setting `advantage.gate.mode` to `fixed` in a newly generated config permits an explicit fixed lambda instead. Lambda below 1 requires the combined-return recipe.

History and v4 learner advantages remain at **lambda 1** independently until their mixed-table gate qualifies. `advantage.table_gate` now uses contract `table-gae-ev-gate/2.0.0`: a **20-iteration evidence window**, at least **8 distinct held-out games** for both the mixed table type and the self-play comparison, at least **3 contributing iterations**, and at least **3 healthy iterations with accepted value fitting** within that window. Self-play comparison statistics come from the same contributing iterations. Merge target/error moments to obtain pooled EV; do not average iteration EV values. The EV tolerance remains `mixed EV >= selfplay EV - 0.05`.

Activation also requires **ten consecutive healthy iterations** (final economy/pass checks, PPO warmup/update/KL checks, no PPG KL trip) and a **fresh paired mixed-table sample in the current iteration**. A missing v4 sample does not erase unexpired evidence or reset the health streak. A fully restored held-out-loss rejection still supplies valid predictions; recent accepted value work is required across the window instead of on every sparse sample. Missing/degenerate pooled variance and insufficient game support cannot qualify. Snapshot rows never supply this evidence. The self-play lambda stage and snapshot-critic adoption rules remain unchanged.

These are row-weighted **online EV measurements from successive retained post-value models**, not a re-evaluation of all old positions using the latest model or independent confidence bounds. The checkpoint stores only game IDs and moments, capped at 20 iterations; no extra tensor cache or model inference is needed. Existing whole-game holdout exclusions and the two-iteration PPG disk-cache lifecycle are unchanged. Qualification takes effect next iteration, persists across restart and follows subsequent self-play lambda stages. Qualification can precede the first lambda below 1. Reports distinguish the qualification `origin` from `effective_stage_origin`; the 30-iteration review age counts actual collections at the current lambda stage, beginning at the later of qualification and that stage's origin. A later lambda stage begins its own review age. Review common PL rating evidence and by-table advantage spread; no automatic rollback or additional evaluations are introduced.

The value phase runs after PPO, using the current iteration plus the previous committed iteration's critic-train rows. The first iteration has only current rows. Set `value_phase.buffer_iterations=2` (default, `ppg-value-phase/2.0.0`); an explicit value of 1 keeps current-only fitting. A stable game-id hash assigns approximately **5% of complete games** to the critic holdout; all seats from those games are excluded from PPO critic batches, head-only utility calibration and value-phase fitting. Their nonforced rows remain eligible for actor and policy-cloning losses. This is a critic holdout, not a completely untouched policy evaluation set. Small collections can have no held-out games; the value phase skips with an explicit reason rather than changing the split.

The previous rows live in `<run>/.ppg-value-cache/` as checksummed NumPy memory-mapped arrays. Cache only the critic inputs: state, one real legal candidate, seat mapping, MC/utility/remaining-cost targets, true descriptors, role/table labels, descriptor generation and game/phase identity. Forced rows are included; hash-held-out games are never cached. Previous behavior probabilities never enter PPO, GAE, head-only ordinary calibration, clone references or policy probes. Cloning and holdout validation continue to use current rows only. The batch mixes current and previous critic rows without duplicating either population within an epoch.

Publication is transactional: write the current cache before saving the checkpoint, bind its receipt to `training.actor_value_phases.buffer`, commit `state.json`, then delete the previous cache. After a successful iteration, only that iteration's cache remains for the next update. Interrupted updates retain their committed input; restart removes uncommitted cache output. Missing/corrupt cache files rebuild shard by shard from the checkpoint's verified ordinary replay receipts and must reproduce the original checksum. Missing or changed authoritative replays cause a clear error, never silent loss of previous data. Retention requires at least **two replay rounds**, including manual cleanup; the current fresh template keeps two. Do not manually delete the newest replay rounds if recovery is needed.

At the current 3,128-state / 327-action dimensions, each cached row is **14,503 bytes** plus the array header. At the plan's illustrative 86k learner + 21k snapshot rows, a 95% train split is about **1.47 GB per cache**, with two arrays coexisting until commit (about **2.95 GB**). Phase 1 has fewer eligible rows. At this size, two iterations give about 50 value minibatches per epoch, up to 150 across three epochs; guard stops can reduce work. Descriptors and small targets are resident tensors; full state/action features remain memory mapped. Missing-cache reconstruction uses bounded learner then snapshot passes to reproduce the staged order. Collection and rating game counts are unchanged. Additional costs are snapshot critic work, one incoming fingerprint probe pass and unknown-descriptor holdout evaluation; production GPU/disk throughput remains unmeasured.

Up to **three value epochs**, batches of **4,096**, fit the existing win/rank/VP losses plus utility MSE. A frozen raw post-PPO policy supplies up to **8,192 nonforced clone rows**, sampled in batches of **512**, with beta initially **1**. Value batches retain one legal candidate for the state-only value path; clone batches retain every legal action. The separate **Adam** runs at base PPO LR **6e-6**, without PPO warmup or the LR multiplier. It does not advance PPO counters or modify PPO moments.

After every value optimizer step, exact raw-policy KL is checked on the **same position IDs as the PPO probe**, against the fixed post-PPO raw policy. The cap remains **0.002**, cumulative across the whole value phase. A violating step restores its preceding weights **and value Adam state**. Earlier safe steps in that epoch can be accepted only if their held-out value loss does not increase; then the phase stops. A first-step violation or increased held-out loss restores the entire attempted epoch, while earlier accepted epochs remain. This avoids discarding a useful safe prefix merely because the next step crossed the cap. Draw RNG is consumed deterministically as before; PPO Adam, warmup and LR-controller state are untouched.

When the **current iteration trips** the cap and at least three of the last five iterations tripped, double beta up to **8**. Clean iterations cannot raise beta repeatedly using the same older trips. Resume preserves the saved beta and trip history, including an existing beta of 8. The trip remains recorded even when a safe prefix is retained. Existing health/GAE gates do not count that iteration as a clean pass. Reports distinguish attempted/accepted steps and presentations, full/partial accepted epochs, last attempted KL, retained KL, the cap and `beta_reason`. Each epoch's held-out attempt describes its safe prefix after restoring any violating step. A final accepted fit issues a new utility-calibration receipt; a skipped or fully rejected fit clears the stale receipt for the next ordinary calibration. The bounded rollback adds one model/value-Adam snapshot per attempted step; it does not add another optimizer or retain snapshots across iterations. See [the PPG and v4 sampling fix](../ai-research/PPG_V4_PROFILE_FIX.md).

The order is collection → critic-train-only head calibration → advantages → PPO → PPO probes/LR decision → value phase → any teaching → final IL-KL/probes → checkpoint. LR feedback keeps the post-PPO signal. Final calibrated IL-KL limits and quality-pool admission see the final weights. Generation checkpoints include both optimizers, the GAE gate's progress/streaks, clone beta/trip history and RNG. Existing runs without these blocks retain their recipe; use a new config and run for this change.

Each iteration and compact history retain `actor.gae` (lambda, TD-error spread, shadows at 1/0.98/0.95/0.9, sign/correlation and era/phase means), `actor.critic_split`, `actor.final_categorical_probe`, and `value_phase` (held-out/train EV, utility MSE, value loss, ten-bin win calibration, accepted/attempted steps, cloning/probe KL, stop reason and seconds). The `value_phase.buffer` report adds cache generations/bytes, rebuild and cleanup results; value totals distinguish current/previous available rows, distinct accepted rows and presentations. Console lines show used/next lambda, accepted value steps and current+previous rows actually used. The report exporter preserves these fields automatically. See [implementation and validation](../ai-research/GAE_VALUE_PHASE_REPORT.md).

## Quality-score history opponents

Ordinary v4 tables cycle through all five admitted training profiles: balanced, cotton, goods-beer, rail-tempo and supplier. Selection advances by the **v4-table ordinal**, derived from committed ordinary progress, rather than the overall game index that aliases the 5% quota. With 320 games at 50/45/5, the 32 v4 opponent seats give each profile 6–7 appearances. Each table uses two distinct variants; learner-pair rotation remains unchanged. `pottery-flex` stays in the evaluation pool. `sampling.legacy_profiles` records actual completed counts by profile and physical seat and is retained in compact history. No extra mutable rollout counter or config regeneration is required.

Each history seat is sampled independently: 20% pinned IL, otherwise `p_i = 0.8 * softmax(q)_i + 0.2 / N` over the scored members. Until the first admission, every history seat uses IL. Both opponent seats may select the same model. The quality probability is conditional on selecting the scored pool; its per-seat probability is `0.8 * p_i`.

After collection, every learner/opponent seat pair contributes its official-tiebreak outcome (win 1, tie 0.5, loss 0). A scored member receives `q_i -= 0.01 * outcome / (N * p_i)`, using the probabilities frozen when the collection was planned. Recenter by subtracting the largest q. IL outcomes are reported but never change a score. No rating result enters this sampler.

Every five committed iterations, admit the current model only after five recorded healthy iterations and when every `KL(member || candidate)` is at least 0.02 on the fixed IL validation probe. Health uses the existing baseline/behavior/evidence checks below; its window is not backfilled. Consequently the first admission can be later than generation 15. New q is the current maximum. When full, evict the lowest-q eligible member, breaking exact q ties by smallest KL to another member. IL, the two newest members, and members younger than 20 iterations are protected. If all are protected, skip admission.

`state.json` stores members, scores, planning probabilities, cached-probe receipts, health window and membership events. Each generation checkpoint binds this state by digest; resume checks it. Reports and compact history retain q/p, seats, comparisons, raw and five-iteration outrank rates, and admission/eviction reasons. Active probability caches, pool checkpoints and all rated checkpoints survive retention cleanup.

## Held-out Plackett–Luce ratings

Every ten committed generations, collect **32 fixed validation setups × 4 candidate-seat rotations = 128 main games**, with exploration 0, neutral focus and no search for either neural or v4 policies. Half the setups have an IL reference seat, half v4. The other seats use the eight nearest rated checkpoints, estimating the new candidate from the latest rated checkpoint. Startup explicitly falls back to IL + v4; with one rated checkpoint it fills both extra seats. Previously evicted training-pool models remain available as rated opponents.

Every slot additionally runs **eight setups × four rotations for each diagnostic panel**: one learner versus three v4 bots, and two learners versus two IL opponents. The all-learner health panel comes from recorded ordinary self-play. This adds **64 games**, so the default total is **192 held-out games per 3,200 training games (6% extra game volume)**, plus fitting and I/O. Scheduled Monitor and Broader Evaluation add no games while disabled. Held-out games never train PPO. Rating v4 is the pinned `balanced-v3-masked-economy` profile in the configured reference bank, with search disabled, matching the search-free setting throughout the new self-play recipe.

Keep the runtime under `artifacts/legacy-ai/search/<rated-runtime-id>/` for the lifetime of the rating series. Rating jobs verify its recorded manifest, executable, dependency and profile hashes and use that exact binary even after current source changes. New ordinary collections still require current source freshness. Startup checks the reference before collecting more ordinary games; a missing or changed reference fails explicitly instead of substituting a new opponent under its old identity.

The MAP model fixes IL theta at zero, uses agent priors N(0, 2²), and four zero-sum seat effects with priors N(0, 0.5²). Official residual ties use a seeded shuffle. Every slot refits all main games with L-BFGS and runs 200 setup-cluster bootstrap refits, preserving pairing across slots and rotations. Reports give 173.7×theta points, 95% intervals, seat-neutral outrank probabilities versus IL/v4, and paired difference intervals. Only a difference interval excluding zero supports a directional strength claim within this model. The nominal intervals require adequate real-game evidence; neither ±40 points nor actual 95% coverage is guaranteed.

Committed compact games, slot specifications, fit reports, panel results, bootstrap arrays and `progress.svg` live under `ratings/slots/generation-NNNNNN/`. Rating work finishes before further PPO on resume; an interrupted slot reuses verified games. Reports flag a new upper interval more than 50 points below the best point estimate and a strict decline in v4 win share over three slots. **The mixture stays 50/45/5. Deployment qualification requires manual panel review; no model is automatically deployed or published.** The highest lower-bound candidate is reported as a review aid.

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
| 61–90 | 0.02 |
| 91–120 | 0.01 |
| 121 onward | 0 |

`rehearsal_floor` is a versioned block with `initial=0.03`, `hold_iterations=60`, `decrement=0.01`, and `interval_iterations=30`. Rehearsal remains loaded past 10,000 games while its floor is positive. Each report records the committed counters, floor, decayed coefficient and effective coefficient. An explicitly zero `rehearsal_weight` still disables rehearsal. Configs without the floor block retain their original schedule.

## PPO learning-rate controller and calibrated IL-KL limits

Fresh configs now use **`ppo-categorical-kl-lr/1.2.0`** plus the strict **`auto-il-kl-limit/1.0.0`** block. Both are part of the run signature. Completed v11.1 IL remains usable through the explicit v11.2 upgrade; its original directory and selected weights stay intact. Generate a new bound config before starting. A previously generated config does not silently acquire these settings; the helper requires a new output filename. Existing committed runs require a separately reviewed source/config transition.

The per-update LR signal remains exact **KL(old behavior || post-PPO behavior)** on the ordinary proportional probe, capped at 2,048 nonforced rows. The active PPO trainer now also uses this exact categorical KL after **each optimizer step**, stopping further steps when it exceeds `update.target_kl=0.01`. The old probabilities and probe positions stay fixed throughout all PPO epochs; both distributions use the same legal masks, temperature and 0.08 exploration mixture. The sampled-action estimate on **all nonforced rows** remains a secondary guard at each epoch end. Nonfinite probe or full-dataset KL aborts the update before any further optimizer steps or checkpoint publication. No KL penalty is added to the loss, and stopping does not undo the last step.

The exact and sampled probe values share one forward pass per microbatch. Reports and compact history retain `actor.kl_guard` (contract `ppo-categorical-kl-guard/1.0.0`, checks, maximum exact KL, stop reason and step), plus each epoch's `probe_categorical_kl`. Existing `probe_kl` and `post_update_kl` retain their sampled-action meaning. The console prints `KLstop=categorical_probe`, `sampled_full_dataset` or `none`; the LR report records the same stop reason and actual completed/planned steps.

Raise by 1.25 after a full three-valid-iteration median below 0.0045 with no stops, no cooldown and passing health/anchor checks. Cut by 1.5 immediately on an early stop before half the planned steps, two stops in three valid iterations, or current exact KL above 0.015. Cuts start a three-valid-iteration cooldown; multiplier bounds remain 0.5–3. Warmup, empty/nonfinite update handling, shadow/off modes and PPO-only multiplier application remain unchanged. **Protective LR cuts never depend on anchor/health approval.**

The categorical guard remains enabled and its preceding guard-only source approvals are preserved. Those approvals do **not** authorize the new quality-pool/rating recipe on an existing PPO run. For this revision, generate a new recipe-3 config and use the earlier `--fresh` command without a source-resume flag. Completed v11.1 IL remains usable; the PPO guard does not alter supervised IL updates. Signed older recipes are not silently upgraded, and pending-collection safeguards remain active.

Validation (2026-10-18): 59 focused regressions passed, covering exact/masked behavior KL, stop boundaries, nonfinite values, sample accounting, LR feedback, reviewed resume and IL extension/cache reuse. A separate CPU smoke with fresh current v11.1 weights and the utility head completed two six-step PPO updates; checkpoint restoration produced bitwise matching next-update weights and Adam state. On a 2,048-position synthetic probe (2–32 candidates, microbatch 128, two CPU threads, three timed repeats), median guard time was 0.990 s for sampled-only versus 1.008 s for exact plus sampled. This measures probe overhead only; CUDA, full collection throughput and playing strength were not measured. The reproducible smoke is `tests/Recovery/smoke_categorical_kl_guard.py` and requires a new `--output` directory under `artifacts/`.

The IL-distance gate now uses a separate **fixed validation probe**, built at generation 0 from the completed IL dataset. It selects up to 2,048 distinct nonforced positions with seed 710193, balanced across era × decision-phase strata using deterministic hash priority; exhausted strata redistribute their allocation. The manifest, selected identities, source hashes and cached IL probabilities live under `il-kl/`. No training RNG is used. The generation-zero report verifies KL(IL || IL) = 0. Validation positions never enter PPO or rehearsal updates; rehearsal still uses the training split.

Both policies use raw temperature-1 probabilities with **no exploration mixture**. For a finite, nonnegative KL that remains zero for identical policies, both raw distributions receive the same **1e-6 floor and renormalization** for the KL calculation. The remaining raw pass-probability check requires, where a non-pass move exists and IL pass mass is below 0.2, learner pass mass no greater than `2 * IL + 0.05`. Every eligible pass position must pass. **IL sale-probability checks are completely removed** from measurement, shared health, LR/limit decisions, pool admission and GAE/critic gates. Ordinary sales counts remain economic diagnostics; their existing aggregate economy threshold is unchanged.

The four fixed-probe groups are **Canal action, Canal other, Rail action, Rail other**. This replaces the early/late-round anchor groups for controller 1.2; explicit 1.1 configs retain their previous behavior. Each group needs at least 64 positions, and LR increases require the full 2,048-position probe. Tiny diagnostic datasets can produce smaller probes with explicit counts; they cannot qualify an increase.

Health uses **all-learner self-play tables only**, with seat-weighted five-iteration rolling averages and a frozen baseline from the first ten iterations that start after LR warmup finishes:

| Metric | Healthy when |
| --- | --- |
| VP per seat | At least 0.85 × baseline |
| Selected sell decisions per seat/game, including continuations | At least 0.70 × baseline |
| Passes with a legal non-pass alternative per seat/game | At most 2 × baseline + 0.5 |
| Zero-VP seat fraction | At most max(0.05, 2 × baseline) |
| Fraction of seats with an insolvency event | At most baseline + 0.10 |

Zero-VP seats, insolvency events and distinct affected seats are recorded separately for every table type. Historical/v4 tables cannot influence the calibrated self-play baseline. Missing evidence and failed pass checks block increases.

The overall limit starts at **0.15 nats**. Every group limit is always **2 × the current overall limit**, initially **0.30**. Before the baseline is ready the limits stay at their initial values. Afterward:

- Any health/behavior/evidence failure immediately tightens the overall limit to `min(limit, max(0.05, last_healthy_KL))`, updates both limits and restarts the ten-iteration clock.
- After ten further healthy iterations, use their largest overall KL, `K`: `min(1.0, 1.25 * limit, max(limit, 1.25 * K))`. Small distances never lower the limit; growth is capped at 25% per update and 1.0 overall / 2.0 per group.
- A KL budget violation latches the LR-increase gate. Recovery requires healthy evidence, mean KL strictly below 0.8 × its current limit, and every group strictly below 0.8 × its corresponding limit. Mean KL above 2 × the current limit raises a diagnostic alarm.

If warmup finishes after generation 3, the baseline completes at generation 13 and the first healthy-window update is generation 23. Actual timing follows committed progress, not that estimate. The limits only gate LR increases; they do not add an anchor loss or automatic rollback.

The calibrated limit, last healthy KL/generation, update clock, baseline, rolling/healthy windows, latch and probe receipt are stored in `state.json` **and the corresponding generation checkpoint**. Resume requires them to match and verifies the probe/data hashes. The PPO multiplier and its independent three-row window/cooldown remain in Adam's `brass_ppo_lr_controller` parameter group.

Full/compact reports retain the fixed-probe measurements, four group means/counts, behavior checks, health metrics, before/after overall/group limits and hold/raise/tighten reason under `il_kl_limit`. They also retain `rehearsal_schedule`, health counts by table type and the existing LR step/median/decision evidence. Console output adds current IL KL, overall/group limits and the limit decision. The report exporter includes the probe manifest and generation-zero report, without copying dense probe tensors.

Probe preparation/validation happens at startup; normal iterations evaluate only the cached probe. Preparation scans the audited validation partition using disk-backed storage. Resume checks source hashes. These costs and the ongoing rehearsal floor are separate from the per-iteration forward-pass cost; the plan's approximate one-second timing is not a guarantee on every device. The limits and rehearsal schedule are experimental, with no demonstrated playing-strength improvement.

## Continue v11.2 self-play

To continue the latest committed generation, including the reviewed generation-55 run:

```bash
python scripts/build-shared-search.py --root .

python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

The `--resume-reviewed-source` flag admits the exact reviewed source transitions from the exported run and preceding fixes, including the health/EV revision, generation-85 corrections and staged snapshot-weight recovery. Keep the existing config, upgraded IL folder and self-play directory; do not rerun config generation, IL upgrade or `--fresh`. An old table-gate block upgrades to the versioned default window through this reviewed transition; all other config fields must match. The existing fixed IL validation probe remains compatible and is reused. Unknown source/config edits and pending old collections still fail closed. Omit the review flag when the source and effective config are unchanged.

Optionally add `--expected-generation 85` only while generation 85 is the intended entry; otherwise omit it or use the actual committed generation. Continuation preserves weights, both Adam optimizers, RNG, active lambdas and enabled table origins, snapshot gate/weight, descriptors/fingerprints, clone coefficient/history, warmup, LR controller, counters, population and lineage. The first snapshot-recovery upgrade adds only an empty recovery streak; later checkpoints resume it exactly. An existing v2 EV window and its streaks are preserved exactly. Only the first upgrade from the old v1 table gate starts an empty window at committed progress and clears its superseded mixed-table streaks; old scalar reports cannot backfill moments or game support. Calibrated IL limits/latch and economic baselines are preserved. See the [generation-85 review](../ai-research/SELFPLAY_G85_REVIEW.md), [health/EV revision](../ai-research/SELFPLAY_HEALTH_EV_REVISION.md) and [generation-55 review](../ai-research/SELFPLAY_G55_REVIEW.md).

Full reports and compact history retain `gate_diagnostics`: pass failures, health and LR blockers, IL-only history fallback, accepted/attempted value steps, beta reason, held-out learner rows and whole-game counts, and mixed-table EV window support/reasons. Its `mixed_ev_status` continues checking the existing pooled tolerance after qualification. An active table with supported evidence below that tolerance emits `mixed_ev_below_tolerance:<kind>` for review; there is no automatic rollback. Console `mixedEV=<kind>:<pooled>vs<paired-selfplay>[<status>]` distinguishes `ok`, `below_tolerance` and `insufficient_support`, alongside `health`, `pass_fail` and game/iteration support. A single-game EV is still reported but does not override the pooled evidence. Removed sale-probability fields are absent from new reports; archived reports remain unchanged. Missing evidence is unavailable, not zero EV.

V11.2 retains v11.1's pre-GELU local action context, fixed-weight action projection and selector-multiply export. Descriptor conditioning affects only critic heads. Existing signed runs/checkpoints stay untouched. The rates remain IL `6e-5` / self-play `6e-6`; config generation does not repeat the earlier width-related rate reduction.

## Export self-play reports

After an iteration finishes, collect the analysis bundle with:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic
```

The explicit `--run` selects the new GAE/value run; the script's older default run name is unchanged. The destination is a new `results/joint7-v11-2-grouped-lr-critic-gNNNN-<UTC timestamp>/` folder. It includes the latest two committed full iteration reports, available compact history and evaluation reports, the saved config, and the originating IL reports. Descriptor/role/EV gate diagnostics, GAE/value diagnostics, exact categorical KL, grouped anchor evidence, LR decisions, actual learner/sample usage, quality scores/membership and disabled-teacher status are preserved. Recent rating specs/reports, compact outcome JSONL, panels and SVG curves are included together with the active cumulative rating archive; dense probe/bootstrap tensors and full replay shards are excluded. Use your actual `--run` if it differs. The original completed IL stays at `artifacts/step5/features4-v11-1-batch7-il`; the new handoff is `artifacts/step5/features4-v11-2-batch7-il`. See [export options](EXPORT_TRAINING_REPORTS.md).

## Small local checks

With the repository Python dependencies available:

```bash
PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery:tests/Reports \
  python -B -m unittest test_critic_descriptors test_critic_export test_gae_value_phase \
    test_value_buffer test_quality_pool test_auto_il_kl test_rehearsal_window \
    test_export_training_reports -v

PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery \
  python -B tests/Recovery/smoke_critic_descriptors.py \
    --output artifacts/critic-descriptors-local-smoke
```

The compute smoke exercises mixed learner/snapshot rows and an exact two-optimizer restart. The optional `--native --imitation-run <upgraded-IL>` uses two one-game trainer iterations in a new directory and checks cache reconstruction/cleanup. Both are bounded diagnostics, not evidence of playing-strength improvement. See the [revision report](../ai-research/OPPONENT_CRITIC_DESCRIPTORS_REPORT.md) for measured results and scope.
