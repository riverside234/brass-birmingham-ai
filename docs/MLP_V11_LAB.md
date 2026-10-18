# Grouped MLP v11.1: fresh IL and self-play

V11.1 is the current MLP default: projections **32/64/48/320/352/48 + 12**, fusion **876 → 512**, **eight width-512 residual blocks**, a width-160 action encoder, and a 192-wide policy scorer. Local slot/link/industry banks include neighboring slots and opponents' industry mats; their bias-free role projections start at zero. The shared per-player outcome head remains width 128. See the [model contract](../ai-research/GROUPED_PROJECTION_CONTRACT.md) and [validation report](../ai-research/MLP_V11_1_REPORT.md).

The October 18 quality-pool/rating revision reuses **completed v11.1 IL without retraining**. Skip the IL commands below if it is already complete. Generate a new bound self-play config and use a new run directory. Model tensors, features, and IL losses are unchanged; the exact preceding v11.1 implementation is admitted by the completed-IL source receipt. This revision does not authorize converting an existing PPO run.

## Existing joint7 IL data

Reuse the already converted, audited **Features-4** dataset as-is. The state/action dimensions remain **3,128 / 327**, and feature, encoder, derived-feature, rules and target contracts are unchanged. The new model checks the static board topology as part of admission. Existing native-board data satisfies those checks, including the three-region Worcester/SouthWest/Kidderminster links.

Keep the manifest, labels, split assignments and raw shards. The normal loader may rebuild source-bound tensor caches in the new IL run's `.imitation-cache/`; do not regenerate demonstrations solely for this upgrade. A path containing v8 is still valid when its contents are Features-4.

Use **fresh v11.1 IL weights and a new self-play directory**. V11 and all earlier checkpoints cannot be resumed or reinterpreted as v11.1, including with source-review flags. Old neural PPO trajectories are not current behavior data.

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

## Fresh v11.1 IL

Use the actual converted manifest path if yours differs. New run paths avoid modifying previous checkpoints.

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/

python scripts/feature-engineering.py inspect

python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-v11-1-batch7-il \
  --epochs 10 --min-epochs 3 --patience 3 --workers 12
```

The epoch/loader-worker arguments are example lab settings. Inspection should show architecture `brass-four-seat-mlp-resnet/11.1.0`, local layout `grouped-mlp-local-entities/2.0.0` and **5,904,644 base parameters**. The IL CLI now takes its default learning rate from the selected config, so it does not silently retain the old hard-coded 1e-4.

To resume an interrupted v11.1 IL run at its committed epoch, repeat the same command with identical settings and source. Warmup, Adam and RNG restore from its checkpoint. Do not use `--from-imitation-run` to load an older architecture.

To extend this completed 10-epoch run to **20 total epochs**, use the same run and its original settings:

```bash
python scripts/step5.py --config configs/step5/joint7-latest.json imitate \
  --manifest artifacts/il-replay/features4-compact-v8-batch7/manifest.json \
  --run artifacts/step5/features4-v11-1-batch7-il \
  --epochs 10 --extend-to-epochs 20 \
  --min-epochs 3 --patience 3 --workers 12
```

`--epochs 10` identifies the unchanged original recipe; `--extend-to-epochs 20` raises its execution cap to 20, **not 20 additional epochs**. Training resumes at epoch 11 after the committed epoch 10. It stops no later than epoch 20 and may stop earlier after three consecutive epochs without an improvement greater than `0.0001` over the recorded best validation loss. Keep the actual original patience/minimum-epoch settings if they differ from this example.

The extension restores the latest checkpoint's weights, Adam moments, CPU/CUDA RNG and warmup counter. It retains all validation history, the previous best checkpoint, the patience counter and the original dataset/checkpoint lineage. Existing prepared shards are reused only through the pinned preparation-equivalence review, with raw-input and cache checksums still verified. No dataset generation or warmup restart is needed.

Repeat the **same extension command** after interruption; the saved total cap is idempotent. It cannot revive a run stopped by patience or silently change the model, dataset, learning rate, batching, loss, precision or other original settings. The exact preceding v11.1 sources are explicitly reviewed; unknown source changes remain rejected. The original signature remains immutable, with the increased cap and actual executing source recorded in `state.json`/`report.json` as `epoch_extensions` and in subsequent checkpoints. After this extension commits, self-play handoff requires it to finish (at the cap or by patience); a stale 10-epoch report is rejected. Generate a new self-play config after choosing the final completed IL checkpoint.

`--from-imitation-run` remains a separate weights-only continuation into a **new** run and resets Adam/RNG/warmup. Do not use it for the exact extension above.

## Fresh self-play from completed v11.1 IL

Install the self-play dependencies and generate a fresh config from the updated default template, bound to your completed IL checkpoint. The commands below use new `joint7-v11-1-grouped-lr-quality` config/run names so your earlier `joint7-v11-1-grouped-lr.json` is left unchanged. The new config includes all current updates: 50/45/5 tables, the 24-member quality-score pool, PL ratings, auto IL-KL/LR controls, rehearsal floor, teaching disabled, `v4_search_enabled=false`, and Monitor/Broader Evaluation disabled. Keep the completed IL directory unchanged; another IL run is not required. The generator refuses to overwrite a config. If the new names also exist, choose another unused suffix consistently in generation, training, continuation and report export.

```bash
python -m pip install -r requirements-step5.txt

python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --output configs/step5/joint7-v11-1-grouped-lr-quality.json

python scripts/step5.py --config configs/step5/joint7-v11-1-grouped-lr-quality.json train \
  --fresh \
  --run artifacts/step5/joint7-v11-1-grouped-lr-quality \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --iterations 1
```

The initial PPO model copies the completed IL weights exactly. PPO receives a fresh Adam optimizer and its own warmup counter. The utility extension is initialized/calibrated through the existing guarded path.

Use your actual completed v11.1 IL path in both commands. The shipped `joint7-latest.json` is a template; `configure` binds the selected IL checkpoint and its rehearsal manifest instead of importing the template's historical lineage.

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

## Quality-score history opponents

Each history seat is sampled independently: 20% pinned IL, otherwise `p_i = 0.8 * softmax(q)_i + 0.2 / N` over the scored members. Until the first admission, every history seat uses IL. Both opponent seats may select the same model. The quality probability is conditional on selecting the scored pool; its per-seat probability is `0.8 * p_i`.

After collection, every learner/opponent seat pair contributes its official-tiebreak outcome (win 1, tie 0.5, loss 0). A scored member receives `q_i -= 0.01 * outcome / (N * p_i)`, using the probabilities frozen when the collection was planned. Recenter by subtracting the largest q. IL outcomes are reported but never change a score. No rating result enters this sampler.

Every five committed iterations, admit the current model only after five recorded healthy iterations and when every `KL(member || candidate)` is at least 0.02 on the fixed IL validation probe. Health uses the existing baseline/behavior/evidence checks below; its window is not backfilled. Consequently the first admission can be later than generation 15. New q is the current maximum. When full, evict the lowest-q eligible member, breaking exact q ties by smallest KL to another member. IL, the two newest members, and members younger than 20 iterations are protected. If all are protected, skip admission.

`state.json` stores members, scores, planning probabilities, cached-probe receipts, health window and membership events. Each generation checkpoint binds this state by digest; resume checks it. Reports and compact history retain q/p, seats, comparisons, raw and five-iteration outrank rates, and admission/eviction reasons. Active probability caches, pool checkpoints and all rated checkpoints survive retention cleanup.

## Held-out Plackett–Luce ratings

Every ten committed generations, collect **32 fixed validation setups × 4 candidate-seat rotations = 128 main games**, with exploration 0, neutral focus and no search for either neural or v4 policies. Half the setups have an IL reference seat, half v4. The other seats use the eight nearest rated checkpoints, estimating the new candidate from the latest rated checkpoint. Startup explicitly falls back to IL + v4; with one rated checkpoint it fills both extra seats. Previously evicted training-pool models remain available as rated opponents.

Every slot additionally runs **eight setups × four rotations for each diagnostic panel**: one learner versus three v4 bots, and two learners versus two IL opponents. The all-learner health panel comes from recorded ordinary self-play. This adds **64 games**, so the default total is **192 held-out games per 3,200 training games (6% extra game volume)**, plus fitting and I/O. Scheduled Monitor and Broader Evaluation add no games while disabled. Held-out games never train PPO. Rating v4 is the pinned `balanced-v3-masked-economy` profile in the configured reference bank, with search disabled, matching the search-free setting throughout the new self-play recipe.

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

Fresh configs now use **`ppo-categorical-kl-lr/1.2.0`** plus the strict **`auto-il-kl-limit/1.0.0`** block. Both are part of the run signature. Completed v11.1 IL remains compatible; its directory and selected weights are reused. Generate a new bound config before starting. A previously generated config does not silently acquire these settings; the helper requires a new output filename. Existing committed runs require a separately reviewed source/config transition.

The per-update LR signal remains exact **KL(old behavior || post-PPO behavior)** on the ordinary proportional probe, capped at 2,048 nonforced rows. The active PPO trainer now also uses this exact categorical KL after **each optimizer step**, stopping further steps when it exceeds `update.target_kl=0.01`. The old probabilities and probe positions stay fixed throughout all PPO epochs; both distributions use the same legal masks, temperature and 0.08 exploration mixture. The sampled-action estimate on **all nonforced rows** remains a secondary guard at each epoch end. Nonfinite probe or full-dataset KL aborts the update before any further optimizer steps or checkpoint publication. No KL penalty is added to the loss, and stopping does not undo the last step.

The exact and sampled probe values share one forward pass per microbatch. Reports and compact history retain `actor.kl_guard` (contract `ppo-categorical-kl-guard/1.0.0`, checks, maximum exact KL, stop reason and step), plus each epoch's `probe_categorical_kl`. Existing `probe_kl` and `post_update_kl` retain their sampled-action meaning. The console prints `KLstop=categorical_probe`, `sampled_full_dataset` or `none`; the LR report records the same stop reason and actual completed/planned steps.

Raise by 1.25 after a full three-valid-iteration median below 0.0045 with no stops, no cooldown and passing health/anchor checks. Cut by 1.5 immediately on an early stop before half the planned steps, two stops in three valid iterations, or current exact KL above 0.015. Cuts start a three-valid-iteration cooldown; multiplier bounds remain 0.5–3. Warmup, empty/nonfinite update handling, shadow/off modes and PPO-only multiplier application remain unchanged. **Protective LR cuts never depend on anchor/health approval.**

The categorical guard remains enabled and its preceding guard-only source approvals are preserved. Those approvals do **not** authorize the new quality-pool/rating recipe on an existing PPO run. For this revision, generate a new recipe-3 config and use the earlier `--fresh` command without a source-resume flag. Completed v11.1 IL remains usable; the PPO guard does not alter supervised IL updates. Signed older recipes are not silently upgraded, and pending-collection safeguards remain active.

Validation (2026-10-18): 59 focused regressions passed, covering exact/masked behavior KL, stop boundaries, nonfinite values, sample accounting, LR feedback, reviewed resume and IL extension/cache reuse. A separate CPU smoke with fresh current v11.1 weights and the utility head completed two six-step PPO updates; checkpoint restoration produced bitwise matching next-update weights and Adam state. On a 2,048-position synthetic probe (2–32 candidates, microbatch 128, two CPU threads, three timed repeats), median guard time was 0.990 s for sampled-only versus 1.008 s for exact plus sampled. This measures probe overhead only; CUDA, full collection throughput and playing strength were not measured. The reproducible smoke is `tests/Recovery/smoke_categorical_kl_guard.py` and requires a new `--output` directory under `artifacts/`.

The IL-distance gate now uses a separate **fixed validation probe**, built at generation 0 from the completed IL dataset. It selects up to 2,048 distinct nonforced positions with seed 710193, balanced across era × decision-phase strata using deterministic hash priority; exhausted strata redistribute their allocation. The manifest, selected identities, source hashes and cached IL probabilities live under `il-kl/`. No training RNG is used. The generation-zero report verifies KL(IL || IL) = 0. Validation positions never enter PPO or rehearsal updates; rehearsal still uses the training split.

Both policies use raw temperature-1 probabilities with **no exploration mixture**. For a finite, nonnegative KL that remains zero for identical policies, both raw distributions receive the same **1e-6 floor and renormalization** for the KL calculation. This explicitly resolves the plan's numerical-floor/identity-check ambiguity. Sale/pass checks use the original raw probabilities: where IL sale mass is at least 0.2, learner sale mass must be at least half; where a non-pass move exists and IL pass mass is below 0.2, learner pass mass must not exceed `2 * IL + 0.05`. Every eligible position must pass.

The four fixed-probe groups are **Canal action, Canal other, Rail action, Rail other**. This replaces the early/late-round anchor groups for controller 1.2; explicit 1.1 configs retain their previous behavior. Each group needs at least 64 positions, and LR increases require the full 2,048-position probe. Tiny diagnostic datasets can produce smaller probes with explicit counts; they cannot qualify an increase.

Health uses **all-learner self-play tables only**, with seat-weighted five-iteration rolling averages and a frozen baseline from the first ten iterations that start after LR warmup finishes:

| Metric | Healthy when |
| --- | --- |
| VP per seat | At least 0.85 × baseline |
| Selected sell decisions per seat/game, including continuations | At least 0.70 × baseline |
| Passes with a legal non-pass alternative per seat/game | At most 2 × baseline + 0.5 |
| Zero-VP seat fraction | At most max(0.05, 2 × baseline) |
| Fraction of seats with an insolvency event | At most baseline + 0.10 |

Zero-VP seats, insolvency events and distinct affected seats are recorded separately for every table type. Historical/v4 tables cannot influence the calibrated self-play baseline. Missing evidence and failed sale/pass checks block increases.

The overall limit starts at **0.15 nats**. Every group limit is always **2 × the current overall limit**, initially **0.30**. Before the baseline is ready the limits stay at their initial values. Afterward:

- Any health/behavior/evidence failure immediately tightens the overall limit to `min(limit, max(0.05, last_healthy_KL))`, updates both limits and restarts the ten-iteration clock.
- After ten further healthy iterations, use their largest overall KL, `K`: `min(1.0, 1.25 * limit, max(limit, 1.25 * K))`. Small distances never lower the limit; growth is capped at 25% per update and 1.0 overall / 2.0 per group.
- A KL budget violation latches the LR-increase gate. Recovery requires healthy evidence, mean KL strictly below 0.8 × its current limit, and every group strictly below 0.8 × its corresponding limit. Mean KL above 2 × the current limit raises a diagnostic alarm.

If warmup finishes after generation 3, the baseline completes at generation 13 and the first healthy-window update is generation 23. Actual timing follows committed progress, not that estimate. The limits only gate LR increases; they do not add an anchor loss or automatic rollback.

The calibrated limit, last healthy KL/generation, update clock, baseline, rolling/healthy windows, latch and probe receipt are stored in `state.json` **and the corresponding generation checkpoint**. Resume requires them to match and verifies the probe/data hashes. The PPO multiplier and its independent three-row window/cooldown remain in Adam's `brass_ppo_lr_controller` parameter group.

Full/compact reports retain the fixed-probe measurements, four group means/counts, behavior checks, health metrics, before/after overall/group limits and hold/raise/tighten reason under `il_kl_limit`. They also retain `rehearsal_schedule`, health counts by table type and the existing LR step/median/decision evidence. Console output adds current IL KL, overall/group limits and the limit decision. The report exporter includes the probe manifest and generation-zero report, without copying dense probe tensors.

Probe preparation/validation happens at startup; normal iterations evaluate only the cached probe. Preparation scans the audited validation partition using disk-backed storage. Resume checks source hashes. These costs and the ongoing rehearsal floor are separate from the per-iteration forward-pass cost; the plan's approximate one-second timing is not a guarantee on every device. The limits and rehearsal schedule are experimental, with no demonstrated playing-strength improvement.

## Continue v11.1 self-play

After the one-iteration run commits generation 1:

```bash
python scripts/step5.py --config configs/step5/joint7-v11-1-grouped-lr-quality.json train \
  --run artifacts/step5/joint7-v11-1-grouped-lr-quality \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --expected-generation 1 \
  --iterations 20
```

Use the actual generation for the optional guard, or omit it to resume the current committed generation. Matching-source continuation preserves weights, Adam/RNG, warmup, LR controller, counters, population and checkpoint lineage. Existing freshness/source-review rules remain active for later source changes and pending collections. This new recipe starts fresh; it does not silently convert an existing staged-curriculum run.

V11.1 adds the selected local vectors **before action GELU and LayerNorm**. Training keeps indexed selectors; new ONNX exports use the fixed action linear weights plus one `143 × 160` per-position projected-bank multiply. The exporter simplifies the graph, checks exact GELU/LayerNorm fusion, and runs ORT at `ORT_ENABLE_ALL`. Role projections still initialize at zero. The Features-4 data is unchanged, but this is a different learned function from v11: **fresh IL is required**, and no source-review receipt bridges v11 weights. See the [v11.1 report](../ai-research/MLP_V11_1_REPORT.md).

Existing signed runs and checkpoints are left untouched. Do not point these commands at an old v11 run directory. The rates remain IL `6e-5` / self-play `6e-6`; config generation from v11 does not apply the earlier 0.6 rate adjustment again.

## Export self-play reports

After an iteration finishes, collect the analysis bundle with:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-1-grouped-lr-quality
```

The explicit `--run` selects the new quality run; the script's older default run name is unchanged. The destination is a new `results/joint7-v11-1-grouped-lr-quality-gNNNN-<UTC timestamp>/` folder. It includes the latest two committed full iteration reports, available compact history and evaluation reports, the saved config, and the originating IL reports. Exact categorical KL, grouped anchor evidence, LR decisions, actual learner/sample usage, quality scores/membership and disabled-teacher status are preserved. Recent rating specs/reports, compact outcome JSONL, panels and SVG curves are included together with the active cumulative rating archive; dense probe/bootstrap tensors and full replay shards are excluded. Use your actual `--run` if it differs. The completed IL directory remains `artifacts/step5/features4-v11-1-batch7-il`. See [export options](EXPORT_TRAINING_REPORTS.md).

## Small local checks

With the repository Python dependencies available:

```bash
PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery \
  python -B -m unittest test_v11_1 test_mlp_v11 test_v11_architecture test_v11_learning_rate -v

PYTHONPATH=src:tests/FeatureEngineering:tests/R2:tests/Recovery:tests/Reports \
  python -B -m unittest test_auto_il_kl test_ppo_lr_controller test_anchor_group_gate \
    test_categorical_kl_guard test_quality_pool test_strength_rating test_audit_retention \
    test_v11_learning_rate test_source_reviews.SourceReviewTests test_export_training_reports -v

PYTHONPATH=src:tests/FeatureEngineering python -B tests/FeatureEngineering/benchmark_v11_1.py \
  --output artifacts/feature-engineering/v11-1-local-benchmark
```

The benchmark compares the indexed and selector-multiply execution forms of the same v11.1 model. It measures CPU tensor-ready inference only. Neither smoke tests nor these timings establish improved playing strength.
