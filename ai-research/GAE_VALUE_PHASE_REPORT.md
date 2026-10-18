# Gated GAE and PPG-style value phase

Date: 2026-10-18. Applies to a newly configured v11.1 self-play run from completed IL. Experimental training change; no demonstrated playing-strength improvement.

## Implemented behavior

The new default starts with lambda 1 and enables the value phase. Gamma stays 1. GAE chains contain the nonforced decisions of one `(game_id, learner_seat)` in record order. Rewards are successive differences of the existing combined-utility returns, including future insolvency costs; terminal continuation value is zero. This preserves forced-decision costs without treating forced rows as extra lambda steps. Lambda 1 uses the original subtraction exactly, including float32 rounding. Critic targets and actor normalization are unchanged.

Shadow diagnostics at 1, 0.98, 0.95 and 0.9 report raw standard deviation, correlation/sign agreement with MC, and era, decision-phase and early/late-period means. TD-error standard deviation is separate. These diagnostics use no RNG or model writes. Baseline explained variance continues to use MC prediction errors, not GAE variance.

The automatic gate replaces the expensive fork experiment:

1. Start at **1.0**. Require at least ten committed iterations with accepted value steps and five consecutive held-out utility EV measurements at least 0.5. Thirty iterations with accepted value steps provide the fallback. Both paths also require current healthy evidence.
2. Switch to **0.98** for subsequent iterations. Count twenty consecutive healthy iterations at this lambda.
3. Switch to **0.95** when that streak is ready and the latest common PL fit has a nonnegative slope across its last three rated checkpoints, with no existing PL regression/v4-watch alert. Missing or declining evidence delays this switch. Existing scheduled ratings supply all evidence; no new rating workload is introduced.

Healthy means the final calibrated IL-KL/economic/behavior gate passes; PPO warmup is finished; PPO is nonempty, finite and not early-stopped; value fitting accepted steps; held-out EV is finite; and the value KL guard did not trip. An epoch rejected for higher holdout loss can follow useful accepted epochs without automatically failing this health check. The twenty-iteration streak resets on unhealthy iterations. There is no automatic rollback or decrease below 0.95. `advantage.gate.mode="fixed"` supports an explicit fixed lambda in a new signed config.

These conservative progression checks are operational choices for the lighter alternative. The point-estimate rating slope has uncertainty and is not a statistical strength claim. The thirty-iteration fallback does not guarantee a sufficiently accurate critic.

## Value phase and isolation

- A deterministic SHA-256 game-id split holds out approximately **5% of complete games**, all seats together. They never enter PPO critic batches, ordinary utility calibration, or value-phase fitting. Their actor and raw-policy-cloning rows remain eligible, as specified in the plan. Thus this measures critic holdout performance, not a fully untouched policy evaluation population.
- Fit current plus previous committed iteration's critic-train rows, including forced rows (disk-buffer revision below): up to **3 epochs**, **4,096-row value batches**, existing win CE + 0.25 rank + 0.1 VP losses + utility MSE. State-only value batches use one legal candidate.
- Freeze raw post-PPO legal policy distributions on up to **8,192 nonforced rows**. Draw **512-row clone batches**, with beta initially **1**, without exploration mixing. Keep full legal candidate lists for cloning.
- Use a separate **Adam at 6e-6**, the base PPO LR, with no PPO multiplier, warmup advancement or PPO-moment changes.
- Check raw `KL(post-PPO || current)` on the same position IDs as the PPO probe after every value step. If it exceeds **0.002**, restore the attempted epoch's model weights and value Adam state. Do the same if held-out value loss rises at the epoch boundary. Earlier accepted epochs remain.
- If three of the last five iterations trip the KL guard, double beta up to **8**. Keep this sliding history and beta in the checkpoint. Reports distinguish attempted versus accepted epochs/steps.
- Refresh the utility-calibration receipt for final accepted weights. A skipped or fully rejected fit clears the old pre-PPO receipt. Empty critic splits or empty policy-reference populations skip explicitly; no split relaxation or invented validation metric.

The order is ordinary collection, critic-train-only head calibration, advantages, PPO, PPO-phase probes/LR decision, value phase, any teaching, final IL-KL/probes, checkpoint. The LR controller retains its post-PPO signal. Final IL-KL limit updates and quality-pool admission use the final weights. The pure IL-KL transition is recomputed from the same prior committed state, so its windows advance only once.

Generation checkpoints save the GAE state, clone coefficient/trip history, disk-buffer receipt and the separate Adam in `training.actor_value_phases`, alongside the existing PPO optimizer and RNG. Resume checks the phase config and generation. Missing phase state in a trained checkpoint is rejected. Weights-only population snapshots omit the auxiliary optimizer while retaining model/calibration/lineage; full resume checkpoints keep both optimizers.

## Compatibility and changed files

The v11.1 architecture and Features-4 dataset/IL contracts are unchanged. The completed IL directory remains `artifacts/step5/features4-v11-1-batch7-il`. The generator was exercised against the local completed diagnostic v11.1 IL; the production lab checkpoint is not present locally. On the lab, generation still verifies its completion, hash, dataset and lineage.

Unchanged recipe: 50/45/5 table mixture, 24-member quality pool, 4/2/2 learners, exploration 0.08, PPO reward/LR/entropy settings, rehearsal floor/window, PL schedule, no v4 search, no teacher training, no Monitor/Broader Evaluation. Existing configs and runs are not rewritten or converted.

Main files:

- `recovery/gae.py`: GAE, shadow metrics, deterministic gate and verified common-fit rating slope.
- `recovery/value_phase.py`: split, critic diagnostics, frozen raw cloning and transactional value epochs.
- `recovery/actor_value.py`: lifecycle, separate optimizer and checkpoint state.
- `recovery/value_buffer.py`: checksummed critic-only disk cache, bounded reads, replay reconstruction and commit-time deletion.
- `recovery/updates.py`, `utility.py`, `league.py`: actor/critic separation, calibration exclusion and iteration ordering.
- `recovery/config.py`, `configs/step5/joint7-latest.json`: strict signed blocks and fresh defaults.
- `recovery/history.py`, `retention.py`: compact metrics and slim opponent snapshots.
- `configs/learning/imitation-handoff-sources.json`: exact predecessor review for completed IL only; no old PPO migration approval.
- `tests/Recovery/test_gae_value_phase.py`, `smoke_gae_value_phase.py`: focused regressions and isolated smoke checks. Existing regression fixtures explicitly retain their older recipe, teacher opt-in and historical source receipts.
- `docs/MLP_V11_LAB.md`: generation, fresh start, continuation, optional fixed-probe reuse and report export commands.

## Initial single-iteration validation and measurements

Evidence is under `artifacts/gae-value-20261018/`. **131 distinct tests passed across the broad run and targeted final reruns.** The initial broad run covered 129 tests and exposed two stale test-fixture assumptions: a prepended existing IL source receipt, and a teacher test that inherited the disabled teacher default. The assertions were corrected to retain exact historical receipt contents and explicitly enable the standalone teacher fixture. Both passed their reruns; two additional GAE/controller cases also passed. Final results are recorded in `final-focused.log`, `final-additional.log` and `validation.json`; no production safeguard was relaxed.

The [compute smoke](../artifacts/gae-value-20261018/compute-final/report.json) used the real medium v11.1 utility model (**5,921,320 parameters**), authored Features-4 tensors, eight fixture game IDs, 128 rows and 1–4 legal actions, on two CPU threads. Its deliberately small batches were 128 value rows and 32 clone rows, with a three-epoch cap. Of 128 rows, 112 were critic-train and 16 held out. It accepted two value steps/epochs, rejected and restored the third for increased holdout loss, and reproduced the next PPO/value update bit for bit after checkpoint restore, including both optimizers and RNG.

| Compute-smoke measurement | Before value phase | Final accepted weights |
| --- | ---: | ---: |
| Held-out value loss | 1.642278 | 1.617137 |
| Held-out utility EV | 0.011820 | 0.023942 |
| Held-out utility MSE | 0.223907 | 0.222235 |
| Post-PPO raw policy probe KL | 0 | 0.00004136 |

Raw advantage standard deviations on the same authored actor rows were **0.468752 / 0.456942 / 0.440084 / 0.414197** at lambda **1 / 0.98 / 0.95 / 0.9**. The training update used lambda 1. These short synthetic chains do not validate long-game GAE bias, the adoption thresholds, or playing strength.

The [native trainer smoke](../artifacts/gae-value-20261018/native-final/report.json) completed one fresh iteration and one same-run continuation: two ordinary games, one PPO step per iteration, **312 / 296 learner rows**, **170 / 166 forced rows**, no teacher or evaluation games. Neither one-game collection contained a hash-held-out game, so value fitting correctly skipped and lambda stayed at 1. The fixed 144-row diagnostic IL probe was reused; its insufficient production-health evidence was not waived. Donor files, completed diagnostic IL and the first committed checkpoint remained unchanged. The LR signal matched the post-PPO probe in both reports. Total smoke time including loading, collection, training, checkpointing and resume was **178.4 seconds**.

On the compute fixture, PPO took **3.09 seconds**, value work **3.50 seconds**, and the whole compute/resume smoke **20.24 seconds**. Tests ran concurrently, so these are illustrative CPU observations, not a controlled GPU or production-throughput benchmark. No playing-strength improvement is established by the loss reduction, entropy or advantage variance.

## Initial runtime observations and current commands

At an illustrative 86,000 collected rows, the expected 95% critic-train portion is about 81,700 rows: roughly 20 value steps per epoch, at most 60 attempted steps over three epochs, plus cloning, per-step 2,048-row policy probes, validation, and rollback copies. Early stopping can reduce accepted work. Full-collection GPU timing remains unmeasured; the plan's “tens of seconds” is not a guarantee.

The extra Adam moments add roughly **45 MiB** at this model size, plus a roughly **68 MiB** temporary model/Adam epoch snapshot and bounded clone-probability storage. Full generation checkpoints grow accordingly; frozen population copies do not inherit those extra moments. The compute-smoke full checkpoint was 118,819,106 bytes. Collection and scheduled rating game counts do not increase.

Generate the new config without rerunning or renaming IL:

```bash
python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-v11-1-batch7-il \
  --output configs/step5/joint7-v11-1-grouped-lr-gae-ppg-buffer2.json
```

Use the fresh-start and twenty-iteration continuation commands in [the lab instructions](../docs/MLP_V11_LAB.md#fresh-self-play-from-completed-v111-il). The config generator refuses overwrites; use another unused suffix if necessary. Reports export with `python scripts/export-training-reports.py --run artifacts/step5/joint7-v11-1-grouped-lr-gae-ppg-buffer2`.

Initial single-iteration tested implementation: `e21348a9f87705983350b5d5b49fc8d17178ed540c9107d79011090cc6b80b2e`. Learning compatibility remains `b3b944dbc492bc1483c41bdb5a9d566f13e2b2a8b7988dbeec8fcb9ebb372674`.


## Two-iteration disk-buffer revision (2026-10-18)

The current default is `value_phase.contract="ppg-value-phase/2.0.0"`, `buffer_iterations=2`. The existing explicit one-iteration contract remains readable; the signed configuration and phase checkpoint state prevent silent conversion of an older run. Fresh config generation still admits the completed v11.1 IL. No architecture, IL dataset, reward, GAE thresholds, PPO recipe, clone/KL/holdout safeguards, or unrelated rehearsal/rating controls changed.

`value_buffer.py` stores only critic-train rows, including forced actions. It retains state features, one real legal candidate, stable seat mapping, win/rank/VP/utility targets, remaining insolvency budget, and game/phase identity. Dense features stay in a read-only NumPy mmap; only small target columns become resident tensors. Checkpoint receipts bind the generation, encoding compatibility, reward, holdout fraction, originating policy/model, verified ordinary replay receipts, row counts and all cache bytes. Cached rows never supply actor advantages, old behavior probabilities, policy cloning or PPO data. Cloning, policy probes and the heldout-loss guard use the current collection.

Cache transaction order:

1. On iteration 1, fit current rows only. Write its critic cache before the generation checkpoint.
2. On iteration N, shuffle current and N-1 critic rows into the value batches, without duplication within an epoch. Never admit hash-held-out games.
3. Publish the N cache and checkpoint, then commit `state.json`. Close disk readers and delete N-1's cache. Only N remains for the next iteration.
4. After interruption, keep the checkpoint-referenced cache and remove abandoned publications/partials. Rebuild missing or corrupt derived data one replay shard at a time; its reconstructed receipt/checksum must match exactly. If authoritative replay is unavailable or changed, fail explicitly rather than skipping data.

The current generated recipe keeps two replay rounds. Both config validation and manual retention enforce at least two. A pending future report cannot push the last two committed rounds out of retention. Cleanup audits owned flat-file paths, refuses linked/unknown targets and never recursively deletes a run. Full optimizer checkpoints retain the buffer receipt; weights-only population snapshots still omit phase training state.

Reports and compact history retain current/previous available rows, forced rows, distinct rows accepted for training, accepted presentations, attempted/accepted steps, cache generations/bytes, write time, replay-rebuild status, and commit/startup deletion receipts. Console output includes accepted `value_rows=current+previous`. A rejected epoch reports zero accepted rows even though its backward passes ran.

### New bounded validation

Evidence: `artifacts/ppg-buffer2-20261018/`.

- **48 tests passed** in completed runs: 24 GAE/value/buffer regressions, 3 source-review/resume regressions, and 21 retention/report-export regressions. Coverage includes disk/value parity, forced rows, heldout exclusion, empty buffers, accepted previous-row updates, exact continuation, corruption/rebuild checks, incomplete publication, safe deletion, committed replay retention and compact reporting. The existing source-review test now calculates its source fingerprint once instead of once per historical entry; all assertions are unchanged. See `focused-tests.log`, `source-review-final.log` and `validation.json`.
- The actual config-generation command accepted the local completed diagnostic v11.1 IL and emitted the new two-iteration defaults (`generated-config.json`). The production lab IL is not available locally; normal completion/hash/lineage checks still apply there.
- [Medium compute smoke](../artifacts/ppg-buffer2-20261018/compute-final/report.json): real 5,921,320-parameter v11.1 utility model; two iterations of authored 128-row Features-4 inputs, eight fixture games and 1-4 legal candidates each, two CPU threads. Iteration 1 admitted 112 critic-train rows and held out 16, accepted two value epochs and restored the third. Iteration 2 exposed **112 current + 112 previous** critic rows and 16 current heldout rows. Instrumentation verified all 112 previous rows participated in backward passes. Its two attempted value steps were rolled back because heldout loss rose: **zero accepted previous rows**, preserving the safeguard. A focused smaller fixture separately verifies an accepted buffered update. Resumed execution reproduced model weights, both optimizers and RNG bit for bit. Expired cache deletion and unchanged earlier checkpoints were verified.
- Compute timings: iteration-1 PPO **3.35 s**, value fitting **3.91 s**, cache write **0.097 s**; iteration-2 PPO **1.96 s**, attempted value fitting **2.76 s**, cache write **0.072 s**. The whole compute/restore check took **23.91 s**. These are small CPU measurements with concurrent tests, different accepted work and warmup; they do not estimate full-run throughput.
- [Native trainer smoke](../artifacts/ppg-buffer2-20261018/native-smoke/report.json): one fresh iteration followed by one real continuation, **two ordinary games**, **312 then 296 current learner rows**. Deliberately deleting the first smoke cache forced exact reconstruction from native replay on resume. The second phase admitted all 312 previous critic rows, then removed their expired cache only after generation 2 committed. The newest 296-row cache remained. Both one-game collections had no hash-held-out game, so value fitting correctly skipped without weakening the split. Total **172.41 s**; no evaluation campaign, production checkpoint writes, donor changes or prior-checkpoint changes.

### Disk and runtime expectations

The current cache record is **13,949 bytes** (3,128 state floats, 327 action floats, targets and identity columns). At 81,700 train rows, one array occupies about **1.14 GB / 1.06 GiB**, compared with caching every legal candidate. Two arrays coexist while committing, about **2.28 GB / 2.12 GiB**, then the older one is removed. Receipts are small additional files. Rebuild reads one source shard at a time, so its temporary memory is bounded by shard size.

For equally sized collections, the critic workload and accompanying clone/KL checks approximately double: 163,400 critic rows give **40 batches per epoch**, up to **120** over three epochs at batch size 4,096. Validation and guards can stop earlier. No extra ordinary or evaluation games are introduced. Production GPU latency and disk throughput remain unmeasured; no playing-strength improvement is claimed.

Current tested implementation: `419e6a74a223db78b2c5209ffdd5b86d50e3449f5b31a70c22daa96587a0b421`. Learning compatibility remains `b3b944dbc492bc1483c41bdb5a9d566f13e2b2a8b7988dbeec8fcb9ebb372674`. Generate the new `joint7-v11-1-grouped-lr-gae-ppg-buffer2` config/run using the lab commands above; the completed IL path is unchanged.
