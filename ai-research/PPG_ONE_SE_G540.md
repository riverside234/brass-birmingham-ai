# g540: one-SE PPG coverage retention and fresh replay

Implemented 22 October 2026. Effective insolvency settings are **event_cost=0.020, max_cost=0.080**. The PPG epoch cap is reset to **2**, with an adaptive range of **2–6**. The completed IL filename and model/feature layout stay unchanged.

## Retention and early stopping

`ppg-value-phase/7.0.0` adds the strictly validated `selection` contract `ppg-one-se-coverage/1.0.0`.

- Compute paired loss differences over all eight iterations' ordinary learner parent games, normally **144 games** under the existing 8/8/2 quotas. Games have equal weight. Snapshot/bot decisions and branch continuations do not supply independent validation evidence; held-out families remain excluded from critic/cloning fitting.
- Keep a separate reference containing the **lowest observed mean game loss**. A worse accepted state never raises that reference.
- A candidate is eligible when its mean loss increase from this reference is **at most one paired standard error**, and it passes both policy KL safeguards. A newly lower minimum rechecks the previous selected state's eligibility.
- Among eligible candidates, prefer the state whose retained training path contains the most **distinct critic rows**. Ties prefer lower mean loss, then a later budget position. Repeated presentations do not increase distinct coverage.
- An improving check updates the minimum-loss reference. A neutral check continues without consuming patience; an increase exceeding one SE fails and rolls back. Patience remains **1**, with checks every **16 steps** and at shorter epoch tails.
- Retain the selected weights and their matching **value Adam** together, including on held-out rejection or policy-KL termination. Rejected/rolled-back training is excluded from retained sample usage.
- Both overall and age-0 exact raw-policy KL remain capped at **0.002**, checked every optimizer step against the frozen post-PPO policy. One-SE tolerance cannot waive these caps. Existing bounded beta retries remain.

`value_phase.selection` records initial/minimum/selected loss, selected-versus-minimum paired mean/SE, minimum/selected budget positions, coverage, neutral checks and failures. Existing usage fields describe the selected state. The adaptive epoch controller continues to use the **minimum-loss position**, rather than counting a late tolerance-safe coverage choice as evidence that loss is still improving.

These safeguards are engineering tolerances, not confidence guarantees or evidence of stronger play. More neutral checks can increase auxiliary runtime, potentially up to the two-epoch budget; judge retained coverage and subsequent critic/rating evidence alongside cost.

## Fresh PPG continuation from g540

The explicit g540 helper changes only `value_phase`, `insolvency` and matching search reward settings. It writes a config derived from locked committed state; it does not edit checkpoint tensors or run progress.

On the first resumed startup:

1. Restore and validate the original model, both optimizers, RNG and the reviewed source/config transition.
2. Set PPG's cache/schedule origin to committed **g540**, epoch cap to **2**, and minimum to **2**. Reset PPG epoch votes, scheduled-phase count and its KL-trip window. Keep the saved beta, ordinary PPO/GAE/LR state, population, lineage and original entropy/economic baselines.
3. After durable adoption, audit both owned `.ppg-eight-cache` and `.ppg-value-cache` directories before deleting any cache files. Unknown files and links are refused. Remove recognized arrays, metadata and partial files, retaining ownership markers. Save released-byte evidence in `ppg-fresh-cache-reset.json`.
4. Collect fresh **g541–548** data with the new reward. PPG next runs at **g548**, then **g556**. The first new checkpoint persists the new replay origin and clock. Later restarts do not repeat the reset or remove valid new caches.

The reward migration preserves weights, Adam moments/counters and RNG, invalidates the old utility-calibration receipt, and requires fresh ordinary head calibration. It never relabels old Monte Carlo targets. At event cost 0.020, the fourth insolvency event reaches the 0.080 cap. Archived .005→.0075 and .0075→.010 migrations retain their original caps and outputs.

The value Adam is restored on continuation; its existing phase-start moment reset and 16-step LR warmup still apply when PPG runs. PPG LR remains **3e-6**. This revision does not halve the LR or increase phase frequency.

## Validation

**67 distinct regression cases passed across the focused runs and corrected reruns**, including 12 new one-SE/restart cases. Coverage includes:

- exact one-SE boundary; neutral continuation; rejection beyond one SE; no upward reference drift; rechecking an older selection after a new minimum;
- preferring distinct rows over repeated presentations, with deterministic ties;
- retaining the matching weights/value Adam when a later interval fails or either independent KL cap trips;
- unchanged ordinary-parent validation scope across all eight ages;
- epoch cap hotfix 1→2, a floor of two, and the existing raise rule;
- .020/.080 reward admission, cap accounting, fresh defaults and unchanged archived migrations;
- owned-cache cleanup only after adoption, audit-before-delete behavior, unknown-file rejection and ordinary restart continuity;
- real checkpoint loading across an interrupted source-adoption transaction, preserving weights, both Adams, RNG and counters;
- existing eight-age PPG, opening PPO, g503 review and disabled-cache cleanup regressions.

The initial broad run had two missing archived source-review routes and one overly late expected epoch-vote assertion. Their exact approved routes/assertion were corrected and the failed cases passed on rerun. No unresolved test-case failures remain. Existing source reviews are preserved; archived recipes retain their old PPG semantics.

The isolated smoke used authored mixed-table/branch fixtures and a **1,511,464-parameter small model**, on two CPU threads. It created an old g540 checkpoint/cache, adopted the new reward and fresh PPG origin, ran g541–548, and repeated g548 from the g547 checkpoint:

| Measurement | Result |
| --- | --- |
| Total smoke wall time | 82.2 seconds |
| Startup cache deletion | 24 files, 20,362,287 bytes |
| First PPG cache generations | g541–548 |
| Auxiliary epochs / planned steps | 2 / 56 |
| Retained / attempted steps | 56 / 56 |
| Distinct critic rows retained | 869 |
| Paired ordinary holdout support | 48 games across all eight fixture ages |
| Retained overall exact KL | 0.0000014781 |
| Retained age-0 exact KL | 0.0000014775 |
| Continuous vs restarted g548 | Identical weights, both Adams, RNG, phase clock and epoch-controller state |
| Incoming checkpoint bytes | Unchanged |

The real smoke improved at each check. Separate controlled regressions exercise neutral/loss-rejected/KL-rejected intervals and verify their retained state. Fixture holdout quotas and tensor budgets are reduced; the production config retains 144 games and its full cloning/probe budgets. No production training, expensive evaluation, full-size GPU benchmark or playing-strength claim was made.

Local test/smoke evidence is under `artifacts/g540-coverage/`. Reproduce the new checks with:

```bash
PYTHONPATH=src:tests/Recovery python -m unittest -v test_ppg_one_se
PYTHONPATH=src:tests/Recovery python tests/Recovery/smoke_ppg_one_se.py \
  --output artifacts/g540-one-se-smoke-new
```

Use a new output directory for each smoke.

## Continuation command

After syncing this revision on the lab machine:

```bash
cd /data/not_backed_up/yxu209/ai-training

python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 540 --apply && \
python scripts/joint7-g540-ppg.py --expected-generation 540 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-one-se-g540.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

This runs **g541–560**, with auxiliary phases at **g548 and g556**. The recovery command archives only an incomplete next collection and refuses a different committed generation or an ambiguous pending checkpoint. No manual cache deletion is needed. Repeat only the train command for later clean continuations.

Principal changes: `ppg_selection.py`, `ppg_auxiliary.py`, `ppg_config.py`, `actor_value.py`, `value_buffer.py`, reward/default/resume validation, startup reporting, the g540 helper/config, source-review registries, focused tests/smoke, and `docs/MLP_V11_LAB.md`.

Reviewed implementation: `056449b7e5177379e545cb499da91a870d22ebe033fa6827ba3b5beac214d534`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
