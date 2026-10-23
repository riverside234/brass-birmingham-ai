# Sixteen-iteration PPG and lossless packed replay

The explicit g620 continuation and newly generated configurations use
`ppg-value-phase/12.0.0`: 16 cached committed iterations, uniform weights,
fresh main-process shuffling each epoch, and a value phase every 16 iterations.
Acceptance uses ordinary learner held-out games from **all 16 ages**: normally
288 games (128 self-play, 128 history, 32 v4). Snapshot and branch families stay
excluded from acceptance, and every held-out family remains excluded from all
auxiliary fitting, cloning and probe sampling.

The 16-step held-out checks, 32-step scheduled KL measurements, patience 1,
1.5-SE paired allowance, five-epoch ceiling, and constant 0.35 times actual final
PPO LR are preserved. Per the follow-up, **PPG's hard KL cap is disabled**:
`kl_cap=null`, `kl_control="penalty-only"`. Neither overall nor age-0 KL can reject,
roll back, retry or stop a PPG interval; final selected weights also need no KL
permission. Both remain measured over complete legal distributions. Nonfinite
inputs, losses and gradients remain errors. The held-out-loss retention rule is
now strict minimum loss, with the matching value-Adam state (details below).

Beta is fixed at the current checkpoint value, with **no automatic increases or
decreases**. The g620 report records `beta_clone_next=1.0`; this continuation keeps
that value. A new run starts at configured beta 1. The former 0.002 is not an
acceptance limit or a beta-adjustment threshold in this recipe. PPO's separate
KL guards, GAE economic/PPO checks and LR-controller rules are unchanged.
PPO continues DCPG output preservation. No model, IL, exploration,
opponent, LR-controller, GAE or snapshot settings are changed.

The pending g620 continuation now also changes insolvency from **0.020/event,
0.080 maximum** to **0.040/event, 0.100 maximum per seat/game**. The first two
events cost 0.040 each, the third costs only the remaining 0.020, and further
events add no penalty. The VP-180 reward and all other
reward coefficients stay unchanged. The explicit config migration updates both
ordinary and search reward specifications, preserving historical signed recipes.
The active cap is 0.100. Validation also retains support for the previously
reviewed 0.160 recipe without adopting it; each frozen
opponent still validates remaining budgets against its own checkpoint's cap.

Policy coverage stays at 2,048 clone rows per age and 512 probes per older age,
with 2,048 age-0 probes: up to **32,768 cloning rows and 9,728 probes**. Exact KL
still covers each selected state's complete legal action distribution. All
age/table/decision-group diagnostics and actual retained sample counts remain.

## Minimum-loss early-stopping follow-up

The g620 continuation has not been started. Its newly generated configuration now
uses `selection.contract="ppg-minimum-heldout-loss/1.0.0"`. The **1.5-SE stopping
rule and patience 1 stay unchanged**, but the final checkpoint is the strict
minimum of mean held-out game loss observed **within each PPG phase**. All 16
ages use the same fixed held-out set throughout that phase; no loss values from
different phases or different sets are compared.

- Include the post-PPO starting state as step zero. At each 16-step loss check,
  immediately copy weights, value-Adam state and auxiliary-head/Adam state when
  mean held-out loss is strictly lower. These snapshots are held in memory; the
  selected state is restored before the ordinary generation checkpoint commit.
- More distinct rows, later training steps and a loss inside the SE allowance
  cannot replace a lower-loss checkpoint. Exact ties keep the earlier state.
- Continue through tolerance-neutral checks as before; a paired mean loss
  increase greater than 1.5 SE triggers the existing patience stop. This affects
  how long the phase runs, never which already-observed checkpoint is best.
- If nothing improves, restore step zero and report zero retained PPG work.
  This does not undo PPO, game progress or the phase schedule. Retained row/step
  accounting follows the selected checkpoint, excluding discarded updates.

Archived recipes keep their signed coverage-selection behavior. Replaying the
logged loss curves under the new selector chooses step **0 at iteration 547**,
**112 at 555**, and **64 at 595**. This changes checkpoint selection; it does not
prove better future EV or playing strength. Loss terms and weights are unchanged.

The follow-up passed **26 focused tests in 65.2 s**, including the three archived
loss-curve replays, exact ties, no improvement, lower loss with fewer rows,
weights/value-Adam/companion snapshot matching, a tolerance-neutral interval
that continues without replacing the minimum, g620 migration and packed replay.
The isolated 16-iteration CPU smoke passed in **47.6 s**, including exact g15
restart. Its 96 ordinary held-out games gave initial loss **1.6809781654** and
minimum/final loss **1.6459169972**, both selecting step **70**. Beta stayed 1.0;
no hard KL cap was applied. Evidence:
`artifacts/ppg-best-loss-g620-tests.log` and
`artifacts/ppg-sixteen-g620-best-loss-smoke/report.json`.

## Storage and continuation

New generations use `ppg-columnar-disk-buffer/2.0.0` in the existing owned
`.ppg-eight-cache` directory. The directory name and ownership marker remain for
compatibility; retention uses the signed buffer length, not the directory name.

- FP32 state columns that are constant across a sealed generation are stored once
  as their exact uint32 bit patterns. Verified canonical 0/1 columns are packed
  into bits. All other columns remain FP32, including variable signed zeros.
- Target, merchant and industry selectors use three byte indices with absent
  sentinels. Other action columns retain FP32. A signed-zero bitmap is included
  only when needed. Full legal candidates, order, masks and exact input values
  are restored before the model runs.
- Original terminal win/rank/VP targets and stable-seat descriptors are stored
  once per game. Every row is checked for agreement before publication. The
  original relative-seat mapping restores descriptors; each row's utility and
  remaining costs stay separate and bit-exact. Branch games have their own IDs
  and metadata. Ownership/VP label sidecars retain their original receipts.
- Separate mmap arrays permit bounded, vectorized random gathers, preserving
  duplicate indices and minibatch order. Held-out partitioning reads the small
  game/role columns directly rather than reconstructing full row descriptors.
- Packing uses bounded blocks. New raw scratch arrays coexist with packed
  outputs during staging; this extra temporary space is checked and reported.
  Only new-generation scratch is removed before commit. Expired **committed**
  arrays are removed only after the new checkpoint/state commit. Unknown files,
  links, missing arrays and changed hashes fail closed.

The g620 helper generates a configuration without editing the run. With the
reward follow-up, old g613–620 MC/remaining-cost targets are excluded immediately
on adoption. At the next training startup, **after the reviewed new configuration
is durably committed and before g621 collection**, delete only the owned old
PPG caches. Unknown files or ownership mismatches refuse cleanup before any
deletion. No old targets are relabeled or reconstructed with the new reward.
The replay origin becomes g620; new caches are packed as they are collected.
The first new phase is **g636**, using g621–636, then g652. Fresh runs first run
PPG at g16. Restarting during this fill does not
restart the schedule or reset beta, either Adam, RNG, GAE, snapshot evidence,
population, lineage or ordinary progress. The previous phase count is retained.
An interruption between configuration adoption and the first new checkpoint
repeats the same empty-window migration safely. Once g621 is committed, restarting
retains its new-reward cache. DCPG still fits fresh value targets only in PPG;
this reward change does not introduce a head-only fitting pass inside PPO.

Do not rerun old beta-reset or fresh-buffer-reset helpers. Backups must include
the sealed cache arrays referenced by the matching checkpoint. Original replay
retention remains unchanged and cannot reconstruct expired 16-age cache data.

## Validation and measurements

The final **0.040/event, 0.100 cap** revision passed **13 focused tests in 59.6 s**.
They cover direct g620 adoption, the earlier 0.160 proposal's exact migration,
third-event partial charging, config generation, cache cleanup/restart, and
preservation of weights, both Adams, RNG and progress. Evidence:
`artifacts/insolvency040-cap100-g620-tests.log`. The continuation command below
is unchanged; generate its configuration with the updated helper before training.
The earlier full PPO/PPG smoke below used the previous 0.160 cap, not 0.100.

Before the cap was reduced to 0.100, the 0.040/0.160 follow-up passed **13 focused tests in 20.3 s**, including g620
combined configuration admission, preservation of archived reward migrations,
per-event saturation, inference budget limits, old eight-age and sixteen-age
cache retirement, fail-before-delete ownership checks, interruption after
adoption, and retention of newly committed reward-compatible caches. Weights,
both Adams, RNG, beta, phase counts and gate state are checked for preservation.
Log: `artifacts/insolvency040-g620-tests.log`.

An isolated **66.9 s CPU smoke with that earlier 0.040/0.160 reward** completed 16 authored PPO
iterations and a 70-step retained PPG phase over 96 held-out fixture games.
The g15 restart reproduced weights, both optimizers, RNG and evidence exactly;
the incoming checkpoint stayed unchanged. Final overall/age-0 KL were
0.0000039773 / 0.0000022737. This validates execution and continuation, not
playing strength or the effect of stronger insolvency penalties on real games.
Report: `artifacts/ppg-sixteen-insolvency040-g620-smoke/report.json`.

Before the minimum-loss follow-up, the buffer/penalty suite passed **45 tests in 238.0 s**, including ten new
packing/schedule/penalty-only regressions, existing DCPG/beta contracts, source
review canonicalization, loss selection and companion-head rollback. Log:
`artifacts/ppg-sixteen-penalty-focused.log`. Tests cover bit-exact tensors including signed zero,
all legal candidates, selector absence, duplicate gathers, mixed legacy/packed
ages, full-window holdouts, branch-family separation, sidecars, post-commit
expiry, corruption/ownership rejection, strict source/config adoption, beta/
Adam/RNG preservation and deterministic restart. Additional injected high-KL
tests verify that both overall and age-0 movement above the old cap are retained
when loss permits, beta stays fixed, and genuine held-out-loss failures still
restore the selected weights and matching value Adam.

An earlier expanded legacy sweep also encountered existing unreviewed g185,
g418/g460 and ancient resume-fixture source routes, plus an obsolete v7 fresh
default assertion. Those source routes were checked against the pre-change
g620 implementation identity and already failed there. They were not added to
the production approval registry to make old fixtures pass. This is a focused
validation result, not a claim that every historical repository test passes.

That initial isolated CPU smoke completed in **74.9 s**, using a 1,511,464-parameter
small preset and 18 authored games per iteration with reduced row budgets. It
ran PPO over 16 committed fixture iterations, then one actual PPG phase:
**70 attempted / 70 retained optimizer steps**, **96 held-out ordinary games**
across all 16 ages, final overall KL **0.0000039903** and age-0 KL
**0.0000022844**. Beta stayed **1.0** with hard-cap control disabled. Restarting
from the g15 fixture checkpoint reproduced g16 weights, PPO/value Adam, RNG,
beta, schedule and GAE state exactly. The incoming checkpoint stayed unchanged.
Phase process RSS high-water was **537,460,736 bytes**; cache staging disk peak
was **39,909,542 bytes**. These are small-fixture CPU measurements, not estimates
for the production 16-iteration buffer or GPU memory. Report:
`artifacts/ppg-sixteen-g620-penalty-smoke/report.json`.

The bounded position benchmark used 64 existing permitted-view fixtures from
`artifacts/rail-era/smoke-2`, re-encoded with current Features-4 and repeated to
8,192 critic rows. It is not a production-cache compression forecast. Dense arrays
were **128,234,024 bytes**, packed arrays **43,646,002 bytes** (about **66.0% less**).
Of 3,128 state columns, 1,524 were constant, 653 binary and 951 FP32 in this sample.
Packing took 0.41 s. Sixteen warm random gathers of 512 rows took median **36.2 ms
dense / 83.0 ms packed**; unpacking adds about **2.9 ms per 512-row gather** here.
All fields and complete policy inputs were bit-exact. This demonstrates storage
savings, not a training-throughput improvement. Evidence:
`artifacts/ppg-sixteen-g620-positions-benchmark/report.json`.

For similarly sized collections, rows and optimizer steps per full epoch roughly
double while phases occur half as often. Amortized value-fitting work is therefore
similar for equal completed epochs, but each phase is longer. Held-out/probe
sets also grow, so unchanged check intervals can increase amortized validation
overhead. Actual keep-best stopping may change. Old data are older, and neither
reduced EV volatility nor stronger play is established by this revision.

## Continue from g620

After syncing source and the review registry, run from the Linux repository:

```bash
python scripts/configure-ppg-sixteen.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 620 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppg16-cost040-g620.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg16-cost040-g620.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 620 --iterations 20
```

This continues **g621–640**, including the g636 PPG phase. If Ctrl+C left an
uncommitted g621 collection and startup reports it, archive that partial
collection first with the existing guarded helper:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 620 --apply
```

After a clean completion, reuse the same generated configuration and only the
training command, with the new committed generation. Do not regenerate to reset
the phase clock. The completed IL path stays unchanged.
