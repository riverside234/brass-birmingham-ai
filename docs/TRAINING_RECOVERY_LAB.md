# Fresh training in conda: selected v4 teachers and Masked ISMCTS NN

## Current g1320 continuation

Use [the g1320 reward/cache continuation command](../ai-research/TRAINING_G1320_REVISION.md).
Insolvency event cost is **0.01**, with the existing **0.075 cap**. The helper
changes only the learner/search reward. After durable reviewed adoption, startup
retires old-reward PPG caches: g1321 uses current data, g1322 uses g1321+g1322.
Weights, PPO/value/auxiliary Adam, beta/trips, RNG, gate evidence, population and
training progress are preserved. Loss16/KL32 and all other PPG settings remain.
Fresh self-play configuration generation also uses 0.01/0.075.

## Archived g1120 continuation

Use [the g1120 continuation command](../ai-research/TRAINING_G1120_REVISION.md).
Each neural seat uses **temperature 1.3 for its first three action decisions**,
including a fresh three-decision window after a fork. Payment/bonus/continuation
choices do not consume this window. The existing decoupled raw-T1 PPO objective
selects the same eligible nonforced rows. Branches skip the first three action
decisions and remain at 1% per eligible decision.

PPG held-out loss checks return to **16 steps**, on all held-out games from both
cached iterations. Periodic KL checks run every **32 steps**, plus epoch tails
and verification of any otherwise unchecked retained state, with the 0.002 cap.
Saved beta (0.5 at g1120), adaptive bounds 0.5–32, LR curve, replay caches and
training state are preserved. Fresh config generation also uses this recipe.

## Archived g1050 continuation

Use [the g1050 continuation command](../ai-research/TRAINING_G1050_REVISION.md).
PPG beta is set to **0.5 once on adoption**, then adapts within **0.5–32**.
Held-out loss checks run every **8 steps**, using the full two-iteration held-out
set. Periodic KL checks remain every **16 steps**, with exact verification of
any otherwise unchecked retained loss checkpoint. The hard KL cap stays 0.002.
Linear warmup to step 16, cosine decay to 0.001 × peak, the two-epoch cap and
0.65 × actual PPO peak LR are unchanged. Replay caches and training state are
preserved; no fresh-buffer reset or IL change is needed.

## Archived g950 continuation

Use [the g950 review and continuation command](../ai-research/TRAINING_G950_REVIEW.md).
PPG uses linear warmup from **0.1 × peak at step 1** to **peak at step 16**,
then cosine decay to **0.001 × peak** at the end of its fixed **two-epoch** budget.
Peak remains **0.65 × the last actual PPO LR**. On adoption, beta is set to **1 once**;
subsequent adaptive updates and restarts respect the new **1–32** bounds.
PPG still runs every iteration, fits both critic ages and uses all held-out games
from both ages for loss checks. Loss and KL checks stay at 16 steps; the hard KL
cap remains 0.002. Current-only gate evidence, cached targets, Adam/RNG, progress,
reward and completed IL are preserved. No fresh-buffer reset is needed.

## Archived g850 continuation

Use [the g850 revision and continuation command](../ai-research/TRAINING_G850_REVISION.md).
Openings use temperature **1.5**, epsilon mixing **0.05**, fixed entropy **0.005**,
and branching **1% per eligible decision**. PPG still runs every iteration on
two critic ages and both ages' held-out games, now with a **two-epoch cap**, beta floor **0.5**, loss/KL checks
every **16** steps, and cosine warmup/decay around a peak of **0.65 × actual PPO LR**.
Insolvency is **0.02 per event / 0.075 cap**. PL refits use four single-threaded
processes by default (configurable 4–8) and full-fit warm starts. The completed
IL directory and training progress are preserved. Use the new config helper;
do not manually edit the saved run signature or reuse incompatible reward caches.
GAE and snapshot evidence remain current-only; pooled acceptance does not count
the previous iteration again as new gate evidence.

## Archived g660 PPG continuation

Use the [g210-style PPG restoration](../ai-research/PPG_G210_G660.md) for the
g660 recipe. PPG runs after every PPO iteration, trains current plus previous
critic data, validates current-iteration holdouts, and allows at most **3 epochs**.
Beta starts at **4 once on adoption**, then adapts in **[1, 32]**. PPG uses
**0.65 times the actual PPO LR**, a **0.002 hard KL guard every 32 steps**
(plus epoch tails and retained-checkpoint verification), and strict
non-increasing loss checks every **16 steps**. Existing PPO/DCPG, rewards,
opponents, branch probability and completed IL identity are preserved.
The linked guide includes the exact g660 recovery/config/continuation command.

## Archived g650 PPG continuation

Use the [dense eight-iteration PPG instructions](../ai-research/PPG_DENSE_EIGHT_G650.md)
for the current v11.2 run: eight cached/held-out ages, a phase every eight
iterations, last-accepted-loss selection, ordered multiprocess preparation and
**0.5% branching per eligible decision**. PPO feedback cuts use **LR / 1.25**;
PPO stopping uses only the **2,048-row exact categorical KL probe**, with sampled
KL retained as a diagnostic.
The reviewed g650 continuation starts its first new phase at g658 and preserves
the completed IL folder. The guide includes the interrupted-collection recovery
and exact config/continuation commands.

## Features 2: fresh IL and a new self-play run

The current encoder/model uses `permitted-features/2.0.0`, width 304 and an explicit
3M parameter cap. Follow [FEATURE_ENGINEERING_LAB.md](FEATURE_ENGINEERING_LAB.md)
to audit/re-encode heuristic demonstrations, train fresh IL, bind its new config
and resume a new-schema self-play run. Old-schema checkpoint continuation commands
below apply only with their original pinned runtime; source-review flags cannot
convert those weights/optimizer states into the new input layout.

## 14 October ordinary learner exploration experiment

The current `joint7-latest.json` preset and `latest_model_config` helper now use
**0.08 exploration** for the current learner in ordinary training. All four
latest-model self-play seats share this setting and contribute eligible data to
the same learner. Historical/v4 tables use it only at their one learner seat.
Frozen historical opponents retain **0.05**, teacher rollout policies retain
their existing settings, and evaluation neural policies retain **0**. The v4
policies and their search budgets are unchanged.

The existing mixture is preserved:

```text
behavior_probability = 0.92 * policy_probability + 0.08 * exploration_distribution
```

The exploration distribution still groups legal choices by action family and
build/develop industry, with the existing context weighting. It is not uniform
over individual legal actions. Temperature remains **1.0**, entropy weight
**0.005**, value weight **0.5**, and actions are sampled probabilistically. No
schedule, opening temperature, Dirichlet noise, new starting positions or
Dropout was added. The latest-model curriculum, four-seat learner collection,
disabled gate, reward/insolvency, teacher clocks/cadence/quality, null cumulative
teacher allowance, population and retention settings remain intact.

`recovery/ordinary.py` reports the exploration/temperature/policy identity in
ordinary job provenance and `sampling.behavior`. Existing compact history retains
this report. Neural policy identities already include exploration; collection
and PPO reconstruction both use the same configured value and family/context
distribution. Each sampled action still records its actual mixture probability.
The strict likelihood verification, masking and forced-action filtering are
unchanged. A 0.05 collection is not relabeled as 0.08. Collection cache identities
bind policies/settings/provenance, and a pending iteration from the previous
source/configuration must be resolved through existing recovery rules before
the continuation can begin.

Continue the existing latest-model run from its committed checkpoint:

```bash
cd /data/not_backed_up/yxu209/ai-training
python scripts/step5.py --config configs/step5/joint7-latest.json train \
  --run artifacts/step5/v4-selfplay-joint-latest \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --iterations 20
```

`--resume-reviewed-source` is required on the first reviewed source/configuration
change. Once this run commits the new source and 0.08 setting, ordinary same-source
resumes can omit it. Keep the existing run directory; this change does not reset
weights, Adam/RNG, iteration/game/seed counters, population, checkpoint lineage
or the persisted zero-to-100,000 curriculum window. For an explicitly configured
run using another copied preset, change only its `exploration` field from 0.05
to 0.08 and use the same reviewed continuation flag. Older preset files retain
their explicit 0.05 setting.

The current reviewed source is
`ab5e74f8d374c9a348d1949c0d7e3361b39f17c44e0edbe9c7caac241bda5a2d`.
Update source and both learning source registries together. All **2,017**
preceding expanded approvals are preserved; **121** exact routes were appended,
including the preceding source to current source with only 0.05 → 0.08 changed,
and the same change for a run already on this source. Config changes beyond the
pinned review still fail validation. IL/data/model compatibility is unchanged.

Files changed in this adjustment:

- `configs/step5/joint7-latest.json` and `src/brass_ai/recovery/config.py`:
  learner exploration default/configuration generation.
- `src/brass_ai/recovery/ordinary.py`: explicit behavior settings in provenance
  and sampling reports.
- `configs/learning/selfplay-resume-sources.json` and
  `configs/learning/imitation-handoff-sources.json`: pinned continuation/handoff
  source reviews.
- `tests/Recovery/test_exploration08.py`,
  `tests/Recovery/smoke_exploration.py`, and
  `tests/Recovery/test_latest_model_recipe.py`: focused coverage and bounded
  cross-source native smoke.
- This training note.

Validation: **26 focused tests passed** in 15.700 seconds. Tests cover normalized
0.08 mixtures, zero masked probability, preserved family/industry/context mass,
4/1/1 learner attribution, historical opponent exploration, real likelihood
reconstruction before PPO, stale/relabelled behavior rejection, cache identity
and stale-shard rejection, exact reviewed continuation and pending-iteration
refusal. Weights, Adam, CPU RNG, counters, population, lineage and curriculum
were checked across a committed generation-75 / 24,000-game transition and
restart. A follow-up assertion confirms that changing ordinary exploration does
not change the derived evaluation configuration.

Evidence is under `artifacts/recovery/exploration08-review/`. The native smoke
uses a separate directory and authored small IL-format fixture weights, with
CPU inference and reduced actor concurrency. Rehearsal, diagnostic panels and
retention are disabled only in the fixture; teacher readiness/cadence and
ordinary v4 clocks remain active. No production checkpoint is modified.

The cross-source native smoke passed: three games at 0.05 on the preceding
source, then a reviewed continuation for three games at 0.08 on the new source.
The stages took **84.109 / 145.218 seconds** (**229.327 seconds** combined).
At continuation admission, the incoming weights, Adam state, CPU RNG, counters,
population, lineage and curriculum matched the prior commit. The old checkpoint
was byte-identical afterward. Progress advanced normally from iteration 1 / 3
ordinary games to iteration 2 / 6 ordinary games. No gate or teacher collection
ran. The existing utility freshness rule required three calibration steps on
the new ordinary data after the source change.

The 0.08 collection had **one self-play table and two v4 tables**, producing six
learner trajectories. It supplied **491 learner decisions**, including **278
forced decisions** excluded from actor updates. PPO used all **213 eligible
actor decisions**, with **426 presentations**, **491 distinct critic decisions**,
**two PPO steps**, final KL **0.000002404**, and no early stop. Historical-seat
0.08/0.05 attribution is covered by the planner and real-likelihood focused
tests; the small native continuation did not include a historical table.

Before calibration/PPO, mean mixture entropy over all **213 nonforced ordinary
learner decisions** was **3.176290 nats**. Recomputing the same model, legal
choices and contexts at 0.05 gave **3.176810 nats**. Raising the exploration
coefficient does not necessarily increase entropy because the exploration
distribution is family/context weighted. Maximum error between recorded and
reconstructed selected-action probability was **5.96e-8**.

| Sampled action family | Count | Fraction of 213 nonforced learner decisions |
| --- | --- | --- |
| Build | 18 | 8.45% |
| Develop | 65 | 30.52% |
| Finish | 3 | 1.41% |
| Liquidate | 2 | 0.94% |
| Loan | 21 | 9.86% |
| Network | 21 | 9.86% |
| Pass | 62 | 29.11% |
| Scout | 20 | 9.39% |
| Sell | 1 | 0.47% |

The 83 sampled build/develop choices were CottonMill 19, CoalMine 20, IronWorks
13, Manufacturer 16, Brewery 15 and Pottery 0. These counts describe selected
actions in six test player trajectories, including continuations; they are not
root-only, per-industry coverage guarantees, independent samples or a strength
panel. The 0.05 baseline used different games and earlier weights, so its
entropy/frequencies are not a controlled before/after strength comparison.
The exact same-position comparison above isolates only the mixture coefficient.
Raw receipts are in `native/report.json`, with the audit in `review.json`.

The larger mixture coefficient adds no forward pass, search worker or optimizer
step. Runtime depends on the game paths generated by the new behavior. This is
a modest experiment; entropy and action-frequency observations alone do not
demonstrate improved playing strength.

## 14 October fresh joint-7 latest-model recipe

`configs/step5/joint7-latest.json` is the new default for `train`. It starts a
separate run from completed joint-7 imitation; it cannot import G160/G179 PPO
state or convert those runs, even with `--resume-reviewed-source`. Explicit
older presets and their reviewed continuation routes remain available.

| Completed ordinary games | Latest-model tables | Historical tables | v4 tables |
| --- | --- | --- | --- |
| 0–19,999 | 15% | 25% | 60% |
| 20,000–99,999 | 45% | 25% | 30% |
| 100,000 onward | 60% | 25% | 15% |

These quotas refer to tables. Scheduling uses each game's absolute ordinary
index, including iterations crossing 20,000 or 100,000. Teacher/evaluation
games, trajectories and optimizer steps never advance this clock. The persisted
v4 window starts at zero and ends at 100,000; validation prevents an older boost
schedule from overriding the base curriculum or resetting on resume.

Every self-play table uses the same frozen current-learner snapshot at all four
seats, with all four eligible trajectories updating one shared learner. The next
collection registers the updated snapshot. Historical and v4 tables retain one
learner seat; frozen opponent decisions are excluded. Seat rotation, exploration,
contexts, behavior-probability checks and forced-action handling are preserved.
At 320 tables wholly within the final phase, the counts are 192/80/48 tables and
768/80/48 learner trajectories: 896 trajectories before decision filtering.

This recipe does not initialize or schedule champion gates, promote champions,
mix champion opponents or publish gate-selected `best.pt`. Regular checkpoints
and latest tracking continue. Monitor/broader panels remain diagnostic. Recipe
validation and resume guards reject champion configuration/state. Older explicit
gate recipes keep their behavior. The helper still applies the unrelated
insolvency reward migration.

The production configuration was derived from the current joint-7 preset, not
an archived run configuration. PPO, reward coefficients, insolvency
`event_cost=0.005` / `max_cost=0.075`, opponent search, population capacity 64,
retention, and evaluation schedules are unchanged. Teacher settings are the
requested **12/12/16/24 seconds** for early/late Canal and early/late Rail,
five games, eight distillation steps, the existing 160-game cadence/readiness,
phase reservations and quality/freshness/reuse/calibration guards.
`search_wall_fraction: null` disables only the cumulative teacher allowance;
shared per-turn deadlines, cancellation and failure timeouts remain active.
Explicit older recipes keep their previous clock migration behavior.

Resolved original IL lineage:

- Run: `artifacts/step5/v4-batch7-joint`.
- Selected checkpoint: `artifacts/step5/v4-batch7-joint/epoch-0006.pt`.
- SHA-256: `29e7577bc374321f54d99b663800c4ccf762659ef0c02cf08a2eb8c1817317cf`.
- Model identity: `ea1d9e4da5820a9bfa60647570d498ecf705dacceeb58221c87a77bbcbb26a53`.
- Manifest: `artifacts/il-replay/v4-batch7-joint/manifest.json`.
- Manifest SHA-256: `7da7720287d7de4ccaae486c70f8f2f4696ae7b409d3b821580d30a8674e954e`.
- IL signature: `08ab674205d8603ebf1827009809a1a3479e00b85a716b521d36e4c64cb101c4`.

G150/G179 exported lineage and both IL completion records agree on this origin.
The actual checkpoint/dataset are absent from the local checkout; their binary
verification must occur on the lab machine. Startup uses the existing completed
imitation validator and checks the pinned origin before creating a learner
checkpoint. Missing, incomplete or changed IL fails without random/checkpoint
substitution. IL weights transfer into a fresh PPO optimizer, zero counters and
a new population rooted at the IL anchor; existing reward-head adaptation and
fresh utility calibration remain active. Subsequent resumes retain this run's
weights, Adam, RNG, scheduling, population and lineage.

Start from the lab repository; this command runs 20 iterations from zero:

```bash
cd /data/not_backed_up/yxu209/ai-training
python scripts/step5.py --config configs/step5/joint7-latest.json train \
  --run artifacts/step5/v4-selfplay-joint-latest \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --fresh --iterations 20
```

The production directory is
`/data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-selfplay-joint-latest`.
It has not been created or trained by local validation. `--fresh` refuses an
existing state; the ordinary initialization path also rejects a nonempty run
directory. Resume this new run later with another 20 iterations:

```bash
cd /data/not_backed_up/yxu209/ai-training
python scripts/step5.py --config configs/step5/joint7-latest.json train \
  --run artifacts/step5/v4-selfplay-joint-latest \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --iterations 20
```

`--expected-generation` is optional: it asserts the committed starting generation
and does not select a checkpoint. No `--resume-reviewed-source` is needed when
resuming with this same source/configuration. Future code changes still require
the existing exact source-review process.

Iteration reports and compact history retain actual table counts, learner
trajectories, all/eligible/forced decisions, and distinct PPO actor/critic samples
used per table type. `actor.sample_usage` also records repeated presentations
and unused eligible decisions, including KL-stopped batches. Calibration,
rehearsal and teacher updates are excluded from these PPO sample counts. Table,
trajectory and actual PPO sample fractions are separate fields.

At fixed 320-table collections, the three phases produce 464, 752 and 896 learner
trajectories before filtering. Four-seat sampling increases replay preparation,
inference and PPO work; unchanged batching and KL stops determine actual work.
The longer 30% v4 phase and permanent 15% v4 quota retain more searched opponents
than the earlier 5% mature recipe. Gate evaluation cost disappears. No
proportional throughput change or playing-strength improvement is established.

The reviewed source for that preceding recipe revision was
`f71f3b85f1b68a4d128d57dbc85ff0f61526385804029f71078e869ce594911f`.
Update source and both `configs/learning/*-sources.json` registries together.
Learning compatibility and all 1,899 preceding expanded continuation approvals
are preserved; 118 routes were added for the unchanged older recipes. These
approvals do not permit importing old PPO state into the new recipe.

Validation: **151 distinct focused tests passed** across the recipe and
compatibility checks. The 79-test primary run passed in 52.242 seconds. An older
CLI placement test still expected `lab.json`; it now asserts the new default,
and all six tests in that affected module passed on rerun. Coverage includes
both curriculum boundaries/crossing batches, persisted progress, actual
collection planning and seat attribution, behavior verification, real PPO sample
usage and KL stopping, completed-IL handoff, fresh optimizer/zero counters,
gate-free startup/resume, teacher timing/null allowance and existing exact
Adam/RNG/source-review safeguards. All 24 existing Step 5 JSON configurations
remain unchanged. Local evidence is under
`artifacts/recovery/latest-model-recipe-review/`; `review.json` records test
coverage, the resolved assertion and the configuration/lineage audit.

The bounded native smoke completed **six games in 241.078 seconds**, in a
separate directory with a small authored IL-format fixture, CPU inference and
reduced actor concurrency. It ran one three-game iteration, restarted from its
checkpoint, then ran another three. These are functional test weights, not the
trained joint-7 checkpoint. Fixture rehearsal, scheduled diagnostic panels and
retention cleanup were disabled; production settings are unchanged. Ordinary
v4 search retained its one-second shared
turn budget and four workers. The two iterations used snapshots with the
expected first-output/second-input model identity, and resume restored Adam and
CPU RNG exactly. No gate function was called; the prior checkpoint remained
byte-identical and progress advanced from 3 to 6 ordinary games.

| Smoke table type | Actual tables | Learner trajectories | Eligible actor decisions | Distinct actor decisions used |
| --- | --- | --- | --- | --- |
| Latest-model self-play | 2 | 8 | 278 | 278 |
| Historical | 1 | 1 | 32 | 32 |
| v4 | 3 | 3 | 119 | 119 |
| Total | 6 | 12 | 429 | 429 |

The small smoke spans only the first six table slots, not a complete quota
block. Its table fractions were 33.3% / 16.7% / 50.0%, trajectory fractions
66.7% / 8.3% / 25.0%, and actor sample fractions 64.8% / 7.5% / 27.7%.
All **968 learner decisions** entered the critic dataset, including **539
forced decisions** excluded from actor updates. Two epochs per iteration gave
**858 actor presentations**, **1,936 critic presentations**, **four PPO steps**,
and zero unused eligible actor decisions. Final KL was **0.000004585** and
**0.000003014** for the two iterations; neither stopped early at target 0.01.
A separate focused test forces a KL stop and verifies unused-sample reporting.
Fresh utility-head calibration took three separate steps on the first
collection; resume reused its calibration receipt without repeating them.

The smoke did not reach the unchanged teacher readiness threshold, so it ran
zero teacher collections/distillation updates and does not measure teacher
policy drift or label yield. Teacher timing and null wall-limit handling are
covered by focused tests. No production iterations, GPU performance benchmark,
or playing-strength evaluation was run. See `native/report.json` for the raw
sampling/update receipts and `preservation.json` for the configuration audit.

## 14 October follow-up self-play correctness review

The follow-up review reproduced and repaired three additional failure cases:

- Local evaluation accepted a utility budget above the frozen model's reward
  cap while shared inference rejected it. Both paths now use one strict
  per-model validator; valid predictions and older four-head models retain
  their behavior.
- Multi-device inference could replace the original replica error with a later
  broken pipe or generic router error. A shared first-failure record now
  preserves the cause. The replica records that cause before exposing its
  local error to the router, closing a reproduced publication race.
- Failed-iteration archival checked the population, champion and evaluation
  anchor but omitted the rollout's separately retained previous champion.
  It now verifies both rollout anchor/previous checkpoint hashes before any
  archival mutation. Missing or changed required opponents leave the partial
  iteration and committed state untouched.

The reviewed source for that preceding recipe was
`b24eef15976abc2895afebbf6191fa465bec85ad8ea49568f324f528ecaddaab`.
Update source and **both** `configs/learning/*-sources.json` registries together.
All 1,782 preceding expanded approvals remain intact; 117 exact continuation
and manual champion-cap routes were added. Learning compatibility and all 19
Step 5 presets are unchanged. This preserves insolvency **0.005/0.075**, teacher
clocks **12/16/20/20 seconds**, five teacher games, the 160-game cadence, eight
steps, `search_wall_fraction: null`, PPO/reward/gate/opponent settings, staged
champion exposure and the 64-model capacity. The approved 47-model restored
pool continues growing normally. Checkpoints, Adam/RNG, committed counters and
lineage are preserved.

Validation: **164 distinct focused tests passed**. Affected inference and
source/resume tests were rerun after the final race fix. Tests cover inline and
shared transports, real queued inference through two CPU-backed replicas,
original-error attribution during a concurrent router failure, ordinary model
compatibility and cancellation, rollout/checkpoint retention, phase budgets,
label/source guards, and an actual generation-160 checkpoint/Adam/RNG retry.
Population tests verify growth from 47 to 64 across restarts. Two stale tests
still expected the old 0.025 reward cap; they now retain historical coverage and
also verify the requested 0.075 cap, including fifteen insolvency events.

The broad exploratory suite was stopped in the slow many-commit retention
performance test; a complete suite pass is not claimed. CUDA hardware and the
remote lab were not exercised. The multi-device tests validate routing/error
behavior using CPU replicas, not multi-GPU performance.

The final native smoke used four workers, a small calibrated test model, frozen
0.025-cap opponents, and one observed two-action turn per phase:

| Phase | Search attempts/completions | Simulation attempts/completions | Qualified labels | Visited/legal root actions | Insufficient coverage | Search / turn seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Early Canal | 1/1 | 16/8 | 1 | 4/328 | 0 | 11.844 / 12.032 |
| Late Canal | 1/1 | 24/16 | 1 | 5/151 | 0 | 15.891 / 16.266 |
| Early Rail | 1/1 | 30/23 | 1 | 7/55 | 0 | 19.890 / 20.078 |
| Late Rail | 1/1 | 32/24 | 1 | 3/3 | 0 | 19.938 / 20.203 |

The probe took **92.703 seconds** including setup/calibration and exercised
833 frozen-opponent requests. No fatal inference error, CPU-admission abort,
teacher failure or insufficient-coverage rejection occurred. There were
4 ordinary deadline-expired inference requests. Search time remains shared
across workers, both actions and continuations; turn timing also includes
retirement/action application. The 4 qualified diagnostic labels produced
**0 distinct training samples used**. No distillation update occurred, so
teacher policy drift is **not measured**, rather than zero.

The smoke helper now records legitimate label-rejection counts instead of
asserting that every searched turn must produce a label before saving its
report. An earlier probe hit that overly strict assertion; its exact rejection
reason was not retained. The trainer's label-acceptance criteria are unchanged.
The successful final probe validates execution, not improved yield, training
stability or playing strength. This fix adds no teacher search time; the current
collection ceiling remains 1,360 shared teacher turn-seconds.

Machine-readable evidence and test logs are under
`artifacts/recovery/selfplay-followup-review/`, with the final probe in
`native-final/report.json` and the review summary in `review.json`.

Use the safe interrupted-iteration continuation command in the next section
if the lab is still committed at generation 160. Its expected-generation guard
rejects a newer committed run. Archive only the failed partial iteration; do not
rerun the generation-160 restore or replace the current population.

## 14 October teacher crash after the generation-160 restore

The reported attempt completed 320 ordinary games and replay preparation, then
failed in the first five-game teacher collection due at 51,360 ordinary games.
`CPU admission aborted` is a secondary failure: inference aborts the shared CPU
admission queue, and another actor can report that abort before the original
inference error reaches the parent. The pasted traceback alone does not identify
the original error.

A local reproduction using the actual search and inference service identifies
a defect consistent with this restore: the learner's new `max_cost=0.075`
ledger was also passed to frozen historical opponents whose utility heads still
use `max_cost=0.025`. The strict inference check rejects that request with
`Invalid inference utility budget`, then aborts shared admission. Both inline
and shared-memory inference reproduce the failure before the fix.

Search now requests only the frozen opponent's action probabilities with that
model's own default utility context. Those probabilities do not depend on the
remaining insolvency budget; opponent utility outputs never supply search leaf
values. Learner root/frontier evaluations retain the current ledger and exact
reward-ID checks. Frozen opponent weights/configs and inference validation are
unchanged. Stale policy-only request budgets are cleared. The parent collector
also surfaces the original service error, and later broken replies cannot
overwrite it; cancellation and failure aborts remain active.

The initial reviewed source for this fix was
`0733832dde1cd773a3a593a4c49fa74c3a18cb4338508d31aae2a8c2ca8b1d79`;
the follow-up review above supersedes it.
Update source and **both** `configs/learning/*-sources.json` registries together.
All Step 5 preset contents are unchanged. This includes the 12/16/20/20-second
teacher clocks, five games per collection, 160-game cadence, eight distillation
steps, `search_wall_fraction: null`, insolvency 0.005/0.075, and staged champion
opponents. Population capacity remains 64; the approved 47-model restore adds
new distinct generations normally (52 after five commits, 64 after seventeen,
assuming distinct new models).

Validation: **105 focused tests passed**. Coverage includes mixed-cap frozen
opponents, unchanged leaf/inference guards, original-error reporting, phase
clocks and shared deadlines, source reviews, 47-to-64 population growth, and an
actual checkpoint/Adam/RNG retry through the original 0.025-cap generation 160,
its already-recorded 0.075 source transition, partial-iteration archival and the
new source. That retry preserves the rollout and all progress counters.

Bounded native validation used four search workers, a small calibrated test
model, frozen 0.025-cap opponents and one observed two-action turn per phase:

| Phase | Search attempts/completions | Simulation attempts/completions | Qualified labels | Visited/legal root actions | Insufficient coverage | Search seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Early Canal | 1/1 | 16/8 | 1 | 4/328 | 0 | 11.890 |
| Late Canal | 1/1 | 24/16 | 1 | 5/151 | 0 | 15.828 |
| Early Rail | 1/1 | 40/32 | 1 | 9/55 | 0 | 19.875 |
| Late Rail | 1/1 | 40/32 | 1 | 3/3 | 0 | 19.860 |

The full local probe took 90.593 seconds including setup/calibration. All 998
frozen-opponent requests succeeded; no inference abort or teacher failure was
recorded. The four qualified labels are diagnostic partial-turn evidence:
**zero labels entered distillation**, and teacher policy drift was not measured
because there was no policy update. This verifies the repaired execution path,
not an improvement in label yield, training stability or playing strength. The
training budget is unchanged: up to 1,360 teacher turn-seconds per collection,
shared across workers/actions/continuations; elapsed lab time depends on actual
parallelism and search completion. Details are in
`artifacts/recovery/teacher-opponent-cap-fix/native/report.json`.

For the already-restored lab run, archive only its failed uncommitted iteration
and retry from the committed generation 160. Run from the repository directory:

```bash
python scripts/recover-step5-iteration.py \
  --root /data/not_backed_up/yxu209/ai-training \
  --run artifacts/step5/v4-selfplay-joint \
  --expected-iteration 160 --apply && \
python scripts/step5.py --config configs/step5/joint7-g150-gate72.json train \
  --run artifacts/step5/v4-selfplay-joint \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --expected-generation 160 --iterations 1
```

The archive command validates the committed checkpoint/Adam, source lineage and
live opponents, then preserves the partial replays under `.interrupted-iterations/`.
It leaves state/checkpoint bytes unchanged. The 320 ordinary games from the
failed attempt must be recollected under the new source, followed by fresh
utility calibration and teacher collection. Do not rerun `joint7_restart160.py`:
the requested restore has already happened. No lab files were accessed or
training launched during this local fix.

## 14 October teacher phase-clock revision

The current teacher defaults use one shared budget per reserved turn:

| Phase | Reserved round range | Seconds |
| --- | --- | --- |
| Early Canal | Canal 1â€“4 | 12 |
| Late Canal | Canal 5â€“8 | 16 |
| Early Rail | Rail 1â€“6 | 20 |
| Late Rail | Rail 7â€“8 | 20 |

Workers, both actions and continuation choices share that turn's deadline.
Cancellation, failure timeouts and label-quality requirements still apply.
Five games with four seats and one reservation per phase allocate up to
**1,360 teacher turn-seconds per collection**, versus 1,280 previously (6.25%
more). Parallel collection and actual search completion determine elapsed wall
time; this is a budget estimate, not a measured runtime or strength improvement.
Collection-cost estimates and the maximum failure bound read the same phase map.

The reviewed `step5-phase-teacher-time/1.2.0` migration adopts these clocks from
the exact preceding default. Earlier migration receipts remain reproducible;
custom clocks are preserved. Update source, presets and both source registries
together, then use the generation-160 continuation command below. Checkpoint,
Adam/RNG, progress, population and lineage are retained. Teacher games/cadence,
distillation, `search_wall_fraction: null`, the staged champion rollout and
insolvency `event_cost=0.005, max_cost=0.075` retain their current settings.

Validation: **77 focused tests passed**, including shared worker/action deadlines,
collection accounting, disabled cumulative admission, historical receipt checks,
and exact checkpoint/Adam/RNG continuation. The exported generation-160 restore
plan also accepts the new configuration; no lab restore or training was run.

## 14 October champion rollout and generation-160 restart

Current defaults replace the previous three-incumbent self-play lineup. Only
self-play tables change; the mature **70/25/5 table composition**, seat rotation,
learner-only PPO attribution, immediate champion gate/promotion and `best.pt`
publication remain intact. Each champion's exposure uses frozen checkpoints:

| Committed ordinary games since introduction/promotion | Self-play opponents |
| --- | --- |
| 0â€“1,599 | One champion, one previous/distinct older model, fixed IL anchor |
| 1,600â€“3,199 | Half of tables have one champion, half two; every table keeps the IL anchor |
| 3,200 onward | Two champions and the fixed IL anchor |

These durations are five configured iterations at 320 games each. Both the
origin and duration are persisted. Introducing the feature into an existing run
starts at its current committed progress. Future accepted promotions reset that
origin; failed gates and restarts do not. Transition tables use deterministic
shuffled pairs, one of each lineup per pair, with the pair cursor committed
alongside normal training counters. An odd collection carries the unmatched
table into the next collection. No phase uses three champion opponents.

The anchor remains the initial/IL evaluation model. Additional older slots
prefer a distinct previous champion, then older population models; limited
populations can duplicate the anchor. Before the first distinct promotion, the
champion itself is the initial anchor. That bootstrap case explicitly uses one
copy plus two fixed admitted search-free v4 opponents, avoiding three identical
champion copies. The report records this fallback. Anchor and previous-champion
checkpoints remain retention dependencies even outside the population.

Set `champion_opponents.max_champion_opponents` to **1** to stay at one champion
opponent. Continue with `--resume-reviewed-source` after a committed iteration;
the exact same-source one/two-cap switches are explicitly reviewed. There are
no automatic rollback thresholds. Changing this option preserves the original
rollout clock. Other configuration changes still require their own review.

The insolvency cost is now **0.005 per event, capped at 0.075** (15 events).
The narrow reviewed 0.025â†’0.075 migration preserves all model tensors, Adam
moments/steps and RNG. The utility evaluator must recalibrate on fresh ordinary
games before teaching. Teacher clocks follow the phase table above. The PPO
settings, gate, opponent search, five teacher games, collection cadence, eight distillation
steps and **`search_wall_fraction: null`** remain unchanged. The extra opponent
models may add inference cache/batch overhead; no additional teacher or gate
games are scheduled. The first resumed iteration includes utility recalibration.

`selfplay_opponents` in each full and compact report records stage, origin,
duration, elapsed ordinary games, anchor identity, actual one/two-champion table
counts, sampled generations and fallback counts. At 320 games per iteration,
the expected self-play counts are 224/0, then 112/112, then 0/224. These are
implementation checks, not evidence of improved economic stability or strength.

For the specifically reviewed generation-179 joint-7 run, restore generation
160 before continuing. The tool cross-checks deterministic seed/context/teacher
counters against the separately saved generation-170 state. Strict mode requires the
**original complete generation-0160.pt** with Adam/RNG plus all 64 target
population/anchor models. JSON reports and weight-only `best.pt` are insufficient.
It validates files before staging, archives the entire later run, and publishes
the restored files at the original run path without rewriting checkpoint bytes.
The archived directory is evidence/rollback storage; its checkpoint run bindings
still name the original path. No generation-160 partial replay is reused.

The reported restore failure omitted 17 historical opponents, at generations
23, 24, 33, 34, 75, 115, 146â€“151 and 153â€“157. It did **not** report the full
generation-160 checkpoint as missing. `history` and `retention-history.json`
retain metadata and metrics, not deleted model parameters. With the previous
zero-hour quarantine, retired unpinned opponents can already be deleted.

The user-approved **`--allow-reduced-population`** option permits only those
reported missing opponents. It keeps at least **47 distinct verified models**,
including the initial anchor and generation-160 champion, and still requires the
complete learner checkpoint. Missing required files, extra missing generations,
future models and invalid model identities remain errors. The original later
run is archived before replacement; no future models fill the missing slots.
The dropped identities and retained counts are recorded in `restart160.json`
and `state.json` under `restart_origin.population_recovery`.

Population capacity stays **64: 16 recent + 48 historical/anchor models**.
Normal committed new models refill the pool without dropping available models
while it is below capacity. Starting at 47, five distinct new commits give 52;
17 give 64 (generation 177). Restarting does not reset this growth. Repeated
identical models do not count as new entries. Once full, normal retention resumes.

From the lab repository, inspect first (no training or run replacement):

```bash
python scripts/joint7_restart160.py --allow-reduced-population
```

If retention removed a required file, supply `--backup-run /path/to/old-run-backup`
and/or `--checkpoint-backup /path/to/generation-0160.pt`. A missing or mismatched
learner checkpoint stops recovery; optimizer state is never fabricated. Omit
`--allow-reduced-population` to require all 64 original opponents. For the
approved 47-model recovery, use:

```bash
python scripts/joint7_restart160.py --allow-reduced-population --apply && \
python scripts/step5.py --config configs/step5/joint7-g150-gate72.json train \
  --run artifacts/step5/v4-selfplay-joint \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --expected-generation 160 --iterations 5
```

Repeat any needed backup flags on the `--apply` command. After those five
iterations, continue normally without `--expected-generation 160` or the restore
command. The generation check prevents accidentally continuing from 179. The
local exported-state audit is `results/reviews/joint7-g0179/restart160-plan.json`;
it explicitly records `applied: false` because lab checkpoint files are not
available in the local workspace. No lab training was launched for this revision.

Reduced-pool validation passed **nine focused tests**, including a real 47-model checkpoint restoration with
exact learner bytes, Adam/RNG and archived future evidence preserved. A bounded
commit simulation verifies 47â†’52â†’64 growth across state reloads, anchor retention,
and the hard 64-model capacity; this is not a playing-strength measurement.

Validation: focused tests cover boundaries, deterministic allocation, promotion,
retention, resume and exact Adam/RNG preservation. The bounded CPU smoke executed
11 real games in about 208 seconds: four ordinary games over two resumed PPO
iterations, six games spanning all three exposure stages, and one gate-format
game that cannot promote. No full evaluation campaign or strength comparison was
run. Reproduce with `tests/Recovery/smoke_champion.py --output <new-artifact-dir>`
using the repository's normal Python dependencies.

**14 October explicit container-check revert:** [current behavior, measurements and continuation receipts](STEP5_CONTAINER_CHECK_REVERT_2026-10-14.md). The extra recursive scans from the [performance correctness audit](STEP5_PERFORMANCE_AUDIT_2026-10-14.md) are removed at the user's request. The known JSON-equivalent list/tuple cache-validation gap is accepted; schema validators, content hashes, attribution, finite-value checks and deadlines remain. Update source and both review registries together; training settings and search clocks are unchanged.

**14 October CPU performance update:** [measurements, benchmark commands and exact continuation](STEP5_PERFORMANCE_2026-10-14.md). Request snapshots and public-history validation do less repeated Python work; reports now separate teacher collection/distillation and expose total iteration time. Search clocks and all training settings remain unchanged. Local CPU savings are measured; a complete single-GPU training speedup is not established.

**14 October teacher crash fix:** [deadline-race repair and generation 170 recovery commands](INFERENCE_DEADLINE_RECOVERY_2026-10-14.md). `Request no longer live` at the final inference gate now retires search normally. The explicit recovery tool preserves unfinished iteration evidence before retrying from the committed checkpoint. Current training settings and search clocks are unchanged.

**13 October self-play review:** [remaining bugs, repairs and continuation evidence](SELFPLAY_MAJOR_REVIEW_2026-10-13.md). Five confirmed issues are repaired in replay integrity, calibration cleanup, worker retirement, actor cache identity and population checkpoint reuse. Current search clocks and all Step 5 configurations are unchanged. Update source and review receipts together, then continue a committed run with its existing config and `--resume-reviewed-source`.

**13 October performance update:** [measurements and continuation instructions](STEP5_PERFORMANCE_2026-10-13.md). Unchanged industry mats reuse validated content instead of rechecking every tile after cash/hand-count changes. Current configurations and search clocks are unchanged: teacher phases remain 12/12/16/24 seconds and v4 opponents one second per whole turn. Exact source receipts preserve checkpoint/Adam/RNG/progress continuation. Old derived caches require normal source revalidation; no lab-GPU or complete-iteration speedup is claimed from the local CPU measurements.

Recipe `step5-training/2.2.0` keeps the **1,840,109-parameter MLP-ResNet** for imitation, then extends it to **1,850,129 parameters** for PPO and **Masked ISMCTS + NN** with a shared combined-utility head. Existing completed IL and reviewed self-play checkpoints can continue through the versioned extension described below. The source-based six-style revision and comparison requirements are documented in [the opening-book note](../ai-research/TOURNAMENT_STYLE_V3.md). The user selected the previous advanced bank, `bank-3dc65253a610191d66a5`, as **v4**, with productive-development stall reset and opening-biased candidates. Further strength tests were stopped; this is an explicit selection rather than a tournament promotion. The rejected network/Scout additions are excluded. `training/opponents/current.json` and the distribution both identify that exact bank. The main lab preset is **24 queued heavy tables, four private search workers per decision**, with one shared 60-slot CPU admission gate under your 64-core allocation. The shared turn clock starts after CPU admission.

New imitation data, searched RL opponents and heuristic evaluation use Masked **v4** with **one second per whole turn/four workers**. Active lab presets now apply a one-time recovery schedule: **60% searched-v4 tables for 20,000 ordinary games, then 30% for 10,000 games**. Historical opponents stay at 25%; current self-play occupies 15% then 45%. PPO and NN search share `W + 0.10*P + 0.001*VP - future insolvency costs`; official performance heads remain separate. The schedule begins when a new run is created or when an existing run adopts this reviewed revision, and persists across restarts. Afterward the original mature curriculum resumes at 70% current self-play / 25% historical / 5% searched v4. Internal `v3-*` profile IDs and `training/runtime/v3*` paths remain compatibility aliases for the selected bank. Start a new model from random weights. A completed, fresh standalone imitation run can explicitly hand its selected weights to self-play via `--imitation-run`; its optimizer, population and neural replay are not imported. Audited entirely heuristic datasets may be reused with their original receipts. Same-run continuation from reviewed versions preserves existing PPO progress; see the 9 October restart and incumbent-selection section below.

## Comprehensive bug audit (12 October)

The [comprehensive audit report](COMPREHENSIVE_BUG_AUDIT_2026-10-12.md) documents the host, replay attribution, search cleanup, run-lock, retention and P0/P1 persistence fixes and their verification. The current reviewed Step 5 source transition is `step5-comprehensive-bug-audit/1.0.0`, from `aeef0917b3645bdb...` to `8d19aff88a93edd9...`; previous approved routes retain their exact recipe migrations. Budgets, objective and concurrency settings below are unchanged. After a committed iteration, stop old-source writers and repeat the existing command with the same run/config/origin plus `--resume-reviewed-source`. Finish unfinished old-source PPO or IL work with its original source before updating. Completed compatible IL handoffs remain supported.

Unlike the earlier throughput-only revision below, this audit changes replay/runtime validation source: stale derived caches are ignored, authoritative replay is revalidated and features are recomputed in memory. The reader does not automatically rewrite cache files. Audited heuristic data and learning schemas remain compatible; do not regenerate or relabel datasets merely to bypass a validation failure. The report also gives the exact P0/P1 migration and interrupted-phase recovery procedure. Local CPU fixtures are not lab-GPU or playing-strength qualification.

## Single-GPU throughput defaults (12 October)

Use **one allocated RTX 6000 Pro** for the learner and the shared inference service. The CLI now defaults to `--device auto`: train, imitate, evaluate and preflight select one visible CUDA GPU when available, otherwise CPU. The `scripts/step5.py` entry point restricts visibility to the first job-visible GPU before importing Torch, so checkpoint RNG capture cannot initialize additional idle GPUs. An explicit inference override preserves the caller's device indices and visibility. `--device cuda` remains useful to fail explicitly if the lab GPU is unavailable. Demonstration generation and audit remain CPU jobs. Omit `--inference-devices`; even an explicit single-device list or `auto` with only one visible GPU now uses the direct service, with no multi-device router, second queue, or replica coordination. Multiple GPUs are an explicit override only.

The lab presets keep **24 actor processes Ã— 2 overlapped ordinary hosts**, 24 queued heavy tables, four search workers, the shared **60 CPU work slots / 64 allocated cores**, 16 replay readers, inference batch limit 64, 1 ms flush and 131,072 padded candidate cells. PPO/IL effective batch sizes, microbatches, precision and model architecture are unchanged. These are bounded concurrency limits, not a guarantee of 96 simultaneous search workers. There is no evidence here to justify larger batches or more CPU workers on the lab hardware.

The measured CPU changes apply to shared pipeline code:

- Strict JSON decoding builds valid objects without a Python check/error-string allocation for each key; duplicate keys and non-finite constants are still rejected.
- The dataset/evaluation/self-play writer reuses input validation only for sealed, previously validated games. It verifies every input, label and context seal first and still checks headers, decision order, legal indices and targets. Raw or changed data cannot take this path. Authoritative gzip/JSON and derived-array formats are unchanged.
- The existing reader pool starts while actors play. Previously it started lazily after the first shard arrived: the local two-shard preparation tail was 7.4 seconds although the slowest actual reader task took 1.3 seconds. Startup is now overlapped without a collection barrier or additional processes; actual preparation still uses the shared CPU admission gate.
- Retention resolves repeated dependency paths once per scan. Each new cleanup obtains fresh dependencies, and deletion still uses the existing containment, link and pin checks.

Dynamic batching, shared-memory transport, fused native decisions, revision-safe caches, actor/reader overlap and the existing efficient optimizer batches remain enabled. Search remains **12 seconds per NN whole turn / 1 second per searched-v4 whole turn**, four workers, with **50% cumulative teacher admission**. Rewards, schedules, evaluation cadence and teacher-quality gates are unchanged. The installed game is not part of this update.

**Resume:** this source-only transition is pinned as `step5-single-gpu-throughput/1.0.0`, from `f19424e3d4da14f8...` to `aeef0917b3645bdb...`; previously reviewed routes retain their explicit recipe migrations. After a committed iteration, repeat your existing command with the same run, config and IL origin plus `--resume-reviewed-source`, omitting a previous multi-GPU override. Model weights, Adam state/steps, RNG, game/iteration counters, incumbent/population, opponent-window progress and cumulative search accounting are preserved. Utility calibration is refreshed from fresh ordinary rollouts before teaching; old-source search targets are rejected. Completed IL handoff remains supported. The replay reader, encoder and validator version remain byte-identical, so existing attested feature caches remain valid; no demonstration regeneration or re-audit is required. Finish or recover an unfinished old-source iteration with its original source before migrating.

### Local matched benchmark and lab verification

The before/after comparison used identical seeds, two ordinary PPO games, two existing complete replays (602 decisions) for IO and supervised-kernel fixtures, and three native search turns. It used two tensor threads, one actor process/two overlapping hosts, two replay readers and four search workers. IL used the medium model; the runtime smoke used a small random fixture-calibrated model. These are bounded performance/correctness fixtures, not new training candidates or strength evidence.

| Measurement | Before | Final revision |
|---|---:|---:|
| Complete PPO iteration, two games | 36.531 s | 26.547 s |
| Games/second, including iteration preparation/update/cleanup | 0.0547 | 0.0753 |
| PPO subprocess including startup and shutdown | 53.858 s | 42.779 s |
| Ordinary collection games/second alone | 0.1031 | 0.0938 |
| Replay preparation tail | 6.094 s | 0.875 s |
| Retention cleanup | 9.500 s | 3.187 s |
| Two-game replay publication | 2.580 s | 2.248 s |
| Cached replay load + IL preparation | 1.959 s | 1.756 s |
| One identical supervised epoch | 0.708 s | 0.725 s |
| Qualified teacher labels / full-turn second | 0.0839 | 0.0836 |
| Sampled mean PPO process-tree CPU use | 1.20 cores | 1.38 cores |
| Sampled peak PPO process-tree RSS | 2,008 MiB | 2,124 MiB |

The final iteration was **27% shorter** and the full subprocess **21% shorter**. The model-weight digests after IL and PPO were identical between versions. The ordinary collection sample was **9% slower**, and supervised epoch time was essentially unchanged; this does **not** demonstrate faster native simulation or GPU updates. Laptop timings varied during investigation. The strongest measured improvement is removing the cold reader-startup tail and reducing cleanup/publication work; repeat on the lab before treating these small-sample ratios as sustained throughput.

Both NN turns completed **8/16 attempted simulations** in each version. The teacher emitted one qualified label: 11.922 s before, 11.968 s after. Evaluation emitted none, as required; its full turn took 11.938 s and 12.031 s respectively, including engine applications, while recorded search deadline overrun was zero. The frozen v4 turn returned a legal incumbent with **0/4 completed simulations**, taking 1.266 s and 1.281 s. Its existing cooperative native retirement exceeded one second on this cold laptop case; this optimization does not fix that limitation. No incomplete simulation or fallback was admitted as a teacher label. This turn-only check does not establish complete searched demonstration/evaluation games per second.

**Verification:** 66 distinct targeted checks passed; two CUDA-only checks and the Linux-specific unlink check were skipped on this Windows CPU machine. Checks cover single-GPU selection and optional routing, reader startup/cleanup, strict JSON and mutation rejection, existing cache receipts, parallel IL/PPO input parity, shared deadlines, old-teacher rejection, exact model/Adam/RNG/counter restoration, retention pins and IL handoff. The original two replay receipts still load without another input-schema pass. The reader/encoder/validator bytes and learning-compatibility digest are unchanged; prior source-review entries are preserved. The installed game AI was not modified.

Local Torch was CPU-only: **GPU utilization and peak VRAM were not measurable**. RSS is a 250 ms process-tree sample including shared pages, not an exact allocation count. Earlier reader startup modestly increases resident memory during collection. Local artifacts: `artifacts/recovery/single-gpu-20261012/{before-run,optimized-run}/report.json`; profiles, verification and source-only migration receipts are in the same directory.

For a lab smoke, install `psutil` if missing and create a small JSON fixture containing `{"paths": ["/data/not_backed_up/yxu209/ai-training/artifacts/.../replay.jsonl.gz"]}` using existing complete replay shards. They are only benchmark fixtures; this does not admit neural replay into the imitation dataset. Then run:

```sh
conda activate ai-training
export CUDA_VISIBLE_DEVICES=1  # replace with your allocated physical GPU
export PYTHONPATH="/data/not_backed_up/yxu209/ai-training/src:/data/not_backed_up/yxu209/ai-training/tests/R2:/data/not_backed_up/yxu209/ai-training/tests/Recovery${PYTHONPATH:+:$PYTHONPATH}"
python /data/not_backed_up/yxu209/ai-training/tests/Recovery/benchmark_single_gpu.py \
  --device cuda \
  --fixture /data/not_backed_up/yxu209/ai-training/artifacts/recovery/benchmark-fixture.json \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/recovery/single-gpu-lab
```

Use a fresh output directory for every comparison and hold fixtures, settings, allocation and background load fixed. This smoke checks one-GPU functionality, not full 64-core scaling. On a normal lab iteration, measure games/second and iteration time, phase-qualified teaching samples/second, native CPU/IPC time, encoding/publication, replay tail, inference batch occupancy/queue wait, optimizer time and sampled GPU SM utilization/peak memory. Validate the real CUDA prefetch/update path and the 24-process/60-slot preset there before tuning batch limits or concurrency. The same `--device cuda` placement serves imitation, PPO, teachers and evaluation; demonstration generation remains CPU-bound.

## Focused search-retirement review (12 October)

This review fixes three confirmed issues without changing the training recipe:

- **Cancellation retried work until the deadline.** The time-primary loop could repeatedly start immediately cancelled trajectories because it no longer had a finite attempt quota. It now checks cancellation before each launch and retires pending paths without backing them up. The regression fixture previously launched 1,200 rejected attempts during a simulated 12-second turn; it now launches zero when already cancelled, or retires the one pending path when cancelled during launch.
- **Private workers built discarded teaching targets.** Four replicas each constructed a teacher label, including a full source fingerprint, before the coordinator built the only label used. Replicas now publish completed statistics only. The coordinator still merges, selects the legal plan, fingerprints and validates its target. The four-worker check verifies five label/fingerprint constructions become one with unchanged merged evidence. A source fingerprint cost about 17 ms on this laptop in a short isolated measurement; removing four calls avoids that repeated work, but is not a measured lab speedup.
- **Forest telemetry omitted opponent decisions.** The coordinator now includes the opponent-choice counts already present in merged statistics. This corrects reporting; it does not change rollout policies or search targets.

The source-only review receipt is **`step5-search-retirement-review/1.0.0`**, from `6a4fce0541ce5080...` to `f19424e3d4da14f8...`. Older already-reviewed continuation routes remain supported with their original, explicit configuration migrations. Active budgets remain **12 seconds NN / 1 second searched v4**, four workers and **50% cumulative teacher admission**. Rewards, PPO, opponent windows, evaluation cadence, generated-data formats, IL handoff, batching, shared memory, actor overlap and GPU routing are unchanged. The installed game and frozen native v4 code are untouched.

**Resume:** finish a committed iteration, sync these source and receipt updates to the lab, and repeat your existing command with **`--resume-reviewed-source`**, keeping the same `--run`, config and IL-origin arguments. Do not replace the run with `--initial` or a new directory. The same-run loader preserves weights, Adam moments/steps, RNG, generation/game counters, population, incumbent, search-time debt and the remaining opponent window. Fresh ordinary rollouts refresh utility calibration before teaching; old-source teacher distributions are rejected. An unfinished old-source iteration must be finished/recovered with that source before migration; no cached work is silently deleted or relabelled.

**Verification:** 60 distinct targeted checks passed across deadline/fallback handling, cancellation, the three fixes, model/Adam/RNG continuation, stale-target rejection, IL handoff, PPO/search remaining-return consistency, shared-memory/model attribution, failure cleanup, GPU selection, evaluation resumption and retention grouping. The generation-70 continuation fixture verifies exact tensor/moment equality and unchanged committed checkpoint bytes. Local tests exercise CPU execution and GPU routing contracts; they do not benchmark the lab GPUs.

The small native search smoke used three two-action Canal turns, four workers and a random, fixture-calibrated small network:

| Path | Full turn elapsed | Completed / attempted simulations | Teaching samples emitted |
|---|---:|---:|---:|
| NN searched evaluation | 11.922 s | 8 / 16 | 0 |
| NN phase teacher | 11.906 s | 14 / 21 | 1 |
| Searched v4 | 1.234 s | 0 / 4 | 0 |

Both NN searches published before the shared deadline; coordinator finalization took approximately 32 ms and 16 ms respectively. All turns completed legally and CPU admission returned to zero. The frozen native v4 call took 1.172 seconds on this cold laptop fixture and returned its legal incumbent: cooperative native retirement can still exceed one second, and this sample completed no MCTS trajectories. No clock was reset or quota failure raised. NN teacher throughput was 1.176 completed simulations per full-turn second; this single short run does not establish a sustained speedup or lab throughput regression. Receipts: `artifacts/recovery/step5-review-20261012/native-search/report.json`.

A separate **four-game native PPO smoke** ran two games, saved a checkpoint, resumed through an older reviewed source fixture, then ran two more games. It performed two PPO optimizer steps per iteration, advanced from generation 1/two games to generation 2/four games, refreshed utility calibration, preserved the saved checkpoint and incumbent bytes, and kept the opponent-window progress. Search was disabled in this ordinary-collection fixture; the three-turn check above exercises actual search. Receipts: `artifacts/recovery/step5-review-20261012/native-resume/report.json`. Neither smoke used a trained lab model or changed the installed game.

## Whole-turn time budgets throughout Step 5 (12 October)

This section supersedes the fixed-count recipe and older 90% wall allowances in the historical notes below. **Active Step 5 search stops on one shared whole-turn deadline:** 12 seconds for neural search and one second for searched v4. Faster hardware can complete more useful simulations within that clock. There is no minimum thinking time, artificial wait, per-action reset, worker multiplier or CPU-speed multiplier. The installed game AI and frozen v4 bank are unchanged.

| Active path | Whole-turn search allowance | Workers | Normal stopping condition |
|---|---:|---:|---|
| NN phase teachers, all four phases | 12 seconds | 4 | Common deadline, cancellation, or no useful remaining work |
| Searched NN evaluation | 12 seconds on each scheduled search turn | 4 | Same common deadline |
| Searched v4 RL opponents, new demonstrations and evaluation | 1 second | 4 | Same common deadline |
| Search-free v4, ordinary neural PPO, policy-only monitor/champion panels | No MCTS | Existing allowance | Unchanged bounded incumbent planning or policy inference |

The active lab, smoke, joint-7 and evaluation presets declare versioned time contracts. The legacy field names `teacher_work_budget`, `legacy_opponent_work_budget` and `nn_work_budget` remain for configuration compatibility:

```json
"search_wall_fraction": 0.50,
"teacher_work_budget": {
  "contract": "teacher-turn-time/1.0.0",
  "mode": "time",
  "seconds_per_turn": 12.0
},
"legacy_opponent_work_budget": {
  "contract": "legacy-opponent-turn-time/1.0.0",
  "mode": "time",
  "seconds_per_turn": 1.0
}
```

Searched evaluation uses `nn_work_budget` with `contract: nn-turn-time/1.0.0`, `mode: time` and `seconds_per_turn: 12.0`. Entry points explicitly migrate valid old count configurations before constructing the run signature. Old replay formats and budget receipts remain readable; archived fixed-count constants are available only for explicit comparisons. `search.simulations` is no longer the active NN stopping rule; v4 retains a high emergency guard, not a required quota.

CPU admission precedes observation, encoding and the turn timer. Both actions, all subchoices and all four workers share the same deadline. Search runs at the first nonforced choice; later choices follow the best completed legal plan. Forced choices and debt resolution do not create extra allowances. Unexpected continuation changes use current legal fallbacks. Private workers and in-flight inference/native calls retire before reply channels or CPU slots are reused.

NN launches stop **200 ms before the 12-second deadline** (1.67%); v4 reserves **20 ms** of its one second for publication. These are bounded finishing reserves, not extra time. There is no sleep. Native calls and running inference are cooperatively retired, not forcibly killed: a slow call can exceed the deadline. Reports expose that overrun, and a late NN result cannot teach. On normal clock expiry, previously completed evidence survives without meeting any former simulation quota. Missing evidence, insufficient root/continuation coverage, cancellation, failures and fallback-only results yield no NN teaching label. Failed/insufficient collections consume their allocation without extra retries. Heuristic demonstration actions remain behavior-cloning data, never fabricated NN search distributions.

Teacher games keep their reproducible four reserved phase turns per NN seat. Generic searched evaluations preserve `turn_interval` and `max_searches_per_seat`; their search cadence is unchanged. Search-free paths retain their prior bounds. Warm-up, 160-game teacher cadence, four teacher games, two-collection maximum, eight maximum distillation updates, PPO/reward coefficients, worker counts, opponent schedules and evaluation cadence are preserved.

### Cumulative 50% collection admission

Available allowance remains `max(0, ordinary_seconds * fraction / (1 - fraction) - prior_search_seconds)`. At **0.50** this is **ordinary seconds minus already charged search seconds**. These are the existing accumulated collection accounting categories; they do not independently meter every search inside ordinary opponents or evaluation.

The gate admits a whole collection using its conservative cost estimate, then charges actual collection, distillation and recalibration time before checking another collection. It starts with the four-game estimate (768 seconds plus ordinary-game overhead) and incorporates matching measured collection costs plus 25% headroom. Estimates do not guarantee an exact share for an unexpectedly slow collection. Old counters are never reset. A run upgraded from 90% may defer teaching until ordinary time catches up. Lower user caps remain lower. Changing the search contract invalidates the old recipe-specific cost estimate, not the accumulated time debt.

### Explicit continuation and compatible data

Configuration migration **`step5-whole-turn-time/1.0.0`** is pinned to reviewed source transitions. Repeat the same run, config and IL-origin arguments with `--resume-reviewed-source` after a committed iteration. Compatible model tensors, Adam moments/steps, RNG, generation/game counters, next teacher threshold, historical population, accepted incumbent and remaining v4 schedule all survive. Existing utility calibration is refreshed from ordinary rollouts before new search teaching. Changing unrelated settings still requires its own reviewed transition.

Use a new directory for new demonstrations or manual searched evaluations because their provenance changes. Existing generated data and completed IL handoff remain valid with their original receipts. Fresh PPO collections and current-source teacher distributions are required; old cached search labels are not silently relabelled. Finish or recover an uncommitted old iteration with its original source before migrating.

### Local checks and small comparison

**68 targeted checks passed** for seeded phase scheduling, CPU-admission ordering, shared deadlines, safe retirement/cancellation, legal fallback, quality gates, archived data/work contracts, 50% whole-collection admission, exact checkpoint/Adam/RNG continuation and completed IL handoff. A stale but internally valid old-source teacher is rejected by distillation. Active preset comparisons confirm no changes to PPO/rewards, workers, GPU routing, opponent windows or evaluation cadence.

The six-turn CPU comparison used the same native two-action Canal position, four workers and a small random fixture-calibrated utility model. Each row is one turn; this is a runtime check, not a strength or lab-GPU benchmark. Elapsed turn time includes applying the completed plan and legal subchoices, separately from the search publication deadline.

| Path | Budget mode | Full turn elapsed | Completed / attempted simulations | NN teaching samples emitted |
|---|---|---:|---:|---:|
| Searched NN evaluation | Archived count | 19.41 s | 16 / 16 | 0 (evaluation) |
| Searched NN evaluation | 12 s | 12.13 s | 8 / 16 | 0 (evaluation) |
| NN teacher | Archived count | 30.47 s | 32 / 32 | 1 |
| NN teacher | 12 s | 12.03 s | 8 / 16 | 1 |
| Searched v4 | Archived count | 3.55 s | 16 / 16 | 0 (heuristic) |
| Searched v4 | 1 s | 1.27 s | 0 / 4 | 0 (heuristic) |

NN teacher publication completed within its deadline; merge/label finalization took about 16 ms. The separate timed evaluation search published about 31 ms late and its internal teaching-quality gate rejected it. The cold native v4 search call took 1.19 s and retained a legal incumbent; the remaining time was legal continuation. In-flight retirement can exceed cooperative clocks. No game failed because it missed an old simulation quota, and all six turns completed legally with CPU admission released.

Teacher turn latency fell about 61%, but completed simulations per full-turn second fell from **1.05 to 0.665 (about 37%)**, because eight pending trajectories did not complete by the cutoff. The single qualified label arrived sooner, but this tiny sample does not establish sustained qualified-label throughput or search strength. Evaluation completed-path throughput also fell (0.824 to 0.660/s); cold v4 completed none within the short clock. The lab must measure these tradeoffs on its own CPUs and GPUs. Increasing GPU memory use is not evidence of useful throughput.

Detailed receipts are in `artifacts/recovery/time-budget-20261012/smoke-final/report.json`; targeted output is in the adjacent `targeted-tests.log`. The smoke predates only the final configuration-validation/null-default cleanup, which was covered by the targeted checks; its search algorithm, worker behavior and margins match the final revision.

After continuing **one committed iteration** with the existing command below, compare its `report.json` against the last old iteration: ordinary games/second; search collection elapsed time; attempted/completed simulations; qualified samples by phase and per second; overrun/fallback counts; CPU-admission wait; native simulator/encoding time; inference queue latency, batch size and per-device throughput. Keep the checkpoint origin, allocated cores/GPUs, seeds and lineups recorded. A changed 50% admission decision can mean no teacher collection that iteration; do not reset counters or force extra games to make a comparison. For a direct tiny paired CPU check, run `tests/Recovery/smoke_time_budgets.py` with `src`, `tests/R2` and `tests/Recovery` on `PYTHONPATH`, an installed training runtime and a new `--output` directory under `artifacts`.

### Why fast self-play systems are fast

[AlphaZero's paper](https://arxiv.org/html/1712.01815v1) used 800 simulations per training search, 5,000 first-generation TPUs for game generation and 64 second-generation TPUs for learning. Its reported per-search times depended on the game; its millions of games were produced with far more hardware than one lab server. Its fixed-work design is useful as a comparison; our active recipe uses the requested shared turn clocks and makes no comparable throughput claim.

[SEED RL](https://research.google/blog/massively-scaling-reinforcement-learning-with-seed-rl/) centralizes accelerated inference, batching observations from many CPU actors and using fast asynchronous communication. [Sample Factory](https://vladlen.info/papers/sample-factory.pdf) similarly separates simulation, inference and learning and reduces actor idle time. Our pipeline already has persistent private simulators, central GPU inference and parallel replay readers. Its PPO collection/update barrier deliberately preserves fresh on-policy likelihoods; copying an asynchronous off-policy learner would change the algorithm and require different correction/validation.

The saved `results/selfplay-v2-70/report.json` records 320 ordinary games in **237.3 s**, inference forward elapsed time **38.0 s** across **33,324 batches**, mean batch **1.47 positions**, and replay preparation **41.3 s** with eight readers. Actor validation and encoding cost **539.4 / 128.0 s summed across overlapping actors**. These are historical host elapsed/CPU measurements, not current GPU kernel utilization. Current lab presets already use sixteen replay readers. Increasing batch capacity or filling 90 GB of VRAM cannot help a GPU waiting for observations from the rules engine.

The encoder/output revision removes a duplicate full-board/choice fingerprint while keeping input validation, and packs neural outputs into one CUDA-to-host transfer instead of four/five separate transfers. The CPU path retains its cheaper zero-copy behavior. The six throughput changes below now keep **48 ordinary game lanes in 24 processes**, 24 queued heavy tables, four workers per search and 60 shared CPU slots under the known 64-core allocation. The 2M model and PPO batches stay unchanged. GPU routing accepts your **2â€“6 available RTX 6000 Pro GPUs**, concentrating sparse requests before activating additional replicas. The PPO learner stays on its selected GPU; its optimizer is not replicated.

### Six throughput changes (11 October)

| Change | Active implementation and limits |
|---|---|
| Native simulator work | Separate authored ordinary/search workers fuse observation, legal choices and state transitions. Choices and known-wild summaries cache by revision. Rollback copies the old journal prefix only on failure, retaining its pre-mutation hash and restoration checks. Ordinary neural play batches at most 16 forced subchoices within the same player/turn, preserving every replay row and RNG draw. |
| Double-buffered actors | Main lab presets use `processes: 24`, `hosts_per_process: 2`; `lab-small` uses 14 Ã— 2. Two private host lanes overlap blocking native/inference IO, with separate policies, RNG and reply channels. This changes process layout, not the 48/28 concurrent game capacity. Heavy search remains 24 tables/four workers. |
| Shared-memory transport | Each reply lane owns a bounded 2 MiB numeric slot (`shared_slot_bytes: 2097152`). Model, observer, revision, choice IDs, cancellation and deadlines remain checked metadata. Oversized batches use the original queue path, retaining every candidate. Slots are released at shutdown, including service failures. |
| Demand-based GPU routing | `inference_routing: demand`, `replica_backlog: 32`. Requests favor the same frozen-model replica until pending demand warrants another GPU. Existing batch/candidate/flush bounds still apply. Models are frozen once on CPU and moved to a device lazily; additional visible GPUs need not become busy. |
| Ready-leaf batching | Ready leaves from a private forest, including different frozen opponent models, share one transport submission. Services regroup by model and candidate size; other games/workers can join compatible batches. Observer-specific features remain separate. Shared deadlines, tree reservations, legality and teacher qualification remain enforced. |
| Incremental replay preparation | Validated finished shards start CPU preparation while other games continue. Readers share the 60-slot admission gate and have a bounded pending queue. The actor serializes each game once. Learner rows merge in deterministic job order; PPO still waits for the complete fresh collection before any optimizer update. |

These apply to the shared training collection infrastructure, including new demonstrations and evaluations where relevant. Rule-based v4 search remains CPU-only with its unchanged frozen bank; it cannot use GPU neural batching. Search-free policy choices retain the same probabilities and sampling. Search batching preserves the objective and shared deadlines but thread timing can change the order of completed tree work, so bitwise identical searched games are not promised.

The optimized EXEs/manifests are under `training/runtime/throughput/{ordinary,search}` and are included in Git. They reuse the existing privately transferred R1 dependencies. The installer checks the qualified baseline, dependency hashes, authored source hashes and builder fingerprint, then materializes separate workers under `artifacts/training-host/`. **No game installation, qualified R1 binary or frozen v4 bank is replaced.** Rebuilding is only needed for authored C# changes (`scripts/build-training-host.py`); the lab can use the shipped workers with Mono.

After syncing, use the existing preflight command. Its output now includes `throughput_workers`, with `optimized: true` for both workers and their manifest hashes; it exercises fused snapshots, legal choices, observations and restoration in addition to the existing full-game parity check:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
```

The implementation borrows separation/overlap principles from [Sample Factory's double buffering](https://www.samplefactory.dev/07-advanced-topics/double-buffered/), bounded transport and actor throughput from [EnvPool](https://arxiv.org/abs/2206.10558), centralized inference from [SEED RL](https://arxiv.org/abs/1910.06591), and leaf batching from [Batch MCTS](https://arxiv.org/abs/2104.04278). It retains this project's Masked ISMCTS and PPO contracts; it does not replace the simulator or make PPO asynchronously off-policy.

### One allocated GPU by default

GPU placement is a **runtime option**, not a learning hyperparameter. Changing it between committed resumes preserves weights, optimizer, population and curriculum. The default inference service shares the learner's one GPU and has no demand router.

Limit visibility to the one GPU allocated to this job. CUDA indices in the program are remapped by `CUDA_VISIBLE_DEVICES`:

```sh
# Example: physical GPU 1 is allocated to this job.
export CUDA_VISIBLE_DEVICES=1
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py \
  --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 1
```

The same placement applies to imitation, self-play, NN teachers and evaluation. All actors and replay readers share the existing 60 CPU slots; search clocks begin after admission and are shared across workers. Collection and optimizer updates retain the existing on-policy phase boundary. Demonstration generation uses CPU v4 search and does not need GPU inference.

Optional multi-GPU support remains available only when explicitly requested with `--inference-devices cuda:0,cuda:1` or `--inference-devices auto` after exposing multiple allocated GPUs. The demand router is constructed only for two or more selected devices. `auto` does not reserve GPUs or detect other users. It is unnecessary for the current workload.

For monitoring during a comparison, keep model/checkpoint, seeds, ordinary games, turn budgets and CPU settings fixed. Compare total iteration time, collection games/second, qualified teacher samples per search second, inference batch size/queue wait, replay loading/publication and optimizer time. Sample the allocated GPU's SM utilization and memory alongside process-tree CPU use, e.g. `nvidia-smi dmon -i 1 -s pucm -d 1`; memory occupancy alone is not utilization. GPU forward timings from optional profiler traces are more specific than host elapsed inference time.

### Continue the existing run

Stop after a committed iteration, update the repository, then repeat the **same run, config and IL-origin arguments** with `--resume-reviewed-source`. No IL restart is needed. For example (substitute your existing run paths):

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 1
```

The exact source transition preserves model tensors, Adam/RNG, committed counters, accepted best, historical pool and the active v4 recovery window. It permits the versioned time-budget migration plus already reviewed source transitions; arbitrary learning or CPU-budget changes remain rejected. Unchanged custom execution settings can continue through the source-only transition. Old partial iterations require recovery with their original source before upgrading. New PPO rollouts and teacher distributions carry the new identity. Existing heuristic replay/manifest formats and completed IL handoff remain compatible. The feature-cache implementation fingerprint changes, so old derived caches may be validated and rebuilt once from authoritative replay.

For new demonstrations, use a **new output directory** because the teacher identity changed. Existing audited datasets remain usable as-is; audit recognizes historical time/count teachers and the new versioned time teachers. New searched evaluations likewise need a fresh output directory; completed old reports are not relabelled. Scheduled evaluations use the existing reviewed source-namespace handling.

### Correctness review after the throughput revision (11 October)

That earlier review used source `fff450449f8fbceaa8c57cffcf826da006d53d44351c2fce4d985fc973d0df7c`; the 12 October time-budget migration supersedes its active recipe. This fixes failure handling and runtime verification without changing training settings, search allocations, reward, replay formats or installed game AI:

- **Changing GPU count can resume.** Step 5 previously rejected a checkpoint when its saved CUDA RNG list differed from the visible GPU count, despite supporting flexible inference placement. Step 5 now restores CPU RNG/Adam and every matching logical CUDA stream, including its `cuda:0` learner, when growing or shrinking the visible set. New inference-only devices keep their initialized streams. Other checkpoint callers retain the strict topology default. Logical device order defines the mapping; physical GPU IDs are not encoded in older checkpoints.
- **Failed collections wake CPU waiters.** A terminated actor could strand a slot or queue ticket, leaving replay readers or the other actor pool waiting forever during executor cleanup. A shared abort flag now wakes waiters and rejects new admission before actors are terminated. Overlapping collection observes the first failed future immediately. Failed partial games cannot enter PPO; restarting creates a fresh admission gate.
- **Inference failure cleanup is bounded.** Shutdown no longer joins a feeder with queued messages and no consumer. A saturated GPU route uses interruptible queue writes. Each envelope receives one reply attempt, so a broken actor pipe cannot duplicate a successful reply or prevent delivery to another lane. Private endpoint handles close, and fatal inference failures abort CPU waiters.
- **The single-reader path respects CPU admission.** `audit_workers: 1` now binds and restores the same shared allowance used by spawned replay readers, without altering the learner's tensor-thread setting.
- **Compiled search transformations are pinned.** Additive host manifests now include `scripts/build-search-host.py`; editing that helper cannot silently leave the old EXE active. The rebuild changed manifest fingerprints but produced identical EXE hashes. Qualified R1 and frozen v4 files are unchanged.

Use the continuation command above from a committed iteration. The immediate previous throughput source and previously supported sources have explicit review receipts. The old checkpoint bytes, model tensors, Adam state, RNG, population, accepted best, v4 recovery window and counters remain preserved. Old-source uncommitted iterations still require recovery with their original source; this review does not relabel partial replay. Existing completed IL handoff remains supported.

Verification: **65 focused checks passed; one CUDA-only check was skipped**. The suite includes 2â†’6 / 6â†’2 logical GPU restoration with mocked CUDA APIs, actual checkpoint/Adam/RNG continuation, a spawned CPU waiter released on abort, dead-reply and saturated-queue cleanup, partial-shard recovery, and native branch/chance/restore checks. An injected engine error after mutation rolled back to the exact saved checkpoint and allowed legal continuation. A two-game PPO smoke committed two optimizer steps; separate four-worker search turns completed **16/16 and 32/32** attempts without timeouts, with one qualified teacher label. Both worker preflight probes and frozen-bank/distribution hashes passed. Reports are under [the review artifact directory](../artifacts/recovery/post-throughput-review-20261011/). These are local CPU checks, not a real multi-GPU/Linux throughput benchmark or playing-strength qualification.

### Verify throughput on the lab

First run one continued iteration, then compare `timings`, `utilization.mean_inference_batch`, `utilization.actor_phase_seconds`, `utilization.replay_ingress`, inference queue/forward time and GPU memory separately. Teacher `search.teaching.phases` reports attempted/completed work, qualified samples, fallbacks, failures, watchdog limitations and elapsed time. Searched evaluations add `search_work` receipts; legacy telemetry includes its work contract and attempted/completed totals. Compare **games/second and qualified labels/second**, not VRAM allocation or attempts alone. Keep official VP, placement and win evaluation separate from this systems comparison.

New native metrics distinguish `host.native.cpu_seconds`, native elapsed and materialization/serialization from `host.ipc.stdout.read_wait`. Native process CPU has platform timer granularity; elapsed phases overlap across hosts. With two lanes, per-game Python-process CPU is omitted to avoid double-counting shared threads. Use OS process metrics alongside the native totals. `shared_requests`, `shared_replies` and `inline_requests` show transport use; multi-GPU `routing.requests_per_replica` and `per_device` show demand distribution. Replay `wall_seconds` is now the remaining join/merge after collection, while `stream_elapsed_seconds` spans overlapping preparation; `worker_average_cores` uses that full interval. Do not add overlapping phase times or compare only the shortened replay tail.

In a second terminal, capture actual GPU compute utilization during that one iteration:

```sh
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1
```

If the inference queue is mostly empty and batches are tiny, improve actor/rules/encoding throughput first. If a sustained queue builds and GPU compute is high, test a larger batch or a separate inference GPU with a controlled comparison. A larger model solely to consume VRAM is not a throughput optimization. No multi-GPU speedup or end-to-end lab speedup is claimed from local CPU tests.

### Local verification of the six throughput changes

Initial throughput source fingerprint: `b883eac48cd8e01b1e197ec842ab33461da19aac82082cbe9a8d5466935c080d` (superseded by the correctness review above). **79 targeted checks passed; one CUDA-only IL check was skipped** on this laptop. Coverage includes private model/seat attribution, shared-memory bounds and failure cleanup, sparse/backlogged routing, exact replay preparation, fused native state/chance/restoration, fixed search allocations and checkpoint/Adam/RNG/IL continuation. Reviewed transitions accept the exact new preset execution layouts and preserve previously supported source upgrades. The [verification receipt](../artifacts/recovery/throughput-20261011/verification.json) records the scope.

Two matched full native games, with alternating run order, preserved every observation/choice trace and final state/result. The optimized host took **18.57 vs 30.99 s** and **17.10 vs 26.95 s**, a **1.58â€“1.67Ã—** speedup for that random-legal-play simulator workload. These compare fused RPCs/caching/forced-choice handling against the pinned R1 host; they are not a neural training or v4 search benchmark. See the [paired native results](../artifacts/recovery/throughput-smoke-20261011-a/native.json).

A [two-game PPO smoke](../artifacts/recovery/throughput-smoke-20261011-b/report.json) exercised two host lanes, shared inference transport and streaming preparation, then committed two optimizer steps. Its two tiny shards did not finish preparing before collection ended, so this check demonstrates pipeline correctness rather than a replay-overlap speedup. [Two native NN turns](../artifacts/recovery/throughput-nn-turns-20261011/report.json) completed **16/16** and **32/32** attempts across four workers, without timeouts; the teacher supplied one qualified label and evaluation exported none. Whole-turn times were **10.81 / 17.83 s** in that small CPU fixture. Earlier turn timings were not an interleaved control, so do not treat their difference as a measured search speedup.

Both optimized workers passed local preflight probes. The selected bank's 158 pinned source/runtime files and all ten original distributed assets passed their hashes. Only the two new authored EXEs/manifests are added to the Git-ready runtime. No trained lab checkpoint, long campaign, installed-game change or CUDA/Linux throughput measurement was used. The smoke receipts retain their source fingerprints; the final revision additionally includes telemetry/preflight and continuation checks.

### Local verification of the fixed-work prerequisite

Previous fixed-work source fingerprint: `f739475371ce3d0a7dfe1c505780513a130474de3c17b8bb4501da0b67b017b3`. Its nine incoming review receipts covered the previous capped-work revision and previously supported self-play sources. The targeted work, scheduling, legality, input parity, inference, GPU routing, config and continuation checks passed. That continuation suite passed **21 tests**, including actual checkpoint/Adam/RNG preservation, old source upgrade, evaluation namespaces, partial-iteration rejection and the personality contract guard. Pipeline/CLI/scheduled-evaluation checks passed **7 tests**. The GPU pool's six checks used real spawned processes with CPU-backed replicas; they do not prove CUDA performance.

The [encoder benchmark](../artifacts/recovery/inference-performance-20261011/encoding-benchmark.json) used 24 saved real positions, both eras and five paired trials. Median time for the measured encode-plus-cache-reuse path fell from **3.027 s to 2.308 s (23.7%)**, with bit-identical features. This is an isolated component result, not a whole-training speedup. Packing outputs was slightly slower on CPU, so that optimization is enabled only on CUDA; actual CUDA benefit remains unmeasured.

[Native NN turn checks](../artifacts/recovery/fixed-work-turns-20261011/report.json) completed **16/16** attempts for generic searched evaluation and **32/32** for the teacher across four workers, with legal complete turns and no watchdog interruption. Search took **35.19 / 60.59 s** on this laptop's CPU fixture; these confirm removal of the former 12-second cutoff, not speed or strength improvement. Evaluation emitted no teacher label; the teacher produced one qualified label.

[Native v4 checks](../artifacts/recovery/legacy-opponent-fixed-work-smoke-20261011/report.json) completed **16/16** attempts in each of one Canal and one Rail turn, with legal continuation and incumbent-only replanning after intentionally discarding the queued plan. All 161 frozen bank/runtime files remained unchanged. The native first Canal search took about **9.41 s** on this laptop; the previous one-second-limited check completed zero simulations. Whole-turn timings include the deliberate extra fallback construction and are not normal game throughput. Native checks ran while unrelated Python integration changes were in progress; their reports retain start/end source fingerprints. No trained lab checkpoint, CUDA benchmark, installed-game modification or long training campaign was used.

## Self-play correctness review and continuation (11 October)

The [review report](../ai-research/SELFPLAY_REVIEW_20261011.md) describes the fixes and verification. **Continue the existing run; this fix does not require repeating IL or resetting PPO.** Stop at a committed iteration before updating. Repeat your original command with its same run, config and IL origin, adding `--resume-reviewed-source` for this approved source change. For the example run below:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 1
```

Substitute your actual saved paths and retain any original `--config` option. If your run used `--demonstrations` instead of `--imitation-run`, retain that original option. `--iterations 1` adds one iteration to the saved count; increase it for the intended continuation. Subsequent resumes may keep the review flag. Do not use `best.pt` as a new initializer: `state.json` selects the latest learner and separately retained incumbent.

The transition preserves model weights, Adam/RNG state, game/seed/search counters, the historical pool, accepted best and the already-started v4 opponent window. It saves a `state-before-source-*.json` backup. Old utility readiness is cleared in memory, then the existing bounded head-only calibration uses fresh ordinary-game returns before search teaching. Successful distillation also clears readiness because it changes the shared encoder; the trainer recalibrates using **that iteration's ordinary returns**, with all other parameters frozen. Fit time is logged as `search.collections[*].post_distill_calibration_seconds` and charged to the search allowance. A failed fit defers any remaining teacher allocation, without consuming its seeds or reservation. Teacher game outcomes never become utility-fit targets.

If a committed iteration has an interrupted monitor/broader/champion evaluation from the old source, its files are preserved. The current source uses a deterministic `*-source-<source-id>` directory with its own exact specification and fresh evidence; retrying reuses that new directory. An **uncommitted ordinary collection or next-generation checkpoint** still blocks a source upgrade: finish/recover that iteration with the previous source first. Never edit source signatures or delete partial data to bypass this check.

In that earlier revision, reward, PPO hyperparameters, the then-90% allowance, 12-second/four-worker teacher turns, 160-game cadence, four games per collection and two-collection maximum remain unchanged. This revision does not authorize previously incompatible reward changes. Completed IL checkpoints from the preceding source remain explicitly compatible; base data and checkpoint formats are unchanged.

Verification: 86 targeted checks passed after affected reruns, one CUDA-only check was skipped, and a four-game native continuation smoke passed across two iterations. This verifies execution and state preservation with fixtures; it does not measure the lab model's strength or GPU throughput.

## Reward revision and best accepted model (9 October)

This section supersedes the 7â€“8 October reward coefficients below. Active training defaults now use **`W + 0.10*P + 0.001*VP - min(0.005*N, 0.025)`**, where `N` counts insolvency events, not removed tiles. Official tiebreakers/shared wins, pairwise placement ties, undiscounted returns, linear uncapped VP and causal future-cost accounting are unchanged. PPO and NN search share the exact configured objective and reward fingerprint. A previously collected teacher distribution with different coefficients cannot train the new objective. There is no Pass, loan, build, sale or industry-specific reward.

### Start again from completed IL

Use a new run directory. The IL policy, original performance heads and generated-data formats remain compatible; a fresh PPO optimizer and the new-objective utility head are initialized through the existing handoff. The head calibrates on fresh ordinary rollouts before teaching. Substitute your actual completed IL directory if it differs from `v4-batch6`.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --iterations 20
```

Repeat the same command to continue another 20 iterations. Resume restores the latest learner's optimizer/RNG and the separately selected incumbent. Do not relabel an 8 October combined-reward checkpoint with the new coefficients: that objective change is not approved for same-run continuation. Existing reviewed migrations from older, unheaded checkpoints remain explicit; the recommended workflow here is a fresh IL handoff.

### Learner versus incumbent

Keeping the best accepted self-play model follows **AlphaGo Zero**, whereas **AlphaZero** used the latest model. This implementation adapts incumbent selection to PPO: the learner keeps advancing with its own fresh, on-policy actions, while the `selfplay` table quota now means **one current learner against three frozen incumbent policies**. Only the learner seat supplies PPO samples. The initial incumbent is the IL/initial anchor. Rejected candidates do not replace it or reset the learner's Adam state. Historical and searched-v4 table quotas are unchanged. This yields fewer learner-seat trajectories per selfplay table than the previous four-learner setup; compare throughput per learner decision as well as per game.

The selected weights are exported as **`best.pt`**, with their immutable source, identity and selection receipt in **`best.json`** and `state.json`. Full latest checkpoints remain for training resume. Cleanup protects the incumbent even when it leaves the historical reservoir. NN teacher labels still come from the calibrated current learner, with the incumbent additionally included in its bounded opponent ensemble.

Every **6,400 ordinary games**, one candidate/incumbent gate runs **48 policy-only four-player games**: eight fresh validation setups, each with all six ways to place two candidate seats and two incumbent seats. These remain individual competitors, not cooperating teams. Both models use neutral policy play without search. The candidate's summed official win share across its two seats has a 50% symmetry baseline. Accept only at **55% or more**, with the lower endpoint of the setup-clustered 95% bootstrap interval above 50%. Report VP and place separately; training reward is not the promotion metric. Missing games, incomplete seat coverage and inconclusive results cannot promote. No favorable-batch retries or early stopping are used.

This is a practical training-selection gate, not proof of global strength: 48 games and repeated selection have statistical limits, and this panel does not establish search-assisted strength or performance against every opponent. Existing periodic IL/v4 monitoring remains separate. An interrupted gate resumes the committed candidate and matching cached evidence before the next learner iteration. Gate games never become PPO or imitation data. The gate adds 48 evaluation games per 6,400 ordinary games (0.75% by game count; wall-time overhead depends on the workload).

All other active training settings remain: PPO LR `1e-5` in lab presets, 320 ordinary games per main iteration, existing v4 schedule, 640-game teacher warm-up, collection every 160 games, four games/eight maximum updates per collection, at most two collections per iteration, **12-second/four-worker NN teacher turns**. The 12 October update lowers the search wall-time cap to **50%**.

### Compact retention history

`retention.history_reports` defaults to **128**. `retention-history.json` retains at most that many compact report summaries: final loss/utility error, reward components, insolvency, phase teaching counts, learner results by composition and phase timings. It omits per-game execution receipts, inference/model telemetry trees, every optimizer epoch and repeated nested search collections. Existing large archives are converted on the next applied cleanup, even if no new files expire. Conversion is idempotent; older summaries beyond the cap are discarded. This file is diagnostic history, not checkpoint or restoration state.

Normal resume files, population weights, pinned artifacts and unfinished cleanup journals retain their existing protections. Two recent completed gate directories and the current accepted gate evidence remain; expired gate replays are subject to the usual sample limit. To compact an old inactive run, use normal retention cleanup, which also retires its other eligible unused artifacts:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 --apply
```

For policy evaluation, select `/data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1/best.pt` as `--candidate` and use `--no-search` for comparison with the incumbent-selection panel. Initially `best.pt` may still be the unextended IL anchor until a challenger passes. Installed game AI files are unaffected.

Verification for this revision: **32 focused tests passed**, covering reward/insolvency accounting, compatibility, promotion and rejection, interrupted-gate recovery, and retention protections. A **five-game CPU smoke** completed four ordinary games across a resume plus one gate-format game; the incomplete gate correctly could not promote. Compacting an in-memory copy of the uploaded 70-iteration history reduced 69 report entries from **5,455,452 to 507,061 bytes (90.7%)** without modifying the original. The full 48-game gate, CUDA/Linux throughput and playing-strength gains have not been measured in this check.

## Transfer and preflight

**Moving an already-running self-play project to another lab machine:** use the independent [Step 5 machine-transfer guide](STEP5_MACHINE_TRANSFER.md). A final iteration folder or `best.pt` alone is insufficient. The default compact export keeps continuation checkpoints, the live opponent pool/incumbent, IL lineage and still-needed rehearsal shards; it omits consumed replays, caches and unused files. Default `--dependencies repo` bundles every rules runtime/DLL from the distribution manifest and the selected host; unchanged code/config and authored workers come from Git at the exact Release tag/commit. Content is deduplicated, compressed as **multithreaded tar.zst**, and split into **1,900,000,000-byte parts for GitHub Release assets**. The manually triggered self-hosted Actions workflow packages and uploads from the source lab machine. It supports public and private repositories, with assets inheriting repository visibility. After a successful upload, the original machine/runner is no longer needed: clone the matching tag, download all parts/index, restore and verify. To use a different absolute directory, pass **`restore --relocate --root <destination>`** with the updated tool; persistent `.step5-relocation.json` maps historical paths without editing signed artifacts. Follow the guide's old-source upgrade sequence before destination preflight/resume. No administrator privileges or bind mounts are needed. There is no separate runtime ZIP/SSH import for default Release packages; Python/Mono/CUDA dependencies still need installation. Old ZIP/zstd/split packages remain supported.

The audited relocation implementation fingerprint is `f6f8028538c13fdaa5cbcd40ffde5e81fe5d9cb58b649875d8a5cd1ea8487191` (source recipe 3, relocation revision 1.1.0); older bundles retain their original recipe/hash checks. Old archives must first be restored/verified with matching archived code, using the new transfer tool from a separate checkout. Then update the destination to this reviewed source and resume with `--resume-reviewed-source`; the source machine is unnecessary. The original relocation version (`c9c9bb914477feea...`), preceding comprehensive-audit version (`8d19aff88a93edd9...`) and previously supported pinned configuration transitions are explicitly registered. A recipe 3 archive lacking the new relocation revision also requests this update when moved. Before exporting again, commit a training iteration under the installed source and stop the run. The utility-head checkpoint, optimizer/RNG, source transitions, whole-turn time budgets and cumulative accounting remain intact; existing reviewed-source utility recalibration rules still apply. Additive worker EXEs/manifests come from Git; destination preflight installs them using restored DLLs. Export also includes separate hosts used by pending evaluations and live runtime source pins; dependencies omitted by an older exporter must be supplied separately with matching bytes.

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

For a first installation using the separate runtime ZIP (skip this ZIP/import step if the machine-transfer Release has already restored the runtime), sync this revision's source/configs and tracked `training/` directory first. Read the selected bank from `training/opponents/current.json`; `training/runtime/distribution.json` must identify the same bank. The game uses **two workers**, lab heuristic and NN search **four**. Copy the freshly exported `artifacts/step5-transfer/private-rules-runtime-v4.zip` to `/data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime-v4.zip`. The distribution manifest records every member and SHA-256; import verifies the exact inventory before writing. The bundle includes proprietary game dependencies and remains outside Git commits. Authored workers and frozen source/profiles are tracked; archival bootstrap weights are never loaded.

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

For a first dataset, 500Ã¢â‚¬â€œ1,000 complete games gives independent train/validation/test setups and broad style coverage. Start smaller to verify throughput. New generation rotates the five admitted teachers. Existing audited heuristic manifests can be loaded with their immutable archived provenance; do not pass old neural replay to `demo-audit`.

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

**2. Evaluate the selected imitation model.** The same imitation run saves `initial.pt`, its random starting model, as a stable comparison. This command runs 16 games: two setups Ãƒâ€” four candidate-seat rotations, against the random anchor and searched v4 in separate panels. Treat this as a development screen; beating random is not a strength qualification. Check VP, rank, sales, stranded tiles and Pottery opportunities against the v4 panel. Evaluation games never become imitation or PPO training data.

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
| Heavy collection | 24 queued tables Ãƒâ€” 4 private workers; searched decisions share 60 active CPU slots |
| CPU allowance | Shared FIFO gate: 60 active work slots + 4 reserved cores; ordinary turns cost one slot, searched turns four; not OS affinity |
| v4 teacher clock | v4 + Masked ISMCTS; 1-second shared turn deadline after admission; 4 private roots; K 4/2 and guarded heuristic Q. Completion/selection may retire late and is measured |
| NN search | 12-second whole-turn teacher allowance after CPU admission, shared by both actions, subchoices and 4 private root workers; K 16/8, unbounded utility with fixed selection scale 1.2, new-root-information frontier, selection-only virtual loss 0.02, 2 pending paths / batch per replica |
| Central CPU/GPU inference | Batch up to 64 positions, 131,072 candidate cells, 1 ms partial flush; CPU feature preparation |
| PPO | Effective batch 2,048; microbatch 512; independent critic batch 2,048; 2 epochs; LR 1e-5; KL stop 0.01 with a proportional era/phase probe of up to 2,048 nonforced decisions |
| Learner preparation | 16 persistent CPU readers (`audit_workers`), checked compact labels/features; two queued pinned CPU batches for CUDA PPO; 20/40/60/80/100% replay-preparation logs |
| BC | LR 1e-4; main effective/microbatch settings; complete legal policy denominator |
| Curriculum | Next 20,000 games: 15% self-play / 25% historical / 60% searched v4; following 10,000: 45/25/30; afterward 70/25/5 |
| Shared reward | PPO and NN search: W + 0.10*P + 0.001*VP minus future insolvency costs; 0.005 per event, capped at 0.025 per seat/game; no Pass penalty |
| Search teachers | Warm-up 640 ordinary games; due every 160 games; at most 2 collections per iteration, each 4 games and at most 8 policy-only updates; 50% accounted wall-time limit (one times ordinary time) |
| Distillation evidence | At least 16 qualified decisions, 2 distinct setups, and 2 decisions in each phase; batch 16, at most 2 presentations per sample |
| Routine evaluation | 4-game monitor every 6,400 games; 8-game broader screen every 25,600 (20 / 80 main-preset iterations) |
| Retention | Up to 64 weight-only opponents: 16 recent + initial/IL anchor + 47 older reservoir entries; 2 full resume checkpoints; 2 replay rounds; immediate deletion of retired unpinned files |

The first search collection remains due after the 640-game warm-up. Later due points advance by **160 ordinary games**. Active lab presets use `search_every_games: 160`, `search_collections_per_iteration: 2`, and `search_wall_fraction: 0.90`. With 320 ordinary games per iteration, two independently collected and qualified teacher batches can run after the PPO update. The second batch searches with the model after the first batch's accepted updates. Each collection keeps `search_games: 4` and `distill_steps: 8`; it never reuses a small first collection to fill the second. BC, rehearsal, PPO optimization settings and per-turn search budgets are unchanged; the shared reward is described in the 8 October section. The 70-iteration review below documents the exact same-run migration. Search actions remain external behavior and never become PPO neural likelihoods.

Each NN seat in a **training teacher game** reserves exactly one seeded random round from each window:

| Phase in `search.teaching.phases` | Reserved round |
| --- | --- |
| `canal_early` | Canal 1Ã¢â‚¬â€œ4 |
| `canal_late` | Canal 5Ã¢â‚¬â€œ8 |
| `rail_early` | Rail 1Ã¢â‚¬â€œ6 |
| `rail_late` | Rail 7Ã¢â‚¬â€œ8 |

Sampling uses the setup seed, sampling seed, absolute seat and independent phase streams. Policy RNG consumption, worker completion order and failed searches cannot change later reservations. The old `turn_interval` / `max_searches_per_seat` fields remain readable for compatibility and generic search/evaluation; training teacher games use these four reservations instead. An unproductive reservation is not redrawn or moved to another phase. A normal four-game collection therefore schedules 64 whole turns: 16 per phase across four seats per game.

With the active work budget above, search returns when its shared simulation allowance finishes. The **12-second safety clock starts immediately after CPU admission**, before observing the turn. Both actions, resource payments and all other subchoices retain that deadline; four workers receive the same remaining time. Tree work stops 1.2 seconds early to leave time for merging and label validation. Native calls already in progress retire before private channels/hosts are reused; any decision returned after the hard deadline loses its teaching label and reports the overrun. This is a cooperative deadline, not process termination. Workers, actions and forced choices never get another 12 seconds. A timeout, fallback or insufficient-coverage result disables further searches in that turn; legal completion can use the supported continuation or the unlabelled fallback without restarting search. Training replicas reuse the coordinator's fallback rather than redundantly simulating their own discarded fallback plans; their private tree statistics still merge normally.

Teaching requires a nonforced choice, completed visits on at least two distinct alternatives, and a supported complete-turn continuation for the chosen visit-leading action. The target still mixes 90% normalized visits with 10% full-support prior. Forced choices, missing evidence, cancelled/late results and prior fallbacks produce no label. Incomplete in-flight rollouts are counted separately; only completed evidence can qualify.

The optional `distillation` config block now declares `min_samples: 16`, `min_games: 2` (distinct setup groups), `min_samples_per_phase: 2`, `batch_size: 16`, and `max_reuse: 2`; omitted blocks receive these defaults. All four phase minima must be met before **any** distillation optimizer step. An insufficient collection is reported and skipped without retries. Qualified rows are shuffled without replacement per pass and presented at most twice, with at most eight optimizer steps: 16 rows permit two steps, 32 permit four, and 64 permit eight at the default batch size. These are initial safeguards, not empirically optimal thresholds. `search` reports the support counts, skip reason, actual steps and sample reuse.

Distillation trains the **policy only**, with the existing weight 0.25. Its terminal game-outcome losses remain diagnostics, not critic targets. PPO and BC retain their four-seat value losses. The value heads receive no distillation gradients, Adam-moment updates or weight decay; the shared trunk can still change their predictions. This uses the teacher action distribution as in [Policy Distillation](https://arxiv.org/abs/1511.06295), while avoiding fitting the critic repeatedly to a tiny searched-game sample.

In each round's `report.json`, inspect `search.teaching.phases`: `scheduled_turns` are planned reservations, `reached_turns` are reservations actually reached, `searched_turns` count turns with a call, `actual_searches` count decision calls, and `qualified_samples` count accepted decision labels. `forced_choices`, `failures`, `timeouts`, `fallbacks`, `insufficient_coverage` and `cancelled` do not count as qualified samples. `successful_turns` means at least one qualified sample; `failed_turns` means reached without one. `unreached_turns` stays explicit. `elapsed_seconds` measures search/merge/qualification calls; `turn_elapsed_seconds` also includes host execution and unlabelled subchoices. `deadline_overrun_seconds`, `completed_rollouts` and `interrupted_rollouts` expose late retirement and evidence coverage. Per-seat reserved rounds and counters also survive cached collection/resume in shard provenance.

The wall-time gate can postpone a due collection. Its available allowance is `ordinary_seconds * fraction / (1 - fraction) - prior_search_seconds`, now **`ordinary_seconds - prior_search_seconds`** under the requested **50%** cumulative share. This is a maximum, not a target to fill. Each four-game collection still requires a conservative estimate of **768 seconds plus overhead** before admission, although parallel tables can finish sooner. The gate is checked separately before each collection; actual elapsed collection/distillation time is charged before checking the next. At most two collections run per main-preset iteration; missed or inadequate teachers never trigger extra games or unlimited catch-up. `search.budget` records admitted/deferred work. The 12-second whole-turn deadline and four workers are unchanged.

Generated demonstrations, replay/teacher schemas and model/checkpoint formats are unchanged. Existing completed fresh IL checkpoints from the immediately preceding, explicitly reviewed compatible source still work with `--imitation-run` or `--from-imitation-run`; model/data compatibility, completion, lineage and hash checks remain required. Do not regenerate your audited demonstrations for this scheduler change. For the exact reviewed same-run transition, retain your existing directory and add `--resume-reviewed-source`, as described above; source-bound PPO runs cannot silently change recipes. IL training itself, generic search/evaluation scheduling and installed game AI are unchanged.

The 5 October scheduler check passed 15 focused tests, including seeded/late-Rail reservations, after-admission shared deadlines, failed-label rejection, an eight-step distillation update, PPO guards and compatible IL handoff. One complete native CPU teacher game reached/searched all 16 reservations (four in late Rail), and cached replay loading preserved its receipts. The fresh random network produced **zero qualified labels**: five searches lacked completed coverage and eleven timed out, so no distillation update ran. Native retirement exceeded individual deadlines by a measured aggregate 1.378 seconds; all late outputs were excluded. This verifies failure handling and scheduling, not useful teacher coverage on the GPU server. The supported-label fixture separately verified the positive eight-step update path.

To repeat only that one-game plumbing smoke on the lab (a fresh network, simulation cap 16 per worker, no saved learner or campaign), use a new output directory:

```sh
PYTHONPATH=/data/not_backed_up/yxu209/ai-training/src python /data/not_backed_up/yxu209/ai-training/tests/Recovery/smoke_teacher_schedule.py \
  --device cuda --output /data/not_backed_up/yxu209/ai-training/artifacts/recovery/teacher-phase-smoke
```

For fewer resources, select `--config /data/not_backed_up/yxu209/ai-training/configs/step5/lab-small.json` before creating a new run. All active lab presets retain four NN search workers and now use the 12-second teacher allowance. The historical filename `lab-16x2.json` currently aliases the 24-table/four-worker preset. All v3/v4 heuristic presets still use one second. `recovery-smoke.json` is a compact local scaffold check with periodic teachers disabled by default. None of these is permission to migrate an old source-bound run in place.

The smaller 32-core preset retains 28 ordinary actors. Heavy tables stay at the selected 24 Ãƒâ€” 4 in the main preset, sharing the existing admission gate with ordinary games. Increasing ordinary actors supplies more concurrent neural requests; it does not change the v4 or NN search clocks. Keep the effective PPO batch and learning rate unchanged when comparing throughput. More GPU memory allocation alone is not a training improvement.

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

**Search/distillation change:** the four teacher games at iteration 0001 reached all **64 reserved turns** but produced only **one qualified label**, in early Canal. There were 14 completed rollouts against 790 interrupted rollouts; 14 turns timed out and 49 lacked sufficient coverage. The former update nevertheless sampled that one label eight times and trained its game-outcome critic each time: win loss fell **0.9903 Ã¢â€ â€™ 0.0004**, while VP loss worsened **0.1671 Ã¢â€ â€™ 0.2558**. The falling combined loss was one-point fitting, not evidence of better search or play. The revised policy-only update, diversity gate and reuse bound prevent this case. The longer 12-second shared allowance, two pending paths per replica and removal of duplicate replica fallback work aim to yield more completed evidence; they do not relax legality, hidden-information or continuation checks. Useful lab teacher coverage still needs measurement.

**Runtime:** CUDA was active. Mean ordinary inference batches were only 1.34 and 1.27 positions; collection dominated, while PPO took 13Ã¢â‚¬â€œ21 seconds. Aggregate host/validation/legacy timings overlap, and host-reported neural forward time is not GPU utilization. The 1.84M model need not fill 90 GB of memory. Keep the existing actor/batch sizes for this comparison; do not increase them merely to allocate memory. Teacher collection plus update cost 114.3 seconds, or 12.24% of accounted ordinary-plus-search time. That recorded run used a 25% admission limit; the 6 October update raised it to 50%, and the current 7 October presets use **70%** at the user's request.

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

**Verification:** 35 distinct focused tests passed (37 executions, including two repeated checks). They cover exact schema/mutation rejection, bounded caches, cached-versus-rebuilt rehearsal producing identical PPO updates, teacher metrics, value-head protection, IL handoff and search scheduling. The [local validation benchmark](../artifacts/recovery/selfplay-v2-review-20261006/validation-benchmark.json) used the same 128 saved permitted positions with fresh caches over three repetitions: median **4.307 s Ã¢â€ â€™ 1.093 s (3.94Ãƒâ€”)**. This is validation speed only, not a measured whole-training or lab speedup. A [two-game native CPU smoke](../artifacts/recovery/selfplay-v2-review-20261006/native/report.json) completed, verified exact R2 feature parity at 32 positions, reused replay caches, recorded process CPU, and ran five PPO updates. No GPU benchmark, strength tournament or long training campaign was launched.

**Continue the existing `selfplay-v2` run.** This review changed performance and reporting, while preserving the learning/search settings. The explicit `--resume-reviewed-source` path accepts only pinned source transitions, including the supplied original v2 source (`f85a8c99Ã¢â‚¬Â¦`, commit `2ee1bf9`). Signature fields must match exactly: config, data/IL receipts, rules/features, teacher bank and budgets, except for the subsequently requested **50% Ã¢â€ â€™ 70% search-wall limit** explicitly recorded in the review receipt. The approval is recorded in `configs/learning/selfplay-resume-sources.json`; this is not a general source/config-check bypass.

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

| Measurement | First two iterations | Iterations 24Ã¢â‚¬â€œ31 |
| --- | ---: | ---: |
| Mean final-epoch value loss | 1.1568 | 0.9263 |
| Mean final-epoch win loss | 1.1390 | 0.9130 |
| Mean full-policy KL | 0.01080 | 0.00479 |
| Mean clip fraction | 5.16% | 5.65% |
| Mean policy entropy | 1.9526 | 1.7933 |
| Mean PPO optimizer steps per iteration | 7 | 38.25 |

**Keep LR 1e-5, two PPO epochs, target KL 0.01, clip 0.2, entropy weight 0.005 and the existing batch sizes.** There is no exploding loss or KL overshoot trend in the continuation. Later iterations complete both epochs, with lower loss and a moderate change in policy entropy. Lowering LR again is unsupported; increasing it simply to fill the KL allowance risks accelerating an undesirable strategy. Win cross-entropy is not a playing-strength score, and the changing opponent mixture changes its difficulty.

Against three search-free v4 opponents, the incoming learner improved from **77.30 VP / 3.703 average place / 2.73% win share** over iterations 2Ã¢â‚¬â€œ5 (512 games) to **87.07 VP / 3.396 place / 7.81% win share** over iterations 24Ã¢â‚¬â€œ29 (192 games). These are different training setups with a seat-balance defect described below, not a controlled head-to-head result. The last report's **95.3 VP / 2.85 place / 15% wins** covers only 20 games, mixing eight fast-v4 and twelve searched-v4 tables as the run crosses 10,000 games; it cannot establish a new win rate.

There is a late economic warning in pure self-play: mean VP falls from roughly 73Ã¢â‚¬â€œ75 in iterations 16Ã¢â‚¬â€œ23 to 66.34 in iteration 31. Comparing iterations 2 and 31, coal builds increase from **2.68 to 4.06 per seat**, sales decrease from **2.61 to 2.11**, development decreases from **5.57 to 3.73**, and liquidation decisions increase from **0.43 to 0.92**. Pottery remains near **one build per seat**, with no missing-Pottery warning. Rehearsal simultaneously decays from 0.0936 to 0.0008 and expires at 10,000 games. This correlation does not prove that forgetting IL caused the decline: stronger competition can also lower total VP. Keep the current reward and rehearsal schedule while checking the frozen IL comparison below; do not hard-code a coal penalty or force Pottery from these aggregate counts.

Only a four-game monitor at 6,400 games is recorded, and its outcome report is absent from this upload. It compared the candidate against a recent checkpoint. The supplied logs therefore do not establish post-update strength or retention of IL skill.

### Search teachers and runtime

The 30 continued iterations collected **1,361 qualified teacher decisions from 1,920 reserved turns (70.9%)**, using 194 bounded distillation steps. Mean fixed-sample teacher KL improved **0.4024 Ã¢â€ â€™ 0.3573** after each update; this is fit to searched training labels, not a test-set result. Phase qualification was early Canal **395/480**, late Canal **369/480**, early Rail **308/480**, late Rail **289/480**. There were **413 timeouts**, 41 fallbacks and 102 insufficient-coverage turns. Failed/forced choices remained excluded. Keep four teacher games per due collection, the 320-game interval, maximum eight distillation steps, two uses per sample and all diversity/coverage guards. The user requested increasing the wall-time admission ceiling from **50% to 70%**; active lab/smoke presets and defaults now apply that one config change. It does not increase the 12-second turn clock or force spending 70% when the existing collection schedule needs less.

The latest iteration took **229.0 s collection + 41.8 s replay loading + 12.6 s PPO + 248.3 s search/distillation + 8.4 s cleanup**, about **540.4 s** overall. Search consumed **42.1%** of cumulative accounted ordinary-plus-search time. PPO preparation was 1.67 s, forward/backward 6.04 s and policy/KL measurement 4.43 s. Peak allocated CUDA memory was about **1.27 GB** (4.05 GB reserved); ordinary inference averaged **1.46 positions per batch**. Enlarging GPU batches or model size to fill VRAM will not fix the dominant simulation/search costs.

The recorded **6.85 average Python actor cores** excludes Mono engine processes, the learner and GPU; it is not whole-server utilization. Actor phase timings overlap across processes. CPU admission wait jumped when searched v4 joined the collection, consistent with 48 ordinary actors sharing 60 admitted slots with four-worker searches. Keep the existing 48 ordinary actors, 24 heavy tables, four search workers and eight replay readers for this correction; the logs do not support increasing concurrency beyond the current CPU budget. Search-free v4 remains the early opponent and searched v4 remains the mature 5% opponent.

### Implemented corrections

1. **Uncouple seat rotation from curriculum quotas.** The 20-game quota cycle and `index % 4` rotation previously put the learner into only seats 1/2 at 10% legacy, then only seat 1 at 5%. Rotation now advances once per quota cycle as well as per game. Every fixed quota position visits all four seats over 80 games, without changing the table mixture or resetting the absolute game counter on resume. Historical tables receive the same correction.
2. **Retain the 12-second whole-turn budget, with 1.2 seconds reserved for retirement/merge/validation.** The old reserve was 0.15 seconds. In the last collection, individual finalization overruns reached 0.89 seconds and caused 14 timeouts despite completed search work. Training teachers now stop tree work at 10.8 seconds into the admitted turn. All workers and actions still share the one 12-second deadline; no extra allowance, retry or late-label acceptance is introduced. This trades a little simulation time for usable evidence; measure qualified Rail coverage and completed rollouts on the lab. Standalone search/evaluation clocks and the installed game AI are unaffected.
3. **Use the fixed initial/IL checkpoint for the scheduled four-game monitor.** It remains every 6,400 games with the same seat rotations and cost. This gives a stable forgetting/progress alarm instead of comparing only neighbouring PPO generations. The broader population/legacy panel remains every 25,600 games. Existing saved monitor results are not rewritten.
4. **Remove duplicate diagnostic board walks and measure replay preparation.** Overall diagnostics are combined from the already computed disjoint table-type summaries. Counts and weighted metrics are preserved; floating-point means can differ only by rounding. `utilization.replay_ingress` now reports reader-task CPU, overlapping worker time, read/validation, diagnostics, feature preparation, compaction and parent wall time. CPU totals explicitly exclude process startup/result serialization. Replay validation, legal choices, targets, features and generated-data formats are unchanged.
5. **Allow the reviewed continuation from the supplied source `760778c9Ã¢â‚¬Â¦`.** The exact source transition and optional **`search_wall_fraction: 0.5 Ã¢â€ â€™ 0.7`** change are pinned in `configs/learning/selfplay-resume-sources.json`. All other config fields, model/data/IL contracts, opponents and checkpoint hashes must still match. The transition prints and records the budget change. Model weights, Adam/RNG, population, warm-up and scheduling counters are retained; this is not a restart from imitation.

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

**Local verification:** 37 focused tests passed, covering the 70% admission calculation, allowed checkpoint continuation, seat balance, teacher deadlines, diagnostic parity and parallel replay ingestion. One real native teacher game reached all 16 reserved turns and qualified **14/16** samples, with **zero timeouts, deadline overruns or fallbacks**; two insufficient-coverage samples were excluded. Its 14 labels and single setup correctly failed the minimum-data gate for an optimizer update. This used a fresh CPU model and a small simulation cap, so it verifies execution and qualification rather than trained strength or lab GPU speed. A five-repeat comparison on two saved native games (641 decisions) preserved diagnostic outputs and reduced median diagnostic-only time from **13.33 ms to 6.53 ms** (2.04Ãƒâ€”). Replay read/validation, IPC and the rest of training are excluded from that microbenchmark; no end-to-end speedup is claimed. Reports are under `artifacts/recovery/selfplay-v2-31-review-20261007/`.

## Review of selfplay-v2-70 (7 October)

Evidence: [`state.json`](../results/selfplay-v2-70/state.json), [`retention-history.json`](../results/selfplay-v2-70/retention-history.json), and the final [`report.json`](../results/selfplay-v2-70/report.json). This is **70 committed iterations / 22,400 ordinary games**, ending at `generation-0070.pt`. Folder/report indices are zero-based. Full reports cover iterations 0Ã¢â‚¬â€œ67 and 69; iteration 68 supplies loss/teacher summaries in state but no full match/runtime report. The aggregates below therefore use nine full reports for the last ten iterations. They describe the incoming collection policy, before that iteration's PPO/distillation update.

### Findings and decisions

| Metric | Iterations 16Ã¢â‚¬â€œ23 | Iterations 60Ã¢â‚¬â€œ69 |
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

The worsening self-play economy warrants intervention, although aggregate self-play VP alone cannot distinguish regression from competition for resources. Against the **same searched-v4 opponent mode**, the learner's aggregate training win share changes from 8.13% over 160 seat-games in iterations 40Ã¢â‚¬â€œ49 to 4.86% over 144 seat-games in the available final reports; VP changes from 83.44 to 79.71 and mean place from 3.34 to 3.56. Seeds differ and samples are small, so these are warning signs, not a controlled strength result. The latest individual iteration reaches only 42.01 self-play VP. No coal penalty, Pottery quota or reset to IL is introduced. Pottery stays fully legal and exploration-supported. The user-requested follow-up below adds the insolvency return separately.

Teacher reliability improved after the earlier retirement fix: the last ten collections qualify **552/640 reserved turns (86.25%)**, with **zero timeouts or deadline overruns**. Phase counts are 160/160 early Canal, 135/160 late Canal, 129/160 early Rail and 128/160 late Rail. There are 21 fallback and 66 insufficient-coverage events; failures still produce no labels. Mean same-collection teacher KL improves 0.3518 Ã¢â€ â€™ 0.3209, but the incoming policy already agrees with the teacher's top action about 95.9% of the time. This is training fit, not proof of search strength. More independent teaching and varied opponent models are preferable to repeatedly fitting these same sparse labels. Keep **12 seconds/four workers**, with the current **200-ms finalization reserve** (the historical review used 1.2 seconds), and keep the existing coverage gates and two-use limit.

The latest iteration spends **237.3 s in collection, 41.3 s in replay preparation, 11.6 s in PPO, 203.1 s in teacher collection/distillation and 8.5 s in cleanup**. Replay readers use about 7.02 CPU cores out of eight allocated; they run after collection. Increase lab `audit_workers` to **16**, preserving all validation and replay formats. Keep 48 ordinary actors, 24 queued heavy tables, four search workers and the 60-slot admission gate. Ordinary inference averages only 1.47 positions/batch. Peak allocated CUDA memory is about 1.07 GB, with 5.53 GB reserved. These are memory/host-phase metrics, not a measured GPU activity percentage; whole-server CPU includes Mono work absent from Python actor counters. No lab end-to-end speedup is claimed until measured.

### Historical population and paper comparison

The uploaded pool contains only generations **0, 1, 61Ã¢â‚¬â€œ70**. Its old selection repeatedly spaced an already-pruned list; middle generations gradually disappeared. Merely enlarging that same algorithm would not reliably preserve history.

AlphaGo Zero generated self-play with its best accepted network; AlphaZero used the continually updated latest network. Neither method uniformly sampled all historical opponent checkpoints. This distinction is explicit in the [AlphaZero paper](https://arxiv.org/html/1712.01815v1). A league mixing current and historical opponents is closer to [AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf), which also used performance-based opponent sampling to address forgetting. Here a simple stratified mixture is used; noisy four-player table results are not converted into two-player PFSP ratings.

The revised bounded archive keeps **16 recent versions + the initial/IL anchor + 47 older versions**, at most **64 distinct weight-only models**. Older slots use fixed, deterministic hash priorities: a streaming bottom-k reservoir. Repeated selection over retained entries has the same result as selection over full history at fixed capacity; it does not drift toward the recent end. Existing unavailable checkpoints cannot be reconstructed. This run grows the new archive from its surviving population and any surviving full checkpoints. Sixty-four 1.84M FP32 models represent roughly **470 MB of raw weights**, plus checkpoint metadata, while only two full optimizer/resume checkpoints are kept.

Within historical tables, each opponent slot samples **25% initial/IL, 25% recent, 50% older reservoir**; missing strata fall back to available versions. The current learner is excluded when an older version exists. Initial training necessarily uses its only available network until distinct versions exist. The table curriculum stays **70% current self-play / 25% historical / 5% v4** at maturity. Thus training remains 95% neural games. Search opponent ensembles now include the current learner plus up to three frozen models spanning IL, recent and older versions. `historical_opponents.generation_seats` records actual sampling. Increasing inference cache capacity from **16 to 72 models** accommodates the 64-model archive and transient learner/teacher versions without eviction churn. The archive is for opponents; old PPO trajectories are never replayed as on-policy data.

### More teaching, with a bounded budget

The clarified request is **150% of the previous search-time allowance**, not an impossible 150% share of total time. The cap rises from 70% to **7/9 Ã¢â€°Ë† 77.78%** of accounted ordinary-plus-search time, equivalent to search using **3.5Ãƒâ€” ordinary time**. The supplied run actually used about **41.24%**, so the old cap was not the bottleneck.

Active lab presets now schedule teaching every **160 ordinary games**, at most **two collections per iteration**, with **four games and at most eight updates per collection**. With the main 320-game iteration this usually means eight teacher games and at most 16 policy-only updates; samples are independent and each collection passes its own diversity/phase gate. Ordinary PPO still collects 320 games and updates once per iteration. Warm-up remains 640 games. Budget admission is rechecked after charging the first collection's actual time; insufficient/failed evidence never triggers extra collection retries. Aggregate `search.teaching` counts both collections, while `search.collections` keeps their distinct source models, thresholds, fit results and elapsed times. Aggregate fit numbers are sample-weighted per-collection comparisons, not one fixed held-out evaluation.

At the observed timings, a second roughly 203-second teacher collection would increase a roughly 502-second iteration to around **705 seconds before replay-reader savings**. This is a workload estimate, not a benchmark: extra teaching deliberately increases runtime. More search cannot by itself correct a consistently wrong value/opponent model; check the fixed-anchor evaluation and economic trends before extending far beyond the next review.

### Continue generation 70 on the lab

The exact source/config transition is pinned in `configs/learning/selfplay-resume-sources.json`. Checkpoint hashes, Adam/RNG, IL lineage, model/data contracts, population and all ordinary progress remain checked and preserved. The future teacher threshold changes **22,720 Ã¢â€ â€™ 22,560**, so the first new 320-game iteration can serve reservations at 22,560 and 22,720 without replaying past allocations. A partially collected old-source next iteration still blocks migration until recovered with its original source. Completed IL and generated-data formats stay compatible. Installed game files are unaffected.

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

## Insolvency recovery and temporary searched-v4 curriculum (7 October; reward superseded below)

This follow-up is enabled in all five active lab presets. It keeps the same model, IL handoff, replay format, learning rate, PPO batches/epochs, teacher search clocks, search frequency and evaluation schedule.

**Insolvency return:** charge **0.02 once per player per income-settlement round with unpaid debt**, capped at **0.10 per seat/game**. A win has utility 1. Multiple industries liquidated to settle the same round count as one event. Taking a loan, carrying affordable negative income, passing, normal merchant sales and grouped-sale continuations receive no additional penalty. **Final Rail has no income settlement and is exempt**, matching the declared B01 rules repair.

Events come from the first debt-phase public view, which exposes every seat's initial shortfall. When no player has an asset requiring a disposal decision, automatic settlement is recovered at the observed round boundary from the last completed action's public cash/income. These derived labels are calculated in the existing parallel replay readers, after validation. Only small scalar returns cross into the learner; serialized demonstrations, replay shards, feature caches and observation contracts do not change.

The 7 October implementation used a win-only baseline and subtracted future capped insolvency costs from actor returns. **The 8 October shared-utility revision below supersedes that actor/baseline formula.** Event detection, causal timing, per-seat caps and the temporary opponent schedule remain the same.

`actor.insolvency` reports learner-only event counts, counts by era, total shortfall, affected trajectories, total/mean capped cost and decisions with future cost. Terminal progress also prints event count and mean cost. `liquidate` remains a separate count of asset-removal decisions in the unchanged economy diagnostics.

**Ordinary training tables after activation:**

| Stage | Duration | Current self-play | Historical NN | Searched v4 |
| --- | ---: | ---: | ---: | ---: |
| First | 20,000 games | 15% | 25% | 60% |
| Second | 10,000 games | 45% | 25% | 30% |
| After recovery | Ongoing | 70% | 25% | 5% |

A v4 table contains one learner and three admitted searched-v4 opponents at **1 second/four workers per complete turn**. The percentages describe table composition, not the fraction of all player seats. The temporary schedule overrides the search-free warm-up while active. It retains the 24-table heavy queue and shared CPU admission gate; it does not raise the core allowance. Main 320-game iterations wholly within stage one contain 48 self-play, 80 historical and 192 v4 tables; stage two uses 144/80/96. Expect more collection time and fewer learner trajectories per iteration than the old 70/25/5 mix.

`state.legacy_boost` persists the activation counter, final counter and configuration digest. `report.legacy_boost.stages` records both boundaries and actual games from each stage, including iterations that cross a boundary. Teacher, evaluation and IL games do not advance this schedule. It never rearms on restart. From the uploaded generation-70 state, boundaries are **22,400 Ã¢â€ â€™ 42,400 Ã¢â€ â€™ 52,400 ordinary games**. If the run has advanced, both boundaries move with the actual adoption counter; no absolute counter is hardcoded.

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

Compare this against unchanged distillation using matched fixed-IL, searched-v4 and older-NN panels. Preserve frozen collection policies and exact stored likelihoods; avoid live weight smoothing during collection. If testing EMA separately, initialize an evaluation copy at adoption and update only after a complete PPO-plus-distillation iteration. A proposed 10-iteration half-life gives decay `2**(-1/10) Ã¢â€°Ë† 0.933`, with only the restart state and one candidate retained. [PyTorch's averaging utilities](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html) provide the averaging mechanism.

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

## Shared PPO / NN-search return and checkpoint continuation (8 October; coefficients superseded above)

The active lab presets now use this undiscounted objective for each stable seat:

```text
terminal utility = reward.win * W + reward.placement * P + reward.vp * final_VP
remaining return at decision t = terminal utility - insolvency charges strictly after t
defaults: reward = {"win": 1.0, "placement": 0.10, "vp": 0.0005}
insolvency = {"event_cost": 0.02, "max_cost": 0.10}
```

`W` is the official win share, including shared wins. `P` compares the final `(VP, income, cash)` tuple against each other seat: one point per player beaten, half per exact tie, divided by three. Competition rank alone cannot recover half-credit ties. For example, four exactly tied seats each receive `W=0.25` and `P=0.5`. The VP term uses actual scored VP: each additional 10 VP contributes exactly `0.005`, at low or high scores. There is no target, ceiling, saturation, relative score, running score normalization, Pass penalty, action-frequency reward or conditional switch. Existing PPO advantage standardization still follows return/baseline computation; it does not change the reward definition or search values.

Each insolvency event consumes its capped budget once in chronological order. Debt choices already occur after that event, so their remaining returns exclude it. Both the offline learner and live search account for all debtors in the first debt observation, group multiple forced tile removals into one event, and handle automatic settlement when a player has no assets. The final Rail round remains exempt from income/debt settlement under the declared game rules. Borrowing and affordable negative income are not penalized.

### Utility head, calibration and search

The new 10,020-parameter head predicts four remaining returns from the observation representation and the four remaining insolvency allowances. It has a 32-unit hidden layer and unconstrained outputs. Initialization uses a detached combination of the existing win/rank/VP predictions plus a zero-initialized learned correction. This is only an initial estimate: the correction learns exact-tie effects and future insolvency costs. Official win probabilities, competition placements and VP predictions keep their existing labels/losses.

Before its first search collection, the new head receives three bounded head-only calibration passes on freshly collected ordinary games, using learning rate `0.001` for those passes. Compatible parameters and their Adam moments are frozen during calibration; the regular PPO learning rate is restored afterward. Calibration logs before/after MSE and MAE. Teaching is deferred if calibration does not produce a finite, non-worsening fit. Subsequent ordinary PPO updates jointly fit the utility head and existing official heads. No extra calibration games or search retries are launched. Calibration is an in-sample initialization check, not evidence of accurate search values or improved play.

PPO advantages use `remaining_return - frozen_combined_utility_prediction`. Search terminal returns use the same scoring formula, and leaves use this same combined head. Each simulated node subtracts only insolvency charges incurred after that node; leaf budgets have already consumed those charges. Costs before the root are never charged again. Private search workers receive identical real-game budget snapshots; inference requests and search caches bind their remaining-budget inputs.

Combined utility is **not a probability**, and neither predicted values nor completed returns are clipped to `[0,1]`. Its supported numeric domain is finite real values, without a VP-derived upper bound. PUCT uses fixed `utility_scale: 1.2`: values are divided by that scale for selection, with existing exploration, FPU reduction and virtual loss in those fixed selection units. Raw backed-up returns remain unscaled. This is not a running Q transform. Keep the scale positive; changing coefficients or scale requires an explicit reviewed configuration transition, because old teacher distributions represent a different objective.

When configuring a new run, change the coefficients in `reward` and its signed mirror `search.reward` together; validation rejects mismatched PPO/search objectives. Reward identity is `brass-combined-return/1.0.0`, and new search/visit targets use `masked-ismcts/1.2.0` / `ismcts-root-visits/1.2.0`. Teacher fingerprints include the full reward coefficients and search scale. Old teacher records remain readable, but are rejected by combined-objective distillation. Old PPO collections are not reused after a source transition; new rollouts retain exact behavior-policy likelihood checks. Heuristic demonstration/replay formats and their cached input features stay unchanged.

### Continue the existing run

Finish a committed iteration and stop training before updating the checkout. Keep the complete run directory and its original imitation run. The transition preserves compatible weights, Adam moments/steps, RNG, historical population, opponent recovery-window progress and scheduling counters. Only the new utility-head parameters start with empty optimizer state. Older historical opponents keep their existing policy weights. Version-1 IL/self-play checkpoints remain loadable; new combined checkpoints use `four-seat-checkpoint/2.0.0` and persist their reward contract and calibration receipt. The original checkpoint files are not rewritten.

Pin the generation used as the comparison control before retention can retire its full checkpoint. Substitute the actual last committed generation if it is later than 70:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py pin \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 \
  --paths /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/generation-0070.pt

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2 \
  --imitation-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/v4-batch6 \
  --resume-reviewed-source --iterations 10
```

Use the same imitation path recorded in the original run. There is no second imitation pass. Known source transitions are pinned in `configs/learning/selfplay-resume-sources.json`; unknown source/config changes, modified checkpoints and old uncommitted iteration data still fail closed. Do not manually edit signatures or delete incomplete data to force a resume. Completed standalone IL handoff also remains supported for a new self-play directory: the utility extension occurs after loading the compatible imitation weights.

The 9 October revision used **90%**; the current 12 October recipe supersedes it with **50%**, allowing one accounted search second per ordinary second. Other active lab settings remain unchanged: PPO `1e-5`, two PPO epochs, 640-game search warm-up, due every 160 ordinary games, at most two collections per iteration, four teacher games per collection, at most eight policy-only distillation steps, 12 seconds per whole turn, and four workers. The 60% searched-v4 / 20,000-game and 30% / 10,000-game recovery windows continue from their saved counters; they do not restart on this migration.

### Compare results separately from reward fit

Inspect `actor.reward.components`, `actor.reward.weighted`, `actor.utility_calibration`, and each epoch's `utility_prediction.mse` / `mae`. Components report raw official win share, pairwise placement fraction, actual VP, capped insolvency cost and combined utility averaged per learner seat-game. Utility prediction errors cover all four heads. Existing official win/rank/VP, insolvency events, phase coverage, search failures, timing and GPU/CPU telemetry remain separate. The new mixed objective deliberately trades off wins, place, VP and solvency; a lower utility loss does not establish higher win rate.

Use matched seeds, all four rotations, fixed IL references and searched v4 when comparing the pinned control against a new generation. First compare their policies without search to isolate training changes. For each candidate, use a distinct output directory:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/step5/lab.json evaluate \
  --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/generation-0080.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-v2/initial.pt \
  --lineups recent-neural mixed-v3 --setups 4 --seed 941000 --partition validation --no-search \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-reward-generation0080
```

Repeat with `generation-0070.pt`, the same reference/seeds/options, and a new output directory. Report official win share, average VP and rank with setup-group uncertainty; small panels are noisy. A separate equal-time search comparison can use the new calibrated candidate with `--search`; using its new combined objective against the old candidate's win-only search changes two factors, so report that comparison separately. No strength promotion, long campaign, installed-game update or lab-GPU benchmark is performed by this source change.

### Local verification and limits

**30 focused checks passed**, covering official tiebreakers/shared wins and pairwise ties, constant marginal VP reward through 100,000 VP, causal grouped insolvency and partial remaining caps, final-Rail exemption, node-local terminal/leaf backup, stable-seat budget mapping, CPU batching/chunking/remote inference, PPO baseline agreement, fresh distillation and old-target rejection, IL handoff, checkpoint/Adam/RNG migration, seeded phase reservations and shared deadlines. All seven active/smoke presets validate, with other preset fields unchanged. The generation-70 and preceding insolvency-recovery source transitions were checked; the existing recovery window and schedule counters persist.

The final native CPU smoke ran **two ordinary games plus one game with a single four-worker NN teacher**, using a fresh small model. It committed five PPO updates, calibrated the new head (in-sample MSE **0.2345 to 0.1853**), and reloaded the new checkpoint/optimizer without repeating migration or calibration. Streaming public insolvency budgets matched offline labels throughout the native teacher game. Four phase turns were reserved/reached: early Canal and late Rail each produced one qualified sample; late Canal and early Rail lacked sufficient completed coverage and correctly produced none. There were **17 completed simulations and no deadline overruns** across those four search calls. The smoke's cap was eight simulations per worker; the production cap remains unchanged.

That single teacher setup correctly failed the production sample/diversity gate, so native distillation performed **zero optimizer updates**; the focused tests separately exercise a valid distillation update and stale-target rejection. No extra games were generated to satisfy the gate. This final smoke took approximately **259 seconds**; an earlier development smoke also ran three games. Receipts are in `artifacts/recovery/combined-reward-20261008/verification.json` and `native-final/report.json`. These checks establish execution and accounting, not playing strength, medium-model calibration quality, or CUDA/Linux throughput. The lab continuation and matched comparisons above are the next measurements.

## Storage and diagnostics

Cleanup runs after every committed iteration, including the last iteration requested on the command line. Main training presets now use `quarantine_hours: 0`: retired unreferenced files are journaled and deleted in the same cleanup, so they do not occupy another 24 hours of disk space. It retains two full resume checkpoints, up to 64 slim opponent checkpoints, the initial evaluation anchor, two replay/report rounds, and two recent completed development screens per panel. Extra explicitly pinned files and dataset/formal-evaluation dependencies are retained.

Older round and development-screen metrics are consolidated into `retention-history.json`, capped at 128 compact summaries by default; their per-game execution receipts and expired raw replays are removed. Two completed cleanup receipts remain; unfinished receipts are preserved for recovery. Empty retired directories are removed. Historical lineage entries in `state.json` remain small audit metadata and may refer to retired full checkpoints: load opponents from the live `population` entries, and resume from the latest learner checkpoint. The selected incumbent is protected separately and exported as `best.pt`. Weight-only opponents no longer carry optimizer state or full report copies, and full checkpoints contain a small update receipt instead of another copy of all game telemetry. The in-memory report list is also limited to the latest two rounds.

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

Use `top` to inspect the Python reader/actor and Mono host processes together, not only the learner PID. In the older `results/1.1` two-game smoke, CPU preparation took about 95Ã¢â‚¬â€œ96% of PPO wall time and neural batches averaged about one position. That does not measure the present 48-actor preset. A local two-game replay check measured 3.69 s for serial read/preparation/diagnostics versus 1.97 s with two warmed readers; process startup made the first parallel pass 5.22 s. Readers persist across rounds to amortize startup. CUDA throughput must be measured on the lab; local checks use CPU.

The selected **v4 / patch 1.4.20** uses C# heuristic Masked ISMCTS, six advanced-opening styles, **two game workers** and the existing **2/5/12-second** options. Lab search remains at four workers. Its previous `smoke-7` screen tied 6Ã¢â‚¬â€œ6 on wins and averaged 88.79 versus 84.71 VP against the economic control; the user then explicitly selected this bank and stopped further strength tests. This selection does not certify tournament superiority. Productive development resets intent age, and opening goals influence build/setup candidates. The [1.4.7 resource lifecycle](AI_EXECUTION_RESOURCES_1.4.7.md) remains in force. Trained NN deployment is a later qualification step.

## Historical rule-based comparison commands

These commands document earlier S1/Masked diagnostics. They are not required for the v4 transfer or fresh training, and no further strength comparison was run for this selection.

Run eight matched four-player games (two setups Ãƒâ€” four seat rotations). Each table has one Balanced serial 1s/240-work opponent, one Balanced 1s/two-worker/720-total-work opponent, and two rotating serial S1 styles.

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
