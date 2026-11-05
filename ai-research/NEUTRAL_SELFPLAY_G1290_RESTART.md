# Fresh neutral self-play from g1290

## Current continuation: g760, automatic reward stages and explicit GPU selection

The pulled `g0760-20261105T004442699955Z` export contains **760 committed
generations / 243,200 ordinary games**. Its entropy schedule is at **k=50**, with
**140 held iterations**, and reward is still **0.01/event, cap 0.075**. Entropy
is **0.003** in all four groups. The schedule had been waiting for a manual
reward step; the old log does not identify the physical GPU.

The new `entropy-anneal-schedule/1.3.0` contract automatically applies the next
listed reward at a committed boundary, before collection:

- Before **g761**: apply the overdue **0.005/event, cap 0.05** stage.
- At **k=100**: apply **0.0025/event, cap 0.025** before the next collection.
- Keep the current entropy clock, baselines and healthy-iteration rules.
  Entropy continues from 0.003 to 0.001. No held iterations are backfilled.

No `REWARD_STEP_PENDING` action or manual `insolvency-step` command is needed
with this contract. At a boundary the trainer closes collection workers and
uses the same verified checkpoint restoration path in the running command.
It preserves main/companion weights, both Adam states, RNG, warmup/controller
progress, population and lineage. It invalidates reward-dependent utility
calibration and retires incompatible PPG targets only after durable adoption,
before collecting new targets. The transition receipt records the committed
generation, entropy progress, old/new reward and first affected collection.
An interruption retries that same transition; it does not advance the clock.
The originally generated config can be reused on later continuation: only its
already committed automatic reward stage is advanced from `state.json`.
Unrelated config changes still require their usual exact source review.

Sync the complete reviewed checkout, then run this from the Linux lab root
for **g761–g780** (change `--iterations` to continue longer):

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 760 --apply && \
python scripts/configure-training-g760.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 760 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-g760.json && \
CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/step5.py \
  --cuda-visible-devices 7 --device cuda \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g760.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 760 --iterations 20
```

`--cuda-visible-devices 7` is applied by `scripts/step5.py` **before importing
Torch**, so it does not depend on an environment prefix on an earlier command.
You can instead `export CUDA_VISIBLE_DEVICES=7` before the entire command chain.
An unexported `CUDA_VISIBLE_DEVICES=7 python recover... && python train...`
sets the variable only for recovery. Existing inherited masks are preserved.
For unambiguous selection across host enumeration orders, the option also
accepts the GPU UUID shown by `nvidia-smi --query-gpu=index,uuid --format=csv`.

With a single visible GPU, PyTorch calls it **logical `cuda:0`** even when the
mask selects physical GPU 7. Do not pass `--inference-devices cuda:7` with that
mask. Startup now prints the effective visibility mask, logical device,
selected mask token and UUID; iteration environment reports retain them.
Match the UUID against `nvidia-smi` to verify physical placement. No lab CUDA
hardware is available in the local test environment, so physical placement
must be confirmed from that startup line on the lab machine.

Validation: **35 distinct checks passed** across the focused batch and two
corrected test-fixture reruns, including all eight g760 cases, exact checkpoint
weights/Adam/RNG restoration, automatic k=50/k=100 boundaries, interrupted
adoption retry, reuse of the original launch config, durable reward-cache
retirement and pre-Torch GPU mask/child-process inheritance. The bounded CPU
PPG smoke retained **4/4 steps**, KL **1.49e-7** versus cap 0.003. No production
training or evaluation games were run. Logs: `artifacts/g760-validation.log`
and `artifacts/g760-fixture-rerun.log`.

The broader legacy `test_single_gpu_performance` batch also has **four unresolved
fixture failures**: three use synthetic R2 links that fail current v11 topology
admission, and one requests an unapproved old `f19424...` source transition.
These fixtures are outside this revision; no topology or source-admission
guard was weakened. The current g760 resume, GPU-selection and smoke checks
pass. Boundary setup adds one worker teardown/checkpoint reload per reward
change, with no additional games; lab wall time has not been benchmarked.

## Earlier continuation: g611, extended insolvency schedule and PPG beta range

The `g0611-20261104T073052792185Z` export records **611 committed generations /
195,520 ordinary games**, with entropy progress **k=41**, no held iterations,
and next-collection entropy **0.00336** in all four groups. Preserve these
counters and the original entropy baseline; the g611 helper makes only these
reviewed changes:

- **Epsilon 0.02 → 0.00** for ordinary learner collection. Probabilistic sampling,
  T1.3 three-decision normal openings, T1.5 two-decision post-fork openings,
  decoupled PPO and the 0.5% T2-only branching recipe remain unchanged. Recorded
  epsilon and policy identities change; stale PPO behavior is rejected.
- **PPG beta floor 1.0 → 0.25**, with the saved beta and trip window preserved.
  The g611 report records beta 1.0. A clean phase with accepted optimizer steps
  and retained exact policy KL **strictly below 0.0015** halves beta for the
  next phase: **1.0 → 0.5 → 0.25**. Phases with no retained steps do not lower it.
  A new cap trip with three trips in the last five phases still doubles beta,
  up to 32. PPG's hard cap becomes **0.003**. Loss checks remain every 16 steps,
  KL checks every 32 plus epoch tails/retention, two epochs, every iteration,
  two replay ages and full two-age held-out loss checks. PPO KL is unchanged.
- Extend the versioned insolvency plan with **0.0025/event, cap 0.025 at k=100**,
  after the existing **0.005/event, cap 0.05 at k=50**. This timing was explicitly
  confirmed by the user. The current reward stays **0.01/event, cap 0.075**
  until the k=50 restart. Entropy still ends at 0.001 at k=100.

Each reward boundary uses the existing explicit `insolvency-step` helper.
Entropy holds at the boundary until that step is applied. Reward changes refresh
incompatible PPG targets after the durable transition; extending the plan and
changing epsilon/beta/cap alone preserve the compatible two-age critic cache.
Main/auxiliary weights, Adam/RNG, warmup/LR multiplier, population, lineage and
committed GAE/entropy evidence remain intact. The new PPG adoption receipt and
feedback report include the floor, cap, threshold and preserved beta.

After syncing the complete reviewed checkout, recover Ctrl+C work and collect
**nine iterations, g612–g620**. Nine healthy commits would reach the first
pending reward boundary; if any iteration is unhealthy, k remains below 50 and
the reward-step helper correctly refuses an early change.

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 611 --apply && \
python scripts/configure-training-g611.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 611 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-g611.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g611.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 611 --iterations 9
```

At **`REWARD_STEP_PENDING`**, use the reward-step command below with the current
committed generation. The same helper applies the next listed stage at k=50
and k=100; it does not skip stages or change reward during a collection. A
longer `--iterations` count is allowed, but entropy will hold at a pending
boundary until the explicit restart. To continue while health holds delay the
boundary, reuse the generated config with the actual committed generation;
do not rerun the g611-specific configuration helper after advancing.

Validation: **26 distinct focused checks passed** across the main batch and
the corrected isolated-checkpoint fixture rerun. Coverage includes old/new
contracts, k=50/k=100 boundaries, preserved k=41/baselines, epsilon-zero behavior
reconstruction, stale-row rejection, beta thresholds/clamps, rollback of weights
and Adam above 0.003, helper retries and exact checkpoint state. The small CPU
PPG smoke retained **4/4 steps**, measured KL **1.47e-7**, and moved beta from
**1.0 to 0.5** while keeping both replay ages. The third-reward-step test verifies
cache retirement happens only after durable config adoption. Existing g570
helper/source-approval regressions pass too. Evidence is in
`artifacts/g611-validation.log` and `artifacts/g611-checkpoint-validation.log`.
No production checkpoints, long training or evaluation games were run. These
checks verify implementation behavior, not improved playing strength or GPU
throughput. A larger allowed PPG KL can permit more retained fitting work;
actual wall time depends on the loss/KL stopping points.

## Earlier continuation: g570, v4 PFSP and trained-start rating reference

The `g0570-20261103T190804654465Z` export records **570 committed generations /
182,400 ordinary games**. Apply the reviewed changes from collection **g571**:

| Setting | Value |
| --- | --- |
| All four entropy coefficients | **0.005 → 0.001**, linear over **100 healthy committed iterations** |
| Insolvency at schedule start | **0.01/event, cap 0.075**, first reviewed reward step |
| Ordinary epsilon | **0.02**, same family/context distribution |
| Normal openings | **T1.3 for three action decisions per seat**, unchanged |
| After a branch fork | **T1.5 for two action decisions per seat**, then T1 |
| Branch eligibility/rate/fork move | Skip the three normal opening actions; **0.5%**; **100% T2** alternative, unchanged |
| Ordinary tables per 320 | **166 self-play / 154 history / 0 standalone v4** |

The branch continuation has a separate versioned behavior contract. Its two
tempered decisions use the existing decoupled T1 PPO objective and actual
exploration-inclusive behavior probabilities. The alternative fork move stays
excluded; legal masks, strict reconstruction and branch-family holdouts remain.

The pending g570 command now starts the
[entropy/insolvency schedule](ENTROPY_ANNEAL_SCHEDULE_PLAN.md) at **g571**.
Entropy starts at **0.005**, including the revised fork recipe; the earlier
fixed **0.0025** proposal is superseded. For each group, use
`0.005 - 0.004 * min(k / 100, 1)`, where `k` counts only healthy committed
iterations. Economic health, a nonempty PPO update, no PPO early stop, no PPG
KL trip, and exact categorical KL at most `1.5 * target_kl` are required.
An interrupted iteration never advances the schedule. Saved adaptive entropy
baselines are retained; adaptive entropy remains off.

The same reviewed restart applies insolvency **0.02/0.075 → 0.01/0.075**.
At **k=50**, entropy reaches **0.003** and holds until a second explicit restart
applies **0.005/event, cap 0.05**. No running iteration changes its reward.
Each reward step refreshes incompatible PPG caches and invalidates utility
calibration; fresh fitting follows the existing DCPG/PPG rules. The first PPG
phase uses one fresh iteration, then the usual two-iteration buffer refills.
Main/value/auxiliary Adam, RNG, GAE and controller evidence remain preserved.

Reports and compact history retain used/next coefficients, healthy progress,
held count, applied reward step, and `waiting_for_insolvency_step`. A report-only
raw-policy entropy/family/industry probe runs every five iterations (first g575)
on the existing cached positions, with separate elapsed time. It does not
compute AnchorKL, compare against IL, reload IL data or restore seat fingerprints.
Its final-weight probabilities can also serve a due PFSP admission check.

V4 moves immediately through an explicit **manual override at g570**; no 100%
win rate or qualification evidence is fabricated. The general automatic gate
requires ten consecutive committed iterations with ordinary v4-table games,
and a **combined first-place win share of 1.0 for the two learner seats in every
game**. A tie with an opposing bot fails this criterion; a zero-game iteration
breaks the evidence window. The gate is persistent and one-way.

The five existing search-free bank-3dc65253a610191d66a5 training variants form
one rule-bot PFSP member. Its initial S/C estimates are copied from the eligible
IL member, giving exactly equal **post-insertion** sampling probabilities. Its
score subsequently follows actual history-table outcomes; variants cycle by
actual sampled bot seats. The frozen neural pool capacity remains 24, with one
additional bot member. Bots retain rule-bot descriptors and supply no PPO or
snapshot-critic trajectories. Standalone v4 EV/GAE checks stop, including old
v4 holdout rows in the retained PPG buffer. Historical-table learner evidence
continues normally, including tables where PFSP sampled a v4 opponent.

Main PL evaluation keeps **128 games** and the existing estimator, priors,
seat correction, paired setup bootstrap and promotion rules. Use **2 IL + 2 v4
+ 4 trained-start + 24 ladder setups**, each rotated through four seats. The
original source g1290 seed is named **`trained-start-baseline`**. Its rating
floats; **IL is the only fixed zero**. The original seed hash/identity is checked
and retained, without confusing it with this run's future generation 1290.
Reference setups rotate across the fixed seeds; nearby opponents remain
distinct and balanced, with the existing explicit startup fallback. The new
baseline first receives measured evidence in the next due slot, g580.

New history/rating/KL-reference snapshots retain every main-model head and the
complete Ownership/VP companion bundle, with its identity and maturity. Slim
filenames include the companion identity so identical main weights cannot
silently reuse different auxiliary weights. Available full checkpoints can
backfill their own generation's old slim references. Already-deleted companion
weights cannot be recovered and are never replaced by random or other-generation
heads. Startup writes `auxiliary-checkpoint-inventory.json`, automatically included
by the report exporter. The historical g1290 seed remains its original verified
artifact; companion heads are unnecessary for its policy-only PL evaluation.

GAE remains **0.95**, PPG beta floor **1**, and the
two-iteration PPG buffer, every-iteration cadence, loss16/KL32 checks and current
LR schedule remain intact. Preserve model/companion weights, all Adam/RNG state,
progress, training lineage and compatible replay targets. Rehearsal, DOMiNO,
seat fingerprints and AnchorKL remain off. The LR controller retains its normal
data-composition window reset while preserving its multiplier and warmup progress.

After stopping training, run from the Linux repository root:

**g570 resume approval repair (2026-11-03):** the Rust lab r10 changes moved the
aggregate source identity to
`f74d78220d8dd622929d36618d374dfc0c4bfde313051cd2d60e5ba6a19cc296`.
Their unchanged-config qualification receipts did not include the pending
`trained-restart-g570/1.0.0` + `entropy-anneal-adoption/1.0.0` migrations. As a
result, the config helper raised `Reviewed resume requires unchanged config
except explicitly pinned review changes` before writing its output or starting
training. The registry now adds these exact routes, including same-source
retries and later entropy/reward helper transitions. Existing approvals remain
unchanged; unknown sources and unrelated config edits still fail admission.
Sync the repaired `configs/learning/selfplay-resume-sources.json` with the
current checkout, then rerun the command below. Do not overlay the older sealed
Rust qualification ZIP afterward: it contains the older registry. This repair
does not change the trainer's source fingerprint or the sealed lab bundles.
Validation: 11 focused checks passed in 49.5 seconds, including the actual
config helper, strict rejection cases, schedule toggles/retries and an isolated
checkpoint resume preserving exact model/Adam/RNG state. All 78,503 previous
approval entries remain intact; five explicit routes were appended. No
production checkpoint or training progress was changed.

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 570 --apply && \
python scripts/configure-training-g570.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 570 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-g570-anneal.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g570-anneal.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 570 --iterations 20
```

This continues **g571–g590**. Recovery archives an uncommitted Ctrl+C collection;
the helper verifies the full g570 checkpoint and original seed before writing
the config. Reviewed resume commits the v4 migration atomically. Retrying before
g571 restores the same receipt rather than adding the bot or resetting it again.
The reward change automatically refreshes PPG replay after the configuration
transition commits; no separate cache deletion command is needed. For a
separate newly prepared trained-restart recipe,
`configure(config, manual_generation=None)` enables the measured v4 gate and
leaves entropy annealing off, without inheriting this run's manual authorization.

When the console reports **`REWARD_STEP_PENDING`**, stop at a committed boundary
and run this from the Linux repository root. `G` is read from the stopped run;
the helper rejects a premature step, stale generation or uncommitted collection.

```bash
G=$(python -c 'import json; print(json.load(open("artifacts/step5/joint7-v11-2-g1290-trained-restart/state.json"))["iteration"])')
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration "$G" --apply && \
python scripts/configure-entropy-schedule.py insolvency-step \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation "$G" \
  --output "configs/step5/joint7-v11-2-g1290-trained-restart-anneal-reward-g${G}.json" && \
python scripts/step5.py \
  --config "configs/step5/joint7-v11-2-g1290-trained-restart-anneal-reward-g${G}.json" train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation "$G" --iterations 20
```

Use the same generic helper with `off` to return entropy to **0.005**, keeping
the current reward and saved progress; `on --start next` resumes that progress.
An explicit `on --start next --reset` resets only schedule progress and is
allowed before reward step two. `insolvency-step --reverse` explicitly moves
one listed reward step back: **0.005/0.05 → 0.01/0.075**, or for the extended
g611 plan **0.0025/0.025 → 0.005/0.05**. Progress stays unchanged and the ramp
waits for the forward step again. Returning to **0.02/0.075** is not authorized
by the existing reward-transition list. None of these helpers edits model files.

Validation: **23 focused tests passed** across `test_entropy_schedule` and
`test_training_g570`, covering schedule boundaries/holds, applied coefficient
provenance, on/off/reset and reward steps, interrupted restart chains, actual
checkpoint/Adam/RNG restoration, PPG cache refresh, and the other g570 changes.
The isolated CPU reward-change PPG smoke retained **two steps**, exact probe KL
**2.20e-8**, with only the fresh reward's cache age. The unchanged-reward
two-age smoke retained four steps. Nine additional legacy regression checks
passed. Two archived tests have pre-existing fixture issues: an obsolete
epsilon-zero fresh-default assertion in `test_exploration_revision`, and a
missing g1340 export in `test_policy_observations`. Equivalent fresh-default
and LR-health checks using the current g570 fixture passed.
No long training or evaluation campaign was launched; these checks establish
implementation behavior, not improved playing strength. Probe overhead is
measured per invocation; no claim of GPU training-speed improvement is made.

## Previous continuation: g370, direct lambda 0.95 and T2 forks

The `g0370-20261102T221517838215Z` export records **370 committed generations /
118,400 ordinary games**, active lambda **0.98** and next PPG beta **0.5**.
This reviewed continuation applies:

| Setting | Effective value from g371 |
| --- | --- |
| All four entropy coefficients | **0.005**, unchanged; adaptive entropy remains off |
| GAE lambda and floor | **0.95 immediately**, explicit user-authorized hotfix |
| Branch rate per eligible decision | **0.5%**, down from 1%; self-play tables only |
| Fork action distribution | **100% T2**, excluding the played move before normalization |
| PPG beta floor | **1.0**; saved 0.5 is raised to 1, values already above 1 are preserved |

Parent/post-fork openings remain **T1.3 for three decisions**, with
`skip_opening_decisions=3`. T2 here selects the alternative fork action only.
Fork moves remain excluded from training; hidden-deck reshuffling and opening
counter resets remain intact. Ordinary epsilon stays **0.05**. Rehearsal,
DOMiNO, fingerprints and AnchorKL stay off. The **160/154/6** table mix and PFSP
remain unchanged.

The direct GAE change is a user setting, not measured gate qualification.
History/v4 qualification flags and their evidence are preserved. Already-enabled
tables use 0.95; unqualified mixed tables still use 1.0 until their existing gates
qualify. Lambda-stage age restarts at this boundary, without resetting training
progress, work counters or independent evidence. No new healthy iterations are
invented. PPG retains its two-iteration buffer, every-iteration cadence,
loss16/KL32 checks, 0.002 safeguards and current LR schedule. Beta can still
increase on KL trips and decrease after low-KL accepted phases, bounded below by 1.

Model/companion weights, all Adam states, RNG, LR state, utility calibration,
compatible PPG caches, population and lineage are preserved. The generation-zero
startup receipt stays historical and unchanged. The helper creates a new config;
reviewed startup applies the checkpoint migration once and logs its before/after
values. Retrying before a new commit repeats the same restoration, not a reset.

After stopping the trainer, run from the Linux repository root:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 370 --apply && \
python scripts/configure-training-g370.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 370 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-g370.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g370.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 370 --iterations 20
```

This continues **g371–g390**. Recovery archives an uncommitted Ctrl+C collection,
without rewinding g370. The helper verifies the actual full checkpoint hash and
rejects stale generations or conflicting output files. The desktop report bundle
cannot replace the Linux model/optimizer checkpoint. No fresh PPG cache is needed:
reward, encoding, targets and holdout contracts have not changed.

Validation: **18 focused checks passed** across the new continuation tests and
existing GAE/branching/PPG regressions. The isolated CPU PPG smoke retained **4 of
4 attempted steps**, with probe KL **1.58e-7**. It used small authored data and
actual two-age disk caches, checking cache hashes, unchanged model/Adam/RNG at
adoption, beta clamps, restart continuity, T2 probabilities, opening resets and
strict source/config checks. One initial smoke assertion used the wrong report
field name; correcting the test to `attempted_steps` resolved it. No production
code or safeguard was relaxed. Logs: `artifacts/g370-tests.log` and
`artifacts/g370-regressions.log`. No long training or evaluation was launched;
these tests do not establish improved playing strength or training throughput.

## Previous continuation: g253, GAE lambda floor 0.98

The `g0253-20261102T075610938330Z` export confirms **253 committed generations /
80,960 ordinary games**. Its latest report (`iteration-0252/report.json`, which
commits generation 253) uses lambda **0.95**. The reviewed continuation below
adopts `gae-advantage/4.0.0` with `lambda_floor=0.98`: before the next advantage
calculation, saved 0.95 becomes 0.98 and cannot automatically fall back to 0.95.
Already-qualified history/v4 tables use 0.98 too; unqualified tables remain at
1.0 until their existing independent EV gates qualify. Lower-lambda shadow
diagnostics remain comparisons only, not the trained advantages.

This changes only the signed `advantage` block. Weights, both main Adam states,
companion heads/Adam, RNG, LR state, ordinary progress, population, reward,
utility calibration and the two-iteration PPG cache remain intact. Mixed-table
qualification origins and evidence remain intact. If lambda changes, its stage
origin becomes g253 and its stage streak starts at zero; prior work counters and
recorded observations are retained. No restart from the original g1290 seed.

After Ctrl+C, run from the Linux repository root:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 253 --apply && \
python scripts/configure-gae-floor.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 253 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-gae098-g253.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-gae098-g253.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 253 --iterations 20
```

This runs **g254–g273**. Recovery archives any uncommitted partial collection;
it does not rewind committed progress. The helper verifies the full checkpoint
hash and writes a new configuration without editing run state. Actual adoption
occurs during reviewed training startup and is safe to retry before the first
new commit. Subsequent continuations use this same configuration and the latest
committed generation. The original generation-zero manifest stays unchanged;
the original launch recipe below documents how this run was created.

**16 focused checks passed**, including the isolated CPU smoke. Validation covers strict floor configuration, healthy-window clamping, mixed
table evidence preservation, reviewed signature/retry checks, helper safety and
an isolated CPU Adam checkpoint/advantage smoke. No long training or evaluation
is launched; this is a requested algorithm setting, not evidence of strength gain.
Logs: `artifacts/gae-floor-tests.log`, `artifacts/gae-floor-regressions.log`,
`artifacts/gae-floor-legacy-fixture.log`. The legacy startup test initially lacked
its archived g1360 export; its unchanged assertions passed with the available
g253 GAE-3 configuration. No production validation was weakened. `git diff --check`
passed. The actual Linux full checkpoint is checked by the continuation command;
the desktop export contains reports, not those model/optimizer tensors.

This starts a **new run at generation 0**, initialized with the retained g1290
model tensors. It does not rewind or modify the g1360 source run. The source
inventory is `results/joint7-v11-2-grouped-lr-critic-g1360-20261101T011723464975Z`.

## Recipe

| Setting | New value |
| --- | --- |
| `canal_action`, `rail_action`, `canal_other`, `rail_other` entropy | Fixed **0.005** each; adaptive control off |
| Ordinary epsilon | **0.05**, existing family/industry distribution |
| Insolvency | **0.02** per event, **0.075** maximum |
| Openings | **T1.3**, first **three** action decisions per neural seat |
| Fork-action mixture | **60% T1 + 30% T2 + 10% q** |
| Branch frequency | **1%** per eligible decision, active from the **first collection** |
| DOMiNO | Off: no intrinsic companion, dual updates, context sampling or context-evaluation panels |
| Policy seat tags | Neutral style `[0,0,0,0]` and intent `[0,0,0,0,0,0,0,0]` for every neural seat |
| Critic fingerprints | Existing 28 zero columns; role/age descriptors and dimensions preserved |
| AnchorKL | Measurement, calibration and LR gating remain off |
| IL rehearsal | **Off for the new run below** via `--no-rehearsal`; global defaults unchanged |
| Main GAE | Starts at **0.98**; automatic 0.95 transition retains 20-iteration health/work/rating evidence |
| PPO warmup | Fresh Adam, **100 optimizer steps**, starting factor 0.1; independent of auxiliary warmup |
| IL/v4 panels | Retired at generation zero by explicit **manual user override**, without qualification games |
| Ordinary tables per 320 | **160 self-play / 154 history / 6 v4**; IL is an ordinary PFSP-eligible member |
| PL main evaluation | 128 games per slot; competitive lineups manually enabled, with existing startup fallback |

The parent opening and each post-fork opening use the same three-decision T1.3
window. Forks skip those first three action decisions. Each fork component
excludes the played action before normalization; if `q` has no alternative
support, the available T1/T2 components renormalize to 2/3 and 1/3. Fork moves
remain excluded from PPO/PPG training. The existing decoupled raw-T1 PPO opening
objective retains its actual behavior-probability checks and importance cap.

PPO/PPG losses, architecture, LR rules, economic checks, PPO KL safeguards, teacher
disablement and the current two-iteration PPG recipe are retained. PPG remains
every iteration, loss checks every 16 steps, KL checks every 32 steps. The fresh
run gets new PPO/value/auxiliary Adam state, warmup, RNG, counters, gates, economic
evidence, replay buffers and PFSP scores. No source DOMiNO or PPG state is copied
into the learner checkpoint. The separate Ownership/VP heads start fresh as
specified below; no complete g1290 checkpoint or old companion is required.
The seed's utility calibration is invalidated for the new reward; ordinary
head-only calibration follows the existing training rules.

## Fresh Ownership/VP heads and accepted-work warmup

Only the separate industry-slot and connection companion heads are initialized.
The imported backbone, policy, **win, rank, terminal-VP and utility value heads**,
and descriptor encoder retain their exact tensors. These existing heads are not
reinitialized. Hidden global/entity projections of the new companion keep normal initialization;
only the final ownership and VP output layers are zero-initialized. Initialization
uses a deterministic forked Torch RNG and restores the learner RNG stream.

`local-scoring-training/2.0.0` has configurable experimental defaults
`head_only_iterations=2`, `minimum_fitting_steps=32`, `ramp_iterations=5`.
Both fitting minima must hold: two **committed iterations with retained PPG
auxiliary fitting**, and at least 32 retained auxiliary optimizer steps. Merely
waiting two iterations is insufficient. PPO output-preservation steps, rejected
intervals, missing labels and zero-weight targets do not count.

During head-only warmup, the auxiliary gradient into the shared state/entity
encoders and descriptor encoder is zero. Ordinary PPO and PPG continue training
the main model. Auxiliary recorded-target fitting occurs only in PPG; DCPG PPO
continues output preservation. After warmup commits, the next iteration uses
encoder-gradient multiplier **0.2**, advancing to **0.4, 0.6, 0.8, 1.0** only after
each further committed iteration with retained fitting. An empty/rejected phase
holds the multiplier. The heads' own gradients and loss weights are not scaled;
the same encoder multiplier applies to PPO auxiliary output preservation.

The auxiliary Adam base LR stays **1e-4**, clipping **1.0**. The new companion
uses the existing PPG warmup/decay *factors* on that base LR. The current two-age
PPG path retains its existing optimizer lifecycle (no new per-phase reset);
acceptance and rollback restore the matching auxiliary Adam and fitting counters.
Main PPO/value optimizer schedules, two-iteration buffer, cadence, holdout family
separation and KL caps are unchanged. The 100-step PPO warmup is independent.

Generation zero saves fresh head weights, empty Adam states, initialization
identity and warmup state. Checkpoints persist fitting counters and ramp stage;
continuation neither repeats nor skips warmup. This initialization path is
independent of `--trained-restart`: the latter controls branching, manual panel
overrides and GAE only. A missing *main* seed file still fails preparation.

Each report includes current/next stage, encoder multiplier, retained fitting
steps and distinct fitting games/rows for that iteration (including eligible
previous-age and branch data). Existing diagnostic intervals remain every five
iterations, at most 512 ordinary learner held-out rows. Slots/connections have
separate ownership CE and VP MAE, support and occupied-entity metrics; occupied
means a nonzero ground-truth owner at era end. No IL diagnostics or added games.

## Manual scheduling permissions are not qualification evidence

`trained_startup` records the user authorization, source generation/model,
manifest hash and generation-zero origin. Manual panel triggers contain no
win rate, game count, confidence interval or fabricated evaluation files.
Reports label them `manual_override`; measured panel evidence remains absent.
They enable IL-in-PFSP, the 160/154/6 mix, panel retirement and competitive PL
scheduling only. They do not satisfy economic, LR, GAE, critic or strength gates.

The original IL remains the PL zero point and opponent identity. Main rating
budget, model/priors, bootstrap and statistical promotion rules are unchanged.
Competitive scheduling uses its existing fallback until enough distinct rated
checkpoints exist. GAE's evidence windows start empty: the automatic 0.98→0.95
transition still needs the existing accepted value-training support, 20 healthy
observations, rating evidence and PPO/PPG KL safeguards. History/v4 EV gates stay
separate and start unqualified.

## Seed and history

The exported state records g1290 model ID
`73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff`.
Its retained weights-only file is named
`population/weights-73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff.pt`
under the old run, with SHA-256
`e14d5b7a4596944ac7dec1b30c676b52ab79beb7bbf2e55a8b8540cc8a312937`.
The preparation helper checks the actual file, architecture, tensor identity and
source generation. It accepts the retained slim checkpoint when the original
full g1290 checkpoint has been cleaned up. Reports cannot replace missing weights.

Only members of the **currently retained source pool** with source generation
**≤1330** are imported. The g1360 inventory contains these 19 members:

`IL, 855, 905, 990, 1020, 1045, 1075, 1105, 1135, 1155, 1175, 1205, 1225, 1240, 1250, 1280, 1300, 1315, 1330`.

Source generations **1335, 1340, 1345, 1350 and 1355** are excluded. This cutoff
applies to imported source models; normal admission of models trained in the
new run continues. PFSP's formula, exposure accounting, decay, admission health,
novelty test and eviction safeguards remain. Scores restart at the configured
prior, because the learner/context/reward regime changed. The initial capacity
is raised from 16 to **18 scored members** to retain all eligible snapshots; the
existing maximum of **23 scored models plus IL** stays unchanged.

Imported snapshots carry explicit original generations and model IDs in
`imported_from`. Their local birth generation is `source_generation − 1330 − 1`,
so they are older than the new generation-zero learner without borrowing old
training progress. Their local admission age starts at zero. The original IL
remains the pool's generation-zero reference and the PL zero point. The new
learner's `initial.pt` is the g1290-derived model, not the IL reference.

All imported weights and probability caches are copied into the new run. The
fixed 2,048-position probe is copied with its existing integrity/source checks;
it is used only for pool admission/eviction. The helper does not compute
AnchorKL or new fingerprints. Preparation is retry-safe while the source stays
at its expected committed generation, and it refuses an existing destination
with training progress. Later retention cannot delete an imported active member.
The seed inventory remains an audit receipt; it does not pin evicted imported
weights or probability caches indefinitely.

## Start on the lab machine

Stop the old trainer first. No old-run recovery/rewind command is needed: only
its committed checkpoints are read. This command uses a new folder distinct from
earlier prepared recipes. `--trained-restart` signs the explicit startup settings;
`--no-rehearsal` keeps weight zero, with no floor or rehearsal shard loading.
Do not reuse an incompatible previously prepared destination or manifest.
Run from the repository root:

```bash
python scripts/prepare-neutral-selfplay.py \
  --source-run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-source-generation 1360 \
  --seed-generation 1290 --history-through 1330 \
  --trained-restart --no-rehearsal \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --output configs/step5/joint7-v11-2-g1290-trained-restart.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-g1290-trained-restart.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --fresh --iterations 20
```

These are **new generations 1–20**, not g1361–g1380. The `--imitation-run` argument
retains the completed IL's reference identity for the pool and ratings; it does
not enable rehearsal when its configured weight is zero. The verified seed
manifest selects g1290 as the learner initialization. The helper prints the
imported/excluded generations and does not launch training.

After 20 completed generations, continue the new run by omitting the preparation
step and replacing `--fresh` with `--expected-generation 20`. If a later command
is interrupted mid-iteration, use the existing recovery helper on the **new run**
with its own committed generation, never 1360.

Export its reports explicitly:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --continue
```

The exporter now includes `fresh-start.json` with source/cutoff receipts, alongside
the ordinary reports. This run is also the exporter's default. `--continue`
reuses verified unchanged reports only from earlier exports of this same new
run; keep any bundles referenced in `export-manifest.json`. Manual startup
overrides need no measured panel files. Checkpoint tensors and PPG arrays are excluded.

## Validation scope

For the trained-startup / accepted auxiliary warmup revision, **63 distinct tests
passed** across the focused and compatibility batches. The final 13-test startup
and warmup batch passed in 83.7 seconds. Checks include exact main tensor equality,
normal-hidden/zero-output initialization without RNG movement, encoder and
descriptor gradient isolation, full head gradients at every ramp stage,
retained-work accounting under loss and deferred-KL rejection, checkpoint restart,
manual receipt provenance, 160/154/6 planning, fresh GAE evidence and PPO step warmup.
An archived g270 test now uses its original approved transition; production
source-review checks were not loosened. Existing review restrictions are preserved.

The bounded CPU smoke used **42 authored games**, **one PPO step**, **two retained
PPG fitting steps**, **36 distinct fitting games / 432 fitting rows**. Auxiliary
scale remained zero, with one qualifying iteration and two steps toward its
2-iteration/32-step requirement. PPO categorical KL was about **4.62e-8**;
PPG probe KL rounded to **0**. This demonstrates pipeline behavior, not playing
strength. No long training, native evaluation games or production checkpoint edits.

The new bookkeeping adds small counters and retained-row tracking to existing
rollback snapshots. Occupied diagnostics reuse the existing bounded forward
pass. There are no additional evaluation games or full-dataset diagnostic passes.
Production throughput was not benchmarked by this authored CPU smoke.

Evidence: `artifacts/trained-startup-validation.json` lists the four result logs;
the final corrected results cover all 63 cases. `git diff --check` passed.
Key implementation files are `trained_startup.py`, `local_scoring_warmup.py`,
`local_scoring_auxiliary.py`, `fresh_selfplay.py`, `league.py`, `value_phase.py`,
the versioned config/GAE/exploration and panel/PFSP gate modules, and
`scripts/prepare-neutral-selfplay.py`. The launch instructions and focused tests
are updated together. Existing signed configurations keep their old auxiliary
schedule; no old age counter is silently treated as accepted fitting evidence.

### Earlier neutral/rehearsal revision checks

Focused checks cover strict settings, old-config behavior, cutoff selection,
weights-only fallback, fresh Adam/progress, original-IL identity, checkpoint
restart, PFSP priors and complete 320-table neutral planning, neutral replay
admission, branch probabilities, post-fork openings, and report export.

**71 distinct focused tests passed**, including the existing PFSP, source-review,
fingerprint/AnchorKL-disable and historical opening/PPG regressions. The final
fresh-start/source-review/export batch passed all 14 tests in 81.1 seconds.
`git diff --check` passed. Validation logs are under
`artifacts/fresh-neutral-{native,regressions,final-regressions,complete}.log`;
reruns resolve the stale archived-migration fixture and an initial test-name typo.

The run-scoped no-rehearsal follow-up passed three focused checks in 32.0 seconds:
the signed override stays off across floor boundaries, defaults remain enabled,
and seed preparation/restart still preserve the original source. See
`artifacts/fresh-no-rehearsal-tests.log`.

An isolated CPU smoke uses authored fixtures: one PPO step and two accepted PPG
steps, with no DOMiNO or AnchorKL calls. A separate initialization smoke creates
and reloads a generation-zero run from synthetic retained weights, checks exact
tensor equality and verifies that its source files are unchanged. These checks
do not demonstrate improved playing strength or measure production throughput.
The actual Linux g1290 weights are not present in this desktop checkout; the
preparation command verifies them before writing the final usable configuration.
