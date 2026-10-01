# Step 5: self-play, search distillation and meta-RL on the lab computer

**Archived pre-2.2 instructions:** use [Fresh Masked ISMCTS training](TRAINING_RECOVERY_LAB.md) for current commands. This document records earlier recipes and checkpoint-import/search workflows; do not run those commands for the new random residual-model run.

**30 September recovery revision:** the default trainer now uses audited v3 imitation, effective-minibatch PPO, a declining 40% → 20% → 10% → 5% v3 table curriculum, and persistent CPU actors with shared batched inference. Follow [TRAINING_RECOVERY_LAB.md](TRAINING_RECOVERY_LAB.md) for current commands and migration. The original recipe described below is retained as historical context in `configs/step5/previous-lab.json`; its initial 95% neural schedule does not describe the new default. Optional meta-RL retains that explicit historical config.

The project root in every lab command is **`/data/not_backed_up/yxu209/ai-training/`**.
This delivery implements the learning loop. Local smoke weights demonstrate that
the code updates a model; they are not a strong trained opponent or a Unity release.
Use the fresh approximately 1.9M-parameter R2 backbone for a new substantial run.
Executed Windows checks and their limits are in [STEP5_EXIT_CHECKS.md](STEP5_EXIT_CHECKS.md).

## Training priorities

The default 20-table cycle has **14 current-agent self-play tables, five tables
against historical neural snapshots, and one table against qualified frozen v3
opponents**. Thus 95% of tables use neural opposition. The three legacy opponents
on the remaining table are independent players. Original Hard has zero training
weight. Set `legacy_every` to `0` to train entirely against neural agents.

Every player optimizes its own terminal first-place share, including official tie
breakers. There is no combined bot reward. The policy update uses clipped PPO on
fresh learner trajectories and the R2 four-seat value losses. Historical and
legacy actions do not receive learner actor gradients or imitation targets.
Valid losing games are retained. Snapshot hashes, replay provenance, behavior
probabilities and setup partitions are checked before an update.

The regular trainer freezes every generation. Historical sampling chooses from
recent checkpoints half the time and the complete history half the time. This
reduces reliance on one latest opponent; it does not by itself guarantee that
the population becomes strong. Promotion requires comparisons with multiple
trained neural references, not just wins over the weak legacy bank.

### Pottery is available throughout training

The held-out item is the **particular frozen `pottery-flex-v3` opponent identity**.
It does not exclude Pottery builds, development, sales, cards, observations, value
targets or search proposals. Each block of twelve new learner seat trajectories
contains six neutral contexts and one focus for each of the six industries,
including Pottery. The initial offset varies, and the context persists for the
whole game. These are learning contexts, not yet calibrated personalities.

Exploration assigns equal mass to available build-industry groups before an
optional focus adjustment. Having more legal Cotton locations cannot consume all
of Pottery's exploration allocation. Complete-turn search retains a proposal for
every available build industry, as well as setup actions such as Loan/Develop.
There is no industry-based replay rejection, anti-Pottery penalty or imitation
of only the five admitted legacy styles.

Reports contain actual builds, legal build opportunities, context exposure and
builds matching the assigned focus. A candidate with at least 30 Pottery build
opportunities and zero Pottery builds receives a warning and fails automatic
promotion. Also evaluate `--focus Pottery` on held-out setups. Equal exploration
cannot promise equal final industry usage: profitable Pottery depends on cards,
locations, timing and other players. The aim is to learn when it is a good main
strategy, not force it into unsuitable positions.

## What travels through Git, and what needs a private copy

The following paths are deliberately **not ignored**:

| Component | Repository paths |
| --- | --- |
| R2 backbone, encoder, batching and losses | `src/brass_ai/learning/`, `configs/learning/` |
| Loadable R2 weights | `training/models/bootstrap/r2-fresh-medium.pt`, `r2-value-scaffold.pt`, `r2-onpolicy-scaffold.pt` |
| Current v3 legacy bank | `training/opponents/current.json`, `pool.json`, active `bank/` snapshot, qualification reports, `configs/legacy-ai/` and original `src/BrassLegacyAI/` code |
| Authored host and v3 worker binaries | `training/runtime/r1/BrassHost.exe`, `training/runtime/v3/BrassLegacyAI.exe` |
| Step 5 code and selected release weights | `src/brass_ai/step5/`, `scripts/step5.py`, `configs/step5/`, `training/models/releases/` |

R3's six "models" are C# policies and JSON profiles, not neural checkpoints.
`training/runtime/distribution.json` pins the shipped weights/workers and the
private runtime. The original Hard AI is inside the proprietary game assembly;
its pinned implementation accompanies the rules engine in the private bundle.
Game DLLs, assets, saves, decompiled source, bulk replay and experiment outputs
remain ignored. A Git clone alone cannot run the proprietary simulator.

The local export command is `scripts/step5.py --root <absolute-local-root> assets`.
It has prepared this private bundle:

`C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham/artifacts/step5-transfer/private-rules-runtime.zip`

Commit and push the authored source/configuration/weights before cloning or
pulling on the server. Being unignored does not mean a file is already committed
or published. Transfer the private ZIP directly to your lab account, preserving
its filename, at:

`/data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime.zip`

For example, create that directory on the lab, then use your normal SCP/SFTP
client from the laptop. The bundle includes the preserved R1 runtime and current
v3 runtime; it does not copy your saves or change the installed game. Its importer
checks every file against the manifest before writing and refuses to overwrite
a different existing runtime.

## Install and qualify the lab runtime

Use the existing **`ai-training` conda environment**. Every `python` command below
assumes `conda activate ai-training`; no venv is needed. Installation, tests and
CUDA smoke training already passed on your lab according to your report. Keep
that working environment; the install commands are for reproducing it, not a
requirement to reinstall packages when fixing the qualification error.

Use Python 3.12, a working `mono` command for the preserved .NET host, and an
NVIDIA driver compatible with the pinned PyTorch CUDA wheel. These commands
retain the tested R2 dependency versions. The CUDA 12.8 installation is listed in
[PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/).

```bash
conda activate ai-training
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r /data/not_backed_up/yxu209/ai-training/requirements-step5.txt

python /data/not_backed_up/yxu209/ai-training/scripts/step5.py import-runtime --bundle /data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime.zip
python /data/not_backed_up/yxu209/ai-training/scripts/check_qualification_reports.py
python /data/not_backed_up/yxu209/ai-training/scripts/verify-step5.py
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
python /data/not_backed_up/yxu209/ai-training/scripts/check-transferred-legacy.py
```

Pass global options such as `--device`, `--config`, `--root` and `--mono` **before**
the subcommand. CPU is an explicit alternative: use `--device cpu` and the CPU
PyTorch wheel. CUDA requests fail rather than silently falling back to CPU.

The transferred-legacy check runs one full game with four original guarded Hard
seats. Its `--v3-seat 0` option instead inserts one frozen Balanced v3 seat. These
are runtime/baseline checks, separate from the neural training distribution.

Preflight checks checkpoint compatibility, all private runtime fingerprints, a
device forward/backward calculation, frozen v3 loading and two complete games
against semantic fingerprints recorded on the qualified Windows host. It records
a machine-specific pass in `artifacts/step5/preflight.json`; the CLI requires that
pass before match collection. Do not use `--record-parity` on the server or
replace expected fingerprints to bypass a mismatch. Linux/Mono and RTX execution
are **not claimed tested by the Windows checks**. If parity fails, resolve that
runtime difference before collecting lab training games.

### Fix an existing `Qualification report changed` checkout

The first published qualification report was hashed with CRLF line endings, but
Git stored it with LF. The expected hash is
`790942901e25720a58afe0bfe6536f54d014cfba7ec4432cb3e8f1843f5520eb`;
the LF-converted copy is
`36821a4da89dc43d66d31943223eddb1167ce09678c78419fb0216689e90d7eb`.
This affects v3 admission in the main configuration. The pure-neural smoke run
does not exercise admission; the original runtime preflight used diagnostic
loading, which also skipped this gate. Conda and CUDA are unrelated to the error.

The fix preserves qualification-report bytes with `-text` in `.gitattributes`.
[Git documents this switch](https://git-scm.com/docs/gitattributes) as disabling
line-ending conversion. The distribution check now exercises all six admitted
v3 policies and includes their qualification report in the required Git assets.

After transferring/pulling the fix into the lab checkout, run:

```bash
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/check_qualification_reports.py --repair-line-endings
python /data/not_backed_up/yxu209/ai-training/scripts/check-step5-distribution.py
```

The repair accepts only an LF/CRLF conversion that produces the **existing exact
SHA-256**. It checks all referenced reports before writing, then invokes the
unchanged admission gate. Content changes, missing files or conflicting pins
remain errors. Do not update the pool's hash to match an arbitrary report.
The command is safe to repeat and does nothing to an already matching report.

Your reported failure occurs before creating `main/state.json` or collecting
games. If that file is absent, rerun your original command with `--initial`.
If a later attempt already created it, resume without `--initial` instead.
This fix does not change model/trainer signatures or frozen bank code, so it
does not invalidate existing smoke checkpoints. No environment reinstall,
private-runtime re-import or completed-game replay is necessary for this repair.

## Start with a short self-play run

The smoke configuration uses four games per iteration, a medium backbone, two
PPO epochs, pure neural opposition and no scheduled search. This measures the
host/inference/replay/update workflow before committing to a longer run.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/smoke.json train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/lab-smoke --initial /data/not_backed_up/yxu209/ai-training/training/models/bootstrap/r2-fresh-medium.pt --iterations 2
```

Inspect both iteration reports: all games must have completed; gradients and
losses must be finite; learner trajectories and context/build counts should be
present. GPU utilization can be modest because legal moves, IPC and rules
simulation run on the CPU. More GPU memory alone does not increase game throughput.
The current main preset scales this to 32 concurrent rule hosts and 320 games per
iteration, with separate inference/training batch limits. See the
[64-CPU lab scaling guide](STEP5_SCALING.md) for settings, bottleneck measurements
and migration from your existing trained run. The small smoke remains unchanged.

## Historical run and resume commands

The original larger workload is retained in `configs/step5/previous-lab.json`.
For the current default use [the recovery guide](TRAINING_RECOVERY_LAB.md).
The historical commands below create a **new** run. If `main` already contains your earlier experiment,
use the scaling guide's `--from-run` command to create `main-scaled`, preserving
its weights, optimizer, historical opponents and next setup seed. Config/source
signatures deliberately prevent resuming an old run with changed settings.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/previous-lab.json train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main --initial /data/not_backed_up/yxu209/ai-training/training/models/bootstrap/r2-fresh-medium.pt --iterations 20

# Resume for another 20 iterations; omit --initial.
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/previous-lab.json train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main --iterations 20
```

`state.json` commits the iteration only after its replay, report and checkpoint
are written. Resume restores the last completed model, optimizer, RNG, setup cursor
and frozen pool. Complete cached collection batches in an interrupted iteration
are reused only if their provenance matches; an unfinished update restarts from
the last committed weights. CUDA scheduling may prevent bitwise reproduction of
an interrupted update. Do not edit an active run's configuration or learning code:
the signature deliberately rejects that resume. Start a new run from its last
checkpoint for an experiment with different settings.

Twenty iterations now collect 6,400 regular games. Start with two when first
measuring the larger preset. This is not a strength guarantee. Check progress against older neural generations. Preserve
older agents with distinct behavior even if they are not the current champion.
The main loop freezes candidates; it does not automatically label each new
generation stronger.

### Budget for a stronger opponent

The earlier estimate of 5,000 iterations applied to the **old 20-game preset**:
100,000 regular games plus 1,996 search games. That preset is preserved in
`configs/step5/lab-small.json`. The new default has 320 regular games per iteration;
5,000 of these would mean 1.6 million regular games. Use staged game/update budgets
rather than carrying the old iteration count over automatically.

The historical PPO algorithm accumulates gradients over a complete epoch, then
takes one optimizer step. Larger `minibatch_size` improves memory/throughput;
larger `games_per_iteration` changes the data per update. Equal total games do not
imply equal learning. Compare learner decisions, optimizer steps, validation
results and wall time as well as game counts. A substantial initial scaled budget
is approximately 313–500 iterations (100,160–160,000 additional regular games),
released in stages below and extended only while measured play improves.

| Total completed iterations | Regular games | Extra search games | Decision at this point |
| --- | --- | --- | --- |
| 2 | 640 | 0 | Measure fresh collection/update throughput and memory |
| 10 | 3,200 | 0 | Check economics, scores and early checkpoint progress |
| 50 | 16,000 | 64 | Compare frozen neural references and inspect industry use/search labels |
| 200 | 64,000 | 304 | Broader held-out comparisons and coherent-play checks |
| 313 | 100,160 | 488 | Substantial training-data budget; assess learning before extending |
| 500 | 160,000 | 784 | Extend only if useful progress continues |

These counts start at the new run's iteration zero; inherited training remains
recorded in its origin/history. Evaluate every 10–25 scaled iterations initially,
then every 50 once progress is clear. Use a stable reference panel and the same prespecified validation setups
to compare checkpoints; keep final-test setups unused until the end. A five-setup
evaluation is a trend check. Use the existing 30-setup, four-seat promotion gate
for release candidates (120 games per lineup). Stock/v3 wins alone are insufficient;
compare historical neural agents and inspect competent economic play too.
If progress stalls across several checkpoints, inspect value calibration,
exploration, complete-game behavior and search-label quality before increasing
the budget. Evaluate search-enabled and plain policy checkpoints separately.

`--iterations` means **additional iterations**, not a target total. After two
scaled iterations, `--iterations 8` reaches ten. Omit `--initial` and `--from-run`
on every normal resume. Keep the configuration unchanged within a run; use a new
run and `--from-run` for another batch/worker-count experiment.

Measure your lab's iteration reports for a time estimate. Once search has started,
one search iteration plus four ordinary iterations is a representative cycle;
use their combined seconds divided by five, multiplied by the remaining count.
Search can dominate runtime even with a fast GPU. The trainer also retains every
optimizer checkpoint and full replay: budget storage from measured bytes per
iteration, and do not delete history checkpoints while a live run references them.
The local medium-model optimizer checkpoints are about 22.8 MB each, so 500
generations alone take approximately 11.4 GB before replay and evaluation data.
Larger game collections also increase replay and in-memory preparation costs.

Reserve about **10–20% additional compute** for evaluation and independent-seed
replications. For early reproducibility checks, run two or three shorter
independent seeds before committing the full extension. RL training can vary
substantially with seed; see [Spinning Up's experimental guidance](https://spinningup.openai.com/en/latest/spinningup/spinningup.html).
Consider further scaled iterations only while validation shows useful progress.
Start meta-RL after the neural population is useful;
meta-RL outer iterations have a different workload and are not included in these
ordinary-training counts.

### Neural search distillation

The default starts additional search self-play at zero-based iteration 10, then
every five iterations. It searches every fourth turn of each seat and collects
eight extra games through eight independent search-game workers in the larger
preset (the preserved small preset uses two games and one search worker). Each
search has one 30-second turn allowance;
the returned plan covers both ordinary actions when two are available. A timeout
uses the neural fallback and produces **no fabricated teacher label**. Inspect
`confirmed`, `incomplete`, teacher-row counts and distillation's `updated` field.

Proposals come from the neural policy, with explicit action/industry coverage.
Candidates are replayed in matched worlds sampled from the root's permitted
observation. Each opponent acts from its own view using a frozen neural policy.
The simulation stops at the root's actual next decision, including changes in
spending order and debt/era transitions. It uses that root's allowed view for
the four-seat value prediction. Current-turn action two cannot vary with hidden
cards or future draws. Finalists are confirmed on fresh worlds; conditional
choice/continuation targets exclude unexamined choices from the policy-loss
denominator. Weak v3 choices are never treated as optimal search labels.

The design follows the search/imitation separation in
[Expert Iteration](https://arxiv.org/abs/1705.08439); this implementation is an
original, bounded Brass planner. R3's S2 underperformed S1 in its historical test,
so adding search is not assumed to improve strength. Compare `evaluate` with and
without `--search`; those commands measure each mode but do not by themselves
create an equal-time ablation. A longer search setting needs an equal-whole-turn
budget study before laptop deployment. `distill_weight` defaults to 0.25; PPO and
terminal results remain central. Early value errors and poor opponent models can
make search a poor teacher, which the reports must expose.

## Diverse-opponent evaluation

Use fresh validation setups, rotate the candidate through every seat, and retain
the same frozen references across repeated comparisons. The example below assumes
the named generations exist after the main run. Evaluation now defaults to the
dedicated `configs/step5/evaluation.json`: **64 setups, four seat rotations,
32 parallel neural simulators, inference batches up to 64, and eight independent
game workers for legacy/search lineups**. With two distinct neural references and
legacy enabled, this is **1,024 games across four lineups**. See the
[evaluation scaling guide](STEP5_EVALUATION_SCALING.md) for configuration and resume
details. Append `--setups 4 --no-include-legacy` with a separate output directory
for a 32-game progress check, or one setup and one reference for a four-game wiring check.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0015.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0010.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-scaled-g20

# The same candidate with the learned Pottery context, without forced Pottery moves.
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0015.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0010.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-scaled-g20-pottery --focus Pottery
```

The four lineups are recent neural opponents, mixed neural history, mixed admitted
v3 styles, and held-out Pottery Flex. The report separates each lineup, seat
exposure, win share, rank, VP, score margins and industry use. Confidence intervals
resample whole setup groups; rotations are not independent samples. A failed game
aborts the evaluation without publishing a complete report. Its successfully
written batches can be resumed under the same spec; failures are not scored as wins.

`--partition test --seed 1910000` reserves a final test partition. Do not use test
results for training, tuning or promotion. Evaluation shards are marked and the
R2 training reader rejects them. Promotion currently requires validation results
over at least 30 setups, two distinct **trained** neural reference identities,
a lower clustered win-share bound above 25% in each neural lineup, and no
Pottery-collapse warning. A win over fresh initialization or v3 alone is insufficient.
These gates still do not certify human-level or enjoyable play.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py promote --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --report /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-scaled-g20/report.json
```

The 64-setup default meets the sample-count requirement; the other promotion gates
must still pass. Small progress checks cannot qualify promotion. Prespecify the
setup count; do not keep adding games until one favorable interval appears.

## Optional meta-RL: learn to adapt between games

Meta-RL here is a **Reptile-style first-order experiment**. A task is one frozen
three-opponent neural lineup. Clone the initialization, make at least two PPO
updates on fresh support games against that task, then move the initialization
toward the mean adapted weights across tasks. This adapts the method described
in [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999).
It is not a claim that ordinary self-play alone is meta-learning.

Query games use different setups and never contribute gradients. The logs show
before/after query performance, including deterioration. No opponent identity,
private hand or engine seed is fed into the model. Adaptation uses the learner's
own permitted trajectories and terminal outcomes. Start after ordinary self-play
has produced useful, distinct neural checkpoints; weak initial tasks do not
provide evidence of useful adaptation.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda meta-train --initial /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0010.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0015.pt --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/meta --iterations 5 --tasks 3 --inner-steps 2 --support-games 4 --query-games 4 --fraction 0.1
```

Repeat the same command to add iterations to that meta run. Each task in the
example uses eight support games and four before plus four after query games:
16 games per task. Meta-training is additional work, not a free replacement for
the main self-play budget. The first implementation keeps adapted models for a
small task batch in memory; reduce tasks if memory is constrained.

Evaluate on an unseen lineup, for example using additional held-out frozen neural
versions. Replace the example paths below with versions you actually froze and
kept outside meta-training. The CLI rejects an exact lineup recorded in the meta
checkpoint's task history.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda meta-evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/meta/meta-generation-0005.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0008.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0012.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0018.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/meta-evaluation --inner-steps 2 --support-games 4 --query-games 8
```

Meta-evaluation updates a disposable clone only. Support shards are marked
evaluation-only, query games are untouched by gradients, the original checkpoint
is verified unchanged, and adapted weights are not published. Compare many
held-out lineups with the same support budget, including ordinary PPO initialization
as a control. The current experiment adapts **between games** using gradient
updates. It does not implement recurrent within-game RL², hidden opponent labels,
or an on-laptop adaptive release.

## Export and keep the evidence

Use R2's checked ONNX export with real permitted-view examples from the run:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/r2.py export --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --shards /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/iteration-0019/actor-0000.jsonl.gz --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/model-g20.onnx
```

Keep ordinary run data under ignored `artifacts/step5/`. Select weights and their
model/evaluation manifests for `training/models/releases/` deliberately; do not
upload every replay or optimizer snapshot. Preserve the R3 frozen banks and the
R2 bootstrap manifests. Step 5 does not install a trained neural model into the
game; laptop search/inference integration, calibrated difficulty and human
playtesting remain the later milestones in the main plan.
