# Step 5: self-play, search distillation and meta-RL on the lab computer

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

Use Python 3.12, a working `mono` command for the preserved .NET host, and an
NVIDIA driver compatible with the pinned PyTorch CUDA wheel. These commands
retain the tested R2 dependency versions. The CUDA 12.8 installation is listed in
[PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/).

```bash
python3.12 -m venv /data/not_backed_up/yxu209/ai-training/.venv
/data/not_backed_up/yxu209/ai-training/.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
/data/not_backed_up/yxu209/ai-training/.venv/bin/python -m pip install -r /data/not_backed_up/yxu209/ai-training/requirements-step5.txt

/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py import-runtime --bundle /data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime.zip
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/verify-step5.py
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/check-transferred-legacy.py
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

## Start with a short self-play run

The smoke configuration uses four games per iteration, a medium backbone, two
PPO epochs, pure neural opposition and no scheduled search. This measures the
host/inference/replay/update workflow before committing to a longer run.

```bash
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/smoke.json train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/lab-smoke --initial /data/not_backed_up/yxu209/ai-training/training/models/bootstrap/r2-fresh-medium.pt --iterations 2
```

Inspect both iteration reports: all games must have completed; gradients and
losses must be finite; learner trajectories and context/build counts should be
present. GPU utilization can be modest because legal moves, IPC and rules
simulation run on the CPU. More GPU memory alone does not increase game throughput.
The first implementation batches inference across a small group of game workers
and bounds replay memory to an iteration; it is not a distributed actor service.
Measure seconds per iteration before increasing workers or games.

## Run and resume the main population trainer

```bash
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main --initial /data/not_backed_up/yxu209/ai-training/training/models/bootstrap/r2-fresh-medium.pt --iterations 20

# Resume for another 20 iterations; omit --initial.
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/main --iterations 20
```

`state.json` commits the iteration only after its replay, report and checkpoint
are written. Resume restores the last completed model, optimizer, RNG, setup cursor
and frozen pool. Complete cached collection batches in an interrupted iteration
are reused only if their provenance matches; an unfinished update restarts from
the last committed weights. CUDA scheduling may prevent bitwise reproduction of
an interrupted update. Do not edit an active run's configuration or learning code:
the signature deliberately rejects that resume. Start a new run from its last
checkpoint for an experiment with different settings.

Twenty iterations are an initial experiment, not a training-duration prescription
or strength guarantee. Check progress against older neural generations. Preserve
older agents with distinct behavior even if they are not the current champion.
The main loop freezes candidates; it does not automatically label each new
generation stronger.

### Neural search distillation

The default starts additional search self-play at zero-based iteration 10, then
every five iterations. It searches every fourth turn of each seat and collects
two extra games per search batch. Each search has one 30-second turn allowance;
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
the named generations exist after the main run. Five setups means 20 games **per
lineup**, not 20 total. With all four lineups it is 80 games. Start with one setup
and omit `--include-legacy` for a small wiring check.

```bash
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0015.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0010.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-g20 --setups 5 --include-legacy

# The same candidate with the learned Pottery context, without forced Pottery moves.
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0015.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-g20-pottery --setups 5 --focus Pottery
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
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py promote --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --report /data/not_backed_up/yxu209/ai-training/artifacts/step5/evaluation-g20/report.json
```

The five-setup example deliberately will not pass that promotion gate. Increase
the prespecified setup count for a real promotion attempt; do not keep adding
games until one favorable interval appears.

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
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda meta-train --initial /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0010.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0015.pt --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/meta --iterations 5 --tasks 3 --inner-steps 2 --support-games 4 --query-games 4 --fraction 0.1
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
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda meta-evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/meta/meta-generation-0005.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0008.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0012.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0018.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/meta-evaluation --inner-steps 2 --support-games 4 --query-games 8
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
/data/not_backed_up/yxu209/ai-training/.venv/bin/python /data/not_backed_up/yxu209/ai-training/scripts/r2.py export --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/generation-0020.pt --shards /data/not_backed_up/yxu209/ai-training/artifacts/step5/main/iteration-0019/actor-0000.jsonl.gz --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/model-g20.onnx
```

Keep ordinary run data under ignored `artifacts/step5/`. Select weights and their
model/evaluation manifests for `training/models/releases/` deliberately; do not
upload every replay or optimizer snapshot. Preserve the R3 frozen banks and the
R2 bootstrap manifests. Step 5 does not install a trained neural model into the
game; laptop search/inference integration, calibrated difficulty and human
playtesting remain the later milestones in the main plan.
