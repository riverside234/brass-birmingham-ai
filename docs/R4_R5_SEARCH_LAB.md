# R4 / R5 additional search experiments

29 September 2026. R4 and S3 are implemented as additional experiments alongside
Step 5. Running them does not resume, edit or replace the current population
trainer, its configuration, its frozen v3 opponents or its checkpoints. Their
entry point is `scripts/r4r5.py`, with separate outputs under
`/data/not_backed_up/yxu209/ai-training/artifacts/searchlab/`.

Implementation is required; promotion is conditional. A working search is not
evidence of stronger play. See [executed local checks](R4_R5_EXIT_CHECKS.md).

The lab preset now targets the reported **64 logical CPUs and 90 GB GPU**. It
runs comparisons and S3 teacher collection in separate spawned processes, with
larger game counts and distillation minibatches. The **quick smoke experiment**
still makes just one decision per variant. It is the quick execution check;
the larger commands below are deliberate additional experiments using frozen
Step 5 checkpoints. See [scaling check results](R4_R5_SCALING_CHECKS.md).

## What is implemented

| Variant | Decisions and information | Purpose |
| --- | --- | --- |
| S2 control | Existing complete-turn proposals, matched hidden worlds, independently masked opponent replies, fresh finalist confirmation | Keep the current training/search baseline unchanged |
| R4 OpenSpiel | Actual OpenSpiel ISMCTS node/prior/PUCT and resampling interfaces; root information-state decisions; fixed masked opponent policies; four-seat returns | Independent algorithm comparison using the same rules and model |
| R5 S3 | Complete root-turn plans, root sampling, future root action/observation nodes, bounded particle histories, own-seat UCT selection, four-seat backups, fresh confirmation | Test whether one or two additional future root turns justify their cost |

S3 depends on the existing S2 interfaces, not on the OpenSpiel adapter. Its
implementation uses a finite horizon and neural leaves. It is **POMCP-inspired**,
not a claim of convergence or a multiplayer equilibrium algorithm. Opponents act
for themselves from their own allowed observations. Root-private information
does not enter opponent search statistics: R4 deliberately has no optimized
opponent nodes. S3 opponents are also fixed observation-based rollout policies.

S3 commits the same complete current-turn plan across sampled worlds. A later
root turn may branch on its newly observed hand and public actions. Its finalist
confirmation freezes the continuation tree and uses fresh samples; unseen future
observations use the same neural fallback. Timed-out or unconfirmed searches
produce no positive-confidence distillation labels. R4 visit counts are never
exported as teacher probabilities.

The separate `BrassSpielGame`/`BrassSpielState` adapter implements four-player
sequential imperfect information, terminal win shares, semantic legal IDs,
isolated reconstruction-based clones, per-seat private hand recall, serialization
and belief resampling. Setup order, merchant placement, initial cards, ordinary
draws and the Canal-to-Rail shuffle are explicit incremental chance events with
normalized probabilities. No private card callback is broadcast to other bots.
Semantic IDs are deterministic bounded hashes; any collision fails explicitly.
They are separate from dense neural candidate indices.

`src/BrassSearchHost/` adds pause/resume support to the project-authored R1 host
in a **separate** worker. `scripts/build-search-host.py` creates the generated
host under ignored artifacts and exports only the authored executable/manifest
under `training/runtime/search/`. The R1 worker, game installation and frozen
R3 workers are unchanged. Both workers use the preserved rule-patch 1.1.0 engine.
The already-installed UI/AI package remains a separate game runtime.

R0's OpenSpiel source pin remains immutable. The executable experiment dependency
is pinned separately to `open_spiel==2.0.2`. Adapted mechanics carry the
[OpenSpiel notice](../src/brass_ai/searchlab/NOTICE.md) and Apache-2.0 license.
References: [OpenSpiel installation](https://openspiel.readthedocs.io/en/latest/install.html)
and [Silver and Veness, POMCP (2010)](https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html).

## Install in the existing conda environment

Keep the working `ai-training` environment and Step 5 private runtime. Run these
after the new source and `training/runtime/search/` have been transferred to the
lab checkout. Files being unignored does not mean they have been committed or
pushed. No game DLL is added to Git.

```bash
conda activate ai-training
python -m pip install -r /data/not_backed_up/yxu209/ai-training/requirements-searchlab.txt
python /data/not_backed_up/yxu209/ai-training/scripts/r4r5.py install-runtime
```

`install-runtime` verifies and reuses dependencies in
`/data/not_backed_up/yxu209/ai-training/artifacts/r1-host/guarded/`, then installs
the separate worker in `artifacts/r4-host/`. It refuses changed dependencies or
an incompatible existing worker. A fresh clone first needs the private import
and baseline preflight from [the Step 5 guide](STEP5_LAB_TRAINING.md).
The lab must have a working Mono runtime. Pass `--mono /absolute/path/to/mono`
consistently to verification and experiments if it is not on PATH.

## Larger lab configuration

`r4r5.py` defaults to [configs/searchlab/experiment.json](../configs/searchlab/experiment.json).
The previous small settings remain available in [small.json](../configs/searchlab/small.json).
Pass global `--config /data/not_backed_up/yxu209/ai-training/configs/searchlab/small.json`
**before** `compare` or `distill-s3` to use them.

| Setting | Larger default | Purpose |
| --- | --- | --- |
| `setups` | **64** | Held-out setups, each with four seat rotations and S2/R4/S3: **768 games**. |
| `workers` | **8** | Independent comparison processes, each running one game at a time. |
| `distill_games` | **64** | Fresh train-partition games for the separate S3 teacher experiment. |
| `distill_workers` | **8** | Independent S3 teacher processes; all four seats use frozen S3 policies. |
| `threads` | **1** | PyTorch CPU threads per search process. |
| `simulations` | **128** | R4/S3 work cap, still bounded by the same whole-turn clock. S2 has its existing proposal/sample limits. |
| `batch_size` | **64** | Shared same-model inference batch limit. |
| `update.effective_batch_size` | **1024** | Optimizer batch for revised S3 distillation. |
| `update.microbatch_size` | **512** | Memory microbatch, also bounded by candidate cells. |
| `max_candidate_cells` | **131072** | Bound on padded legal-choice batches; oversized minibatches split without dropping choices. |
| `distill_steps` | **8** | Update steps after completed teacher collection. |
| `seconds` | **3** | Whole-turn allowance, shared by both actions; override for 1/3/10-second comparisons. |
| `future_turns` | **1** | Initial R4/S3 future-root-turn horizon; compare two separately. |

The default now uses schema `searchlab-experiment/1.2.0`. Search workers retain
private CPU models for identity/context, rules hosts, trees and RNG contexts;
one coordinator service owns device models and dynamically batches same-model
requests. Processes use `spawn` on Linux and Windows. Each game gets fresh
policy context and a schedule-independent sampling seed. Workers publish
validated completed replay shards.

Independent search requests share inference, with a 2 ms partial flush and a
131,072 padded-candidate-cell limit. R4 caches combined policy/value predictions
for the same complete permitted input, including own cards, context, model and
ordered choices. Replies preserve identity, seat, revision and deadlines. Start
with eight search processes and one tensor thread per worker; try 16 only after
measuring CPU use, confirmed searches, memory and fresh-run games/second.
Concurrency can reduce work finished per timed search even when aggregate
throughput improves. There is no promise to fill 90 GB of GPU memory.

This revision also changes shared Step 5 source fingerprints. Use the fresh-run
migration in [TRAINING_RECOVERY_LAB.md](TRAINING_RECOVERY_LAB.md). Existing R4/R5
correctness records become stale; requalify the lab device before a full
comparison or teacher run. Use fresh output directories. The prior schema is
preserved in `previous-experiment.json` for historical comparisons.

## Quick smoke experiment

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-searchlab.py --device cpu --seconds 1

# Exercise spawned workers on the lab GPU: still only three decisions total.
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-searchlab.py --device cuda --seconds 1 --workers 2

# Current central-inference path, still just three decisions.
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-searchlab.py --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/searchlab/experiment.json \
  --seconds 1 --workers 3
```

This runs **one decision each** for S2, R4 and S3 against the same initial
four-player position. It checks checkpoint compatibility and that the rules
host accepts each selected choice, then prints timing and search-completion
telemetry. It uses the distributed smoke weights by default; pass `--checkpoint`
with an absolute path to a completed main-run snapshot to try that model.
Use `--device cuda` to try the lab GPU. No output checkpoint, replay, promotion,
training update or full-game match is produced. A fallback legal move is an
acceptable wiring result; it is not proof of useful search or stronger play.
The smoke command uses the small preset independently of the larger defaults.
The full integration suite is **not required for this quick experiment**.

## Full correctness gate — only before longer comparisons

When ready to measure strength, qualify the selected machine/device first:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/verify-searchlab.py --integration --device cpu
python /data/not_backed_up/yxu209/ai-training/scripts/verify-searchlab.py --integration --device cuda
```

Run only the device you will compare. The full local CPU check took about eight
minutes; it is deliberately separate from the quick smoke command.

The verifier checks a complete R1-versus-R4 game with matched random outcomes,
legal choices, restoration, private chance delivery and all four allowed views;
it then checks hidden-state counterfactual invariance in both planners for every
seat. The report binds source, runtime, Step 5 implementation, OpenSpiel, Python,
PyTorch, OS and device. It is stored as `correctness-cpu.json` or
`correctness-cuda.json` under `artifacts/searchlab/`. A source/runtime/device
change requires verification again. Do not copy a Windows pass to Linux.

## Larger paired comparison

After the correctness gate passes for CUDA, choose existing immutable trained
snapshots. The following example runs **64 setups × four seats × three variants
= 768 complete games**, on eight worker processes:

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/r4r5.py \
  --device cuda \
  --config /data/not_backed_up/yxu209/ai-training/configs/searchlab/experiment.json \
  compare \
  --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled/generation-0050.pt \
  --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled/generation-0025.pt /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled/generation-0010.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/searchlab/scaled-g50-3s
```

Replace checkpoint names with files that exist. For a smaller check using the
same runner, append `--setups 1 --workers 2` and use a separate output directory:
12 complete games. This is still much longer than the three-decision smoke.
The `--workers` option overrides comparison concurrency; it does not change
setup identities or RNG seeds. The default variant list retains the S2 control.

Parallel runs are **screening evidence**. Every variant has the same time cap,
but shared CPU/GPU contention can distort its work allowance. Reports mark
parallel runs ineligible for promotion. Use `--workers 1` with a new output
directory to confirm a selected comparison at equal budgets, with no competing
training workload. Reducing workers on an existing directory is a spec change,
so cached parallel games cannot be reused as serial timing evidence.

## Small paired comparison using smoke weights

The distributed smoke weights let this command run without waiting for main
training. They are not strong opponents and are not suitable strength evidence.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/r4r5.py --device cpu compare \
  --checkpoint /data/not_backed_up/yxu209/ai-training/training/models/releases/step5-selfplay-medium-smoke.pt \
  --references /data/not_backed_up/yxu209/ai-training/training/models/bootstrap/r2-fresh-medium.pt /data/not_backed_up/yxu209/ai-training/training/models/bootstrap/r2-onpolicy-scaffold.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/searchlab/wiring-3s \
  --setups 1 --workers 1 --seconds 3 --simulations 32 --future-turns 1
```

This is a longer tournament command, not the quick smoke above. One setup means
four candidate seat rotations for each of S2/R4/S3: **12 complete
games**. The other seats are frozen neural references. All variants share the
same candidate weights, references, setups, rotation seeds, rules, utility and
whole-turn allowance. Jobs use a reproducibly shuffled variant order. With
`--workers 1` they execute sequentially; with more workers they finish in any
order but retain the prescribed setup/seat assignments. A completed
record can be resumed with the exact same command. Changing any setting,
checkpoint, device or dependency requires a new output directory. Failures are
retained and prevent promotion; a later success does not erase a failed trial.

At the default allowance, 90% is reserved for search and 10% for live validation.
Recorded whole-turn time includes observation/choice generation, search,
inference and live action application. Action two shares the original deadline.
Python/IPC/inference calls cannot be interrupted in the middle of a call; actual
overshoots are recorded and disqualify a claimed equal-budget advantage.
Startup/model-loading cost is outside the per-turn timing. Do not run a formal
timing comparison concurrently with main training on the same saturated GPU or
CPU. Taking a frozen snapshot for later experiments does not require stopping
or modifying the training run.

## Compare trained snapshots at equal budgets

Choose a completed, immutable checkpoint from the main run, plus at least two
frozen earlier neural opponents. Use their full paths in `--checkpoint` and
`--references` above. The initial main weights, stock Hard and weak scaffold
weights are not adequate final strength benchmarks. This experiment does not
increase the legacy AI's training share or remove Pottery from the action space.

Prespecify the comparison, then run:

1. One setup to check the selected checkpoints and device.
2. Parallel validation screening at 1, 3 and 10 seconds in distinct output
   directories: the 64-setup default is 768 games per three-variant comparison.
   Use the same validation seed across variants; the runner already pairs them.
3. Compare S3's `--future-turns 1` and `--future-turns 2` in distinct directories,
   keeping the same whole-turn cap. The work cap is now 128; try 256 only
   when telemetry shows the work limit is reached before the clock. More work
   is not assumed better.
4. Confirm the selected comparison with `--workers 1 --setups 30` or more in a
   new directory. This is 360 games with all three variants, or 240 with
   `--variants s2 s3` / `--variants s2 r4`. Only this uncontended serial timing
   comparison can meet the existing promotion gates.
5. Lock one candidate configuration before a fresh `--partition test` run with a
   different `--seed` (for example `4930000`). Test reports are final reporting
   evidence; the promotion command refuses using them for model selection.

Thirty setups is a minimum screening threshold, not sufficient evidence for a
human-level or production difficulty claim. Use more held-out setup groups and
mixed opponent populations for a release. The main plan's larger acceptance
tournament and human playtests still apply. Repeated configuration selection on
the same validation seeds must be followed by the untouched test partition.

`report.json` records first-place share, mean rank/VP, paired win-share difference
against S2 with a setup-cluster bootstrap interval, timing, errors and Pottery
opportunity/build counts. It also records worker count, execution mode, elapsed
time and per-process peak allocated CUDA memory. Per-process peaks are not total
GPU memory, and resumed throughput includes cached games. Shared inference
reports queue/batch/pack/forward/result timings and actual versus padded
candidate cells. Per-match files preserve
search completion, reply
counts, horizons and tree statistics; compressed complete-game replays are
marked **evaluation**, and the training loader rejects them as training data.
Pottery warnings detect zero use over many legal opportunities; they do not force
Pottery in every game or claim an optimal strategy frequency.

## Evidence-gated promotion

Promotion requires a current correctness pass, a **one-worker comparison**,
at least 30 paired validation
setups with all four rotations, a positive lower 95% bound for paired win-share
improvement over S2, no failed matches, no recorded over-budget turns in the
candidate or control, and no aggregate Pottery zero-use warning. Source, model,
opponent and replay hashes are checked again.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/r4r5.py --device cpu promote \
  --report /data/not_backed_up/yxu209/ai-training/artifacts/searchlab/validation-3s/report.json \
  --variant s3 \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/searchlab/selected-s3.json
```

Use the actual comparison directory; `validation-3s` is an example. For a CUDA
comparison use `--device cuda`. Even a successful command writes only an
experiment selection record. It never edits main training, its checkpoints, the
frozen bank or Unity's AI. Deployment and difficulty/personality calibration
remain separate measured decisions. A refused promotion is an expected result
when deeper search does not help.

## Additional S3 distillation experiment

This standalone path collects new train-partition games, uses only completed
fresh-confirmation targets, and applies the existing coverage/confidence-aware
distillation update to an in-memory copy of the checkpoint. Its output must be
separate from training. The first run uses an empty directory. Completed per-game
teacher shards can be resumed with the exact spec; a completed report is reused
without applying the update twice. The source checkpoint and main training state
remain unchanged. The default is **64 games and eight workers**, followed by
confirmed-target updates on the coordinator with effective batches of **1,024**,
memory microbatches up to **512**, and a cap of **eight optimizer steps**.
Reports include actual useful rows and optimizer steps; a nominal budget cannot
create labels when search confirmation fails.

```bash
python /data/not_backed_up/yxu209/ai-training/scripts/r4r5.py --device cuda distill-s3 \
  --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/main-scaled/generation-0050.pt \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/searchlab/s3-teacher-g50-10s \
  --seconds 10 --future-turns 1
```

Use a trained frozen snapshot that exists. For a small two-game check, append
`--games 2 --workers 2 --minibatch-size 64 --distill-steps 4` and use another
output directory. Here `--workers` overrides `distill_workers`; it does not alter
the comparison setting. Teacher replays are named `teacher-<seed>.jsonl.gz`.

If the time allowance never finishes confirmation, the report explicitly says
`updated: false`; a
copied candidate checkpoint is not a trained improvement. Evaluate a produced
`candidate.pt` against its unchanged parent and diverse opponents on fresh
held-out setups before considering integration. These extra S3 games do not
change the main run's self-play-first schedule or optional meta-RL path.

## Known practical limits

The reconstruction-based host is correctness-first and can dominate search cost.
Short budgets can fall back to the neural policy without completing useful
search. Inspect confirmed searches, completed simulations, reached horizons and
actual turn time before spending a large lab budget. Search quality also depends
on the trained value model, belief approximation and frozen opponent pool.
Neither framework choice nor depth establishes better play. The local Windows
checks do not qualify Linux/Mono, the RTX GPU, Intel inference or the Unity UI;
run the corresponding lab/laptop checks on those machines.
