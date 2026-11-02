# Fresh neutral self-play from g1290

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
