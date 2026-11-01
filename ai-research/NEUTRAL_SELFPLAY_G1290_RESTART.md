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
| Branch frequency | Existing **1%** per eligible decision; existing fresh-run activation after 20 iterations |
| DOMiNO | Off: no intrinsic companion, dual updates, context sampling or context-evaluation panels |
| Policy seat tags | Neutral style `[0,0,0,0]` and intent `[0,0,0,0,0,0,0,0]` for every neural seat |
| Critic fingerprints | Existing 28 zero columns; role/age descriptors and dimensions preserved |
| AnchorKL | Measurement, calibration and LR gating remain off |

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
into the learner checkpoint. The separate auxiliary ownership heads start fresh.
The seed's utility calibration is invalidated for the new reward; ordinary
head-only calibration follows the existing training rules.

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
its committed checkpoints are read. Run from the repository root:

```bash
python scripts/prepare-neutral-selfplay.py \
  --source-run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-source-generation 1360 \
  --seed-generation 1290 --history-through 1330 \
  --run artifacts/step5/joint7-v11-2-neutral-from-g1290 \
  --output configs/step5/joint7-v11-2-neutral-from-g1290.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-neutral-from-g1290.json train \
  --run artifacts/step5/joint7-v11-2-neutral-from-g1290 \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --fresh --iterations 20
```

These are **new generations 1–20**, not g1361–g1380. The `--imitation-run` argument
retains the completed IL's reference/rehearsal identity; the verified seed
manifest selects g1290 as the learner initialization. The helper prints the
imported/excluded generations and does not launch training.

After 20 completed generations, continue the new run by omitting the preparation
step and replacing `--fresh` with `--expected-generation 20`. If a later command
is interrupted mid-iteration, use the existing recovery helper on the **new run**
with its own committed generation, never 1360.

Export its reports explicitly:

```bash
python scripts/export-training-reports.py \
  --run artifacts/step5/joint7-v11-2-neutral-from-g1290
```

The exporter now includes `fresh-start.json` with source/cutoff receipts, alongside
the ordinary reports. It never exports checkpoint tensors or PPG arrays.

## Validation scope

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

An isolated CPU smoke uses authored fixtures: one PPO step and two accepted PPG
steps, with no DOMiNO or AnchorKL calls. A separate initialization smoke creates
and reloads a generation-zero run from synthetic retained weights, checks exact
tensor equality and verifies that its source files are unchanged. These checks
do not demonstrate improved playing strength or measure production throughput.
The actual Linux g1290 weights are not present in this desktop checkout; the
preparation command verifies them before writing the final usable configuration.
