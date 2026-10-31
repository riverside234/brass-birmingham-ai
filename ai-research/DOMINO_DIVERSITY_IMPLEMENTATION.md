# DOMiNO-based episodic PPO: g1326 adoption

Implemented 2026-10-30 against the g1326 export
`results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z`.
This implements D0–D4 of [the plan](DOMINO_DIVERSITY_TRAINING_PLAN.md).
D5, empirical strategy and playing-strength qualification, remains a separate experiment.

Follow-up: [debug review and performance fixes](DOMINO_DIVERSITY_DEBUG_REVIEW.md).
Latest: [g1327–g1330 log review and g1330 continuation](SELFPLAY_G1330_REVIEW.md).
Already adopted runs use their existing config and a reviewed source continuation;
they do not repeat the g1326 adoption or reset compatible PPG replay.

**Archived g1326 command update:** at the time of this update, adoption had not run.
The command below now explicitly sets all four fixed PPO entropy coefficients to
**0.10**, through `--entropy-010` and `fixed-group-entropy/1.1.0`. Adaptive entropy
stays off; its saved baseline is retained. Fresh config generation uses the same
four coefficients. Existing signed configs keep their original values until an
explicit reviewed transition. The new output filename avoids overwriting a
previously generated DOMiNO config with different settings.

The entropy follow-up passed six focused tests (19.82 s), including raw-policy
entropy gradients and masks, baseline/report preservation, strict old/new config
validation, the generation-bound helper, and one isolated authored PPO/PPG
iteration. Log: `artifacts/domino-entropy-tests.log`. No production checkpoint was
modified. This changes a training coefficient, not a demonstrated strength result.

## Behavior and objective

Every ordinary neural seat receives one IID uniform ID, including frozen history
and IL opponents: neutral, CottonMill, Manufacturer, Pottery, Brewery, IronWorks,
CoalMine. Assignment uses its own deterministic stream keyed by run seed,
committed ordinary-game index and final physical seat. It does not consume PFSP,
action, PPO or initialization RNG. V4 seats have no neural ID. Branches inherit
contexts and retain their existing three-decision post-fork opening window.

An explicit context-bound behavior family and seat-policy receipt accompanies the
legacy policy identifier. Model identity and learner/frozen role remain separate,
including equal-weight learner and opponent models. Python collection, native
collection, replay admission, critic descriptors and disk replay verify these
receipts. Evaluation, teaching and the fixed anchor probe remain unchanged.

The versioned descriptor has 112 observable features per phase, 448 coordinates
per centroid. The confirmed phases are Canal rounds 1–4 / 5–8 and Rail rounds
1–6 / 7–8. Main-action family, own investment/development state, realized versus
unrealized assets, financing and spatial/network footprint are separately scaled.
Each complete seat-game gives each phase weight 1/4, divided among that phase's
main actions. Payment records cannot inflate occupancy; a missing terminal phase
is explicitly zero. No outcome, model/strategy label or hidden opponent state
enters the descriptor. The source-field and scaling contract is in
`configs/learning/domino-descriptor-1.0.0.json`.

The intrinsic direction is the frozen prior committed centroid minus its nearest
other centroid, with deterministic tie handling. Coincident centroids yield zero;
neutral has exactly zero intrinsic reward and task coefficient 1. Six other task
coefficients use sigmoid logits, initially zero, with separate float64 Adam at
0.01. The quality constraint is each strategy's ordinary combined task return
against 0.95 times neutral's return. The existing insolvency cost is included once.
One dual update uses the prior committed estimates; new statistics then enter the
0.9 EMA. Bootstrap stays task-only until every ID has at least 32 training
seat-games, 16 distinct games, complete finite estimates and positive neutral
quality. Reward/descriptor/context/table-share changes archive the old evidence
and repeat readiness; ordinary historical-pool turnover does not reset it.

Task GAE is unchanged. Intrinsic advantages use undiscounted MC reward-to-go and
the independent intrinsic baseline. Streams combine before the existing single
advantage normalization. The current decoupled T2 opening objective and actual
epsilon-inclusive behavior checks remain in place. Branch actor advantages stay
task-only; historical seats never become learner actor data.

The training-only head is `(backbone + acting descriptor) → 64 → 1`, with a
zero-initialized output and isolated initialization RNG. Its PPO preservation
gradient reaches the trunk and true descriptor encoder. PPG fits only current
ordinary learner training rows, including forced critic rows. Older ages,
snapshots, IL/bots, branches and whole-game holdouts cannot become intrinsic
targets. Both auxiliary Adam states and the head participate in keep-best and
rollback. Task-value selection and all policy KL caps remain authoritative.
The base network and exported ONNX tensor interface are unchanged; export metadata
now identifies the supported existing context semantics.

## Continuation and preserved recipe

Run from the lab repository root, after updating the code:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 1326 --apply && \
python scripts/configure-domino.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 1326 --mode apply --entropy-010 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-domino-entropy010-g1326.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-domino-entropy010-g1326.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1326 --iterations 20
```

This starts g1327 and requests 20 iterations through g1346. It does not rewind to
g1324. The helper locks and reads the real state, checks the generation and complete
checkpoint SHA, inventories every selectable frozen checkpoint, verifies the
reviewed source/config transition and writes only the new configuration. Repeating
it at the same pre-adoption generation is idempotent. It refuses stale generations,
missing checkpoints, incompatible context interfaces and conflicting output files.
The local export contains reports rather than the lab's full g1326 checkpoint;
the actual checkpoint verification therefore runs on the lab.

The config changes are `diversity` and the explicit four-group fixed entropy
block. The current two-age PPG phase still runs
every iteration, for at most two epochs, with loss checks every 16 steps and KL
checks every 32. Saved beta, value Adam, PPO Adam, warmup/LR multiplier, local-head
Adam, RNG, weights, games/iteration counters, population, lineages, economic
baselines, GAE/snapshot evidence and qualified KL reference remain intact.
Epsilon stays 0.08, three opening actions stay T2, branching stays 1% with 50% T2 /
50% q, entropy is now 0.10 for all four groups, insolvency stays
0.005 per event with cap 0.05, and snapshot weight stays 1.0.

At adoption, the old neutral-opponent PFSP S/C values are archived and the
uniform-context estimator starts from the existing prior/half-life. Its initial
draw weights therefore change intentionally. Membership, admission ages, IL gate,
capacity and qualified v4 table-share settings do not change. PFSP still has one
member per checkpoint and records model probability separately from 1/7 context
probability. No sevenfold population expansion occurs.

The first new PPG window contains only g1327; g1328 onward uses the latest two
new-context collections. Previous manifests are archived. Old owned caches survive
adoption and failed updates, and expire only after the new checkpoint and state
commit. Incompatible pending actor collections follow existing recovery/freshness
rules; they are never relabelled. The original IL directory is unchanged.

## Reports and validation

Iteration and retained compact history contain readiness, assignments/completions
by ID/table/role/seat, PFSP context outcomes, task quality/support, coefficients,
constraint residuals, centroid distances/ties, intrinsic reward/return and advantage
scales, preservation drift, actual fitting support/steps, held-out intrinsic error,
PPO/PPG KL by ID, fingerprint work and cache bytes. Existing economic, KL, EV/R²,
memory and phase timings remain. The short console field is `DOMiNO=mode:readiness`.
`python scripts/export-training-reports.py` also copies the adoption receipt and
retains the new report sections, without copying model weights or replay arrays.

Validation completed on CPU; no active training checkpoint was changed:

- **153 distinct focused regressions passed**, across the recovery suite and
  follow-up native, ONNX and report-export suites. Coverage includes descriptor
  phase normalization, neutral/bootstrap behavior, dual signs and nearest ties,
  independent context RNG, same-weight learner/frozen identities, branch
  inheritance, strict behavior checks, intrinsic gradient/masking and joint
  rollback, checkpoint inventory, shadow-mode weight/Adam/RNG parity, deferred
  cache expiry, and generation-bound adoption. Archived resume tests now pin
  their historical approved target or the actual compatible g1326 source;
  production approval rules were not broadened to satisfy those fixtures.
- **Authored training smoke:** three bounded iterations, plus an exact repeat of
  the third after save/reload. Each collection used 126 synthetic complete tables,
  one PPO step and at most six PPG steps. PPG retained 4/6/6 steps. The first
  iteration was task-only; the second used ready prior statistics. Base weights,
  both base Adam states, both auxiliary heads and their Adam states, controller
  state and RNG matched exactly after restart. Wall time was **122.67 s** under
  concurrent CPU load. Artifact: `artifacts/domino-smoke-final/report.json`.
- **Native collection smoke:** **28 unique ordinary tables** total, with matched
  setup schedules: 14 self-play, 12 history and 2 search-free v4 tables. It used
  representative 512-wide random models, four lanes and real continuations
  (three control branches and four DOMiNO branches). The DOMiNO arm recorded
  **1,302 descriptor events**; neural-seat support for IDs 0–6 was
  **11 / 6 / 7 / 9 / 8 / 6 / 7**. Sealed family retries, replay admission,
  actual contexts and identities passed. The final invocation took **467.80 s**
  including setup/validation. Its control arm reused 14 cached families and the
  DOMiNO arm reused three, so their 77.77/238.10 s arm times are **not a valid
  throughput comparison**. The new work issued 1,706 inference requests with
  only two resident model identities; the largest reported host RSS was
  **3.45 GB**, sampled rather than an OS peak. Artifact:
  `artifacts/domino-collection-03/report.json`.
- **Fingerprint microbenchmark:** 64 real smoke positions, 3,672 legal actions,
  the 5,928,616-parameter utility model plus one frozen model, all seven contexts.
  Neutral took **0.208 s**; the remaining 13 cold context passes took **2.587 s**.
  A repeat took **0.156 s**, with **14 hits / zero forward passes**, unchanged RNG
  and one model load per weight identity. Sampled host RSS was **477 MB**.
  Artifact: `artifacts/domino-fingerprints-01/report.json`. This small CPU test
  does not estimate the lab GPU's 2,048-position probe latency.
- ONNX base/utility export parity, default descriptor handling, masking and fused
  execution checks passed. Report export retains DOMiNO adoption and compact
  evidence. `git diff --check` passed. The exact historical source-review prefix
  (75,124 entries) remains unchanged; 120 narrow continuation entries were added.

For the current 512-wide backbone, the training-only intrinsic head adds
**34,945 parameters**. It does not enter the inference model or ONNX graph. The
training smoke uses the smaller fixture head (10,369 parameters) to bound work;
native collection and the fingerprint benchmark use the current 512-wide shape.
Extra steady-state work includes six learner-context probe passes each iteration,
descriptor statistics, and the intrinsic preservation/PPG term. Frozen-context
fingerprints persist across iterations and restarts, keyed by weights, context
and the unchanged fixed probe; initially uncached opponents incur additional
passes. End-to-end training overhead and peak GPU memory remain unmeasured.

The real native smoke exposed two pre-existing adapter failures: an absent
nonterminal `result` caused bootstrap to dereference null, and JSON parsing could
shift a branch probability by one float64 ULP. Bootstrap now emits explicit null.
Native branch admission allows at most two ULPs for probability transport while
keeping identities exact and independently replaying the original branch sampling
evidence. It does not rewrite actual behavior probabilities or loosen PPO checks.

## Implementation files

New `recovery/domino_{config,features,sampling,state,fingerprints,auxiliary,adoption}.py`
modules isolate the contracts, assignment, observable occupancy, committed
statistics, fingerprint caches, intrinsic training and adoption. Integration spans
ordinary/execution/replay/ingress/critic data, actor-value/PPO/DCPG/PPG, PFSP,
resume/league/history, native adapters, ONNX metadata, source-review registration
and report export. `scripts/configure-domino.py` is the generation-bound helper.
Focused tests and isolated smoke scripts are under `tests/Recovery/*domino*`.

This is DOMiNO-based episodic PPO, not a reproduction of the paper's V-trace
system. Neither a passing smoke, seven sampled IDs, increased separation nor
entropy demonstrates seven strong, distinct strategies. A later matched-setup
trial must measure task quality and whole-game diversity, including a random-ID
null and a metric with industry counts removed.
