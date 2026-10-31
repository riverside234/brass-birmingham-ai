# DOMiNO context evaluation addition plan

Date: 2026-10-31 (America/Chicago).
Status: proposed implementation; this document does not enable evaluations or change training.
Applies to the existing seven-context DOMiNO trainer and the continuing
`joint7-v11-2-grouped-lr-critic` run.

## 1. Decision

Keep the existing **neutral PL rating every 10 committed iterations**. Add a
separate, report-only context panel **every 50 committed iterations**:

| Setting | Initial choice |
| --- | --- |
| Contexts | neutral, CottonMill, Manufacturer, Pottery, Brewery, IronWorks, CoalMine |
| Checkpoint | One committed checkpoint, shared by all seven contexts |
| Setups per context | 16 matched setups |
| Seat rotations | All four; 64 candidate games per context |
| Neutral control | Reuse the exact 64 selected games from that generation's main PL slot |
| Additional games | **6 × 16 × 4 = 384** |
| Total analyzed games | 448, including the 64 reused neutral games |
| Policy | Temperature 1, epsilon 0, probabilistic action sampling; context fixed for the whole game |
| Opponents | Exactly the frozen opponents and contexts in the selected neutral PL games |
| Statistical work | Paired summaries and cheap resampling of compact arrays; no additional PL fit |
| Visual summary | Automatically generated `progress.svg`, matching the existing PL report's visual style |
| Automatic training decisions | None |

This replaces the earlier tentative 30-iteration suggestion with a larger,
less frequent panel. Sixteen setup clusters give more useful within-checkpoint
comparisons than spreading a small evaluation budget across frequent seven-way
tests. The panel is for detecting substantial weaknesses and behavioral
differences; it cannot guarantee resolution of small win-rate differences.

The names identify the existing **style-plus-intent input mappings**, not seven
separate model files or a promise that seven distinct strategies already exist.

## 2. Why this frequency and size

The supplied g1330 report records **309.59 seconds** for scheduled evaluation,
with 128 main rating games and no active IL/v4 panels. Its total iteration time
is 865.63 seconds; subtracting evaluation leaves 556.04 seconds. The g1327 and
g1329 non-rating iterations took approximately 405 and 578 seconds.

For planning only, divide the complete rating-stage time by 128:

```text
time proxy per additional game = 309.59 / 128 = 2.419 seconds
new panel time proxy           = 384 × 2.419 = 928.78 seconds (15.48 minutes)
amortized addition             = 928.78 / 50 = 18.58 seconds per iteration
existing amortized iteration   = 556.04 + 309.59 / 10 = 587.00 seconds
estimated incremental overhead = 18.58 / 587.00 = 3.16%
```

Using the broader recent iteration-time range gives roughly **3.1–4.3%** extra
wall time. These are estimates, not measured panel throughput or upper bounds:
the proxy includes the expensive existing PL fit, which the context panel will
not repeat, while different context trajectories can have different lengths.
Report actual collection, extraction, analysis and total additional time.

| Alternative | Setups/context | New games/panel | New games/ordinary training games | Added seconds/iteration, proxy |
| --- | ---: | ---: | ---: | ---: |
| Every 30 iterations | 8 | 192 | 2.0% | 15.48 |
| Every 50 iterations, smaller panel | 8 | 192 | 1.2% | 9.29 |
| **Every 50 iterations, recommended** | **16** | **384** | **2.4%** | **18.58** |
| Every 100 iterations, fallback | 16 | 384 | 1.2% | 9.29 |

The recommended panel adds 60% to the number of evaluation games over a
50-iteration period, but only 2.4% relative to its 16,000 ordinary training games.
The relevant acceptance criterion is measured end-to-end overhead, not either
game-count percentage in isolation.

Use **5% additional wall time as a review budget**, not a training gate. After
the first two complete panels, compare cumulative added panel time with the
normal trainer time over the same committed interval, including normal PL work
and excluding the added panel. Include failed attempts, replay extraction and
cleanup in the numerator. If consistently above budget, optimize collection or
explicitly change the interval to 100. Keep 16 setups rather than silently
reducing statistical support. Do not automatically alter cadence or sample size
in response to favorable or unfavorable results.

## 3. Evidence being measured

The main PL series continues to measure neutral-policy strength on its existing
scale. The new panel answers two different questions:

1. At the **same checkpoint**, how do each of the six contexts and the intended
   uniform seven-context mixture perform relative to neutral against matched
   frozen opponents?
2. Do the contexts produce distinguishable investment, realization, financing
   and network trajectories, beyond the named industry bias and ordinary game
   variation?

DOMiNO seeks behavioral diversity while maintaining task quality. Both are
needed to assess the local implementation; increasing occupancy distances alone
does not establish useful strategies. [DOMiNO paper](https://arxiv.org/abs/2205.13521)

The default panel changes only the candidate's context. Neural opponents remain
neutral as in the main PL slot, and any v4 opponent remains search-free. It
therefore measures context quality against this declared opponent distribution,
not robustness to every combination of opponent styles or the entire training
mixture. A full style-versus-style matrix and deployment qualification remain
separate experiments.

## 4. Setup selection and exact neutral reuse

Create and seal the context-panel selection from the due main rating
specification **before inspecting its outcomes**. Under the current competitive
coverage scheme, select:

| Main PL setup type | Selected | Available |
| --- | ---: | ---: |
| Candidate + IL + two rated checkpoints | 2 | 4 |
| Candidate + v4 + two rated checkpoints | 2 | 4 |
| Candidate + three rated checkpoints | 12 | 24 |

Select using a deterministic, versioned schedule derived from the evaluation
seed, generation and setup identity. Balance frozen-model appearances and
prefer coverage of the previous rated checkpoint and active KL reference.
Report their actual appearances and any shortage. Never replace opponents to
force coverage after seeing game outcomes. At startup, if competitive coverage
is not active, select 16 setups deterministically from the existing main slot,
balance its reference types, and report the inherited startup fallback.

For every selected setup and rotation:

- Reuse its candidate checkpoint, opponents, physical-seat mapping, setup seed,
  action-sampling seeds and RNG-index semantics exactly.
- Run six additional jobs, changing only the candidate's explicit context.
- Hold that context through the complete game. Disable training openings,
  epsilon mixing, branching and search in all new jobs.
- Use the same stochastic sampling streams across contexts where the current
  collector permits. Once trajectories diverge their decisions need not stay
  aligned; the valid comparison unit remains the complete setup block.

The neutral control must match the candidate SHA/model ID, parent spec digest,
opponents, contexts, temperatures, seeds, rotations, runtime and source receipts.
Never substitute a previous generation's neutral games. Reuse all four rotations
for each selected setup, even if some neutral outcomes are unusually good or bad.

**Current implementation gap:** `ratings.game_record()` retains terminal results,
but not the behavioral descriptors or insolvency accounting needed here. On due
generations, extract compact candidate summaries for the selected neutral games
while `ratings.collect_slot()` still has the validated replay. Save them in a
separate immutable companion artifact before raw-game cleanup. Preserve the
original PL game records and fit inputs.

Do not silently launch 64 replacement neutral games if reuse is impossible.
Report the missing/incompatible evidence and leave the panel incomplete. A
deliberately requested standalone full panel would cost 448 games and is outside
the default 384-new-game budget.

## 5. Strength and economic reporting

Report one row per context plus the uniform mixture. Use the single candidate
seat, not all four table seats, as the unit whose outcomes are summarized.

- Official first-place win share, preserving ties as fractional wins.
- VP, placement and opponent-outrank share, with ties handled consistently.
- Combined task return using the exact recorded reward contract, including the
  configured insolvency cost once. Do not substitute raw VP or intrinsic reward.
- Insolvent-seat count/rate, insolvency events, zero-VP count/rate and pass rate.
- Paired differences from the matched neutral control for win share, VP and task
  return, with 95% intervals and explicit game/setup support.
- The task-quality margin `mean(J_context) - 0.95 × mean(J_neutral)`, using the
  current DOMiNO quality ratio. This is diagnostic evidence, not a new gate.

If a task-return ratio is shown, require a positive, supported neutral
denominator; otherwise report the difference/margin. The training constraint's
0.95 task-return ratio does **not** mean 95% win rate.

The mixture estimate is the equal-weight average of the seven context outcomes
within each setup, with uncertainty calculated jointly. Label it "uniform
candidate context mixture versus neutral-context frozen opponents." This costs
no additional games. Do not average separate PL scores or take the best observed
context as the mixture's result.

No added result enters main PL history, its 200 bootstrap refits, KL-reference
promotion, PFSP, economic health, LR/GAE/snapshot controllers, DOMiNO centroids or
duals. Preserve the retired IL/v4 panels and disabled temperature comparison.
The small IL/v4 strata here are parts of existing main lineups, not reopened
qualification panels or supported per-style v4/IL win-rate tests.

## 6. Behavioral diversity and uncertainty

Use candidate-only public-state/action descriptors from the existing versioned
DOMiNO manifest. Preserve its four phase definitions, main-action counting,
per-game phase normalization and explicit missing-terminal-phase convention.
Do not include context IDs, style/intent inputs, checkpoint identities, hidden
state or terminal outcomes in the descriptor.

Compute from complete panel games:

- Phase-resolved mean descriptors and a 7 × 7 centroid-distance matrix.
- Between-context separation and within-context variation; show their scale
  together rather than displaying only positive distances.
- Action-family frequencies, build/develop industry shares, flipped/unflipped
  investment, financing and network summaries with denominators.
- A second distance analysis with **both industry-resolved investment and
  realization families removed**. Retain the original scales of the remaining
  coordinates and version this evaluation-only projection. Do not change the
  training descriptor or claim the remaining spatial features are completely
  independent of industry.

Use a matched label-permutation null: independently permute the seven context
labels within each setup, applying the same permutation to all four rotations
of that setup. Recompute the predeclared aggregate separation statistic over
1,000 deterministic permutations. This uses existing games and helps show
whether labels explain more than setup variation. It is an exploratory
exchangeability-based diagnostic, not proof of useful strategy diversity or a
dedicated same-policy random-ID trial. A formal D5 qualification still needs its
own larger trial and controls.

For quality intervals, average the four rotations into setup-level observations,
then perform **2,000 paired setup-bootstrap resamples** of the compact arrays.
Use the same resampling indices for all contexts and the mixture. Under the
competitive layout, resample within the three setup strata, retaining counts
2 / 2 / 12. Pairing preserves covariance in context-minus-neutral differences;
the relevant resampling principle is documented by
[SciPy's paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

Use explicitly configured percentile intervals for this descriptive panel;
record the method, seed, replicate count and degeneracy warnings. Resampling
does not turn 16 independent setup groups into 64 or 448 independent samples.
The two small reference strata cannot support precise standalone conclusions.
Always show raw event counts; a zero-event or zero-width empirical interval does
not establish zero risk. Individual 95% intervals are marginal diagnostics, not
simultaneous certification of six strategies.

Display trends across panels, but keep each checkpoint's comparison separate.
Changed weights and opponent lineups prevent pooling successive panels as if
they tested one fixed policy. Repeated setup seeds also remain repeated clusters.
Do not infer a DOMiNO causal improvement without a matched pre-change comparison.

Preserve up to one representative trace per context from already collected
games: choose the trajectory nearest its held-out context centroid, with a
deterministic tie break. Label it illustrative and retain its setup/opponent
receipt; it must not be a hand-picked winning game.

## 7. Scheduling, restart and compatibility

Add a strictly validated, versioned `context_evaluation` block with mode `off`
or `report`. Proposed initial values:

```json
{
  "contract": "domino-context-evaluation/1.0.0",
  "mode": "report",
  "every_iterations": 50,
  "setups": 16,
  "seat_rotations": 4,
  "reuse_main_neutral": true,
  "bootstrap_samples": 2000,
  "permutation_samples": 1000,
  "overhead_review_fraction": 0.05,
  "raw_replay_panels": 2
}
```

The context mapping, evaluation distribution and 384-new-game limit belong to
the contract. Reject unknown fields, invalid types/nonfinite values, unsupported
context interfaces, insufficient main setups or a cadence incompatible with the
main rating interval. Derive existing execution resources from the current run;
do not introduce another CPU/GPU worker budget.

Fresh DOMiNO config generation may enable `report`. Existing configs with no
block remain off until an explicit reviewed evaluation-only transition. Include
the block in the run signature and preserve the ordinary training signature
components and their existing transition safeguards.

Persist an activation generation, next due generation, sealed pending spec and
completed/failed panel receipts. The first panel is the first main rating slot
at or after `activation_generation + 50`; later due generations advance by 50.
For example, adoption at committed g1330 would first evaluate **g1380**, then
g1430. Fresh adoption at g0 first evaluates g50. Do not backfill older checkpoints
or restart the interval on process restart.

Run synchronously after the due generation's committed main PL evidence and
before ordinary training resumes, using the same run lock. A pending panel
resumes against its sealed checkpoint and spec, never the latest live weights.
Reuse verified completed jobs and generate only missing jobs. Publish the report
atomically after all seven contexts have complete support; an interrupted or
failed panel must display incomplete support rather than a full-panel comparison.

Keep at most one automatic pending panel. Bound retries using existing failure
handling; record permanent failures explicitly rather than repeatedly launching
an evaluation backlog. Missing/corrupt required receipts fail panel admission.
Failed diagnostic work must not commit partial training changes. Include all
retry time in overhead reports and distinguish attempted from unique games.

Snapshot/restore training RNG state and model mode around evaluation, including
error paths; use independent evaluation RNGs. No weights, optimizer moments,
ordinary progress, population, DOMiNO evidence, PPG cache, baselines or controller
states are reset. Pending jobs pin their exact checkpoint dependencies until
completion or explicit abandonment.

## 8. Collection performance and artifact budget

- Register each checkpoint once by model identity. The six contexts reuse the
  same candidate GPU weights; do not load seven model copies.
- Submit context jobs through one shared inference service/actor pool. Interleave
  context jobs deterministically so common candidate/opponent weights can batch.
  Scheduling order must not alter per-game RNG, context, receipts or decisions.
- Preserve existing CPU/thread limits, model routing and legal-action masking.
  Do not overlap the new panel with PPO or another evaluation process.
- Extract summaries while reading each replay once. Do not create PPO/PPG tensors,
  policy fingerprints, teacher labels or training feature caches for the panel.
- Do not run a PL fit for each style or repeat the main historical bootstrap.
  Paired means, distance statistics and their resampling operate on small CPU
  arrays. Measure their time separately from rollout time.
- Store immutable specs, terminal outcomes, per-game descriptors, summaries and
  receipts. Keep raw new-panel replays for the latest two complete panels and
  bounded representative traces; protect unfinished work. Prune only after
  report/index commit and only within the owned evaluation directory.
- Export the compact report/spec/evidence and SVG summary through
  `export-training-reports.py`; omit bulk replay and model tensors. Record
  current/peak disk use where available.

Suggested artifacts are `ratings/contexts/generation-XXXXXX/{spec.json,
games.jsonl,summaries.jsonl,report.json,report.md,progress.svg}` plus receipts and
bounded raw games. Reused neutral games are referenced by immutable parent receipts, not
counted again as newly collected games. Preserve their compact summaries even
after the main rating slot's raw replay is retired.

Report new versus reused games; per-context and per-setup completeness; model
loads; collection/summary/analysis/total seconds; games/second; batch distribution;
peak GPU/RSS where measurable; bytes; retries; and incremental wall fraction.
Include the originating generation's DOMiNO readiness and retained intrinsic
fitting summary for interpretation, without allowing those fields to select
which panels or outcomes get reported.

## 9. Visual summary panel (required)

Every completed context panel must produce a polished, self-contained SVG
dashboard, comparable in presentation quality to the current main PL
`progress.svg`. This is a required output, not an optional later visualization.
It should let the reader identify the checkpoint, sample support, context
performance, behavioral differences and evaluation cost without opening JSON.

### 9.1 Visual language

Use `recovery/rating_plot.py` as the design reference: warm off-white canvas
(`#F6F8F7`), white cards, dark ink (`#203B43`), muted secondary text (`#5D737C`),
teal accents (`#167D8D` / `#0A6473`), subtle grid lines (`#E5ECEE`) and restrained
amber (`#9C753F`) for caveats. Preserve its generous spacing, clear title hierarchy,
rounded summary cards and small but readable methodological captions.

Use a fixed context order and stable colors across charts and generations:
neutral first, then the six existing industry contexts. Neutral uses gray;
the other contexts receive distinct colors with text labels, so color is never
the sole identifier. Use readable display names and retain the canonical names
in SVG metadata and report JSON. Keep the uniform mixture visually separate
with a diamond marker and a label identifying it as a derived estimate.

Start with an **18 × 14 inch vector canvas**, with a two-column chart grid.
Allow height to grow for wrapped labels rather than shrinking the text. Use
DejaVu Sans as in the existing renderer, approximately 24 pt for the main title,
12–14 pt for panel titles, 11–12 pt for chart labels and at least 10 pt for notes.
Verify readability at a browser display width of 1280 px. Avoid 3D charts, radar
charts, gradients on cards, large decorative elements and seven overlapping
history lines.

### 9.2 Layout and chart contents

```text
BRASS BIRMINGHAM / CONTEXT EVALUATION                 GENERATION gXXXX
Strategy quality and diversity                      T=1 / epsilon=0

[ Support ] [ Mixture vs neutral ] [ Diversity ] [ Evaluation cost ]

[ Paired VP differences with 95% intervals ] [ Context outcome table ]

[ Full descriptor distance matrix ] [ Distance excluding industry families ]

[ Recent paired quality history ] [ Actual evaluation overhead history ]

Methods, uncertainty, opponent scope, report identity and completeness
```

The four top cards show:

1. **Support:** 16 paired setups, four rotations, 64 games/context; 384 new and
   64 reused games. Keep independent setup count visible alongside game count.
2. **Mixture versus neutral:** paired VP difference and its recorded 95%
   interval. Use neutral wording when the interval spans zero; no "best style"
   badge selected from noisy point estimates.
3. **Diversity:** the recorded aggregate separation with the industry-resolved
   families removed, alongside its permutation-null reference. Include a short
   "exploratory" label; larger distance is not automatically better play.
4. **Cost:** actual additional wall time and measured amortized overhead. Before
   sufficient timing evidence exists, show "not yet available" rather than the
   planning estimate as if measured. Highlight the 5% review budget as advisory.

The first chart row presents quality:

- A horizontal forest plot of **paired VP difference from same-checkpoint
  neutral**, for the six contexts and uniform mixture. Include a zero reference
  line and recorded asymmetric 95% intervals. Mark neutral as the reference,
  not as an estimated zero-width confidence interval. Keep context order fixed
  instead of sorting by this panel's winner.
- A compact outcome table with one row per context and a separate mixture row:
  first-place win share and its interval, mean VP, task-quality margin, and
  **insolvent seats** as `affected candidate seats / candidate games`, distinct
  from the separately reported insolvency-event total. Mixture event rates
  are derived averages, not new event counts or additional independent support.

The second row contains the two **7 × 7 distance heatmaps**, with legible numeric
cell values, labeled colorbars and matching context order. Use the same
nonnegative color scale for both matrices within a panel, including zero and
the observed maximum, so removing feature families cannot appear to increase
distance merely through rescaling. Annotate the different descriptor scope and
show within-context variation alongside each matrix. Color scales may change
between generations only with explicit numeric limits; historical comparisons
must use recorded values, not color intensity. Mark unavailable cells clearly.

The bottom row uses at most the latest **12 completed panels**, retaining exact
generation coordinates and recorded intervals:

- Uniform-mixture paired VP difference versus that panel's own neutral control.
  Each point is a within-checkpoint comparison against its recorded opponents;
  this is not a PL strength curve. Do not pool the panels or smooth their evidence.
- Measured added wall-time fraction over the applicable committed intervals,
  with the 5% advisory line and actual panel runtime available in the label or
  caption. Missing intervals remain gaps, not zeros.

Break histories at incompatible protocol, descriptor or reward changes. With
one panel, show its exact point and "first panel"; never invent earlier history.
The footer states the evaluation policy, neutral-context opponent scope,
setup-paired bootstrap method, marginal interval interpretation, descriptor
ablation and exploratory status. Keep full derivations in `report.md` rather
than crowding the SVG.

### 9.3 Evidence fidelity, rendering and publication

- Implement a headless Matplotlib renderer in a separate context-report module,
  using the existing plotting dependency. Read **committed report values** and
  a bounded, recorded history selection. Rendering must not collect games,
  re-fit PL, rerun resampling, or read training tensors.
- Store the scalar interval endpoints, table values, both matrices, variation
  summaries and null references needed by the renderer in `report.json` during
  analysis. The SVG is a view of those values, not an alternative calculation.
- Include point estimates and complete interval endpoints in axis bounds, even
  when a percentile interval excludes its point estimate. Different units use
  separate labeled axes. Missing/degenerate evidence has explicit text, and
  incomplete panels have an unmistakable status rather than success styling.
- Publish the immutable per-generation `progress.svg` after its data report is
  committed. Atomically update `ratings/contexts/progress.svg` as a convenience
  copy of the latest complete rendered panel. Keep the main PL SVG separate.
- Record the renderer version and history input receipts. Use deterministic SVG
  identifiers/metadata, selectable text, and an SVG title/description containing
  generation, support and evidence scope. A redraw uses the frozen history
  selection through that generation, never future panel results.
- A rendering failure leaves valid collected evidence intact and records a
  pending/failed visualization status. Retrying the render must not rerun games.
  Keep the previous valid latest SVG with its original generation clearly shown;
  report a stale/missing latest visualization explicitly.
- Export both the per-generation SVG and the latest convenience copy with the
  report bundle, verifying that their displayed generations and receipts match.
  Retain committed SVGs with compact reports when bulk replays expire.
- Target **under 5 seconds CPU rendering time and under 2 MB per SVG**, measured
  rather than assumed. Include rendering in the evaluation-overhead accounting.
  Generate PNG previews for visual QA or an explicit export request; normal
  training only needs the SVG and does not launch a GUI/browser.

## 10. Implementation order and focused validation

| Stage | Work | Completion evidence |
| --- | --- | --- |
| CE0 | Config contract, due schedule, immutable spec and identities | Exact counts, deterministic selection, persisted restart schedule |
| CE1 | Neutral summary extraction and six-context collection | Actual replay contexts and paired seeds/opponents verified on both supported execution paths |
| CE2 | Quality, diversity, null, paired uncertainty reports and SVG dashboard | Analytic fixtures, tie handling, honest support, accurate chart values and passed visual QA |
| CE3 | Reviewed continuation, report export and owned retention | Training/RNG isolation, exact pending-job retry, protected dependencies |
| CE4 | Small isolated collection smoke, SVG preview and measured overhead | Bounded real-game artifact and inspected dashboard; default-size runtime remains explicitly unmeasured until first live panel |

Likely integration points are `recovery/ratings.py`, `recovery/league.py`,
configuration/resume validation, retention, compact history and report export.
Keep panel logic in new `context_evaluation` modules. `ratings.jobs_for()` currently
hardcodes neural focus to neutral: introduce an explicit panel context path while
regression-testing that ordinary PL specs/jobs/results keep their original
semantics. Do not copy the temperature-comparison path's extra PL fit.

Focused regressions must cover:

1. 16 setups × four rotations × six added contexts; 64 exact neutral reuses;
   one candidate per table; stratified selection independent of outcomes.
2. Context applied to the candidate's actual rotated seat, fixed throughout play;
   frozen opponents unchanged; shared weights with distinct context policy/job
   identities; no ordinary DOMiNO random-assignment hook in evaluation.
3. Temperature 1 / epsilon 0, no openings/search/branches, correct tie win shares,
   true task reward, complete descriptor phases and explicit absent data.
4. Neutral admission rejects different checkpoint/runtime/source/RNG/opponents;
   replay-derived summaries survive later raw-replay cleanup.
5. Known paired effects, identical-context controls, setup-cluster resampling,
   mixture covariance, degenerate/rare-event reporting and deterministic nulls.
6. Industry-removal metric excludes both specified families and context inputs;
   zero extra model copies; out-of-order execution preserves game identities.
7. Activation and due boundaries, Ctrl-C/restart, partial/failed jobs, atomic
   publication and bounded retention; exact preservation of training state/RNG.
8. Main PL inputs, anchor promotions, retired panels, PFSP, health/controller
   evidence and training-data eligibility are unchanged by panel results.
9. SVG numbers, context ordering, interval endpoints, counts and heatmap values
   match report JSON; both heatmaps share the declared scale; history is bounded
   and contains no later evidence. Rerendering preserves evidence/RNG and never
   invokes collection or bootstrap. Render failure/retry and exported latest
   aliases preserve generation identity.

For visual QA, render authored fixtures for a normal panel, all-identical
contexts, wide/asymmetric intervals, zero insolvencies, missing/incomplete data,
one-panel history, 12-panel history, long labels and large generation numbers.
Inspect PNG previews at full size and 1280 px width for overlap, clipping,
contrast, legend placement and uncertainty visibility. Include an example SVG
and preview in the implementation review. Validate SVG structure and required
labels with focused assertions; avoid brittle cross-platform pixel-exact tests.
Fixture images must be labeled synthetic and must never appear as live results.

The implementation smoke should use **one setup × four rotations × seven
contexts = 28 real games** in a separate output directory: four neutral games
plus 24 added games. Use a reduced test-only spec, label it statistically
insufficient, and exercise interruption/reuse without duplicate collection.
Default 16-setup scheduling can be tested with fixtures. Do not launch a full
384-game panel, long training or a strength campaign merely to implement this
plan.

## 11. Review and interpretation

A useful first result provides per-context quality differences with uncertainty,
behavioral separation beyond the industry-resolved features, and measured cost.
It does not need to declare a strongest style. Four recent training iterations,
positive centroid distance, seven populated IDs or a successful smoke cannot
establish competitive strategy diversity.

After two complete live panels, review whether the observed intervals and
behavioral variation justify the cadence. If support is too weak for an intended
decision, commission a separately budgeted evaluation on one frozen checkpoint;
do not merge changing generations or keep sampling until a desired conclusion
appears. No automatic deployment qualification or training rollback is introduced.

Related local material: `DOMINO_DIVERSITY_TRAINING_PLAN.md` (especially D5 and
reporting), `DOMINO_DIVERSITY_IMPLEMENTATION.md`, `SELFPLAY_G1330_REVIEW.md`, and
the supplied g1330 `ratings/slots/generation-001330/{spec.json,games.jsonl,report.json}`.
