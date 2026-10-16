# P4-P6: qualification and deployment

[Documentation index and required follow-up](README.md) | [Design](DESIGN.md#design) | [Audit history](AUDIT_HISTORY.md)

Sections: [P4 rosters](#p4) | [P5 difficulty](#p5) | [P6 deployment](#p6). See [encoder reuse status and remaining validation](README.md#encoder-reuse-follow-up) and [deployment performance](DEPLOYMENT_PERFORMANCE.md). Existing commands and historical verification records are preserved.

**Deployment performance update (`20261016` evidence):** Cross-worker feature sharing, fused native snapshots and snapshot-local public-view reuse are implemented. Local initialized-model tests measured 12/10 completed simulations with two workers and 15/12 with three workers on Canal/Rail fixtures; v9 and W352 matched in this short test. These results do not qualify P4/P5 or change their declared resources, tier budgets or evidence requirements. See [measured scope and limits](DEPLOYMENT_PERFORMANCE.md#measurements).

<a id="p4"></a>

## P4: unified deployment styles

**13 October audit:** [P0–P6 findings and verification](AUDIT_HISTORY.md#audit-2026-10-13-p0-p6). Deployment, roster and roster-check contracts are now **1.1.0**. Rosters retain fixture provenance from their handoff/readiness; qualification requires explicit nonfixture provenance and the new `real_artifacts` gate. Full synthetic panels cannot qualify even if every statistical gate passes. Regenerate P4 evidence and dependent P5/P6 artifacts; do not retag old reports.

**11 October review:** [P0–P4 corrections and compatibility notes](AUDIT_HISTORY.md#audit-2026-10-11-p0-p4). Neighborhood selection now uses each pair's own noise threshold. Blind examples cover members across setups/seats and stay fixed when only partial chunks exist. Rebuild the upstream personality evidence and P4 artifacts under the revised contracts; the 10 October verification below describes historical fixtures.

P4 implements the **one frozen shared network** option in [the personality plan](DESIGN.md#design-56-bring-discovered-behavior-back-into-one-network). It selects 16 archetypes, each containing its tested representative and seven nearby tested variants. There are 128 unique controllers and one set of neural weights. The five pilot neural profile inputs remain supported inputs; archetypes combine those trained inputs with tested intent/prior settings. Sixteen distinct behaviors must be demonstrated, not inferred from 128 different numbers.

The actual Step 5 parent is still training on the lab. Complete the [P0/P1 handoff](TRAINING_P0_P3.md#p0-p1) and [P2/P3 calibration/discovery](TRAINING_P0_P3.md#p2-p3) first. Local authored fixtures do not qualify a learned roster. This workflow changes neither Step 5 nor the installed game. P5 handles difficulty and UI; P6 handles laptop export, once-per-game sampling/save restoration and installation.

<a id="p4-1-required-input"></a>

### 1. Required input

Use a P3 archive with all candidates evaluated under **the same frozen ready model, opponent pool, reward and search configuration**. For a full roster, the archive must contain searched-controller confirmation, not just direct-policy discovery. Expand the confirmed proposal to include enough neighbors: a 16-entry representative shortlist alone cannot supply 128 members.

Selection requires each member's existing P3 preliminary quality gate, at least eight matched setup groups, mean own-seat win share at least 0.20 and a control-difference lower endpoint of at least −0.05. It compares FPS and K-means representatives, uses the chosen method fixed before new calibration, and keeps actual tested recipes. Neither a cluster centroid nor an averaged set of weights becomes a style input.

Neighborhoods are relative to each representative's nearest other representative, not one global radius. Defaults bound radius to 0.40 of that distance, retain a 0.10 separation margin and require each pair of members to differ by more than 0.50 times that pair's combined game-to-game dispersion. An incompatible candidate is skipped while other eligible neighbors remain. Missing, indistinguishable, noisy or shared members cause selection to fail with the affected count. Expand discovery instead of duplicating variants or lowering evidence gates after seeing the calibration results.

<a id="p4-2-select-and-name-a-roster"></a>

### 2. Select and name a roster

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
P4=$ROOT/artifacts/personality/p4

python "$ROOT/scripts/personality_deployment.py" --root "$ROOT" select \
  --archive "$ROOT/artifacts/personality/discovery/searched-archive" \
  --output "$P4/candidate" --groups 16 --members 8 --method fps

python "$ROOT/scripts/personality_deployment.py" --root "$ROOT" cards \
  --roster "$P4/candidate" --output "$P4/naming"
```

`searched-archive` is the P3 archive directory you actually produced; substitute its path. For a smaller preliminary workflow, use `--groups 2 --members 2` (or another smaller size). A preview is explicitly ineligible for full P4 qualification, even when its other checks pass. Direct-policy previews are supported.

`candidate/manifest.json` pins the shared model once, source archive/encoder, exact profile/search recipes, member mapping, uniform sampling policy and evidence dependencies. It contains no per-member model copies or optimizer state. `resolve()` returns the tested `Profile` and `SearchConfig`; descriptor coordinates never enter inference. The behavior encoder remains an offline calibration dependency.

Read `naming/cards.json`: it contains labeled **public discovery plans and visible board snapshots**, with hidden cards, model IDs and final scores removed. Fill the distinct names and short descriptions in `naming/labels-template.json`, saving the result as `naming/labels.json`. Names should describe observed behavior. Freeze these descriptions before calibration. Do not invent 16 labels to disguise a lack of distinctness.

<a id="p4-3-freeze-a-bounded-collection-plan"></a>

### 3. Freeze a bounded collection plan

```bash
python "$ROOT/scripts/personality_deployment.py" --root "$ROOT" plan \
  --roster "$P4/candidate" --labels "$P4/naming/labels.json" \
  --output "$P4/panel" --setups 16 --repetitions 2 --seed 121000000
```

The plan excludes descriptor-fit and P3 selection setup groups and uses only fresh validation groups. It freezes all jobs, opponent mixtures, independent per-seat sampling streams and named descriptions. Each setup has:

- All 128 members in all four seats against the same three frozen neural references.
- The neutral/no-bias control and four pilot-profile controls, with matching model/search budgets.
- Separate games for each archetype's uniform eight-member mixture and its representative with ordinary policy randomness. Two repetitions are the default; mixture members are drawn independently once per game, including independently for different seats.

This full default is **12,608 games**, deliberately planned separately from execution. A two-by-two preview with two setups/two repetitions is 136 games. The plan does not launch games. Use a small preview to check interfaces before budgeting a full calibration; no local campaign was run. Eight setups is the minimum for P4 quality estimates, but more may be needed for uncertainty or diversity gates. P5's final test groups remain unused.

<a id="p4-4-collect-chunks-on-the-lab"></a>

### 4. Collect chunks on the lab

```bash
python "$ROOT/scripts/personality_deployment.py" --root "$ROOT" --device cuda collect \
  --panel "$P4/panel" --output "$P4/chunk-00000" \
  --start 0 --games 128 --tables 24

python "$ROOT/scripts/personality_deployment.py" --root "$ROOT" --device cuda collect \
  --panel "$P4/panel" --output "$P4/chunk-00128" \
  --start 128 --games 128 --tables 24
```

Continue with disjoint slices until the frozen plan is covered. Each invocation accepts at most 4,096 games; use smaller chunks for recovery. Existing chunk directories are immutable and rejected. A failure leaves an incomplete reservation; it supplies no evidence. Do not silently drop its planned jobs or select among favorable reruns. Checking overlapping chunks is an error; a partial set produces an explicitly incomplete report.

Execution reuses P2's central dynamic inference queue, CPU admission, private search forests, actual per-seat contexts and legal choices. Search seconds/workers come **unchanged from the selected archive** (normally 12 seconds per whole turn/four workers for the lab NN reference). Ordinary options, subchoices and workers share the existing deadline. Direct previews create no search hosts. This does not alter PPO, IL, teacher cadence, budgets, retention presets or the 24-table lab allowance. Do not launch this independent calibration alongside a training job that already consumes the allocation.

These are calibration/evaluation records. They do not increment ordinary-game counts, supply PPO/search teaching or fit the descriptor. Base generated-data formats and P2 sidecars remain unchanged. The extra job attribution is in the P4 panel/receipts.

<a id="p4-5-re-encode-and-check-the-deployed-roster"></a>

### 5. Re-encode and check the deployed roster

```bash
python "$ROOT/scripts/personality_deployment.py" --root "$ROOT" --device cuda check \
  --panel "$P4/panel" --chunks "$P4"/chunk-* \
  --output "$P4/check-automatic"
```

Every replay is checked against its exact planned job, model, reward, pool and search recipe. The frozen encoder transforms the new trajectories; it is never refitted on calibration. The report checks:

- Complete fresh coverage, legal completed four-player games and both eras; every member has actual completed search work for searched qualification.
- Each member **and each actual uniform mixture** against the neutral control: own official win share, VP and rank, grouped by setup. Default tolerated control differences are −0.05 win share, −5 VP and +0.20 rank, using preliminary 95% intervals. These are P4 screens, not P5's multiplicity-adjusted release certificates.
- Recomputed member centroids/dispersion, same-archetype neighborhoods and differences beyond ordinary sampling noise. A formerly diverse archive cannot qualify a collapsed deployment model.
- At least 60% held-out nearest-representative recognition for every archetype, plus better between/within behavior separation than the four pilot controls.
- Canonical plan variety in each era, comparing the mixture with its representative plus ordinary action randomness. Either opening or later-plan diversity must improve by at least 0.02, with a positive setup-grouped lower endpoint, in **both** eras. Changing payment/card IDs or ordering interchangeable subchoices does not count as variety.
- Pottery opportunity/build/observed-flip coverage across the roster. This is a coverage diagnostic, not a per-style usage quota or a reward bonus; an unobserved or unsupported Pottery case remains unresolved. Human review also checks tactical soundness.

`report.json` separates official performance, geometry, variety and search throughput. It records completed/interrupted simulations, fallbacks, prior preparation time, overruns and simulations per search second for each member and the neutral control. A small fixture run cannot establish playing strength or performance improvement.

Automatic analysis also writes `blind-packet.json` and `review-template.json`. Supply the frozen named descriptions and discovery cards to at least two reviewers. They independently label the **unlabeled public calibration plans** and mark tactical soundness using the included visible board snapshots. Do not give reviewers the source job/controller mapping or outcomes. In a completed roster, the packet samples at least eight examples per archetype, covering all eight members. Duplicate ratings, mismatched packets or descriptions changed after calibration are rejected. Per-archetype recognition must reach 60%; an identified tactical failure blocks that group. The public-only review cannot reconstruct hand-dependent alternatives, so it supplements the legal/quality checks rather than claiming exhaustive tactical proof.

Merge the responses into one file matching `review-template.json`, with one rating per `(example, rater)` and at least two distinct raters per example. `guess` is the frozen archetype ID. Save, for example, as `blind-review.json`, then check the **same** games:

```bash
python "$ROOT/scripts/personality_deployment.py" --root "$ROOT" --device cuda check \
  --panel "$P4/panel" --chunks "$P4"/chunk-* \
  --review "$P4/blind-review.json" --output "$P4/check-reviewed"
```

Only a complete 16/128 searched roster passing every gate is marked `p4_qualified`. `release_qualified` remains false, supported difficulty tiers remain empty, and no unqualified same-archetype fallback is invented. P5/P6 must still qualify difficulty, enjoyable interaction, runtime sampling/restoration, export parity, hardware timing and installation. An unresolved check still preserves its report and pinned manifest for diagnosis.

<a id="p4-6-changed-or-distilled-networks"></a>

### 6. Changed or distilled networks

P4's implemented path needs no extra distillation: the P3 entries already share one frozen network. It rejects archives mixing checkpoints and does not average weights, relabel hindsight descriptors as policy inputs or turn calibration decisions into PPO samples.

If a new network is produced by P2 policy distillation/joint continuation, first run P2 readiness again, then collect a **new P3 discovery/quality archive** under the new model identity and reselect/recheck P4. All 128 deployed controllers must be re-encoded and requalified. Old coordinates and qualification receipts cannot authorize changed weights. Cross-checkpoint consolidation is not a second implementation path in this version; use the shared-generation path unless a later explicit importer/training design is added.

<a id="p4-verification"></a>

### Verification

Run focused checks in the existing conda environment:

```bash
PYTHONPATH="$ROOT/src:$ROOT/tests/R2:$ROOT/tests/Recovery:$ROOT/tests/Personality" \
  python -m unittest test_deployment test_personality test_search test_search_training test_repertoire
```

The optional native fixture smoke is one four-player direct-preview game, not a qualification campaign:

```bash
python "$ROOT/tests/Personality/smoke_deployment.py" \
  --output "$ROOT/artifacts/personality/p4-native-smoke"
```

It creates an explicitly untrained fixture parent/descriptor, exercises panel collection and reporting, checks compatibility fingerprints and verifies that an incomplete preview cannot qualify. Production work must start from the completed accepted Step 5 handoff instead.

<a id="p4-local-verification-10-october-2026"></a>

#### Local verification, 10 October 2026

- **14 P4 checks passed**, alongside **33 existing P0/P1/P2/P3 checks**. Coverage includes exact 16/128 membership, shared weights, noise/collapse rejection, seeded rotations/mixtures, changed-model and duplicated/reused-evidence rejection, public-only review inputs, immutable panels, variety controls and unqualified partial reports.
- The final native smoke completed **one four-player direct-preview game in 24.0 seconds**, exercising collection, both eras, public board/plan extraction and immutable reporting. It intentionally collected only one job from its 104-job fixture plan and correctly failed qualification. Its artifacts are under `artifacts/personality/p4-native-final-20261010/` and remain fixtures.
- CLI help, new-source syntax/whitespace, guide links and `git diff --check` passed. The final smoke report loads under the current P4 contract; its pinned weights/evidence validate.
- Step 5 and P0/P1 compatibility fingerprints are unchanged. No installed game files, main training settings, production models or generated training-data schemas were modified. P4 has no new online model parameters or search algorithm changes. CUDA utilization, real style recognition/strength and a complete 16/128 roster remain lab measurements; this small CPU smoke is not a throughput comparison.

<a id="p5"></a>

## P5: difficulty calibration and intent interfaces

**13 October audit:** [P0–P6 findings and verification](AUDIT_HISTORY.md#audit-2026-10-13-p0-p6). P5 now requires the P4 1.1.0 report's explicit nonfixture provenance and `real_artifacts` gate. The P5 version remains 1.2.0, but its nested source contract changed: regenerate panels and reports from current P4 evidence. No additional independent P5 defect was confirmed in this pass.

P5 tools are implemented. **Real calibration, human review, and qualification are pending.** The actual Step 5 parent is still training, and there is no qualified P4 roster yet. All local P5 results are explicitly untrained fixtures. No Easy/Medium/Hard level is currently certified, and this package does not install or export a game AI.

The current verifier is `personality-difficulty/1.2.0`. The [P5/P6 audit](AUDIT_HISTORY.md#audit-2026-10-12-p5-p6) added panel-wide job ownership, full derived-report revalidation and preparation-failure cleanup. Earlier P5 evidence must be regenerated under new panels; Step 5 and P0–P4 continuation are unchanged.

The separate `scripts/personality_difficulty.py` command reuses P0's handoff, P1's unchanged shared network, P2's intent/search/central inference, and P4's roster, profiles, frozen reference pool and pinned qualification evidence. It neither resumes nor writes into Step 5. Existing datasets, IL handoff, model weights, optimizer state and training progress retain their existing contracts; **P5 requires no Step 5 checkpoint migration**.

<a id="p5-1-environment-and-prerequisites"></a>

### 1. Environment and prerequisites

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
cd "$ROOT"
python -m pip install -r "$ROOT/requirements-personality-p5.txt"
export PYTHONPATH="$ROOT/src:$ROOT/tests/R2:$ROOT/tests/Recovery:$ROOT/tests/Personality${PYTHONPATH:+:$PYTHONPATH}"

python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" build-host
```

The SciPy dependency is used only for P5's statistical intervals. The build uses the existing pinned compiler and privately transferred R1 dependencies, under `artifacts/personality-host/guarded/`. On Linux it uses Mono (or the global `--mono` argument). It does not read or modify the installed Unity package. Transfer the existing compiler/dependencies first if this is a new lab checkout.

The baseline is **`stock-hard-guarded-1.1.0`**: original Hard's selector and liquidation behavior, running on the declared repaired/guarded R1 rules. This is the existing R3 comparison identity, not a claim about unpatched vendor rules or a substitute v4 policy. Native stock failures are recorded separately, are not credited as candidate wins and block qualification. Stock actions cannot honestly be expressed as R1 legal-choice replay, so P5 writes a separate versioned evaluation record rather than fabricating training rows.

The isolated bridge reconstructs schema-valid public history from observable stock events for subsequent neural observations. It does not relax the observation validator or expose stock hands. When a brewery-funded stock sale does not bind a merchant, the history uses a deterministic compatible reachable merchant witness; it does not claim that the original selector explicitly chose that merchant.

Before real work, finish P0–P4 using [P0/P1](TRAINING_P0_P3.md#p0-p1), [P2/P3](TRAINING_P0_P3.md#p2-p3), and [P4](QUALIFICATION_P4_P6.md#p4). P5 requires the full 16/128 searched roster, immutable common model, P4 report, public names/descriptions, blind recognition/tactical review, and all pinned sources. Real commands reject fixture parents and unresolved/preview rosters.

<a id="p5-2-versioned-tier-candidates"></a>

### 2. Versioned tier candidates

The starting recipe is `configs/personality/difficulty-1.0.0.json`:

| Candidate tier | Whole-turn cap | Search workers |
| --- | ---: | ---: |
| Easy | 2 seconds | 2 |
| Medium | 5 seconds | 2 |
| Hard | 12 seconds | 2 |

These are **candidates for measurement**, not presumed strength levels. They preserve the plan's laptop choices and two tensor threads. All inherit the same P4 shared weights, per-member style/intent/bias settings, reward, candidate masks, utility scale and supported-root-visits selection. Near-best tolerance is zero in this version: there is no new untested selection algorithm or deliberate blunder injection. Goal-bias strength is not used as the difficulty slider.

P2's existing whole-turn CPU lease and common deadline cover both actions, subchoices and private workers. Search stops early when appropriate. The shared model and frozen references are registered once per collection; tables reuse hosts/search sessions and central dynamic inference. The launcher uses only the first already-visible GPU, with no multi-GPU router. Use `CUDA_VISIBLE_DEVICES` to select an allocated device before running it.

Step 5 remains unchanged: NN teachers 12 seconds/four workers, searched v4 one second/four workers, 50% cumulative search-wall admission, the current PPO/reward/opponent schedules and single-GPU settings. P5 evaluation games are never teaching, PPO, IL or descriptor-fitting data.

<a id="p5-3-freeze-a-calibration-panel-when-p4-is-qualified"></a>

### 3. Freeze a calibration panel (when P4 is qualified)

```bash
P5="$ROOT/artifacts/personality/p5"
ROSTER="$ROOT/artifacts/personality/p4/check-reviewed"

python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" plan \
  --roster "$ROSTER" \
  --profiles "$ROOT/configs/personality/difficulty-1.0.0.json" \
  --output "$P5/calibration-panel" --stage calibration \
  --setups 32 --repetitions 2 --seed 151000000 \
  --tables 24 --hardware-label lab-rtx6000pro
```

Planning freezes seeds, all four seat rotations, clockwise opponents, independently seeded once-per-game member draws, profiles, original-Hard runtime, table allowance and acceptance thresholds. Validation groups exclude P3/P4 evidence groups. Member draws/action streams are matched across tiers and reference comparisons. The hidden member draw is uniform; actual mixture games are evaluated independently from member results.

The panels include:

- Every member and every archetype mixture against the same frozen neural reference pool at all three tiers.
- Every Easy member and mixture as one candidate plus three original Hard seats.
- Same-tier homogeneous-style and mixed-style games, reported separately.
- Medium versus Easy and Hard versus Medium, with two independent seats per tier and all six seat-pair allocations, separately for every archetype.

**The full default is 100,352 games (3,136 per setup). Planning does not run them.** This is a formal 128-member/384-configuration study, not a routine training monitor. Budget it explicitly after a small interface preview. A small panel remains unresolved; do not lower thresholds or count correlated rotations as independent setups to manufacture a pass. More independent groups may be necessary; the tool bounds a panel to 512 groups. If power or costs are unacceptable, revise and preregister the study before using new untouched evidence.

`--gates /absolute/path/gates.json` can freeze an explicitly reviewed alternative within validated ranges. Default minimum is 32 setup groups, family error level 0.05, Easy gain +0.02 above 25% win share, same-tier equivalence ±0.05, and positive tier gains of at least 0.01. Gates cannot be edited after collection. For another hint mode, copy the profile JSON and change `hint_mode` to `intent-hints` or `hidden` **before** planning a separate panel.

<a id="p5-4-collect-immutable-bounded-chunks"></a>

### 4. Collect immutable bounded chunks

```bash
# Select the one GPU allocated to this job, for example physical GPU 0.
export CUDA_VISIBLE_DEVICES=0

python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" --device cuda collect \
  --panel "$P5/calibration-panel" --output "$P5/calibration-chunk-000000" \
  --start 0 --games 96 --tables 24

python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" --device cuda collect \
  --panel "$P5/calibration-panel" --output "$P5/calibration-chunk-000096" \
  --start 96 --games 96 --tables 24
```

Continue disjoint slices only when the study is budgeted. A command accepts at most 4,096 games. `--tables` must match the frozen plan; fewer remaining jobs naturally activate fewer tables. CPU admission comes from the pinned handoff's allocation. Do not run a second calibration workload alongside training if training already occupies that allocation.

A sealed panel-owned `reservations.json` assigns jobs to chunk paths before collection under an OS-backed lock. A different output directory cannot retry an already reserved job, and copied chunks cannot replace its evidence. Expected game failures are retained as failures, without legal-policy replacements or favorable retries. A crashed invocation can leave an incomplete reservation; it is not valid evidence. Preserve the ledger and diagnose/recover under an explicit study amendment rather than overwriting it, skipping the failed jobs or silently rerunning for a better result. Changed models, search profiles, sources or runtime bytes require new evidence; no hash rewriting is supported.

Reports include completed/interrupted simulations, fallbacks, prior/search seconds, overruns, game time, admission/batching telemetry and the measured machine/backend. Timing on the lab is not a laptop certificate. A changed table allowance, backend or clock does not inherit an existing strength claim. For eventual laptop validation, make a separate declared panel with its actual resources; P6 owns integration and runtime timing.

<a id="p5-5-analyze-and-collect-human-play-feedback"></a>

### 5. Analyze and collect human play feedback

```bash
python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" check \
  --panel "$P5/calibration-panel" --chunks "$P5"/calibration-chunk-* \
  --output "$P5/calibration-automatic"
```

`report.json` separates each member, each mixture, reference lineups, tier comparisons, original-Hard failures and timing. Partial chunks can be inspected, but cannot qualify. Exact official VP/income/cash tiebreakers and shared wins are reused. VP is a reported/corroborative metric, never capped, normalized or a changed training reward.

Qualification rules:

- **Every Easy member, and every mixture separately**, needs a grouped lower win-share endpoint above 0.27, with positive mean VP margin and better mean official rank than its three original-Hard rivals. A strong archetype average cannot hide a weak member.
- Every member/tier and mixture/tier needs its entire paired win-share interval inside ±0.05 of the equally weighted archetype-mixture target on the frozen neural pool. A nonsignificant difference or a wide interval does not prove equivalence.
- Medium must exceed Easy and Hard must exceed Medium on the common neural pool **and** the direct two-versus-two family-win comparison. Direct games have a 50% family baseline, not 25%. Per-archetype direct results/exceptions are reported rather than reduced to an Elo score.
- All jobs, legal complete games, both eras and actual search work for every member/tier are required. An unresolved P4 gate, missing human review or any fixture flag blocks P5 qualification.

Rotations and repetitions are averaged **within setup**. One prespecified Bonferroni family covers the 144 Easy claims, 432 equivalence claims and four aggregate ordering comparisons (580 intervals for the full roster). Student-t intervals assume sufficiently regular independent setup means; 32 groups is a minimum, not guaranteed validity/power. Zero observed variance uses a conservative bounded Hoeffding interval rather than pretending uncertainty is zero. Validate these assumptions and study power before treating results as release evidence.

`public-cases.json` contains outcome-independent member/seat/setup examples: completed public plans, visible boards and permitted display text. It omits micro-variant IDs, model identities, hidden cards, private goals/targets and search traces. It supplements P4 blind recognition review; it is not itself human play evidence.

`review-template.json` starts with an empty `sessions` list. Record completed four-player human sessions using pseudonymous rater IDs, one unique session ID per play, the public archetype, tier, exact hint mode and a reference to the session evidence. Each session has:

```json
{
  "session_id": "human-session-001",
  "rater": "reviewer-a",
  "archetype_id": "the-public-archetype-id",
  "tier": "easy",
  "hint_mode": "public",
  "completed": true,
  "players": 4,
  "tactically_sound": true,
  "ratings": {
    "understandable_plans": 4,
    "varied_openings": 4,
    "resilience": 4,
    "resource_interaction": 4,
    "enjoyment": 4,
    "lack_of_repetition": 4,
    "appropriate_interference": 4,
    "waiting_comfort": 4
  },
  "evidence": "reference to the actual completed human session"
}
```

This is a **schema example, not a real review**. Keep the template's panel/model/profile/hardware identity fields intact. At least two independent raters per style/tier, no reported tactical failure, and mean ratings of at least three in each category are required. Tools validate attribution and coverage; they cannot prove reviewer independence or enjoyment. The current delivery provides interfaces/records, not a playable Unity UI. Actual human sessions await a suitable review harness or P6 integration.

```bash
python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" check \
  --panel "$P5/calibration-panel" --chunks "$P5"/calibration-chunk-* \
  --review "$P5/calibration-human-review.json" \
  --output "$P5/calibration-reviewed"
```

Calibration-stage reports always remain provisional. Tune only on calibration evidence, freeze the final candidate and register a **new test-partition** release panel referencing the complete calibration report:

```bash
python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" plan \
  --roster "$ROSTER" --profiles "$ROOT/configs/personality/difficulty-1.0.0.json" \
  --output "$P5/release-panel" --stage release \
  --calibration "$P5/calibration-reviewed" --setups 32 --repetitions 2 \
  --seed 181000000 --tables 24 --hardware-label lab-rtx6000pro
```

Collect/check that frozen panel in disjoint chunks and provide its own bound human-review record. Final test groups exclude all prior discovery/calibration groups. Do not repeatedly retune on final results. Only complete non-fixture evidence passing every gate can set `p5_qualified`. **`release_qualified` remains false even then**: P6 export, hardware parity, save/restore, runtime sampling, cancellation/Play Now and installation are separate.

<a id="p5-6-public-selection-and-intent-interfaces"></a>

### 6. Public selection and intent interfaces

```bash
python "$ROOT/scripts/personality_difficulty.py" --root "$ROOT" menu \
  --roster "$ROSTER" --report "$P5/release-reviewed" \
  --output "$P5/public-interface"
```

The JSON interface has exactly 16 named archetypes and three tier choices only where measured. Internal members, centroids, controller/model hashes and micro-variant IDs never enter it. Unqualified entries are disabled; `--fixture` emits an explicit unqualified preview. Actual once-per-game sampling/restoration is P6, not a second P5 runtime implementation.

`IntentDisplay.update()` has three modes:

- `hidden`: no text.
- `public` (default): deterministic text from the static public style description and completed public actions. It does not read observations, private goals or head probabilities.
- `intent-hints`: opt-in broad, uncertain intention from the displayed seat's validated `IntentState` snapshot. Calibrated confidence must reach 0.65; changed hints require two turns. Updates are at most once per own turn. Exact locations/targets, cards, probabilities, future draws and search traces are never emitted. Hiding text takes effect immediately.

Hints deliberately reveal some of the AI's own plan. They are not claimed causal explanations of neural computations. Another seat's intent is rejected. Switching hint modes cannot inherit a different mode's measured difficulty. No new public-only prediction head or language model is introduced.

<a id="p5-7-focused-checks-and-fixture-smoke"></a>

### 7. Focused checks and fixture smoke

```bash
python -m unittest test_difficulty \
  test_search.SearchTests.test_one_deadline_after_admission_both_actions_workers_and_cancel \
  test_search.SearchTests.test_zero_bias_exact_and_bounded_support \
  test_search.SearchTests.test_central_queue_named_heads_context_cancel_and_expiry \
  test_deployment.DeploymentTests.test_online_resolution_uses_tested_profile_not_descriptor \
  test_deployment.DeploymentTests.test_public_variety_does_not_count_payment_or_hidden_card_changes \
  test_personality.PersonalityTests.test_old_reader_rejects_new_checkpoint_and_restore_is_exact -v

python "$ROOT/tests/Personality/smoke_difficulty.py" \
  --output "$ROOT/artifacts/personality/p5-fixture-smoke"
```

Use a new smoke output directory. The smoke creates an explicitly untrained fixture handoff/roster, runs two actual four-player games (one neural-reference table and one original-Hard table) without search, then one two-second/two-worker searched turn. It verifies the 16-choice public interface, source compatibility and zero remaining CPU leases. The fixture panel stays incomplete/unqualified. No long campaign, GPU training or installed-game update is performed.

Real parent quality, all 128 member/tier baselines, same-tier equivalence, tier ordering, human enjoyment, real goal confidence, single-GPU lab performance and laptop timing remain to be measured.

<a id="p5-local-verification-12-october-2026"></a>

#### Local verification, 12 October 2026

- **23 focused tests passed**, covering member/mixture jobs, official ties, grouped/multiplicity intervals, equivalence and failing-member gates, reservations/evidence pins, fixture rejection, UI privacy/hysteresis, shared deadlines/cancellation and exact checkpoint restoration.
- The isolated host built successfully. Two actual four-player **untrained fixture** games completed: one against neural references and one against three original Hard seats. Collection took 36.09 seconds on local CPU; neither game is difficulty evidence.
- A two-second/two-worker probe completed zero simulations and returned a legal fallback (2.09 seconds externally measured, including host observation/application). A separate 12-second probe completed seven simulations, supported its legal plan and returned in 11.14 seconds. Search reported no deadline overrun in either probe; both retired all CPU leases. These are execution checks, not timing or strength qualification.
- Step 5, P0/P1, P2, P3, P4 and learning compatibility fingerprints matched the pre-P5 snapshot. P5 changes no training/checkpoint/data contract. The installed game was not modified.

Local evidence is under `artifacts/personality/p5-20261012/`: `targeted-final-fixed.log`, `native-smoke-fixed/smoke-report.json` and `hard-turn.json`. An initial stock-history bridge failure was corrected before the successful smoke; failed evidence was retained separately. No real calibration, human review, GPU benchmark or release campaign ran.

<a id="p5-p5p6-review-corrections"></a>

#### P5/P6 review corrections

The earlier review introduced `personality-difficulty/1.1.0`, now superseded by 1.2.0 above. A qualified flag alone cannot enable tiers: the portable summary must contain all named gates, the complete member/mixture/tier set, source identities, release coverage and human review. Loading a report also recomputes its metrics and gates from the pinned game records. An original-Hard failure is disqualifying even if a record inconsistently omits the outer failure flag.

Release planning and loading now follow the calibration report's underlying chunk/game/review dependencies, check its exact recipe, and preserve the excluded training/calibration setup groups. Keeping a report after deleting its evidence cannot support qualification. Public menus remain disabled for incomplete evidence.

Previous P5 evidence uses a different verifier/source identity and must not be relabeled. Keep it for comparison; create fresh calibration/release panels under the corrected verifier. No real roster was qualified locally. This change does **not** migrate or invalidate Step 5/IL weights, optimizer state, counters, datasets or P0–P4 artifacts. Their existing continuation commands remain valid.

Focused review checks: `python -m unittest test_difficulty test_laptop test_p56_review test_search -v` with the existing test paths on `PYTHONPATH`. See [P6 deployment](QUALIFICATION_P4_P6.md#p6) for fixture integration and package migration boundaries.

Review verification: **54 targeted Python checks passed**, including the P0 checkpoint restore and P4 profile/public-data boundaries. Two further fixture games completed in 22.157 seconds, with no failed games or remaining CPU leases; the incomplete panel remained unqualified, and loading its report recomputed identical results. Evidence: `artifacts/personality/p56-review-targeted.log` and `p56-calibration-smoke/review-smoke.json`. No real calibration campaign ran.

<a id="p6"></a>

## P6: laptop packaging and game integration

**Current runtime performance:** New source-built packages select the verified optimized native search host, and both private search workers and plan revalidation request its fused path. Legacy runtime capability fallback remains supported. The [performance guide](DEPLOYMENT_PERFORMANCE.md) explains shared encoding, private-hand isolation, benchmark reproduction and future proposals. Existing packages are immutable and do not receive these changes automatically; regenerate compatible evidence and build a new candidate through the normal qualification flow. The three-worker and W352 comparison was an isolated experiment, not a supported replacement for the required two-worker laptop qualification profile.

**13 October audit:** [P0–P6 findings and verification](AUDIT_HISTORY.md#audit-2026-10-13-p0-p6). New bundles use `personality-laptop-package/1.2.0`. Validation now binds the roster's content hash, search reward, P5 profiles and copied `readiness.json` calibration to the same evidence chain. A late benchmark button timer cannot affect the following case. Re-export/restage candidates after regenerating upstream evidence; preserve old immutable packages with their matching runtimes/helpers for existing saves. The save schema is unchanged.

P6 tooling is implemented. **No trained personality package is qualified or installed.** The current Step 5 model is still on the lab server, and the real P4 roster and P5 difficulty evidence are pending. The local authored fixtures exercise deployment mechanics only; they do not establish playing strength, recognizable styles or Easy/Medium/Hard difficulty.

The [P5/P6 implementation audit](AUDIT_HISTORY.md#audit-2026-10-12-p5-p6) fixes first-save integrity, rollout-pool validation, persistent-service failure cleanup and malformed requests. Certification now requires observed controls and valid Play Now clocks. Re-export/restage candidates for the new source identities and P5 verifier 1.2.0; preserve old immutable packages for their existing valid saves.

<a id="p6-prerequisites-and-boundaries"></a>

### Prerequisites and boundaries

| Stage | Available implementation | Still required for a release |
| --- | --- | --- |
| P0/P1 | Accepted-parent handoff, versioned model/utility/goal contracts and traces | Completed eligible Step 5 accepted model; real goal/utility validation |
| P2/P3 | Masked ISMCTS intent/search integration, shared batching, discovery and learned repertoire tools | Real calibration, searched-controller evidence and learned descriptors |
| P4 | Shared-network roster selection and member/mixture qualification | Qualified **16 groups × eight members**, actual names/descriptions and human review |
| P5 | Tier profiles, every-Easy-member versus original Hard gates, mixture/equivalence/order gates, public intent interface | Complete untouched panels and human evidence for the exact two-worker laptop profiles and hint mode |
| P6 | Export, persistent local service, staged game helper, benchmarks, validation, installation and rollback | Actual-model export parity, representative laptop/backend tests, interactive Unity acceptance and release certificate |

P6 adds `src/brass_ai/personality/laptop/`, `src/BrassPersonalityGame/` and `scripts/personality_laptop.py`. It reuses the existing P0–P5 implementation and does not change Step 5, its datasets, optimizer/checkpoint continuation, reward, architecture or training settings. Staging reads the accepted **1.4.31** game archive, so unrelated uninstalled source changes are not imported.

The candidate package contains one shared float32 ONNX actor for all 128 members and the small frozen rollout-reference pool required by the qualified P2 controller. It preserves policy, official win/placement/VP, raw remaining utility, goal factors and confidence. It does not ship 128 copies of the actor, an optimizer, discovery archive or behavior-encoder weights. The bundled Python library remains byte-pinned; no learner entry point is launched by the game.

Native simulator dependencies include proprietary game DLLs. These are **private local packages**, not artifacts to upload to a public repository. The Python interpreter and its installed dependencies are provisioned separately. No installer downloads dependencies or models automatically.

<a id="p6-export-on-the-lab-server"></a>

### Export on the lab server

Use the existing conda environment and native R4 host prerequisites. Existing R2/search/personality requirements must already be installed.

The exporter uses the source-checked additive search runtime from `training/runtime/throughput/search` when present, verifies its native dependencies and bundles the compiled host under `runtime/artifacts/r4-host`. The deployed package needs neither a compiler nor a training directory. If the additive runtime is absent, the legacy host remains usable but does not provide the measured fused-path gain. Source/binary mismatches fail validation; do not suppress them. See [runtime provenance and reproduction](DEPLOYMENT_PERFORMANCE.md#reproduction) before building a new candidate.

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
cd "$ROOT"
python -m pip install -r "$ROOT/requirements-personality-p6.txt"

python "$ROOT/scripts/personality_laptop.py" export \
  --roster "$ROOT/artifacts/personality/p4/qualified-roster" \
  --qualification "$ROOT/artifacts/personality/p5/qualified-report" \
  --output "$ROOT/artifacts/personality/p6/candidate-001"

python "$ROOT/scripts/personality_laptop.py" validate \
  --package "$ROOT/artifacts/personality/p6/candidate-001"
```

Replace the two input directories with the actual P4 roster and P5 report directories. Export refuses an incomplete roster, changed sources/model/pool, a preview, missing qualification or a mismatched reward/profile. The output is an immutable **candidate**, not a release. Adding `--fixture` is only for authored development fixtures and always records `qualified_sources: false`.

Export checks every named head against PyTorch using multiple batch/candidate shapes, legality masks, style inputs and seat rotations. It stores parity vectors for testing the actual execution backend later. A backend never silently changes to FP16 or another device. Source or membership changes require a new package; keep old immutable packages for existing saves. To validate an older package after updating the checkout, use its bundled runtime:

```bash
PKG="$ROOT/artifacts/personality/p6/candidate-001"
PYTHONPATH="$PKG/runtime/src" python -m brass_ai.personality.laptop.cli validate --package "$PKG"
```

<a id="p6-stage-the-windows-helper"></a>

### Stage the Windows helper

On the laptop, use a tested Python environment with the existing CPU PyTorch, NumPy, JSON Schema, OpenSpiel, SciPy and P6 requirements. The game launches that absolute Python executable with the package's bundled modules. `requirements-searchlab.txt`, `requirements-personality-p5.txt` and `requirements-personality-p6.txt` declare these dependencies. OpenSpiel and the native host remain the rules adapter used by Masked ISMCTS; this does not reintroduce a separate search algorithm.

```powershell
conda activate ai-training
$Project = 'C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham'
Set-Location -LiteralPath $Project
$Python = (Get-Command python).Source
$Package = "$Project/artifacts/personality/p6/candidate-001"
$Build = "$Project/artifacts/personality/p6/game-build-001"

python "$Project/scripts/personality_laptop.py" stage-game `
  --base-version 1.4.31 --output $Build
```

This compiles staged `BrassLegacyAI.dll` and `BrassPatch.dll` under artifacts. The second DLL adds only a dispatch hook for neural mandatory-liquidation decisions; the accepted rules and human/legacy liquidation handling stay intact. The archived source anchors, game assembly and build bytes are checked. **It does not write to the installed game's Managed directory.** An unexpected baseline fails rather than overwriting another patch.

The integration adds the qualified 16-style catalog to the existing Gamma selection page, with Easy/Medium/Hard and Random. Four-player standard games only are enabled. It keeps two private search workers and the **2/5/12-second whole-turn options**. Both actions/subchoices share one CPU-admitted deadline; Play Now requests the best completed legal incumbent and cancellation retires workers before resource restoration. No action starts merely because a budget expired. A response is validated against the current game, actor, revision and complete native legal path before mutation. If a legal incumbent cannot be produced or validation fails, the game stops visibly rather than silently changing the personality or playing an unsafe move.

Mandatory liquidation uses the same legal, policy-only neural choice as P2/P5. It cannot silently fall through to the stock AI's liquidation heuristic, reset the ordinary turn's saved allowance or count as a search teaching sample. The existing round-end save/UI flow commits one validated liquidation at a time.

The native adapter explicitly maps the game's `local-game/0.3.0` standard four-player extension to the existing `public-state/0.2.0` model boundary. It rejects introductory games, other player counts and non-eight-round configurations, removes the local-only configuration fields, and retains the existing strict schema. Confirmed public action history and the actual round-turn owner are reconstructed separately from temporary liquidation focus. Focused cases check exact permitted-state reconstruction in the unchanged training rules adapter; this is not a general conversion of arbitrary game versions or old datasets.

<a id="p6-sampling-saves-and-public-information"></a>

### Sampling, saves and public information

Each seat receives independent SHA-256 streams derived from package, game, independent sampling seed, seat and stream purpose. A selected style samples one of its eight members uniformly once. Random first selects one of the 16 styles. This does not use the deck-shuffle seed, and difficulty or seat creation order cannot reroll a member.

The versioned outer save envelope preserves the original legacy extension inside it. It pins the package, selected member and profile; saves policy RNG, strategic intent, permitted own-hand recall, public display hysteresis, pending action and remaining whole-turn allowance. A revision/history journal supports save/load and history rewind without replaying a future choice. Mixed legacy/neural tables retain both states. Loading a save with a different package is rejected explicitly; install its pinned compatible package instead of resampling. Existing legacy-only saves retain their existing format and saved clocks, including older 10-second choices.

Only the 16 names/descriptions and measured tiers appear in the menu. Internal member IDs remain private to saves/diagnostics. Default intent text uses public personality/action templates and appears only after the action commits. It does not display private goals, hands, model values or search traces. Broad goal-head hints require a separately P5-qualified `intent-hints` profile and explicit installation opt-in. They are not enabled by default.

<a id="p6-benchmark-cpu-and-optional-intel-graphics"></a>

### Benchmark CPU and optional Intel graphics

Use ORT CPU first. Its initial configuration is two tensor threads, sequential graph execution and disabled idle spinning. Two private search lanes reuse one dynamic inference queue and persistent model sessions. Whole-turn work, including native search and completed results, determines backend choice. GPU memory use is not a performance metric.

Optional Intel comparisons require a working Intel driver and OpenVINO:

```powershell
python -m pip install 'openvino>=2025,<2027'
python "$Project/scripts/personality_laptop.py" benchmark `
  --package $Package --cases "$Project/artifacts/personality/p6/laptop-cases.json" `
  --output "$Project/artifacts/personality/p6/laptop-benchmark-001.json" `
  --backends ort-cpu openvino-cpu openvino-gpu --tiers easy medium hard `
  --unity-running
```

`--unity-running` records an operator assertion; use it only while the isolated review game is actually running. Unsupported devices fail explicitly and are excluded. Optional GPU execution is **not yet verified locally**. CPU defaults follow ORT's [thread-pool guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html); explicit OpenVINO GPU selection follows the [Intel GPU runtime documentation](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html). Backend float32 parity is checked before search measurements.

Cases are a JSON array (1–24 positions) of `{"view": <permitted observation>, "choices": <complete legal root choices>, "seed": 123, "insolvencies": [0,0,0,0]}`. Obtain `view` and `choices` together from the existing native permitted-observation boundary at a completed action boundary; never substitute a full vendor save or fabricated opponent hand. Record actual public insolvency counts. Keep exactly the same cases/seeds across backend comparisons. Add separate copies with `"interrupt": "play-now"` or `"cancel"` and `"interrupt_after": 0.2` for controls.

Real acceptance requires ordinary positions from **both eras**, two distinct seeds per tier/era, useful completed simulations in each stratum, build/develop/sell/network opportunities, no reset deadlines, at most 50 ms measured turn overrun, and successful cancellation/Play Now. Collect representative invested and congested positions using the real model; the small pass-only fixture below is insufficient. The benchmark records cold startup, whole-turn elapsed time, completed simulations, search-quality teaching samples, batching, parent CPU time and process-tree peak RSS. Native child CPU totals and GPU utilization/memory are explicitly unavailable unless measured separately. Search-quality samples are diagnostics, not P4/P5 qualification.

<a id="p6-isolated-unity-review-before-release"></a>

### Isolated Unity review before release

Actual Unity layout, four-player play, pending-action restore and presentation/resource restoration still need human checks. To make those possible before certification, the installer supports a narrowly gated **review copy**. Prepare a separate private copy of the game under a **short** workspace `artifacts` path, e.g. `artifacts/p6-review`, and create a `.p6-fixture-sandbox` marker there. The normal installed game directory is refused for this mode. Use a separate Windows test user/profile so Unity uses a different persistent-data directory: copying the executable alone does not isolate user saves. No interactive game was launched in this task.

```powershell
$ReviewGame = "$Project/artifacts/p6-review"
# Only after preparing the isolated copy:
New-Item -ItemType File -Path "$ReviewGame/.p6-fixture-sandbox"
python "$Project/scripts/personality_laptop.py" install `
  --package $Package --build $Build --game $ReviewGame --python $Python `
  --review-copy --review-backend ort-cpu
# Review the dry-run output, then repeat with --apply for this isolated copy.
```

Launch the copied game with `--brass-personality-review`. Both the explicit flag and marker are required. Fixture entries say **FIXTURE**; unqualified real candidates say **REVIEW**. Neither mode asserts measured difficulty or creates a release certificate. For Intel acceptance, use the matching `--review-backend` after parity succeeds. Fixtures cannot pass the release command. This task used only a minimal isolated DLL sandbox for installer/rollback tests; it did not launch or modify the installed game.

Write a real acceptance record after testing. Its format is `personality-laptop-acceptance/1.0.0`, with `package_id`, `build_id`, `backend`, `target: "Intel Core Ultra 5 135U"`, a named `reviewer`, and boolean `checks`:

```json
{
  "four_player_game": false,
  "sixteen_styles": false,
  "save_load_pending_action": false,
  "history_rewind": false,
  "play_now": false,
  "menu_cancel": false,
  "resource_restoration": false,
  "old_save_unchanged": false,
  "same_profile_strength_review": false
}
```

These are pending checks, not an example passing review. Record evidence for each before changing its value. Confirm that export/backend differences preserve the calibrated profiles' behavior; a faster backend does not automatically establish stronger or equally balanced difficulties.

<a id="p6-certification-installation-and-rollback"></a>

### Certification, installation and rollback

Once qualified real P4/P5 artifacts, matching real laptop benchmarks and human acceptance exist:

```powershell
$Certificate = "$Project/artifacts/personality/p6/release-001.json"
python "$Project/scripts/personality_laptop.py" certify `
  --package $Package --build $Build `
  --benchmark "$Project/artifacts/personality/p6/laptop-benchmark-001.json" `
  --acceptance "$Project/artifacts/personality/p6/laptop-acceptance-001.json" `
  --backend ort-cpu --output $Certificate

python "$Project/scripts/personality_laptop.py" install `
  --package $Package --build $Build --game $Project --python $Python `
  --certificate $Certificate
# Future release installation, with the game closed: repeat with --apply.
```

Installation is a dry run by default. `--apply` requires exact real qualification/certificate pins and the expected installed helpers. It copies an immutable package, saves both prior helpers and the pointer, replaces `BrassLegacyAI.dll` and `BrassPatch.dll`, adds `BrassPersonality/`, and writes a rollback journal. `Assembly-CSharp.dll`, saves and training are untouched. Short content-addressed paths avoid the old Windows runtime's path limit; full hashes remain checked. Copy validation precedes replacement; ordinary failures restore both helpers/pointer. Hashes detect changed bytes; they are not a publisher-signature system.

```powershell
python "$Project/scripts/personality_laptop.py" rollback `
  --journal 'ABSOLUTE-JOURNAL-PATH-PRINTED-BY-INSTALL'
# After reviewing the dry run, repeat with --apply and the game closed.
# For a fixture review sandbox, also supply --review-copy.
```

Rollback verifies both current and backup bytes, restores the exact prior helper and pointer, leaves saves untouched and retains packages. An old helper cannot run a new P6 save; restore the pinned P6 installation to resume it. A forcibly interrupted installation may require restoring the backed-up helper/pointer manually; do not bypass hash checks or delete an unknown installation. No release installation occurred during P6 implementation.

<a id="p6-focused-development-checks-and-measured-fixture-timing"></a>

### Focused development checks and measured fixture timing

From the project root with the existing dependency environment:

```powershell
$env:PYTHONPATH="$Project/src;$Project/tests/R2;$Project/tests/Recovery;$Project/tests/Personality"
python -m unittest discover -s "$Project/tests/Personality" -p test_laptop.py -v
python "$Project/tests/Personality/smoke_laptop.py" `
  --ready "$Project/artifacts/personality/p5-20261012/native-smoke-fixed/ready" `
  --output "$Project/artifacts/personality/p6-smoke-new" --tiers easy medium hard
```

The `--ready` argument reuses a labeled fixture; omit it to generate a fresh fixture if the old fixture/source identity no longer matches. Use a fresh output path. `check_laptop_native.py` compiles focused native history/legality/save checks against a staged helper. `check_laptop_pipeline.py` tests the bundled runtime in its own module path, Play Now, cancellation, exact save restore, late-control isolation, installation dry run and byte-exact rollback in a marked sandbox.

The native report can supply `--debt-cases` to `check_laptop_pipeline.py`; this also checks seven actual native observation/choice pairs against the strict Python schema and reconstructed simulator. The liquidation probe must return a legal policy choice without search or changing the saved ordinary-turn budget. Public error messages omit rejected observations/private card values.

Final fixture artifacts: `artifacts/personality/p6-package-final`, `p6-game-build-final`, `p6-native-final2/report.json` (42 assertions), and `artifacts/p6f/report.json` (standalone service, seven native reconstructions, controls, liquidation, install/rollback). Cancellation returned in 0.422 s and a warm mandatory-liquidation request in 0.156 s. The first cold Play Now request took 7.891 s including lazy model/native-worker initialization; the completed plan itself took 0.422 s. Play Now interrupts planning but does not skip that one-time initialization. Real-model startup latency and interactive presentation still require review before release. `artifacts/personality/p6-preservation.json` verifies unchanged installed DLLs, P0–P5 identity and Step 5 implementation identity.

Measured on this **Core Ultra 5 135U**, ORT CPU, two workers, one ordinary two-action Canal position, untrained small fixture:

| Whole-turn cap | Warm request work, seconds | After CPU admission, seconds | Completed simulations | Search-quality samples |
| --- | ---: | ---: | ---: | ---: |
| 2 s | 1.938 | 1.844 | 0 | 0; legal incumbent fallback |
| 5 s | 4.593 | 4.547 | 0 | 0; legal incumbent fallback |
| 12 s | 10.891 | 10.844 | 8 | 1 |

No deadline resets or overruns occurred. The first request additionally needed 2.437 s to initialize its private native workers; model/service loading took 2.172 s. The 22.16 s benchmark (including startup/cleanup) completed eight simulations and one quality sample. Peak process-tree RSS was about **633 MiB**. These startup costs are reported separately, not hidden inside an apparent 2-second end-to-end latency. A previous probe completed one additional Medium simulation; this illustrates why a single small fixture probe is not a throughput or strength qualification.

Reusing a single already-completed ordered turn trace removed a redundant post-search native replay that had caused roughly 0.22–0.45 s overruns in an earlier fixture probe. Unity still validates every full action against the live game before applying it. This small probe does not establish a general throughput gain or real-model latency. Artifact: `artifacts/personality/p6-final-benchmark.json`; the subsequent mandatory-debt hook preserves this ordinary-turn search path and is checked separately.

Remaining release work: finish the real model and P0–P5 evidence; export and test that model; create representative both-era benchmark cases; compare CPU/Intel graphics on the actual laptop with Unity running; complete interactive save/rewind/cancel/UI/resource and strength review; then certify and explicitly install. No long training or qualification campaign is part of P6 tooling verification.

<a id="p6-p5p6-review-update"></a>

### P5/P6 review update

Packages now use `personality-laptop-package/1.1.0` and the stricter P5 verifier. Real candidates carry the exact P4 report bytes pinned by the roster, plus the complete P5 qualification summary. Missing gates, incomplete members/mixtures/tiers, mismatched actor identities, named outputs, seat contracts or rewards fail validation. Export parity rejects nonfinite legal policy/value/goal outputs, including matching NaNs; masked policy entries must remain negative infinity. Combined utility remains unbounded by probability limits.

The native helper explicitly rejects unsupported neural-save versions instead of treating them as legacy saves. Python restoration validates saved turn keys and tier allowances. Native error popups omit exception payloads that can contain cards or simulated choices. The Random menu text now refers to available styles. These changes preserve two workers, 2/5/12-second clocks, search behavior and legacy-only saves.

Installation dry runs and applied installations now probe the **supplied interpreter** using the same bundled module path as the game. The probe loads every exported graph, checks backend parity and starts/closes the private native host. Missing dependencies or an unavailable backend stop installation before game files change. It installs nothing and runs no full game. Run it directly in the existing conda environment if needed:

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
python "$ROOT/scripts/personality_laptop.py" check-runtime \
  --package "$ROOT/artifacts/personality/p6/package" --backend ort-cpu
# Add --fixture only for a clearly labeled fixture/review package.
```

On Windows use the tested absolute Python executable and the same command. The interpreter must have its dependencies installed in its own environment; development-only `PYTHONPATH` entries are deliberately not inherited by the game. No dependency download happens during the probe. Release certification additionally requires the benchmark's measured CPU identity to match the 135U target; a lab-machine result cannot substitute for that measurement.

An ordinary I/O failure during rollback now restores the current installation so rollback can be retried. Focused failure injection tests cover interruption between the two DLL replacements. Abrupt OS/process termination remains a manual-recovery case; retain the backup and journal. Fixtures still install only into a marked isolated artifacts directory.

Earlier 1.0 packages remain pinned to their own immutable runtime and saves. Do not edit their IDs, replace their runtime in place, or substitute a new package into an old neural save. Re-export/restage new candidates after obtaining compatible P5 evidence; keep the original package to resume its old saves. No real P6 release exists locally, and no live installation occurred. Step 5/IL, P0–P4 and their model/optimizer/counter continuation are unchanged.

Review artifacts: `artifacts/personality/p56-package-review`, `p56-game-review`, `p56-native-review/report.json`, `artifacts/p6r/report.json`, and `artifacts/personality/p56-benchmark.json`. The fixture native checks passed 44 assertions; the standalone pipeline passed exact save restoration, cancellation, Play Now, seven native reconstructions, dependency preflight and isolated installation/rollback.

The new one-position CPU smoke measured admitted planning times of **1.828 / 4.531 / 10.828 seconds**, with **0 / 1 / 8 completed simulations** and **0 / 0 / 1 qualified teaching samples**, respectively. All three caps retained one deadline and had zero measured overruns. Cold model/service loading took 1.86 seconds; the first turn additionally initialized private workers. The cold Play Now RPC took 5.047 seconds including initialization, while cancellation took 0.281 seconds and warm liquidation took 0.078 seconds. Peak process-tree RSS was about 681 MiB. These untrained fixture timings do not qualify difficulty, real-model speed or Intel GPU performance, and are too small for a search-throughput claim.

The review finished with **54 targeted Python checks passing**. A same-bundle microbenchmark replayed the previous duplicate hash traversal under the current validation checks: median 1.450 seconds versus 1.405 seconds for one pass across 274 files (three alternating, warm-cache measurements). This is a small startup saving within measurement variation, not an end-to-end speed claim. Models remain persistent and search settings are unchanged. Artifacts: `p56-hash-benchmark.json` and `p56-review-preservation.json`; the latter verifies that all 231 snapshotted upstream/config/installed-DLL files and the Step 5 implementation fingerprint are unchanged.
