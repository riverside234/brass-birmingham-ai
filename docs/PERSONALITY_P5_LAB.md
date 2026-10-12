# P5: difficulty calibration and intent interfaces

P5 tools are implemented. **Real calibration, human review, and qualification are pending.** The actual Step 5 parent is still training, and there is no qualified P4 roster yet. All local P5 results are explicitly untrained fixtures. No Easy/Medium/Hard level is currently certified, and this package does not install or export a game AI.

The separate `scripts/personality_difficulty.py` command reuses P0's handoff, P1's unchanged shared network, P2's intent/search/central inference, and P4's roster, profiles, frozen reference pool and pinned qualification evidence. It neither resumes nor writes into Step 5. Existing datasets, IL handoff, model weights, optimizer state and training progress retain their existing contracts; **P5 requires no Step 5 checkpoint migration**.

## 1. Environment and prerequisites

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

Before real work, finish P0–P4 using [P0/P1](PERSONALITY_P0_P1_LAB.md), [P2/P3](PERSONALITY_P2_P3_LAB.md), and [P4](PERSONALITY_P4_LAB.md). P5 requires the full 16/128 searched roster, immutable common model, P4 report, public names/descriptions, blind recognition/tactical review, and all pinned sources. Real commands reject fixture parents and unresolved/preview rosters.

## 2. Versioned tier candidates

The starting recipe is `configs/personality/difficulty-1.0.0.json`:

| Candidate tier | Whole-turn cap | Search workers |
| --- | ---: | ---: |
| Easy | 2 seconds | 2 |
| Medium | 5 seconds | 2 |
| Hard | 12 seconds | 2 |

These are **candidates for measurement**, not presumed strength levels. They preserve the plan's laptop choices and two tensor threads. All inherit the same P4 shared weights, per-member style/intent/bias settings, reward, candidate masks, utility scale and supported-root-visits selection. Near-best tolerance is zero in this version: there is no new untested selection algorithm or deliberate blunder injection. Goal-bias strength is not used as the difficulty slider.

P2's existing whole-turn CPU lease and common deadline cover both actions, subchoices and private workers. Search stops early when appropriate. The shared model and frozen references are registered once per collection; tables reuse hosts/search sessions and central dynamic inference. The launcher uses only the first already-visible GPU, with no multi-GPU router. Use `CUDA_VISIBLE_DEVICES` to select an allocated device before running it.

Step 5 remains unchanged: NN teachers 12 seconds/four workers, searched v4 one second/four workers, 50% cumulative search-wall admission, the current PPO/reward/opponent schedules and single-GPU settings. P5 evaluation games are never teaching, PPO, IL or descriptor-fitting data.

## 3. Freeze a calibration panel (when P4 is qualified)

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

## 4. Collect immutable bounded chunks

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

A reservation is written before collection. Expected game failures are retained as failures, without legal-policy replacements or favorable retries. A crashed invocation can leave an incomplete reservation; it is not valid evidence. Diagnose/recover under an explicit study amendment rather than overwriting it, skipping the failed jobs or silently rerunning for a better result. Changed models, search profiles, sources or runtime bytes require new evidence; no hash rewriting is supported.

Reports include completed/interrupted simulations, fallbacks, prior/search seconds, overruns, game time, admission/batching telemetry and the measured machine/backend. Timing on the lab is not a laptop certificate. A changed table allowance, backend or clock does not inherit an existing strength claim. For eventual laptop validation, make a separate declared panel with its actual resources; P6 owns integration and runtime timing.

## 5. Analyze and collect human play feedback

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

## 6. Public selection and intent interfaces

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

## 7. Focused checks and fixture smoke

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

### Local verification, 12 October 2026

- **23 focused tests passed**, covering member/mixture jobs, official ties, grouped/multiplicity intervals, equivalence and failing-member gates, reservations/evidence pins, fixture rejection, UI privacy/hysteresis, shared deadlines/cancellation and exact checkpoint restoration.
- The isolated host built successfully. Two actual four-player **untrained fixture** games completed: one against neural references and one against three original Hard seats. Collection took 36.09 seconds on local CPU; neither game is difficulty evidence.
- A two-second/two-worker probe completed zero simulations and returned a legal fallback (2.09 seconds externally measured, including host observation/application). A separate 12-second probe completed seven simulations, supported its legal plan and returned in 11.14 seconds. Search reported no deadline overrun in either probe; both retired all CPU leases. These are execution checks, not timing or strength qualification.
- Step 5, P0/P1, P2, P3, P4 and learning compatibility fingerprints matched the pre-P5 snapshot. P5 changes no training/checkpoint/data contract. The installed game was not modified.

Local evidence is under `artifacts/personality/p5-20261012/`: `targeted-final-fixed.log`, `native-smoke-fixed/smoke-report.json` and `hard-turn.json`. An initial stock-history bridge failure was corrected before the successful smoke; failed evidence was retained separately. No real calibration, human review, GPU benchmark or release campaign ran.

### P5/P6 review corrections

The P5 verifier is now `personality-difficulty/1.1.0`. A qualified flag alone cannot enable tiers: the portable summary must contain all named gates, the complete member/mixture/tier set, source identities, release coverage and human review. Loading a report also recomputes its metrics and gates from the pinned game records. An original-Hard failure is disqualifying even if a record inconsistently omits the outer failure flag.

Release planning and loading now follow the calibration report's underlying chunk/game/review dependencies, check its exact recipe, and preserve the excluded training/calibration setup groups. Keeping a report after deleting its evidence cannot support qualification. Public menus remain disabled for incomplete evidence.

Previous P5 evidence uses a different verifier/source identity and must not be relabeled. Keep it for comparison; create fresh calibration/release panels under the corrected verifier. No real roster was qualified locally. This change does **not** migrate or invalidate Step 5/IL weights, optimizer state, counters, datasets or P0–P4 artifacts. Their existing continuation commands remain valid.

Focused review checks: `python -m unittest test_difficulty test_laptop test_p56_review test_search -v` with the existing test paths on `PYTHONPATH`. See [P6 deployment](PERSONALITY_P6_DEPLOYMENT.md) for fixture integration and package migration boundaries.

Review verification: **54 targeted Python checks passed**, including the P0 checkpoint restore and P4 profile/public-data boundaries. Two further fixture games completed in 22.157 seconds, with no failed games or remaining CPU leases; the incomplete panel remained unqualified, and loading its report recomputed identical results. Evidence: `artifacts/personality/p56-review-targeted.log` and `p56-calibration-smoke/review-smoke.json`. No real calibration campaign ran.
