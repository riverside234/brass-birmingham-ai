# P4: unified deployment styles

P4 implements the **one frozen shared network** option in [the personality plan](../ai-research/PERSONALITIES_AND_DIFFICULTY_PLAN.md#56-bring-discovered-behavior-back-into-one-network). It selects 16 archetypes, each containing its tested representative and seven nearby tested variants. There are 128 unique controllers and one set of neural weights. The five pilot neural profile inputs remain supported inputs; archetypes combine those trained inputs with tested intent/prior settings. Sixteen distinct behaviors must be demonstrated, not inferred from 128 different numbers.

The actual Step 5 parent is still training on the lab. Complete the [P0/P1 handoff](PERSONALITY_P0_P1_LAB.md) and [P2/P3 calibration/discovery](PERSONALITY_P2_P3_LAB.md) first. Local authored fixtures do not qualify a learned roster. This workflow changes neither Step 5 nor the installed game. P5 handles difficulty and UI; P6 handles laptop export, once-per-game sampling/save restoration and installation.

## 1. Required input

Use a P3 archive with all candidates evaluated under **the same frozen ready model, opponent pool, reward and search configuration**. For a full roster, the archive must contain searched-controller confirmation, not just direct-policy discovery. Expand the confirmed proposal to include enough neighbors: a 16-entry representative shortlist alone cannot supply 128 members.

Selection requires each member's existing P3 preliminary quality gate, at least eight matched setup groups, mean own-seat win share at least 0.20 and a control-difference lower endpoint of at least −0.05. It compares FPS and K-means representatives, uses the chosen method fixed before new calibration, and keeps actual tested recipes. Neither a cluster centroid nor an averaged set of weights becomes a style input.

Neighborhoods are relative to each representative's nearest other representative, not one global radius. Defaults bound radius to 0.40 of that distance, retain a 0.10 separation margin and require member differences greater than 0.50 times their combined game-to-game dispersion. Missing, indistinguishable, noisy or shared members cause selection to fail with the affected count. Expand discovery instead of duplicating variants or lowering evidence gates after seeing the calibration results.

## 2. Select and name a roster

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

## 3. Freeze a bounded collection plan

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

## 4. Collect chunks on the lab

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

## 5. Re-encode and check the deployed roster

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

## 6. Changed or distilled networks

P4's implemented path needs no extra distillation: the P3 entries already share one frozen network. It rejects archives mixing checkpoints and does not average weights, relabel hindsight descriptors as policy inputs or turn calibration decisions into PPO samples.

If a new network is produced by P2 policy distillation/joint continuation, first run P2 readiness again, then collect a **new P3 discovery/quality archive** under the new model identity and reselect/recheck P4. All 128 deployed controllers must be re-encoded and requalified. Old coordinates and qualification receipts cannot authorize changed weights. Cross-checkpoint consolidation is not a second implementation path in this version; use the shared-generation path unless a later explicit importer/training design is added.

## Verification

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

### Local verification, 10 October 2026

- **14 P4 checks passed**, alongside **33 existing P0/P1/P2/P3 checks**. Coverage includes exact 16/128 membership, shared weights, noise/collapse rejection, seeded rotations/mixtures, changed-model and duplicated/reused-evidence rejection, public-only review inputs, immutable panels, variety controls and unqualified partial reports.
- The final native smoke completed **one four-player direct-preview game in 24.0 seconds**, exercising collection, both eras, public board/plan extraction and immutable reporting. It intentionally collected only one job from its 104-job fixture plan and correctly failed qualification. Its artifacts are under `artifacts/personality/p4-native-final-20261010/` and remain fixtures.
- CLI help, new-source syntax/whitespace, guide links and `git diff --check` passed. The final smoke report loads under the current P4 contract; its pinned weights/evidence validate.
- Step 5 and P0/P1 compatibility fingerprints are unchanged. No installed game files, main training settings, production models or generated training-data schemas were modified. P4 has no new online model parameters or search algorithm changes. CUDA utilization, real style recognition/strength and a complete 16/128 roster remain lab measurements; this small CPU smoke is not a throughput comparison.
