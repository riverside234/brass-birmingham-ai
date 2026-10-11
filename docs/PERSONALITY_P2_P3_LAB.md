# Post-Step-5 P2 search and P3 learned repertoire

Implemented 10 October 2026 as separate lab entry points. Finish Step 5 and the [P0/P1 handoff and goal pilot](PERSONALITY_P0_P1_LAB.md) first. Use the accepted, frozen lab parent; local smoke weights are test fixtures. No production personality model or qualified roster has been created locally.

The installed game, Step 5 configs/readers, original generated-data formats, reward and P0/P1 checkpoint fingerprint are unchanged. New code lives in `personality/search/` and `personality/repertoire/`. These packages have their own source contracts. Finish a command before updating code; changed contracts require fresh preparation/collection rather than editing recorded hashes.

## Environment and paths

Use the existing conda environment, PyTorch/OpenSpiel installation and the already imported R1/R4 rules runtimes. No robotics framework or additional ML dependency is needed. CUDA serves the shared neural queue; private rules hosts and search workers remain on CPU. CPU operation is supported for checks.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
brass_root=/data/not_backed_up/yxu209/ai-training
personality_root="$brass_root/artifacts/personality"
```

The examples use `handoff-v1` and `goal-pilot-v1` from the P0/P1 guide. Substitute your actual directories. Run these stages sequentially; do not give a simultaneous Step 5 campaign and discovery job independent use of all 64 cores/GPU memory.

## P2: prepare the current learner

```sh
python "$brass_root/scripts/personality_search.py" --device cuda prepare \
  --handoff "$personality_root/handoff-v1" \
  --pilot "$personality_root/goal-pilot-v1" \
  --output "$personality_root/search-ready-v1" \
  --games 20 --calibration-games 8 --workers 4
```

This copies the current P1 learner into a separate readiness bundle. Fresh direct-play games run the actual causal P2 intent controller and record its saved-intent inputs. Predicted goals are explicitly unqualified during this collection: no search tilt, teacher or promotion is permitted. `context_coverage` reports how many training/validation rows contain an actual saved intent. These calibration games cannot enter PPO or descriptor fitting.

Three bounded utility-head passes use the fresh training games; policy, official outcome and goal tensors stay bit-for-bit fixed. The fit must not worsen training utility MSE. Separate validation games also check utility error: after-fit MSE must not exceed `1.10 * before-fit MSE + 0.005`. This is a numerical readiness screen, not a strength gate.

Goal-confidence temperature is selected on separate calibration setups. Bias stays disabled unless there are at least two setups, 64 observed labels, Brier ≤0.25 and ECE ≤0.10. The goal proposal also needs confidence ≥0.55 and public feasibility. Inspect `ready.json`: an unsuccessful utility check blocks search, while an unsuccessful goal-confidence check permits the unbiased control. Neither result qualifies a release style. Improve the P1 goal fit before expanding discovery if confidence or behavior repeatability is poor.

The bundle pins the source handoff, calibrated checkpoint and both collection receipts. Every search model, reward, profile and target is bound to this readiness identity. Preparing a new learner does not overwrite the source pilot.

## P2: compare root priors under the same budget

```sh
python "$brass_root/scripts/personality_search.py" --device cuda probe \
  --ready "$personality_root/search-ready-v1" \
  --seconds 12 --workers 4 --repeats 2 \
  --report "$personality_root/search-ready-v1/paired-root-timing.json"
```

The probe alternates α=0 and α=0.15 on identical public roots/seeds. It reports legal choices, completed simulations, qualified targets, elapsed time, deadline overruns, prior overhead and shared queue/CPU admission statistics. It is a small initial-position timing check, not a strength comparison. `--unpruned` provides a bounded full-candidate reference under the same time limit. Do not interpret root timing as proof of stronger play or use the probe to qualify 16 styles.

Search behavior:

- The same Masked ISMCTS engine keeps raw four-seat remaining-utility backups and node-local causal insolvency budgets. No new value bonus, heuristic Q normalization or fixed future-turn depth limit is introduced.
- Only the initial action root receives the bounded goal tilt. Both actions and payment/development/sale subchoices share the original clock. Goal predictions do not feed recursively into that same network pass; the input contains the actual prior saved intent.
- The neural tilt keeps a 10% unbiased prior mixture and the existing grouped exploration. At least a quarter of the ordinary root Top-K allowance is selected by unbiased rankings. Industry/action protection, visited candidates, progressive reopening and legal tails remain available, including Pottery.
- Each seat owns its intent, history, age and progress. Productive development toward a higher tier counts as progress. Era changes, lost opportunities and bounded stalling revise commitments. Other simulated players use their own permitted observations and fixed neutral reference policies.
- Four workers own separate trees, host lanes and inference reply channels. Existing within-worker pending-path reservations are selection-only virtual loss. The shared queue batches named policy/value/goal requests across tables. All workers retire before CPU admission is released.
- The **12-second whole-turn clock starts after CPU admission**. Work stops 10% early, capped at 1.2 seconds, for retirement/merging. A completed legal continuation survives timeout/cancellation as fallback; failed/forced/insufficient searches produce no teaching label.

## P2: reserved teachers and policy-only distillation

Ordinary-game counters start with this personality stage, not the completed Step 5 run. The initial warm-up is **640 ordinary games**, followed by one collection due every **160** games. Only committed fresh P1 `joint-pilot` collections count. Held-out diagnostics, descriptor games and teacher games do not advance that counter.

```sh
python "$brass_root/scripts/personality_search.py" --device cuda teach \
  --ready "$personality_root/search-ready-v1" \
  --pilot "$personality_root/goal-pilot-v1" \
  --run "$personality_root/teachers-v1" --tables 4
```

If not due or over budget, this writes the accounting state and returns a reason without collecting games. Continue ordinary fresh P1 joint pilots when needed; each existing invocation is bounded to 32 games. After changing the learner, prepare it again into a new readiness directory before teaching. `teach` rejects readiness tied to an older pilot checkpoint. Keep the same teacher accounting directory across successor pilots.

Each due collection has four all-NN games and at most two collections may be charged to one committed ordinary iteration. For every NN seat, deterministic independent seeds reserve one own turn in Canal 1–4, Canal 5–8, Rail 1–6 and Rail 7–8. Earlier failures cannot consume later phases. Counters distinguish scheduled/reached turns, actual searches, completed simulations, qualified samples, failures and elapsed time per phase.

The **90% wall allowance** is retained: accounted search time may be at most nine times ordinary collection time. Admission reserves an estimated full collection before launch; actual elapsed time, including failures, replaces the reservation on completion. A crashed attempt remains charged, has no unbounded retry and cannot silently gain another allocation. The teacher clock is 12 seconds/four workers, never 12 seconds per action or worker.

```sh
python "$brass_root/scripts/personality_search.py" --device cuda distill \
  --ready "$personality_root/search-ready-v1" \
  --collection "$personality_root/teachers-v1/collection-0000" \
  --output "$personality_root/distilled-v1"
```

Collection admission requires at least 16 qualified positions across two setups, with at least two samples in every phase. Each sample needs two completed alternatives and a selected complete plan with support ≥2. A low-coverage collection returns `updated: false`; it does not manufacture labels.

Successful distillation uses at most eight policy-only steps, LR `1e-5`, and two presentations per sample. The target is 90% completed visits plus 10% original full-legal prior. Official-value, utility-head and goal-head parameters are frozen; no search Q value becomes a value-training target. The shared encoder can change, so the result clears utility readiness. A consumed collection cannot be reused indefinitely. Old Step 5 teachers or labels from a different model/reward/context are rejected. All new teachers stay in versioned sidecars; the base replay remains readable by existing tools.

### Continue PPO from a distilled learner

```sh
python "$brass_root/scripts/personality_search.py" continue-pilot \
  --pilot "$personality_root/goal-pilot-v1" \
  --distilled "$personality_root/distilled-v1" \
  --output "$personality_root/goal-pilot-v2"

python "$brass_root/scripts/personality.py" --device cuda joint-pilot \
  --handoff "$personality_root/handoff-v1" \
  --run "$personality_root/goal-pilot-v2" --games 20 --workers 4 --epochs 2
```

This creates a separate compatible pilot, preserves its imported control and joint counters, and transfers PPO Adam/RNG state when the source was a joint PPO checkpoint. The small teacher optimizer does not replace PPO Adam. It refuses to overwrite a learner that moved since teacher preparation. Joint training collects fresh current-model data, never teacher actions relabeled as PPO. Prepare the successor again before further search teaching.

To use the distilled checkpoint directly for a frozen discovery pilot, run `prepare` with the original `--pilot`, a new `--output`, and `--distilled "$personality_root/distilled-v1"`. This recalibrates its utility and goal confidence. Do not point Step 5 `--initial` at a personality checkpoint.

## P3: inexpensive discovery first

```sh
python "$brass_root/scripts/personality_repertoire.py" propose \
  --ready "$personality_root/search-ready-v1" \
  --output "$personality_root/proposals-v1" --count 64 --decision-mode direct

python "$brass_root/scripts/personality_repertoire.py" --device cuda collect \
  --proposal "$personality_root/proposals-v1" \
  --output "$personality_root/discovery-v1" --partition train \
  --setups 2 --tables 24 --seed 83000000

python "$brass_root/scripts/personality_repertoire.py" --device cuda fit \
  --collections "$personality_root/discovery-v1" \
  --output "$personality_root/descriptor-v1" --epochs 30
```

Sixty-four candidates × two setups × four seat rotations means **512 discovery games**. These are suggested lab commands, not a campaign run by this implementation task. Start smaller if inspecting operation; two candidate controls still require two setup groups to fit. Each call is bounded to 4,096 games and 24 tables. Search confirmation uses four workers per searched decision, within the same shared 64-core budget with four cores reserved; direct discovery uses one admitted host slot and creates no search hosts.

One frozen personality model is shared by every candidate. The neutral control and supported pilot styles are combined with small intent-profile mutations. Direct discovery applies no MCTS goal tilt; its α is zero. Candidate IDs bind model, profile, mode, reward, search contract and opponent pool. The tested seat plays against the same frozen neural reference mix under matched setup/sampling seeds and all four rotations. Archive entries refer to weights rather than duplicating models or optimizer state.

Per-seat descriptors contain public chronological actions, economic changes, industry levels/resources, geographic activity and public merchant connectivity. Both eras and sequence masks are retained. No final win/rank/VP, actual hand/deck, controller/style labels, absolute seat, random seed or machine time enters the encoder. Such metadata is stored separately for quality and split diagnostics. The optional all-seat reader retains opponent trajectories for inspection; active candidate rows fit the descriptor.

The small masked GRU autoencoder produces a 2D descriptor. Normalization and a PCA comparison fit on **training/discovery groups only**. Reports include reconstruction loss, collapse detection, controller centroid dispersion, within/between variance, nearest neighbors, seat/setup associations and correlation with own win share. An authored action-summary baseline provides a third comparison. Low loss or widely separated points alone do not demonstrate human-recognizable personalities.

## P3: independent preliminary quality and an archive

```sh
python "$brass_root/scripts/personality_repertoire.py" --device cuda collect \
  --proposal "$personality_root/proposals-v1" \
  --output "$personality_root/quality-v1" --partition validation \
  --setups 8 --tables 24 --seed 93000000

python "$brass_root/scripts/personality_repertoire.py" archive \
  --proposal "$personality_root/proposals-v1" \
  --encoder "$personality_root/descriptor-v1" \
  --discovery "$personality_root/discovery-v1" \
  --qualification "$personality_root/quality-v1" \
  --output "$personality_root/archive-v1" --method fps --count 16
```

The default preliminary screen requires two discovery setups, eight complete four-seat-rotation quality setups, own mean win share ≥0.20, and the lower end of a setup-clustered approximate 95% interval on the paired difference from neutral ≥−0.05. Descriptor RMS dispersion must be ≤3 in frozen standardized latent units. Win share is primary; own VP and rank are separate diagnostics. Total table points and novelty cannot qualify a weak candidate. These pilot thresholds and intervals are not release certification or a correction for selecting among thousands of candidates.

The unstructured archive holds up to 128 distinct qualified elites with a discovery-space separation of 0.5. Better-supported quality is considered before nearby competitors; single lucky games remain provisional. All observed candidate evidence is retained for later refits. Entries include their exact controller, ancestry, trace IDs, descriptor centroid/dispersion and uncertainty. Weights are deduplicated and `pins.json` protects descriptor, traces, proposals and frozen-model dependencies from existing cleanup.

`--method kmeans` supplies the alternative eight-restart K-means selection. Both methods choose actual tested controllers. K-means centroids are never passed to the policy as style vectors. If fewer than 16 qualify, the output contains fewer than 16 and explicitly marks the shortlist incomplete. A 16-member shortlist still has `release_qualified: false`; P4/P5 must establish recognizable styles, eight nearby variants each, difficulty and release strength.

## Extend, refit and confirm with search

Propose a larger batch around qualified elites while retaining old controller IDs:

```sh
python "$brass_root/scripts/personality_repertoire.py" propose \
  --ready "$personality_root/search-ready-v1" --from-archive "$personality_root/archive-v1" \
  --output "$personality_root/proposals-v2" --count 128 --seed 17702 --decision-mode direct
```

Collect new discovery/quality setups with different seed ranges, fit a new encoder using the old and new discovery collections, then build a new archive with `--previous "$personality_root/archive-v1"`. Pass the new collections as well. The rebuild reloads old retained evidence and re-encodes **every retained candidate**; it never compares old coordinates with new encoder output. Repeated candidate/setup/seat evidence is rejected instead of inflating confidence. The previous archive remains immutable.

For shortlisted search confirmation:

```sh
python "$brass_root/scripts/personality_repertoire.py" propose \
  --ready "$personality_root/search-ready-v1" --from-archive "$personality_root/archive-v1" \
  --output "$personality_root/search-proposals-v1" --decision-mode searched \
  --seconds 12 --workers 4
```

A mode/budget change transfers the tested shortlist's profile recipes plus the neutral control, creates **new** controller IDs, and resets descriptor/quality evidence. Collect fresh matched discovery and validation games and build a separate archive; do not use the direct-policy archive as `--previous`. A changed model or opponent pool likewise requires fresh discovery/qualification. Final-test and held-out Pottery-policy games never train the encoder or select archive entries.

The workflow is explicitly bounded and manually staged. It does not launch an unattended discovery campaign, evolve network weights, export a UI roster, calibrate difficulty or replace the accepted Step 5 model.

## Verification and limitations

Focused tests cover bounded/zero priors, protected candidates, private contexts, cancellation, shared deadlines after admission, reserved late-Rail turns, target provenance, policy-only updates, PPO continuation, public descriptor inputs, grouped splits, encoder refits, elite filtering and actual-member shortlists. Native smoke uses authored model/readiness fixtures with the real four-player rules host. It exercises both searched teacher collection and direct discovery; it does not qualify playing strength.

```sh
cd /data/not_backed_up/yxu209/ai-training/
PYTHONPATH=src:tests/R2:tests/Recovery:tests/Personality \
  python -m unittest test_search test_search_training test_repertoire test_personality -v

python "$brass_root/tests/Personality/smoke_search.py" \
  --output "$personality_root/p2-p3-native-smoke"
```

Real-parent confidence, saved-intent conditioning quality, skill preservation, CUDA throughput, stable repertoire geometry and release qualification remain lab measurements. The separate P1 joint pilot still uses its bounded one-learner/three-parent collection; the full 70/25/5 personality league and P4–P6 product work are not implemented by P2/P3.

The next stage is now available separately: [P4 shared-network rosters](PERSONALITY_P4_LAB.md). It requires enough searched, qualified P3 neighbors for the intended membership; a 16-representative shortlist alone does not supply 128 distinct members. P4 keeps one frozen network and rechecks deployed behavior, quality and replayability before any later tier or laptop release.

### Local verification record, 10 October 2026

- **20 new P2/P3 checks passed**, alongside 52 existing P0/P1, teacher-schedule, combined-reward, champion and selected active Masked ISMCTS checks. CLI help and a two-controller proposal command passed; new source files are not gitignored.
- The final native fixture smoke completed one four-player reserved-teacher game, one direct discovery game and four paired initial-root probes in **111.6 seconds**. All four seats produced valid 16-turn public trajectories with the 32×93 padded schema. The smoke caught and repaired multi-region link connectivity in descriptor opportunities.
- At the normal **12 seconds/four workers**, α=0 completed **19** simulations across two roots; α=0.15 completed **22**. Rates were **0.872 vs 1.009 completed simulations/second**; final publication took **10.875–10.922 seconds**, with **zero whole-turn overruns** and four qualified targets. Prior/context preparation took **1.8–3.7 ms** per root. This tiny fixture screen showed no throughput regression; the apparent 15.7% increase is not a statistically established speed or strength gain.
- The full teacher game's deliberately tiny **0.35-second negative-test clock** yielded no qualified teachers. All 15 attempted searches retired to legal fallback; total retirement overrun was 0.293 seconds. These failures were recorded, not credited as teaching. This is not the normal lab clock.
- Three additional direct calibration games passed the utility-health check and recorded **17 training / 37 validation saved-intent rows**. Goal ECE was **0.214**, so the untrained fixture correctly kept goal bias disabled. The separate native search probe used clearly authored high-confidence goal fixtures to exercise the bias path.
- An exploratory invocation of the retired `tests/SearchLab/test_parallel.py` still fails to import its removed `ParallelS3` symbol. That pre-existing inactive test was left unchanged; the active Masked ISMCTS forest, virtual-loss, candidate and admission checks passed.

The unchanged Step 5 implementation fingerprint is `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`; unchanged P0/P1 source fingerprint is `e8e11585642e084845a62265880a20e23c94025c6f743c8c8633ad1edb4a19a1`. No installed game patch, main training config, production checkpoint or long campaign was changed/run.
