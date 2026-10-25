# P0-P3: handoff, training, search and discovery

[Documentation index and required follow-up](README.md) | [Design](DESIGN.md#design) | [Audit history](AUDIT_HISTORY.md)

Sections: [P0/P1](#p0-p1) | [P2/P3](#p2-p3). See [encoder reuse status and remaining validation](README.md#encoder-reuse-follow-up). Existing commands and historical verification records are preserved.

**Current workflow decision (24 October 2026):** [Skip the goal-head stage](README.md#goal-head-skipped) and [temporarily skip P2 search preparation](README.md#p2-skipped). Do not run goal-head warm-up, `validate-goals`, the goal-loss `joint-pilot`, P2 readiness/calibration, search probes, teacher collection or distillation in the upcoming workflow. Retain the implementations and commands as historical reference. After P0 and a validated direct-policy checkpoint/context adapter, proceed to P3 with the seven supported Step 5 inputs. [P4 prioritizes a smaller qualified set](README.md#small-roster), initially targeting seven distinct styles; the 16-by-8 target is canceled. The old P1/P2 model/readiness dependencies still need adaptation; this note implements no bypass.

**Performance update (`20261016` deployment evidence):** Step 5's [endpoint encoders and shared feature caches](README.md#step5-encoder-reference) are now integrated across personality `SearchSession` coordinator/worker/opponent lanes. The session also requests fused native snapshots. [Local deployment measurements](DEPLOYMENT_PERFORMANCE.md) cover the shared search path, not real P1-P3 collection throughput. These are implementation records for the retained P1/P2 branch; its execution is currently skipped. The experimental one-path batching, three-worker comparison and W352 model do not change training defaults or existing checkpoints.

<a id="p0-p1"></a>

## Post-Step-5 personalities: P0 handoff and P1 goal head

**13 October audit:** [P0–P6 findings and verification](AUDIT_HISTORY.md#audit-2026-10-13-p0-p6). P0 now revalidates manifest claims against its copied Step 5 evidence. P1 retains saved RNG across the warm-up-to-joint transition while creating the required fresh optimizer. The exact migrations below supersede the older blanket instruction to start a new pilot.

**11 October review:** [P0–P4 corrections and compatibility notes](AUDIT_HISTORY.md#audit-2026-10-11-p0-p4). P1 now rejects externally selected/search behavior and non-unit sampling temperature before PPO attribution. Personality source contracts changed; create a fresh P0 handoff and derived pilot after updating. The accepted Step 5 checkpoint and its training readers are unchanged.

Implemented 10 October 2026. These commands use the **completed, stopped Step 5 run on the lab**. They do not need a model on the laptop. Wait for your current training run to finish; the importer refuses an active run. Replace `selfplay-best-v1` below with the actual completed run folder.

P0/P1 is a separate, bounded experiment. The installed game and Step 5 training, IL handoff, search settings and checkpoint readers are unchanged. Goal-biased Masked ISMCTS, persistent intent and archive discovery are now implemented in the separate [P2/P3 workflow](TRAINING_P0_P3.md#p2-p3). The full personality league and release qualification remain later work. The four pilot profile codes are input controls, not yet proven playing personalities.

<a id="p0-p1-1-inspect-the-accepted-model"></a>

### 1. Inspect the accepted model

Use your existing conda environment and dependencies. No new package installation is required.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py inspect \
  --from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1
```

The check resolves `best.json` to the immutable `state.json` champion entry and checks its hash against `best.pt`. It validates the accepted gate's specification, report and result, the model's run/generation lineage, the rules/features contract and the calibrated utility head. An IL-only initial incumbent, mismatching reward, incomplete gate or changed dependency is rejected. It does **not** fall back to the latest learner. If no PPO candidate has passed Step 5's gate, the real P0 handoff must wait.

This implementation accepts the current reward fingerprint:

```text
W + 0.10 * pairwise_placement + 0.001 * scored_VP - min(0.005 * insolvency_events, 0.025)
```

Returns stay undiscounted and subtract only future insolvency costs at each decision. The utility output remains unbounded; it is not a probability. Changing the reward requires an explicit future migration, rather than editing a receipt. A passed Step 5 selection gate is preliminary training evidence, not a personality release certificate.

<a id="p0-p1-2-freeze-the-p0-handoff"></a>

### 2. Freeze the P0 handoff

Use a new output directory:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py handoff \
  --from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1 \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --trace-shards 4 --references 3
```

The handoff takes the existing run/cleanup lock, pins dependencies **before copying**, and verifies every copied hash. It preserves the parent, an age-spread reference sample, reward/calibration metadata, accepted selection receipts, resolved configuration, runtime/bank contracts and available ordinary training shards with their authoritative receipts. It records every shard's game/setup/split identity. Old style/intent fields retain their original meaning; absent new personality context is explicitly absent.

`manifest.json` and `pins.json` describe the protected bundle. Dependency pins are written in the handoff directory; retention discovers them there without changing any source-run files. With zero-hour Step 5 cleanup, keep these pins while the handoff is in use. Optional cleanup review:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1
```

This command is the existing cleanup dry run; do not add `--apply` merely to inspect it. P0 never recovers missing training data from evaluation games. If ordinary traces were already retired, the manifest says so; use fresh warm-up collection in the next section. You can explicitly set `--trace-shards 0` when only fresh discovery data is wanted.

<a id="p0-p1-3-warm-up-the-new-head-with-the-core-frozen"></a>

### 3. Warm up the new head with the core frozen - skipped

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda warmup \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --epochs 3
```

This uses the copied ordinary training shards only for auxiliary supervision. It starts a new optimizer, copies all trained policy/official-value/utility tensors, and keeps those tensors and the new context projection frozen. It checks that the copied tensors remain bit-for-bit unchanged. The goal learning rate is `1e-4`.

If no suitable traces remain, add `--games 20 --workers 4` to collect fresh policy games first. Each table has one learner seat and three frozen parent seats, with private rules hosts and batched inference. Twenty games cover all five input codes in all four seats: neutral, industrialist, networker, supplier and investor. These are provisional controls; no heuristic teacher or recovery-opponent schedule is started. `--device cpu` is also supported.

The medium architecture has **1,915,962 total parameters**: the copied 1,850,129-parameter network plus 65,833 goal/context parameters. The new contract checks a 100,000-parameter extension limit and a 2,100,000 total limit. Its named outputs are policy logits; four-seat win share, placement, VP and remaining utility; four goal-factor logits; and six-by-three event-confidence logits. Import initially preserves all five old outputs, including the trained utility correction and remaining cost budgets.

Goal labels use **public realized outcomes**, never future features or opponents' cards. They cover sales, a newly built industry's later flip, resource consumption, network extension, liquidity recovery and flexibility. Industry factors include Pottery and all other industries. Region uses public city areas. Horizons count the current remaining own turn and up to two/four own turns in the same era. Missing final effects, short horizons and ambiguous simultaneous completions are masked. An unsold build is not a completed industry goal. These are behavioral proxies, not proof of optimal strategy or the model's mental intent.

<a id="p0-p1-4-measure-held-out-goal-diagnostics-then-run-a-small-joint-pilot"></a>

### 4. Measure held-out goal diagnostics, then run a small joint pilot - skipped

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda validate-goals \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --games 20 --workers 4

python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda joint-pilot \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --games 20 --workers 4 --epochs 2

python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda validate-goals \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --games 20 --workers 4
```

The joint pilot collects **fresh current-model learner trajectories**. Frozen parent actions never become learner PPO targets. It records and checks the exact action probability, actual predecision profile and model identity. Saved intent is explicitly absent in this P1 collector; the separate P2 controller produces it causally. Old traces cannot enter PPO through the auxiliary importer.

The pilot takes LR, clipping, KL limit, entropy/value weights, batch limits and gradient clipping from the handoff's resolved Step 5 settings. Current values are LR `1e-5`, clip `0.2`, KL `0.01`, entropy `0.005`, value weight `0.5`, gradient clip `5`, actor/critic batch `2048`, microbatch `512`, and `131072` candidate cells. Goal loss has weight `0.05`. Actor batches exclude forced choices; separate critic/goal batches can use them. The current CLI defaults to two joint epochs. There are no style rewards, search-value pseudo-targets or Pass penalties.

`validate-goals` uses fresh grouped validation seeds, applies no gradients and leaves model bytes unchanged. Reports include masked goal loss, confidence Brier/ECE, separate official-head losses, remaining-utility MSE/MAE, official win share/place/VP by profile, and raw/weighted reward components. These small diagnostics are neither a matched strength tournament nor calibrated difficulty levels. Confidence is reported as **unqualified**. A joint update clears inherited search readiness; P2 must check/recalibrate the changed utility evaluator on appropriate fresh data before teaching.

<a id="p0-p1-5-checkpoints-traces-and-scope"></a>

### 5. Checkpoints, traces and scope

The 13 October audit adds exact `p0-p1-audit-continuation/1.0.0` receipts in `configs/personality/source-migrations.json`, from either reviewed source `3bb0eab8a753a5fb29e33b595e19e6ac0ab6d2e2d09875c059e93c9387bee068` or `31b8d44872bbfcd9b89c8e3cc591ef38337fe39fbdebf64d4dc8a76ba46464ff` to `7857312a03491be8102a621826f1bcfc58018eaa4416f9dca26de5049ae54176`. The historical 12 October receipt remains intact. Migration requires identical model, goal, feature and base contracts, validates old weights against their original identity, and revalidates copied P0 evidence. Import retains compatible tensors, Adam state, RNG and pilot counters while assigning the current model identity. Old artifact files are not rewritten on load; successful updates record the migration and new checkpoint/state. Stop pilot commands, preserve the run, update code and this receipt together, then use the same P1 command, handoff and run path to continue a completed compatible pilot. Other source versions remain rejected. Recreate P2 readiness, teacher/discovery collections and P3–P6 qualification/export under the changed identity; this migration does not authorize old search labels or qualification receipts.

The separate `personality-checkpoint/1.0.0` stores the named-output model, context/goal/reward contracts, source fingerprint, optimizer and RNG. Step 5 readers intentionally reject this new model format. Repeating a completed warm-up or joint command on the same run continues its model and same-phase optimizer. Transitioning from frozen warm-up to joint training creates a fresh optimizer for the enlarged trainable parameter set and restores the saved CPU/CUDA RNG stream independently. Every joint invocation recollects current-model PPO data. After joint training, start a separate run if you want to repeat frozen-core preparation.

`initial.pt` preserves the imported control. Pilot state version `personality-pilot/1.1.0` writes each update's immutable checkpoint in its phase directory before atomically publishing `state.json` with its relative `model_path` and hash. `model.pt` is a convenience copy of the committed P1 learner, **not an automatically promoted incumbent**; P1 and P2 read the authoritative state pointer if a final convenience-copy operation was interrupted. The original `1.0.0` state layout without `model_path` continues to resolve `model.pt`, with the same strict hash and source checks, and upgrades on the next successful update. The frozen Step 5 parent is retained throughout this pilot. The base generated-replay format is unchanged: new profile/role/model contexts live in a hash-bound sidecar. Training, joint and diagnostic seed ranges are separate, with grouped train/validation partition checks. Old generated IL datasets and their readers are unchanged.

Operations are bounded to 1–32 fresh games, at most eight host workers and 1–10 passes per call; up to 32,768 prepared examples are accepted. Use fewer handoff shards if that limit is reached. Four tensor threads are the CLI default. P0/P1 uses small batched policy collection and does not yet implement the planned 70/25/5 full personality league or P2's resource-managed search queues. Do not treat this pilot as the large Step 5 training driver.

Completed operation reports are retained, with the last eight referenced in `state.json`. These bounded pilots do not run automatic campaign cleanup. A partial operation is rejected rather than silently overwritten. If interruption preceded the state commit, preserve the named uncommitted phase directory under `<run>/interrupted/`, then rerun the same command: it restores the previous committed model, optimizer, RNG and counters. For example, after stopping all pilot commands, move `warmup-0001` to `interrupted/warmup-0001` when `state.json` still records `warmup_runs: 1`. Do not change the state or checkpoint hash, and do not move a phase directory referenced by `state.json.model_path`. If only the final convenience copy failed, the state already records the completed operation and the next invocation continues from its immutable checkpoint without moving anything. Source changes invalidate P0/P1 compatibility explicitly, so finish an operation before updating code. Do not use local smoke fixtures as a production handoff.

The interrupted-phase procedure above applies to operations run with the audited persistence code. If an older implementation already overwrote the committed `model.pt` before failing, moving the orphan phase cannot reconstruct the prior bytes. Preserve all files and recover a matching known-good checkpoint/state backup or use the original-source recovery procedure; do not edit the recorded hash to admit an unrelated checkpoint.

<a id="p0-p1-verification"></a>

### Verification

Local verification on 10 October passed **13 P0/P1 tests and 23 existing combined-reward/champion tests**. The final native smoke completed three four-player games in **68.7 seconds** on CPU, with 70 non-forced learner decisions, 32 labeled training turn roots and 16 held-out turn roots. Reload, frozen-parent preservation and unchanged validation checkpoint bytes passed. These numbers describe fixture operation, not trained-model performance. The unchanged Step 5 implementation fingerprint is `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`.

The focused suite checks accepted-parent rejection, hashes/pins, old-reader rejection, neutral output equivalence including nonzero utility corrections, parameter count, public-only labels, era/horizon masks, all-industry access, frozen-core warm-up, learner-only PPO probabilities and checkpoint/optimizer restoration:

```sh
cd /data/not_backed_up/yxu209/ai-training/
PYTHONPATH=src:tests/R2:tests/Recovery:tests/Personality \
  python -m unittest discover -s tests/Personality -p 'test_*.py' -v
```

A small optional native smoke creates **authored fixture weights/evidence**, runs two fresh joint games and one held-out diagnostic game, and checks reload and source preservation:

```sh
python /data/not_backed_up/yxu209/ai-training/tests/Personality/smoke.py \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/personality/native-smoke-local
```

Local CPU verification does not establish CUDA throughput, playing strength, real-parent goal quality, or P2 search readiness. Those require the completed lab parent and later comparison gates in [the personality plan](DESIGN.md#design).

<a id="p2-p3"></a>

## Post-Step-5 P2 search and P3 learned repertoire

**Current scope (updated 24 October 2026):** [P2 search preparation is temporarily skipped](README.md#p2-skipped), including generic readiness/calibration, search probes, teacher collection and policy-only distillation. Goal-generated intent and goal-biased search remain excluded with P1. The P2 commands below are historical reference, not current prerequisites to execute. Implement and validate a direct-policy entry for P3 without fabricating P1/P2 receipts; keep model/input compatibility, provenance and independent quality checks. Future use of search requires its own readiness and controller evidence.

**13 October audit:** [P0–P6 findings and verification](AUDIT_HISTORY.md#audit-2026-10-13-p0-p6). Readiness calibration must match the saved model receipt and count only setups with observed labels. Teacher context, goal, profile and search settings must match the executed collection. Preparation failures now retire inference resources. Regenerate downstream evidence after the P0/P1 source migration; P3 schemas and selection settings are unchanged.

**11 October review:** [P0–P4 corrections and compatibility notes](AUDIT_HISTORY.md#audit-2026-10-11-p0-p4). Rebuild personality artifacts under the current contracts. P3 now uses the 32×95 `public-behavior-trace/1.1.0` schema; the older verification below describes historical fixtures. Step 5 and base generated-data formats remain unchanged.

Implemented 10 October 2026 as separate lab entry points. Finish Step 5 and the [P0/P1 handoff and goal pilot](TRAINING_P0_P3.md#p0-p1) first. Use the accepted, frozen lab parent; local smoke weights are test fixtures. No production personality model or qualified roster has been created locally.

The installed game, Step 5 configs/readers, original generated-data formats and reward are unchanged. P2/P3 code lives in `personality/search/` and `personality/repertoire/`. These packages have their own source contracts. Finish a command before updating code; changed contracts require fresh preparation/collection rather than editing recorded hashes. The accompanying P1 review fix also changes its source fingerprint.

<a id="p2-p3-environment-and-paths"></a>

### Environment and paths

Use the existing conda environment, PyTorch/OpenSpiel installation and the already imported R1/R4 rules runtimes. No robotics framework or additional ML dependency is needed. CUDA serves the shared neural queue; private rules hosts and search workers remain on CPU. CPU operation is supported for checks.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
brass_root=/data/not_backed_up/yxu209/ai-training
personality_root="$brass_root/artifacts/personality"
```

The examples use `handoff-v1` and `goal-pilot-v1` from the P0/P1 guide. Substitute your actual directories. Run these stages sequentially; do not give a simultaneous Step 5 campaign and discovery job independent use of all 64 cores/GPU memory.

<a id="p2-p3-p2-prepare-the-current-learner"></a>

### P2: prepare the current learner

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

<a id="p2-p3-p2-compare-root-priors-under-the-same-budget"></a>

### P2: compare root priors under the same budget

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

<a id="p2-p3-p2-reserved-teachers-and-policy-only-distillation"></a>

### P2: reserved teachers and policy-only distillation

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

The master personality plan §4.4 instead specifies a 50% accounted ceiling, matching current Step 5. This is an unresolved specification discrepancy: active P2 code and this workflow use 90%. The 13 October audit preserves that behavior and all clocks/worker counts; choose and explicitly version the intended P2 accounting policy before a real teacher campaign.

```sh
python "$brass_root/scripts/personality_search.py" --device cuda distill \
  --ready "$personality_root/search-ready-v1" \
  --collection "$personality_root/teachers-v1/collection-0000" \
  --output "$personality_root/distilled-v1"
```

Collection admission requires at least 16 qualified positions across two setups, with at least two samples in every phase. Each sample needs two completed alternatives and a selected complete plan with support ≥2. A low-coverage collection returns `updated: false`; it does not manufacture labels.

Successful distillation uses at most eight policy-only steps, LR `1e-5`, and two presentations per sample. The target is 90% completed visits plus 10% original full-legal prior. Official-value, utility-head and goal-head parameters are frozen; no search Q value becomes a value-training target. The shared encoder can change, so the result clears utility readiness. A consumed collection cannot be reused indefinitely. Old Step 5 teachers or labels from a different model/reward/context are rejected. All new teachers stay in versioned sidecars; the base replay remains readable by existing tools.

<a id="p2-p3-continue-ppo-from-a-distilled-learner"></a>

#### Continue PPO from a distilled learner

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

To use the distilled checkpoint directly for a frozen discovery pilot, run `prepare` with the original `--pilot`, a new `--output`, and `--distilled "$personality_root/distilled-v1"`. This recalibrates its utility and goal confidence. The original pilot must still match the exact generation used to prepare the teacher; sharing the same Step 5 parent is insufficient. Do not point Step 5 `--initial` at a personality checkpoint.

<a id="p2-p3-p3-inexpensive-discovery-first"></a>

### P3: inexpensive discovery first

**Current direction (24 October 2026):** [Reuse one frozen network with supported contexts](README.md#frozen-network-discovery), while skipping P1 and temporarily skipping P2. The commands below describe the existing goal-enabled pipeline; they are not yet runnable instructions for handing the current Step 5 checkpoint directly into direct-policy discovery without P1/P2. First adapt the checkpoint/context entry and the old readiness dependency. Start with the source run's supported `neutral` and six industry-focus contexts, not the separate five personality profiles. Remove inactive goal controls from candidate identity/variation rather than counting different `alpha`, `patience` or `hysteresis` values as additional styles. Preserve matched evaluations, held-out quality checks and provenance. No code or command-line bypass is implemented by this note.

The initial target is seven styles corresponding to the seven supported inputs, with priority given to good playing quality, repeatability and clear behavioral separation. Report a smaller qualified subset when fewer styles meet those requirements, and identify indistinguishable, weak or inconclusive candidates separately. The former 16-archetype/eight-variant target is canceled; the old shortlist counts and archive examples below are retained implementation history. If evidence motivates learning new behaviors through weight updates, [PPGA adaptation is an optional independent experiment](README.md#optional-ppga-experiment). Define and validate that adaptation before running a bounded pilot; the historical commands below do not implement PPGA or automatically launch weight training.

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

Larger proposals can now use `--candidate-start` and `--candidate-count` without creating new proposals or changing the matched random streams. For an existing **2,048-candidate** proposal, two discovery setups fit into four slices:

```sh
for start in 0 512 1024 1536; do
  python "$brass_root/scripts/personality_repertoire.py" --device cuda collect \
    --proposal "$personality_root/proposals-large" \
    --output "$personality_root/discovery-large-$start" --partition train \
    --setups 2 --tables 24 --seed 83000000 \
    --candidate-start "$start" --candidate-count 512
done
```

Each slice contains 4,096 games; the loop is a lab campaign command, not a smoke test. Include **all four output directories** in `fit --collections` and `archive --discovery`. The neutral control is collected only in the slice containing candidate zero. Keep proposal, partition, setup count and seed identical across slices; use unique outputs and nonoverlapping candidate ranges. Eight-setup validation uses slices of **128 candidates**, the same validation seed across slices, and `archive --qualification` receives all slices. Duplicate candidate/setup/seat evidence is rejected.

One frozen personality model is shared by every candidate. The neutral control and supported pilot styles are combined with small intent-profile mutations. Direct discovery applies no MCTS goal tilt; its α is zero. Candidate IDs bind model, profile, mode, reward, search contract and opponent pool. The tested seat plays against the same frozen neural reference mix under matched setup/sampling seeds and all four rotations. Archive entries refer to weights rather than duplicating models or optimizer state.

Per-seat descriptors contain public chronological actions, economic changes, industry levels/resources, geographic activity and public merchant connectivity. Both eras and sequence masks are retained. No final win/rank/VP, actual hand/deck, controller/style labels, absolute seat, random seed or machine time enters the encoder. Such metadata is stored separately for quality and split diagnostics. The optional all-seat reader retains opponent trajectories for inspection; active candidate rows fit the descriptor.

Cash/income changes and tile flips use the observation immediately after each own choice, rather than the next own turn after opponents act. `effects_observed` marks the fraction with observable same-era consequences; `era_boundary` marks censored era-clear transitions. Missing final/era-clear observations are not invented. Merchant beer follows the native `merchant:` target format. Existing descriptor weights must be refitted for these corrected 95 fields.

The small masked GRU autoencoder produces a 2D descriptor. Normalization and a PCA comparison fit on **training/discovery groups only**. Reports include reconstruction loss, collapse detection, controller centroid dispersion, within/between variance, nearest neighbors, seat/setup associations and correlation with own win share. An authored action-summary baseline provides a third comparison. Low loss or widely separated points alone do not demonstrate human-recognizable personalities.

<a id="p2-p3-p3-independent-preliminary-quality-and-an-archive"></a>

### P3: independent preliminary quality and an archive

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

<a id="p2-p3-extend-refit-and-confirm-with-search"></a>

### Extend, refit and confirm with search

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

<a id="p2-p3-verification-and-limitations"></a>

### Verification and limitations

Focused tests cover bounded/zero priors, protected candidates, private contexts, cancellation, shared deadlines after admission, reserved late-Rail turns, target provenance, policy-only updates, PPO continuation, public descriptor inputs, grouped splits, encoder refits, elite filtering and actual-member shortlists. Native smoke uses authored model/readiness fixtures with the real four-player rules host. It exercises both searched teacher collection and direct discovery; it does not qualify playing strength.

```sh
cd /data/not_backed_up/yxu209/ai-training/
PYTHONPATH=src:tests/R2:tests/Recovery:tests/Personality \
  python -m unittest test_search test_search_training test_repertoire test_personality -v

python "$brass_root/tests/Personality/smoke_search.py" \
  --output "$personality_root/p2-p3-native-smoke"
```

Real-parent confidence, saved-intent conditioning quality, skill preservation, CUDA throughput, stable repertoire geometry and release qualification remain lab measurements. The separate P1 joint pilot still uses its bounded one-learner/three-parent collection; the full 70/25/5 personality league and P4–P6 product work are not implemented by P2/P3.

The next stage is now available separately: [P4 shared-network rosters](QUALIFICATION_P4_P6.md#p4). It requires enough searched, qualified P3 neighbors for the intended membership; a 16-representative shortlist alone does not supply 128 distinct members. P4 keeps one frozen network and rechecks deployed behavior, quality and replayability before any later tier or laptop release.

<a id="p2-p3-local-verification-record-10-october-2026"></a>

#### Local verification record, 10 October 2026

- **20 new P2/P3 checks passed**, alongside 52 existing P0/P1, teacher-schedule, combined-reward, champion and selected active Masked ISMCTS checks. CLI help and a two-controller proposal command passed; new source files are not gitignored.
- The final native fixture smoke completed one four-player reserved-teacher game, one direct discovery game and four paired initial-root probes in **111.6 seconds**. All four seats produced valid 16-turn public trajectories with the 32×93 padded schema. The smoke caught and repaired multi-region link connectivity in descriptor opportunities.
- At the normal **12 seconds/four workers**, α=0 completed **19** simulations across two roots; α=0.15 completed **22**. Rates were **0.872 vs 1.009 completed simulations/second**; final publication took **10.875–10.922 seconds**, with **zero whole-turn overruns** and four qualified targets. Prior/context preparation took **1.8–3.7 ms** per root. This tiny fixture screen showed no throughput regression; the apparent 15.7% increase is not a statistically established speed or strength gain.
- The full teacher game's deliberately tiny **0.35-second negative-test clock** yielded no qualified teachers. All 15 attempted searches retired to legal fallback; total retirement overrun was 0.293 seconds. These failures were recorded, not credited as teaching. This is not the normal lab clock.
- Three additional direct calibration games passed the utility-health check and recorded **17 training / 37 validation saved-intent rows**. Goal ECE was **0.214**, so the untrained fixture correctly kept goal bias disabled. The separate native search probe used clearly authored high-confidence goal fixtures to exercise the bias path.
- An exploratory invocation of the retired `tests/SearchLab/test_parallel.py` still fails to import its removed `ParallelS3` symbol. That pre-existing inactive test was left unchanged; the active Masked ISMCTS forest, virtual-loss, candidate and admission checks passed.

The unchanged Step 5 implementation fingerprint is `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`; unchanged P0/P1 source fingerprint is `e8e11585642e084845a62265880a20e23c94025c6f743c8c8633ad1edb4a19a1`. No installed game patch, main training config, production checkpoint or long campaign was changed/run.
