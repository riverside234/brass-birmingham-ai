# Exploration revision: fresh starts and staged continuation

Implemented on 20 October 2026. This is an experiment in exploration, not evidence of improved playing strength. The existing v11.2 model, Features-4 encoding and completed IL checkpoint are unchanged. The utility model still has **5,928,616 parameters**.

## Effective behavior

Newly generated self-play configs use **epsilon 0.03** for ordinary learners and neural history opponents. Action-phase q v3 assigns build/develop/network weights 4, loan/sell 2, scout 1, and unknown kinds 1. Pass receives no exploration mass when another action family is legal. A position containing only pass/discard choices can still select a pass. Industry focus doubles the focused subgroup inside build/develop, without changing that family's total mass. Non-action phases retain q v2 exactly.

Each neural seat's first **two action decisions** use temperature **1.5**; their intervening subdecisions and subsequent actions use **1.0**. Every sampled row records q version, actual temperature, epsilon, latent mixture component, decision counter and marginal behavior probability. The policy identity and collection recipe bind these settings. PPO and its exact categorical probe reconstruct the same row-specific mixture. Forced-action filtering, legal masks and strict probability verification remain active.

The actor entropy bonus now uses the raw temperature-1 policy, including on opening rows. Four coefficients cover Canal/Rail × action/other. Each starts at **0.005**, is bounded by **0.05**, and follows the supplied median/floor/gain/hysteresis rule. Raw and normalized entropy come from the existing fixed IL probe pass on final iteration weights, without consuming training RNG. Ten prospective post-warmup measurements establish the baseline. Before that, coefficients remain at their base. The first automatic adjustment can occur in the next healthy iteration after arming.

The entropy controller holds on unhealthy or invalid updates. Existing economic checks, PPO KL stopping, PPG KL guards and automatic IL-KL limits remain. Retired IL sale/pass-probability checks are not reintroduced. Alpha-cap and excess-entropy alarms are reporting only. `off`, `shadow`, and `apply` are strictly validated modes; the default is `apply`, gated by baseline readiness. In `shadow`, virtual decisions are reported while the applied coefficients stay at 0.005.

V4 policies, teacher/search policies and normal evaluations retain their prior settings. Q v2 remains the default for other policy callers. Teacher search, Monitor, Broader Evaluation and v4 search remain disabled in the current recipe.

## Persisted adoption stages

| Collection age since adoption | Fresh run | Existing run, staged |
| --- | --- | --- |
| 0 | q v3/.03, openings, raw entropy at .005 | q v3/.03 only |
| 1 | Continue | Add openings |
| 2 | Continue | Add raw entropy at .005; start prospective baseline when warmup is complete |
| Baseline ready | Arm the entropy controller | Arm the entropy controller |
| At least 20, baseline ready | Enable branching | Enable branching |

Age counts completed ordinary iterations since the saved origin. If adoption starts at committed **g185**, q changes in the collection producing g186, openings in g187, raw entropy in g188, and branching no earlier than g206. With complete probe evidence and warmup already finished, the entropy baseline completes at g197 and can affect subsequent collections after controller decisions.

The origin, current generation, baseline observations, coefficients, windows and alarm streaks are stored in both the generation checkpoint and committed state. They must agree on restore. Existing economic baselines are retained rather than recalculated. A restart before the first new generation uses the same committed origin; after a generation is saved it restores the recorded rollout state exactly.

Weights, PPO Adam, value-phase Adam, RNG, counters, population, curriculum, fixed IL probe/limits and checkpoint lineage survive the migration. At each exploration regime change, the LR controller clears its short decision window while retaining its multiplier and cooldown. Random variation in branch count does not repeatedly clear that window. This prevents a streak measured under the old collection regime from immediately driving a raise under the new one.

## Branch collection and training

Only four-learner self-play tables can branch. After the two opening decisions per seat, each eligible action decision has probability **0.005**, capped at **one branch per parent game**. Rail positions with an empty draw pile and at most four cards in the acting hand are excluded.

Selection and continuation seeds are hashes of the stable parent ID and decision index. They do not consume the parent's policy RNG. A branch uses the same frozen collection weights, seats, contexts and opening counters. The alternative move excludes the parent's selected choice, then samples the supplied 70% temperature-1 / 25% temperature-2 / 5% q mixture. If exclusion leaves a component with zero support, the remaining components are renormalized and recorded explicitly. Conditional and marginal probabilities are checked on replay admission.

The additive training-host fork snapshots the engine at the decision. Resuming can shuffle only the undrawn deck; hands, discard/wild decks, public state and engine RNG are preserved. A journal entry makes that shuffle replayable. The [g210 scheduling revision](SELFPLAY_G210_REVIEW.md) queues the trusted snapshot immediately to the same bounded ordinary worker pool. A different free lane can continue it while the parent is running; longer ready continuations go first. The initial implementation instead ran each branch after its parent on the same lane.

Nested branch replays are sealed with their parent, permitted only for ordinary self-play, and cannot be imported as standalone PPO games. Prefix rows are not copied. The branch move is marked `branch_move` and excluded from both actor and critic. Subsequent rows form separate per-seat trajectories. Prior insolvency expenditure is carried into the remaining-cost budget, so branching does not reset its cap.

The critic holdout hashes parent families and assigns every branch to its parent's split. Disk-backed previous-iteration critic rows retain branch identity. Branch outcomes and actions are excluded from ordinary health/economy, PFSP, ratings, curriculum and completed-game counts. Gate EV and mixed-table support use ordinary games; branch EV is a separate diagnostic. Both PPO and PPG report actual retained branch row usage.

Every 50 absolute generations, the latest main rating checkpoint also receives a paired temperature-0.7 evaluation with identical setups, seat rotations and opponents. The extra comparison has a separate PL fit and setup-cluster bootstrap; it cannot alter the main rating series or GAE rating slope. The report identifies evidence favoring either temperature, but deployment qualification remains subject to the existing human panel review. No temperature evaluation campaign was launched for this implementation.

## Validation

- **173 distinct focused regressions passed**, counting each test's final outcome across the bounded suite and targeted reruns. Coverage includes exact q weights/masks, opening counters/probabilities, controller/restart sequences, stale-cache rejection, 320-table planning, checkpoint preservation, parent-family holdouts and critic disk caches, separate temperature ratings/retention/export, existing GAE/PPG/IL-KL safeguards and LR behavior. `artifacts/exploration-revision-20261020/validation.json` links the test outcomes to their logs.
- **100 native complete-game fork checks passed** in 320.20 seconds with eight isolated workers. Unshuffled continuations reproduced every subsequent decision context and terminal result exactly. Shuffling changed only the undrawn deck, preserved all four permitted seat observations and preserved acting-seat Features-4 encoding. The encoder admits only the acting seat at a decision; other-seat observations were compared directly.
- The final isolated neural smoke used two native ordinary games, one deliberately forced early branch, the current randomly initialized v11.2 utility model, and one bounded PPO step repeated from the same checkpoint. It passed in **211.33 seconds**. Collection took **72.13 seconds** with the forced branch and **38.50 seconds** without one; the branch itself took **31.48 seconds**.
- All recorded on-policy probabilities reconstructed before PPO. Repeating the update after restoring the checkpoint produced identical weights, Adam and RNG. The original checkpoint remained unchanged.
- Eight ordinary seat-games produced exactly **16** tempered opening decisions and two distinct first actions per physical seat. Ordinary health counted **two** games, while the branch was reported separately.
- The branch generated **294 critic rows / 129 actor-eligible rows** after excluding its move. The bounded PPO step actually used **129 distinct branch actor rows / 17 distinct branch critic rows**. These are usage counts, not additional independent game support.
- The post-update categorical probe covered **411 nonforced learner positions**, including branch continuation rows. Mean raw entropy was **3.197607 nats**, normalized entropy **0.999945**, and categorical KL **0.000001122**. This near-uniform random initialization is not representative of the trained run.
- Ordinary nonforced action counts were build **24**, develop **46**, network **50**, loan **32**, sell **1**, scout **24**, pass **66**. Only four action decisions selected the exploration component: two build, one develop, and one pass in a **pass-only** position with six discard choices. Zero exploration passes occurred with a non-pass alternative. Four draws are far too few to estimate the family mixture; exact arithmetic regressions test its intended shares.

The fork receipts are in `artifacts/exploration-revision-20261020/forks100-parallel/report.json`; the final neural smoke is in `artifacts/exploration-revision-20261020/native-smoke-final/report.json`. A current-code reread of its sealed native games records the explicit pass-only metrics in `native-smoke-final/diagnostics.json`. Fixtures removed independently from the workspace are read from Git for tests only; no deleted user files are restored.

## Runtime and review

Q weighting, opening temperature and raw entropy reuse existing policy/probe work. Branches add collection, encoding, storage and PPO/PPG work. With about 100 eligible decisions per self-play table, the probability of a branch is about `1 - 0.995^100 = 39.4%`, or about 63 of 160 self-play tables; real eligibility and continuation length determine cost. The plan's proposed 8% iteration-overhead target has **not** been verified by this forced-branch smoke. Review production collection time, total iteration time, accepted sample usage and ordinary/branch EV after activation.

At 50-generation intervals the temperature comparison adds one existing-sized rating slot: currently 128 main plus 64 panel games. This is periodic evaluation overhead, separate from branching. The original temperature-1 slot is reused.

No long training was launched, no remote/live checkpoint was edited, and the finished IL directory was not renamed or retrained. Current-run compatibility was checked against the exported g185 signature; optimizer/RNG preservation was tested with isolated authored checkpoints. The actual remote checkpoint remains subject to normal hash, signature and pending-collection checks at resume.

## Commands

From the lab repository, create the staged config from the actual committed run:

```bash
python scripts/joint7-exploration.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic

python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-exploration.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The helper also includes the already requested PPG acceptance/window changes if missing, preserves all other saved settings, and refuses conflicting output. It reads state and writes only the config. Pending old collections must first be recovered with their original source. Repeating the train command continues; it does not restart the stages.

For a future fresh run, use the normal config generator with the completed IL:

```bash
python scripts/feature-engineering.py configure \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-exploration-fresh.json

python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-exploration-fresh.json train \
  --fresh --run artifacts/step5/joint7-v11-2-grouped-lr-critic-exploration \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il --iterations 1
```

Use a new output directory for a fresh run. [The lab instructions](../docs/MLP_V11_LAB.md) explain the optional v11.1-to-v11.2 IL upgrade; an already completed v11.2 IL needs no upgrade.

## Changed components

- `step5/policy.py`, `recovery/exploration.py`: q, sampling identities, opening counters and persisted controller.
- `recovery/config.py`, `ordinary.py`, `league.py`, `lr_controller.py`, `resume.py`: fresh defaults, stage planning, state/signature integration and exact reviewed continuation.
- `recovery/updates.py`, `review_metrics.py`, `il_kl_probe.py`: raw entropy and row-specific probability reconstruction.
- `BrassTrainingHost/FastHost.cs`, `scripts/build-training-host.py`, both matching throughput runtime bundles: training-only fork support.
- `recovery/branching.py`, `execution.py`, `replay.py`, `ingress.py`, reward/insolvency helpers, critic holdout and value-phase reporting: branch admission, budgets, sample attribution and independent evidence.
- `exploration_metrics.py`, `diagnostics.py`, `history.py`, `temperature_rating.py`, `ratings.py`, `retention.py`, report exporter: compact diagnostics and separate evaluation evidence.
- Fresh/staged configs, exact source/handoff review registries, `scripts/joint7-exploration.py`, lab instructions and focused/native smoke tests.

The source review preserves the pre-existing review entries and adds only exact unchanged-config or staged-exploration paths. Learning compatibility is unchanged; no weight or feature reinterpretation is permitted.
