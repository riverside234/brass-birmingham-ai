# B04.5 plan: all-seat teacher tables and a search-value correction head

**Current production stage specification (2026-11-08):** the user replaces the
auxiliary gates below with **A10 head-only → B20 unconditional blend warm-up →
C10 unconditional trunk warm-up**. No 256-game test or validation-based rollback.
Four held-out games provide diagnostics every tenth iteration, except each
pending beta-qualification iteration when the new config override is true.
Remove the head's policy-staleness gate (which starved it through g1280); keep
independently validated value shards, the hard joint KL cap .003 and PPG beta
floor .25. All original gated-stage/window/H2H specifications below are historical
for this revised recipe. See [replacement specification and implementation](RUST_TEACHER_STAGE_SCHEDULE_REVISION.md).

**Current Stage A activation (2026-11-08):** B04.5/B05/B07 lab gates and four
full 28+4 shadow gradient iterations have passed through g1261. The user now
requests teacher on-mode and residual-head Stage A training. The reviewed
on-mode config starts 32 head-only Adam steps per iteration after PPG,
`w=0` and trunk scale zero; shadow did not train the head. First installation
at this boundary preserves the committed collection identity. Later eight-age,
held-out/blend and trunk gates remain. See [review, fixes and continuation](RUST_TEACHER_ON_STAGE_A_REVIEW.md).
Actual first on-mode training remains Pending on the lab. Earlier status lines
below describe the original implementation/qualification scope.

Date: 2026-11-06 (America/Chicago). Status: **implemented; g1250 lab qualification Pending.**
See [implementation and compatibility notes](RUST_BACKEND_B045_IMPLEMENTATION.md) and
[qualification commands](../distribution/rust-lab-g1250/README.md). Production checkpoint
and run state are unchanged; production integration remains gated by B05/B07.

This plan inserts a new step, **B04.5**, between B04 (passed) and B05 in
[RUST_AI_BACKEND_MIGRATION_PLAN.md](RUST_AI_BACKEND_MIGRATION_PLAN.md). It extends
the [teacher search phase revision](TEACHER_SEARCH_PHASE_REVISION_PLAN.md).
Everything not changed here stays exactly as written there: policy labels, π′,
the PPO teacher term and its weight calibration, the 4-age label buffer, and
held-out isolation.

## 1. Goal

Close the expert-iteration loop:

    search action → full-game return → update search value → better next search → better distillation targets

The two values in the system answer different questions:
- **The PPO critic** keeps learning the value of ordinary play: the sampled policy
  with exploration. That is what PPO's advantages are measured against.
- **A new correction head Δ** learns the gap between that value and what search
  play actually achieves.

The search reads `U_main + w · Δ`. PPO and the ordinary critic never read Δ.

## 2. Decisions

| Item | Setting |
| --- | --- |
| Tables | 32 per iteration: 28 train and 4 held-out (validation-split seeds), as today |
| Who searches | All four seats, every turn: 64 searched turns per table |
| Full searches | The 16 reserved turns (one per seat per phase). `search_v1`, 4 s plus ≤1 s post-work, K = 16, Gumbel noise on. These produce all the policy labels: about 31 per table (B04 measured 30) |
| Fast searches | The other 48 turns. `search_v1`, 0.8 s plus ≤0.5 s post-work, K = 4, Gumbel noise on. No labels |
| Move played | search_v1's recommendation, with Gumbel noise, at every turn |
| Value rows | One per searched turn: 64 per table, so 1,792 train and 256 held-out rows per iteration |
| Value target | Per-seat λ-return over the game's root values, ending at the final utility-to-go (insolvency costs included) |
| Correction head | Same inputs and shape as the utility head; final layer zero-initialised; ≈17.7k parameters |
| Search value | `U_search = U_main + w · Δ`, with w ∈ [0, 1] recorded on every search |
| PPO and ordinary critic | Unchanged. Teacher rows never enter ordinary replay, GAE, critic or PPG data |
| Training | Companion Adam, following the local-scoring pattern. Stages A (head-only, not used) → B (w ramp) → one-off head-to-head → C (trunk gradients in a guarded pass) |
| Value buffer | The last 8 iterations, bound to the reward id. Held-out rows are never trained on |
| Legacy value path | `teacher_value` and `TEACHER_AUXILIARY_VALUE` stay disabled. That path trained the shared critic; this one does not |
| Default | Off. Every new block is inert until it is switched on |

The mixed budget follows KataGo's playout-cap randomisation: a full search on 25%
of turns and fast searches on the rest. KataGo ran 600 versus 100 playouts, later
1,000 versus 200, which is close to the 1:5 time ratio used here. KataGo records
only the full-search turns for training. Here that rule applies to the policy
labels, while value rows come from every searched turn (section 6).

## 3. B04 evidence that shapes this step

Source: `results/rust-lab-qualification/20261104T022520138504Z-20261104T022431456929Z-g570-b04-845a2a56-36de5c8d`
(B04 passed on g570, `cuda:0`).

| Measurement | Value |
| --- | --- |
| Shadow tables | 2 (1 train, 1 held-out); 60 labels, of which 32 root, 27 continuation and 1 replanned |
| Completed trajectories per search | Median 46, p10 46, p90 47 |
| Effective K | 8 in 59 of 60 searches (K = 16 requested) |
| Completed halving rounds | 3; no deadline hits and no overruns |
| Timed probes | Frozen N of 46–47; search finished in 1.13–1.15 s of its 4 s; post-work 0.15–0.17 s |
| GPU-path estimate | 84.7 ms per trajectory: the slower of two probes (8 trajectories in 677.5 ms, then 8 in 176.1 ms) |

The frozen quota is set by a warm-up-inflated estimate. Actual cost was about
24–25 ms per trajectory (46 in 1.13 s). Two consequences:
- **Full searches are underpowered.** They use about 29% of their allowance, and
  K = 16 runs as K = 8.
- **Fast searches would be very short.** At the same estimate, a 0.8 s search
  would freeze N ≈ 9. That is just enough for K = 4's two halving rounds, which
  need 4 + 4 = 8 simulations.

Step 1 of section 9 therefore fixes the calibration first. After it, expect
N ≈ 160 at 4 s (K = 16 needs 64 for four rounds) and N ≈ 32 at 0.8 s.

## 4. Starting conditions

### 4.1 From IL, or continuing the current run

- **A real value exists from iteration 0.** BC trains the official value heads
  with value weight 0.5 on the heuristic demonstrations. The utility head starts
  as a zero residual on top of them. A continued run has its mature critic.
- **Teacher start.** Run 32 tables from the first iteration that has a ready
  utility receipt for the current reward. B04's null-receipt authorisation
  covers g570 only. The user also explicitly authorizes null readiness for the exact g1250 B04.5 shadow checkpoint; this does not authorize production teaching.
- **Order:** A → B → head-to-head → C. This is the LP-FT order. A new, randomly
  initialised head trained together with a pretrained trunk distorts the trunk's
  features while the head is still wrong, so train the head on frozen features
  first and unfreeze later.

### 4.2 From zero

- **Why not run teacher tables straight away.** The value and prior start
  random, and the search sees about one turn ahead. Until the PPO critic learns
  something, π′ is roughly the prior plus noise, so 32 tables buy little.
- **Shadow probe.** Run only the 4 held-out all-seat tables each iteration
  (about +11 s per iteration). Use no labels and train no value.
- **Gate G0 (the search adds information).** On the probe's 8-iteration window,
  compare at the same positions:
  - prediction error of the final utility using the search's root value
  - prediction error using the network's own value

  Take the paired squared-error difference and a game-cluster bootstrap with 200
  refits. G0 passes when the 95% lower bound is above 0. Report it by phase,
  because late-Rail positions, where search reaches terminal states, will pass
  first.
- **After G0.** Turn on the 32 tables and start stage A.
  - With no pretrained trunk to protect, the trunk scale may ramp together with
    stage B. It stays inside the guarded pass and behind the PPG KL cap.
  - Run the head-to-head before any deployment use.

## 5. Stages

| Stage | Search value | Δ training | Trunk gradient | Advance when | On failure |
| --- | --- | --- | --- | --- | --- |
| A | `U_main` (w = 0) | Head-only on detached features: 32 steps per iteration | None | G1 passes (needs a full 8-iteration window) | Stay in A |
| B | `U_main + w·Δ`; w rises 0.05 per iteration that passes G1 (20 passing iterations) | Head-only | None | w = 1 and G1 still passes | Hold w; reset w to 0 if G1's 95% upper bound is below 0 |
| Head-to-head (one-off) | w = 0 versus w = 1 | — | — | The w = 1 search finishes ahead in ≥ 50% of comparisons and its 95% interval is not entirely below 50% | Stay at B; reset w to 0 if clearly worse |
| C | `U_main + Δ` | Search-value pass with the trunk | Scale ramped 0 → 1 over 10 iterations | G2 and the PPO guard hold | Scale back to 0 (head-only) |

Typical duration from IL: A takes about 10–20 iterations and B at least 20. The
head-to-head is a single 25-minute run. C's ramp takes 10 iterations. These are
expectations, not fixed counts; the gates decide.

### Gates

- **G1: the correction helps.**
  - Data: the last 8 iterations × 4 held-out tables, which is 32 games, 128
    seat-games and 2,048 rows.
  - Per row: `d = ‖u − U_main‖² − ‖u − U_main − Δ‖²`, where u is the final
    utility-to-go (λ = 1), which is an unbiased target for "what search play
    achieves".
  - Take the mean of d with a game-cluster bootstrap (200 refits, seeded).
    G1 passes when the 95% lower bound is above 0.
- **G2: the trunk gradients help.** 20 iterations after the scale reaches 1,
  G1's mean improvement must be at least its value just before stage C.
  Otherwise revert to head-only; the trunk gradients only added risk.
- **PPO guard (stage C).** Over the 20 iterations after the scale reaches 1,
  each of these must stay within its min–max range from the 20 iterations before
  C:
  - the ordinary critic's held-out explained variance
  - PPO's categorical KL per iteration
  - the pooled probe entropy
  - the economic health metrics

  Any breach sets the scale back to 0. Check the PL rating trend separately, over
  at least 200 iterations. Over g370–g570, the slope's 95% interval was still
  ±15 points per 100 iterations.

## 6. Value targets and buffer

- **Root value for each searched turn** (full or fast), as a 4-seat absolute
  vector:

      r_t = (v_t + Σ_a R_t[a]) / (1 + N_t)

  - v_t is the root's network value under the value source in use.
  - R_t[a] is the summed returns of edge a.
  - N_t is the number of completed trajectories.
  - If N_t = 0, then r_t = v_t.
- **Final utility-to-go u_t** for each seat: terminal utility minus the capped
  insolvency costs incurred after t, exactly as `learning/reward.replay_returns`
  computes it today.
- **λ-return** for each seat, over the game's global turn order (all 64 turns):

      G_T = u_T,   G_t = −c_t + (1 − λ) · r_{t+1} + λ · G_{t+1}

  - c_t is the capped insolvency cost the seat is charged between turns t and
    t + 1; it is usually 0. With λ = 1 the recursion gives exactly u_t.
  - Choose λ from {0.5, 0.8, 0.95, 1.0} once, at the first full held-out window,
    by prediction error against the final outcome. Then fix and record it.
  - Root values from fast searches (N ≈ 9–32) enter only as bootstrap terms.
  - Once w > 0, root values already include the correction. That is the intended
    bootstrapping.
- **Each row stores:**
  - the encoded state, with unknown descriptors and neutral context, exactly as
    the search sees it
  - the relative remaining budgets and the seat mapping
  - r_t, N_t, v_t and u_t
  - generation, model and reward ids
  - the value source and w
  - the phase, the profile (full or fast) and the table split
- **Buffer.**
  - 8 ages, against 4 for labels, because outcomes are scarcer and noisier.
  - Retire any age whose reward id differs from the current one; for example,
    an insolvency step from the entropy/insolvency plan does this.
  - Apply the label buffer's per-age staleness rule.
- **Volume.**
  - Train: 1,792 rows per iteration, so 14,336 in the buffer (896 seat-games).
  - Held-out: 256 rows per iteration, so 2,048 in the window (128 seat-games).
  - Rows from one seat-game share one outcome, so gates cluster by game.

## 7. Head and training

- **Head (`search_value_head`).**
  - Inputs, the same as the utility head: context (512), relative remaining
    budgets (4) and the mean descriptor embedding (32).
  - Shape: Linear(548 → 32) → GELU → Linear(32 → 4), final layer zero, so Δ = 0
    at creation.
  - Size: about 17.7k parameters, bringing the model to about 5.946M against the
    6.5M limit.
  - Unlike the trainer-owned local-scoring heads, it is registered on the model
    and saved in checkpoints, because the search must read it.
- **Companion optimiser**, following `local_scoring_auxiliary`:
  - Adam, learning rate 1e-4, gradient clip 1.0.
  - Weight decay 1e-4 (the run's value), so Δ shrinks toward zero wherever it has
    no data.
- **Loss.** MSE over the 4-seat utility between G and `U_main.detach() + Δ`. Δ
  learns only the residual, and the main heads never receive this loss.
- **Stages A and B.**
  - 32 steps per iteration, batch 512, uniform over the buffer's train rows.
  - Inputs go through `ppg_phasic.stop_value_gradient`, so only the head trains.
  - No rollback is needed: the head cannot affect the policy or the ordinary
    critic.
- **Stage C: the search-value pass.** It runs after the ordinary PPG phase.
  - Steps: at most 2 intervals of 16 steps.
  - Trunk gradient: `local_scoring_heads.scale_gradient`, with the scale ramped
    0 → 1 over 10 iterations. Trunk steps use PPG's value optimiser and
    learning-rate schedule. The cloning term uses the current β.
  - KL: the ordinary PPG phase plus this pass, measured on the PPG probe, must
    stay within the existing 0.002 cap.
  - Acceptance per interval: the search held-out loss falls, and the ordinary
    held-out loss rises by no more than 0.013. That is about 2 SD of the
    restart's PPG interval changes (SD 0.0065).
  - A rejected interval restores the model, companion and Adam state through the
    existing PPG rollback.
- **Why a separate pass,** rather than adding the loss to the ordinary PPG phase:
  - B05 qualifies the ordinary phase as it is, and this keeps it unchanged.
  - The pass gets its own acceptance test, rather than the ordinary critic's
    coin-flip one.
  - It can be switched off cleanly.

## 8. Search integration

- **Lab path.** Leaf evaluations are served by the PyTorch GPU service,
  `rust_backend/lab_teacher_inference.predict`, which today requires exactly five
  outputs, including `utility`.
  - Add a sixth output, `search_utility = utility + w·Δ`, computed in the
    service.
  - The Rust search reads `search_utility` for Q and v.
  - Bump the protocol version. The binding records `value_source` (main or
    corrected) and w, and the reply echoes both.
- **w = 0 reproduces B04 bit for bit.** A full-profile search from the same
  root, seed and model gives the same Q, π′, recommendation and labels as B04.
- **Ordinary (non-search) inference is unchanged.** PPO and the ordinary critic
  never see Δ.
- **Deployment (Unity/ORT) is unchanged in B04.5.** The deployed search keeps the
  main value until a separate decision after the head-to-head. At that point the
  ONNX export gains the Δ output and w in its metadata, with Rust/ORT parity.

## 9. Implementation steps

1. **Recalibrate the GPU trajectory estimate.** Exclude the warm-up probe and use
   the median of at least 3 warm probes per root type. Re-run B04's timed probes
   at 4 s/K16 and 0.8 s/K4, and record N, effective K and rounds.
2. **Rust.**
   - All-seat scheduling inside each table's one-thread session.
   - The fast profile, with no label construction or verification.
   - Sub-stage decisions inside a searched turn follow search_v1's existing
     plan/continuation rules under that turn's cutoffs.
   - Root-value export (r_t, v_t, N_t) and a root state tensor for every
     searched turn.
   - Read `search_utility`, and add the new profile and receipt fields.
3. **Python.**
   - Head, companion optimiser, checkpoint and restore.
   - The sixth inference output.
   - Value buffer and targets.
   - Stage state machine and gates G0, G1 and G2.
   - The search-value pass.
   - Reports, config validation and toggles.
4. **Head-to-head tool.** w = 0 against w = 1, using 2v2 tables with seat
   rotations and the deployment profile (4 s/K16 on every turn), 256 games.
5. **Bounded lab smoke.**
   - Two all-seat shadow tables (1 train, 1 held-out) at w = 0. Measure timings,
     N, effective K, labels and value rows.
   - One authored stage-A update on frozen rows.
   - No production training.

**Exit:**
- The recalibrated estimate is recorded.
- Native and Python parity holds for root values and `search_utility`.
- At w = 0, full-profile searches from the same roots reproduce B04's searches
  and labels.
- The two-table all-seat shadow passes timing and correctness.
- A stage-A update moves only the head.
- Production teaching remains gated by B05 and B07.

## 10. Configuration

```json
"teacher_phase": {
  "contract": "teacher-phase/2.0.0",
  "mode": "off",
  "startup": "immediate",
  "tables": {"train": 28, "held_out": 4},
  "seating": "all-seat-search",
  "search": {"method": "search_v1", "k": 16, "allowance_seconds": 4.0, "post_search_seconds": 1.0,
             "turns": "four-phase-reserved", "labels": true},
  "fast_search": {"method": "search_v1", "k": 4, "allowance_seconds": 0.8, "post_search_seconds": 0.5,
                  "turns": "all-other-turns", "labels": false},
  "search_value": {
    "contract": "search-value-correction/1.0.0",
    "stage": "off",
    "head": {"hidden": 32, "inputs": "utility-head-inputs", "init": "zero-final-layer"},
    "optimizer": {"kind": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0001, "grad_clip": 1.0},
    "steps_per_iteration": 32,
    "batch_size": 512,
    "target": {"kind": "seat-lambda-return", "lambda_candidates": [0.5, 0.8, 0.95, 1.0], "lambda": null,
               "root_value": "network-plus-completed-returns"},
    "buffer": {"iterations": 8, "reward_bound": true, "staleness": "label-buffer-rule"},
    "gate": {"window_iterations": 8, "bootstrap": 200, "cluster": "held-out-game",
             "evaluation_target": "final-utility-to-go"},
    "blend": {"w": 0.0, "step": 0.05},
    "head_to_head": {"games": 256, "tables": "2v2-seat-rotations", "profile": "4s-k16-every-turn"},
    "trunk": {"scale": 0.0, "ramp_iterations": 10, "intervals": 2, "interval_steps": 16,
              "ordinary_heldout_tolerance": 0.013, "kl": "shared-ppg-cap"}
  }
}
```

- `startup` is `immediate` for a start from IL or a continued run, and
  `shadow-until-search-informative` for a start from zero (4 held-out tables,
  gate G0).
- `stage` takes the values `off`, `A`, `B`, `head_to_head` and `C`.

## 11. Validation

- All-seat seating requires both profiles. The full profile is exactly 4 s plus
  1 s with K = 16 and labels. The fast profile is 0.8 s plus 0.5 s with K = 4
  and no labels.
- Stages advance only through the state machine. A manual change may only go to
  `off` or move back a stage.
- w ∈ [0, 1] in steps of 0.05, and the scale ∈ [0, 1].
  - w > 0 requires the `search_utility` protocol.
  - The scale may be above 0 only in stage C, or in stage B for a start from
    zero.
- λ is fixed once chosen. Changing it retires the buffer through a reviewed
  transition.
- The buffer is bound to the reward id, and held-out rows are never sampled for
  training.
- A teacher-phase signature with `search_value.stage = off` and all-seat seating
  off must reproduce the B04 profile exactly.

## 12. Reports (per iteration)

- **Searches, by profile:** N (median, p10, p90), effective K, rounds, deadline
  hits, overruns and post-work.
- **Volumes:** labels; value rows by split and phase; buffer ages and rows.
- **Stage state:** stage, w, scale and λ.
- **Gates:** G1 (mean, 95% interval, by phase), plus G0 or G2 when active.
- **Δ size:** mean |Δ| and p90 |Δ|, by phase.
- **Stage C pass:** intervals accepted, KL used, and the change in ordinary
  held-out loss.
- **Cost:** lane-seconds, per-table wall time, inference evaluations, GPU queue
  wait and iteration wall time.

## 13. Tests

**Native (Rust)**
- Fast profile:
  - the 0.8 s and 0.5 s cutoffs
  - the frozen quota and effective K = 4 with two rounds
  - no labels
  - cancellation and late replies
  - continuation and replanning
- All-seat scheduling covers every action-stage turn: reserved turns get the
  full profile and all others the fast profile.
- Root-value export matches a Python recomputation from the exported edge
  statistics.
- `search_utility` is read, and at w = 0 full-profile searches from the same
  roots are identical to B04's.

**Python**
- λ-return construction: seat mapping, terminal utility with insolvency costs,
  and bootstraps.
- Buffer behaviour: retirement on reward change, staleness, held-out isolation.
- Δ = 0 at creation, and stages A and B send zero trunk gradient and cause zero
  policy KL. Check this with a gradient allowlist.
- The companion optimiser saves and restores, and resumes mid-stage.
- Gates are deterministic, and the stage machine survives interruption.
- Stage C pass: KL accounting under the shared cap, the ordinary held-out
  tolerance, and rollback.
- With `search_value` off, ordinary training is bitwise unchanged.

**Lab (bounded)**
- Recalibration, then timed probes for both profiles.
- The two-table all-seat shadow.
- One authored stage-A update.

## 14. Cost

On the teacher plan's basis: 48 lanes, a 367 s iteration and about 27 lane-seconds
per ordinary game. The restart's current median iteration is about 322 s, so the
percentages would be higher.

| Item | Policy-only plan | B04.5, budget-bound (after recalibration) | B04.5, quota-bound (current estimate) |
| --- | --- | --- | --- |
| Search per table | 64 s | 16 × 4 + 48 × 0.8 ≈ 102 s | 16 × 1.15 + 48 × 0.23 ≈ 29 s |
| Lane-seconds per table | ≈ 91 | ≈ 102–129 | ≈ 56 |
| Added time per iteration | +60 s | ≈ +86 s | ≈ +37 s |
| Search inference per iteration | 512 searches, ≈ 24k–82k evaluations | 2,048 searches, ≈ 131k evaluations (512 × 160 + 1,536 × 32) | 2,048 searches, ≈ 37k evaluations |
| ~1,300-iteration run | +22 h | ≈ +31 h | ≈ +13 h |

- **Quota-bound rows** are cheaper only because the searches are weaker: K runs
  as 8 and N ≈ 46.
- **The tables can become the critical path.** A teacher table is sequential, at
  about 130 s budget-bound. If Rust ordinary collection drops below that, keep
  scheduling teacher jobs first.
- **One-off and probe costs:**
  - The head-to-head (256 games × about 283 lane-seconds) takes about 25 minutes
    on 48 lanes.
  - The from-zero shadow probe adds about 11 s per iteration.

## 15. Changes to B05–B07

- **B05.** Add:
  - value-buffer reload and resume across commit and restart
  - stage-machine persistence (stage, w, scale, λ, gate history)
  - head-only pass fixtures (zero trunk gradient, zero policy KL)
  - stage C pass acceptance, rollback and shared-cap KL accounting
  - interruption at each stage transition
  - teacher off/shadow invariance with `search_value` off
  - an allowlist proving that the Δ loss never reaches the main value heads
- **B06.** Measure all-seat tables separately from the policy-only profile:
  - lane-seconds and per-table wall time
  - inference evaluations, latency and GPU queue
  - the search-value pass cost
- **B07.** Run the canary with at most two all-seat tables per pass at w = 0. The
  later explicit enabling decision covers the stages, and deployment use of Δ is
  a separate decision after the head-to-head.

## 16. Proposed insert for the migration plan

Add this row to the B4 table, between B04 and B05:

| Step | Implementation / current lab status | Exit scope |
| --- | --- | --- |
| B04.5 | **Implemented; g1250 lab qualification Pending.** All-seat mixed-budget teacher tables (16 × 4 s/K16 labelled + 48 × 0.8 s/K4 unlabelled); root-value export; search-value correction head served as `search_utility`; value buffer, targets and staged training (A/B/head-to-head/C) | GPU estimate recalibrated; root-value and `search_utility` parity; at w = 0, full-profile searches reproduce B04; two-table all-seat shadow; one head-only update; no production teaching |

Then add a short `#### B04.5 — Search-play value` section linking to this file.

## 17. Interactions

- **Entropy/insolvency anneal plan.** That plan keeps the teacher phase off
  during its ramp. If the teacher phase runs throughout:
  - make no stage, w or scale changes during the 100-iteration ramp
  - remember that each insolvency step retires the value buffer, so the G1
    window has to refill (8 iterations)
- **Teacher term.** Labels carry their w. The 4-age label buffer mixes labels
  made under different w, which is expected while w ramps.
- **PPG.** The ordinary phase is unchanged. Stage C shares its KL cap, so it gets
  whatever room the ordinary phase leaves; the ordinary phase's accepted KL has a
  median of about 0.0007.

## 18. Risks

- **Self-confirming bias.** A search that uses its own value can avoid positions
  it undervalues and never collect the outcomes that would correct them.
  - Mitigations: Gumbel noise on every turn; weight decay toward Δ = 0; PPO games
    keep training the main value over broader states; G1 is measured on held-out
    seeds.
- **Thin, correlated data.** The train buffer holds 896 seat-games. Keep the
  head small, zero-initialised and decayed, and gate on game-clustered intervals.
- **Trunk interference.** Stage C changes features shared by PPO and the
  ordinary critic. It is limited by the KL cap, the ordinary held-out tolerance,
  the PPO guard and G2.
- **Weak fast searches.** With N ≈ 9 at the current estimate, fast searches are
  close to a one-ply look at 4 sampled candidates. Recalibration matters more for
  them than for full searches.
- **Inference load.** There are 2,048 searches per iteration instead of 512, all
  as singleton requests. This is the main unknown; measure it in the B04.5 smoke
  and in B06.
- **Attribution.** w, the trunk scale, the teacher term and any entropy or
  insolvency schedule all affect the policy. Change one at a time, and keep the
  stage gates.

## Sources

- B04 lab evidence:
  `results/rust-lab-qualification/20261104T022520138504Z-20261104T022431456929Z-g570-b04-845a2a56-36de5c8d`
  (`gpu-calibration.json`, `timed-turns.json`, `teacher-report.json`).
- Wu, *Accelerating Self-Play Learning in Go* (KataGo), arXiv 1902.10565,
  playout-cap randomisation: https://arxiv.org/abs/1902.10565
- Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform
  Out-of-Distribution* (ICLR 2022), arXiv 2202.10054:
  https://arxiv.org/abs/2202.10054
