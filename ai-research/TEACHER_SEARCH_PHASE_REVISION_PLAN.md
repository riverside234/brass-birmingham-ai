# Teacher search phase revision plan

Date: 2026-10-29 (America/Chicago). Status: **T1/B02.5 and T2/B04 implemented; lab qualification Pending.** See the [T1 report](RUST_BACKEND_B025_REPORT.md), [B04 native export/collection report](RUST_BACKEND_B04_REPORT.md) and [B04 transfer/qualification commands](../distribution/rust-lab-b04/README.md). The phase defaults to off; the explicit B04 collector supplies the bounded lab shadow route. B05/B07 still gate production update/adoption. No existing run state was changed.

This plan replaces the legacy Masked-ISMCTS teacher with an policy-only search_v1 teacher phase. The phase is meant to be switched on for roughly the last 100–500 iterations. It fits the Rust migration's B04 step (search_v1 teachers) and B05 step (PPO/PPG qualification), plus Python trainer changes that can be made now.


## 1. Decisions

| Item | Setting |
| --- | --- |
| Switch | `teacher_phase.mode`: `off` (default), `shadow` or `on`. It changes only at a committed iteration boundary. |
| Tables | 32 per iteration: 28 training tables and 4 held-out tables (validation-split seeds) |
| Searched turns | 16 per table. Each seat searches one reserved turn per phase, using the existing four-phase schedule. |
| Labels | Both actions of each searched turn: about 31 per table, or about 870 training labels per iteration |
| Search | Rust `search_v1` Gumbel IS-MCTS. K = 16, c1 = 50, c2 = 0.1. Gumbel noise on, prior temperature 1. One session with one thread, and one 4 s whole-turn allowance. |
| Target | π′ = softmax(l + σ(completed Q)) over every legal action, built from all completed trajectories |
| Quality gates | None. The timeout, coverage, fallback, two-visit and diversity gates are removed. Integrity checks stay. |
| Value | Policy-only. `teacher_value.enabled = false`, the `TEACHER_VALUE` default is off, and no teacher row enters any critic data. |
| Buffer | The last 4 iterations, with per-label age weights 4 : 3 : 2 : 1 and a staleness check per age |
| Training | A KL(π′ ‖ π_θ) term inside every PPO step, with 64 label rows per step, under the existing KL guard and clipping |
| Weight | β is calibrated so the teacher gradient is 0.25 × the PPO actor gradient on the shared encoders. |

The deployed search uses Li and Wellman's constants: c1 = 50 and c2 = 0.1, with raw actor-utility Q and no Q normalisation. The Go/chess setting (c_scale = 1 with Q normalised to [0, 1]) does not apply. π′ must come from the search's own function, so the two cannot diverge.

## 2. Today versus this revision

| Area | Today | This revision |
| --- | --- | --- |
| Search | Python Masked ISMCTS (`search.contract` `masked-ismcts/1.2.0`): 8 workers, 9/12/12/12 s per phase | Rust `search_v1`: 1 session × 1 thread, 4 s, K = 16 |
| Timing | `collect_due()` runs after PPO and PPG (`league.py`): 2 collections × 5 games when due | 32 tables run in the same collection as the 320 ordinary games, every iteration while enabled |
| Target | `ismcts-root-visits/1.2.0`: a visit/prior mixture over covered actions, for the turn's first decision only | π′ over all legal actions, for action one and action two |
| Update | `distill_update()`: 8 steps × 16 rows, weight 0.25, outside the PPO KL guard, followed by utility recalibration | A term inside each PPO step. No separate steps and no recalibration. |
| Data | The current collection only | A 4-iteration buffer, age-weighted |
| Auxiliary value | `teacher_value.enabled = true` in the run config. It is dormant because `teacher_enabled = false`. | Disabled |
| Readiness | `calibration_ready()`, which was false in 67 of 128 iterations in g1190–1317 | The last ready receipt for the current reward (section 8) |

## 3. Iteration flow when enabled

For iteration t with incoming model θ_t:

1. **Readiness.** Apply the search-readiness rule (section 8). If it fails, run no teacher tables this iteration and keep using the existing buffer.
2. **Plan.** Queue the 32 teacher jobs ahead of the 320 ordinary jobs. Both use the frozen θ_t.
3. **Collect.** The teacher tables publish a training label shard L_t and a held-out shard H_t. Teacher records stay out of the ordinary game list, the streaming PPO preparation and `games_done`.
4. **Admit.** Validate the labels (section 8), build the buffer {L_t, …, L_t−3} and run the staleness check.
5. **Measure (pre).** On H_t, measure KL(π′ ‖ π_θt) and top-1 agreement.
6. **PPO.** Every step adds the teacher term. In shadow mode the term is measured only.
7. **Unchanged.** The LR controller, the PPG value phase and the final probes run as today.
8. **Measure (post).** Repeat the H_t measurements with the final weights.
9. **Commit.** The checkpoint and state store the buffer receipts and β. After the commit, delete owned shards older than t − 3.

## 4. Configuration and switching

### 4.1 New block

Add a separately versioned block. Leave the legacy teacher fields unchanged so historical configs still validate. The new path does not read them. The legacy fields are:
- `teacher_enabled`, `search`
- `search_games`, `search_every_games`, `search_collections_per_iteration`
- `distill_steps`, `distill_weight`, `distillation`
- `teacher_work_budget`, `execution.teacher_search_workers_per_turn`

```json
"teacher_phase": {
  "contract": "teacher-phase/1.0.0",
  "mode": "off",
  "tables": {"train": 28, "held_out": 4},
  "schedule": "four-phase-teacher-turns/2.0.0",
  "unsearched": {"temperature": 1.0, "epsilon": 0.0, "opening_temperature": null, "branching": false},
  "search": {
    "method": "search_v1",
    "algorithm": "brass-gumbel-is-mcts/count-v1",
    "session": "brass-gumbel-is-mcts/session-v1",
    "v205_enabled": false,
    "allowance_seconds": 4.0,
    "sessions": 1,
    "threads_per_session": 1,
    "pending_paths": 1,
    "k": 16,
    "c1": 50,
    "c2": 0.1,
    "gumbel_noise": true,
    "prior_temperature": 1.0,
    "watchdog_seconds": 30.0,
    "reward": "<identical to the PPO reward specification>"
  },
  "labels": {"source": "search-v1-improved-policy/1.0.0",
             "decisions": "action-stage-both-actions",
             "evidence": "completed-trajectories"},
  "buffer": {"iterations": 4, "age_weights": [4, 3, 2, 1], "max_staleness_ratio": 1.0},
  "term": {"rows_per_step": 64, "share_target": 0.25, "shadow_iterations": 2, "beta_bounds": [0.01, 10.0]}
}
```

`pending_paths: 1` matches the installed local profile, which uses one pending path per session. Changing it creates a different scheduling variant.

### 4.2 Validation (`config.validate_config`)

- If `teacher_phase` is present, `teacher_enabled` must be False. This stops the legacy Masked teacher from ever running alongside the new phase.
- `mode` must be one of `off`, `shadow` or `on`.
- `search` must pass `rust_backend.lab_contract.admit_search_request`: an explicit `search_v1`, with no focus, composite or reuse overrides.
- `search.reward` must equal `reward_spec(config)`.
- If `mode` is not `off`, `teacher_value.enabled` must be False.
- The auto IL-KL rule at `config.py:712` (`teacher_enabled is False`) stays. Add an explicit allowance for `teacher_phase`. The teacher term moves the policy inside PPO steps, where the same per-step probe, LR controller and IL-KL probe measure it. This differs from the legacy teacher, which updated outside the guard.
- `age_weights` must have `iterations` entries that are positive and non-increasing.
- `rows_per_step` must be at most `effective_batch_size`, and `share_target` must satisfy 0 < share_target ≤ 1.

### 4.3 Modes

| Mode | Teacher tables and labels | PPO teacher term |
| --- | --- | --- |
| `off` | None. Existing shards age out. | None |
| `shadow` | Collected, admitted and buffered | Loss and gradients computed at β = 1 for measurement only. Nothing is added to the update. |
| `on` | Collected, admitted and buffered | Added with the calibrated β |

### 4.4 Policy-only

- Change `TEACHER_VALUE['enabled']` (`config.py:85`) to `False`, so fresh recipes start disabled. Keep its contract and fields so existing receipts still validate. `teacher_value_config(None)` already means disabled.
- In the adoption transition, set the run's `teacher_value.enabled` to `false`. The existing `teacher_value_changes` route covers this block.
- The new path never builds `TeacherValue` and never calls `recalibrate_teacher_utility`. It attaches no value, utility, descriptor or local-scoring graph to the teacher loss.
- Teacher-table rows never enter:
  - PPO actor or critic data
  - PPG buffers
  - local-scoring targets
  - ratings, the quality pool or curriculum counts

### 4.5 Enabling and disabling

- **One-time adoption.** At the next planned revision, add the block with `mode = off` and disable `teacher_value`, as one reviewed transition. With `off` it changes nothing.
- **Toggle route.** Add a `teacher_phase_changes` flag to source-review transitions. Handle it the same way as `teacher_value_changes` in `resume.check_signature` (lines 70–79 and 197–199), but let only `teacher_phase.mode` and `teacher_phase.term.share_target` differ. Such a toggle must also keep the utility receipt (section 8).
  - Register the flag as a self-transition for the adopted implementation, and carry it into later revisions.
  - Any other teacher-setting change needs a normal review.
- **Helper.** Add `scripts/configure-teacher-phase.py --run RUN --expected-generation G --mode shadow|on|off [--share-target X] --output CONFIG`. It follows the `configure-training-g1324.py` pattern: read the locked committed state, check the generation and checkpoint hash, write only the new config, and be safe to repeat.
- **Restart.** Resume with `--resume-reviewed-source --expected-generation G` as usual.

| Change | Effect |
| --- | --- |
| off → shadow or on | The buffer starts from any admissible shards still in the window (usually none) and fills over 4 iterations. |
| shadow → on | β = share_target / (shadow median share at β = 1) |
| on → off | No teacher jobs and no term. Shards expire normally, and β stays in state. |
| off → on within 4 iterations | Shards still in the window are reused. |
| share_target change | β is scaled by new / old, since the share is proportional to β. |

## 5. Teacher tables

- **Count and split.** 28 tables use train-split seeds and 4 use validation-split seeds (`seeds_for(..., partition='validation')`). Held-out labels are never trained on. To train on all 32, add the 4 held-out tables on top (36 tables, about 12% more teacher cost) rather than dropping them.
- **Seeds.** Use a separate `teacher_seed_cursor` in state, so enabling or disabling the phase never shifts the ordinary setup seeds. The legacy path consumed the ordinary cursor (`seeds_for(cursor, 1)` in `teacher_collection.py`).
- **Seats.** All four seats are the current learner θ_t. `search_v1` evaluates every actor with the root model, so the legacy rollout-opponent list is not part of the profile.
- **Reserved turns.** Keep `reserved_rounds()`: one turn per phase per seat, from independent streams. Bump the schedule contract to `four-phase-teacher-turns/2.0.0`, because it now labels both actions with `search_v1`.
- **Unsearched decisions.** Use the raw learner policy at T = 1, with ε = 0, no opening temperature and no branching.
  - Teacher games never feed PPO, so they do not need the ordinary exploration settings.
  - Deployment-like play gives deployment-like label positions.
  - The legacy teacher base used ε = 0.1.
- **Move played at searched turns.** search_v1's normal recommendation (with Gumbel noise), plan support and fallback rules, unchanged.
- **Scheduling.** Use the same frozen job planner and collection barrier as ordinary games. Put teacher jobs first: each takes about 90 lane-seconds, against about 27 for an ordinary game.

## 6. Search and label production (B04)

### 6.1 Profile

- Use exactly the `teacher_phase.search` block, admitted only through the B04 dispatch boundary (`admit_search_request`).
- Record per search: method, algorithm and session versions, source fingerprint, model and export identity, reward id, and requested and effective K.
- `session-v1` pins the finalisation reserve: min(1.2 s, 10% of the allowance), which is 0.4 s at 4 s.
- The timed quota N is frozen from about 3.6 s of work time and a trajectory-cost estimate. That estimate must be calibrated for the lab GPU path (B04 item 6); the local CPU estimate does not apply.
- Record N, completed trajectories and completed halving rounds for every search. For reference, the local 4 s release completed about 490 trajectories per session (L14: 24,550 trajectories over 25 searched cases with two sessions).

### 6.2 Decisions that get labels

| Label kind | Decision | Statistics used |
| --- | --- | --- |
| `root` | Action one's action-stage decision (the tournament root) | The root edges |
| `continuation` | Action two's action-stage decision, when it follows the validated plan | The action-two node of the action-one tree, on the played path |
| `replanned_root` | Action two, when search_v1 discards the continuation and replans | The replanned root |

- search_v1 does not reuse trees after action one (L11). B04 must therefore snapshot the action-two node's edge statistics when the action-one search finishes, and publish them with the plan. This is a new export requirement. Without it, labels fall to about 16 per table.
- Sub-stage decisions are not labelled in this revision: payment, continuation, bonus and debt. Forced (single-choice) decisions get no label.
- Canal round 1 has one action. A canal_early turn therefore yields 1.75 labels on average and other phases yield 2, giving about 31 per table.

### 6.3 Target

For a labelled node with acting seat i, legal logits l, prior p = softmax(l), cached value v, completed edge counts n, summed absolute-seat returns R and N = Σ n:

```text
q[a]   = R[a][i] / n[a]                                            if n[a] > 0
v_mix  = (v[i] + N · Σ_visited p[b]·q[b] / Σ_visited p[b]) / (1 + N)   (v_mix = v[i] if N = 0)
q[a]   = v_mix                                                     if n[a] = 0
σ(q)   = c2 · (c1 + max_b n[b]) · q,     c1 = 50, c2 = 0.1
π′     = softmax(l + σ(q))     over every legal action
```

- These are the same equations as the [reference](GUMBEL_IS_MCTS_REFERENCE.md). The visited-prior average is not visit-weighted. There is no Q normalisation and no prior floor. π′ does not include the Gumbel draws.
- **Evidence.** Use every trajectory completed through the node when the search stops (the final completed-work watermark).
  - The recommendation still uses search_v1's last completed-round snapshot, so labels are decoupled from move selection.
  - Labels need no completed round.
- A node with no completed edge visits gives no label, counted as `no_evidence`. Its π′ would equal the prior exactly.
- Compute π′ in Rust with the search's own function, in double precision, and export it. Python recomputes it as a check.

### 6.4 Exported fields per label

| Group | Fields |
| --- | --- |
| Identity | Label source, method, algorithm and session versions, source fingerprint, model id, export hash, reward id, job and game id, seat, turn key, revision, phase, kind |
| Decision | Ordered legal ids, prior logits l (float32), node value v (actor component) |
| Evidence | Counts n over all legal actions, mean returns q for visited actions, v_mix, c1, c2, max n, π′ (float32) |
| Search | Requested and effective K, frozen N, completed trajectories (search and node), completed rounds, deadline hit, overrun seconds, played action, whether the played action had plan support or was a fallback, Gumbel draw id |

### 6.5 Deadlines and failures

- Reaching the work deadline is the normal end of a timed search. The completed evidence is the label.
- Finishing after the 4 s allowance (late finalisation) does not invalidate the label. Record the overrun.
- A 30 s watchdog cancels a stuck search. It gives no label, and bounded completion plays the move.
- These are integrity failures and give no label:
  - cancellation (game replaced, shutdown or restart)
  - a stale, misrouted, wrong-model or wrong-epoch reply
  - a non-finite output

## 7. Gates

Quality gates removed for labels:

| Location | Today | New |
| --- | --- | --- |
| `teacher_schedule.py:204–207` | Clock already past the work deadline → `timeouts`, turn disabled | No search ran, so there is no evidence. Counted as unsearched, not as a failure. |
| `teacher_schedule.py:221, 251, 253, 263, 267` | Deadline or finalisation overrun → `timeouts`, label dropped | Label kept, overrun recorded |
| `teacher_schedule.py:252, 255` | Fewer attempts than allotted, or confidence ≤ 0 → `insufficient_coverage` | Removed |
| `teacher_schedule.py:260–262` | Played move is not the visit leader or lacks plan support → `fallbacks` | Removed, because π′ does not depend on the played move. The fallback is kept as provenance. |
| `updates.py:868` | ≥ 2 covered actions and confidence > 0 | Only "≥ 1 completed visit at the node" |
| `updates.py:892–894` | < 16 labels, < 2 games or < 2 per phase → no update | Removed. The term runs whenever the buffer has rows. |
| `updates.py:899–901` | Labels must come from the current model | Labels must come from a lineage model in the buffer window |
| `learning/replay.py:43` | ≥ 2 visited root actions | Not applied to the new source |
| Migration plan B04 step 2, "completed-only labels" | Completed rounds only | Completed trajectories |

The fallback gate was not on the requested list. It still has to go: a search that stopped early often lacks complete-plan support, so leaving the gate in would drop most of the timed-out searches being kept.

The last run with teachers enabled was `selfplay-joint`. At iteration 59, these gates discarded 20 of 127 searched turns:
- 7 timeouts
- 12 `insufficient_coverage`
- 1 fallback

canal_late lost 9 of 32. The discards fall on slow, uncertain positions, which is where search adds the most.

Integrity checks kept: the failures in 6.5, a legal-set or observer mismatch, reward, model and source identity, a π′ recomputation mismatch, and a logits-parity failure (section 8).

## 8. Admission and search readiness (Python)

Add a new module `recovery/teacher_labels.py` with `validate_improved_policy(label, choices)`. It checks:

- The source is `search-v1-improved-policy/1.0.0`, the method is `search_v1` and V205 is off.
- The legal ids equal the replay row's choice ids, in order, and the observer equals the row's seat.
- All arrays are finite. π′ ≥ 0 and sums to 1 within 1e-6. Counts are integers ≥ 0, and at least one edge was visited.
- π′ recomputed in float64 from (l, n, q, v, c1, c2) matches the export, with max abs difference ≤ 1e-5.
- The reward id equals the current reward. The model id equals the collection model recorded for that generation in `state['history']`. The source fingerprint is admitted.
- **Logits parity at admission.** Run θ_t on the new label rows and compare the result with the exported l at B01's tolerances (atol 2e-5, rtol 1e-5). This also proves that the Rust and Python encoders agree on teacher rows.

Search readiness replaces the gate at `teacher_collection.py:101–109`:

- Track the latest non-null calibration receipt in state. Run teacher tables for θ_t when that receipt has the current reward id and `ready = true`.
- After a reward change, `resume.py:276` writes a not-ready receipt, so the teacher waits for the first ready PPG fit under the new reward.
- A receipt with `ready = false` defers teacher tables for that iteration.
- **Why.** PPG writes a null receipt only when it leaves the critic unfitted (phase skipped, no holdouts, or zero retained steps).
  - In g1190–1317, all 67 null receipts came from zero-step PPG phases.
  - Through those iterations, DCPG preservation held the four-seat utility within a median 0.0043 RMS per PPO update (max 0.0065).
  - The strict `calibration_ready()` check would skip about half of all iterations.
- Leave `model.utility_calibration` and `calibration_ready()` unchanged. The deployment handoff (`personality/handoff.py:61`) still uses them.
- **Exempt mode toggles from the receipt reset.** `resume.py:270–279` replaces the receipt with `ready = false` whenever the checkpoint's signature differs from the new one. Without an exemption, every switch to `shadow` or `on` would delay the first teacher tables until a PPG fit is accepted, which happens in about half of all iterations. Skip that reset when the only signature difference is `teacher_phase.mode` or `teacher_phase.term.share_target`: the weights, reward and critic are unchanged.

## 9. Label buffer

Add a new module `recovery/teacher_buffer.py`, following the ownership rules in `value_buffer.py`.

- **Storage.** Write one sealed shard per generation and split, `teacher-labels/generation-{g:06d}-{train|heldout}.jsonl.gz`, plus a receipt (hash, rows, split, model id, reward id, profile digest). Full teacher-game replays follow the existing replay retention and are never admitted anywhere else.
- **Window.** PPO at iteration t reads the train shards from t, t−1, t−2 and t−3 (ages 0–3).
- **Age weights.** Each label gets weight 4, 3, 2 or 1 by age, renormalised over the ages present. Each PPO step draws 64 rows without replacement, with probability proportional to weight. Draws are independent across steps.
- **Sampling RNG.** Use a dedicated generator seeded from (run seed, iteration, step). It must not consume the global torch RNG, so PPO's minibatch order is identical whether the phase is on, off or in shadow.
- **Staleness check.** At the start of each PPO, run one forward pass over the buffer. For each age a, compute:
  - I_a = mean KL(π′ ‖ softmax(l)): the search's correction, from stored values.
  - D_a = mean KL(softmax(l) ‖ π_θ): how far the policy has moved since collection.
  - If D_a > max_staleness_ratio · I_a, skip age a this iteration and report it.
  - At the current drift (final categorical KL ≈ 0.008 per iteration), this is not expected to trigger.
- **Commit.**
  - Publish shards and receipts before the checkpoint/state commit; state lists the window receipts.
  - Delete owned shards older than t − 3 only after the commit, and never delete unknown files.
- **Invalidation.** Empty the window on any change to:
  - the reward
  - the feature or action schema
  - the architecture
  - the search profile (method, K, c1, c2, allowance, sessions, pending paths)
  - the label source

  Entropy, exploration, LR and PPG changes do not invalidate labels.
- **Resume.** Rebuild the window from receipts. A missing or changed shard is a hard error, as for other sealed data. Timed searches are not bit-reproducible, so a recollected iteration produces new labels; fixed-data tests use saved shards.

Expected volumes at about 870 training labels per iteration and 50 PPO steps:

| Quantity | Value |
| --- | --- |
| Buffer size | About 3,500 rows |
| Teacher presentations per iteration | 50 × 64 = 3,200 |
| Share of draws by age 0 / 1 / 2 / 3 | 40 / 30 / 20 / 10% |
| Presentations per label over its life | About 1.5 + 1.1 + 0.7 + 0.4 = 3.7 |
| Iterations for 100k training labels | About 115 |

## 10. Teacher term in the PPO step

The term goes in `_ppo_nonforced()` in `updates.py`, inside the step loop. Place it after the `gradient_terms.finish(...)` block (lines 617–618), beside the rehearsal block (lines 619–625) and before `clip_grad_norm_` (line 628). Pass the buffer through `ppo_update()` → `ppo_nonforced()` the same way as `rehearsal`.

Each step:

1. Draw 64 buffer rows (one microbatch).
2. Run a policy forward pass with unknown seat descriptors (as in search) to get raw T1 logits.
3. Take the log-softmax over each row's legal candidates, using the same masking as PPO's raw-T1 probabilities, with no exploration or temperature mixing. Do not reuse the legacy loss, which masked to covered actions only (`masked_fill(~cov, …)` in `supervised_epoch`).
4. Compute `loss_T = β · mean_rows Σ_a π′(a) · (log π′(a) − log π_θ(a))`.
5. Take `torch.autograd.grad` over the actor parameters and measure the norm on the shared state/entity encoders. Add the gradient into `.grad` (on), or discard it (shadow).
6. Clipping, the Adam step and the per-step KL probe are unchanged, and now include the teacher term.

How the term interacts with the existing update:

- **KL guard and LR controller.** The 0.01 per-step categorical-KL stop, the behaviour guard, the LR controller and the IL-KL probe all see the combined movement. Do not exempt the teacher term; measuring it there is the reason for moving it inside PPO. Headroom is modest: in g1190–1317 the final categorical KL had a median of 0.0076 and a p90 of 0.0095, with 1 KL stop in 128 iterations.
- **Clipping.** Nearly every PPO step in g1190–1317 (99.98%) was clipped at 1.0. The teacher gradient enters before clipping. At share 0.25, with roughly orthogonal directions, it takes about 24% of the clipped step's norm and shortens PPO's own component by about 3%.
- **Critic.** DCPG preservation in the same step counteracts trunk movement caused by the teacher term. No separate recalibration is needed.
- **Optimizer.** The term uses PPO's AdamW and adds no extra steps.

## 11. Teacher weight

- **Share.** Define r = ‖g_T‖ / ‖g_A‖, measured before clipping on the shared state/entity encoders. This is the `TrunkGradients` parameter set, so r is comparable with the DCPG actor and preservation norms. g_A is the PPO actor gradient that `TrunkGradients.add_actor` already accumulates (without entropy). `TrunkGradients` runs only while DCPG preservation is active, as in the current recipe; without DCPG, measure g_A with the same `autograd.grad` call.
- **Calibration.** Shadow mode (2 iterations) measures r at β = 1 on every step. On switching to `on`, set β = 0.25 / median r, bounded to [0.01, 10]. β is stored in state and then held fixed.
- **No shadow receipt.** Start at β = 1 for one iteration, then set β from that iteration's measurement.
- **Monitoring.** Report the median r and the cosine between g_T and g_A every iteration. If the median r stays outside [0.125, 0.5] for 3 iterations, warn, and recalibrate with one shadow iteration if needed. A share that falls as labels are absorbed is expected.
- **Adjusting after 10 iterations.**
  - Raise share_target to 0.4 if held-out KL is not falling faster than it did in shadow mode and PPO health is normal.
  - Lower it to 0.15 if KL stops increase or action-group entropy drops (section 14).

## 12. Reports

Add a compact `teacher_phase` section to each iteration report and to `history.py`:

- **Mode and weight.** Mode, β, share (median, p10, p90) and encoder cosine; the shadow r at β = 1.
- **Counts.**
  - Tables (train and held-out).
  - Reserved, reached and searched turns by phase.
  - Labels by kind and phase, `no_evidence` and forced decisions.
- **Search.** Completed trajectories per search (median, p10, p90), effective K, completed rounds, deadline-hit fraction, overrun seconds (median and max), fallback-played fraction and cancellations.
- **Improvement and staleness.** I by phase (age 0), D_a / I_a by age, and any skipped ages.
- **Held-out.** KL(π′ ‖ π_θ) before and after the iteration, the change, and top-1 agreement before and after, all by phase.
- **Buffer.** Rows, presentations and distinct rows used by age and phase; mean teacher KL at the first and last PPO step.
- **Side effects.** PPO steps, KL stop, final categorical KL, raw-T1 entropy for the four groups, and IL-KL health.
- **Cost.**
  - Teacher lane-seconds and the change in collection wall time.
  - Admission/encoding seconds and teacher-term seconds.
  - Teacher inference requests, mean batch size and queue wait.

The `league.py` console line should print labels, β, share and the held-out KL change, replacing the legacy `teacher rows` and `steps`.

## 13. Tests

Python tests, runnable before B04 with fixture labels:

1. **Config.** Defaults are off; the section 4.2 rules hold; `teacher_value` is forced off; `teacher_phase` excludes `teacher_enabled`; the toggle route permits only `mode` and `share_target`.
2. **π′.** It matches the Gumbel reference equations and golden trace (`tests/GumbelISMCTS/`). Unvisited actions keep the prior's relative logits, and zero evidence gives the prior.
3. **Gates.** Searches with a partial round, an overrun, a short quota or a fallback-played move keep their labels. Zero evidence, cancellation and stale replies give none.
4. **Validator.** Tampered π′, wrong legal order, observer, reward or model, and non-finite values are all rejected.
5. **Buffer.**
   - Sampling frequencies match 4 : 3 : 2 : 1, and missing ages renormalise.
   - The staleness skip works.
   - Deletion happens only after commit, and a reward change resets the window.
   - Resume reproduces the draws.
6. **PPO.**
   - Shadow mode leaves weights, Adam and RNG bit-identical to a run without the term (on CPU).
   - In `on` mode the term gives no gradient to the value, utility, descriptor or local-scoring heads.
   - The KL guard includes the term.
   - A restart reproduces weights, Adam and RNG, as in the g1324 smoke.
7. **Readiness.** The g1190–1317 pattern (null receipts after zero-step PPG) is admitted. A reward change requires a fresh ready receipt, and `ready = false` defers. A mode-only or share-only toggle keeps the receipt, while any other reviewed change still resets it.
8. **Seeds.** Toggling the phase leaves the ordinary job seeds unchanged.

Rust / B04 tests:

9. **Export.** Root, continuation and replanned labels export, and Python's recomputation and logits-parity checks pass.
10. **Interrupted rounds.** With fake clocks, labels use completed trajectories, and the recommendation matches today's.
11. **Cancellation.** Cancel during GPU waits and between actions. Late replies are rejected and give no label.
12. **Lab smoke.** Run 2 teacher tables in shadow mode and report N, effective K, deadline hits, labels and timings.

## 14. Rollout near the end of training

1. **Adopt.** Add the block with `mode = off` and disable `teacher_value` at the next planned revision.
2. **Shadow.** Run 2 shadow iterations about 120 iterations before the planned end, which yields about 100k training labels by the end. Check:
   - label yield: at least 95% of reached reserved turns give a root label
   - N and effective K, and the deadline-hit rate
   - improvement I by phase
   - r at β = 1
   - cost
3. **On.** Use the first 10 iterations as acceptance:
   - Held-out KL falls faster per iteration than it did in shadow, when PPO acted alone.
   - At most 2 KL stops (baseline: 1 in 128), and median final categorical KL ≤ 0.010 (baseline 0.0076).
   - Raw-T1 action-group entropy stays within 10% of the shadow average, and IL-KL health is normal.
   - Iteration time rises by no more than 25%.
4. **Keep it on to the end.** A reward change resets the buffer, so expect about 3 iterations of ramp.
5. **Final check.** Evaluate under deployment search (K16, two sessions, at 4 s and 10 s), comparing the final checkpoint with the checkpoint at enabling time.
   - The clean causal test is a short off/on branch from the same committed generation (for example, 20 iterations each).
   - Choose a final checkpoint that has a ready calibration receipt, because the personality handoff requires one. In g1190–1317, 52% of checkpoints had none.

## 15. Cost estimate

These estimates come from g1190–1317: a median iteration of 367 s, a collection phase of 211 s, today's 48 lanes, and about 27 lane-seconds per ordinary game including branch continuations.

| Item | Estimate |
| --- | --- |
| Search per table | 16 turns × 4 s = 64 lane-seconds |
| Unsearched play per table | About 27 lane-seconds |
| 32 tables | About 2,900 lane-seconds, about 29% more than the roughly 10,100 ordinary lane-seconds |
| Iteration wall time | About +60 s (about +16%), if lanes stay the bottleneck and inference keeps up |
| Teacher term in PPO | About +3% of PPO presentations |
| Teacher inference | 512 searches × N, about 0.25–0.9 M evaluations per iteration, against about 64k ordinary. N is unmeasured on the lab GPU path; local CPU sessions reach about 490 at 4 s. |
| 120 iterations | About +2 h in total |

Inference load is the main unknown: about 20 concurrent single-path sessions, each issuing singleton requests. Measure it in the B04 smoke and in B06. Faster Rust ordinary collection does not shrink the 64 s of search per table, so the relative cost rises as B02 speeds up collection.

## 16. Implementation order

| Step | Where | Content |
| --- | --- | --- |
| T1 | Python, now | Config block and validation; `teacher_value` off; toggle route and helper; readiness state; label validator; buffer; PPO term; reports; tests 1–8 with fixture labels |
| T2 | B04 | search_v1 teacher job type and routing; root, continuation and replanned label export; deadline and gate semantics; GPU trajectory-cost calibration; tests 9–12 |
| T3 | B05 | Fixed-data PPO/PPG with the teacher term; resume with a partly filled buffer; mode toggles; rollback |
| T4 | B07 canary | Two shadow iterations on an isolated copy before production adoption |

## 17. Risks and open items

- **Inference throughput.** If the GPU service saturates, N per search falls and the labels weaken. Mitigations, in order:
  - stagger teacher starts
  - give teacher requests a dedicated inference lane
  - set pending_paths above 1, which is a separately identified scheduling variant
- **PPO trust region.** The teacher term may cause more KL stops and fewer PPO steps. Watch this in the first 10 iterations.
- **Sharper targets.** π′ is sharper than the prior, so watch action-group entropy.
- **Correlation.** About 31 labels per table come from one game, which is why the held-out split is by table.
- **Weaker training search.** One session at 4 s gives a smaller correction than deployment's two pooled sessions at 8–12 s. It points the same way, because σ grows with max n.
- **Temperature option.** Labels use the T = 1 prior. The game's T = 1.5 option only flattens the prior at play time.
- **New B04 requirement.** The continuation-node export is new. Without it, labels drop to about 16 per table (roots only).
