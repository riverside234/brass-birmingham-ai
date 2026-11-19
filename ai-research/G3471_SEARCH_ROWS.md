# g3471: expanded search rows, era-head λ targets, soft weight 6, cotton timing, self-play VP by table type

Adopted together at the committed g3471 boundary, with one continuation:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3471 --iterations 19 --share-target 0.1 --value-revision search-rows --soft-policy-weight 6 --update-cotton-timing
```

That reaches the g3490 rating. Then continue without the adoption flags:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3490 --iterations 20 --share-target 0.1
```

The continuation rebuilds the CPU teacher and collector (`ensure_cpu`; their overlay changed, `lab-native/` did not, so
the teacher job fingerprint is unchanged). Execution identity `e4afd4e2…` (predecessor `259d39df…`).

## 1. Search-teacher tables keep more states (`--value-revision search-rows`)

Code: `src/brass_ai/recovery/search_rows.py`, `lab-native-cpu/src/cpu_capture.rs`, overlay patch in `lab_cpu_overlay.py`.

Each table keeps its 64 turn-start roots and adds two row types. The CPU teacher encodes them before the decision is
applied, so their seat mapping, state, costs and remaining budget are exactly the decision's:

| kind | rows per table | what |
| --- | --- | --- |
| `turn` | 64 | turn starts, as before |
| `second` | 60 | the state before every played second main action (4 × 15 turns; Canal round 1 has one action) |
| `sampled` | ≤ 24 per era | other played decisions with two or more legal choices: a develop's second tile, a sale, a payment source, a bonus, a liquidation |

These decisions are rare. A local CPU-teacher table (fixture model) had only 15 of them, 6 Canal and 9 Rail, so the
cap seldom binds and a table grows from 64 to roughly 140 rows (about 2.2×).

- **Sampling.** The 24 decisions per era with the smallest keys of `splitmix64(seed ^ splitmix64(revision ^ 0x34715EA2…))`.
  - The seed is the job's sampling seed. The stream is the sampler's own, so no search, policy or exploration draw moves.
  - Each decision is seen once.
  - Only decisions entering the sample are encoded.
  - The finish message lists every candidate revision, and admission recomputes the sample (Python twin `sample_key`).
- **No new search.** Extra rows bootstrap from the existing turn-start search values.
- **Admission checks.** Each extra row must sit strictly between its turn's root and the next root. Its costs must be monotone, and its mapping and state shape are checked. A second-action row must belong to its turn's seat and turn key. Admission then stores the rows after the roots in the table's search-value shard. Old 64-row tables stay as they are.

## 2. Search value on all three row types (λ .95, once per turn)

- Root targets are unchanged, compared exactly with the pre-g3471 code in the tests.
- An extra row x in turn t bootstraps at the next original turn start, so its own costs from x onward count once:
  `G_x = −(spent_{t+1} − spent_x) + (1 − λ) r_{t+1} + λ G_{t+1}`.
  - The value coordinate is the full-game utility, so no reward falls between.
  - At λ = 1, `G_x = u_x`, x's realized utility-to-go (tested).
- Held-out: the whole-game rotation is unchanged. Every row of a validation table scores against `u`. The 4-game diagnostic stays on roots; `extra_validation` reports the other kinds.
- The Stage C pass still takes at most 32 steps of 512 rows. It now packs only the drawn rows, which saves about 1.1 GB of host memory per iteration.

## 3. Ownership and local VP on λ-returns of their own frozen predictions

- **Target.** The actual era-end map Z plus a correction C = G − Z, with `G_i = (1 − λ) P_next + λ G_next` and G = Z at the era's last row (its scoring horizon).
  - P are the heads' own predictions: ownership probabilities (none and four seats) and per-seat local VP / 5.
  - The owner loss is cross-entropy against the blended distribution. The VP loss is smooth-L1 against the blended VP.
- **Ordinary PPO rows.** Each seat's chain of critic rows is used, as for the utility blend, and cut where the era map changes. P are the heads' outputs frozen before PPO, which `FrozenValues` already computed.
  - They train on Z + 0.5 C, half actual and half return.
  - The corrections are staged per age as `generation-N.local-bootstrap.npy` (float16, 118 entities × 9). An age staged before adoption reads zeros, which means the actual maps.
- **Search rows (all kinds).** The table's turn clock is used. P are the heads' predictions at the turn-start roots, frozen at the start of the value phase.
  - An extra row in turn t takes root t+1's step, the same correction as its turn's root. Frames are aligned through the stable seats.
  - They train on the full return, Z + C.
- **Checks.** The ownership stop and its pooled criterion keep scoring the actual maps, and the by-era report keeps the actual-map CE. At λ = 1 the losses equal the run without the item (tested).
- **Rail-end.** The Rail-end head (weight 0 in this run) keeps its actual map. This run removes the Rail-end heads at
  g3528; a run that switches them on gets λ-returns too (ai-research/G3528_CONTINUATION.md).

## 4. Soft policy weight 8 → 6 (`--soft-policy-weight 6`)

`soft_policy.weight` = 6, with `weight_change = {generation: 3471, before: 8}`. The head, its Adam and its warm-up carry over. The committed head state, stamped under the weight-8 settings, is read once at 3471 and re-stamped at commit.

## 5. Cotton exploration timed within each era (`--update-cotton-timing`)

Applies to ordinary PPO cotton tables (`ordinary-cotton-exploration/5.0.0`) and teacher cotton tables (`teacher-cotton-exploration/9.0.0`):

| seat's action in the era | Canal (1–15) | Rail (1–16) |
| --- | --- | --- |
| no cotton exploration | 1, 15 | 1, 2, 16 |
| cotton builds only (no cotton develop, no cotton-city network) | 14 (and 15) | 15 (and 16) |

- **Action index.** A seat's action is `2·round − actions_remaining`, plus one in the Rail era. A payment or continuation decision keeps its action's index, because the engine counts actions down only when an action finishes.
- **Where the rule lives.** It is implemented identically in `step5.policy.cotton_timing`, used by the actors and the replay audit (`ordinary_cotton.TableExploration`) and by teacher labels (`teacher_cotton.validate_label`), and in its Rust twins `cpu_cotton::timing` (teacher) and `cpu_autoplay` (ordinary games).
- **Unchanged.** Seats, quotas, rates, the three build-and-develop targets per era, the one network per era and the Rail-era build level 3 are as in g2654. Cotton seats are drawn from the new contract's stream.

## 6. Self-play VP by table type

- `economy.cotton_by_selfplay.{standard_exploration,cotton_exploration}` gains `seat_games` and `vp_sum` (learner seats).
- The iteration log prints `Self-play mean final VP per learner seat: all … (N tables), standard … (n), cotton … (m)`.
- `progress_vp.svg` draws the standard and cotton tables as two extra lines, wherever 10-iteration groups record the split, which is from g3471 on.

## Cost (estimates; nothing here was measured on the lab)

| item | per iteration |
| --- | --- |
| teacher capture: about 80 extra rows (and encodings) and about 2.4 MB of JSON per table, parsed once at finish | about 1 s of trainer GIL, mostly during ordinary games |
| admission: shards about 2.2× larger | about +4 s, mostly in the ordinary tail |
| search-value buffer | rows about 41k → 92k; RAM about 0.5 → 1.1 GB; disk about 2.2× |
| search-value fit | same 32 steps; targets about +1 s; held-out rows 2,048 → about 4,600 |
| era heads, search rows | pool 10,240 → about 23,000 rows (about 430 per PPG step instead of 190; search share of head presentations about 4.7% → 9.5%); about +1.5 s; pooled-criterion search share about 6% → 13% |
| era heads, ordinary rows | corrections about 3 s (1.5 s for 60k rows on the laptop); a 267 MB sidecar per age (about 1 s to write and hash); +1.6 GB RAM held from PPO's end to PPG staging |
| soft weight, cotton timing, VP report | none |

The total is about +5–9 s on a 263 s iteration (2–3%). No extra search, tables or held-out games are needed.

## Checks after the first iteration (g3472)

- `search_value.extra_rows` = `{second: 2400, sampled: several hundred}` (40 tables; 15 in the local fixture table).
- The `search_value.rows` counts include them.
- The log shows `search_rows=10240+2160+…` (only the new train tables carry extras at first; held-out tables reach the pool as they rotate out).
- `value_phase.buffer.local_bootstrap` shows about 125k rows, with `mean_owner_l1` > 0.
- `value_phase.search_auxiliary.lambda_targets` shows `mean_owner_l1` > 0.
- `teacher_phase.collection.cotton_exploration.contract` is `teacher-cotton-exploration/9.0.0`, with more `blocked_window` decisions.
- The new self-play VP line appears.

## Tests

- `tests/Recovery/test_search_rows_g3471.py` (17 tests).
  - Covered: root parity, λ = 1 outcomes, the bootstrap formula, admission against fit targets, the sample stream twin, refused tampering, mixed 64-row and expanded rotation, buffer keys, ordinary and search corrections (λ = 1 zero, frame alignment), the loss, the soft weight, the recipe and job blocks, and the overlay.
  - End to end through `ppg_g210.fit`.
- `tests/Recovery/test_cotton_timing_g3471.py`: the rule table, PPO targets under each contract, teacher labels and settings, the recipe, the transition, and the VP split plus chart.
- Rust: `cpu_capture` (3 tests), `cpu_cotton` timing, and `cpu_autoplay` timed targets. All 43 `cpu_` tests pass.
- Native (`tests/RustBackendLab/test_cpu_inference.py`, the built CPU teacher, one fixture-model table): the extra rows pass
  the Python admission, the sample equals the one recomputed from the stream, λ = 1 recovers `u`, and every cotton
  label follows the timed rule (60 second-action rows, 15 sampled).
