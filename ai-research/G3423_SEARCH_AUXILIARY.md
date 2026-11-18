# g3423 search auxiliary: search-teacher tables train the ownership and local-VP heads

- **Date:** 2026-11-18. **Run:** `joint7-v11-2-g1190-restart-3`, newest committed generation g3423.
- **Adopted with** `--value-revision search-aux` at a committed boundary (planned: g3423, together with `--soft-policy`).
- **Needs** three things the run already has: the era heads (`heads`, g3275), the ownership stop (`ownership-stop`, g3318) and the g3270 teacher revision.

## What it does

### 1. Each table sends its era-end scoring maps once, with its finish message

- **The maps were already computed.** The CPU teacher passes every transition through `ordinary_after` with a native scoring recorder (`lab-native/src/teacher.rs`), the same recorder ordinary games use. It records both era-end maps (owner and VP of every slot and link, with the official scoring audit) but never sent them. The `finished` message now carries `"scoring": recorder.finish(&state.export())`, exactly what an ordinary game's final message sends.
- **Where the change lives.** The change is an exact-text patch in `rust_backend/lab_cpu_overlay.py`; `lab-native/` itself is unchanged.
  - The patch compiles to the same binary a direct edit would, so there is no performance difference.
  - Teacher jobs are fingerprinted from `lab-native/*.rs`, so a direct edit would have relabelled every teacher job and the qualified producer. With the patch, only the CPU worker is rebuilt, and the continuation does that by itself (`ensure_cpu`).
- **Admission (with the item on).** The receipt goes through the same era audit as an ordinary game's (`local_scoring_targets.audit_eras`) and is bound to the table's own result (final VP). It is then sealed and packed onto the table's first root in its search-value shard.
  - The map is stored once per table, about 4 KB.
  - A held-out table still has its map when it leaves the held-out set eight iterations later.

### 2. Each played root is matched to its era and seat mapping

Each root of an all-seat table becomes one row (64 per table), built from the root's search-value row:
- **Input:** the acting seat's encoded state.
- **Seat mapping:** the root's `mapping`, stable seat → relative. This equals the encoder's `stable_to_relative`; both are `(s − seat) mod 4`, and admission checks it.
- **Era:** `turn_key[0]`.
- **Target:** that era's end map, plus the Rail-end map for Canal rows once `rail-end` is on.
- **Unchanged from ordinary rows:** labels, masks, the VP scale (/5) and the loss weights (0.15 per era now). The rows go through the companion's own `_auxiliary`.
- **Seat descriptors:** search rows have none, so they read as unknown seats, as for the search-value head.

### 3. One batch per iteration joins the first PPG step

- **The training tables (40):**
  - the 36 train tables of the iteration;
  - the 4 held-out tables that leave the 32-game rotating search-value held-out set this iteration, collected 8 iterations earlier.
  - That is 2,560 roots, which fit one 4,096-row PPG batch.
- **The step:** the batch is added to the first joint step of the value phase. That step's ordinary batch is unchanged.
- **Losses:** search rows add only the era heads' ownership and local-VP losses (and Rail-end when on), with no value, utility or policy term. Each search row is weighted as one ordinary row of that step: the step's loss is divided by its ordinary batch size.
- **Existing machinery:** the heads' trunk-gradient scale, the PPG KL checks, the value check with its interval rollback, and the ownership stop apply to that step as to any other.
- **Accounting:** search rows are counted apart. The ordinary attempted, retained and fitting counters, the warm-up counters and `by_era` stay those of the ordinary rows. The search rows' own era sums are in `search_auxiliary.by_era`.

### 4. The 4 current search validation tables early-stop and evaluate

- **Scoring.** This iteration's 4 validation tables (256 roots) are scored by the ownership stop at every boundary, beside its ordinary validation games, with the same owner-CE criterion (each root weighted 1).
- **The interval that holds the search batch** (steps 1–16) is kept only if neither owner CE increased. An increase of either one restores the heads and their Adam to the start, and ends head fitting for the phase. That is the existing rollback.
  - The reason is `heldout_ce_increased`, or `search_heldout_ce_increased` when only the search tables rose.
- **Later intervals** are judged on the ordinary games as before. Their search CE is only reported.
- **No overlap:** the validation tables are never in that iteration's search batch. They train only after they leave the held-out set.

### When the 40 tables are complete

- Tables collected from g3423 on carry maps.
- **The first 8 iterations** (state g3423–g3430) train on the 36 current train tables, 2,304 roots, because the tables leaving the held-out set were collected before the rebuild. They are reported as `without_maps.rotated = 4`.
- **From the iteration that starts at g3431**, the leaving tables are the g3423 ones, so the batch is 40 tables and 2,560 roots.

### Without the item

- Admission keeps no map, and PPG gets no search batch.
- The rebuilt binary still sends the maps, and the Python side ignores them.

## Cost

| Item | Cost (*estimate* except where read) |
| --- | --- |
| CPU worker | One rebuild, done by the continuation with its focused regressions. The laptop compile of the patched teacher took 50 s (incremental). |
| Native work | None new: the recorder already ran. The finish message grows by about 15 KB per table. |
| Admission | Under 0.1 s for 40 tables (one audit and seal each); about 4 KB per table in the shards |
| PPG | About +0.3–0.5 s per iteration: staging 2,816 roots (35 MB on the GPU, transient), one forward-backward of 2,560 roots through the trunk and heads, and 256 held-out roots at each of up to five boundaries. For scale, at g3422 the whole value phase took 20 s, 9.2 s of it in training. |
| Search, collection, held-out data | None: the 40 tables and the 4 validation tables already exist |

**Risk.** The extra held-out criterion on the first interval can end ordinary head fitting more often.
- From g3318 to g3417 the stop already kept 0 head steps in 43 of 100 iterations.
- The 4 tables are 256 roots, so their CE is noisier than the 31,000-row ordinary criterion.
- If `aux_stop=search_heldout_ce_increased` turns up in many iterations while the search CE is not falling, that rule is too strict for 4 tables.

## What to read

| Field | Meaning |
| --- | --- |
| `value_phase.search_auxiliary.selection` | Current train, rotated and held-out tables; `without_maps`; `rotated_generations` (should be G−8) |
| `value_phase.search_auxiliary.train` / `.heldout` | Rows, games and rows by era of the batch and of the validation tables |
| `value_phase.search_auxiliary.weighted_loss`, `.by_era` | The batch's auxiliary loss, and its owner CE and VP error by era, kind and open/settled |
| `value_phase.search_auxiliary.heads_kept`, `.trunk_kept`, `.stopped_by` | Whether the ownership stop kept the interval with the search batch, whether the value check kept the trunk, and which criterion stopped it |
| `value_phase.search_auxiliary.heldout_owner_ce` | Search validation CE before, after the search interval, and at the end |
| `value_phase.ownership_stop.search`, `.intervals[*].search_*` | The same, from the stop's side, at every boundary |
| `search_value.scoring_maps` | Tables admitted with a map |
| log line | `search_aux=Nrows/Ngames search_ce=before->after search_kept=…`, after `aux_stop=` |

## Checks run (laptop, fixture models)

`tests/Recovery/test_search_auxiliary_g3423.py`, 11 tests:
- **Maps.**
  - The receipt is audited, sealed and bound to its table. A changed total, a missing era, the wrong producer and a final VP that disagrees with the result are all refused.
  - The ordinary receipt check behaves as before.
  - Admission stores the map once, on the first root, and only with the item on.
- **Selection.** The batch is this iteration's train tables plus exactly the G−8 held-out tables. While the held-out set fills, none leave. Tables without maps are counted and skipped.
- **Rows.** Each root reads its era's end map through its own seat mapping (the classes are checked one by one), and a wrong mapping is refused.
- **Companion.**
  - Search rows add only era-head losses: the outcome, utility and policy parameters get no gradient.
  - The trunk follows the heads' scale.
  - The ordinary counters and era sums are unchanged.
  - A stopped head returns nothing.
- **Stop.**
  - The ordinary criterion is identical with or without the search tables.
  - A search-CE increase stops the heads only in the interval that holds the search batch, with reason `search_heldout_ce_increased`.
- **Through `ppg_g210.fit`.**
  - The batch joins step 1 only.
  - The ordinary presentations equal those of the same phase without the search batch, and no teacher game enters the ordinary accounting.
  - Without maps, the batch is skipped.
- **Recipe.** The item needs the teacher revision and the ownership stop, is idempotent and forward-only, and `is_on` still recognizes the recipe.
- **Native.** The overlay adds the field once, and `lab-native/src/teacher.rs` is untouched. The patched CPU teacher compiles (`cargo build --features teacher --bin brass-lab-teacher`).

Also run with this change: soft policy S1 (12), g3318 ownership stop (6), g3140 (11), g3270 (18), local scoring, `test_cpu_inference.Overlay` and `test_ppg_g210`, all passing except two `test_ppg_g210.Restored` tests. Those need an old g0660 export that is not in the local results repository.

No teacher table was played locally. The first lab iteration is the check that the rebuilt teacher's maps pass admission, in `search_value.scoring_maps`.

## Code

| File | Change |
| --- | --- |
| `rust_backend/lab_cpu_overlay.py` | The finish message carries `scoring` (CPU teacher only) |
| `rust_backend/lab_teacher.py` | `collect` returns it with the table |
| `recovery/local_scoring_targets.py` | `audit_eras` (shared by ordinary games and tables), `validate_table_receipt` |
| `recovery/search_auxiliary.py` (new) | Table envelopes, table selection (train + leaving held-out; current validation), `SearchRows`, the one-batch trainer and report |
| `recovery/search_value_phase.py`, `search_value_iteration.py` | Keep the map on the table's shard; hand the tables to PPG in `prepare_ppg` |
| `recovery/local_scoring_v3.py` | `search_loss` (era-head losses only, counted apart), `search_report` |
| `recovery/ownership_stop.py` | The search validation tables: scored at every boundary; they decide the interval that holds the search batch |
| `recovery/value_phase.py`, `ppg_g210.py` | The batch at step 1; the report |
| `recovery/training_revision_g3140.py` | Item `search-aux` (block `search_auxiliary`, forward-only, tag `searchaux`) |
| `recovery/league.py`, `scripts/continue-rust-g1491.py`, `continue-rust-g1190-restart-2.py` | Log line, help text |

The execution identity is re-recorded together with soft policy S1 (`e62799b4…`; `0dcb6b30…` is kept as a predecessor).

## Continue command

You have not run the soft-policy command yet, so this one replaces it. It adopts soft policy S1 and the search auxiliary together at g3423 and runs to the g3430 rating:

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3423 --iterations 7 --share-target 0.1 --soft-policy --value-revision search-aux
```

- **Before the first iteration** the continuation rebuilds the CPU teacher, which takes a few minutes once.
- **After the first iteration (g3424), check:**
  - `search_value.scoring_maps` is 40 of 40;
  - `search_aux=2304rows/36games`;
  - a `search_ce` before → after;
  - the soft-policy checks in [G3423_SOFT_POLICY_S1.md](G3423_SOFT_POLICY_S1.md).
- **Then 20 iterations to the g3450 rating.** Both settings are stored with the run:

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3430 --iterations 20 --share-target 0.1
```

- **From the iteration that starts at g3431** the batch reaches 40 tables (`search_aux=2560rows/40games`).
- **Adopting both at once** means the first iterations change two things. They act in different places, and each is reported separately:
  - the soft head trains on the teacher term inside PPO (`teacher_phase.term.soft`);
  - the search batch trains the era heads in PPG (`value_phase.search_auxiliary`).
