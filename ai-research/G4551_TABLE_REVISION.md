# g4551: standard PPO self-play tables, 54 teacher tables, 176 teacher rows, no search head at w = 0

The user's requests from 2026-11-22, built for one continuation from the committed g4551:

1. Skip the search-value head in CPU (ONNX) inference while its weight is 0.
2. Make every PPO self-play table a normal game. Keep the cotton tables among the search (teacher) tables.
3. Run 54 search tables, with a teacher batch of 176 rows. Allow at most 64 CPU workers.

Items 2 and 3 form one recipe revision, `--table-revision` (module `recovery/training_revision_g4551.py`, top-level
block `table_revision`). It is adopted once at a committed boundary and then kept. Item 1 and the 64-worker cap are
code-only, with no flag.

Execution identity: `configs/learning/b07-replay-location-review.json` (`g4551_table_revision`,
`g4551_headless_cpu_bundles`, `g4551_cpu_worker_cap`; `7e52ba1d…`, predecessor `ca3ea6e9…`).

## Continue command

From the newest export (g4551) to the g4570 rating:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 4551 --iterations 19 --share-target 0.1 --table-revision
```

The flag is needed once. Later continuations keep the revision without it. The Rust workers are unchanged, so
nothing is rebuilt.

## 1. No search head in CPU inference unless a table blends it

**Before.** Every CPU bundle of a model with a search head had a sixth output, `search_delta_4`. ONNX Runtime runs
every node of a graph, whatever outputs a caller fetches. So every forward computed the head:
- the 48 teacher tables (about 15,900 forwards each at g4550), although every job bound w = 0;
- every ordinary game and rating, which never read it.

The GPU teacher path already skipped the head at w = 0.

**Now** (`rust_backend/lab_cpu_models.py`):
- `export_bundle(..., search_delta=False)` exports the graph without the head (`<model_id>-main`).
- `ensure` and `bundle_ready` hand ordinary games, ratings and teacher tables the head-less bundle.
  - The exception is a teacher collection where some job or seat binding blends the head (w > 0;
    `needs_search_delta`). Those tables get the full bundle, exactly as before.
- A model without a search head (league anchors) keeps its single bundle, which serves both cases.
- The CPU qualification script keeps the full bundles, because it checks the blend.

**Exactness.** The head-less graph's five outputs are bit-identical to the full graph's on every parity sample
(tested). So searches at w = 0, labels, ordinary games and ratings do not change.

**Saving** (laptop, one thread, a model of the lab's size: 6.07M parameters, of which the head is 0.14M). The forward
was 3–6% faster at 4–48 candidates and about 2% at 160. At the lab's 28.7 s of forward per table that is about
1–1.5 s per table, plus the same share of every ordinary and rating forward.

**One-time cost.** The first iteration exports a head-less bundle for each model it uses, about 23 at g4550. Each takes
a checkpoint load (about 3 s) and an export (2 s locally), so roughly 1–2 minutes, once. The first rating (g4570)
does the same for its agents. After that, only the new learner is exported each iteration, as now.

## 2. Every PPO self-play table is a standard game

`ordinary_cotton.selfplay_settings(config)` returns None once the revision is adopted. With it:
- the planner (`ordinary.plan_collection`) assigns no self-play cotton seats;
- the admission audit (`lab_admission.binding`) refuses one;
- the sampling report drops `selfplay_exploration`.

**Before:** 32 of the 224 self-play tables had one cotton seat (g3528 contract `ordinary-cotton-exploration/6.0.0`).

**Unchanged:**
- the committed `table_plan` block, which still names the cotton rule;
- the 70/30 self-play/historical mix and every table's seeds, policies and opponents;
- the teacher's cotton tables (`teacher-cotton-exploration/10.0.0`): 3/20 of the train tables and one of the four
  held-out tables. At g4552 that is 7 + 1 of the 54.

**Report.** The VP line shows `all` and `standard` only. The cotton line of `progress_vp.svg` ends at the last
10-iteration group that had cotton tables.

## 3. 54 teacher tables, 176 rows per PPO step, at most 64 CPU workers

- **Tables:** 50 train + 4 validation every iteration (was 44 + 4), through `g3270.tables`.
  - The train seeds continue the same stream: the first 44 are the tables the old count would have run.
  - The search-value shards (the head stays paused) and the era heads' search pool grow with them.
  - Held-out tables are still 4 an iteration, and the rotating search-value held-out set is still 32 games.
- **Rows:** 176 teacher label rows per PPO step (was 160), through `g3270.phase_options`.
  - The draw is unchanged: by age weight, no repeats inside the update, grouped by legal width.
  - The soft head trains on the same rows. The adaptive beta carries over.
  - At 48 PPO steps the update draws 8,448 rows from a pool of about 11,900 now, growing to about 13,400 as the
    larger tables fill the 8-iteration buffer.
- **Workers:** `collector_runtime.MAX_WORKERS = 64`. `validate_cpu` and the continuation's `--workers` refuse more.
  - Each table holds one of the 64 lanes while it searches.
  - Ordinary games use the other 10 (was 16) and take each table's CPU as it finishes.
  - Helper processes (replay readers, row checkers, publishers) are not lanes; they are unchanged.

**Kept:** the qualified `teacher_phase` and its label-buffer identity, the search profile (long 640, short 128 at
C_scale .6), the teacher beta, the soft head, and the paused search-value head with its stage.

The revision changes the collection, so any pending work is archived at adoption, as for g4138.

## Cost (estimates)

| item | per iteration |
| --- | --- |
| 6 more teacher tables | ordinary games get 10 free CPUs instead of 16 while the tables search; collection about +3–10 s (it was publication-bound at 72–75 s with 16) |
| more labels (about +12%) | admission and logit checks about +1–2 s, mostly in the ordinary tail |
| 176 rows per step | teacher term about +0.25 s (2.25 s at 160), soft head about +10% |
| era heads, 14% more search rows | well under 1 s |
| no search head at w = 0 | about −1 to −1.5 s on the teacher wall, plus 2–6% off every ordinary forward |
| standard self-play | none |

Total: about +4–10 s on a 249 s iteration (2–4%), plus a one-time 1–2 minutes of bundle exports at g4552 and at the
first rating. Nothing here was measured on the lab.

## Checks after the first iteration (g4552)

- `teacher_phase.collection.tables` is `{train: 50, heldout: 4}`, and `cpu_inference.lanes` is 54.
- In `collection_metrics[*].cpu_inference.model`, `search_delta` is false, and `forward_ns` per position is slightly
  lower.
- `teacher_phase.term.revision.plan.rows_per_step` is 176, with `demand` 8,448 at 48 steps.
- `economy.cotton_by_selfplay.cotton_exploration.tables` is 0 and `quality_pool.table_mix` has no
  `selfplay_exploration`.
- `teacher_phase.collection.cotton_exploration` still reports the teacher cotton tables.
- `native_backend.collection.collector_runtime.cpu_inference.exported_bundles` is about 23 once, then about 1.

## Tests

- `tests/Recovery/test_table_revision_g4551.py` covers:
  - the recipe overlay, peel and validation, and the forward-only transition;
  - 54 jobs with the teacher cotton tables kept;
  - the admission audit refusing a self-play cotton seat after adoption;
  - the VP split and chart with no cotton tables;
  - the 64-worker cap;
  - `needs_search_delta`.
- `tests/RustBackendLab/test_cpu_inference.py` covers:
  - the head-less export: same model id, five outputs, fewer nodes, the head's weights gone, and bit-identical
    outputs on every parity sample;
  - bundle selection and caching, with an anchor's single bundle serving both cases.
- Recipe dry run on the export's g4551 state:
  - `table_revision` is the only changed key, and the forward transition is accepted;
  - 44 + 4 → 50 + 4 tables, with 7 + 1 teacher cotton tables at g4552;
  - 160 → 176 rows; self-play cotton `6.0.0` → off;
  - every job has w = 0, so no table needs the head.
- Six failures in `test_training_g1041`, `test_fresh_neutral_selfplay`, `test_quality_pool` and `test_pfsp_pool`
  fail the same way on unmodified HEAD; they are recorded in `docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md`.
