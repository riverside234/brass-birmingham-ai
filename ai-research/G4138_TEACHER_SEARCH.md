# g4138 teacher search revision: 48 tables, short search 128 at C_scale .6

The user asked for three changes on 2026-11-20:
- 48 search teacher tables;
- short searches of 128 simulations;
- short-search C_scale .6.

Long-search labels must be kept, and anything built from the short search rebuilt.

This is one recipe revision, `--teacher-search-revision` (module `recovery/training_revision_g4138.py`, top-level block
`teacher_search_revision`). It is adopted once at a committed boundary and then kept. Execution identity:
see `configs/learning/b07-replay-location-review.json` (`g4138_teacher_search`, predecessor `7f8399ea…`). The same
identity also carries the engine speedup (`ai-research/ENGINE_SEARCH_SPEED.md`).

Continue command, from the newest export (g4138) to the g4150 rating:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 4138 --iterations 12 --share-target 0.1 --teacher-search-revision
```

The flag is needed once, at adoption. Later continuations keep the revision without it. `--teacher-fast` is not
needed: the recipe sets the short search, and a CPU budget without it is refused.

## What was running (g4137 report)

- Tables: 40, as 36 train + 4 validation (g3270 revision).
- Long search: 640 simulations, K 8, C_scale .20 (`count_search` G2600), 16 per table (the reserved turns).
- Short search: 96 simulations, K 8, C_scale .70, 48 per table. (The short count was 96, not 94: CPU budget
  `fast: 96` in every table's receipt.)
- Simulations per table: 14,848 = 16 × 640 + 48 × 96.
- Short-search labels have been off since the g1490 restart (`labels: long-only`, `short_labels.labels: 0`). The
  policy-label buffer holds only long labels: 1,244 per iteration, 9,942 in the 8-age pool.

## Change

| | before | after |
| --- | --- | --- |
| teacher tables | 36 train + 4 validation | 44 train + 4 validation |
| short search | 96 simulations, K 8, C_scale .70 | 128 simulations, K 8, C_scale .60 |
| long search | 640, K 8, C_scale .20 | unchanged |
| simulations per table | 14,848 | 16,384 (+10.3%) |
| cotton train tables | 3/20 of consecutive tables | the same rule (7 of 44 at g4138) |
| validation tables | 4 (3 standard, 1 cotton) | unchanged |

Where each setting lives:
- The qualified recipe is not edited: `teacher_phase`, its g1491 marker and `count_search`. The new block overlays
  what the trainer and the CPU teacher run.
- `g3270.tables` / `phase_options` read the tables from the block.
- `search_profile(config)` is the committed profile plus `fast 128, fast_k 8, fast_c_scale .6`. Three places use it:
  `continue-rust-g1491.py` (the CPU budget), `collector_runtime.validate_teacher_budget` (refuses any other budget
  before training) and table admission in `lab_shadow` (refuses a table whose receipt or searches used another
  budget or scale).
- C_scale .6 is a new allowed value in `collector_runtime.C_SCALES` and in the CPU worker's budget parser
  (`lab-native-cpu/src/cpu_eval.rs`). The continuation rebuilds the CPU teacher (`ensure_cpu`).

## Buffers

**Long labels: kept.** The policy-label buffer identity (`buffer_identity`) reads the qualified teacher phase and
the g1491 marker, which do not change. All 17 committed shards still match it (dry run on the g4138 state), so:
- the long labels, their 8 ages and the teacher beta carry over;
- adaptive beta continues, with no re-measurement: rows per PPO step stay 128 and the label target is unchanged;
- the soft head carries over.

A long label is a search from a reached state, so a different short search on earlier turns does not change what
the label says.

**Short search: there is no short-label buffer to rebuild.** The short search's results reach training in two ways:
1. The moves played at short-searched turns (75% of searched turns). They set which states the teacher games
   reach, for every row type. Any reached state is valid data, so nothing is retired for this.
2. The root search values in the search-value buffer. 48 of a value game's 64 roots are short searches. Every
   search-value training return (λ .95 per turn) bootstraps from the following roots' search values. These
   targets change with the short search.

**Search value: rebuilt.** The search-value head trains only on games collected from adoption (g4138 on):
- It waits until eight such ages exist. That is the reviewed rebuild length (`required_ages 8`, as in the earlier
  teacher search retunes).
- It pauses from g4138 to g4144 and trains again from g4145. Older games leave as they age out of the
  16-iteration buffer.
- Held-out games leaving the latest 32 train it only if collected from adoption.

What stays unfiltered:
- The dedicated held-out check keeps every game, because it scores realized utility-to-go, not search values.
- The era heads keep their search tables of every age (`search_auxiliary.tables` reads the same shards). Their
  targets are maps and their own λ-returns, not search values.
- No shard is deleted or retired. The head ignores older games only for its own training.

Unlike the earlier retunes (`teacher_search_retune`), this needs no checkpoint rewrite. The rule is
`training_revision_g4138.rebuild(config)`, applied in `SearchValuePhase._scheduled_after_ppg`, with report
`search_value.buffer_rebuild` (`fresh_ages`, `kept_rows`, `excluded_rows`, `paused`).

Cost of the pause: at g4137 the head's 32 steps moved its held-out loss from .16476 to .16474, and the whole
search-value phase took 3.3 s. Seven paused iterations lose very little and save that time.

## Cost (estimate; check after the first iteration)

These figures are on top of the engine speedup in the same identity. That speedup is expected to bring a table
from about 46 s to about 27 s and the collection from about 63 s to about 42 s.

- **Teacher table time**: +1,536 simulations, about +2.5–3 s per table, so about 29–30 s. The forward pass sets
  most of it: about 1.2 ms per position at g4137.
- **More tables**: 48 tables run in parallel on 48 of the 64 CPU lanes, so the time per table is unchanged. While
  tables run, ordinary games get 16 lanes instead of 24. At about 1 s of native work per game, 320 games take about
  20 s on 16 lanes, which still fits inside the tables. Ordinary lanes also take a teacher CPU once its last search
  ends.
- **Teacher finalization**: the Python-side collate and check after the last search grow about 20%, about +1–2 s.
- **Collection**: about 45–48 s instead of about 42 s, so about +4–6 s per iteration.
- **Value phase**: the era heads' search pool grows from 160 to 192 tables (+20% search rows mixed into the PPG
  batches), about +1 s. The search-value head saves about 3 s per iteration while it is paused.
- **PPO**: unchanged, with 128 label rows per step. The label pool grows about 20% (to about 1,490 labels per
  iteration, about 12,000 at 8 ages), so each label is drawn somewhat less often.
- **Total**: about +5–8 s per iteration over the engine-speedup baseline, about 175–178 s. That is still about
  12–15 s below g4137's 190 s.
- **CPU and memory**: teacher CPU time is about 1,440 core-seconds per iteration instead of about 1,080 after the
  speedup (about 1,850 at g4137). Label and value memory grow about 20%.
- **Data per iteration**: +22% long labels, +22% search-value training games, +20% era-head search rows. The
  validation tables and held-out games do not change.

## Checks

Dry run on the exported g4138 state (`state.json`, no checkpoint):
- The revised recipe validates and is on.
- The label identity is unchanged, and 17 of 17 committed label shards match it.
- Tables are 44 + 4 (7 cotton train tables at g4138), and the validation jobs are unchanged.
- The CPU budget is accepted only with 128 simulations, K 8 and C_scale .6.
- The old run's g2600 buffer rebuild is long finished (ready at g2607).

New tests in `tests/Recovery/test_teacher_search_g4138.py`:
- the recipe overlay keeps the labels;
- adoption once, plus the collection change;
- 48 jobs;
- the budget refusals;
- the search-value head trains only on games from adoption after eight ages, rotated held-out games included.

The CPU worker's budget test now parses 128 at .6 and refuses .65.

## What to check after the first iteration

- `teacher_phase.collection.collection_metrics[*].cpu_inference.budget`: `fast 128, fast_c_scale 0.6`. Each table
  should have 16,384 simulations and about 29–30 s of `elapsed_ns`.
- `teacher_phase.collection.tables`: `{"train": 44, "heldout": 4}`. `labels` should be about 1,490.
- `search_value.buffer_rebuild`: `fresh_ages 1`, `paused true`, through g4144. At g4145, `fresh_ages 8`,
  `paused false`.
- `teacher_phase.term.beta` continues from about .0312, with no re-measurement.
- `timings.collection_seconds` should be about 45–48 s.
