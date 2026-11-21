# g4288: teacher tables share one worker process and one copy of the network

Code-only. There is no recipe flag, no run state change and no training change: searches, labels, value rows,
search rows and every recorded output are unchanged. Execution identity: see
`configs/learning/b07-replay-location-review.json` (`g4288_teacher_group`, predecessor `25c6c3ad…`). The next
continuation rebuilds the CPU workers by itself (`ensure_cpu`). `BRASS_TEACHER_GROUP=0` restores one process per
table.

Continue command, from the newest export (g4288). The run rates at g4290 and g4310:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 4288 --iterations 22 --share-target 0.1
```

## What g4284–g4287 show

The g4138 engine speedup worked as intended. The 48 tables then moved the bottleneck.

- **Engine.** 0.41 ms per simulation, down from 1.69 ms. A table is 31–35 s and runs 16,384 simulations.
- **Forward pass.** About 25 s of a table's 32 s of search, 1.56 ms per position. At g4137, with 40 tables, it was
  1.18 ms.
- **Collection.** 71–79 s from g4160 to g4280, against 62–69 s before g4138. Teacher work ends at about 57 s:
  - last search at 37 s;
  - tables published at 46 s;
  - admission done at 57 s.
  Self-play games end at about 67–70 s.

### The forward pass is limited by memory, not arithmetic

The same kind of search costs far less once other tables finish. These are g4287's short Rail-late searches, by
the time they ended:

| ended at | tables still running | microseconds per simulation |
| --- | --- | --- |
| 28–32 s | all 48 | 1,860–1,880 |
| 32–34 s | about half | 1,180 |
| 34–36 s | a few | 480 |

Each lane was a separate process with its own 17.1 MB copy of the weights, and a batch-1 forward reads all of it.
48 lanes doing about 24,000 forwards per second read about 400–450 GB/s of weights. That is the peak DRAM bandwidth
of a large server socket.

The same saturation starves everything else on the machine:
- The trainer's parity collate took 27 ms per row. Its intrinsic cost is about 2 ms on this laptop, and the lab
  CPU is about 5× faster: the lab's row helpers take 18 ms per row against 92 ms here.
- The crash-cache write took 0.58 s per table, against about 0.06–0.25 s of intrinsic JSON work here.

### Self-play waits behind the trainer

- During the tables, only 16 self-play processes run, and never more than 16 games are in flight.
- Unfinished games hit the cap of 128, so dispatch stops while games wait to be published and taken in.
- Publication itself is cheap: 0.24 s median per game, plus 0.43 s to read the export. Yet the pipeline cleared
  only 4.8 games per second.
- 35 of 320 replay shards were still unsealed when collection ended.

## Change

A collection's streamed CPU count tables run in one worker process: `lab_teacher.TableGroup` on the trainer side,
`lab-native-cpu/src/cpu_group.rs` in the worker. Within that process:
- Each table runs the unchanged single-table teacher (`teacher::run_request`, split from `run()` by an overlay
  patch) on its own thread, with its own ONNX session.
- The trainer pins each table thread to that table's CPU, as it pinned table processes. The thread reports its
  Linux thread id first.
- Every session is built on one loader thread that owns the process's prepacked-weights container
  (`cpu_eval::shared_session`). All tables therefore read one prepacked copy of the weights. Packing is
  deterministic, so the shared matrix is exactly the one each session would have built.
- One trainer reader thread tails the group's spool and routes tagged frames to each table's unchanged
  `Worker.collect`. It replaces 48 readers that each polled every 10 ms.

Unchanged:
- the lanes, CPU pinning, core release to self-play games, and CPU admission;
- turn verification, parity checks, admission, buffers and every output format;
- the teacher execution core and the sealed `lab-native` tree.

A table failure still cancels the collection, as before. The group process then exits or is killed, and a panic is
reported as that table's failure.

## Exactness

- One table with the shared container equals the HEAD binary bit for bit.
- Three tables in one group equal the same three tables run alone on group threads. This compares labels, value
  rows, search rows, the result, every search's statistics and the position counts.
- On this Windows build, a table on any spawned thread can differ from one on the main thread in a rare last
  digit of an `exp`-based value (one `v_mix` in 29 labels). MinGW's runtime gives the main thread different x87
  precision from new threads. On Linux, `exp` uses SSE code and new threads inherit the parent's floating-point
  state.
- To check this on the lab before any training, the CPU rebuild's focused regressions now include
  `test_grouped_tables_equal_single_tables`. It compares grouped tables with one process per table, each on its
  main thread, on Linux.

## Expected on the lab (estimate)

- If weight reads are the limit, a simulation should approach the cost seen when few tables ran (about 0.5 ms),
  instead of 1.86 ms. Tables would then take roughly 10–15 s instead of 33 s.
- If per-core arithmetic (two lanes per physical core, all-core clock) dominates instead, the gain is smaller. In
  both cases the searches are the same.
- Less memory traffic and 47 fewer polling threads should also unblock the trainer's finalization and the
  self-play pipeline. The 9 s from the last search to publication, the 11 s of admission, and the 10–13 s
  self-play tail all ran while the machine was saturated.
- Cost: sessions are built one at a time. A table's model load is about 50 ms, so the last of 48 tables can start
  about 1–2 s later.

## The other three items

- **Python finalization (item 2).** Its intrinsic cost per table is small: cache 0.06–0.25 s, label shard about
  0.1 s, value shard about 0.35 s and parity about 0.04 s on this laptop. The lab's 9–13 s comes from contention,
  so this change targets the cause. Re-measure before rewriting any of it.
- **Leaf legal moves (item 3).** These are about a quarter of the remaining engine time: about 5% of a table, or
  1.7 s. An exact rewrite of `can_build` could save about half. It is worth doing once the forward pass is
  smaller, when it becomes a larger share.
- **Leaf string view (item 4).** About a fifth of the engine time, or 4% of a table. Removing it needs a feature
  encoder that works from the native state and matches the string encoder bit for bit. That is large and risky
  for about 1.4 s per table, so it is not worth doing now.

## What to check after the first iteration

- `teacher_phase.collection.finalization.table_group`: `{"tables": 48, "processes": 1}`.
- `collection_metrics[*].cpu_inference.forward_ns / positions`: about 1.56 ms per position at g4287.
- `collection_metrics[*].elapsed_ns`: tables were 31–35 s.
- `finalization.last_search_seconds`, `tables_published_seconds` and `collate_seconds` (were 37, 46 and 10.5 s).
- `timings.collection_seconds`: was 71–74 s.
- `native_backend.collection.dispatch.peak_active_games` and `peak_unfinished_families`: were 16 and 128.
- The CPU rebuild's `regressions.log` passes `test_grouped_tables_equal_single_tables`.

## If collection is still longer than the tables

- **Duplicate checks in the trainer.** The trainer re-checks every publisher-prepared self-play row's arrays
  (40–60 µs per row, about 18 ms per game). The publisher already checked them, so this could be skipped. It is a
  removed duplicate check, not a data change.
- **Serial parity batcher.** The logit-parity batcher runs its table groups one at a time: 13 groups at about 1 s
  each under contention.
