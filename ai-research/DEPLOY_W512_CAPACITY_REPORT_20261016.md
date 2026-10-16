# W448×6 versus W512×8: CPU search with a shared 12-second turn budget

16 October 2026. **W512×8 has a measurable neural inference cost, but this
experiment did not find a uniform reduction in completed search work.** Its
single-position utility forward was **32.2% slower** on a geometric mean across
six fixed inputs. Whole-search totals were higher, mainly because one Canal
root produced substantially more short completed paths. The median paired
completion-count ratio was **1.00** with both worker counts. These are initialized
networks; neither the higher count nor the larger model establishes stronger play.

## Models and controlled settings

| Setting | W448×6 | W512×8 |
| --- | ---: | ---: |
| Six projections | 32 / 64 / 48 / 256 / 288 / 48 | Same |
| Context passthrough | 12 | 12 |
| Fusion | 748 → 448 | 748 → 512 |
| Residual blocks | 6 | 8 |
| Action / policy / outcome hidden width | 128 | 128 |
| Base parameters | 3,656,781 | **5,512,717** |
| Including utility | 3,671,409 | **5,529,393** |
| Including personality/goal extension | 3,762,154 | **5,631,210** |
| Utility dense MACs | 3,599,520 + N × 58,368 | 5,451,680 + N × 58,368 |

W512 retains the exact group-projection and action-encoder weights and initializes
the resized layers with seed 610512. W448 uses the previous benchmark's exact
initialization. Its utility and actor ONNX hashes are **identical to the earlier
W352/W448 experiment**, and every shared source pin from that experiment still
matches. Both are benchmark-only architectures rejected by normal production
config validation. No trained weights or production configurations were changed.

Intel Core Ultra 5 135U; Windows; Python 3.12.14; Torch 2.8.0+cpu; ONNX Runtime
1.22.1 CPUExecutionProvider; two tensor threads, sequential ORT execution,
one inter-op thread and spinning disabled. Power snapshots during and after
the test show **battery supply**, with battery saver off. Frequency, temperature
and unrelated system load were not controlled. Absolute latencies from the older
session should not be compared directly with this session; the two arms here
were measured alternately under the current conditions.

The actual production SearchSession and portable inference queue were used:
one pending path and leaf batch 1 per worker; central batch cap equal to 2 or 3
workers; 1 ms flush; shared immutable encoding cache; fused native transport;
`alpha=0`. Every turn retains **one 12-second allowance shared across workers,
both actions and all continuations**. The existing finalization reserve leaves
about 10.8 seconds for search. Initial native process construction and driver
root reconstruction are outside that allowance; search-worker reconstruction,
encoding and inference are included.

## Roots and repetitions

The exact six roots and legal-game trace from the earlier W448 experiment were
copied without changing their bytes. Four late roots came from a completed legal
game, selected before timing by the largest full-turn branching factor in each
target era/round. No model-selected fixture, fabricated round change or hidden
state extension was added.

| Root | Legal root actions | Occupied industry slots |
| --- | ---: | ---: |
| Rail round 2 | 119 | 4 |
| Canal round 2 | 387 | 0 |
| Canal round 6 | 122 | 20 |
| Canal round 8 | 30 | 23 |
| Rail round 6 | 110 | 18 |
| Rail round 8 | 40 | 18 |

There were **48 complete measured turns**: two models × two worker counts × six
roots × two paired seed repetitions. Model order alternated and reversed for the
second repetition. Paired arms used the same root, seed and search configuration.
Full-feature LRUs were cleared for every turn while persistent encoders remained
available. The search matrix took 606.4 seconds, below its 15-minute cap.

## Completed search and complex positions

Numbers below are mean completed simulations per turn over two repetitions,
summed across all workers. Interrupted simulations do not count as completed.

| Root | 2 workers W448 → W512 | 3 workers W448 → W512 |
| --- | ---: | ---: |
| Rail round 2 | 4.0 → 4.0 | 4.5 → 5.0 |
| Canal round 2 | 4.0 → 4.0 | 6.0 → 6.0 |
| Canal round 6 | **22.5 → 32.0** | **28.0 → 46.5** |
| Canal round 8 | 2.5 → 2.0 | 3.0 → 3.0 |
| Rail round 6 | 4.0 → 4.0 | 4.5 → 4.5 |
| Rail round 8 | 8.5 → 10.0 | 12.0 → 12.5 |

| Aggregate over 12 turns per arm | W448 / 2 | W512 / 2 | W448 / 3 | W512 / 3 |
| --- | ---: | ---: | ---: | ---: |
| Attempts | 115 | 136 | 152 | 191 |
| Completed | **91** | **112** | **116** | **155** |
| Interrupted | 24 | 24 | 36 | 36 |
| Sum of visited root actions | 48 | 58 | 56 | 64 |
| Structurally qualified diagnostic labels | 11 | 10 | 8 | 9 |
| Fallbacks | 1 | 2 | 4 | 3 |
| Reported insufficient-coverage reason | 0 | 0 | 0 | 0 |
| Whole-turn seconds, min–max | 11.110–11.562 | 11.110–11.453 | 11.109–11.609 | 11.079–11.469 |
| Pure ONNX forward seconds, total | 3.508 | 3.550 | 5.181 | 6.188 |
| Inference positions, including unfinished paths | 862 | 826 | 1,118 | 1,120 |
| Encoder warm hits / misses | 103 / 762 | 131 / 695 | 158 / 964 | 232 / 890 |
| Native state advances, including real/unfinished work | 2,845 | 2,784 | 3,678 | 3,622 |

All 48 turns completed their legal real actions, preserved a shared deadline and
stayed below 12 seconds. No search deadline overrun was recorded. **Zero labels
were used for training.** The unqualified roots above were reported as
`fallbacks`, which is why zero `insufficient_coverage` counters must not be read
as universal label success. Root-coverage sums count separate turn occurrences,
not globally distinct actions; even on occupied boards coverage can remain sparse.

The raw completion totals increase by 23.1% / 33.6%, but excluding the high-count
Canal-round-6 root gives only **46 → 48 (+4.3%)** and **60 → 62 (+3.3%)**. Across
all paired root/seed conditions, the median count ratio is 1.00. Native advance
counts are slightly lower despite more completions. This is consistent with a
changed traversal workload, including shorter completed paths and different cache
reuse, rather than evidence that the larger architecture executes faster.

These counters do not measure average simulation length exactly: they include
the real-turn driver and unfinished paths. Worker profiler durations overlap and
cannot be added together as wall-clock time. No claim of an end-to-end 23–34%
architecture speedup follows from this result.

## Fixed-input neural cost

Each cell used five warmups and 30 timed calls per model with alternating model
order. Inputs were identical, pre-encoded, and repeated across the batch. Encoding,
native rules, queueing and transfers are excluded.

| Root / root candidates | Utility B1 W448 ms | W512 ms | Increase |
| --- | ---: | ---: | ---: |
| Rail round 2 / 119 | 1.816 | 2.335 | 28.6% |
| Canal round 2 / 387 | 2.160 | 2.548 | 18.0% |
| Canal round 6 / 122 | 1.742 | 2.275 | 30.6% |
| Canal round 8 / 30 | 1.448 | 1.847 | 27.6% |
| Rail round 6 / 110 | 1.656 | 2.326 | 40.5% |
| Rail round 8 / 40 | 1.376 | 2.069 | 50.3% |

| Batch | Utility geometric-mean increase | Actor increase |
| --- | ---: | ---: |
| 1 | **32.2%** | 19.8% |
| 2 | 16.9% | 19.6% |
| 3 | 25.7% | 19.5% |

Actual search forward batches were predominantly B1, with some B2 and no B3.
Pure forward time represents roughly 2.6–4.6% of measured whole-turn time in this
matrix, so its slowdown does not translate directly into the same percentage
loss of completed searches. Changing model predictions also changes the workload.

Encoder parity remained exact. New-encoder / persistent-miss / warm-hit median
times were 29.808 / 14.926 / 2.101 ms, with p95 47.031 / 28.556 / 3.191 ms.
These measure this battery-powered session; no encoder implementation or cache
key was changed for the test.

## Validation, cleanup and limits

- Exact parameter counts, shared projection/action weights and eight W512 blocks
  were checked. Experimental configs remain rejected by production validation.
- Four exports passed named-output Torch/ORT parity and row-versus-batch
  invariance on the existing synthetic shape grid. All **72 real-root
  model/head/batch combinations** also passed.
- Root legality, paired seeds/configurations, complete turn actions, shared
  deadlines, the complete 48-turn matrix and unchanged source hashes were checked.
- **All four temporary ONNX files were deleted: 51,843,845 bytes.** The owned model
  directory is empty, with hashes/sizes in `model-cleanup.json`. No `.pt` files
  were created. Original fixtures and earlier reports were preserved.

Normal zero-initialized residual-output and utility branches, untrained values,
two seeds per root, six selected positions and uncontrolled laptop power/thermal
conditions limit generalization. A trained model's traversal distribution and
weight working set may differ. This experiment does not establish stronger play,
training stability, reliable label yield or an optimal worker count.

Evidence: `artifacts/feature-engineering/deploy-w448-w512-12s-20261016/` contains
`prepared.json`, `search.json`, `summary.json`, exact roots/trace, power snapshots,
the executed harness and cleanup/verification receipts. The adjacent `.log`
contains the complete run output.

Reproduction uses the same retained fixture directory and normal repository
dependencies, with `src` and `tests/FeatureEngineering` on `PYTHONPATH`:

```text
python tests/FeatureEngineering/benchmark_deploy_w512.py --output artifacts/feature-engineering/deploy-w448-w512-new-check --repeats 2 --delete-models
```

Use a new output directory. This command runs isolated initialized-model tests
and removes its temporary exports after successful completion.
