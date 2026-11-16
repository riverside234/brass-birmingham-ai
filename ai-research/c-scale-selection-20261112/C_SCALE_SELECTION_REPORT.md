# c_scale selection for the teacher searches (long and short)

## Latest records: g2400 (updated 2026-11-14)

All C_scale reports are kept together in this folder:

| Report | Evidence and current interpretation |
|---|---|
| [g2400 short-search ablation](G2400_SHORT_SEARCH_REPORT.md) | All three new console records: 250 and 500 positions from g2399, then 600 positions pooled from g2396–g2398. At normal:cotton = **85:15**, **0.90** is the provisional Q-gap choice, **0.70** the other leading candidate, and **0.50** the latest sample's action-agreement alternative. |
| [g2400 long-search ablation](G2400_LONG_SEARCH_REPORT.md) | Updated K8/640 versus K8/3000 protocol, w=0, opening T=2 and 60 workers. The latest training export records C_scale **0.10**; a new g2400 long-search numerical result table is still awaiting its source. |
| [Historical g1760 record](#historical-g1760-record-2026-11-12) | Original tests A–E and their decisions, preserved below. |

The g2400 reports use actual searches rerun at each candidate scale. The older short-search recommendation of 0.5 followed by 0.7 below belongs to the historical g1760 rescore-only study; the new short-search report supersedes it for current parameter selection. Recommendations are offline findings, not automatic changes to training or local-game settings.

## Historical g1760 record (2026-11-12)


- **Date:** 2026-11-12 (tests and discussion 2026-11-09 to 2026-11-11)
- **Run:** restart-3 (branch from g1491 of the parent run "restart-g1290"); offline tests ran on the frozen g1760 checkpoint
- **Status:** decision record. No code or config has been changed by this document.
- **Terms used here**
  - **Teacher** = the Gumbel search. It has two profiles:
    - **Long search** (K16, 640 trajectories today; K8 proposed). Its improved policy π′ is the label distilled into the student.
    - **Short search** (K8, 96 trajectories). It plays most turns of the teacher games and feeds value data, but makes no policy labels.
  - **Student** = the policy network (and its heads) trained by PPO plus the teacher term KL(π′ ‖ π_θ).
  - `c_scale` is a property of the teacher search. It decides how hard the search's Q estimates move the student's target (long search) or the played move (short search).

---

## 1. Decisions at a glance

### Long search (teacher whose labels the student learns)

| Setting | Current | Decision |
|---|---|---|
| Candidates K | 16 | **8** |
| Trajectories | 640 | 640 (800 not worth it) |
| Label `c_scale` | 1.0 | **0.15** at K8 (0.1 if staying at K16) |
| `c_visit` | 50 | 50, unchanged |
| Configured share target | 0.05 | **0.15** after the switch; **0.3** after 30–40 healthy iterations; ceiling 0.6 |
| Teacher rows per step | 256 | 96 (about 4 uses per label) |
| `c_scale` below 0.1 | n/a | No |

### Short search (teacher that plays moves; no student labels)

| Setting | Decision |
|---|---|
| `c_scale` | **0.5 now**; step to 0.7 after about 10 healthy iterations; 0.8 only if tested end to end |
| Optional exploration knob | Root prior temperature 1.1 on short searches (untested) |

The two scales are chosen on different criteria. Long-search labels are distilled into the student, so they are judged on **calibration** (NLL of the long reference move). The short search makes no labels, so it is judged on **pick quality and exploration**.

---

## 2. Background

### 2.1 What `c_scale` does

Gumbel search label: π′ = softmax(logits + σ(q)), with σ(q) = (c_visit + max visits) × c_scale × q, and `c_visit` = 50.

Label sharpness multiplier = (50 + visits of the most-visited finalist) × `c_scale`.

Visit counts assume the budget splits equally per halving round (Gumbel MuZero paper). That fits the logged 4 rounds but was **not checked in the Rust code**.

| Profile | Weakest candidate visits | Finalist visits | Multiplier at `c_scale` 1.0 | At the chosen scale |
|---|---|---|---|---|
| K16 / 640 (long, today) | 10 | 150 | 200 | 20 at 0.1 |
| K16 / 800 | 12 | 187 | 237 | n/a |
| K8 / 640 (long, proposed) | 26 | about 185 | 235 | 35 at 0.15 |
| K8 / 800 | 33 | 232 | n/a | n/a |
| K8 / 96 (short) | about 4 | about 28 | 78 | 39 at 0.5, 62 at 0.8 |

Probability boost for a move whose Q is 0.03 better, exp(0.03 × multiplier):

| `c_scale` | Short search (K8/96) | Long K8/640 | Long K16/640 |
|---|---|---|---|
| 0.10 | × 1.26 | × 2.0 | × 1.8 |
| 0.15 | × 1.42 | × 2.9 | × 2.5 |
| 0.30 | × 2.0 | × 8.3 | × 6.1 |
| 0.50 | × 3.2 | × 34 | × 20 |
| 0.80 | × 6.5 | × 281 | × 122 |
| 1.00 | × 10.4 | × 1,153 | × 403 |

At 1.0 with K16 the label is nearly one-hot: a 0.01 utility gap (about 1% win chance) already becomes 2 logits.

### 2.2 Code state (read in earlier sessions)

- `teacher_c_scale()` in `recovery/collector_runtime.py`.
- `TEACHER_BUDGET = dict(mode='count', full=640, full_k=16, fast=96, fast_k=8, c_scale=1.)`, with `C_SCALES = (.1, 1.)`. The recorded config schedule used 0.1 or 1.0.
- A change of `c_scale` retires older buffered labels (`receipt['c_scale']` in `teacher_buffer.py`), so the 4-age buffer needs about 4 iterations to refill.
- The code has one `c_scale`. Whether the **play** scale can differ from the **label** scale (labels can be rebuilt from stored visit counts and Q means) was **not checked in the code**.

---

## 3. Why the long-search scale was in question

With `c_scale` 1.0 labels, every β above 0.01 coincided with higher policy entropy and falling PL ratings. With 0.1 labels (main run g1261–1291) entropy stayed flat.

| Run and period | Labels | β | Policy entropy | PL rating (same run) |
|---|---|---|---|---|
| Main, g1261–1290 | 0.1 | 0.16 | flat at 0.49 | about 705, against about 728 before (within noise) |
| Main, g1293–1340 | 1.0 | 0.064 | 0.49 → 0.78 | 715 → 650 → 595 |
| Parent, g1290–1400 | 1.0 | 0.04 (logged from g1360) | 0.76–0.84 | 722 → 537 |
| Parent, g1493–1499 | 1.0 | 0.28 | 0.80 → 0.98 | 576 → 491 |
| Restart-3, g1600–1729 | 1.0 | 0.01 | 0.55–0.60 | 650–728, up from 561 at the branch |

- Cleanest comparison: parent against restart-3 (entropy coefficient 0.001 in both). Entropy tracked β (0.57 at 0.01, about 0.78 at 0.04, 0.98 at 0.28) and ratings fell as it rose.
- The main run is confounded: its entropy coefficient was ramped 0.001 → 0.003 over g1293–1342, and the search-value stages changed in the same window.
- The search-value trunk pass does not look like the cause: restart-3 has run it at full scale since g1652 and its rating stayed 650–730.
- PL ratings are not comparable across runs; only within-run changes count.
- In the main run at 0.1, the teacher and PPO gradient cosine was +0.06 to +0.15 (g1261–1291), against about zero or negative at 1.0. A raw share at 0.1 was about 1.1–1.5, against about 7 at 1.0 (both at 64 rows).

---

## 4. Test results

All tests rescore or rerun positions from teacher games and compare against a deeper reference search. They were run by the user.

### 4.1 Test A: repeat search, K16/640, `c_scale` 1.0 (200 positions)

| Measure | Result |
|---|---|
| Two seeds agree on top move | 104 / 200 (52.0%) |
| Seed A top move = policy top move | 66 / 200 (33.0%) |
| Seed B top move = policy top move | 66 / 200 (33.0%) |
| Both seeds pick the same non-policy move | 48 / 200 (24.0%) |

Derived: when one search picks the policy move, the other does too 56 of 66 (85%). When one search overrides the policy, the other picks the same override only 48 of 134 (36%).

### 4.2 Test B: K16/640 labels vs a long reference

`short_scale` is the `c_scale` the label was rebuilt at. `match_long_move` = top move equals the long reference's move. `match_long_label` = label top move equals the reference's label top move. `move_NLL` = NLL of the long search's move under the label.

**All 200 positions (64 held-out + 136 train)**

| `c_scale` | match_long_move | match_long_label | move_NLL |
|---|---|---|---|
| policy | 33.0% | 27.5% | 2.8117 |
| 0.1 | 39.0% | 33.5% | **2.5520** |
| 0.3 | 38.5% | 37.0% | 2.7406 |
| 0.5 | 42.5% | 40.5% | 3.2931 |
| 1.0 | **44.5%** | **44.5%** | 5.4232 |

**Held-out (64/64 complete, supported references)**

| `c_scale` | match_long_move | match_long_label | move_NLL |
|---|---|---|---|
| policy | 35.9% | 28.1% | 3.0262 |
| 0.1 | 43.8% | 37.5% | 2.6627 |
| 0.3 | 37.5% | 34.4% | **2.5027** |
| 0.5 | 42.2% | 39.1% | 2.7632 |
| 1.0 | 45.3% | 43.8% | 4.6084 |

**Train (136/136 complete, supported references)**

| `c_scale` | match_long_move | match_long_label | move_NLL |
|---|---|---|---|
| policy | 31.6% | 27.2% | 2.7107 |
| 0.1 | 36.8% | 31.6% | **2.4998** |
| 0.3 | 39.0% | 38.2% | 2.8526 |
| 0.5 | 42.6% | 41.2% | 3.5425 |
| 1.0 | 44.1% | 44.9% | 5.8066 |

### 4.3 Test C: K8/640 labels vs a long reference

**All (200/200 valid pairs)**

| `c_scale` | match_long_move | match_long_label | move_NLL |
|---|---|---|---|
| policy | 55.0% | 69.5% | 1.4242 |
| 0.10 | 59.5% | 75.5% | **1.3737** |
| 0.15 | 60.0% | 74.0% | 1.4402 |
| 0.20 | 61.5% | 74.5% | 1.5342 |
| 0.25 | **63.0%** | 76.0% | 1.6506 |
| 0.30 | 61.0% | 75.5% | 1.7855 |

**Held-out (53/53 valid pairs)**

| `c_scale` | match_long_move | match_long_label | move_NLL |
|---|---|---|---|
| policy | 47.2% | 69.8% | 1.5593 |
| 0.10 | 54.7% | 77.4% | 1.3985 |
| 0.15 | 56.6% | 75.5% | 1.3565 |
| 0.20 | 58.5% | 73.6% | **1.3393** |
| 0.25 | 60.4% | 77.4% | 1.3474 |
| 0.30 | 60.4% | 77.4% | 1.3790 |

**Train (147/147 valid pairs)**

| `c_scale` | match_long_move | match_long_label | move_NLL |
|---|---|---|---|
| policy | 57.8% | 69.4% | 1.3756 |
| 0.10 | 61.2% | 74.8% | **1.3648** |
| 0.15 | 61.2% | 73.5% | 1.4704 |
| 0.20 | 62.6% | 74.8% | 1.6045 |
| 0.25 | 63.9% | 75.5% | 1.7600 |
| 0.30 | 61.2% | 74.8% | 1.9321 |

**Do not compare Test B and Test C numbers.** The policy row moved from 33.0% / 2.81 to 55.0% / 1.42, so the reference changed between the tests. Whether K8 beats K16 cannot be read from these tables.

**Test definitions (given by the user)**
- `train`: positions sampled from the 32 teacher training games that supplied policy-distillation labels.
- `heldout`: positions from four separate teacher validation games whose labels were kept out of policy-distillation updates.
- Both groups were rerun with the same frozen g1760 checkpoint. Each position got a fresh K8 base search and a 4× reference search. The scales were scored from the same base-search statistics at that position.
- The training games used cotton exploration; the validation games did not. Both reruns disabled it, which can change which positions each group holds.
- Held-out is only 53 positions from four games.

### 4.4 Test D: short search (K8, 96 trajectories) vs a 2,560-trajectory long reference

Top-move match (hits out of positions):

| `c_scale` | All (200) | Held-out (53) | Train (147) |
|---|---|---|---|
| policy | 55.0% (110) | 47.2% (25) | 57.8% (85) |
| 0.10 | 57.0% (114) | 52.8% (28) | 58.5% (86) |
| 0.15 | 58.0% (116) | 50.9% (27) | 60.5% (89) |
| 0.20 | 57.0% (114) | 50.9% (27) | 59.2% (87) |
| 0.25 | 58.0% (116) | 50.9% (27) | 60.5% (89) |
| 0.30 | 58.5% (117) | 50.9% (27) | 61.2% (90) |
| 0.35 | 58.0% (116) | 50.9% (27) | 60.5% (89) |
| 0.40 | 58.5% (117) | 50.9% (27) | 61.2% (90) |

### 4.5 Test E: short search, Q-gap columns, scales 0.10–1.00

Column definitions (from the test):
- `differs_from_policy`: how often the short label changes the raw policy's top move.
- `mean_Q_gap`: average of Q_long(reference move) − Q_long(short pick) over all positions, with matches counting zero. Positive means the long search rates its own move higher.
- `miss_Q_gap`: the same, averaged over disagreements where the long search visited both moves.
- `Q_supported_misses`: how many misses contributed to `miss_Q_gap`. Read it with the gap, because coverage differs by scale.

**All (200/200 valid pairs)**

| `c_scale` | match | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---|---|---|---|---|
| policy | 55.0% | 0.0% | +0.01730 | +0.03869 | 89/90 |
| 0.10 | 57.0% | 8.0% | +0.01446 | +0.03432 | 83/86 |
| 0.15 | 58.0% | 12.0% | +0.01386 | +0.03370 | 81/84 |
| 0.20 | 57.0% | 13.5% | +0.01300 | +0.03107 | 82/86 |
| 0.25 | 58.0% | 16.5% | +0.01256 | +0.03076 | 80/84 |
| 0.30 | 58.5% | 17.5% | +0.01193 | +0.02984 | 78/83 |
| 0.35 | 58.0% | 18.0% | +0.01201 | +0.02964 | 79/84 |
| 0.40 | 58.5% | 19.5% | +0.01174 | +0.02935 | 78/83 |
| 0.50 | 58.0% | 21.5% | +0.01130 | +0.02789 | 79/84 |
| 0.60 | 58.0% | 23.5% | +0.01143 | +0.02821 | 79/84 |
| 0.70 | 58.0% | 25.5% | +0.01083 | +0.02694 | 78/84 |
| 0.80 | 57.0% | 27.0% | +0.01051 | +0.02550 | 80/86 |
| 0.90 | 56.5% | 29.0% | +0.01015 | +0.02448 | 80/87 |
| 1.00 | 56.0% | 30.0% | +0.01055 | +0.02514 | 81/88 |

**Held-out (53/53 valid pairs)**

| `c_scale` | match | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---|---|---|---|---|
| policy | 47.2% | 0.0% | +0.01727 | +0.03268 | 28/28 |
| 0.10 | 52.8% | 7.5% | +0.01435 | +0.03042 | 25/25 |
| 0.15 | 50.9% | 9.4% | +0.01463 | +0.02982 | 26/26 |
| 0.20 | 50.9% | 9.4% | +0.01463 | +0.02982 | 26/26 |
| 0.25 | 50.9% | 13.2% | +0.01300 | +0.02650 | 26/26 |
| 0.30 | 50.9% | 13.2% | +0.01300 | +0.02650 | 26/26 |
| 0.35 | 50.9% | 13.2% | +0.01300 | +0.02650 | 26/26 |
| 0.40 | 50.9% | 15.1% | +0.01234 | +0.02515 | 26/26 |
| 0.50 | 50.9% | 15.1% | +0.01234 | +0.02515 | 26/26 |
| 0.60 | 52.8% | 17.0% | +0.01183 | +0.02507 | 25/25 |
| 0.70 | 52.8% | 20.8% | +0.00932 | +0.02019 | 24/25 |
| 0.80 | 52.8% | 22.6% | +0.00862 | +0.01869 | 24/25 |
| 0.90 | 52.8% | 22.6% | +0.00862 | +0.01869 | 24/25 |
| 1.00 | 50.9% | 26.4% | +0.01012 | +0.02105 | 25/26 |

**Train (147/147 valid pairs)**

| `c_scale` | match | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---|---|---|---|---|
| policy | 57.8% | 0.0% | +0.01732 | +0.04145 | 61/62 |
| 0.10 | 58.5% | 8.2% | +0.01450 | +0.03600 | 58/61 |
| 0.15 | 60.5% | 12.9% | +0.01358 | +0.03554 | 55/58 |
| 0.20 | 59.2% | 15.0% | +0.01240 | +0.03166 | 56/60 |
| 0.25 | 60.5% | 17.7% | +0.01239 | +0.03281 | 54/58 |
| 0.30 | 61.2% | 19.0% | +0.01154 | +0.03151 | 52/57 |
| 0.35 | 60.5% | 19.7% | +0.01164 | +0.03118 | 53/58 |
| 0.40 | 61.2% | 21.1% | +0.01152 | +0.03145 | 52/57 |
| 0.50 | 60.5% | 23.8% | +0.01091 | +0.02923 | 53/58 |
| 0.60 | 59.9% | 25.9% | +0.01128 | +0.02967 | 54/59 |
| 0.70 | 59.9% | 27.2% | +0.01138 | +0.02994 | 54/59 |
| 0.80 | 58.5% | 28.6% | +0.01121 | +0.02842 | 56/61 |
| 0.90 | 57.8% | 31.3% | +0.01071 | +0.02697 | 56/62 |
| 1.00 | 57.8% | 31.3% | +0.01071 | +0.02697 | 56/62 |

Reduction in `mean_Q_gap` against the raw policy (all positions): 0.10 → 16%; 0.30 → 31%; 0.50 → 35%; 0.80 → 39%; 0.90 → 41%.

**Limit of Tests D and E.** Every row rescored one base search. A search actually *run* at a given scale also uses that scale to drop candidates between halving rounds (where the short search has about 4 visits per candidate), and the tables do not measure that.

---

## 5. Long search: reasoning and decision

### 5.1 "Search noise" or "search is better"?

The logs could not separate the two. "Mostly search noise" was a hypothesis, not a finding.

Evidence for noise:
- Labels at 1.0 are extremely sharp (finalists about 150 visits), while a candidate dropped in the first halving round has about 10 visits and its Q is much less certain. If per-trajectory values vary by 0.05–0.15 (an assumption, not measured), sampling error alone can decide which action the label picks.
- Labels sit about 1.7–1.9 nats from the policy in every phase, including early canal where the value explains almost none of the outcome (0.001).
- Across 31 held-out checks at three β settings the label–policy gap did not close.

Evidence for "the search is better":
- 1.0 is the Gumbel MuZero paper's setting; sharp labels are intended when Q is trustworthy.
- The G0 gate passed: the search's root value predicts outcomes better than the network's own value.
- Noise-dominated gradients are normal at these batch sizes, and β was small in the held-out checks, so the policy failing to follow says little about label quality.

### 5.2 Reading Tests A and B → 0.1 at K16

- There is signal: two seeds agree 52% against 33% with the policy, and both land on the same non-policy move in 24% of positions.
- A single search's override is mostly not reproducible (36%), and at 1.0 the label puts nearly all its weight on it.
- Sharper labels pick the long search's move more often (1.0: 44.5% vs 0.1: 39.0%; about one standard error on 200 positions) but are badly overconfident. At 0.5 and 1.0 the label predicts the long search's move worse than the raw policy does (NLL 3.29 and 5.42 vs 2.81).
- NLL is the training-relevant measure, because the student is trained toward the whole label distribution. 0.1 wins it overall (2.55) and on train; 0.3 wins on held-out (2.50 vs 2.66). On 200 positions 0.1 and 0.3 cannot be separated.
- Mechanism proposed for the entropy and rating pattern: training toward near-one-hot labels that keep changing their pick teaches the student the average of those picks. This fits the data but was **not tested directly**.

Decision at K16: **0.1**.

### 5.3 `c_scale` below 0.1, and `c_visit`

- **Below 0.1: no.** At scale 0 the label is the policy. The gap between label and policy falls roughly with the square of the scale: about 0.02–0.05 nats at 0.1 (main run), about 0.01 at 0.05, which is below the policy's own drift per iteration (0.006–0.025). The run has already been in that state: around g1282–1290 the label gap was 0.003–0.019, the staleness check threw out every older buffer age, and the teacher did almost nothing. 0.1 is about the floor. Rows 0.03 and 0.05 could be added to the same recompute to see where the NLL turns.
- **`c_visit`: leave at 50.** Every full search has the same visit count, so `c_visit` only rescales the multiplier, which `c_scale` already does. At K16/640: `c_visit` 0 with 0.1 is about `c_scale` 0.075 with `c_visit` 50; `c_visit` 100 is about 0.125. `c_visit` only matters when visit counts differ between searches, which is not the case for the label searches.

### 5.4 Why 0.2–0.3 was mentioned although 0.1 tested best

0.2–0.3 is only the fallback if 0.1 proves too weak to teach anything, and only after a larger test (about 800 positions). A sharper label carries more to learn (the gap grows with scale squared), 0.3 won held-out in Test B, and 0.2 was untested there. 0.5 and above are ruled out.

### 5.5 Budget versus K

800 trajectories buys almost nothing. K is the better lever.

| Setup | Weakest-candidate visits | Finalist visits | Usable label strength vs now |
|---|---|---|---|
| K16, 640 (now) | 10 | 150 | 1.0× |
| K16, 800 | 12 | 187 | about 1.1× |
| K8, 640 | 26 | 185 | about 1.6× |
| K8, 800 | 33 | 232 | about 1.8× |

- 800 improves Q precision by only about 11% while the multiplier grows from 200 to 237, so labels get slightly more overconfident at the same scale. To get K8's gain from budget alone would take about 1,700 trajectories.
- K8 costs nothing in search work. Only 8 of about 77 legal moves are examined. Cotton-exploration candidates take 2 of the 8 slots in searches where they fire. I expect the effect of the dropped K16 candidates to be small, since they are the 10-visit moves the search cannot judge anyway.
- Equal noise: K8 at about 0.14–0.15 matches K16 at 0.1.
- **Priorities for extra search work** (restart-3's 91 s of teacher collection, about 77 s overlapped with ordinary collection):

| Option | Extra search work per iteration | What it buys |
|---|---|---|
| 64 teacher games instead of 32 | about +100% (about +90 s) | Twice the labels, value rows and held-out tables |
| Long search 640 → 1,280 | about +69% (about +65 s) | Each label's Q error down about 29% |
| Short search 96 → 192 | about +31% (about +28 s) | Better moves on turns that make no labels |

  Games first in principle, but the user's overlap cost (each extra 48 games lengthens iterations a lot) made the K8 switch the practical choice.
- **Visit-aware label** (a suggestion, not established practice): shrink each candidate's Q toward the root value in proportion to how few visits it has. It can be rebuilt from stored counts like the `c_scale` rows. Not tested.

### 5.6 Test C (K8) → 0.15, not 0.2

- 0.15 is the highest scale that is not worse than the raw policy on pooled NLL (1.44 vs 1.42), with a 5-point top-move gain. It also matches the equal-noise estimate (K8 about 0.15 ≈ K16 0.1).
- 0.20 has mixed support: it wins on the 53 held-out positions (1.34), but on the 147 train positions every scale above 0.10 is worse than the policy (1.36 → 1.93).
- NLL is somewhat harsh on sharper labels because training averages over many labels (an inference, not shown by the test).
- The risk is lopsided: too-sharp labels cost 120–185 rating points twice; too-soft labels only slow learning.

**Effect of the train/held-out definitions (the user's note).** I kept 0.15. The details make me more cautious about 0.2, not less:
- The checkpoint had not memorised the train positions. The policy moved about 0.005 nats at buffered train positions after one iteration and 0.012 after three (restart-3, g1729), and at most about 0.06 at the strongest teacher setting in the main run. The policy's top move matches the long label equally often in both groups (69.4% and 69.8%).
- Held-out is four games; its preference for 0.20 over 0.15 is 0.017 nats. Train's preference for 0.15 over 0.20 is 0.134 nats over 32 games.
- Train looks like the data the student is actually distilled from (cotton-explored games).

NLL above each group's best scale:

| `c_scale` | Train (32 games) | Held-out (4 games) |
|---|---|---|
| 0.10 | best | +0.06 |
| 0.15 | +0.11 | +0.02 |
| 0.20 | +0.24 | best |
| 0.25 | +0.40 | +0.01 |
| 0.30 | +0.57 | +0.04 |

For 0.20 to win on NLL, held-out would need about 89% of the weight; for 0.15 to beat 0.10 it needs about 72%. Held-out is 27% of the positions and 11% of the games. If anything moved, it would be down toward 0.10; I stay at 0.15 because of the K8 equal-noise argument.

**When to move to 0.2:** after 30–40 iterations at 0.15 if ratings and entropy hold, or sooner if the K8 repeat-search test shows the same override reproduced in about 55–60% of cases or more (it was 36% at K16).

**If the labels should also pick teacher-game moves:** the label scale and the play scale are separate roles. 1.0 gave the best top-move match at K16 and 0.25 the best at K8, but whether play at those scales is stronger is untested (head-to-head against the raw policy has never been run; `head_to_head.games` = 0).

### 5.7 Share target and β (depends on label `c_scale`)

The share target is the teacher-gradient norm as a fraction of the PPO-gradient norm, measured at qualification. β is fixed after qualification; the realised share runs about 1.4–1.8× below the qualified target in every run (main 0.25 → about 0.18; parent 0.40 → 0.28; restart-3 0.05 → 0.033 at its β floor 0.01).

| Labels | Configured share target | Realised share | Notes |
|---|---|---|---|
| `c_scale` 1.0 (old) | 0.05 | about 0.05 | β at its 0.01 floor; do not raise |
| K8 / 0.15, start | **0.15** | about 0.1 | re-qualify β; guess β about 0.04–0.05 (stronger labels qualify to a lower β than the 0.07 estimated for K16/0.1) |
| K8 / 0.15 after 30–40 healthy iterations | **0.3** | about 0.2 | β was about 0.15 in the main run with 0.1 labels; expect lower with K8/0.15; noise cost to PPO's step about 2% |
| Further steps, only with a clear benefit | 0.45 → 0.6 | about 0.3 → 0.4 | noise cost about 4% → 7%; ceiling 0.6 is judgment, not tested |

- 0.3 (β about 0.15) is the strongest soft-label setting the logs cover; it was neutral there (entropy flat, rating within noise). Anything above is untested.
- **Earlier recommendation retracted.** I had called a share of about 0.35 (β about 0.064) "safe" from PPO KL alone. Joining ratings and entropy showed that with 1.0 labels every β above 0.01 cost strength.
- **No β tried has clearly moved the policy toward the labels** (held-out KL after minus before: β 0.01 at 256 rows → −0.002 mean, 7 of 13 checks fell; β 0.04 at 64 rows → +0.004, 5 of 13; β 0.064 at 64 rows → −0.020, 4 of 5, which is weak evidence because one check has noise of about ±0.02). So more share will not fix a label that is too soft; check whether it is the label first.
- The measured share is mostly noise: in the parent run, raw share fell from 6.94 (64 rows) to 1.45–1.76 (1,024 rows), and pure noise predicts about 1.73. The target is a noise budget; β carries the useful push.
- In restart-3 (g1600–1729) the ratio wandered 0.026–0.042 (autocorrelation 0.54), tracking label sharpness (0.30) and not PPO gradient size (−0.10). That supports a slow adaptive β, not a frozen one.
- **If 0.3 shows no benefit after about 50 iterations, do not raise it.** At a soft scale the label is only about 0.02–0.05 nats from the policy, so a larger share just closes a tiny gap slightly faster. Then the label itself has to carry more: `c_scale` 0.2–0.3 after the larger test, or a bigger budget.
- **Signs that justify raising it:** rating up beyond the confidence interval over 3–4 rating points; the policy row in the long-search comparison (33% / NLL 2.81 in the K16 test) improving when the test is rerun with the current checkpoint every 30 or so iterations; guards clean.

---

## 6. Short search: reasoning and decision

The short search makes no policy labels, so the long-search risk (sharp noisy labels distilled into the student) does not apply. The criterion is the quality of the played move and the exploration it adds. The student sees the short search only through the games it generates (which moves are played, the value data), not through a policy label.

The short search plays about 48 of the 64 turns in each teacher game (figure from the earlier discussion). If those moves were nearly policy samples, teacher games would be close to ordinary self-play and the search-value correction would have almost no gap to learn.

### 6.1 How the recommendation moved, and why

| Step | Data available | Recommendation | Reason |
|---|---|---|---|
| 1 | K8 long-search table only | 0.15, test 0.25–0.3 | Same `c_scale` is much weaker at 96 trajectories (multiplier 7.8 at 0.1 vs 23.5 in the long search). At 0.1 the short search is close to a plain policy sample. |
| 2 | Test D (match only) | 0.3 | Flat quality 0.1–0.4 (114–117 hits of 200), so exploration decides. A higher scale moves the choice more (× 2.0 at 0.3 vs × 1.26 at 0.1 for a 0.03 Q edge). Ceiling at about 0.3 rested on an *assumed* noise level for 4-visit candidates. |
| 3 | Test E, 0.10–0.40 | 0.3 | Q gap down 31% vs policy at 0.30, departures from policy 8% → 17.5%; plateau from 0.30. Suggested extending to 0.5–1.0. |
| 4 | Test E, 0.50–1.00 | 0.8 | Gap kept falling to 0.90 (0.01015) with no quality cost, so the assumed noise ceiling was too cautious. Faint turn at 1.00 (gap 0.01055, match 56.0%), so stop at 0.8 for margin; 0.7 for more margin. |
| 5 | No rerun possible | **0.5** | Keeps about 88% of the 0.8 gain over the raw policy (gap 0.0113 vs 0.0105; departures 21.5% vs 27%). The untested effect on candidate dropping scales with the scale, so at 0.5 it is about 37% weaker than at 0.8. Middle of the flat range, far from the turn. |

### 6.2 Final choice: 0.5 now

- Both groups agree on direction: the gap falls with scale on train (0.0173 → about 0.0107–0.0114 over 0.5–1.0) and held-out (0.0173 → 0.0086–0.0123). Held-out is best at 0.8–0.9; train is flat from 0.5 to 1.0.
- Coverage does not change the picture: high scales have a few more unsupported misses (7 at 0.9–1.0 vs 5 at 0.3). Even charging each at three times a typical miss, 0.8–0.9 still comes out lowest.
- The misses get milder as well as rarer: the per-miss gap falls from 0.039 (policy) to 0.030 at 0.30 and 0.026 at 0.80.
- **Correction to an earlier claim:** I said the short search adds little because the match rate improved only 4–7 hits in 200. By Q gap the edge over the raw policy is 31–39%.
- **If the K8 base searches in the user's test ran at a high scale** (for example 1.0), the high rows are better tested than I assumed and 0.7–0.8 is fine. If they ran at 0.1–0.15, stay with 0.5. This is not yet known.
- **Exploration knob if 0.5 is not enough:** most variety in short-search turns comes from Gumbel noise. The more direct setting is the root prior temperature on short searches (1.0 in the config). KataGo uses about 1.1–1.25 at the root. The cost is that teacher games drift further from strong play; try 1.1 first. Untested.

---

## 7. Corrections made during the discussion

- "Hold β fixed" was too strong; revised to a slow, outcome-based adaptive β.
- The 0.35 / β 0.064 recommendation was retracted after ratings and entropy were joined.
- "Labels are mostly search noise" was a hypothesis; evidence for both readings is in 5.1.
- "About 12 uses per label" corrected to about 10.6 (age-weighted) at 256 rows; the main run at 64 rows was about 3.4.
- The short-search edge was undersold (see 6.2).
- The short-search ceiling of 0.3 rested on an assumed noise level and was raised after Test E.
- Tests B and C cannot be compared across tables (policy row changed).
- Earlier in this session, "student" was misread as the short search. Here the teacher is the search (long and short) and the student is the policy.

---

## 8. Validation and guards in the live run

**Long-search labels (K8, 0.15, share 0.15 → 0.3)**
- Re-qualify β after the switch; the buffer needs about 4 iterations to refill.
- Halve β if policy entropy rises more than about 0.05 above restart-3's 0.55–0.60 (entropy moved within about 10 iterations in the main run, ahead of the rating).
- Halve β on a rating drop beyond the confidence interval on two consecutive ratings.
- Halve β if PPO KL exceeds 0.01 or the LR controller cuts, or if the teacher/PPO gradient cosine stays negative for 10 iterations.
- Do not let the adaptive β absorb step changes: after a change of rows per step, `c_scale` or budget, set the new target by hand.
- Optional adaptive β: check every 10 iterations; raise by 1.25× if held-out KL fell in at least 3 of the last 4 checks and PPO KL ≤ 0.01; lower by 0.8× on bad evidence; bounds about 0.03–0.1.

**Short search (0.5, then 0.7)**
- Compare average VP per seat in teacher games for about 10 iterations before and after the change. All four seats use the same search, so this is a rough health check and not a strength measure. A clear drop suggests the short search is playing worse moves; fall back to the previous scale.
- If it holds at 0.5, step to 0.7 and repeat.

---

## 9. Tests still open

1. Repeat-search test at K8 (does 36% override reproducibility rise to 55–60%?).
2. Long-search rows at 0.03 and 0.05 to find where NLL turns; a larger test (about 800 positions) for 0.2 and 0.3.
3. Split the 147 train positions by whether cotton had already been built, to see where the NLL rise at higher scales sits.
4. A proper no-cotton group (validation games from the last 8 iterations give 32 games); report 0.20 minus 0.15 NLL per game, first checking that the four held-out games agree in sign.
5. Per-trajectory value spread from the repeat-search data (compare the two seeds' Q for the same move). My noise sizes rest on an assumed spread of about 0.1.
6. A visit-aware label row on the existing searches.
7. Head-to-head, search against the raw policy, at least 256 games with 2v2 seat rotations. Never run.
8. Rerun the 200 positions with the short search actually run at 0.8 and score its real pick against the long reference. If the Q gap stays near 0.011, keep 0.8; if it is clearly worse, fall back to 0.5.

## 10. Unverified assumptions

- Equal budget per halving round in the Rust search (visit counts in 2.1).
- Whether the play scale and the label scale can be set separately in the code.
- Which `c_scale` the K8/96 base searches in Tests D and E ran at.
- The mechanism in 5.2 (near-one-hot, changing labels teach the average of the picks).
- The ceiling of 0.6 for the configured share target is judgment, not a measured limit.

## 11. Data sources

- User-run offline tests A–E (frozen g1760 checkpoint), tables reproduced from the session.
- Training logs and exports: main run `g1290 trained-restart`, parent `restart-g1290`, branch `restart-3` (`iteration-*/report.json`, ratings reports, recorded configs, teacher-buffer receipts).
- Gumbel MuZero for the 1.0 default and the label formula; KataGo for root prior temperature and lower-confidence-bound ideas.
