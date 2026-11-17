# Training design for a stronger four-player AI (review through g2910)

- **Date:** 2026-11-16. **Run:** `joint7-v11-2-g1190-restart-3`.
- **Evidence:** the 17 exports from g1730 to g2910 (1,150 iteration summaries, g1600–g2909, plus the full reports of the last iterations of each export), the training code, and the earlier PPG and value-head reports in this folder. Latest export: [g2910](../results/joint7-v11-2-g1190-restart-3-g2910-20261116T231214598848Z). Section 4.5 also uses 25 games of the g1190 model recorded on the laptop.
- **Scope:** a design. No code or recipe was changed. Numbers marked *estimate* are derived or extrapolated; all others are read from the reports.

## Summary

The run gains strength slowly and has been flat since about g2690. Three things hold it back, and the design changes each of them. PPO stays as it is, and so do the value phase's cap, floor and strict early-stop rule.

1. **The search teacher's work barely reaches the policy.** It makes about 1,000 labels per iteration from 32 games, throws away three times as many from its short searches, and gets about 2% of the policy's movement. The design uses every search as a label, keeps labels twice as long, and trains on them in a phase of its own with a KL budget.
2. **A quarter of every iteration trains nothing.** About 50 s of 229 s is waiting, repeated host work and reporting. Ordinary games also spend most of their CPU on the protocol between Python and the game worker.
3. **The value phase's loss gives the critic little to learn from.** The critic learns from one noisy outcome per seat and game, and more value steps do not help it. The local-scoring auxiliary should add detail, but on held-out games it scores like a rule that copies the board: its heads cannot tell one empty slot from another, its VP part is 0.2% of it, and it treats the Canal and Rail eras as one problem. The early stop decides on 32 games and keeps the first interval 49% of the time. The design rebuilds the auxiliary's heads, separates the eras, gives the win and utility heads a less noisy target, and runs the early stop on more games without holding any back from training.

### The design in one table

| Part | Today | Design |
| --- | --- | --- |
| Teacher tables per iteration | 32, plus 4 held-out every fifth iteration | 40; 4 of them validate before they train |
| Searches per table | 16 long (640 simulations), 49 short (96) | the same |
| Labels per iteration | 993, long searches only | about 5,000, every searched turn |
| Label target | fixed at collection: old policy plus search bonus | rebuilt at training time: current policy plus the stored search bonus; one label scale of 0.20 |
| Label life | 4 iterations, pool 3,970 from 128 games | 8 iterations, pool about 40,000 from 320 games |
| How labels train the policy | inside PPO steps, 96 rows × 46, gradient share 0.10 | own phase after PPO: up to 48 steps of 512 rows, KL budget 0.001 rising to 0.002 |
| Teacher's part of the policy's movement | about 0.0002 per iteration (*estimate*) | 0.002 |
| PPO | 0.0074 per iteration | unchanged |
| Value phase | cap 0.003, beta floor 0.25, strict held-out early stop | unchanged |
| Early-stop games | 32 games (5% of two iterations), never trained | 80 new games, trained in the next iteration |
| Local-scoring heads | one shared trunk vector added to each slot's own features; one output for both eras | each slot and link reads its own part of the trunk; separate Canal and Rail outputs |
| Local-scoring loss | owner 0.05, VP 0.01; the VP terms are 0.2% of it | VP on the same footing as owner; 0.05 while the new heads warm up, then 0.15, then 0.5 |
| Win and utility targets | final outcome | half final outcome, half bootstrapped return |
| Total-VP weight | 0.10 | 0.5 |
| Search-value head | trained, used with weight 0 | paused while the weight is 0 |
| Ordinary games | about 10 core-seconds each in effect | about 5 as a target, then 480–640 games per iteration |
| Iteration time | 229 s | about 185 s (*estimate*) |

### Your concerns

| Concern | Answer |
| --- | --- |
| 1. Distill design | Sections 2 and 3: every search is a label, labels from more games live longer, and distillation becomes a phase of its own. |
| 2. Beta | Replaced by a KL budget. A gradient share of 0.15 is only about 2% of the policy's movement, which is why 0.10, 0.15 and 0.30 look the same. |
| 3. PPG KL cap | Keep 0.003 and the 0.25 floor. A 0.125 floor needs a cap near 0.005 and has nothing to buy: kept value steps do not make the critic better on new games. |
| 4. Collection | One table costs about 0.8 s per iteration. Removing the tables' waiting pays for 8 more tables. No other table is added. |
| 5. Local-scoring auxiliary | Sections 4.5 and 4.6. The heads cannot say anything about a particular empty slot, the VP part has no weight, and Canal and Rail share one output and one sum. New heads, separate eras, a VP term with real weight, then a larger weight. |
| 6. Held-out early-stop loss | Sections 4.7 and 4.8. The formula is sound: its parts move together. The weakness is 32 games. Validate on a quarter of the new games and train on them one iteration later. |

### Cost

The design holds no game back from training. The 4 validation tables and the critic's 80 validation games all train one iteration later. It adds no search time beyond what removing the waiting frees, gives about five times the labels per iteration, and takes about 45 s less per iteration (*estimate*). The value-loss changes of section 4 add under 1 s per iteration.

## 1. Where the run stands

### Strength

Rating points are Plackett–Luce points against the trained-start baseline (174 points is one unit of log-strength). Each rating uses 128 games and has a 95% interval of about ±30.

| Generations | Rating |
| --- | --- |
| g1500–g1600 | 60–75 |
| g1900–g2400 | 100–120 |
| g2410–g2530 | 120–158 |
| g2540–g2620 | 72–106 |
| g2630–g2790 | 105–168, mean 148 from g2690 |
| g2800–g2910 | 95–174, mean 127 |

### Time per iteration (g2840–g2909, mean 229 s, 15.7 iterations per hour)

| Part | Mean seconds | Share |
| --- | ---: | ---: |
| Collection, games and teacher tables | 90 | 39% |
| PPO update, including the teacher term | 60 | 26% |
| Value phase | 36 | 16% |
| Cleanup (75–84 s every fifth iteration) | 16 | 7% |
| Rating games (68–78 s every tenth iteration) | 8.5 | 4% |
| Other | 18 | 8% |

## 2. Search labels: use every search, from more games (concern 1)

### What the logs show

- **Few labels from few games.** About 993 labels per iteration from 32 games, against 46,700 PPO decisions from 320 games. Inside one iteration, 4,416 presentations cover about 2,510 distinct rows, so a row repeats 1.76 times.
- **The term fits its own rows.** On the label buffer the loss falls by 0.0057 ± 0.0019 inside an iteration although the policy moves 0.0080. That is a pull of about 0.014 toward the teacher on trained rows.
- **Almost none of it reaches new positions.** On held-out tables, one iteration's update raises the KL to the teacher's target by 0.0074 ± 0.0005 (202 measurements since g1800), and an update that ignored the teacher would raise it by the policy's own movement of 0.0080. The pull on new positions is about 0.0006 ± 0.0005, roughly a twentieth of the pull on trained rows (*estimate*; the movement is measured on PPO positions).
- **The weight does not change this.** The held-out rise was 0.0057 ± 0.0011 at share 0.15, 0.0058 ± 0.0014 at 0.30 and 0.0071 ± 0.0021 at 0.10.
- **Three quarters of the searches give no label.** Each table runs 49 short searches and uses none of them. In the g2400 study a 96-simulation search closed a third to a half of the raw policy's value gap to a 3,000-simulation search (0.0060–0.0076 against 0.0105–0.0121).
- **Labels are dropped while still valid.** At age 3 the policy has drifted 0.029 from the label's prior while the label's improvement is 0.20.
- **Search play is only a little better than policy play.** Teacher tables score 136.6 VP per seat against 135.8 in PPO self-play (g2700–g2909). The teacher is an improvement step, not a much stronger player, which is why it gets a modest budget in section 3.

### 2.1 Every searched turn is a label

Turn the short-search labels on. A table then gives about 126 labels instead of 31 (*estimate*: 16 long turns and 49 short turns, 1.94 labels per turn), for the same search time.

- The worker and the label checks already support short labels (the g1491 code path).
- Their cost is verification. Labelled turns cost about 34 ms of helper CPU each, and the logit parity check takes 8–16 s for 1,112 rows today. Check a fixed sample of rows per iteration, so the cost stays where it is.
- Short labels were on once, at g1491, together with five other changes. Three things differ here: their label scale (2.2), each label is used at most once per iteration (section 3), and PPO keeps its two epochs and its whole step.

### 2.2 One label scale for both kinds of search

A label's sharpness is the scale times (50 + the most visits of any move) times the value difference. A long search gives each of its 8 candidates 27, 80 or 187 visits. A short search gives 4, 12 or 28.

| Label | Scale | Sharpness per unit of value | Visits behind the weakest candidate | Noise relative to a long label (*estimate*) |
| --- | ---: | ---: | ---: | ---: |
| Long, today | 0.20 | 47 | 27 | 1.0 |
| Short at its play scale | 0.70 | 55 | 4 | 3.0 |
| Short at the label scale | 0.20 | 16 | 4 | 0.9 |

The short search plays at 0.70, which is right for choosing a move. As a label at that scale it would be as sharp as a long label on a seventh of the visits. Recompute short labels at 0.20 when they are loaded. The label already stores the visit counts and values, so nothing changes in the search. A short label is then a softer push with about the same noise as a long one.

### 2.3 Rebuild the target from the current policy

Today a label's target is fixed at collection: the policy of that iteration plus the search bonus. A few iterations later, part of what the label teaches is to return to the old policy.

Store the bonus and rebuild the target each time the label is used:

`target = softmax(current logits + scale × (50 + most visits) × completed value)`

- The completed values are in the label already (search value for visited moves, the mixed value for the rest).
- An old label then only adds the search's preference. It never pulls the policy back.
- This is what lets short labels live. Their bonus is small, about a ninth of a long label's improvement (*estimate*), so with a fixed target the policy's drift would outweigh it after two or three iterations.
- The exported target stays in the label for validation and for the held-out measurement.

### 2.4 Keep labels for 8 iterations

With rebuilt targets a label's age only matters through its stored values, which change slowly. Keep 8 iterations with age weights 8 to 1. The pool grows from 3,970 labels from 128 games to about 40,000 from 320 games. `TeacherBuffer` accepts only 2 or 4 today.

### 2.5 Forty tables, and validation without held-out tables

- **40 tables.** Removing the tables' waiting (section 5) frees the time of about 10 tables. Use 8 of them. Collection stays at about 80 s (*estimate*).
- **Validation on tables before they train.** Each iteration, 4 of the 40 new tables are kept out of that iteration's training. Teacher KL is measured on them before and after the update, together with the policy's movement on the same rows. From the next iteration they train like the others, with their age counted from then. No label is lost and no extra table is searched.
- **The held-out tables every fifth iteration go.** Their job is done every iteration by the 4 validation tables. If you keep the search-value head, keep its held-out games out of value training only and let their labels train the policy from the next iteration, as you suggested.

## 3. Distillation as its own phase (concerns 1 and 2)

### Why the share is the wrong knob

- **A share of the gradient norm is a much smaller share of the movement.** The teacher gradient and the PPO gradient are almost at right angles (cosine 0.04), so their movements add as squares. A share of 0.15 is about 2% of the step's KL, roughly 0.0002 of the 0.0074. A share of 0.30 is about 8%, and 0.40, the g1491 setting, about 14% (*estimates*). Between 0.10 and 0.30 the teacher always had only a few percent.
- **PPO pays for it.** PPO's learning-rate controller measures KL after the PPO step, teacher included. A larger share shrinks the PPO step.
- **It cancels the labels.** When C_scale went from 0.10 to 0.20, the teacher's gradient norm rose from 5.98 to 8.92 and beta fell from 0.066 to 0.043. Sharper labels, or more of them, get the same influence.
- **It repeats rows.** 96 rows in each of 46 steps present a row 1.76 times per iteration, which helps the term fit rows and not positions.

### What changes

The teacher term leaves the PPO step. It runs as a short phase right after PPO, with its own optimizer, and stops at a KL budget measured on ordinary PPO positions. Order inside an iteration:

1. Collection, label admission, held-out measurement "before".
2. PPO update with no teacher term, then the PPO learning-rate decision. PPO keeps its whole step.
3. **Distillation phase.**
4. Value phase. Its policy-clone reference is taken at its start, so it protects the policy after distillation.
5. Final probe, held-out measurement "after", checkpoint.

### One phase, step by step

1. **Snapshot** the weights and the phase's optimizer state.
2. **Freeze the reference.** Record the post-PPO action probabilities on the PPO probe rows, the 2,048 rows that PPO's own KL checks use. `RawPolicyReference` in `value_phase.py` already does this for the value phase.
3. **Draw batches.** Sample label rows from the staged pool by age weight, 512 rows per batch, without repeating a row inside the phase, grouped by legal width.
4. **Loss.** The mean KL from the rebuilt target (2.3) to the policy over the batch, on the same policy parameters as today (state encoder, entity encoder and projections, action encoder, policy scorer). Long and short labels are one pool with equal weight per label; the label scale already makes a short label the weaker push.
5. **Step** with the phase's own Adam. The learning rate is PPO's last applied rate times a multiplier, with a 4-step warmup. Gradient clip 1.0, as in PPO.
6. **Check every 8 steps.** Measure the KL from the reference on the probe rows. If it is above the budget, restore the last check that was inside it and stop. Otherwise continue to 48 steps.
7. **Guard the critic.** Measure how far the four-seat utility on the probe rows moved. PPO's value preservation reports the same number, about 0.005. If it moved more than 0.01, restore the snapshot and skip the phase for this iteration.
8. **Report** the KL at each check, the steps used, the label loss before and after, and the held-out teacher KL and policy movement on the validation tables.
9. **Adjust for the next iteration.** Multiply the learning-rate multiplier by 1.25 if the phase ended below half the budget, divide it by 1.25 if it reached the budget in the first half of its steps, and hold it otherwise. Bounds 0.25 to 4.

### Settings

| Setting | Value | Why |
| --- | --- | --- |
| KL budget on the PPO probe | 0.001 for 20 iterations, then 0.0015 for 20, then 0.002 | 0.002 is a quarter of PPO's own movement and about ten times what the teacher gets today (*estimate*). It is below the 0.003 that the value phase may already move the policy, and it has a direction. |
| Upper end | 0.003 | The value phase's cap is the natural ceiling. Go there only with the label pool of section 2 in place. |
| Steps | at most 48 | 24,576 presentations per iteration: each label is used about five times in its 8 iterations and at most once per iteration. |
| Batch | 512 rows | Each step averages over many more games than the 96 rows used now. |
| KL check | every 8 steps | Six checks per phase, each a forward pass over 2,048 rows. |
| Learning-rate multiplier | starts at 1.0, then the rule in step 9 | It finds the step size at which the phase lands on its budget, whatever the labels' count or sharpness. This is the job the beta controller did. |
| First 4 iterations | run and restore | As the beta qualification does today: the phase runs, its numbers are recorded, and the snapshot is restored, so the multiplier settles before the phase moves the policy. |

The share target, the adaptive beta and the fixed-beta qualification are not used by the phase. The in-PPO term stays in the code as the way back.

### How this differs from g1491

| | g1491 | This design |
| --- | --- | --- |
| Teacher's part of the movement | about 14% of a fixed total, taken from PPO: about 0.001 (*estimate*) | 0.001 rising to 0.002, on top of an unchanged PPO step |
| PPO | cut to one epoch | two epochs, unchanged |
| Rows | 2,048 per step from a 2-iteration buffer, each label presented about four times per iteration (*estimate*) | 512 per step from an 8-iteration pool, each label at most once per iteration |
| Short labels | at the search's own scale | at the label scale, one pool |
| Changes at once | six | one per build step (section 8) |

### Cost

About 8–10 s per iteration (*estimate*): 48 steps of 512 rows through the policy path are about a quarter of PPO's row volume, plus six probe checks and one snapshot. The in-PPO term it replaces takes 1.7 s when its steps are fast and 31 s when they are slow.

### Where it touches the code

| File | Change |
| --- | --- |
| `recovery/teacher_term.py` | Reuse `forward_policy`, the label rows and `heldout_measure` in a new phase class; build the target from stored statistics and current logits. The in-PPO `apply` stays as the fallback. |
| `recovery/league.py` | Call the phase between the PPO learning-rate decision and `phases.fit`. |
| `recovery/teacher_phase.py`, `teacher_phase_config.py` | Phase settings and state (optimizer, multiplier, budget step) in place of the term's weight; validation of the new fields. |
| `recovery/teacher_buffer.py` | 8-iteration window, GPU staging, validation tables released one iteration late. |
| `recovery/training_revision_g1491.py` or a new revision module, `lab_shadow_config.py`, `continue-rust-g1491.py` | Short labels on, 40 tables, the reviewed recipe transition and its flag. |
| Checkpoint | Persist the phase's Adam like the value optimizer. Re-record the execution identity last. |

## 4. Critic and value phase (concerns 3, 5 and 6)

Sections 4.1 to 4.3 answer the cap question. Sections 4.4 to 4.9 are about what the value phase learns from: its loss, the local-scoring auxiliary and the held-out early stop.

### 4.1 Direct answer on the cap and floor

Keep the cap at 0.003 and the floor at 0.25. If you lower the floor to 0.125 you must raise the cap to about 0.005, because the two only work as a pair. I recommend against the pair: it adds policy drift, and the logs show no critic gain to pay for it.

### 4.2 What the logs show about the phase (g2600–g2909, floor 0.25, 310 iterations)

- **How phases end.** The held-out check ends 87% of phases, the KL cap 4%, and 9% run all 59 steps. The phase keeps 0 steps in 51% of iterations, 16 in 23%, 32 in 14%, 48 in 4% and 59 in 9%: a mean of 15.
- **You are right that the value network overfits.** A full phase raises explained variance on its training rows by 0.024. On held-out games the mean changes at the first three checks are +0.0011 ± 0.0005, −0.0005 ± 0.0005 and −0.0004 ± 0.0002 on a loss near 1.18: about zero in sum. Your g1730 head ablation shows the same thing: from 256 to 512 steps the fit error fell and the held-out error rose, for every head.
- **Kept value steps do not make the critic better on new games.** The PPO baseline's explained variance is measured every iteration on 320 new games, before any training on them. Between g2160 and g2909 its correlation with the number of value steps kept in the five iterations before is −0.04 ± 0.04. The third of iterations with the fewest kept steps (26 on average) and the third with the most (140) differ by 0.0014 ± 0.0020.
- **Critic accuracy is slowly falling.** Explained variance on new games was 0.328 at g1600–g1799, 0.316–0.322 until g2399, 0.299 at g2600–g2739 and 0.311 at g2740–g2909.
- **Drift by beta.** A lower beta is a weaker policy-clone penalty, so the policy drifts more.

| Beta | Drift at the step-32 check: median, 90th percentile, maximum | Checks above 0.003 |
| --- | --- | ---: |
| 0.5 (g2150–g2399) | 0.0014, 0.0018, 0.0029 | 0% |
| 0.25 (g2600–g2909) | 0.0022, 0.0028, 0.0043 | 7% |
| 0.125 (*estimate*) | 0.0035, 0.0045, 0.007 | most |

The drift stops growing after step 32. Halving beta from 0.5 to 0.25 raised it by 60% and changed the training-row gain from 0.022 to 0.024.

### 4.3 Should the cap go to 0.005 with a 0.125 floor?

- **They are a pair.** With the cap at 0.003, a beta of 0.125 would put most phases over the cap at the step-32 check. The phase stops there, and after three stops in five the beta rule doubles beta. Beta would move between 0.125 and 0.25 and phases would be cut at 32 steps. This is the more frequent cap hit you describe.
- **0.005 is not unstable.** PPO moves the policy 0.0074 per iteration, and its learning-rate controller reads the KL before the value phase, so it does not react to value-phase drift.
- **It is drift without a direction.** The value phase does not optimise the policy. At beta 0.125 a full phase would move the policy about 0.0035–0.0045, half or more of PPO's own step, against 0.0022 now.
- **There is no gain to pay for it.** Kept value steps do not improve the critic on new games, and the last halving of beta bought no measurable fit. The movement that a lower beta allows is better spent on the distillation phase, where it has a direction.

### 4.4 The value-phase loss today, by size

The sizes below are means on the held-out games of g2600–g2909. The reports give the early-stop loss (1.183) and the utility error (0.160). How the remaining 1.02 divides between win, rank and total VP is an *estimate*.

| Term | Weight | Size | Share of the early-stop loss |
| --- | ---: | ---: | ---: |
| Win cross-entropy | 1 | about 1.01 | 85% |
| Utility squared error | 1 | 0.160 | 14% |
| Rank | 0.25 | about 0.012 | 1% |
| Total VP | 0.10 | about 0.001–0.002 | 0.1–0.2% |
| **Early-stop loss** (the four above) | | **1.183** | 100% |
| Local-scoring auxiliary (training only) | 0.05 owner, 0.01 VP | 0.056 | 4.7% of the main loss |
| of which its two VP terms | | about 0.0001 | 0.2% of the auxiliary |

Three things follow.

- **The baseline PPO uses is mostly the win head.** The utility output is the fixed sum 1.0 × win + 0.1 × placement + 0.001 × VP of the three outcome heads, plus a learned correction. The utility loss does not train the three heads. It trains only the correction.
- **Almost nothing asks the network to count points.** The total-VP term and the auxiliary's VP terms are together under 0.3% of the loss. The points decide who wins.
- **The auxiliary is small.** It is 0.056 beside 1.18, and its heads' gradient norm is about 0.006 against a clip of 1.0. The report does not measure how much of the trunk's gradient it supplies.

### 4.5 Local-scoring auxiliary: what the held-out history shows

The auxiliary asks, for each of the 49 industry slots and each link valid in the era (31 in Canal, 38 in Rail): which seat scores it when the era ends, or none, and how many VP each seat gets from it. Its held-out check runs every fifth iteration. There are 230 checks between g1604 and g2909.

To have something to read those checks against, I scored a rule with no learning in it on the same row window: **copy what is on the board, and give every other slot or link the average result for that quarter of the era.** The rule uses 25 games of the g1190 model recorded on this laptop (9,777 decisions, 139.5 VP per seat against 135.9 in the lab at g2909). I stopped the recording there when you asked for no large local runs.

| | Canal slots | Canal links | Rail slots | Rail links |
| --- | ---: | ---: | ---: | ---: |
| Owner cross-entropy: head | 0.83 | 0.65 | 0.72 | 0.82 |
| Owner cross-entropy: rule | 0.97 | 0.74 | 0.73 | 0.88 |
| Wrong seat on entities that score: head | 66% | 54% | 32% | 38% |
| Wrong seat on entities that score: rule | 67% | 55% | 35% | 47% |
| VP error per scoring entity and seat: head | 1.19 | 0.96 | 1.00 | 1.11 |
| VP error: rule | 1.04 | 0.85 | 0.63 | 1.06 |
| VP error: always answer zero | 1.19 | 1.20 | 1.26 | 1.22 |

Head: self-play groups, mean of the 230 lab checks. Rule: first 128 decisions of each era of the local games, with its rates taken from the other 24 games. An entity is a slot or a link. It scores if it gives some seat VP when the era ends.

**Weaknesses**

1. **On the entities that score, the head names the wrong seat as often as the rule.** In the Canal era the two are equal (66% and 67%, 54% and 55%). In the Rail era the head is a little better, mostly on links, where it can see who owns the industries at both ends. Its lower cross-entropy in the Canal era comes from the entities that stay unscored.
2. **The VP part does not work.** The head's VP error is worse than the rule's in all four groups, and for Canal slots it equals always answering zero. At an average Rail decision 69% of the era's industry VP sits on tiles that are already flipped, with the VP printed in the input, and the head still misses it. The cause is in 4.4: the VP terms are 0.2% of the auxiliary. The weight is 0.01, the target is divided by 20, and the loss is a mean over four seats of which three are zero by rule.
3. **Nothing has changed in 1,300 iterations.** Wrong-seat rates were 63%, 50%, 29% and 35% at g1600–g1799 and are 64%, 52%, 29% and 34% at g2800–g2909. The training-row cross-entropy moved from 0.606 to 0.590 for slots and from 0.540 to 0.536 for links.
4. **The heads cannot tell one empty slot from another.** A head sees a slot through that slot's own input features and those of the other slots in its town. An empty slot in an empty town looks the same anywhere on the board. The trunk enters as one 64-wide vector that is added to every slot alike. So the head can learn how many builds are still to come, but not who is placed to build where, and the trunk is never asked for that either.
5. **Canal and Rail are one problem to it.** This is the weakness you named. One output layer, one weight and one reported sum (`slots_ce_sum`, `links_ce_sum`) serve two different tasks:

   | | Canal | Rail |
   | --- | --- | --- |
   | Share of final VP scored at the era's end (g2909) | 26% (35.7 VP per seat) | 74% (100.2) |
   | Slots that score at the era's end | 18 of 49 | 41–42 of 49 |
   | Links that score | 10.5 of 31 | 37.7 of 38 |
   | Entities that end unscored | slots 63%, links 66% | slots 15%, links 1% |
   | Entities still open at an average decision (local games) | slots 87%, links 79% | slots 39%, links 42% |
   | Critic's explained variance on held-out rows | 0.12 | 0.47 |

   An open entity is an empty slot, an empty link or a tile that is not flipped: its result is not on the board yet. A settled entity is a flipped tile or a built link. In Canal the question is mostly whether anyone will score a slot. In Rail almost every slot and link scores, and the question is who. The shared output layer must get the difference from the trunk's era flag through the one shared vector. The pooled sums hide which era is learning, and the weight cannot be set by era.

6. **Canal rows never see the Rail-end result.** The target is the current era only. In the Canal era that is 26% of the points, and it is where the critic is weakest.
7. **The heads lose what they learn in rejected intervals.** They are restored together with the model. They trained on kept steps in 43% of iterations, 15 steps per iteration on average.
8. **The check cannot be read.** It takes 512 rows from 3 games. Each self-play group is the first 128 rows of one era of one game, so all its rows share one target map and early decisions are over-represented: on all decisions the rule scores 0.79 for Canal slots, on that window 0.97. There is no reference to compare with, and the VP error is averaged over the three seats that are zero by rule. From one check to the next the owner cross-entropy of a self-play group varies by 0.05–0.13.

### 4.6 Local-scoring auxiliary: the design

KataGo, which you named as the model for the value network, gives its ownership target the same weight as the game result (1.5 and 1.5), and its ablation without the ownership and score targets lost a factor of 1.65 in learning efficiency ([Wu 2019](https://arxiv.org/abs/1902.10565), sections 4.1 and 5.2). The local-scoring target is this project's form of the same idea. It needs heads that can express it and a weight that reaches the trunk.

1. **Each slot and link reads its own part of the trunk.** New trainer-owned heads replace the present ones. They are not exported:
   - `shared = GELU(Linear(640 → 256))` of the trunk output and the seat descriptors.
   - `slice = Linear(256 → 118 × 16)`: sixteen numbers for each of the 49 slots and 69 links.
   - `hidden = GELU(Linear(64 + 16 → 96))` of the entity's own vector and its slice, one layer for slots and one for links.

   Now "who is placed to build in this slot" has a path, and its gradient asks the trunk to hold that information. The heads grow from 91,000 to about 670,000 parameters. Games and search cost the same, because the heads are not in the exported model.
2. **Canal and Rail get separate outputs, weights and sums.** One owner output and one VP output per era on the shared hidden layer. The loss, the training report and the check each carry a Canal and a Rail number, split again into open and settled entities. The era weights start equal.
3. **The VP term gets real weight.** Divide the VP target by 5 instead of 20 (5 VP is the mean of a scoring tile or link), sum over the four seats instead of taking the mean, and give it the owner term's weight. At today's weight the VP terms become about a seventh of the auxiliary instead of 0.2% (*estimate*).
4. **Start at today's size, then raise the weight.** The new heads start through the existing warm-up: heads only for two fitted iterations and 32 steps, then the trunk gradient ramps over five. Keep 0.05 for that. Then raise the owner and VP weights to 0.15, and then to 0.5, each at a committed boundary. At 0.5 the two kinds together stand to the win term as KataGo's ownership stands to its game result. The early stop reads the main loss only, so the auxiliary cannot buy a kept step, and the KL cap still bounds the policy drift.
5. **Canal rows also predict the Rail-end result.** A second pair of outputs for Canal rows: who scores each slot and link at the end of the game, and for how many VP. The labels are already in each game's record. Start at half the Canal-end weight. This is the least certain item: early in the Canal era the Rail-end map is mostly base rates. It is last in the build order for that reason.
6. **The heads keep what they learn in a rejected interval.** They are not exported, so a head step is harmless when the model's step is undone. This doubles their training and matters most while the new heads warm up. It needs the warm-up counter to count attempted head steps.
7. **Make the existing check readable, at the same cost (0.06 s).** Take one row per game, era and quarter of the era from all held-out games (the validation games of 4.8) instead of the first 128 rows of one game. Report open and settled entities apart, per era. Report the copy-the-board rule beside the head, and the VP error on the scoring seat beside the always-zero error.

The total-VP head belongs to the same repair. Raise its weight from 0.10 to 0.5 together with item 4. Its term is about 0.1% of the loss today, and its gradient into the layer it shares with the win head is about one fortieth of the win head's (*estimate*).

### 4.7 Held-out early stop: what the logs show

- **The first check is a coin flip.** Over g2600–g2909 the first 16-step interval is kept in 49% of phases. Its mean change of the held-out loss is +0.0011 ± 0.0005, and the spread from phase to phase is 0.0096.
- **What is kept looks like chance.** Kept phases lower the held-out loss by 0.0091 on average, and rejected intervals raise it by 0.0059. Keeping the better half of changes that scatter by 0.0096 around zero gives −0.0077 and +0.0077. This agrees with 4.2: kept steps do not make the critic better on new games.
- **The formula is not the weakness.** The win part and the utility part of a rejected change move together (correlation 0.95). A check on the utility error alone would have reversed 7% of the rejections and 9% of the keeps.
- **The number of games is.** The check has 32 games, and every row of a game carries the same result. From one iteration to the next, explained variance on the 16 held-out games of an iteration varies by 0.08, and on the 320 new games by 0.02.
- **A critic that has not been updated is as accurate.** Explained variance on new games is 0.311 right after a kept phase, 0.310 after one, two or three phases with nothing kept, and 0.313 after four or more. So it costs nothing measurable to train on a game one iteration later.
- **The check is cheap.** One check of 12,400 rows takes 0.1 s.

### 4.8 Held-out early stop: the design

The rule stays: the same four terms on real outcomes, a check every 16 steps, and any increase rejects the interval. The games change.

- **Validate on new games before they train.** Each iteration, set aside a quarter of the new games (80: 56 self-play, 24 against the pool). The check uses only these. In the next iteration they are ordinary training games. Today's permanent 5% hold-out goes.
- **What it costs.** The phase trains on 560 games instead of 608, about 219,000 rows instead of 238,000, and plans 54 steps instead of 59. The phase uses 15 of its planned steps on average and all of them in 9% of iterations, so the smaller set is rarely felt. The check reads 31,000 rows instead of 12,400, which adds under 1 s per iteration (*estimate*).
- **What it gives.** The check has 80 games instead of 32, and none of them has been used by an earlier check. If game-to-game differences cause most of its noise, the noise falls to about 0.6 of today's (*estimate*). No game is lost to the critic: today 5% never train it.
- **The check never uses the auxiliary or the blended targets of 4.9.** It measures what PPO needs: the main heads against what really happened.
- **A larger form, if 80 games still decide like a coin.** Validate on all 320 new games and train on the two iterations before them. The noise falls to about a third. The delay itself costs nothing by the staleness numbers above. It costs one more iteration of critic rows on disk (1.8 GB) and about 5 s to read them, unless the previous iteration's rows stay in memory (*estimate*).

A sharper check may keep fewer first intervals, because the average first interval makes the held-out loss slightly worse. That is the check working. It also makes the share of kept intervals a better signal when the loss changes in 4.6 or 4.9. An interval that raises the held-out loss by 0.003 is kept 38% of the time by today's check, 31% of the time with 80 games and 16% with 320 (*estimates*).

Between g418 and g650 this project ran longer windows (8 and 16 iterations), standard-error tolerances on the early stop, and phases every eighth iteration, and returned to the two-iteration recipe at g660. I do not propose those again.

### 4.9 A less noisy target for the win and utility heads

Each fit sees 238,000 rows but only about 2,400 independent outcomes (608 games, four seats), because every row of a seat carries the same final result. That, and not the number of steps, is what the critic cannot get past.

- For every critic row, form the bootstrapped return along the seat's own decisions with lambda 0.95, from the predictions PPO already freezes before its update: the usual GAE return for the utility, and for the win share the same weighted average of later predictions, ending in the real result. A forced row takes the return of the seat's next decision. Store both beside the outcome.
- Train the win head and the utility output on half the outcome and half the bootstrapped return. The rank, total-VP and local-scoring heads keep their outcome targets, and so does the search's reward.
- Rows of the previous iteration keep the returns stored when they were collected.

This revises the earlier version of this plan in two ways. The win head is included, because the utility output is mostly the win head and its loss trains only the correction (4.4). And the early-stop check stays on real outcomes: your g1730 ablation measured exactly that, and a head fitted to the lambda 0.8 return predicted the real utility of 100 held-out games better than a head fitted to the outcome (0.16779 against 0.16845 at 256 steps, 0.16823 against 0.16964 at 512).

The risk is that the value network partly learns from itself. The half share of real outcomes and the outcome-trained rank and VP heads anchor it. It costs no time.

### 4.10 Two time savings

**Skip the two full-buffer metric scans** except every tenth iteration, or use a 16,000-row sample. With `skip_utility_qualification` they only feed the report. A phase that keeps steps takes 44 s against 20 s for one that keeps none, and about 19 s of that difference is these scans.

**Pause the search-value head while the teacher runs at weight 0.** Over the 84 held-out checks from g2199 to g2909 its corrected value was worse than the main value in 49, by 0.0004 ± 0.0002 in squared error (the main value's error is about 0.17). After the change of lambda to 0.95 at g2880, five of six checks are slightly better (+0.0004). At weight 0 its accuracy changes nothing in training, and it costs 5–9 s per iteration. The blended target above puts the same idea, learning from bootstrapped returns, where there is sixty times more data.

### 4.11 Where the value-loss changes touch the code

| Place | Change |
| --- | --- |
| `learning/local_scoring_heads.py` | New head contract: shared layer, per-entity slice, one output pair per era (and per horizon for item 5). |
| `recovery/local_scoring_auxiliary.py` | Loss by era with the new VP scaling; sums by era and by open or settled, read from the owner and flipped inputs of the row; the check's row choice and reference rule; head steps kept on rejection. |
| `recovery/local_scoring_targets.py` | A Canal row can also return the Rail-end map. |
| `recovery/local_scoring_adoption.py`, `local_scoring_warmup.py` | Adopt the fresh heads at a committed generation; count attempted head steps. |
| `recovery/critic_holdout.py`, `actor_value.py`, `ppg_replay.py` | A quarter of the new games validate; the cached rows of the previous iteration all train. The cache already holds the held-out rows. |
| `recovery/value_phase.py`, `updates.py` | Soft win target and blended utility target in training only; total-VP weight; the check keeps `value_losses` on outcomes. |
| `recovery/value_preservation.py` | Source of the frozen win and utility predictions for the bootstrapped returns. |
| `recovery/ppg_g210.py`, a new `training_revision_gNNNN.py`, `continue-rust-g1491.py` | New value-phase contract fields and the flags that adopt each step. |
| Last | Re-record the execution identity. |

## 5. Collection and time (concern 4)

[RUST_AND_LAB_COLLECTION_PERFORMANCE_PLAN.md](../docs/RUST_AND_LAB_COLLECTION_PERFORMANCE_PLAN.md) holds the detailed performance work. This section adds what the g2739 and g2909 reports show about cost.

### Where a collection's CPU goes (iteration 2909, 64 CPUs, 81 s)

| Teacher table, median of 36 | Seconds | Share |
| --- | ---: | ---: |
| 16 long searches (1.98 s each) | 31.8 | 55% |
| 49 short searches (0.25 s each) | 12.2 | 21% |
| Waiting on the trainer after a search | 13.4 | 24% |
| Whole table | 58.0 | |

- **A table costs about 0.8 s of collection time per iteration.** Iterations with 36 tables take 89.3 ± 0.9 s to collect, iterations with 32 take 86.0 ± 0.4 s.
- **The waiting is 484 lane-seconds per iteration** (553 at g2739). Most turns wait 0.03 s and a few wait up to 14 s, in the middle of the game, while the trainer process is busy with ordinary games.
- **One simulation costs 2.94 ms:** 1.21 ms in the network and 1.73 ms in native search and rules.
- **Ordinary games are mostly protocol.** The collection time fits a simple model in which the 320 ordinary games need about 3,300 core-seconds in effect, about 10 per game (*estimate*). Of the work the reports show, the network takes 66 s and the rules engine with feature encoding 86 s. The rest is Python sampling (164 s), the Python-to-Rust protocol (506 round trips and 58 MB per game), the hand-off of each game's result (729 s of shared writes), publication (237 s) and replay preparation (157 s).

### What the model says (*estimates*)

| Setup | Collection time |
| --- | --- |
| Today, 32 tables | 81 s |
| No teacher tables | about 52 s |
| 32 tables without the waiting | about 74 s |
| 40 tables without the waiting (the design) | about 80 s |
| 40 tables without the waiting, ordinary games at half their cost | about 54 s |
| The same, with 640 ordinary games | about 80 s |

### Design

1. **Search workers never wait for Python.** Drain each table's output as raw bytes as it arrives, or enlarge the pipe, and verify afterwards. This frees about 480–550 lane-seconds per iteration and is what pays for tables 33 to 40. It must come before the short labels, which quadruple the labelled turns the trainer has to accept.
2. **Admit teacher tables while the last ordinary games finish.** Tables are published 8–17 s before the ordinary pool ends, and admission then takes about 5 s after it.
3. **Stage the label rows on the GPU once per iteration.** From g2740 to g2909 the teacher term took more than 10 s in 61% of iterations (31 s on average, against 1.7 s otherwise), and PPO took 70 s instead of 40 s in those iterations. Consecutive iterations of one process differ (29.3 s at 2908, 1.2 s at 2909). The term collates its rows from Python objects on the host at every step, and one wide row pads the whole batch. The distillation phase needs the staged rows anyway.
4. **Take cleanup hashing off the path between iterations.** It is 75–84 s every fifth iteration, 7% of wall time.
5. **Halve the cost of an ordinary game.** This is the largest reserve in the system. Only about 150 of the visible core-seconds are rules and network. Write each game's training features from the worker straight to the shared buffer, and move action sampling into the worker. At half the cost, the same collection time holds 640 ordinary games instead of 320, which doubles what PPO learns from per iteration. This is the largest piece of engineering in the design and the last in the build order.

## 6. Exploration and strategy coverage

The policy is narrow. It takes 0.6% of its legal cotton-mill builds and 6% of its pottery builds, against 12–26% for the other industries, and its entropy has fallen from 0.57 to 0.53 since g1600. Distillation pushes toward sharper choices, so three things keep the policy from narrowing further:

- **Labels on every turn explore.** Each search evaluates 8 candidates drawn with Gumbel noise, including moves the policy rarely plays, and the target raises a rare move when its searched value is higher. With five times the labels this happens on five times as many positions.
- **Cotton tables stay** at their present share of the teacher tables and PPO games.
- **Arm the entropy floor when the phase is switched on.** The floor controller exists and is off. Set its floor at 90% of the entropy measured before the switch, so a fall in entropy raises the entropy weight instead of going unnoticed.

## 7. One iteration, today and in the design

| | Today | Design |
| --- | --- | --- |
| Ordinary games | 320 | 320, then 480–640 after step 9 |
| Teacher tables | 32 | 40 |
| Labels made | 993 | about 5,000 |
| Label presentations | 4,416 (rows repeat 1.76 times) | 24,576 (no repeats inside an iteration) |
| Policy movement: PPO | 0.0074 | 0.0074 |
| Policy movement: teacher | about 0.0002 (*estimate*) | 0.002 |
| Policy movement: value phase | 0.0009 on average, cap 0.003 | the same cap |
| Critic training games | 608, plus 32 never trained | 560; the 80 validation games train one iteration later |
| Early-stop games | 32 | 80, all new |
| Local-scoring auxiliary, size beside the main loss of 1.18 | 0.056 | 0.056 at first, then about 0.2 and 0.65 (*estimate*) |
| Collection | 90 s | about 82 s |
| PPO update | 60 s | about 41 s |
| Distillation phase | inside PPO | about 9 s |
| Value phase | 36 s | about 22 s |
| Cleanup, rating, other | 43 s | about 31 s |
| Whole iteration | 229 s | about 185 s (*estimate*) |

## 8. Build order

Each step is one change that can be adopted at a committed boundary.

| Step | Change | Effect on time per iteration | Needs |
| --- | --- | --- | --- |
| 1 | Stage label rows on the GPU; value-phase metric scans every tenth iteration; cleanup hashing off the path; pause the search-value head | about −45 s | Python only |
| 2 | Search workers never wait; teacher admission during the ordinary tail; 40 tables with 4 validating before they train; held-out tables dropped | about −8 s | Worker output handling, recipe revision |
| 3 | Short-search labels on, label scale 0.20 for both, targets rebuilt from the current policy, 8-iteration pool, parity check on a sample | up to +4 s | Recipe revision, `teacher_buffer.py`, target builder |
| 4 | Distillation phase: 4 run-and-restore iterations, then on at 0.001, 0.0015, 0.002; entropy floor armed | about +8 s, and the in-PPO term goes | Section 3 |
| 5 | Early stop on a quarter of the new games, which train one iteration later (4.8) | under +1 s | `critic_holdout.py`, `actor_value.py` |
| 6 | New local-scoring heads with a per-entity trunk slice and separate Canal and Rail outputs; VP term rescaled; sums and check by era; heads keep rejected steps; weight stays 0.05 (4.6, items 1–3, 6, 7) | 0 | Head contract and its adoption |
| 7 | Blended target for the win and utility heads; the check stays on outcomes (4.9) | 0 | Value-phase targets |
| 8 | Local-scoring weight 0.15, then 0.5; total-VP weight 0.5; then the Rail-end result in Canal rows (4.6, items 4 and 5) | 0 | Recipe revisions |
| 9 | Ordinary games at half their cost; then 480–640 games per iteration | −26 s, or twice the games | Worker and protocol |

**Status (2026-11-17):** steps 1, 5, 6, 7 and 8 are implemented for adoption from g3140 with `--value-revision`; see [G3140_VALUE_REVISION.md](G3140_VALUE_REVISION.md). One change from the table: the search-value head is not paused, so step 1 saves about 5–9 s less than estimated. Step 9's sampling half is implemented: the CPU worker plays each ordinary game, with the same RNG draws and the publisher's audits unchanged. The run stays at 320 games. Workers writing features straight into shared memory, and the publisher's quadratic decode, are still open; see [STEP9_ORDINARY_AUTOPLAY.md](STEP9_ORDINARY_AUTOPLAY.md).

Steps 1, 5, 6, 7 and 9 do not depend on the others, so the value-loss steps 5 to 8 can go before the teacher steps 2 to 4. Step 3 needs step 2, step 4 needs step 3, and step 8 needs step 6. Step 5 is best taken before steps 6 to 8: with a sharper early stop, the share of kept intervals shows sooner whether a change to the loss helps or hurts.

## 9. Risks and what to watch

These use numbers the reports already contain.

| Risk | Sign | Response |
| --- | --- | --- |
| Distillation moves the policy the wrong way | The self-play health check fails, or VP per seat falls below its band | Step the budget back one level; the multiplier and the phase snapshot make this immediate |
| The policy narrows | Raw entropy at the floor for several iterations; cotton and pottery builds fall | The armed floor raises the entropy weight; lower the budget if it stays there |
| Labels are still memorised | The label loss falls but the held-out KL on the validation tables rises by as much as the policy moves | More games per label pool before more budget: more tables or labels on ordinary-game positions (section 10) |
| The critic follows itself | Explained variance of the PPO baseline on new games falls after step 7 | Lower the bootstrapped share from a half to a quarter |
| The larger auxiliary pulls the trunk away from the main loss | Kept value steps fall toward zero for many iterations after step 8, or the baseline's explained variance on new games falls | Step the auxiliary weight back one level |
| The larger auxiliary moves the policy | The value phase's KL at the step-32 check rises and the beta rule raises beta | Step the weight back one level; the cap of 0.003 holds meanwhile |
| The new heads do not learn the open entities | After step 6 the owner cross-entropy on open entities stays at the copy-the-board rule's | Do not raise the weight; check that the trunk slice receives gradient |
| Short labels are too noisy at 0.20 | The short labels' held-out KL rises faster than the long labels' | Lower their label scale to 0.15 |

## 10. After this design

Two larger steps follow naturally once the above is running. They are listed for direction, not as part of the build order.

- **Labels on positions of the ordinary games.** A search on a sampled position of a PPO game gives a label with no game around it, and the labels then come from 320 games per iteration instead of 40. With short searches only, the search time of today's 32 tables would give about 11,000 labels per iteration (*estimate*).
- **More of the CPU to labels.** One short-search label costs about the CPU of two PPO decisions, and a long one about fourteen (*estimates*). If distillation proves to be the faster teacher, games can move from PPO to search at that exchange rate.
- **A critic that adds up the local predictions.** Final VP is the VP a seat has now plus what its slots and links score at each era's end. Once the local-scoring heads predict that well, the total-VP and win heads could take their sums as an input. That needs the heads in the exported model, so it is a change to the model and to inference, not to the loss.

## 11. Estimates and open points

- The collection model (3,300 core-seconds of ordinary work, 52 s with no tables, the times in section 5) is fitted to two measured points, 32 and 36 tables. It may not hold far from them.
- The share-to-movement figures assume right-angle gradients and a fixed Adam scaling.
- The teacher's pull uses policy movement on PPO positions as the reference, and the pull on trained rows compares different batches at the first and last step. The validation tables of 2.5 replace both with exact numbers.
- The noise comparison in 2.2 assumes that a candidate's value error falls with the square root of its visits.
- The short-label count, the phase's cost, the iteration time of the design, and the drift at beta 0.125 are estimates.
- The value-step result covers budgets between about 5 and 28 steps per iteration over five iterations. It does not show what a much longer or a zero budget would do.
- The copy-the-board rule of 4.5 comes from 25 games of the g1190 model played on the laptop. The lab's games at g2909 are slightly slower in the Canal era (18 scoring slots at its end against 20), and the lab check reads learner rows only. The comparison shows the size of the gap, not its third decimal.
- The reports do not give the win, rank and total-VP terms apart, and they do not measure how much of the trunk's gradient the auxiliary supplies. The split in 4.4 and the weights 0.15 and 0.5 in 4.6 rest on estimates and on KataGo's ratio. The early stop and the KL cap are what protects the run while the weight rises.
- The noise of the 80-game check is an estimate that assumes game-to-game differences cause most of it.
- Whether the new heads beat the rule on open entities, and whether that makes the critic better on new games, is not known until step 6 has run. Nothing in the logs can show it in advance.
- The cause of the slow teacher-term steps is a hypothesis; logging the widest row and the time of each step confirms it.
- No lab run was made for this review.

## Appendix: reproducing the numbers

The analysis scripts and the extracted per-iteration table are in `tmp/training-review-g2740/` (not tracked by git). Run them with `artifacts/p6e/Scripts/python.exe`.

| Script | Use |
| --- | --- |
| `extract.py results series.json`, then `slim.py series.json slim.json` | Flatten every `restart-3` export into one table (about 3 minutes) |
| `trend.py slim.json 50 2600` | Windowed PPG, PPO, teacher, held-out and timing tables |
| `ppg.py slim.json 2600 2909` | Value-phase endings, interval statistics and KL by beta |
| `tables.py <report.json>` | Time split of the teacher tables in one full report |
| `auxextract.py results aux.json` | Local-scoring report and value-phase loss parts of every iteration (about 3 minutes) |
| `auxhist.py aux.json 200` | Local-scoring held-out and training-row history by window |
| `earlystop.py aux.json 2600 2909` | Which loss parts decide the early stop; win-head calibration |
| `baserates.py sim`, `baseline_window.py sim 128` | The copy-the-board rule on the 25 recorded local games in `sim/` |
| `simulate.py sim <first seed> <games> <workers>` | Records local games with the g1190 model (about one minute per game and core). Not needed again: the 25 games are kept. |
