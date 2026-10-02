# Review of the rejected network/Scout revision

Reviewed 2 October 2026. Candidate `bank-fcab56543a2c86f30c66`, package source `1.4.20-r12`. This is a diagnosis and proposed follow-up, not a promoted game/training release.

**Subsequent user decision:** stop strength testing and select the preceding advanced bank, `bank-3dc65253a610191d66a5`, as **v4** for game and training. That bank already contains productive-development stall reset and opening-biased candidate selection. The repairs proposed below are retained as research notes; none were applied to the selected v4. Statements about the unchanged installed/training version below describe the state at the failed screen.

## Result and evidence

All 24 games completed at the same nominal one-second/four-worker decision allowance. Three setups/style pairs per control, each with four seat rotations. The previous revision is the last advanced-opening bank before the network/Scout additions, `bank-3dc65253a610191d66a5`; economic control is `bank-f5107c644af2fbf4ac99`.

| Comparison | Wins, new/control | Mean VP, new/control | Mean rank, new/control (lower is better) | Outcome |
| --- | --- | --- | --- | --- |
| Previous advanced revision, 12 games | 5 / 7 | 83.458 / 85.458 | 2.667 / 2.333 | Fail |
| Economic revision, 12 games | 8 / 4 | 92.542 / 87.375 | 2.458 / 2.542 | Pass |

The user requires improvement on **every metric against both controls**. Overall promotion fails. Game UI 1.4.19 / AI 1.4.18 and admitted training bank `bank-340a20d72b23c652c267` remain unchanged.

Evidence is retained in `artifacts/economy-v3/smoke-8/report.json`, its 24 game traces, and `diagnosis.json`. The [diagnosis script](../scripts/diagnose-economy-v3.py) re-applied every recorded choice through the pinned rules host. **All 24 terminal results matched exactly**, and each era's tile/link award matched the official score change. Merchant bonus VP was checked separately because it is banked immediately. No new AI decisions, training samples or strength games were generated during replay.

This supports a policy/budget diagnosis. It does not establish that every rule is correct, or that one added feature alone caused the loss. Each style has only four seat-games per control, from one setup. Style rankings need broader confirmation; the failed screen must not be replaced by favorable seeds.

## 1. The one-second search is mostly running out of useful time

Against the previous revision:

| Work per decision, averaged over all four private roots | New | Previous |
| --- | --- | --- |
| Host/search nodes, including incumbent construction | 41.96 | 49.16 |
| Completed MCTS simulations | **0.250** | **0.701** |
| Mean observed decision latency | 1.156 s | 1.123 s |

Completed simulations fell about **64%**. The new average is one completed simulation per four decisions across the entire four-root forest. Complete heuristic incumbents still exist, so the AI can play legally, but many decisions lack enough completed, matched search evidence to challenge them. The older implementation also has little support under this short clock; it is not an adequate performance target in the long term.

The new revision adds expensive wild-hand investment quotes and repeated public rail-demand forecasts. `ScoutBase` is computed once per proposal batch, but the extra incumbent branch calls it again directly. `RailDemand` prepares reachability and hashes a full observation for each actor/sub-quote. `HeuristicBatchQueue.Key` includes growing public history, and `Drain` evaluates distinct queued requests sequentially in one caller. Reentrant execution fixed a deadlock risk; it did not make different CPU evaluations parallel or inexpensive.

**Confirmed scheduling defect:** [FastIncumbents](../src/BrassMaskedSearch/MaskedSearch.cs) constructs industry candidates, then new rail candidates until the 55% clock threshold, then Scout candidates until 60%, and only afterward checks the older **45% development/setup** and **40% ordinary sale** thresholds. New early work can therefore make necessary setup/conversion branches ineligible before they are reached. Final sales still have a critical fallback. This explains how adding legal candidates can reduce coverage of more useful plans. The precise VP effect of this defect still requires an ablation.

### First proposed fix: preserve useful candidate coverage and search time

1. Prepare a cheap permitted-state summary once: connectivity, reachable buyers, actual market quotes, relevant mat fronts, affordable upgrades and current beer commitments. Reuse it across first-choice/card variants and complete-plan scoring.
2. Construct one complete legal incumbent immediately. Prioritize urgent sales/cash repair, then funded development-plus-build and industry-plus-sale candidates **before** proactive rail/Scout widening. Assign different mandatory families to private workers; avoid four owners independently recreating the same extended greedy plans.
3. Use one ordered candidate budget, rather than later branches with earlier time thresholds. Reserve approximately half the one-second clock for actual search. If an incumbent operation is unusually expensive, preserve the completed legal fallback and skip discretionary widening.
4. Quote Scout only when the current legal build set/card support is materially poor. Cache its result for that exact state and discard multiset. Do not repeat a full wild-hand scan merely to decide whether to construct the Scout incumbent.
5. Cache rail/beer demand for one prepared state. For economic quote caches, use an explicit list of the fields that influence that quote, retaining the observer/own hand whenever required. Exclude history only after verifying that the particular computation does not read it; belief/information-tree keys retain the information needed by search.
6. Allow distinct CPU requests to execute on the already admitted private workers, with a short lock only for cache/pending ownership. Coalesce identical inputs. Do not serialize all different rule-based evaluations behind one drain leader or add extra uncontrolled threads. Keep two workers in-game and four in the lab.
7. Log incumbent time, setup/development coverage, quote time, queue wait, completed simulations, matched-plan support and actual retirement latency **per worker**. Preserve these fields when merging the forest. Node count alone is not a strength measure.

The NN keeps its central dynamic CPU/GPU batching service, raw Q and Masked ISMCTS. The proposed CPU heuristic scheduling repair does not change neural targets, masks, model weights or search horizon.

## 2. The regression is uneven; Zero-Loan collapses in Rail

| Style, four seat-games each against previous | New VP | Previous VP | Difference | Wins, new/previous |
| --- | --- | --- | --- | --- |
| Adaptive Advanced Book | 79.50 | 83.25 | -3.75 | 0 / 1 |
| Coventry Goods Opening | 85.25 | 91.25 | -6.00 | 2 / 1 |
| External-Beer Three Cotton | 91.75 | 77.50 | +14.25 | 2 / 0 |
| Worcester Iron-Cotton | 89.25 | 83.00 | +6.25 | 1 / 1 |
| **Zero-Loan Industry** | **70.25** | **93.75** | **-23.50** | **0 / 2** |
| Advanced Pottery Variants | 84.75 | 84.00 | +0.75 | 0 / 2 |

Zero-Loan's era-end board awards:

| Award per seat | New | Previous |
| --- | --- | --- |
| Canal industries + links | 36.25 | 36.25 |
| Rail industries | **18.50** | **30.25** |
| Rail links | **14.75** | **25.75** |
| Rail merchant bonus VP | 0.00 | 0.75 |

The 23.50-VP deficit is accounted for by **11.75 fewer industry points, 11.00 fewer link points and 0.75 fewer bonus points in Rail**. It cannot be explained simply by building fewer rails: the routes and the industries supporting their points were both worse.

The new Zero-Loan style averaged 6 Canal network actions versus 4.5, reached the era boundary with 13.75 cash versus 22.75, and then borrowed 1.75 times in Rail versus 0.5. Rail coal builds rose from 1.5 to 2.75 per seat, iron builds fell from 2.5 to 0.75, and level 3+ builds fell from 3.5 to 2.25. It finished Rail with **4.5 unsold VP** versus zero. These observations show that the intended market-financed strategy was not reliably carried through to the better investments.

### Second proposed fix: compare productive industry sequences before expansion

* Give a funded cheap-iron development-plus-build plan an explicit completed candidate. When the intended tier needs two development actions, retain its bounded intermediate option and the required build/sale funding. Do not give a bonus for removing tiles without a feasible replacement.
* Compare the marginal resource receipt against an identified spending opportunity. A connected coal mine can be profitable; receiving money alone is insufficient justification for repeating coal while missing higher-value iron/goods/cotton investments. Keep shortage and low-income recovery exceptions.
* Quote links together with the industries they can actually support. Suitable empty slots are access options, never earned link VP. For a supplier, preserve the card-supported resource/market cycle before selecting a city merely for its capacity.
* Stop using one universal expansion cue as the main tie preference for all six books. Link specialists, sale-focused styles and resource-market styles have different concrete access requirements. These remain bounded conditional preferences, with shared legal/cash/finishability safeguards.
* Reserve working capital for a credible industry/sale sequence. Reassess loans and late cash against that sequence. Do not impose a permanent coal ban or a rigid requirement to develop regardless of iron price, cards and time.

## 3. The total loss is not solely a Rail-era problem

Across all six styles against the previous revision, the score accounting is:

| Mean VP difference, new minus previous | Difference |
| --- | --- |
| Canal industry/link award | -1.500 |
| Canal merchant bonuses | -0.625 |
| Rail industry/link award | +0.375 |
| Rail merchant bonuses | -0.250 |
| **Total** | **-2.000** |

Rail link VP increased overall (22.79 versus 20.58), but Rail industry VP fell (24.75 versus 26.58). The aggregate Rail award slightly improved; Zero-Loan's large Rail failure was offset by other styles. Removing all new rail logic would discard a benefit without addressing the unevenness.

The new version still spent 5.625 Canal actions per seat on connections versus 6.208 previously. The decrease is small and not uniform: Zero-Loan spent more. Sale/access necessity and the supporting industry sequence need to govern canals more directly.

## 4. Scout, double rails and beer need tighter exploitation checks

Against the previous revision, the new AI used Scout nine times and built using the resulting wilds 17 times. Thus Scout was often exploited; the data does **not** justify banning Scout. Nevertheless, the current quote adds wild access to the old hand before evaluating opportunities, rather than scoring the actual three-card discard and its subsequent completed plan. It can overestimate which options survive, and its scan costs search time even when no Scout is selected.

Double-rail actions increased only from three to four over the 24 candidate seat-games against the previous revision. Against the economic control, they increased from three to fourteen and produced much better Rail link points. This supports retaining complete double candidates when they are productive, while repairing their opportunity cost and setup budget.

All compared banks made only single-tile sales in this screen. That is a shared weakness, not a new-only regression. The external-beer books often sell immediately to claim a merchant bonus or avoid losing public beer, which is legitimate. Group sales still need an explicitly funded build-and-batch plan when safe own supply makes waiting profitable.

### Third proposed fix: rank complete exploitation plans

1. Score Scout with its **actual discarded cards** and one or two concrete follow-up builds/conversions. Compare it with the best currently supported investment. Use a cheap poor-hand trigger before the expensive quote. Keep wild access available without making Scout a default opening.
2. For double rails, apply both links and exact payments to a detached rules world. Compare their real scored/credible future points with the strongest industry-and-sale plan. Avoid choosing two separately attractive links without proving the complete funded pair.
3. Replace `OwnBeer > CommittedBeer` as the sole own-surplus test with a feasible one-time allocation for urgent sales: compatible merchant beer, reachable public supply with a contest discount, and own beer. Count every barrel once. Do not reserve beer indefinitely for an unreachable or already infeasible industry; do not spend beer required by a credible imminent batch.
4. Construct a brewery-supply candidate only when it enables a specific affordable profitable double rail or sale within the remaining actions. Generic rail demand is too weak to justify speculative brewing or its preparation cost.
5. Preserve urgent conversion candidates and merchant free-development valuation before discretionary widening. Near era end, prefer finishing existing sellable VP over another low-tier build unless the complete alternative actually scores better.

## Follow-up implementation and acceptance order

1. Start from the frozen previous revision as a control, retaining the rejected network bank as another diagnostic reference. Fix candidate scheduling, repeated quote work and CPU queue serialization first. Keep the one-second/four-worker lab allowance; do not conceal the regression by increasing the clock.
2. Test the same permitted observations from a small set of failed Zero-Loan/Adaptive Rail and opening decisions. Record candidate coverage and timing. Compare scheduling-only with old scheduling at the same budget. This is diagnosis, not a strength promotion.
3. Add funded development/industry conversion candidates. Verify their exact legal continuation, cash, beer and conversion deadlines with a few focused scenarios. Stop at the intended tier; respect non-developable pottery.
4. Add actual-discard Scout and feasible double-rail/beer allocation independently. Keep useful city/card/access guidance. Use component toggles to separate their effects; avoid tuning all six styles from four games each.
5. Freeze one final candidate and run **12 rotated fresh games against each required previous/economic control**, 24 total. Declare setups before running. Require strictly more official wins/first places, higher VP and lower mean rank against **each**. Keep this failed screen unchanged and exclude every comparison from learner replay.
6. Only a winner proceeds to fresh native/save/resource qualification, game installation at two workers with existing 2/5/12 clocks, and lab-bank admission/export at four workers/24 queued tables. Preserve the Pottery opponent holdout, unrestricted learner Pottery actions, fresh residual initialization and NN search/distillation settings.

The immediate recommendation is to repair **budget allocation and funded industry follow-through first**. There is evidence that selected network/Scout ideas help against the economic control, and evidence that their current implementation weakens some advanced books. The evidence does not call for replacing Masked ISMCTS or changing the game rules.
