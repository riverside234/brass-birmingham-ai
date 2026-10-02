# Advanced opening books: selected v4 and historical v3 experiments

**Current selection (2 October 2026):** the user stopped further strength testing and selected previous advanced bank **`bank-3dc65253a610191d66a5` as v4** for the game and training. It already includes productive-development stall reset and opening-biased build/setup candidates. The later `NetworkAndHand.cs` revision is excluded. Public style labels end in **v4**; existing `*-v3-masked-economy` IDs remain save/transfer aliases. Patch **1.4.20** changes the AI helper over UI 1.4.19; game workers remain two and lab workers four. The selection receipt is `training/opponents/evaluations/user-directed-v4-1.0.0/selection.json`; it records user selection, not a strength promotion.

The latest revision replaces all six traditional public styles with **Adaptive Advanced Book, Coventry Goods Opening, External-Beer Three Cotton, Worcester Iron-Cotton, Zero-Loan Industry, and Advanced Pottery Variants**. Five specialists share an economic fallback. Existing internal keys remain save-compatible aliases, not the old strategy definitions.

These implement the creator's advanced strategy families. Human tournament strength, tournament credentials and advertised win rates have not been independently established.

## Source evidence

Read on 2026-10-02: the creator's published video descriptions, the primary evidence requested. The videos were not transcribed. The creator is **随变不是随便哦**, [profile](https://space.bilibili.com/172386514/upload/video).

Where descriptions lacked a full sequence, the public comments were inspected, including rendered open shadow DOM, and the comment header clicked to load it. Guides 16, 17–18 and 20 stayed at “正在玩命加载…” (loading). No unavailable comment is represented as evidence. Sites, tile removals and card discards omitted from the descriptions are implementation decisions resolved through the rules host and current legal mask.

| Source | Verified description | Implementation adaptation |
| --- | --- | --- |
| [Guide 16: Coventry factory](https://www.bilibili.com/video/BV1NnfbBBEp5/) | Fourth-seat link opening into Coventry with a suitable hand; warns about punishment and giving away an iron site. No full sequence in the description. | Prefer Coventry access only for the last initial seat, an actual supporting card, affordable factory and nearby buyer. Otherwise use funded goods/iron. Goods II/VI targets and the rail transition are authored hypotheses. |
| [Guides 17–18: poor-hand Iron–Cotton](https://www.bilibili.com/video/BV1T4dUBsEGe/) | No-own-brewery iron/cotton; fourth-seat/bad-hand recovery with Worcester. No exact sequence. | Compare iron receipts and finishable cotton. Bounded Worcester preference only with its card and a supported site. Cotton II/IV and iron tiers are authored targets. |
| [Guide 20: new no-own-brewery Three Cotton](https://www.bilibili.com/video/BV1LV5u6xEUN/) | Three cotton/goods industry cards and a coal card; high cotton against resource-heavy opponents. No exact removals/sites. | Prefer high cotton with current public beer; avoid speculative coal. Cotton III/IV and development scheduling are engine-based adaptations, not a transcription. |
| [Guides 21–23: zero-loan variants](https://www.bilibili.com/video/BV1K7416kEui/) | Resource-market receipts fund industry/access; high-goods and high-cotton variants preserve income/actions by avoiding loans. | Require actual market buyers and receipts. Pay before recognizing receipts. High-tier investments need real cards, access, beer, cash and time; rescue loans remain possible. |
| [Guides 7–10: pottery variants](https://www.bilibili.com/video/BV177qbYyEe3/) | Brewery-led pottery, financing, sale windows, development; four-brewery and cotton-to-pottery alternatives. Flexible actions remain in the basic route. | Compare pottery, necessary supply, iron income and cotton pivots. Build non-developable pottery tiers first; stop development at the intended replacement. No blind brewery/loan quotas. |
| [Guides 11–15: seat/market pottery](https://www.bilibili.com/video/BV1FGC1YyEC2/) | First/later-seat iron/pottery, research-iron and special-market Pottery I/III routes. | Select support from real markets/access/cards; reassess after losing a site or beer. |
| [Guide 19: external-beer pottery](https://www.bilibili.com/video/BV1EMosBVE29/) | No-own-brewery pottery/carry-over and later iron potential. Headline totals include later-era scoring. No full sequence. | Additional pottery variant with reachable current public beer. Carry-over is a forecast, never immediate earned VP. |

Traditional Standard Iron & Beer, Rail & Pottery, Three Cotton and Goods & Beer are no longer the public presets. Archived executable/profile bytes remain historical controls and save-validation evidence.

## Six styles and era transitions

| Public style (save alias) | Canal priorities | Rail priorities |
| --- | --- | --- |
| Adaptive Advanced Book (`balanced`) | External-beer cotton, pottery, market-financed industry and supported Worcester recovery. | Funded scoring industries and useful complete rail plans; preserve sale funding. |
| Coventry Goods Opening (`rail-tempo`) | Conditional fourth-seat Coventry access; funded goods/iron when unavailable. | Finishable higher goods and connections that improve coal, market or buyer access. No automatic rail quota. |
| External-Beer Three Cotton (`cotton`) | High cotton with current merchant/opponent beer; income/coal support only when justified. | Cotton IV and grouped conversions when supply allows. Contested external beer can require immediate sale. |
| Worcester Iron-Cotton (`goods-beer`) | Iron market income plus cotton; bounded Worcester preference with its card. | Higher cotton and profitable iron, public beer and affordable supply contingencies. |
| Zero-Loan Industry (`supplier`) | Recycle connected resource receipts into goods/cotton and access. | Profitable resources, high goods/cotton and secured rails; stop the cash cycle when markets saturate. |
| Advanced Pottery Variants (`pottery-flex`) | Brewery-led, iron-led, cotton-to-pottery or external-beer routes by seat/hand/market. | High pottery and necessary income/supply; abandon blocked routes. |

Tiers and quantities are soft goals. No style restricts legal actions, forbids brewing/borrowing, or ignores an existing investment after pivoting. Rail transitions absent from descriptions are implementation hypotheses.

## Shared economic and opening fixes in selected v4

`TournamentOpenings.Intent` quotes supported sites, development iron, buyer distance, connections, available beer, brewery count, financing and negative-income runway. It rejects missed conversion deadlines. Reaching the tier or funding the plan ends old development/loan incentives.

Reachable opponent breweries count alongside own and compatible merchant beer. Existing unsold investments have a claim on supply. Longer setups discount contested barrels without inventing future breweries. Own brewing is a contingency, not a compulsory companion to every industry.

Necessary brewery funding is separate from predicting its flip: supply may precede its customer/access. Quotes reserve the industry site, distinct brewery sites and rail prerequisites. Cheap unconsumed coal is penalized even at low income. Resource opportunity quotes recognize actual immediate market sales and receipts.

A first canal may be anywhere, but changes subsequent industry-card access. Opening/expansion proposals consider supporting cards, affordable follow-ups, coal, markets and buyers. Early rails still pass the host's coal/payment mask. Access quotes use cloned boards and never change real links, markets or hands. Existing conversions get connection priority; hypothetical link points alone do not justify expansion.

Productive mat removals, owned focus industries and buyer-access progress reset intent age. An unchanged intent may time out; advancing toward a high tier does not. `FastIncumbents` applies the bounded opening-goal bias to both build-family and loan/develop/network setup ranking, so the goal influences candidate selection before complete-plan scoring. The focused economy checks cover productive development, stalled intent, completed tiers and six v4 labels.

## Historical rejected network/Scout additions (excluded from v4)

The rejected network/hand revision compared explicit **double-rail complete plans** with industry investments. It charged both rail fees, coal and the beer actually consumed through host replay; a forced double candidate needed surplus own beer or reachable opponent beer. Existing sale commitments protected own barrels. Where there was a credible funded double-rail opportunity without spare beer, a brewery/develop-then-brewery supply plan also entered the comparison. This experiment remains frozen for diagnosis, not active play or training.

Later early-era canals carry a bounded action opportunity cost unless they improve an owned industry's buyer route or enable a same-turn build. The first canal stays exempt. Network-oriented route priors consider city capacity and brewery slots; industry-oriented priors consider suitable vacant slots, hand support and compatible buyers. These are small access preferences: an empty brewery slot earns **zero official link VP**. Real link values still use `tile.link_vp` and the tile's completion forecast.

Scout compares affordable investment opportunities from the real hand with hypothetical wild access, allowing for the Scout action, development, financing, building and sale deadlines. Discard proposals protect useful goal cards. The host still chooses the legal three-card discard and supplies the two wilds; quotes cannot mutate the real hand or expose hidden cards. A hand already holding wild access and a late hand without time to exploit it receive no Scout incentive.

`NetworkAndHand.cs` supplies these bounded proposals and action costs. Headless/game `Evaluator`, `Conversion`, `CashManagement` and `Tactics` are byte-identical. The CPU evaluation queue executes nested sub-quotes inline outside its lock, so its drain owner never waits on its own queued request. Private search trees remain unshared.

Masked ISMCTS remains the engine. Four private lab roots explore complementary industries with a dynamic evaluation queue; two roots run in-game. Soft style priors remain subject to complete-plan scoring. NN search/rewards and fresh MLP-ResNet initialization are unaffected.

## Previous/economic comparison and promotion

The user's newest request expands the screen to **previous advanced revision `bank-3dc65253a610191d66a5` AND economic control `bank-f5107c644af2fbf4ac99`**. The former is the last revision before these double-rail/canal/Scout changes, not the installed 1.4.18 strategy. Three predetermined fresh setups/style pairs, four seat rotations each, cover six styles in **12 full four-player games per control (24 total)**, one second/four workers for every seat. Games never enter learner replay.

The candidate must achieve **strictly more first places and official win share, higher mean VP, and lower mean numeric rank against EACH control**. Equality fails. Runner, bank admission and deployment independently enforce all metrics. New receipt format: `masked-economy-smoke/1.3.0`. Historical economic-only and three-control screens retain their original requirements.

The newest screen used seeds **843700–843702**, `artifacts/economy-v3/smoke-8`, frozen candidate **`bank-fcab56543a2c86f30c66`**. Focused checks passed: 93 search/runtime, 17 industry and 46 economy/queue/network/hand checks. All 24 games completed without errors. **Promotion failed** because the previous-revision comparison did not improve. These are directional results, not an established tournament-level win rate.

| Control (12 games each) | First places, new/control | Mean VP, new/control | Mean rank, new/control (lower is better) | All metrics |
| --- | --- | --- | --- | --- |
| Previous advanced revision | 5 / 7 | 83.458 / 85.458 | 2.667 / 2.333 | Fail |
| Economic revision | 8 / 4 | 92.542 / 87.375 | 2.458 / 2.542 | Pass |

At that failed screen, the game was UI 1.4.19 / AI 1.4.18 and the admitted training bank was `bank-340a20d72b23c652c267`. The subsequent **v4** selection supersedes that state: active bank **`bank-3dc65253a610191d66a5`**, with ten authored transfer assets and 904 private runtime files. The rejected candidate remains unadmitted. Its full native/save qualification was not run after strength failure. The [network revision review](MASKED_V3_NETWORK_REVIEW.md) retains the observed losses and proposed follow-up.

Historical rejected screens remain intact:

| Screen | Economic result | Disposition |
| --- | --- | --- |
| Generic `bank-206883ed41ee7d4bb5dd`, `smoke-3` | 4–8 first places; 71.50 vs 85.29 VP; 2.92 vs 2.08 rank | Not installed/admitted. |
| Traditional source-based `bank-63d38386aa9370349f6e`, `smoke-4`, seeds 843300–843302 | 5–7 first places; 84.25 vs 87.13 VP; 2.54 vs 2.46 rank | Not installed/admitted; traces motivated brewery funding and coal repairs. |
| First advanced `bank-c92efb835d31fd1ca363`, `smoke-5`, seeds 843400–843402 | 6–6; 86.67 vs 88.13 VP; 2.75 vs 2.25 rank | Not installed/admitted. |
| Access revision `bank-e8191140f5030f79c377`, `smoke-6`, seeds 843500–843502 | 6–6; 89.08 vs 84.21 VP; 2.375 vs 2.625 rank | Not installed/admitted: equal wins fail. |
| Progress-aware revision `bank-3dc65253a610191d66a5`, `smoke-7`, seeds 843600–843602 | 6–6; 88.79 vs 84.71 VP; 2.417 vs 2.583 rank | Subsequently selected as **v4** by the user; not a strength promotion. |

Winning candidates also require legal replay, merchant/cash/conversion checks, native execution of all styles, clocks, cancellation, resource restoration and older saves. Proposed **1.4.20** changes only the AI helper over qualified UI **1.4.19**. Game: **two workers**, existing **2/5/12-second** choices. Lab: **four workers**, **24 queued tables**, shared 60 CPU slots plus four reserved cores.

Only after qualification does the selected bank supply searched imitation, early search-free opponents and later one-second searched opponents. Advanced pottery's opponent identity remains held out; all learner Pottery actions remain available. No old neural checkpoint or NN replay is imported.

See [the conda lab guide](../docs/TRAINING_RECOVERY_LAB.md). This historical comparison command is retained for reproducibility; further tests were stopped and it is not required for v4 installation/training:

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-economy-v3.py --root /data/not_backed_up/yxu209/ai-training/ --bank bank-fcab56543a2c86f30c66 --output /data/not_backed_up/yxu209/ai-training/artifacts/economy-v3/lab-screen --tables 24 --seed 843700
```


The follow-up reduced speculative network-access proposal credit, granted opening potential only to the first own link, and included every legal industry family in first-turn complete-incumbent comparisons. It also protected the target build card from unrelated setup discards. That historical screen used seeds 843500–843502 and `artifacts/economy-v3/smoke-6`, under the earlier economic-only rule.


The progress-aware fix distinguished productive mat development, funding and buyer-access progress from a stalled goal, and applied the same bounded opening prior when choosing forced build/setup candidates. Its completed `smoke-7` tied on wins. The user subsequently chose those exact strategy bytes as **v4**; no new strength games were run. The all-metric gate above still applies to other experimental candidates.
