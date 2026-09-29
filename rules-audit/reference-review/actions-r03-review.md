# Independent R03 reachability review

Source review on 27 September 2026 of `REFERENCE_CROSSCHECK_BUG_FIX_PLAN.md`, `ai-brewery-legal-trace.ps1`, its retained results, original AI rules and current patch overrides. No production files, native tests or user saves were changed or run for this review.

**The scripted history is legal. It is not a history that the original Hard AI would select unchanged. Original-policy brewery exhaustion during Canal is nevertheless not excluded: its Gloucester development-bonus selection can remove the final Brewery IV.** A naturally selected complete trajectory remains unproven, as the R03 text already states.

## Legal scripted history

- The three loans move income from 0 to −3, −6 and −9. Each is legal; each action discards a card and receives £30. The main plan now correctly cites **printed p. 11** for Loan.
- All seven actual brewery tiles are developable, including Brewery IV (`PlayerMat.cs:61–67`). The era flags restrict building: the Build action checks those flags, while the Develop rule on printed p. 11 permits the next one or two eligible tiles without an era restriction. Developing Brewery IV in Canal is legal. Normal `DevelopAction.Develop` at lines 27–46 uses `CanBeDeveloped`, not build-era eligibility.
- The ordered brewery batches `2,2,2,1` remove levels `I/I`, `II/II`, `III/III`, then `IV`. The subsequent seven Coal Mine removals are also legal development targets. No player ever builds anything in this trace, so iron must come from the market and then unlimited general supply. The script's market use therefore does not evade board-first consumption.
- Arithmetic is consistent. Starting £17 plus three £30 loans gives £107. Brewery iron costs £23; coal-development iron costs £41. Income paid through round five is £39 (3 + 4×9), leaving £4 after round-six development. The round-six £9 income obligation produces a £5 shortfall. With no board industries and 0 VP, printed p. 6 permits exhausting cash and losing VP only if possible; the patched liquidation flow finishes without an industry choice and leaves cash £0, VP 0. This is not a skipped debt phase.
- The target performs eleven actions through round six: one in round one, two in each later round. Counts 24/37/50 also include other players' actions before the target's round-seven turn, because its paid development places it last in round-seven order. They are not eleven target actions mislabeled as 24/37/50.
- The harness uses normal confirmation, EndTurn and EndRound. The latter receives the debt state started by EndTurn/BeginEndRound; it does not manually reset income, cash, phase or tiles.

The script explicitly asserts hand decrease, nonnegative cash, resource payment, development permission and affordability. Following this review it also records the remaining action count before each action and asserts an exact decrement of one after confirmation. The updated trace was rerun successfully for all three player counts, preserving the original and event-reconstructed crashes.

## Original Hard-AI policy

All normal brewery Develop candidates in the original rule set are Canal-only Brewery I:

- `DevelopCanalEraIndustries.cs:76–80`
- `DevelopCanalEraBrewery.cs:59,73–75`
- `DevelopManufactureForMerchantBeer.cs:116,122–124`

Thus it would be incorrect to describe the all-seven-Develop script as an original-policy sequence. Ordinary development stops selecting breweries once the first Level II is exposed. Brewery IV cannot be built during Canal either (`PlayerMat.cs:67`; `BuildAction.cs:59–62`).

There is, however, another policy path. `SellActionAIHelper.SellIndustries` first prefers a Canal-only Iron Works for Gloucester's development bonus (`SellActionAIHelper.cs:125–130`), then Canal-only Brewery I (`:132–138`), then **any remaining brewery** (`:140–143`). That fallback has no level/era restriction and can legally develop Brewery IV in Canal after the first six breweries have been built or otherwise removed. The current B12 patch scopes the bonus state to its owner but leaves this selection policy intact. The existing resource manifest only replaces this class's `FindBeerSource` and `SlotToSell`, not `SellIndustries`.

After that policy path, the crash additionally requires that all owned breweries on the board be flipped/absent and that higher-priority rules fail before priority 1230. Own brewery beer can be consumed by that player's sales or by connected opponents. These conditions are allowed; this review does not demonstrate a full naturally selected trajectory satisfying them.

The exact failing source is as claimed: `BuildBreweryForCanalEra.cs:43` evaluates no-owned-unflipped-brewery, then Canal era, then dereferences the next brewery, and only afterwards checks opening rounds. `PrioritySelector.cs:22–27` evaluates applicability in descending priority until one succeeds; `HardAiBehavior.cs:24–29` exposes that path. Easy/Medium omit the Hard-only rule, and Rail short-circuits before the null dereference. Those controls are well chosen.

Recommended classification remains **confirmed decision failure in a legally reachable game state; conditional P1; original-policy full-history reachability/frequency unproven**. Do not downgrade the state to rules-impossible, and do not upgrade the scripted trace to a naturally played original-AI failure.
