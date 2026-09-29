# R03 brewery rule guard

`AiFixes.BreweryRuleApplicable` wraps the shipped Hard-AI opening brewery rule.
It returns false outside the first two Canal rounds or when the next brewery is
absent. Eligible opening states delegate to the preserved original method, so
owned unflipped breweries, tile-era restrictions, valid slots and heuristic
priority keep their existing behavior. Exceptions inside otherwise valid
original logic propagate; the patch does not replace the original AI or force a
fallback action.

The recursive helper compilation and manifest loader pick up this directory.
`manifest.json` replaces only
`BrassApplication.AI.RuleBasedSystem.Rules.BuildBreweryForCanalEra.IsApplicable`.
The existing B14 manufacturer guard is unchanged.

Regression entry points are `tests/BrassPatch/AI/run.ps1` for the staged managed
candidate and `AiRegression.Run()` for the explicit Unity Mono self-test.
`Run()` returns its assertion count; `Main()` returns process status. The suite
checks original opening applicability, executable public decisions, three legal
scripted brewery-exhaustion histories and event replays, and the previous B14
manufacturer boundary. Synthetic early-round empty stacks are boundary probes,
not claimed natural histories. No real saves are read or written.
