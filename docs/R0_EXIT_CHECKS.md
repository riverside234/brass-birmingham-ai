# R0 exit checks

27 September 2026. **R0 passed its local exit checks.** Scope: reference pins,
component mapping, versioned interface definitions and four-seat semantic
fixtures. R1 and later implementation stages have not been started by this work.

| Gate | Executed evidence | Result |
| --- | --- | --- |
| Immutable upstream revisions | Resolved the selected Daiyu/OpenSpiel commits and root trees; matched each selected file's Git blob hash and SHA-256 | PASS — 2 commits, 16 files |
| Origins and notices | Pinned both LICENSE files; checked root NOTICE presence; recorded reference-only status and adaptation obligations | PASS — [provenance record](R0_REFERENCE_NOTICES.md) |
| Source-to-contract mapping | Verifier checks all 10 source paths are pinned and every mapped contract exists; inventory covers scalar values, seat indexing, hidden state and two-player assumptions | PASS — [component map](../configs/references/components.json) |
| Versioned project boundary | JSON Schema 2020-12 schema, content-signature manifest and 12 validated message examples | PASS — contract `0.1.0` |
| Invalid-message rejection | Unit cases cover changed rules/schema/objective, non-four-player/scalar values, non-finite numbers, stale identities, misplaced private fields, plan/budget boundaries and incompatible model metadata | PASS |
| Four-seat semantics | Asymmetric selfish choices, consecutive decisions, chance edges/expectations, no bot coalition, all 24 seat permutations, scoring ties and tie breaks | PASS |
| Existing rules-patch handoff | Read installed Assembly-CSharp.dll and BrassPatch.dll and compared both SHA-256 values with the recorded 1.1.0 profile | PASS — exact pair matches |
| Repository boundary | `git check-ignore` confirms cached reference source/dependencies and both installed DLLs are ignored; tracked-file query finds no game files/artifacts | PASS |
| Fetch/cache failure behavior | Simulated downloads verify bytes before caching; altered cache files fail without replacement; subsequent validation works offline | PASS |
| Unit suite | Python 3.12.14, `unittest discover -s tests/R0 -p "test_*.py" -v` | PASS — 37 tests |

Reproduction commands, using an environment with `requirements-r0.txt` installed:

```sh
python scripts/verify-r0.py
python -m unittest discover -s tests/R0 -p "test_*.py" -v
python scripts/verify-r0.py --check-cache --game-root "/path/to/Brass Birmingham"
```

Use `--fetch-references` once when the ignored source cache is absent. The first
two commands require neither the game nor that cache. Executed local output is
retained in ignored `artifacts/r0-tests.txt` and `artifacts/r0-verification.txt`.
Windows was tested; a lab/Linux run remains unverified.

The recorded installed binary pair is:

```text
Assembly-CSharp.dll  1c8cef281b81474200268eed23448faa3f5247f8ea49409a69efc3982c5a2d0a
BrassPatch.dll       8315180fba0a7d94d757bac270c8944c56cfba315f15db2537aceaca13f6eefb
```

No patch was rebuilt or reinstalled, and no game assembly was executed by these
checks. The bug-fix agent's broader regressions are prior evidence in
[LOCAL_PATCH.md](LOCAL_PATCH.md), not additional R0 test runs. Matching this pair
does not establish a complete rules-host dependency fingerprint.

Schema tests establish structural rejection, not actual game legality, engine
restoration or hidden-state noninterference. The multiplayer tests are small
original semantic prototypes, not execution of OpenSpiel or a Brass match. See
[the contract boundary and limits](R0_CONTRACTS.md). These distinctions are the
R0-to-R1 handoff, not unfinished R0 host implementation.
