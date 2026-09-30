# Link placement replay repair — 1.4.6

**Date:** 30 September 2026.  
**Status:** Installed and verified in the Steam game directory.

This release repairs the presentation paths that can leave a placed canal or railway on the map while the action controls never return. The replay advances through the existing network action flow, including required coal, the optional second railway and required brewery beer.

## Reproduced defects

Native tests against archived patch 1.4.5 reproduced four related problems:

- Killing a placement animation never calls its replay completion delegate. Normal placement works; interrupted placement stalls for both eras.
- A queued animation can start after intentional replay cancellation and call the old delegate. The old cancellation only searches playing tweens, missing a sequence waiting to start.
- Repeated link highlighting creates uncancelled infinite pulses. Twenty updates left twenty running pulses even after clearing the highlights.
- Interrupted end-of-era link scoring leaves its token and VP label reparented outside the link. The subsequent railway animation throws while finding `LinkToken`, with no completion callback.

These are controlled native reproductions of the reported symptom. The precise timing of the user's rare live-game occurrence has not been captured.

## Repairs

Link placement now owns its camera, token and outline effects in one replay sequence. Selection outlines own one pulse, which stops on refresh or teardown. Scoring owns its animation and restores the token and label parents, text and transforms when it finishes, is interrupted, or is superseded by a new board binding or placement.

A guard attached to the replay view completes each link event once. It recovers a killed animation, a lost completion callback or an unexpectedly stalled active sequence. Its presentation allowance is at least three seconds of animation time; explicit replay pause and Unity time pause do not consume it. Missing visual targets or animation construction failures use a short finite fallback.

Intentional cancellation invalidates the guard before killing the animation. Queued starts, hidden replay views and a replaced game router cannot resume an old replay. Recovery calls the existing event replay delegate; it does not confirm a network action, bypass resource choices or advance an AI turn directly.

## Validation

- **765 native link checks:** both eras, all occupied player colors, 2/3/4 players, normal and fast-forward speed; killed, paused, cancelled, superseded and refreshed sequences; lost callbacks and missing visuals; interrupted scoring followed by rail placement; opening/later canal turns and single/double rail actions. Actual network and replay use cases check action controls, resources, card/tile counts and exact money costs.
- **12 fresh managed suites:** existing rules, gameplay histories, AI, liquidation and save regressions.
- **10 fresh native suites:** existing rules and UI regressions, including 46,779 city locator checks.
- **184 presentation checks**, **239 first-open/menu checks**, and **2,339 S1/runtime/save checks** against the candidate DLLs.

The AI DLL is byte-identical to 1.4.5. All existing AI, rule, persistence and input-guard source is unchanged; seven link presentation hooks are added. Prior 108-turn AI difficulty/strategy and archived save compatibility evidence is retained explicitly, rather than claimed as freshly rerun. Qualification verifies the frozen research profiles, the unchanged source scope and retention of every previous hook.

## Installation

Upgrade and interrupted-install recovery passed **79 assertions**. The installed Steam player passed the same **765 link checks** and **239 startup/menu checks**. All three installed DLL fingerprints, all **133 hooks**, the original stock backup and the frozen AI profiles were verified. Online play remains disabled. Personal saves were not opened or modified.

The release includes earlier gameplay, AI and UI fixes, including 1.4.5's immediate first-open AI controls. AI difficulty allowances remain 2/5/12 seconds for new games and 10 seconds for older local-AI saves; training bank settings are unaffected.

Binaries, source snapshots, regression reports and installation records are archived in `artifacts/patch-releases/1.4.6`. Patch 1.4.5 and the installer's verified recovery copy are retained. Restart the game to load the updated DLLs.
