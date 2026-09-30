# AI execution resource management — 1.4.7

**Date:** 30 September 2026.  
**Status:** Installed and verified in the Steam game directory.

This package adds a general execution lifecycle for turn computations. Resource
management follows computation, presentation, cancellation and actual completion;
it does not depend on an AI strategy, model, difficulty name or search allowance.
The current S1 executor is the first backend integration.

## Runtime behavior

- Computation and waiting for a retiring backend use a **30 FPS** software cap.
  The controller temporarily sets VSync to zero because the shipped Windows
  player otherwise ignores `targetFrameRate`. An existing lower software cap is
  respected. Both original settings are restored for move presentation, actual
  completion, game/history replacement, menu exit, errors and destruction.
- The turn clock starts after a preceding backend retires and initial board
  preparation finishes. Both actions and replanning retain the original running
  deadline. Easy/Mid/Hard remain **2/5/12 seconds**; old saves retain **10 seconds**.
  The S1 240-node planning setting and bounded legal-completion behavior remain.
- One turn computation executes at a time. Waiting requests await completion
  without blocking the Unity main thread or occupying a dedicated waiting thread.
  A backend may use its own internal threads or inference batches within its job.
- Cancellation requests do not imply completion. A GPU/inference backend must
  keep its task outstanding until its device work has finished. Turn-buffer
  cleanup runs after that completion, including cancellation before dispatch.
  Cancellation sources and detached observations are released. Model weights and
  model caches remain owned by the backend/application.
- Registered optional work can pause particles or explicitly selected scripts,
  including decorative animation/physics controllers. Its previous playing,
  paused and enabled states are restored. The package does not automatically mark
  shipped objects as optional: profiling has not established significant physics
  or particle costs in those scenes. Global time and physics settings are preserved.
- Search polling uses typed ticket checks instead of per-frame JSON conversion.
  The status label changes only at a Thinking/Playing transition. Published plans
  remain detached copies, prepared outside the coordinator lock and transferred
  once to the main thread. All existing identity/revision/generation guards remain.

There is no forced GC, thread-pool reset, per-turn model unloading, or global
simulation pause. Shared pool threads can remain idle for reuse. This lifecycle
does not cap the CPU utilization of the search itself; cheaper operations or a
separate search-budget policy would be needed to reduce its sustained CPU work.

## Backend integration

`Execution/IAiExecutionBackend` reports a task covering the entire computation.
`IAiTurnResources.ReleaseTurnResources` optionally returns borrowed turn buffers;
it must preserve shared model weights. Cleanup runs on the worker side, after
execution ends. Cancellation callbacks should be nonblocking.

Submit a backend through `AiExecutionScheduler.Submit`. On the Unity main thread,
create an `AiExecutionResources.BeginComputation` scope and call `Observe(job)`.
Use `Presenting()` before live action/animation dispatch and dispose the scope on
every ownership exit. The existing game/history/menu hooks also reset and cancel
independently registered backend scopes. Main-thread resource changes are guarded.
Backend adapters must obey their inference runtime's CPU/GPU thread requirements;
the resource controller does not invoke model APIs or receive live engine state.

`AiExecutionResources.Configure(new AiResourcePolicy(fps, suspendOptionalWork))`
changes resource policy independently of search settings. A zero cap disables FPS
limiting. `Register(IAiOptionalWork)` accepts custom reversible suspension tokens;
`AiOptionalSimulation` is an explicit marker for selected particles and scripts.

A neural model is not installed by this package. Its future adapter can use this
same scheduler, completion contract and resource scope without changing them.

## Policy and package preservation

`Assembly-CSharp.dll` and `BrassPatch.dll` remain byte-identical to installed 1.4.6,
preserving its link replay repair and all existing rules/UI/save hooks. Runtime
changes are limited to the coordinator, turn bridge, status cache and three new
execution/resource source files. Search, evaluation, profiles, observation
boundaries, saved identities and save format retain their prior source bytes.

Provenance verifies H5's immutable v2 bank and the deployed v3 bank
`bank-af17b1aadbe7f9b21fa7`. Deployment policy is compared directly with frozen
source. Independent edits to the active research worker are recorded separately
and preserved; they are not compiled into this game patch or relabeled as frozen
qualification. Compilation records source hashes before and after building.

## Verification

Fresh qualification is tied to the candidate DLL hashes and compilation source
pins. The following checks passed:

| Check | Result |
| --- | --- |
| Native Unity AI matrix | 108 configurations, 18,444 assertions, no native errors |
| Native resource lifecycle | 20 checks, including unregistering during suspension |
| Native execution retirement | 214 submitted/completed jobs, maximum 1 executing, 0 remaining |
| Managed asynchronous backend lifecycle | 10 checks |
| S1 runtime, deadlines and save contracts | 2,339 assertions |
| Complete in-memory gameplay | 6 games, 2,033 assertions |
| Generated saves from 1.3.2, 1.4.0 and 1.4.3 | 54 assertions per producer |
| Immutable H5/profile provenance | Passed; frozen v2 and deployed v3 preserved |
| Upgrade, interruption and locked-file recovery | 79 assertions from installed 1.4.6 |
| Installed native startup, menu and game creation | 259 assertions, including all 20 resource checks |
| Installed DLLs and retained hooks | All 3 hashes match; 133 hooks and 6 merchant rewrites verified |

The synthetic menu fixture recorded 60 frames in 1.014 seconds (59.20 FPS)
normally and 30 frames in 1.002 seconds (29.95 FPS) during computation.
Process CPU samples include engine/startup work; they do not establish a sustained
CPU reduction during real search. GPU power, temperatures, thermal throttling,
playing strength and all physical mouse/display configurations were not measured.

`local-ai-resource-baseline.json` records the expected failure on an intermediate
1.4.7 candidate before the optional-registration cleanup fix. It is a regression
reproduction, not a failure attributed to installed 1.4.6. The final native suite
passes that case and restores the returned token exactly once.

Installed SHA-256 fingerprints:

| File | SHA-256 |
| --- | --- |
| Assembly-CSharp.dll | `1540124CEAAED7C3E31F4762F6EC3616C85344CF5A56FED108FD29364724D907` |
| BrassPatch.dll | `AA0000B3007FB0479A9553A1243C34F5320C4671B6AECCD5883FFC405602BB51` |
| BrassLegacyAI.dll | `CA25E04FB7C4EA9AFACD2927818B236A3ED999230702773172F8BDB22B4A0C36` |

Installed at `2026-09-30T21:28:12.5184929Z`. The installed native fixture also
measured 59.57 FPS normally and 29.86 FPS during computation, with no native
errors and no remaining execution jobs. Installed H5 provenance passed separately.
The original stock backup is verified; the coherent pre-upgrade 1.4.6 package is
retained under `artifacts/patch-recovery/93284585576c4ce08d31abda32436d86`.
Release binaries, source pins, reports and this note are preserved in the
fingerprinted `artifacts/patch-releases/1.4.7` archive.
