# P6: laptop packaging and game integration

P6 tooling is implemented. **No trained personality package is qualified or installed.** The current Step 5 model is still on the lab server, and the real P4 roster and P5 difficulty evidence are pending. The local authored fixtures exercise deployment mechanics only; they do not establish playing strength, recognizable styles or Easy/Medium/Hard difficulty.

## Prerequisites and boundaries

| Stage | Available implementation | Still required for a release |
| --- | --- | --- |
| P0/P1 | Accepted-parent handoff, versioned model/utility/goal contracts and traces | Completed eligible Step 5 accepted model; real goal/utility validation |
| P2/P3 | Masked ISMCTS intent/search integration, shared batching, discovery and learned repertoire tools | Real calibration, searched-controller evidence and learned descriptors |
| P4 | Shared-network roster selection and member/mixture qualification | Qualified **16 groups × eight members**, actual names/descriptions and human review |
| P5 | Tier profiles, every-Easy-member versus original Hard gates, mixture/equivalence/order gates, public intent interface | Complete untouched panels and human evidence for the exact two-worker laptop profiles and hint mode |
| P6 | Export, persistent local service, staged game helper, benchmarks, validation, installation and rollback | Actual-model export parity, representative laptop/backend tests, interactive Unity acceptance and release certificate |

P6 adds `src/brass_ai/personality/laptop/`, `src/BrassPersonalityGame/` and `scripts/personality_laptop.py`. It reuses the existing P0–P5 implementation and does not change Step 5, its datasets, optimizer/checkpoint continuation, reward, architecture or training settings. Staging reads the accepted **1.4.31** game archive, so unrelated uninstalled source changes are not imported.

The candidate package contains one shared float32 ONNX actor for all 128 members and the small frozen rollout-reference pool required by the qualified P2 controller. It preserves policy, official win/placement/VP, raw remaining utility, goal factors and confidence. It does not ship 128 copies of the actor, an optimizer, discovery archive or behavior-encoder weights. The bundled Python library remains byte-pinned; no learner entry point is launched by the game.

Native simulator dependencies include proprietary game DLLs. These are **private local packages**, not artifacts to upload to a public repository. The Python interpreter and its installed dependencies are provisioned separately. No installer downloads dependencies or models automatically.

## Export on the lab server

Use the existing conda environment and native R4 host prerequisites. Existing R2/search/personality requirements must already be installed.

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
cd "$ROOT"
python -m pip install -r "$ROOT/requirements-personality-p6.txt"

python "$ROOT/scripts/personality_laptop.py" export \
  --roster "$ROOT/artifacts/personality/p4/qualified-roster" \
  --qualification "$ROOT/artifacts/personality/p5/qualified-report" \
  --output "$ROOT/artifacts/personality/p6/candidate-001"

python "$ROOT/scripts/personality_laptop.py" validate \
  --package "$ROOT/artifacts/personality/p6/candidate-001"
```

Replace the two input directories with the actual P4 roster and P5 report directories. Export refuses an incomplete roster, changed sources/model/pool, a preview, missing qualification or a mismatched reward/profile. The output is an immutable **candidate**, not a release. Adding `--fixture` is only for authored development fixtures and always records `qualified_sources: false`.

Export checks every named head against PyTorch using multiple batch/candidate shapes, legality masks, style inputs and seat rotations. It stores parity vectors for testing the actual execution backend later. A backend never silently changes to FP16 or another device. Source or membership changes require a new package; keep old immutable packages for existing saves. To validate an older package after updating the checkout, use its bundled runtime:

```bash
PKG="$ROOT/artifacts/personality/p6/candidate-001"
PYTHONPATH="$PKG/runtime/src" python -m brass_ai.personality.laptop.cli validate --package "$PKG"
```

## Stage the Windows helper

On the laptop, use a tested Python environment with the existing CPU PyTorch, NumPy, JSON Schema, OpenSpiel, SciPy and P6 requirements. The game launches that absolute Python executable with the package's bundled modules. `requirements-searchlab.txt`, `requirements-personality-p5.txt` and `requirements-personality-p6.txt` declare these dependencies. OpenSpiel and the native host remain the rules adapter used by Masked ISMCTS; this does not reintroduce a separate search algorithm.

```powershell
conda activate ai-training
$Project = 'C:/Program Files (x86)/Steam/steamapps/common/Brass Birmingham'
Set-Location -LiteralPath $Project
$Python = (Get-Command python).Source
$Package = "$Project/artifacts/personality/p6/candidate-001"
$Build = "$Project/artifacts/personality/p6/game-build-001"

python "$Project/scripts/personality_laptop.py" stage-game `
  --base-version 1.4.31 --output $Build
```

This compiles staged `BrassLegacyAI.dll` and `BrassPatch.dll` under artifacts. The second DLL adds only a dispatch hook for neural mandatory-liquidation decisions; the accepted rules and human/legacy liquidation handling stay intact. The archived source anchors, game assembly and build bytes are checked. **It does not write to the installed game's Managed directory.** An unexpected baseline fails rather than overwriting another patch.

The integration adds the qualified 16-style catalog to the existing Gamma selection page, with Easy/Medium/Hard and Random. Four-player standard games only are enabled. It keeps two private search workers and the **2/5/12-second whole-turn options**. Both actions/subchoices share one CPU-admitted deadline; Play Now requests the best completed legal incumbent and cancellation retires workers before resource restoration. No action starts merely because a budget expired. A response is validated against the current game, actor, revision and complete native legal path before mutation. If a legal incumbent cannot be produced or validation fails, the game stops visibly rather than silently changing the personality or playing an unsafe move.

Mandatory liquidation uses the same legal, policy-only neural choice as P2/P5. It cannot silently fall through to the stock AI's liquidation heuristic, reset the ordinary turn's saved allowance or count as a search teaching sample. The existing round-end save/UI flow commits one validated liquidation at a time.

The native adapter explicitly maps the game's `local-game/0.3.0` standard four-player extension to the existing `public-state/0.2.0` model boundary. It rejects introductory games, other player counts and non-eight-round configurations, removes the local-only configuration fields, and retains the existing strict schema. Confirmed public action history and the actual round-turn owner are reconstructed separately from temporary liquidation focus. Focused cases check exact permitted-state reconstruction in the unchanged training rules adapter; this is not a general conversion of arbitrary game versions or old datasets.

## Sampling, saves and public information

Each seat receives independent SHA-256 streams derived from package, game, independent sampling seed, seat and stream purpose. A selected style samples one of its eight members uniformly once. Random first selects one of the 16 styles. This does not use the deck-shuffle seed, and difficulty or seat creation order cannot reroll a member.

The versioned outer save envelope preserves the original legacy extension inside it. It pins the package, selected member and profile; saves policy RNG, strategic intent, permitted own-hand recall, public display hysteresis, pending action and remaining whole-turn allowance. A revision/history journal supports save/load and history rewind without replaying a future choice. Mixed legacy/neural tables retain both states. Loading a save with a different package is rejected explicitly; install its pinned compatible package instead of resampling. Existing legacy-only saves retain their existing format and saved clocks, including older 10-second choices.

Only the 16 names/descriptions and measured tiers appear in the menu. Internal member IDs remain private to saves/diagnostics. Default intent text uses public personality/action templates and appears only after the action commits. It does not display private goals, hands, model values or search traces. Broad goal-head hints require a separately P5-qualified `intent-hints` profile and explicit installation opt-in. They are not enabled by default.

## Benchmark CPU and optional Intel graphics

Use ORT CPU first. Its initial configuration is two tensor threads, sequential graph execution and disabled idle spinning. Two private search lanes reuse one dynamic inference queue and persistent model sessions. Whole-turn work, including native search and completed results, determines backend choice. GPU memory use is not a performance metric.

Optional Intel comparisons require a working Intel driver and OpenVINO:

```powershell
python -m pip install 'openvino>=2025,<2027'
python "$Project/scripts/personality_laptop.py" benchmark `
  --package $Package --cases "$Project/artifacts/personality/p6/laptop-cases.json" `
  --output "$Project/artifacts/personality/p6/laptop-benchmark-001.json" `
  --backends ort-cpu openvino-cpu openvino-gpu --tiers easy medium hard `
  --unity-running
```

`--unity-running` records an operator assertion; use it only while the isolated review game is actually running. Unsupported devices fail explicitly and are excluded. Optional GPU execution is **not yet verified locally**. CPU defaults follow ORT's [thread-pool guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html); explicit OpenVINO GPU selection follows the [Intel GPU runtime documentation](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html). Backend float32 parity is checked before search measurements.

Cases are a JSON array (1–24 positions) of `{"view": <permitted observation>, "choices": <complete legal root choices>, "seed": 123, "insolvencies": [0,0,0,0]}`. Obtain `view` and `choices` together from the existing native permitted-observation boundary at a completed action boundary; never substitute a full vendor save or fabricated opponent hand. Record actual public insolvency counts. Keep exactly the same cases/seeds across backend comparisons. Add separate copies with `"interrupt": "play-now"` or `"cancel"` and `"interrupt_after": 0.2` for controls.

Real acceptance requires ordinary positions from **both eras**, two distinct seeds per tier/era, useful completed simulations in each stratum, build/develop/sell/network opportunities, no reset deadlines, at most 50 ms measured turn overrun, and successful cancellation/Play Now. Collect representative invested and congested positions using the real model; the small pass-only fixture below is insufficient. The benchmark records cold startup, whole-turn elapsed time, completed simulations, search-quality teaching samples, batching, parent CPU time and process-tree peak RSS. Native child CPU totals and GPU utilization/memory are explicitly unavailable unless measured separately. Search-quality samples are diagnostics, not P4/P5 qualification.

## Isolated Unity review before release

Actual Unity layout, four-player play, pending-action restore and presentation/resource restoration still need human checks. To make those possible before certification, the installer supports a narrowly gated **review copy**. Prepare a separate private copy of the game under a **short** workspace `artifacts` path, e.g. `artifacts/p6-review`, and create a `.p6-fixture-sandbox` marker there. The normal installed game directory is refused for this mode. Use a separate Windows test user/profile so Unity uses a different persistent-data directory: copying the executable alone does not isolate user saves. No interactive game was launched in this task.

```powershell
$ReviewGame = "$Project/artifacts/p6-review"
# Only after preparing the isolated copy:
New-Item -ItemType File -Path "$ReviewGame/.p6-fixture-sandbox"
python "$Project/scripts/personality_laptop.py" install `
  --package $Package --build $Build --game $ReviewGame --python $Python `
  --review-copy --review-backend ort-cpu
# Review the dry-run output, then repeat with --apply for this isolated copy.
```

Launch the copied game with `--brass-personality-review`. Both the explicit flag and marker are required. Fixture entries say **FIXTURE**; unqualified real candidates say **REVIEW**. Neither mode asserts measured difficulty or creates a release certificate. For Intel acceptance, use the matching `--review-backend` after parity succeeds. Fixtures cannot pass the release command. This task used only a minimal isolated DLL sandbox for installer/rollback tests; it did not launch or modify the installed game.

Write a real acceptance record after testing. Its format is `personality-laptop-acceptance/1.0.0`, with `package_id`, `build_id`, `backend`, `target: "Intel Core Ultra 5 135U"`, a named `reviewer`, and boolean `checks`:

```json
{
  "four_player_game": false,
  "sixteen_styles": false,
  "save_load_pending_action": false,
  "history_rewind": false,
  "play_now": false,
  "menu_cancel": false,
  "resource_restoration": false,
  "old_save_unchanged": false,
  "same_profile_strength_review": false
}
```

These are pending checks, not an example passing review. Record evidence for each before changing its value. Confirm that export/backend differences preserve the calibrated profiles' behavior; a faster backend does not automatically establish stronger or equally balanced difficulties.

## Certification, installation and rollback

Once qualified real P4/P5 artifacts, matching real laptop benchmarks and human acceptance exist:

```powershell
$Certificate = "$Project/artifacts/personality/p6/release-001.json"
python "$Project/scripts/personality_laptop.py" certify `
  --package $Package --build $Build `
  --benchmark "$Project/artifacts/personality/p6/laptop-benchmark-001.json" `
  --acceptance "$Project/artifacts/personality/p6/laptop-acceptance-001.json" `
  --backend ort-cpu --output $Certificate

python "$Project/scripts/personality_laptop.py" install `
  --package $Package --build $Build --game $Project --python $Python `
  --certificate $Certificate
# Future release installation, with the game closed: repeat with --apply.
```

Installation is a dry run by default. `--apply` requires exact real qualification/certificate pins and the expected installed helpers. It copies an immutable package, saves both prior helpers and the pointer, replaces `BrassLegacyAI.dll` and `BrassPatch.dll`, adds `BrassPersonality/`, and writes a rollback journal. `Assembly-CSharp.dll`, saves and training are untouched. Short content-addressed paths avoid the old Windows runtime's path limit; full hashes remain checked. Copy validation precedes replacement; ordinary failures restore both helpers/pointer. Hashes detect changed bytes; they are not a publisher-signature system.

```powershell
python "$Project/scripts/personality_laptop.py" rollback `
  --journal 'ABSOLUTE-JOURNAL-PATH-PRINTED-BY-INSTALL'
# After reviewing the dry run, repeat with --apply and the game closed.
# For a fixture review sandbox, also supply --review-copy.
```

Rollback verifies both current and backup bytes, restores the exact prior helper and pointer, leaves saves untouched and retains packages. An old helper cannot run a new P6 save; restore the pinned P6 installation to resume it. A forcibly interrupted installation may require restoring the backed-up helper/pointer manually; do not bypass hash checks or delete an unknown installation. No release installation occurred during P6 implementation.

## Focused development checks and measured fixture timing

From the project root with the existing dependency environment:

```powershell
$env:PYTHONPATH="$Project/src;$Project/tests/R2;$Project/tests/Recovery;$Project/tests/Personality"
python -m unittest discover -s "$Project/tests/Personality" -p test_laptop.py -v
python "$Project/tests/Personality/smoke_laptop.py" `
  --ready "$Project/artifacts/personality/p5-20261012/native-smoke-fixed/ready" `
  --output "$Project/artifacts/personality/p6-smoke-new" --tiers easy medium hard
```

The `--ready` argument reuses a labeled fixture; omit it to generate a fresh fixture if the old fixture/source identity no longer matches. Use a fresh output path. `check_laptop_native.py` compiles focused native history/legality/save checks against a staged helper. `check_laptop_pipeline.py` tests the bundled runtime in its own module path, Play Now, cancellation, exact save restore, late-control isolation, installation dry run and byte-exact rollback in a marked sandbox.

The native report can supply `--debt-cases` to `check_laptop_pipeline.py`; this also checks seven actual native observation/choice pairs against the strict Python schema and reconstructed simulator. The liquidation probe must return a legal policy choice without search or changing the saved ordinary-turn budget. Public error messages omit rejected observations/private card values.

Final fixture artifacts: `artifacts/personality/p6-package-final`, `p6-game-build-final`, `p6-native-final2/report.json` (42 assertions), and `artifacts/p6f/report.json` (standalone service, seven native reconstructions, controls, liquidation, install/rollback). Cancellation returned in 0.422 s and a warm mandatory-liquidation request in 0.156 s. The first cold Play Now request took 7.891 s including lazy model/native-worker initialization; the completed plan itself took 0.422 s. Play Now interrupts planning but does not skip that one-time initialization. Real-model startup latency and interactive presentation still require review before release. `artifacts/personality/p6-preservation.json` verifies unchanged installed DLLs, P0–P5 identity and Step 5 implementation identity.

Measured on this **Core Ultra 5 135U**, ORT CPU, two workers, one ordinary two-action Canal position, untrained small fixture:

| Whole-turn cap | Warm request work, seconds | After CPU admission, seconds | Completed simulations | Search-quality samples |
| --- | ---: | ---: | ---: | ---: |
| 2 s | 1.938 | 1.844 | 0 | 0; legal incumbent fallback |
| 5 s | 4.593 | 4.547 | 0 | 0; legal incumbent fallback |
| 12 s | 10.891 | 10.844 | 8 | 1 |

No deadline resets or overruns occurred. The first request additionally needed 2.437 s to initialize its private native workers; model/service loading took 2.172 s. The 22.16 s benchmark (including startup/cleanup) completed eight simulations and one quality sample. Peak process-tree RSS was about **633 MiB**. These startup costs are reported separately, not hidden inside an apparent 2-second end-to-end latency. A previous probe completed one additional Medium simulation; this illustrates why a single small fixture probe is not a throughput or strength qualification.

Reusing a single already-completed ordered turn trace removed a redundant post-search native replay that had caused roughly 0.22–0.45 s overruns in an earlier fixture probe. Unity still validates every full action against the live game before applying it. This small probe does not establish a general throughput gain or real-model latency. Artifact: `artifacts/personality/p6-final-benchmark.json`; the subsequent mandatory-debt hook preserves this ordinary-turn search path and is checked separately.

Remaining release work: finish the real model and P0–P5 evidence; export and test that model; create representative both-era benchmark cases; compare CPU/Intel graphics on the actual laptop with Unity running; complete interactive save/rewind/cancel/UI/resource and strength review; then certify and explicitly install. No long training or qualification campaign is part of P6 tooling verification.

## P5/P6 review update

Packages now use `personality-laptop-package/1.1.0` and the stricter P5 verifier. Real candidates carry the exact P4 report bytes pinned by the roster, plus the complete P5 qualification summary. Missing gates, incomplete members/mixtures/tiers, mismatched actor identities, named outputs, seat contracts or rewards fail validation. Export parity rejects nonfinite legal policy/value/goal outputs, including matching NaNs; masked policy entries must remain negative infinity. Combined utility remains unbounded by probability limits.

The native helper explicitly rejects unsupported neural-save versions instead of treating them as legacy saves. Python restoration validates saved turn keys and tier allowances. Native error popups omit exception payloads that can contain cards or simulated choices. The Random menu text now refers to available styles. These changes preserve two workers, 2/5/12-second clocks, search behavior and legacy-only saves.

Installation dry runs and applied installations now probe the **supplied interpreter** using the same bundled module path as the game. The probe loads every exported graph, checks backend parity and starts/closes the private native host. Missing dependencies or an unavailable backend stop installation before game files change. It installs nothing and runs no full game. Run it directly in the existing conda environment if needed:

```bash
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
python "$ROOT/scripts/personality_laptop.py" check-runtime \
  --package "$ROOT/artifacts/personality/p6/package" --backend ort-cpu
# Add --fixture only for a clearly labeled fixture/review package.
```

On Windows use the tested absolute Python executable and the same command. The interpreter must have its dependencies installed in its own environment; development-only `PYTHONPATH` entries are deliberately not inherited by the game. No dependency download happens during the probe. Release certification additionally requires the benchmark's measured CPU identity to match the 135U target; a lab-machine result cannot substitute for that measurement.

An ordinary I/O failure during rollback now restores the current installation so rollback can be retried. Focused failure injection tests cover interruption between the two DLL replacements. Abrupt OS/process termination remains a manual-recovery case; retain the backup and journal. Fixtures still install only into a marked isolated artifacts directory.

Earlier 1.0 packages remain pinned to their own immutable runtime and saves. Do not edit their IDs, replace their runtime in place, or substitute a new package into an old neural save. Re-export/restage new candidates after obtaining compatible P5 evidence; keep the original package to resume its old saves. No real P6 release exists locally, and no live installation occurred. Step 5/IL, P0–P4 and their model/optimizer/counter continuation are unchanged.

Review artifacts: `artifacts/personality/p56-package-review`, `p56-game-review`, `p56-native-review/report.json`, `artifacts/p6r/report.json`, and `artifacts/personality/p56-benchmark.json`. The fixture native checks passed 44 assertions; the standalone pipeline passed exact save restoration, cancellation, Play Now, seven native reconstructions, dependency preflight and isolated installation/rollback.

The new one-position CPU smoke measured admitted planning times of **1.828 / 4.531 / 10.828 seconds**, with **0 / 1 / 8 completed simulations** and **0 / 0 / 1 qualified teaching samples**, respectively. All three caps retained one deadline and had zero measured overruns. Cold model/service loading took 1.86 seconds; the first turn additionally initialized private workers. The cold Play Now RPC took 5.047 seconds including initialization, while cancellation took 0.281 seconds and warm liquidation took 0.078 seconds. Peak process-tree RSS was about 681 MiB. These untrained fixture timings do not qualify difficulty, real-model speed or Intel GPU performance, and are too small for a search-throughput claim.

The review finished with **54 targeted Python checks passing**. A same-bundle microbenchmark replayed the previous duplicate hash traversal under the current validation checks: median 1.450 seconds versus 1.405 seconds for one pass across 274 files (three alternating, warm-cache measurements). This is a small startup saving within measurement variation, not an end-to-end speed claim. Models remain persistent and search settings are unchanged. Artifacts: `p56-hash-benchmark.json` and `p56-review-preservation.json`; the latter verifies that all 231 snapshotted upstream/config/installed-DLL files and the Step 5 implementation fingerprint are unchanged.
