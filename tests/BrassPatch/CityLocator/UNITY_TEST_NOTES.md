# City locator verification

The fixture loads the shipped `GameScene` and instantiates its actual `BoardView` prefab. It uses the normal hand presenter, board/region/slot components, patched method entry points, and Unity's normal DOTween iterator. Application startup is bypassed in explicit self-test mode: no repository, player save, network client or account login is opened.

The surrounding router is an in-memory test double. Games use deterministic artificial state. All active cities are tested with no green spaces, a mixed set and all-green spaces, for 2/3/4 players and both eras. These supplied eligibility sets test presentation; they do not claim that every combination is reachable through legal gameplay. A separate case passes a city card through the real hand use case and checks against its actual build-eligibility query.

The suite also exercises occupied spaces, repeated clicks, original outline transform changes, cloned region cleanup, actual region click routing, board/resource/link/merchant refreshes, card-delegate ownership, the common action-in-progress boundary, end-turn handoff, disable/reopen, and uncached-view cleanup. All seven normal action presenters enter through the checked action-in-progress boundary. End-turn testing stops at the use-case handoff after checking cleanup.

Rendered checks use the same prefab and a Unity camera at 1600 x 900. They export baseline, mixed and all-white screenshots for day/night maps at the shipped minimum (0.028), middle (0.064) and maximum (0.1) zoom. Native `GraphicRaycaster` results with the white frames enabled must equal results with those frames hidden. The screenshots are fixture renders, not screenshots of a user's save or a complete interactive match. Physical mouse sequences and every display/DPI configuration remain outside this automated coverage.

The original `BuildingSlotOutline` sprite has glow padding and a 1.4 image scale. Rendering revealed that copying this padded image's bounds oversized the white boxes. The implementation instead fits the slot's own border rectangle and leaves the original textured green graphic untouched.

Commands from the game directory:

```powershell
pwsh -NoProfile -File tests/BrassPatch/CityLocator/run-unity.ps1
pwsh -NoProfile -File tests/BrassPatch/CityLocator/run-unity.ps1 -Render
```

The runner uses a private player under `artifacts/city-locator-player`. Only immutable engine/assets are hard-linked; candidate assemblies are copied. It checks process completion and output status, verifies the tested assembly hashes, and writes evidence to `artifacts/patch-build/city-native-result.json` or `city-render-result.json`. The full installed Unity self-test also runs the city suite after the existing nine suites.
