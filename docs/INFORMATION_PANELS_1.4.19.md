# Information panel UI — patch 1.4.19

This revision is designed for **1920 × 1200**, with **1920 × 1080** also supported and tested. It retains the game and AI binaries from **1.4.18**, including its development and group-sale improvements, and the Card Tracker and statistics models from **1.4.17**.

## Card Tracker

- The button is fixed at the bottom right, above the action controls. It no longer follows the iron market, moving hand cards or map inspection.
- The second text preset, **115%**, is the initial size. All three presets remain available; the selected size is retained while reopening the same tracker.
- A compact 48-pixel toolbar separates the title, paused state, public counts and controls. Its maximum width is 1,260 pixels, leaving more of the board visible.
- A separate **Wild cards** strip groups information by player. Each player has labeled **Location** and **Industry** rows with distinct pin/factory symbols and held counts. Gold rows indicate possession; `1` means held, `0` means none, and `?` indicates incomplete public history. Clicking a row opens the full explanation.
- The tracker still pauses gameplay when opened. It remains limited to four-player games and unavailable during AI turns, replay, pending actions and scoring animation.

## Statistics

- The initial text size is the second preset, **115%**. Existing size controls, navigation, filtering and scroll retention remain available.
- The results entry is centered below the original results controls with a clear gap at 1080p and 1200p. It has a rounded ivory/brass surface, a contrasting dark label and a chart icon.
- The board-review entry uses the lower row beside the original summary control and is hidden while the results or statistics modal is open.
- The eleven statistics pages and their calculations are retained from 1.4.17.

## Verification and installation

Qualification uses the native Unity player, including 1200p and 1080p captures, the three text presets, day/night tracker views, animation positioning, native action-control clearance and original Review/Exit clearance. Completed-match accounting, UI lifecycle, link replay and custom-game menu checks are also included.

The combined candidate passed **2,638 statistics**, **458 tracker**, **184 presentation**, **765 link-replay** and **259 menu** assertions. Nine completed matches and all eleven statistics pages were included. Native screenshots also verify the light button's rendered color. The preservation check verified **12,584 existing method bodies**.

The installer passed **58 checks**, including rejection of failed UI/visual evidence, changed binaries, unknown installations and locked files. After installation, the actual game files passed the tracker, statistics and menu suites again (**458 / 2,638 / 259** assertions). The installed game and AI DLLs match 1.4.18; the helper DLL fingerprint is `F503683EFDD3B6D9B7FB2EECC48B3E74C390D39F564A592E34CAD6A5D723E483`.

The release changes only four helper source files. The game and AI DLLs must match the accepted baseline byte for byte, and all helper methods outside the revised UI classes are checked for preservation. Installation requires passing native and packaging checks, verifies the live DLL fingerprints and retains a recoverable copy of the previous release. Online play remains disabled.

Exact assertion counts and binary fingerprints are recorded in the release's `validation.json`, installed native reports and archive inventory under `artifacts/patch-releases/1.4.19`.
