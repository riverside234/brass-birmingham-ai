# Statistics readability — patch 1.4.10

**Status:** Installed on 30 September 2026. All native statistics and startup/menu checks pass in the installed game.

The end-game Statistics panel uses larger text by default. Body/table text increases from 17 to 21 pixels, secondary text and chart labels use 18–19 pixels, section headings use 24 pixels, and main headings use 30 pixels. These are physical display sizes; the game canvas scale is accounted for.

## Using the panel

- Select A+ or A− at the top right to choose Large, Larger or Largest. Large is the enlarged default; Larger and Largest add 15% and 30%. The choice stays with the current match's panel across closing/reopening.
- Player and era filters are easier to click. The status line shows the selection and the number of ledger rows on the page. Search still affects tables, while charts use player/era filters.
- Clear search with its × button. Reset filters also restores every hidden chart series.
- Collapse rows closes expanded details while preserving filters, sorting and scroll position. Top returns to the beginning of the current page.
- Table rows alternate contrasting brown backgrounds. Selected pages, sorted columns and expanded rows have a gold accent. Column headings wrap and stay pinned while scrolling.
- Larger columns and horizontal scrolling keep numbers readable. Wheel/vertical drag scroll the main page over tables; Shift+wheel or sideways drag moves a wide table horizontally. Horizontal positions are retained across rebuilding a page or resizing text.
- Expanded details wrap to the readable page width and have no fixed height cap. Hover definitions size to their content, and chart points have larger hover targets.
- Keyboard focus reveals controls in scrollable lists. Escape dismisses open category choices first; otherwise it closes the panel using its original results/board return behavior.

The panel occupies more of the available screen, with larger navigation buttons, increased spacing and brighter ivory text. The ten pages and all statistics calculations remain the same.

## Scope and compatibility

Only StatisticsPanel.cs and StatisticsWidgets.cs change in the compiled runtime sources. The report collector, projection, page definitions, gameplay rules, save handling and existing presentation hooks are preserved.

The AI DLL and patched Assembly-CSharp.dll remain byte-identical to patch 1.4.9. This preserves the current Masked ISMCTS integration, six strategy choices and Easy/Mid/Hard clocks. The existing atomic three-DLL installer installed the update and created a verified recoverable backup of 1.4.9. Online play remains disabled.

## Verification

The candidate and installed game each pass all **688 native statistics checks** over nine completed fixtures and all ten pages, plus **259 startup/menu checks**. The candidate also passes **184 presentation checks**, **765 link replay checks**, all **12 managed regression suites**, and **79 installer/recovery assertions** against archived 1.4.9.

Native tests cover nine completed fixtures, all ten pages at 1280 × 720, 1920 × 1080 and 2560 × 1440, and the Largest setting on all pages at 720p and 1440p. Offscreen captures use the actual render-target pixel size so they remain representative when Unity batch mode ignores a window-size request. They check text size, wrapped heading height and toolbar bounds, plus reset, clear search, collapse, top, category dismissal, keyboard reveal, modal input and replay lifecycle.

Release qualification requires fresh statistics, presentation, link replay, menu smoke and managed regression evidence for the candidate, and verifies unchanged AI/rules binaries against the fingerprinted 1.4.9 archive.

Twenty-two native offscreen renders cover every page at 1080p and representative pages at 720p/1440p in both Large and Largest. Representative renders were visually inspected for typography, spacing, chart labels, row contrast and navigation. The tests verify every page's font sizes, toolbar bounds and heading fit at the stated resolutions/settings.

Installed runtime fingerprints:

| File | SHA-256 |
| --- | --- |
| BrassPatch.dll | E4E2DC2E34F8E53C1FECA2D72233EC7D8CA76CCF944F62CC2CDC301FB1476669 |
| BrassLegacyAI.dll | 7DDBFAE1B4C600496C5586BBA66631D2634F4EF406B02571E0CA4B8E47CB9C3D |
| Assembly-CSharp.dll | C39FE4084D4AB4DECDC025A883A15A25E731366A954B80D753EAF685B9A5833C |

The verified prior-release backup is artifacts/patch-recovery/136d49180f87482eb06e7a130af46bbd/.

The release archive is artifacts/patch-releases/1.4.10/. To stage the preserved prior release, run scripts/stage-archived-patch.ps1 -Version 1.4.9, then scripts/install-patch.ps1 with the game closed.
