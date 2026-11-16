# Local game script migration plan

Draft for implementation and revision. Inventory checked on 2026-11-16 at
`1ff3fbd`.

## Scope and target

Move the local game tools into `scripts/local-game/<feature>/`, with each
feature's build, install, verification, and archive tools together. Move the
scripts by default and revise their root lookup, imports, callers, tests, and
command examples where needed. Preserve filenames, CLI arguments and defaults,
exit behavior, candidate outputs, and install, recovery, and rollback behavior.

The [local-game index](../scripts/local-game/README.md) contains **154 files**
across 49 rows. It lists 99 PowerShell scripts, 54 Python scripts, and the
legacy C# weaver. Include the P6 laptop install and rollback CLI and the
Mono allocation probe used in the Rust release audit, for **156 candidates**
in total. Target **145 moves in the first pass**, reserving the nine Rust
runtime files, the allocation probe, and the shared installer helper for
closer review. Reassess that group after the bulk moves and relocate any
files that can pass their existing checks.

Step 5 training, lab runtime builders and validators, CI entry points, and
standalone report and benchmark commands keep their own workflow locations.
Update their calls into local-game tools if they have any. Keep the inventories
of sealed releases and historical lab bundles intact.

## Destination map

Every file in the named index rows goes to the indicated folder under
`scripts/local-game/`. The index supplies the exact filenames and roles.
Keep variants of the same patch in the same folder. This covers all 154
indexed files, including helpers and the C# weaver.

### Package and runtime

| Proposed folder | Rows in the current local-game index | Files |
| --- | --- | ---: |
| `managed-patch/` | Rules and local AI bundle | 8 |
| `_shared/` | Shared install helper | 1 |
| `legacy-bridge/` | Legacy bridge and overlay | 8 |
| `mixed-ai/` | Mixed AI | 3 |
| `rust-runtime/` | Rust local runtime | 9 |

### UI and rules

| Proposed folder | Rows in the current local-game index | Files |
| --- | --- | ---: |
| `ai-selection-ui/` | AI selection UI | 2 |
| `player-selection/` | Player selection | 2 |
| `game-history/` | Game history | 2 |
| `turn-history/` | Turn history fix | 3 |
| `game-finishing-margin/` | Game finishing margin | 1 |
| `network-input/` | Network input fix | 2 |
| `public-history-bonus/` | Public history bonus | 1 |
| `information-ui/` | Information UI | 2 |
| `card-tracker/` | Card tracker, Card tracker light, Card tracker log button, Card tracker for three players | 8 |
| `statistics/` | Statistics base and readability, Statistics light, Statistics review, Statistics geography, Statistics geography refinement, Statistics geography polish, Statistics efficiency | 14 |
| `setup-preferences/` | Setup preferences | 3 |
| `sell-replay-speed/` | Sell replay speed | 2 |
| `execution-history/` | Execution history | 1 |
| `link-replay/` | Link replay | 1 |

### Local AI and search

| Proposed folder | Rows in the current local-game index | Files |
| --- | --- | ---: |
| `era-ai/` | Era AI | 2 |
| `masked-ismcts/` | Masked ISMCTS and strength | 3 |
| `v4-search/` | V4 search time, V4 four-times search | 4 |
| `nn-beta/` | NN beta | 3 |
| `nn-beta2/` | NN beta 2 | 3 |
| `nn-c-scale/` | NN C_scale | 2 |
| `nn-delivery/` | NN delivery | 2 |
| `nn-il/` | NN IL | 4 |
| `nn-industry-focus/` | NN industry focus | 3 |
| `nn-k32/` | NN K32 | 6 |
| `nn-neutral-ui/` | NN neutral UI | 3 |
| `nn-opening-temperature/` | NN opening temperature | 3 |
| `nn-search-choice/` | NN search choice | 4 |
| `nn-temperature/` | NN temperature | 5 |
| `fixed-game-reserve/` | Fixed game reserve | 4 |
| `host-delivery/` | Host delivery | 10 |
| `search-delivery-grace/` | Search delivery grace | 6 |
| `search-v1/` | Search v1 default | 4 |
| `search-v2/` | Search v2 | 7 |
| `search-v3/` | Search v3 | 3 |

Also move [personality_laptop.py](../scripts/personality_laptop.py) to
`personality-laptop/` with its P6 deployment role. The P0–P5 experiments
stay in their experiment folder. Place [measure-mono-allocation.py](../scripts/measure-mono-allocation.py)
in `rust-runtime/` if its release audit can still verify the original receipt.

## How to revise paths

1. **Trace callers before each batch.** Build an old-to-new path list from
   the index. Check direct calls, dot-sources, sibling imports, dynamically
   constructed filenames, compiler response files, fixture copies, generated
   commands, archive member lists, and current docs. Search by filename as well
   as full path across `scripts/`, `tests/`, `src/`, `configs/`,
   `docs/`, `ai-research/`, and `.github/`. Classify receipt references
   by whether the reader opens them from the live checkout or from an archive.
2. **Keep the checkout, game, and package roots distinct.** From a directory
   such as `scripts/local-game/nn-k32/`, the checkout is three levels above
   `$PSScriptRoot`, or `Path(__file__).resolve().parents[3]` in Python.
   Update PowerShell defaults in the existing `param` blocks and use the
   resolved checkout for `src/`, `tests/`, and `artifacts/`. Keep read
   and write targets governed by the existing `-GameRoot`, `-BuildDirectory`,
   and other arguments where present. Test from another working directory and
   in a path with spaces. A copied installer fixture must resolve its own root
   even when it contains no Git metadata or source tree.
3. **Use the script's directory for siblings.** Keep the managed builder and
   its local AI builder together, and keep each qualifier with the builder it
   loads using `Path(__file__).with_name(...)`. Scripts in other feature
   folders use the explicit path from the checkout or local-game root. Update
   the C# compiler input in the legacy overlay builder when moving
   `LegacyWeaver.cs`. Load the same shared install helper for all features
   at its chosen location. Move the helper and its remaining callers in one
   change when it can move safely.
4. **Separate the source copy path from the archive member name.** For
   example, `archive-era-release.ps1` currently copies multiple scripts
   from its own directory and records them under `scripts/<name>`. After
   the move, the live sources will be in several feature folders. Update the
   source list explicitly. Preserve member paths required by existing archive
   readers. If a new source snapshot uses the new layout, record those paths
   in its new receipt and test staging and qualified installation from it.
   If a captured script is executed, stage it with the directory layout it
   expects so its repository and sibling lookups work. Keep old ZIPs,
   member inventories, source hashes, and installer receipts as recorded.

### Callers that span groups

- `check-r1-runtime.ps1` and the legacy overlay checks dot-source
  `sync-research-managed.ps1`. Update the current callable path in the R1
  check if the helper moves. The archived compatibility record retains its
  original evidence. [R1 runtime check](../scripts/check-r1-runtime.ps1) ·
  [Compatibility record](../docs/UI_PATCH_RESEARCH_COMPATIBILITY.md)
- Search v2's `search-v2-tools.py` is used by its local builders and
  verification, the v2 and V205 benchmarks, and the two search v2 summaries.
  Move it with the feature and switch those callers together.
  [Search v2 builder](../scripts/build-search-v2-managed.py) ·
  [Search v2 benchmark](../scripts/benchmarks/benchmark-search-v2.py) ·
  [Search v2 summary](../scripts/reports/summarize-search-v2.py)
- The managed package tests copy installers into a disposable
  `<fixture>/scripts/` tree. Copy them into the new relative layout and
  launch the new fixture paths. Keep the original artifact directories,
  separate game fixture, and artificial saves.
  [Managed packaging regression](../tests/BrassPatch/packaging-regression.ps1) ·
  [Information UI fixture](../tests/BrassPatch/InformationUI/packaging.ps1)

## Implementation order

1. **Pilot one complete patch.** Move the turn-history builder, installer, and
   qualifier as a unit. Adjust their root and sibling paths, the applicable
   tests, and the runnable examples. Use the shared installer helper from its
   existing path for the pilot. Compare the build and packaging behavior in a
   disposable project and game fixture.
2. **Move the managed package commands.** Move build, staging, installation,
   combined installation, uninstall, and smoke checks together. Update the
   packaging fixtures and archive commands that invoke the stager. Inventory
   the dot-sources of `patch-install-common.ps1` before moving it. If a
   retained release script still loads the original sibling, keep the helper
   there for both sets of callers until that dependency is resolved.
3. **Move UI, rules, and search feature families.** Land each feature or
   closely related family with its caller and documentation updates. Reuse
   the pilot's path fixes for the NN builders, installers, and qualifiers.
   Cover NN C-scale and K32 as representative managed and native packages.
   Move the legacy bridge and Search v2 with their cross-workflow updates.
   Move the P6 CLI with its documented staging, certification, and review-copy
   install and rollback flows.
4. **Handle the sealed Rust runtime group last.** Establish which historical
   checks still need the files in the current checkout. When a matching
   checkout or the verified source snapshot satisfies a check, move the
   current tool and document how to reproduce the old release. If a current
   operation still needs exact live source files, retain the smallest
   complete set it needs and record the path and reason in the local-game
   index. Move the rest of the group once its callers work at the new paths.
5. **Finish the cutover.** Update both script READMEs, affected release
   guides, test launch commands, and any instructions printed by the
   scripts. Remove the previous copies of moved scripts once their current
   callers have switched. Keep batches independently revertible, with moves
   and caller updates in the same change. Historical commands can stay in
   sealed evidence with the original checkout as their execution context.

## Rust source pins and other exceptions

The local Rust `release-acceptance.json` pins
`finalize-rust-local.py`, `install-rust-local-release.ps1`,
`test-rust-local-upgrade.ps1`, and `measure-mono-allocation.py` by
checkout-relative path and SHA-256. The L15 acceptance record also pins
`build-rust-local-package.py`, `qualify-rust-release.py`,
`test-rust-release-deployment.ps1`, and the shared native replay
diagnostic. The deployment test invokes `install-rust-alpha.ps1` as a
sibling, and the installers load `patch-install-common.ps1`. A file can
therefore be required even if it does not appear directly in a pin list.
[Final release audit](../scripts/finalize-rust-local.py) ·
[L15 qualification and audit](../scripts/qualify-rust-release.py) ·
[Deployment fixture](../scripts/test-rust-release-deployment.ps1)

Use the matching checkout for old receipts that require live paths. If a
current verifier is revised to consume a retained release snapshot, check
the snapshot and member against the recorded digests and replay the same
verification before removing the original path. Keep all installed and staged
binary checks in place. For current doc references, ordinary imports, and
fixed directory depth, revise the path and move the script.

## Verification for each batch

- Parse migrated PowerShell without invoking installers. Compile or parse the
  Python scripts and run help or read-only checks where supported. Exercise
  calls from the checkout root, another working directory, and the copied
  package fixture. Check parameter and output parity for the supported commands.
- Build the same feature from the same pinned base into separate candidate
  directories before and after migration. For scripts with fixed output
  paths, use separate disposable checkouts. Compare file sets, managed/native
  binaries, and manifest contents. Fresh source locators and script hashes
  can reflect the new paths. Investigate any unexplained binary or rules
  difference before accepting the move.
- Check a sealed managed release with the moved stager's `-VerifyOnly`.
  Exercise qualified install, tamper rejection, and uninstall or rollback in
  a disposable game fixture. For the Rust move, repeat recovery at both
  `NativePublished` and `ManagedPublished` interrupt points and compare
  the previous managed/native hashes, installation record, and artificial
  saves. Use existing fixture checks and keep live Steam and user saves out
  of the test target. [Rust upgrade fixture](../scripts/test-rust-local-upgrade.ps1)
- Rerun the focused callers for each cross-workflow move. Compare Step 5's
  implementation identity and the CPU worker overlay fingerprint against the
  baseline. Check that the transfer workflow still calls the original entry
  point and that new fixture or bundle inventories include the relocated
  dependencies. [Transfer workflow](../.github/workflows/step5-transfer-release.yml)
- Check the final script inventory against this map and click-test the
  updated indexes. Report the moved count, each retained path and its
  dependency, and any test unavailable because its matching private
  candidate or runtime is missing.

Revise folder assignments, batch order, and the retained list when the caller
audit finds a concrete dependency. Update this plan and the index in the same
batch so the inventory reflects what actually shipped.
