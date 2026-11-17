# Local game patches

Run the commands from the repository root. Feature folders group build,
install, verification, and archive commands. The managed package folder
contains the shared staging and install commands.
A dash means the feature has no dedicated script for that step.
See the [local patch guide](../../docs/LOCAL_PATCH.md) for the managed package.

## Package and runtime

| Feature | Build or package | Install or restore | Verify or archive |
| --- | --- | --- | --- |
| Rules and local AI bundle | [build-patch.ps1](managed-patch/build-patch.ps1)<br>[build-local-ai.ps1](managed-patch/build-local-ai.ps1) | [stage-archived-patch.ps1](managed-patch/stage-archived-patch.ps1)<br>[install-patch.ps1](managed-patch/install-patch.ps1)<br>[install-combined-patch.ps1](managed-patch/install-combined-patch.ps1)<br>[uninstall-patch.ps1](managed-patch/uninstall-patch.ps1) | [test-unity-patch.ps1](managed-patch/test-unity-patch.ps1)<br>[test-installed-local-ai.ps1](managed-patch/test-installed-local-ai.ps1) |
| Shared install helper | — | [patch-install-common.ps1](../patch-install-common.ps1) | — |
| Legacy bridge and overlay | [build-legacy.ps1](legacy-bridge/build-legacy.ps1)<br>[build-legacy-overlay.ps1](legacy-bridge/build-legacy-overlay.ps1)<br>[LegacyWeaver.cs](legacy-bridge/LegacyWeaver.cs) | [restore-legacy-overlay.ps1](legacy-bridge/restore-legacy-overlay.ps1)<br>[sync-research-managed.ps1](legacy-bridge/sync-research-managed.ps1) | [verify-legacy-integration.py](legacy-bridge/verify-legacy-integration.py)<br>[check-legacy-bridge.ps1](legacy-bridge/check-legacy-bridge.ps1)<br>[check-legacy-overlay-ui.ps1](legacy-bridge/check-legacy-overlay-ui.ps1) |
| Mixed AI | [build-mixed-ai.py](mixed-ai/build-mixed-ai.py) | [install-mixed-ai.ps1](mixed-ai/install-mixed-ai.ps1) | [verify-mixed-legacy.py](mixed-ai/verify-mixed-legacy.py) |
| Rust local runtime | [build-rust-local-package.py](../build-rust-local-package.py) | [install-rust-local-release.ps1](../install-rust-local-release.ps1)<br>[install-rust-alpha.ps1](../install-rust-alpha.ps1) | [qualify-rust-local.py](../qualify-rust-local.py)<br>[qualify-rust-release.py](../qualify-rust-release.py)<br>[verify-rust-local.py](../verify-rust-local.py)<br>[finalize-rust-local.py](../finalize-rust-local.py)<br>[test-rust-local-upgrade.ps1](../test-rust-local-upgrade.ps1)<br>[test-rust-release-deployment.ps1](../test-rust-release-deployment.ps1)<br>[measure-mono-allocation.py](../measure-mono-allocation.py) |
| P6 laptop personality | [personality_laptop.py](personality-laptop/personality_laptop.py) | Same CLI for install and rollback | [P6 qualification guide](../../docs/personality/QUALIFICATION_P4_P6.md) |

## UI and rules patches

| Feature | Build or package | Install or restore | Verify or archive |
| --- | --- | --- | --- |
| AI selection UI | [build-ai-selection-ui.ps1](ai-selection-ui/build-ai-selection-ui.ps1) | — | [archive-ai-selection-ui-release.ps1](ai-selection-ui/archive-ai-selection-ui-release.ps1) |
| Player selection | [build-player-selection.ps1](player-selection/build-player-selection.ps1) | — | [archive-player-selection-release.ps1](player-selection/archive-player-selection-release.ps1) |
| Game history | [build-game-history.ps1](game-history/build-game-history.ps1) | — | [archive-game-history-release.ps1](game-history/archive-game-history-release.ps1) |
| Turn history fix | [build-turn-history-fix.py](turn-history/build-turn-history-fix.py) | [install-turn-history-fix.ps1](turn-history/install-turn-history-fix.ps1) | [qualify-turn-history-fix.py](turn-history/qualify-turn-history-fix.py) |
| Game finishing margin | [build-game-finishing-margin.ps1](game-finishing-margin/build-game-finishing-margin.ps1) | — | — |
| Network input fix | [build-network-input-fix.ps1](network-input/build-network-input-fix.ps1) | — | [archive-network-input-release.ps1](network-input/archive-network-input-release.ps1) |
| Public history bonus | — | — | [test-public-history-bonus.py](public-history-bonus/test-public-history-bonus.py) |
| Information UI | [build-information-ui.ps1](information-ui/build-information-ui.ps1) | — | [archive-information-ui-release.ps1](information-ui/archive-information-ui-release.ps1) |
| Card tracker | [build-card-tracker.ps1](card-tracker/build-card-tracker.ps1) | — | [archive-card-tracker-release.ps1](card-tracker/archive-card-tracker-release.ps1) |
| Card tracker light | [build-card-tracker-light.ps1](card-tracker/build-card-tracker-light.ps1) | — | [archive-card-tracker-light-release.ps1](card-tracker/archive-card-tracker-light-release.ps1) |
| Card tracker log button | [build-card-tracker-log-button.ps1](card-tracker/build-card-tracker-log-button.ps1) | — | [archive-card-tracker-log-button-release.ps1](card-tracker/archive-card-tracker-log-button-release.ps1) |
| Card tracker for three players | [build-card-tracker-three.py](card-tracker/build-card-tracker-three.py) | [install-card-tracker-three.ps1](card-tracker/install-card-tracker-three.ps1) | — |
| Statistics base and readability | — | — | [archive-statistics-release.ps1](statistics/archive-statistics-release.ps1)<br>[archive-statistics-readability-release.ps1](statistics/archive-statistics-readability-release.ps1) |
| Statistics light | [build-statistics-light.ps1](statistics/build-statistics-light.ps1) | — | [archive-statistics-light-release.ps1](statistics/archive-statistics-light-release.ps1) |
| Statistics review | [build-statistics-review.ps1](statistics/build-statistics-review.ps1) | — | [archive-statistics-review-release.ps1](statistics/archive-statistics-review-release.ps1) |
| Statistics geography | [build-statistics-geography.ps1](statistics/build-statistics-geography.ps1) | — | [archive-statistics-geography-release.ps1](statistics/archive-statistics-geography-release.ps1) |
| Statistics geography refinement | [build-statistics-geography-refinement.ps1](statistics/build-statistics-geography-refinement.ps1) | — | [archive-statistics-geography-refinement-release.ps1](statistics/archive-statistics-geography-refinement-release.ps1) |
| Statistics geography polish | [build-statistics-geography-polish.ps1](statistics/build-statistics-geography-polish.ps1) | — | [archive-statistics-geography-polish-release.ps1](statistics/archive-statistics-geography-polish-release.ps1) |
| Statistics efficiency | [build-statistics-efficiency.ps1](statistics/build-statistics-efficiency.ps1) | — | [archive-statistics-efficiency-release.ps1](statistics/archive-statistics-efficiency-release.ps1) |
| Setup preferences | [build-setup-preferences-patch.py](setup-preferences/build-setup-preferences-patch.py) | [install-setup-preferences-patch.ps1](setup-preferences/install-setup-preferences-patch.ps1) | [qualify-setup-preferences-patch.py](setup-preferences/qualify-setup-preferences-patch.py) |
| Sell replay speed | [fix-sell-replay-speed.py](sell-replay-speed/fix-sell-replay-speed.py) | [install-sell-replay-speed.ps1](sell-replay-speed/install-sell-replay-speed.ps1) | — |
| Execution history | — | — | [archive-execution-release.ps1](execution-history/archive-execution-release.ps1) |
| Link replay | — | — | [archive-link-replay-release.ps1](link-replay/archive-link-replay-release.ps1) |

## Local AI and search patches

| Feature | Build or package | Install or restore | Verify or archive |
| --- | --- | --- | --- |
| Era AI | [build-era-ai.ps1](era-ai/build-era-ai.ps1) | — | [archive-era-release.ps1](era-ai/archive-era-release.ps1) |
| Masked ISMCTS and strength | — | — | [verify-masked-ismcts.py](masked-ismcts/verify-masked-ismcts.py)<br>[archive-masked-release.ps1](masked-ismcts/archive-masked-release.ps1)<br>[archive-masked-strength-release.ps1](masked-ismcts/archive-masked-strength-release.ps1) |
| V4 search time | [build-v4-search-time.ps1](v4-search/build-v4-search-time.ps1) | — | [archive-v4-search-time-release.ps1](v4-search/archive-v4-search-time-release.ps1) |
| V4 four-times search | [build-v4-four-times.ps1](v4-search/build-v4-four-times.ps1) | — | [archive-v4-four-times-release.ps1](v4-search/archive-v4-four-times-release.ps1) |
| NN beta | [nn-beta.py](nn-beta/nn-beta.py) | [install-nn-beta.ps1](nn-beta/install-nn-beta.ps1) | [smoke-installed-nn-beta.ps1](nn-beta/smoke-installed-nn-beta.ps1) |
| NN beta 2 | [build-nn-beta2.py](nn-beta2/build-nn-beta2.py)<br>[package-nn-beta2.py](nn-beta2/package-nn-beta2.py) | [install-nn-beta2.ps1](nn-beta2/install-nn-beta2.ps1) | — |
| NN C_scale | [build-nn-c-scale.py](nn-c-scale/build-nn-c-scale.py) | [install-nn-c-scale.ps1](nn-c-scale/install-nn-c-scale.ps1) | — |
| NN delivery | [fix-nn-delivery.py](nn-delivery/fix-nn-delivery.py) | [install-nn-delivery.ps1](nn-delivery/install-nn-delivery.ps1) | — |
| NN IL | [build-nn-il.py](nn-il/build-nn-il.py)<br>[nn-il.py](nn-il/nn-il.py) | [install-nn-il.ps1](nn-il/install-nn-il.ps1) | [smoke-installed-nn-il.ps1](nn-il/smoke-installed-nn-il.ps1) |
| NN industry focus | [nn-industry-focus.py](nn-industry-focus/nn-industry-focus.py) | [install-nn-industry-focus.ps1](nn-industry-focus/install-nn-industry-focus.ps1) | [smoke-installed-nn-industry-focus.ps1](nn-industry-focus/smoke-installed-nn-industry-focus.ps1) |
| NN K32 | [build-nn-k32.py](nn-k32/build-nn-k32.py)<br>[build-nn-k32-native.py](nn-k32/build-nn-k32-native.py) | [install-nn-k32.ps1](nn-k32/install-nn-k32.ps1) | [qualify-nn-k32.py](nn-k32/qualify-nn-k32.py)<br>[finalize-nn-k32.py](nn-k32/finalize-nn-k32.py)<br>[smoke-installed-nn-k32.ps1](nn-k32/smoke-installed-nn-k32.ps1) |
| NN neutral UI | [nn-neutral-ui.py](nn-neutral-ui/nn-neutral-ui.py) | [install-nn-neutral-ui.ps1](nn-neutral-ui/install-nn-neutral-ui.ps1) | [smoke-installed-nn-neutral-ui.ps1](nn-neutral-ui/smoke-installed-nn-neutral-ui.ps1) |
| NN opening temperature | [build-nn-opening-temperature.py](nn-opening-temperature/build-nn-opening-temperature.py) | [install-nn-opening-temperature.ps1](nn-opening-temperature/install-nn-opening-temperature.ps1) | [qualify-nn-opening-temperature.py](nn-opening-temperature/qualify-nn-opening-temperature.py) |
| NN search choice | [build-nn-search-choice.py](nn-search-choice/build-nn-search-choice.py) | [install-nn-search-choice.ps1](nn-search-choice/install-nn-search-choice.ps1) | [qualify-nn-search-choice.py](nn-search-choice/qualify-nn-search-choice.py)<br>[smoke-installed-nn-search-choice.ps1](nn-search-choice/smoke-installed-nn-search-choice.ps1) |
| NN temperature | [build-nn-temperature.py](nn-temperature/build-nn-temperature.py)<br>[build-nn-temperature-native.py](nn-temperature/build-nn-temperature-native.py) | [install-nn-temperature.ps1](nn-temperature/install-nn-temperature.ps1) | [qualify-nn-temperature.py](nn-temperature/qualify-nn-temperature.py)<br>[smoke-installed-nn-temperature.ps1](nn-temperature/smoke-installed-nn-temperature.ps1) |
| Fixed game reserve | [fixed-game-reserve.py](fixed-game-reserve/fixed-game-reserve.py) | [install-fixed-reserve.ps1](fixed-game-reserve/install-fixed-reserve.ps1) | [check-fixed-reserve-user-data.ps1](fixed-game-reserve/check-fixed-reserve-user-data.ps1)<br>[smoke-installed-fixed-reserve.ps1](fixed-game-reserve/smoke-installed-fixed-reserve.ps1) |
| Host delivery | [build-host-delivery.py](host-delivery/build-host-delivery.py)<br>[build-host-delivery-release.py](host-delivery/build-host-delivery-release.py) | [install-host-delivery.ps1](host-delivery/install-host-delivery.ps1) | [test-host-delivery.py](host-delivery/test-host-delivery.py)<br>[smoke-host-delivery.py](host-delivery/smoke-host-delivery.py)<br>[smoke-host-delivery-release.py](host-delivery/smoke-host-delivery-release.py)<br>[finalize-host-delivery-installation.py](host-delivery/finalize-host-delivery-installation.py)<br>[check-host-delivery-edges.ps1](host-delivery/check-host-delivery-edges.ps1)<br>[check-host-delivery-user-data.ps1](host-delivery/check-host-delivery-user-data.ps1)<br>[smoke-installed-host-delivery.ps1](host-delivery/smoke-installed-host-delivery.ps1) |
| Search delivery grace | [search-delivery-grace.py](search-delivery-grace/search-delivery-grace.py) | [install-search-delivery-grace.ps1](search-delivery-grace/install-search-delivery-grace.ps1) | [check-search-delivery-budget.ps1](search-delivery-grace/check-search-delivery-budget.ps1)<br>[check-search-delivery-grace-user-data.ps1](search-delivery-grace/check-search-delivery-grace-user-data.ps1)<br>[finalize-search-delivery-grace.ps1](search-delivery-grace/finalize-search-delivery-grace.ps1)<br>[smoke-installed-search-delivery-grace.ps1](search-delivery-grace/smoke-installed-search-delivery-grace.ps1) |
| Search v1 default | [build-search-v1-default.py](search-v1/build-search-v1-default.py) | [install-search-v1-default.ps1](search-v1/install-search-v1-default.ps1) | [qualify-search-v1-default.py](search-v1/qualify-search-v1-default.py)<br>[smoke-installed-search-v1.ps1](search-v1/smoke-installed-search-v1.ps1) |
| Search v2 | [build-search-v2-managed.py](search-v2/build-search-v2-managed.py)<br>[search-v2-tools.py](search-v2/search-v2-tools.py) | [install-search-v2.ps1](search-v2/install-search-v2.ps1) | [smoke-search-v2-unity.py](search-v2/smoke-search-v2-unity.py)<br>[verify-search-v2-delivery.py](search-v2/verify-search-v2-delivery.py)<br>[verify-search-v2-native.py](search-v2/verify-search-v2-native.py)<br>[smoke-installed-search-v2.ps1](search-v2/smoke-installed-search-v2.ps1) |
| Search v3 | [build-search-v3.py](search-v3/build-search-v3.py)<br>[search_v3_patch.py](search-v3/search_v3_patch.py) | — | [run-search-v3-smoke.py](search-v3/run-search-v3-smoke.py) |

## Paths retained at the top level

The ten Rust local runtime entries in the table above include the Mono
allocation probe. The L15 and final local release audits pin several of
these files by path and SHA-256 and use the related tools for deployment
checks. The shared `scripts/patch-install-common.ps1` stays where those
installers load it. See the
[migration plan](../../ai-research/migration-plans/LOCAL_GAME_SCRIPT_MIGRATION_PLAN.md) for
the source pins and the first-pass inventory.

For a fresh finalization, use upgrade-test receipts generated with the
relocated installer. To reproduce sealed evidence, use the checkout that
produced it, including its original installer bytes.
