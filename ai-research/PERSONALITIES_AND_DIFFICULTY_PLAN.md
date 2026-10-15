# Personalities and difficulty calibration

This document has been consolidated into [DESIGN.md](../docs/personality/DESIGN.md#design). Use the [P0-P6 documentation index](../docs/personality/README.md) for the maintained guides and pending work. This file contains navigation only.

<details>
<summary>Previous section links</summary>

<a id="1-decisions-and-boundaries"></a>

[1. Decisions and boundaries](../docs/personality/DESIGN.md#design-1-decisions-and-boundaries)

<a id="11-current-step-5-foundation-to-carry-forward"></a>

[1.1 Current Step 5 foundation to carry forward](../docs/personality/DESIGN.md#design-11-current-step-5-foundation-to-carry-forward)

<a id="12-entry-gate-completed-step-5-self-training"></a>

[1.2 Entry gate: completed Step 5 self-training](../docs/personality/DESIGN.md#design-12-entry-gate-completed-step-5-self-training)

<a id="13-shared-return-and-utility-contract"></a>

[1.3 Shared return and utility contract](../docs/personality/DESIGN.md#design-13-shared-return-and-utility-contract)

<a id="2-separate-personality-current-intent-and-discovered-behavior"></a>

[2. Separate personality, current intent and discovered behavior](../docs/personality/DESIGN.md#design-2-separate-personality-current-intent-and-discovered-behavior)

<a id="3-unified-network-and-persistent-intent"></a>

[3. Unified network and persistent intent](../docs/personality/DESIGN.md#design-3-unified-network-and-persistent-intent)

<a id="31-architecture"></a>

[3.1 Architecture](../docs/personality/DESIGN.md#design-31-architecture)

<a id="32-continuity-without-rigid-commitments"></a>

[3.2 Continuity without rigid commitments](../docs/personality/DESIGN.md#design-32-continuity-without-rigid-commitments)

<a id="33-goal-labels-and-joint-training"></a>

[3.3 Goal labels and joint training](../docs/personality/DESIGN.md#design-33-goal-labels-and-joint-training)

<a id="4-soft-priors-that-preserve-tactical-alternatives"></a>

[4. Soft priors that preserve tactical alternatives](../docs/personality/DESIGN.md#design-4-soft-priors-that-preserve-tactical-alternatives)

<a id="41-definition-and-bounded-influence"></a>

[4.1 Definition and bounded influence](../docs/personality/DESIGN.md#design-41-definition-and-bounded-influence)

<a id="42-integration-with-masked-ismcts--nn"></a>

[4.2 Integration with Masked ISMCTS + NN](../docs/personality/DESIGN.md#design-42-integration-with-masked-ismcts--nn)

<a id="43-candidate-coverage-and-final-selection"></a>

[4.3 Candidate coverage and final selection](../docs/personality/DESIGN.md#design-43-candidate-coverage-and-final-selection)

<a id="44-teacher-reservations-readiness-and-time-accounting"></a>

[4.4 Teacher reservations, readiness and time accounting](../docs/personality/DESIGN.md#design-44-teacher-reservations-readiness-and-time-accounting)

<a id="5-aurora-inspired-discovery-of-additional-personalities"></a>

[5. AURORA-inspired discovery of additional personalities](../docs/personality/DESIGN.md#design-5-aurora-inspired-discovery-of-additional-personalities)

<a id="50-product-target-16-visible-archetypes-and-128-hidden-variants"></a>

[5.0 Product target: 16 visible archetypes and 128 hidden variants](../docs/personality/DESIGN.md#design-50-product-target-16-visible-archetypes-and-128-hidden-variants)

<a id="51-offline-role-and-descriptor-data"></a>

[5.1 Offline role and descriptor data](../docs/personality/DESIGN.md#design-51-offline-role-and-descriptor-data)

<a id="52-small-initial-encoder-and-archive"></a>

[5.2 Small initial encoder and archive](../docs/personality/DESIGN.md#design-52-small-initial-encoder-and-archive)

<a id="53-select-16-archetypes-from-qualified-candidates"></a>

[5.3 Select 16 archetypes from qualified candidates](../docs/personality/DESIGN.md#design-53-select-16-archetypes-from-qualified-candidates)

<a id="54-populate-eight-nearby-micro-variants-per-archetype"></a>

[5.4 Populate eight nearby micro-variants per archetype](../docs/personality/DESIGN.md#design-54-populate-eight-nearby-micro-variants-per-archetype)

<a id="55-runtime-sampling-persistence-and-ui"></a>

[5.5 Runtime sampling, persistence and UI](../docs/personality/DESIGN.md#design-55-runtime-sampling-persistence-and-ui)

<a id="56-bring-discovered-behavior-back-into-one-network"></a>

[5.6 Bring discovered behavior back into one network](../docs/personality/DESIGN.md#design-56-bring-discovered-behavior-back-into-one-network)

<a id="6-named-styles-and-player-facing-intent"></a>

[6. Named styles and player-facing intent](../docs/personality/DESIGN.md#design-6-named-styles-and-player-facing-intent)

<a id="7-difficulty-and-enjoyable-four-player-play"></a>

[7. Difficulty and enjoyable four-player play](../docs/personality/DESIGN.md#design-7-difficulty-and-enjoyable-four-player-play)

<a id="71-separate-strength-style-and-waiting-time"></a>

[7.1 Separate strength, style and waiting time](../docs/personality/DESIGN.md#design-71-separate-strength-style-and-waiting-time)

<a id="72-calibration-evidence"></a>

[7.2 Calibration evidence](../docs/personality/DESIGN.md#design-72-calibration-evidence)

<a id="73-keep-development-evaluation-inexpensive"></a>

[7.3 Keep development evaluation inexpensive](../docs/personality/DESIGN.md#design-73-keep-development-evaluation-inexpensive)

<a id="8-versioned-interfaces-and-implementation-map"></a>

[8. Versioned interfaces and implementation map](../docs/personality/DESIGN.md#design-8-versioned-interfaces-and-implementation-map)

<a id="81-lab-execution-and-retention-contract"></a>

[8.1 Lab execution and retention contract](../docs/personality/DESIGN.md#design-81-lab-execution-and-retention-contract)

<a id="9-delivery-sequence-and-focused-exit-checks"></a>

[9. Delivery sequence and focused exit checks](../docs/personality/DESIGN.md#design-9-delivery-sequence-and-focused-exit-checks)

<a id="10-first-lab-experiment-after-step-5-finishes"></a>

[10. First lab experiment after Step 5 finishes](../docs/personality/DESIGN.md#design-10-first-lab-experiment-after-step-5-finishes)

</details>
