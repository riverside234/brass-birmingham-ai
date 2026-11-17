# Known test failures: the 111 on unmodified HEAD (2026-11-14)

This is the record of the 111 failures that were seen both before and after the CPU-preparation performance work. They fail identically on unmodified HEAD with the same messages, so none of them is caused by a change under review. **Do not analyse them again.** They are all gone from the tree now.

## Status

- **Found:** 2026-11-14, running 23 test modules that touch the teacher, PPG and resume code (266 tests: 4 failures, 107 errors, 1 skipped). The full 748-test run showed the same modules and nothing else.
- **Deleted:** 2026-11-15, at the user's request. 47 methods and 7 emptied classes were removed from 21 modules, and `test_selfplay_major_resume.py` and `test_training_revision_g1307.py` were deleted whole. A check on 2026-11-17 found none of the 111 test names still in `tests/`. 76 of the 111 entries (all of `test_selfplay_major_resume` and `test_training_revision_g1307`) went with their deleted files; the other 35 were removed method by method.
- **Do not restore** the deleted tests, even when rolling code back (user, 2026-11-15 and g2682). Restoring a test file that goes with restored code is fine, as long as these methods stay deleted.
- **A failure today is new.** If a test fails now, check it before calling it pre-existing. The exception is a failure that matches one of the causes below and names a module listed here, which would mean a deleted test came back.

## Causes

| Cause | Failures | What it means |
|---|---|---|
| Old source transition not reviewed | 83 | A test replays an old source-to-source transition (`ValueError: This self-play source transition has not been reviewed`). The reviewed-source list does not cover the old exports' sources. |
| Missing old results export | 15 | The test reads an old `results/joint7-v11-2-g…` export (g1307, g1340, g320 era) that is not in the results repository. |
| Old backend rollback manifest | 2 | `Backend rollback manifest changed`: the manifest the test checks no longer matches current source. |
| Old key, attribute or module | 7 | The test expects a key, attribute or helper module that no longer exists (`provenance`, `contract`, `teacher_phase`, `target_cache_misses`, `TeacherPhase.config`, `test_native_transfer`). |
| Assertion on an old contract or message | 4 | The assertion names an old message or old values (for example "teacher on-mode remains gated", `reference_games`). |
| **Total** | **111** | |

## The failures

Format: `module.Class.test` (number of errors when a test failed more than once). The message is the last error line from the baseline log.

### Old source transition not reviewed

- `test_b025.Resume.test_exact_adoption_and_reviewed_self_toggle` — `ValueError: This self-play source transition has not been reviewed`
- `test_b025.Resume.test_helper_is_readonly_idempotent_generation_hash_and_adoption_bound` — `ValueError: This self-play source transition has not been reviewed`
- `test_gae_floor.Floor.test_exported_signature_review_rejects_unrelated_changes_and_retry` — `ValueError: This self-play source transition has not been reviewed`
- `test_gae_floor.Floor.test_helper_retry_stale_generation_and_missing_checkpoint` — `ValueError: This self-play source transition has not been reviewed`
- `test_local_scoring_resume.Adoption.test_diagnostics_fix_resumes_already_adopted_source_without_config_reset` — `ValueError: This self-play source transition has not been reviewed`
- `test_local_scoring_resume.Adoption.test_g560_review_route_helper_and_mode_only_changes` — `ValueError: This self-play source transition has not been reviewed`
- `test_local_scoring_resume.Adoption.test_source_commit_after_completed_adoption_and_publication_crash_is_restart_safe` — `ValueError: This self-play source transition has not been reviewed`
- `test_ppg_disabled.DisabledEight.test_config_helper_is_locked_idempotent_and_never_edits_run` — `ValueError: This self-play source transition has not been reviewed`
- `test_ppg_throughput.Throughput.test_g580_reviewed_config_helper_is_narrow_and_leaves_run_untouched` — `ValueError: This self-play source transition has not been reviewed`
- `test_selfplay_major_resume.SelfplayMajorResumeTests.test_all_active_presets_require_exact_settings_and_reviewed_source` ×71 — `ValueError: This self-play source transition has not been reviewed`
- `test_selfplay_major_resume.SelfplayMajorResumeTests.test_committed_weights_optimizer_rng_population_and_progress_survive` — `ValueError: This self-play source transition has not been reviewed`
- `test_throughput_g470.Throughput.test_reviewed_g470_helper_preserves_state_and_rejects_other_edits` — `ValueError: This self-play source transition has not been reviewed`
- `test_selfplay_major_resume.SelfplayMajorResumeTests.test_prior_partial_iteration_and_teacher_labels_cannot_be_reused` — `AssertionError: "Uncommitted iteration" does not match "This self-play source transition has not been reviewed"`

### Missing old results export

- `test_diagnostics_cleanup.CurrentContinuation.test_actual_g1340_export_accepts_same_helper_without_rewriting_run` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_diagnostics_cleanup.CurrentContinuation.test_direct_performance_only_review_preserves_calibration_and_exact_config` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_entropy95_revision.Entropy95.test_g320_floors_and_long_term_freeze_without_rebaselining` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_entropy95_revision.Entropy95.test_policy_and_collection_identity_change_and_old_temperatures_rejected` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_entropy95_revision.Entropy95.test_restart_controller_decision_is_identical_and_health_still_blocks` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_entropy95_revision.Entropy95.test_reviewed_resume_needs_explicit_approval_and_no_other_recipe_edits` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_entropy95_revision.Entropy95.test_signed_legacy_recipe_and_only_three_blocks_change` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_entropy95_revision.HotfixHelper.test_isolated_apply_preview_idempotence_and_state_preservation` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_ppg_dense_eight.DenseEight.test_exact_config_schedule_and_fixed_beta` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_ppg_dense_eight.DenseEight.test_g650_review_and_idempotent_config_helper_no_run_mutation` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_ppg_eight.Replay.test_real_g418_transition_review_and_unrelated_config_rejection` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_ppg_sixteen.Schedule.test_g620_source_config_helper_and_unrelated_change_rejection` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_training_revision_g1307.Controller.test_a_multiplier_below_the_new_floor_is_refused_at_resume` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_training_revision_g1307.Controller.test_the_current_rate_is_kept_and_cuts_stop_at_the_floor` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`
- `test_training_revision_g1307.Recipe.setUpClass` — `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Brass Birmingham\\results\\joint`

### Old backend rollback manifest

- `test_b07_shadow_resume.PendingResume.test_transition_commit_and_crash_retry_keep_old_checkpoint_and_pending_bindings` — `ValueError: Backend rollback manifest changed`
- `test_teacher_on_stage_a.Activation.test_real_loader_activation_commit_and_restart_keep_full_state_and_pending` — `ValueError: Backend rollback manifest changed`

### Old key, attribute or module

- `test_b07_shadow_runtime.Cache.test_32_table_post_accounting_charges_each_table_and_its_own_split` — `AttributeError: 'TeacherPhase' object has no attribute 'config'`
- `test_entropy95_revision.BranchOpenings.test_counts_restart_for_all_seats_only_for_new_contract` — `KeyError: 'provenance'`
- `test_g1490_restart.Cotton.test_long_only_labels_allow_short_search_spending_between_them` — `KeyError: 'contract'`
- `test_nn_work_budget.NNWorkTests.test_execution_starts_watchdog_after_cpu_admission_and_shares_both_actions` — `KeyError: 'provenance'`
- `test_post_transfer_reward.Rewards.test_export_queues_new_config_but_restores_original_checkpoint_and_state_bytes` — `ModuleNotFoundError: No module named 'test_native_transfer'`
- `test_ppg_throughput.Throughput.test_auxiliary_stats_deferred_without_changing_gradients` — `KeyError: 'target_cache_misses'`
- `test_shadow_verification.FailureLogs.test_main_exports_child_log_and_bounded_error_tail` — `KeyError: 'teacher_phase'`

### Assertion on an old contract or message

- `test_b045_failures.CollectionFailures.test_value_sealing_charges_each_table_and_still_rejects_an_overrun` — `ERROR: test_value_sealing_charges_each_table_and_still_rejects_an_overrun (test_b045_failures.CollectionFailures.test_value_sealing_charges_`
- `test_b025.Lifecycle.test_live_enablement_fails_before_any_collection_or_state_write` — `AssertionError: "teacher on-mode remains gated" does not match "Teacher on-mode requires the reviewed qualified 32-table recipe"`
- `test_diagnostics_cleanup.AuxiliarySchedule.test_default_512_rows_reduced_metrics_eligibility_and_modes` — `AssertionError: Items in the first set but not the second:`
- `test_ppg_disabled.DisabledEight.test_defaults_and_exact_g460_config_review` — `+  'reference_games': 320}`

## How this was established

The same 266 tests ran on unmodified HEAD (`baseline.log`) and on the working tree (`after2.log`). Both reported `FAILED (failures=4, errors=107, skipped=1)` and the per-test, per-message comparison had no difference in either direction. The modules were: test_b025, test_b045_failures, test_b07_shadow_resume, test_b07_shadow_runtime, test_diagnostics_cleanup, test_entropy95_revision, test_g1190_restart, test_g1490_restart, test_gae_floor, test_local_scoring_resume, test_nn_work_budget, test_post_transfer_reward, test_ppg_dense_eight, test_ppg_disabled, test_ppg_eight, test_ppg_sixteen, test_ppg_throughput, test_selfplay_major_resume, test_shadow_verification, test_teacher_on_stage_a, test_throughput_g470, test_training_revision_g1307.

To run focused tests: `PYTHONPATH="src;scripts;tests/Recovery;tests/RustBackendLab;tests/FeatureEngineering;tests/R2"` with `artifacts/p6e/Scripts/python.exe -m unittest <module>`.

## Also failing on unmodified HEAD (found 2026-11-17, still in the tree)

Found while testing the g3140 value revision. Each fails the same way with HEAD `1bab411b` and all g3140 edits stashed. They are not among the 111 above and have not been deleted.

| Test | Message | Cause |
|---|---|---|
| `test_ppg_g210.Restored.test_exact_g210_options_with_requested_beta_and_rate_changes` | `FileNotFoundError: …grouped-lr-critic-g0660-…\state.json` | Missing old results export |
| `test_ppg_g210.Restored.test_g660_helper_is_idempotent_narrow_and_preserves_run` | same g0660 export | Missing old results export |
| `test_dcpg.Delayed.test_config_narrow_strict_defaults_and_fixed_dynamic_rate` | `FileNotFoundError: …grouped-lr-critic-g0590-…\state.json` | Missing old results export |
| `test_dcpg.Delayed.test_g590_cadence_preserves_origin_phase_counts_and_restart` | `FileNotFoundError: …g0590-…\iteration-0589\report.json` | Missing old results export |
| `test_dcpg.Delayed.test_g590_config_helper_narrow_review_and_no_run_writes` | same g0590 export | Missing old results export |
| `test_dcpg.Delayed.test_ppo_has_no_fresh_fitting_uses_frozen_GAE_and_actual_lr` | `AssertionError: 2 != 8` (`len(v['gradients'])` against `optimizer_steps`) | Assertion on old behaviour; not analysed further |
| `test_value_work_optimization.Targets.test_flat_target_copy_matches_old_recurrence_and_isolates_numeric_and_nested_mutation` | `AssertionError: 999 != 0.0` | Assertion on old behaviour; not analysed further |

## Collection-suite failures not caused by step 9 (found 2026-11-17, still in the tree)

Found while testing build step 9 (ordinary autoplay, [STEP9_ORDINARY_AUTOPLAY.md](../../ai-research/STEP9_ORDINARY_AUTOPLAY.md)). `test_b02`, `test_b06` and `test_b07` give the same 10 failures with `BRASS_ORDINARY_AUTOPLAY=0`, which restores the old code path exactly. None of their tracebacks passes through a file changed since HEAD `1bab411b`. A HEAD worktree fails more of these tests because it lacks the ignored fixture files, so it is not a clean comparison. They have not been deleted.

| Test | Message |
|---|---|
| `test_b02.Contract.test_current_search_v1_b01_receipt_is_accepted_without_lab_work` | `FileNotFoundError: …distribution\rust-lab-b01\b01-search-v1-inputs.zip` |
| `test_b02.Contract.test_reference_route_never_silently_substitutes_unsupported_seat` | `AssertionError: 'rust' != 'reference'` |
| `test_b02.Publication.test_child_can_finish_first_and_join_exactly_once` | `KeyError: 'branch'` |
| `test_b02.Publication.test_one_lane_schedules_child_without_waiting_parent_and_reuses_complete_family` | `KeyError: 'branch'` |
| `test_b06.B06.test_exact_g1250_source_route_preserves_config_restrictions` | `ValueError: This self-play source transition has not been reviewed` |
| `test_b06.B06.test_native_branch_lane_installs_continuation_window_before_sampling` | `AttributeError: … lab_collection … does not have the attribute 'Collector'` (the test patches a name the module no longer imports) |
| `test_b07.B07.test_attested_reads_equal_full_validation_including_branches_and_reference_fallback` (4 subtests) | `TypeError: Admitted actor inputs are immutable` |

## Rating-suite errors with no local export (found 2026-11-17, still in the tree)

Found while testing the g3170 publisher and rating-cadence work ([G3170_PUBLISHER_LEASES_RATING.md](../../ai-research/G3170_PUBLISHER_LEASES_RATING.md)). `test_panel_gates` gives the same 6 errors on the tree without these edits. Each reads `results/joint7-v11-2-grouped-lr-critic-g0270-20261020T233304700657Z`, which is not on this laptop. They have not been deleted. The 10 collection-suite failures above are also unchanged by this work.

| Test | Message |
|---|---|
| `test_panel_gates.PanelGates.test_config_upgrade_exact_blocks_and_temperature_default` | `FileNotFoundError: …grouped-lr-critic-g0270…` |
| `test_panel_gates.PanelGates.test_generation270_reviewed_transition_preserves_counters_and_pool` | same |
| `test_panel_gates.PanelGates.test_schedule_partial_retry_restart_and_main_only_next_slot` | same |
| `test_panel_gates.PanelGates.test_v4_share_90_boundary_and_threshold_migration_rechecks_latest` | same |
| `test_panel_gates.RecoveryTolerance.test_g270_evidence_and_compact_fields` | same |
| `test_panel_gates.ScoredIL.test_real_g270_planner_quotas_probabilities_and_checkpoint_plan_restart` | same |
