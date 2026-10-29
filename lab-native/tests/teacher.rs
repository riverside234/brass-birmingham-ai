#![cfg(feature = "teacher")]
// Re-run the sealed baseline session suite against the additive export overlay.
// Existing recommendation and lifetime assertions remain unchanged.
mod baseline {
    include!("../../native/brass-ai-core/tests/sessions.rs");

    #[test]
    fn b04_exports_keep_recommendations_and_final_partial_watermark() {
        for (n, late, k) in [
            (0, 0, 16),
            (1, 0, 16),
            (24, 13, 4),
            (24, 15, 4),
            (24, 0, 16),
        ] {
            let run = |enabled| {
                let i = input();
                let c = Fake::default();
                let mut e = Eval {
                    clock: &c,
                    calls: 0,
                    late_at: late,
                    cancel: false,
                };
                let mut options = settings(n);
                options.search.k = k;
                let mut s = Session::new(
                    identity(),
                    &i,
                    options,
                    Budget::new(0, 4_000_000_000).unwrap(),
                )
                .unwrap();
                if enabled {
                    s.enable_teacher_exports().unwrap();
                }
                let out = decide(&mut s, i, &c, &mut e);
                let export = s.take_teacher_export();
                (out, export)
            };
            let (reference, none) = run(false);
            assert!(none.is_none());
            let (candidate, export) = run(true);
            assert_eq!(
                serde_json::to_value(&candidate).unwrap(),
                serde_json::to_value(reference).unwrap()
            );
            let evidence = export.unwrap();
            assert_eq!(
                evidence.stats.count.iter().sum::<u64>(),
                candidate.completed
            );
            assert_eq!(evidence.completed, candidate.completed);
            if late == 15 {
                assert_eq!(evidence.completed, 13);
                assert_eq!(candidate.snapshot.unwrap().support_watermark, 12);
            }
            assert!((evidence.policy().unwrap().iter().sum::<f64>() - 1.).abs() < 1e-12);
        }
    }

    #[test]
    fn b04_continuation_snapshot_survives_tree_retirement_but_not_wrong_successor() {
        for changed in [false, true] {
            let mut i = ordinary_input();
            i.ledger.settled = vec![(brass_ai_core::types::Era::CanalEra, 1)];
            let c = Fake::default();
            let mut e = Eval {
                clock: &c,
                calls: 0,
                late_at: 0,
                cancel: false,
            };
            let mut s = Session::new(
                identity(),
                &i,
                settings(24),
                Budget::new(0, 4_000_000_000).unwrap(),
            )
            .unwrap();
            s.enable_teacher_exports().unwrap();
            let out = decide(&mut s, i.clone(), &c, &mut e);
            assert_eq!(out.actions.len(), 2);
            assert!(!s.take_teacher_export().unwrap().continuation);
            let info = &out.actions[1].action.steps[0].precondition;
            let mut next = i;
            next.view = info.key.information.observation.view.clone().unwrap();
            next.context = info.key.context.clone();
            next.recall = vec![];
            if changed {
                next.view.players[info.actor].cash = 0;
            }
            c.now.set(3_700_000_000);
            let second = s
                .decide(
                    &identity(),
                    2,
                    next,
                    4_000_000_000,
                    Mode::Count,
                    &c,
                    &mut e,
                    None,
                )
                .unwrap();
            assert_eq!(second.budget.deadline_ns, 4_000_000_000);
            if changed {
                assert!(s.take_teacher_export().is_none());
            } else {
                let evidence = s.take_teacher_export().unwrap();
                assert!(evidence.continuation);
                assert_eq!(evidence.info, *info);
                assert!(evidence.stats.count.iter().sum::<u64>() > 0);
                assert!(s.take_teacher_export().is_none());
            }
        }
    }

    #[test]
    fn b04_cancel_retires_labels_and_original_allowance() {
        let i = ordinary_input();
        let c = Fake::default();
        let mut e = Eval {
            clock: &c,
            calls: 0,
            late_at: 4,
            cancel: true,
        };
        let mut s = Session::new(
            identity(),
            &i,
            settings(24),
            Budget::new(0, 4_000_000_000).unwrap(),
        )
        .unwrap();
        s.enable_teacher_exports().unwrap();
        let out = decide(&mut s, i, &c, &mut e);
        assert_eq!(out.status, "cancelled");
        assert!(s.take_teacher_export().is_none());
    }

    #[test]
    fn b04_invalid_second_action_replans_with_new_evidence_and_same_clock() {
        let mut i = ordinary_input();
        i.ledger.settled = vec![(brass_ai_core::types::Era::CanalEra, 1)];
        let c = Fake::default();
        let mut e = Eval {
            clock: &c,
            calls: 0,
            late_at: 0,
            cancel: false,
        };
        let mut s = Session::new(
            identity(),
            &i,
            settings(24),
            Budget::new(0, 4_000_000_000).unwrap(),
        )
        .unwrap();
        s.enable_teacher_exports().unwrap();
        let first = decide(&mut s, i.clone(), &c, &mut e);
        let info = &first.actions[1].action.steps[0].precondition;
        let mut next = i;
        next.view = info.key.information.observation.view.clone().unwrap();
        next.context = info.key.context.clone();
        next.recall = vec![];
        next.view.players[info.actor].cash = 0;
        c.now.set(1_000_000_000);
        let second = s
            .decide(
                &identity(),
                2,
                next,
                4_000_000_000,
                Mode::Count,
                &c,
                &mut e,
                None,
            )
            .unwrap();
        assert_eq!(second.budget.deadline_ns, 4_000_000_000);
        assert_eq!(second.completed, 24);
        let evidence = s.take_teacher_export().unwrap();
        assert!(!evidence.continuation);
        assert_ne!(evidence.info, *info);
        let row = brass_lab_collector::teacher::label(
            &evidence,
            &evidence.info,
            &serde_json::json!({}),
            &second.action_ids[0][0],
            "replanned_root",
            second.actions[0].support >= 2,
            second.budget,
            c.now_ns(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row["teacher"]["kind"], "replanned_root");
        assert_eq!(row["view"]["context"]["actions_remaining"], 1);
    }

    #[test]
    fn b04_label_overrun_fallback_zero_and_replanned_cases() {
        use brass_lab_collector::teacher::label;
        for n in [0, 1, 24] {
            let i = input();
            let c = Fake::default();
            let mut e = Eval {
                clock: &c,
                calls: 0,
                late_at: 0,
                cancel: false,
            };
            let mut s = Session::new(
                identity(),
                &i,
                settings(n),
                Budget::new(0, 4_000_000_000).unwrap(),
            )
            .unwrap();
            s.enable_teacher_exports().unwrap();
            let out = decide(&mut s, i, &c, &mut e);
            let evidence = s.take_teacher_export().unwrap();
            let row = label(
                &evidence,
                &evidence.info,
                &serde_json::json!({}),
                &evidence.info.actions[0],
                "replanned_root",
                false,
                out.budget,
                4_700_000_000,
            )
            .unwrap();
            if n == 0 {
                assert!(row.is_none());
                continue;
            }
            let row = row.unwrap();
            assert_eq!(row["teacher"]["overrun_seconds"], 0.7);
            assert_eq!(row["teacher"]["fallback_played"], true);
            assert_eq!(row["teacher"]["kind"], "replanned_root");
            let mut wrong = evidence.info.clone();
            wrong.actor = (wrong.actor + 1) % 4;
            assert!(label(
                &evidence,
                &wrong,
                &serde_json::json!({}),
                &evidence.info.actions[0],
                "root",
                false,
                out.budget,
                0
            )
            .is_err());
        }
    }
}
