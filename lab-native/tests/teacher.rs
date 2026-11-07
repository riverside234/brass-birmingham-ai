#![cfg(feature = "teacher")]
// Re-run the sealed baseline session suite against the additive export overlay.
// Existing recommendation and lifetime assertions remain unchanged.
mod baseline {
    include!("../../native/brass-ai-core/tests/sessions.rs");

    #[test]
    fn teacher_later_turns_keep_absolute_deadlines_and_completed_work() {
        use brass_lab_collector::teacher::{decide_turn, profile_budget};
        for fast in [false, true] {
            for started in [0, 12_000_000_000, 120_000_000_000] {
                let i = input(); let c = Fake::default(); c.now.set(started);
                let budget = profile_budget(started, fast).unwrap();
                let mut s = Session::new(identity(), &i, settings(8), budget).unwrap();
                s.enable_teacher_exports().unwrap();
                let mut e = Eval { clock: &c, calls: 0, late_at: 0, cancel: false };
                let out = decide_turn(&mut s,&identity(),1,i,Mode::Count,&c,&mut e).unwrap();
                assert_eq!(out.budget.deadline_ns,budget.deadline_ns);
                assert_eq!(out.budget.work_deadline_ns,budget.work_deadline_ns);
                assert_eq!(out.completed,8);
                assert!(s.take_teacher_export().is_some());
            }
        }
    }

    #[test]
    fn teacher_second_action_keeps_original_later_turn_cutoffs() {
        use brass_lab_collector::teacher::{decide_turn, profile_budget};
        for fast in [false,true] {
            let mut i=ordinary_input();
            i.ledger.settled=vec![(brass_ai_core::types::Era::CanalEra,1)];
            let c=Fake::default(); let started=120_000_000_000; c.now.set(started);
            let budget=profile_budget(started,fast).unwrap();
            let mut s=Session::new(identity(),&i,settings(24),budget).unwrap();
            s.enable_teacher_exports().unwrap();
            let mut e=Eval { clock:&c,calls:0,late_at:0,cancel:false };
            let first=decide_turn(&mut s,&identity(),1,i.clone(),Mode::Count,&c,&mut e).unwrap();
            assert_eq!(first.actions.len(),2);
            let info=&first.actions[1].action.steps[0].precondition;
            let mut next=i; next.view=info.key.information.observation.view.clone().unwrap();
            next.context=info.key.context.clone(); next.recall=vec![];
            c.now.set(budget.work_deadline_ns-50_000_000);
            let second=decide_turn(&mut s,&identity(),2,next,Mode::Count,&c,&mut e).unwrap();
            assert_eq!(second.budget.deadline_ns,budget.deadline_ns);
            assert_eq!(second.budget.work_deadline_ns,budget.work_deadline_ns);
            assert!(s.take_teacher_export().unwrap().continuation);
        }
    }

    #[test]
    fn fast_profile_cutoffs_and_root_mean() {
        use brass_lab_collector::teacher::{profile_budget, root_value};
        let mut b = profile_budget(50, true).unwrap();
        assert_eq!(b.work_deadline_ns, 800_000_050);
        assert_eq!(b.deadline_ns, 1_300_000_050);
        assert_eq!(b.reserve_ns, 500_000_000);
        b.narrow(700_000_050, 9_000_000_000);
        assert_eq!(b.deadline_ns, 1_300_000_050);
        let c = Fake::default();
        c.now.set(800_000_050); assert!(!b.work_open(&c)); assert!(b.delivery_open(&c));
        c.now.set(1_300_000_049); assert!(b.delivery_open(&c));
        c.now.set(1_300_000_050); assert!(!b.delivery_open(&c));
        assert!(profile_budget(u64::MAX, true).is_err());
        let mut stats = brass_ai_core::search::math::Statistics::new(vec![0., 0.], [1.,2.,3.,4.]).unwrap();
        assert_eq!(root_value(&stats).unwrap(), [1.,2.,3.,4.]);
        stats.count = vec![2, 1];
        stats.returns = vec![[2.,4.,6.,8.], [1.,3.,5.,7.]];
        assert_eq!(root_value(&stats).unwrap(), [1.,2.25,3.5,4.75]);
    }

    #[test]
    fn fast_k4_keeps_completed_rounds_and_late_reply_guards() {
        for late in [0, 3, 7] {
            let i = input(); let c = Fake::default();
            let mut e = Eval { clock: &c, calls: 0, late_at: late, cancel: false };
            let mut options = settings(32); options.search.k = 4;
            let mut s = Session::new(identity(), &i, options,
                brass_lab_collector::teacher::profile_budget(0,true).unwrap()).unwrap();
            s.enable_teacher_exports().unwrap();
            let out = decide(&mut s,i,&c,&mut e);
            assert!(out.effective_k <= 4);
            assert!(out.completed <= 32);
            if late == 0 { assert_eq!(out.snapshot.unwrap().completed_rounds,2); }
        }
    }

    #[test]
    fn full_four_second_search_plus_one_second_post_never_replenishes() {
        use brass_lab_collector::teacher::search_budget;
        let mut b = search_budget(10).unwrap();
        assert_eq!(b.work_deadline_ns, 4_000_000_010);
        assert_eq!(b.deadline_ns, 5_000_000_010);
        assert_eq!(b.reserve_ns, 1_000_000_000);
        let c = Fake::default();
        c.now.set(3_900_000_010); assert!(b.work_open(&c));
        b.narrow(c.now.get(), 99_000_000_000);
        assert_eq!(b.work_deadline_ns, 4_000_000_010);
        c.now.set(4_000_000_010); assert!(!b.work_open(&c)); assert!(b.delivery_open(&c));
        c.now.set(5_000_000_010); assert!(!b.delivery_open(&c));
        assert!(search_budget(u64::MAX).is_err());
    }

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
