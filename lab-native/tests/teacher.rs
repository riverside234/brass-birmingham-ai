#![cfg(feature = "teacher")]
// Re-run the sealed baseline session suite against the additive export overlay.
// Existing recommendation and lifetime assertions remain unchanged.
mod baseline {
    include!("../../native/brass-ai-core/tests/sessions.rs");

    fn objective(i:&brass_ai_core::search::brass::Input,z:usize)->brass_ai_core::search::domino::Objective {
        let manifest:serde_json::Value=serde_json::from_str(include_str!("../../configs/learning/domino-descriptor-1.0.0.json")).unwrap();
        brass_ai_core::search::domino::Objective{owner:u8::from(i.view.seat) as usize,strategy:z,
            coefficient:if z==0 {1.} else {0.5},direction:vec![if z==0 {0.} else {0.02};448],
            bundle_id:"a".repeat(64),slots:serde_json::from_value(manifest["slots"].clone()).unwrap(),
            regions:serde_json::from_value(manifest["regions"].clone()).unwrap(),prefix:0.}
    }

    struct IntrinsicEval<'a> {base:Eval<'a>,owner:usize,intrinsic_calls:usize,expire:usize}
    impl brass_ai_core::search::LeafEvaluator<brass_ai_core::search::brass::Key,String> for IntrinsicEval<'_> {
        fn evaluate(&mut self,i:&brass_ai_core::search::Info<brass_ai_core::search::brass::Key,String>)->brass_ai_core::search::Result<brass_ai_core::search::math::Statistics> {self.base.evaluate(i)}
        fn intrinsic(&mut self,i:&brass_ai_core::search::Info<brass_ai_core::search::brass::Key,String>)->brass_ai_core::search::Result<f64> {
            assert_eq!(i.actor,self.owner);self.intrinsic_calls+=1;
            if self.expire==self.intrinsic_calls {self.base.clock.now.set(3_700_000_000);}
            Ok(-0.2)
        }
    }

    #[test]
    fn domino_neutral_has_identical_count_search_and_no_intrinsic_calls() {
        let run=|enabled| {
            let i=ordinary_input();let c=Fake::default();
            let mut e=IntrinsicEval{base:Eval{clock:&c,calls:0,late_at:0,cancel:false},owner:0,intrinsic_calls:0,expire:0};
            let mut s=Session::new(identity(),&i,settings(24),Budget::new(0,4_000_000_000).unwrap()).unwrap();
            s.enable_teacher_exports().unwrap();if enabled {s.enable_diversity(objective(&i,0)).unwrap();}
            let out=s.decide(&identity(),1,i,4_000_000_000,Mode::Count,&c,&mut e,None).unwrap();
            assert_eq!(e.intrinsic_calls,0);(out,s.take_teacher_export().unwrap())
        };
        let (a,x)=run(false);let (b,y)=run(true);
        assert_eq!(a.action_ids,b.action_ids);assert_eq!(a.gumbels,b.gumbels);
        assert_eq!(x.stats.count,y.stats.count);assert_eq!(x.stats.returns,y.stats.returns);
        assert_eq!(x.policy().unwrap(),y.policy().unwrap());
    }

    #[test]
    fn domino_completed_or_expired_continuation_preserves_evidence() {
        for expire in [0,4] {
            let i=ordinary_input();let c=Fake::default();let o=objective(&i,1);
            let mut e=IntrinsicEval{base:Eval{clock:&c,calls:0,late_at:0,cancel:false},owner:o.owner,intrinsic_calls:0,expire};
            let mut s=Session::new(identity(),&i,settings(24),Budget::new(0,4_000_000_000).unwrap()).unwrap();
            s.enable_teacher_exports().unwrap();s.enable_diversity(o).unwrap();
            let out=s.decide(&identity(),1,i,4_000_000_000,Mode::Count,&c,&mut e,None).unwrap();
            let evidence=s.take_teacher_export().unwrap();let d=evidence.stats.diversity.as_ref().unwrap();
            assert!(e.intrinsic_calls>1);
            assert_eq!(evidence.stats.count.iter().sum::<u64>(),out.completed);
            for edge in 0..evidence.stats.count.len() {
                assert!((evidence.stats.returns[edge][d.owner]-(0.5*d.task_returns[edge]+0.5*d.intrinsic_returns[edge])).abs()<1e-10);
            }
            if expire==0 {assert_eq!(out.completed,24);} else {assert!(out.completed<24);assert!(out.retired_pending>0);}
        }
    }

    #[test]
    fn domino_context_corpus_and_zero_contribution_control() {
        use brass_ai_core::types::Era;
        // Authored permitted roots: finite phase/context math coverage, not a game campaign.
        for (era,round) in [(Era::CanalEra,2),(Era::CanalEra,8),(Era::RailEra,2),(Era::RailEra,8)] {
            for z in 0..7 {
                let mut i=ordinary_input();i.view.context.era=era;i.view.context.round=round;
                if era==Era::RailEra {
                    for p in &mut i.view.players {p.hidden_discards=0;}
                    i.view.deck_count+=4;
                }
                if z>0 {i.context.style[if z<=3 {0} else {2}]=1.;i.context.intent[z-1]=1.;}
                let run=|mode| {
                    let c=Fake::default();let mut o=objective(&i,z);
                    if mode==1 {o.coefficient=1.;}
                    let mut e=IntrinsicEval{base:Eval{clock:&c,calls:0,late_at:0,cancel:false},owner:o.owner,intrinsic_calls:0,expire:0};
                    let mut s=Session::new(identity(),&i,settings(2),Budget::new(0,4_000_000_000).unwrap()).unwrap();
                    s.enable_teacher_exports().unwrap();if mode>0 {s.enable_diversity(o).unwrap();}
                    let out=s.decide(&identity(),1,i.clone(),4_000_000_000,Mode::Count,&c,&mut e,None).unwrap();
                    (out,s.take_teacher_export().unwrap(),e.intrinsic_calls)
                };
                let (a,x,_)=run(0);let (b,y,calls)=run(1);let (active,evidence,_) =run(2);
                assert_eq!(a.action_ids,b.action_ids);assert_eq!(a.gumbels,b.gumbels);
                assert_eq!(x.stats.count,y.stats.count);assert_eq!(x.stats.returns,y.stats.returns);
                assert_eq!(x.policy().unwrap(),y.policy().unwrap());assert_eq!(calls,0);
                assert_eq!(active.completed,2);
                let channels=evidence.stats.diversity.as_ref().unwrap();
                for edge in 0..evidence.stats.count.len() {
                    assert!((evidence.stats.returns[edge][channels.owner]-(channels.coefficient*channels.task_returns[edge]
                        +(1.-channels.coefficient)*channels.intrinsic_returns[edge])).abs()<1e-10);
                }
            }
        }
    }

    #[test]
    fn domino_action_two_prefix_and_context_are_revalidated() {
        for corrupt in [false,true] {
            let i=ordinary_input();let c=Fake::default();let o=objective(&i,3);
            let mut e=IntrinsicEval{base:Eval{clock:&c,calls:0,late_at:0,cancel:false},owner:o.owner,intrinsic_calls:0,expire:0};
            let mut s=Session::new(identity(),&i,settings(24),Budget::new(0,4_000_000_000).unwrap()).unwrap();
            s.enable_teacher_exports().unwrap();s.enable_diversity(o).unwrap();
            let first=s.decide(&identity(),1,i.clone(),4_000_000_000,Mode::Count,&c,&mut e,None).unwrap();
            let info=&first.actions[1].action.steps[0].precondition;
            let mut next=i;next.view=info.key.information.observation.view.clone().unwrap();
            next.context=info.key.context.clone();next.recall=vec![];
            s.take_teacher_export();
            s.set_intrinsic_prefix(info.key.intrinsic_prefix+if corrupt {0.01} else {0.}).unwrap();
            c.now.set(1_000_000_000);
            let second=s.decide(&identity(),2,next,4_000_000_000,Mode::Count,&c,&mut e,None).unwrap();
            assert_eq!(second.budget.deadline_ns,4_000_000_000);
            if let Some(label)=s.take_teacher_export() {
                assert_eq!(label.continuation,!corrupt);
                assert_eq!(label.info.key.intrinsic_prefix,info.key.intrinsic_prefix+if corrupt {0.01} else {0.});
            } else {assert!(corrupt);}
        }
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
