use brass_ai_core::{
    features::FeatureContext,
    state::{dto::StateDto, initial::InitialSetup, NativeState},
    types::Seat,
};
use brass_lab_collector::{scoring::Recorder, Lane};
use serde_json::{json, Value};

fn initial() -> StateDto {
    // Use the exact shared archive's catalog, never a second rules definition.
    let catalog: Value = serde_json::from_str(include_str!(
        "../../native/brass-ai-core/data/state-catalog.json"
    ))
    .unwrap();
    let deck = catalog["cards"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| !name.starts_with("wild:"))
        .flat_map(|(name, count)| {
            std::iter::repeat_n(name.clone(), count.as_u64().unwrap() as usize)
        })
        .collect();
    NativeState::initial(InitialSetup {
        order: std::array::from_fn(|s| Seat::try_from(s as u8).unwrap()),
        merchants: serde_json::from_value(catalog["merchant_types"].clone()).unwrap(),
        deck,
    })
    .unwrap()
    .export()
}
fn context() -> FeatureContext {
    FeatureContext {
        model_id: "test".into(),
        reward_id: "policy-only".into(),
        profile_id: "test".into(),
        style: [0.; 4],
        intent: [0.; 8],
        remaining: [0.; 4],
        max_cost: 0.,
        stable_descriptors: None,
    }
}

#[test]
fn rejected_reply_does_not_advance_or_publish() {
    let d = initial();
    let mut lane = Lane::new(d.clone(), "job".into(), "request".into(), vec![0; 64], 10).unwrap();
    let decision = lane.decision().unwrap()["decision"].clone();
    lane.features("request", 0, context(), 0).unwrap();
    let row = json!({"seat":decision["view"]["seat"],"view":decision["view"],"choices":decision["choices"],
        "chosen":0,"teacher":null,"style":[0.,0.,0.,0.],"intent":vec![0.;8],"behavior":{"probability":1.}});
    assert!(lane.append("stale", 0, row.clone(), json!({})).is_err());
    assert!(lane.finish().is_err());
    assert!(lane.row(0).is_err());
    assert_eq!(lane.export(), d);
    let mut corrupt = row;
    corrupt["chosen"] = json!(999999);
    assert!(lane.append("request", 0, corrupt, json!({})).is_err());
    assert_eq!(lane.export(), d);
}

#[test]
fn fork_is_portable_bound_and_deck_only() {
    let d = initial();
    let lane = Lane::new(d.clone(), "job".into(), "request".into(), vec![0; 64], 10).unwrap();
    let cp = lane.checkpoint("request", 0).unwrap();
    let indices = vec![0; d.cards.deck.len() - 1];
    let child = Lane::fork(&cp, "job".into(), "child".into(), 10, &indices).unwrap();
    let mut changed = child.export();
    assert_ne!(changed.cards.deck, d.cards.deck);
    changed.cards.deck = d.cards.deck.clone();
    assert_eq!(changed, d);
    assert!(Lane::fork(&cp, "foreign".into(), "child".into(), 10, &indices).is_err());
    let mut corrupt = cp;
    corrupt["payload"]["state"]["cards"]["deck"][0] = json!("bad");
    assert!(Lane::fork(&corrupt, "job".into(), "child".into(), 10, &indices).is_err());
}

#[test]
fn scoring_cannot_publish_before_completion() {
    let d = initial();
    assert!(Recorder::new(&d).finish(&d).is_err());
}

#[test]
fn paging_rejects_a_context_or_revision_change() {
    let mut lane = Lane::new(initial(), "job".into(), "request".into(), vec![0; 64], 10).unwrap();
    assert!(lane.features("request", 1, context(), 0).is_err());
    lane.features("request", 0, context(), 0).unwrap();
    let mut c = context();
    c.model_id = "changed".into();
    assert!(lane.features("request", 0, c, 64).is_err());
}

#[test]
fn ordinary_draw_keeps_hidden_deck_order_and_one_revision_per_stage() {
    let mut d = initial();
    d.cards.deck.rotate_left(3);
    let deck = d.cards.deck.clone();
    let mut lane = Lane::new(d, "job".into(), "request".into(), vec![0; 64], 10).unwrap();
    for (revision, kind) in ["pass", "finish"].into_iter().enumerate() {
        let decision = lane.decision().unwrap()["decision"].clone();
        let chosen = decision["choices"]
            .as_array()
            .unwrap()
            .iter()
            .position(|v| v["kind"] == kind)
            .unwrap();
        lane.features("request", revision as u32, context(), 0)
            .unwrap();
        lane.append("request", revision as u32, json!({
            "seat":decision["view"]["seat"],"view":decision["view"],"choices":decision["choices"],
            "chosen":chosen,"teacher":null,"style":vec![0.;4],"intent":vec![0.;8],
            "behavior":{"probability":1.}
        }), json!({})).unwrap();
    }
    let after = lane.export();
    assert_eq!(after.context.revision, 2);
    assert_eq!(after.cards.deck, deck[1..]);
    assert_eq!(after.cards.hands[0].last(), deck.first());
    assert!(lane.row(0).is_err()); // Complete-game publication only.
}

fn ordinary_context() -> FeatureContext {
    FeatureContext {
        reward_id: "ordinary-policy-only".into(),
        profile_id: brass_lab_collector::CONTRACT.into(),
        ..context()
    }
}

fn multi_beer_sale() -> StateDto {
    use brass_ai_core::{state::dto::Industry, types::Phase};
    let mut d = initial();
    let owner = usize::from(u8::from(d.context.turn_owner));
    let mut place = |industry, beer| {
        let mat = d.players[owner]
            .mat
            .iter_mut()
            .find(|m| m.industry == industry)
            .unwrap();
        let index = mat
            .tiles
            .iter()
            .position(|t| match beer {
                Some(n) => t.beer == n,
                None => true,
            })
            .unwrap();
        let mut tile = mat.tiles.remove(index);
        if industry == Industry::Brewery {
            tile.resource = 1;
        }
        let slot = d
            .slots
            .iter_mut()
            .find(|s| s.tile.is_none() && s.types.contains(&industry))
            .unwrap();
        let id = slot.id.clone();
        slot.tile = Some(tile);
        id
    };
    let target = place(Industry::Manufacturer, Some(2));
    place(Industry::Brewery, None);
    // The merchant's type assignment is shuffled; preserve its exact multiset.
    let merchant = d
        .merchants
        .iter()
        .position(|m| m.id == "Gloucester:0")
        .unwrap();
    let donor = d
        .merchants
        .iter()
        .position(|m| {
            m.types
                .as_ref()
                .is_some_and(|t| t.contains(&Industry::Manufacturer))
        })
        .unwrap();
    let types = d.merchants[merchant].types.clone();
    d.merchants[merchant].types = d.merchants[donor].types.clone();
    d.merchants[donor].types = types;
    for m in &mut d.merchants {
        m.beer = u8::from(m.types.as_ref().is_some_and(|t| !t.is_empty()));
    }
    d.context.phase = Phase::Payment;
    d.pending.kind = "sell".into();
    d.pending.target = target;
    d.pending.merchant = "Gloucester:0".into();
    d.pending.parts = 1;
    d.pending.beer = 2;
    NativeState::import(&d).unwrap();
    d
}

#[test]
fn ordinary_multi_beer_bonus_keeps_typed_state_for_skip_and_develop() {
    use brass_ai_core::{state::dto::Kind, types::Phase};
    for bonus in [Kind::BonusSkip, Kind::BonusDevelop] {
        let d = multi_beer_sale();
        let mut state = NativeState::import(&d).unwrap();
        let mut recorder = Recorder::new(&d);
        for kind in [Kind::Beer, bonus, Kind::Beer] {
            let choices = state.legal_choices().unwrap();
            let choice = choices
                .iter()
                .find(|c| {
                    c.kind == kind
                        && (kind != Kind::Beer
                            || c.target.as_deref() == Some("merchant:Gloucester:0")
                            || state.context().phase == Phase::Payment
                                && state.export().pending.beer == 1)
                })
                .unwrap();
            let expected = state
                .after_choice(state.context().revision, &choice.id)
                .unwrap();
            state =
                brass_lab_collector::ordinary_after(&state, &choice.id, &[0; 64], &mut recorder)
                    .unwrap();
            // Exact legal transition, not a second implementation of the rules.
            assert_eq!(state.export(), expected.export());
            assert_eq!(
                state.legal_choices().unwrap(),
                expected.legal_choices().unwrap()
            );
            if state.context().phase == Phase::Bonus {
                assert_eq!(state.export().pending.beer, 1);
                // The frozen external importer has a stricter idle-bonus rule.
                assert!(NativeState::import(&state.export())
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("bonus state mismatch"));
                assert!(Lane::new(
                    state.export(),
                    "job".into(),
                    "request".into(),
                    vec![0; 64],
                    10
                )
                .is_err());
            }
        }
    }
}

#[test]
fn compact_multi_beer_bonus_preserves_features_forced_choices_and_stale_rejection() {
    let make = || {
        Lane::new(
            multi_beer_sale(),
            "job".into(),
            "request".into(),
            vec![0; 64],
            30,
        )
        .unwrap()
    };
    for bonus in ["bonus_skip", "bonus_develop"] {
        let (mut full, mut compact_lane) = (make(), make());
        let mut forced_count = 0;
        for kind in ["beer", bonus] {
            let frame = compact_lane.decision().unwrap();
            assert_eq!(frame, full.decision().unwrap());
            let choices = frame["decision"]["choices"].as_array().unwrap();
            let chosen = choices
                .iter()
                .position(|c| {
                    c["kind"] == kind && (kind != "beer" || c["target"] == "merchant:Gloucester:0")
                })
                .unwrap();
            let value = compact(&frame, chosen);
            let revision = value["revision"].as_u64().unwrap() as u32;
            assert_eq!(
                compact_lane
                    .feature_arrays("request", revision, ordinary_context())
                    .unwrap(),
                full.feature_arrays("request", revision, ordinary_context())
                    .unwrap()
            );
            let before = compact_lane.export();
            let mut stale = value.clone();
            stale["revision"] = json!(revision + 1);
            assert!(compact_lane.append_compact("request", &stale).is_err());
            assert_eq!(compact_lane.export(), before);
            let start = compact_lane.public_context().unwrap();
            old_step(&mut full, &frame, &value);
            compact_lane.append_compact("request", &value).unwrap();
            let frames = compact_lane.preview_forced(&start, 16).unwrap();
            forced_count += frames.len();
            if !frames.is_empty() {
                let values: Vec<_> = frames.iter().map(|f| compact(f, 0)).collect();
                for (frame, value) in frames.iter().zip(&values) {
                    assert_eq!(frame, &full.decision().unwrap());
                    old_step(&mut full, frame, value);
                }
                let held = compact_lane.export();
                let mut stale = values.clone();
                stale[0]["action_id"] = json!("foreign");
                assert!(compact_lane.acknowledge_forced("request", &stale).is_err());
                assert_eq!(compact_lane.export(), held);
                compact_lane.acknowledge_forced("request", &values).unwrap();
            }
            assert_eq!(compact_lane.export(), full.export());
        }
        assert!(forced_count > 0);
        assert!(compact_lane.finish().is_err());
        assert!(compact_lane.row(0).is_err());
    }
}
fn compact(decision: &Value, chosen: usize) -> Value {
    let choices = decision["decision"]["choices"].as_array().unwrap();
    json!({"revision":decision["decision"]["view"]["context"]["revision"],
        "action_id":choices[chosen]["id"],"context":ordinary_context(),"seconds":0.,
        "behavior":{"probability":1.,"policy_id":"test"},
        "evidence":{"kind":"ordinary","model_id":"test","policy_id":"test",
            "forced":choices.len()==1,"semantic_id":choices[chosen]["id"],
            "feature_version":"permitted-features/4.0.0"}})
}
fn old_step(lane: &mut Lane, decision: &Value, value: &Value) {
    let d = &decision["decision"];
    let chosen = d["choices"]
        .as_array()
        .unwrap()
        .iter()
        .position(|x| x["id"] == value["action_id"])
        .unwrap();
    let revision = value["revision"].as_u64().unwrap() as u32;
    lane.features("request", revision, ordinary_context(), 0)
        .unwrap();
    lane.append("request",revision,json!({"step":0,"seat":d["view"]["seat"],"view":d["view"],
        "choices":d["choices"],"chosen":chosen,"teacher":null,"style":vec![0.;4],"intent":vec![0.;8],
        "seconds":0.,"behavior":value["behavior"]}),value["evidence"].clone()).unwrap();
}

#[test]
fn binary_features_equal_pages_and_reject_stale_identity() {
    let mut lane = Lane::new(initial(), "job".into(), "request".into(), vec![0; 64], 1000).unwrap();
    let (meta, bytes) = lane
        .feature_arrays("request", 0, ordinary_context())
        .unwrap();
    let count = meta["total_candidates"].as_u64().unwrap() as usize;
    assert_eq!(bytes.len(), (3128 + 327 * count) * 4 + 32);
    for start in (0..count).step_by(64) {
        let page = lane
            .features("request", 0, ordinary_context(), start)
            .unwrap();
        for (i, x) in page["state"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                &bytes[i * 4..i * 4 + 4],
                &(x.as_f64().unwrap() as f32).to_le_bytes()
            );
        }
        for (i, x) in page["actions"].as_array().unwrap().iter().enumerate() {
            let offset = (3128 + start * 327 + i) * 4;
            assert_eq!(
                &bytes[offset..offset + 4],
                &(x.as_f64().unwrap() as f32).to_le_bytes()
            );
        }
    }
    assert!(lane.feature_arrays("stale", 0, ordinary_context()).is_err());
    assert!(lane
        .feature_arrays("request", 1, ordinary_context())
        .is_err());
    let d = lane.decision().unwrap();
    let mut bad = compact(&d, 0);
    bad["action_id"] = json!("invalid");
    let before = lane.export();
    assert!(lane.append_compact("request", &bad).is_err());
    assert_eq!(before, lane.export());
    bad = compact(&d, 0);
    bad["context"]["model_id"] = json!("foreign");
    assert!(lane.append_compact("request", &bad).is_err());
    assert_eq!(before, lane.export());
}

#[test]
fn compact_forced_batches_preserve_every_record_state_scoring_and_export_seal() {
    let create =
        || Lane::new(initial(), "job".into(), "request".into(), vec![0; 64], 1000).unwrap();
    let (mut old, mut new) = (create(), create());
    let mut forced_count = 0;
    while !new.export().context.terminal {
        let d = new.decision().unwrap();
        assert_eq!(d, old.decision().unwrap());
        let choices = d["decision"]["choices"].as_array().unwrap();
        let chosen = choices
            .iter()
            .position(|x| x["kind"] == "pass")
            .unwrap_or(0);
        let value = compact(&d, chosen);
        let revision = value["revision"].as_u64().unwrap() as u32;
        let start = new.public_context().unwrap();
        new.feature_arrays("request", revision, ordinary_context())
            .unwrap();
        old_step(&mut old, &d, &value);
        new.append_compact("request", &value).unwrap();
        let frames = new.preview_forced(&start, 16).unwrap();
        if !frames.is_empty() {
            forced_count += frames.len();
            let state = new.export();
            assert!(new.finish().is_err());
            assert!(new.checkpoint("request", state.context.revision).is_err());
            assert!(new.record_arrays("request", 0).is_err());
            assert!(new.acknowledge_forced("request", &[]).is_err());
            let values = frames.iter().map(|f| compact(f, 0)).collect::<Vec<_>>();
            let mut bad = values.clone();
            bad[0]["revision"] = json!(99999);
            assert!(new.acknowledge_forced("request", &bad).is_err());
            assert_eq!(state, new.export());
            for (frame, value) in frames.iter().zip(&values) {
                assert_eq!(frame, &old.decision().unwrap());
                old_step(&mut old, frame, value);
            }
            new.acknowledge_forced("request", &values).unwrap();
        }
        assert_eq!(old.export(), new.export());
    }
    assert!(forced_count > 0);
    let (a, b) = (old.finish().unwrap(), new.finish().unwrap());
    for key in [
        "rows",
        "records_sha256",
        "evidence_sha256",
        "result",
        "scoring",
    ] {
        assert_eq!(a[key], b[key], "{key}");
    }
    let total = b["rows"].as_u64().unwrap() as usize;
    let mut start = 0;
    while start < total {
        let (page, raw) = new.record_arrays("request", start).unwrap();
        let mut offset = 0;
        for (i, pair) in page["lengths"].as_array().unwrap().iter().enumerate() {
            let row = old.row(start + i).unwrap();
            for (k, key) in ["row_json", "evidence_json"].iter().enumerate() {
                let size = pair[k].as_u64().unwrap() as usize;
                assert_eq!(
                    &raw[offset..offset + size],
                    row[key].as_str().unwrap().as_bytes()
                );
                offset += size;
            }
        }
        assert_eq!(offset, raw.len());
        start = page["end"].as_u64().unwrap() as usize;
    }
    assert!(new.record_arrays("stale", 0).is_err());
    assert!(new.record_arrays("request", total).is_err());
    let mut start = 0;
    while start < total {
        let (meta, raw) = new.drain_training("request", start).unwrap();
        assert_eq!(
            (meta.clone(), raw),
            old.training_arrays("request", start).unwrap()
        );
        assert!(new.drain_training("request", start).is_err());
        start = meta["end"].as_u64().unwrap() as usize;
    }
    let mut start = 0;
    while start < total {
        let (meta, raw) = new.drain_records("request", start).unwrap();
        assert_eq!(
            (meta.clone(), raw),
            old.record_arrays("request", start).unwrap()
        );
        assert!(new.drain_records("request", start).is_err());
        assert!(new.row(start).is_err());
        start = meta["end"].as_u64().unwrap() as usize;
    }
    assert_eq!(new.finish().unwrap()["records_sha256"], b["records_sha256"]);
    assert_eq!(
        new.finish().unwrap()["evidence_sha256"],
        b["evidence_sha256"]
    );
}

#[test]
fn external_policy_uses_current_state_and_retains_features_without_actor_encoding() {
    let mut lane = Lane::new(initial(), "job".into(), "request".into(), vec![0; 64], 1000).unwrap();
    let decision = lane.decision().unwrap();
    let mut value = compact(&decision, 0);
    value["behavior"]["kind"] = json!("external");
    value["context"]["model_id"] = json!("ordinary-external");
    value["evidence"]["model_id"] = Value::Null;
    value["evidence"]["raw_probabilities"] = Value::Null;
    let mut bad = value.clone();
    bad["evidence"]["policy_id"] = json!("foreign-opponent");
    assert!(lane.append_compact("request", &bad).is_err());
    assert_eq!(lane.decision().unwrap(), decision);
    lane.append_compact("request", &value).unwrap();
    assert!(lane.append_compact("request", &value).is_err());
    assert_eq!(lane.export().context.revision, 1);
}
