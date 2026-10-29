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
