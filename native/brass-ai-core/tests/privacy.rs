use brass_ai_core::{
    state::{
        belief::{PreparedRoot, SamplingDraws},
        dto::StateDto,
        observation::{ObservationMemory, RecallEntry},
        NativeState,
    },
    types::Seat,
};

fn initial() -> NativeState {
    let dto: StateDto = serde_json::from_str(include_str!("data/initial-state.json")).unwrap();
    NativeState::import(&dto).unwrap()
}
fn random() -> SamplingDraws {
    SamplingDraws::Chacha12 {
        seed: 890,
        stream: 0,
    }
}

#[test]
fn detached_worlds_preserve_each_root_and_sample_other_observations() {
    let state = initial();
    for s in 0..4u8 {
        let seat = Seat::try_from(s).unwrap();
        let view = state.observation(seat).unwrap();
        let recall = vec![RecallEntry {
            revision: 0,
            hand: view.own_hand.clone(),
        }];
        let root = PreparedRoot::new(&view, recall.clone()).unwrap();
        let (sample, memory, draws) = root.sample(random()).unwrap();
        assert_eq!(sample.observation(seat).unwrap(), view);
        assert_eq!(memory.information(seat).recall, recall);
        assert_eq!(
            sample.export(),
            NativeState::import(&sample.export()).unwrap().export()
        );
        assert_eq!(
            root.sample(SamplingDraws::Recorded { indices: draws })
                .unwrap()
                .0
                .export(),
            sample.export()
        );
        for other in 0..4u8 {
            let other = Seat::try_from(other).unwrap();
            assert_eq!(
                memory.information(other).observation.view.unwrap(),
                sample.observation(other).unwrap()
            );
        }
        let second = root
            .sample(SamplingDraws::Chacha12 {
                seed: 890,
                stream: 1,
            })
            .unwrap()
            .0;
        assert_ne!(second.export().cards, sample.export().cards);
        assert_eq!(second.observation(seat).unwrap(), view);
    }
}

#[test]
fn recall_changes_identity_but_never_sampling_conditioning() {
    let state = initial();
    let seat = Seat::Zero;
    let view = state.observation(seat).unwrap();
    let recall = vec![RecallEntry {
        revision: 0,
        hand: view.own_hand.clone(),
    }];
    let a = PreparedRoot::new(&view, vec![])
        .unwrap()
        .sample(random())
        .unwrap();
    let b = PreparedRoot::new(&view, recall)
        .unwrap()
        .sample(random())
        .unwrap();
    assert_eq!(a.0.export(), b.0.export());
    assert_eq!(a.2, b.2);
    assert_ne!(a.1.information(seat), b.1.information(seat));
    assert_ne!(
        a.1.information(seat).reference_digest(),
        b.1.information(seat).reference_digest()
    );
}

#[test]
fn hidden_live_counterfactuals_cannot_change_recorded_worlds() {
    let state = initial();
    let seat = Seat::Zero;
    let mut dto = state.export();
    let card = dto.cards.hands[1][0].clone();
    dto.cards.hands[1][0] = dto.cards.deck[0].clone();
    dto.cards.deck[0] = card;
    dto.cards.deck.reverse();
    dto.cards.hands[0].reverse();
    let changed = NativeState::import(&dto).unwrap();
    let a = state.observation(seat).unwrap();
    let b = changed.observation(seat).unwrap();
    assert_eq!(a, b);
    let first = PreparedRoot::new(&a, vec![])
        .unwrap()
        .sample(random())
        .unwrap();
    let other = PreparedRoot::new(&b, vec![])
        .unwrap()
        .sample(SamplingDraws::Recorded {
            indices: first.2.clone(),
        })
        .unwrap();
    assert_eq!(first.0.export(), other.0.export());
    assert_eq!(first.1.information(seat), other.1.information(seat));
}

#[test]
fn both_eras_hide_partial_chance_and_keep_only_changed_hands_in_recall() {
    let mut state = initial();
    let mut memory = ObservationMemory::new(&state);
    let mut partial = 0;
    let mut rail = false;
    for _ in 0..650 {
        let before = memory.clone();
        let dto = state.export();
        if dto.context.terminal {
            assert!(rail && partial > 60);
            return;
        }
        if let Some(chance) = state.chance_outcomes() {
            state
                .resolve_chance(dto.context.revision, &chance.outcomes[0].card)
                .unwrap();
        } else {
            let choices = state.legal_choices().unwrap();
            let choice = choices
                .iter()
                .find(|c| c.kind == brass_ai_core::state::dto::Kind::Pass)
                .unwrap_or(&choices[0]);
            state
                .apply_choice(dto.context.revision, &choice.id)
                .unwrap();
        }
        memory.refresh(&state);
        if state.chance_outcomes().is_some() {
            partial += 1;
            assert!(state.observation(Seat::Zero).is_err());
            for s in 0..4u8 {
                let s = Seat::try_from(s).unwrap();
                let a = before.information(s);
                let b = memory.information(s);
                assert_eq!(a.observation.view, b.observation.view);
                assert_eq!(a.recall, b.recall);
            }
        } else {
            rail |= state.export().context.era == brass_ai_core::types::Era::RailEra;
            for s in 0..4u8 {
                let s = Seat::try_from(s).unwrap();
                let info = memory.information(s);
                assert_eq!(info.observation.view, Some(state.observation(s).unwrap()));
                assert!(info
                    .recall
                    .windows(2)
                    .all(|w| w[0].hand != w[1].hand && w[0].revision < w[1].revision));
            }
        }
    }
    panic!("full game did not finish");
}

#[test]
fn malformed_sampling_and_future_recall_fail_without_consuming_prepared_root() {
    let view = initial().observation(Seat::Zero).unwrap();
    let root = PreparedRoot::new(&view, vec![]).unwrap();
    let original = root.sample(random()).unwrap().0.export();
    for indices in [
        vec![],
        vec![999; root.draw_count()],
        vec![0; root.draw_count() + 1],
    ] {
        assert!(root.sample(SamplingDraws::Recorded { indices }).is_err());
        assert_eq!(original, root.sample(random()).unwrap().0.export());
    }
    assert!(PreparedRoot::new(
        &view,
        vec![RecallEntry {
            revision: 1,
            hand: view.own_hand.clone()
        }]
    )
    .is_err());
    let mut invalid = view.clone();
    invalid.players[1].known_wild.push("location:Belper".into());
    assert!(PreparedRoot::new(&invalid, vec![]).is_err());
}

#[test]
fn empty_merchant_field_absence_survives_sampling_roundtrip_and_saves() {
    let mut dto = initial().export();
    let empty: Vec<_> = dto
        .merchants
        .iter()
        .enumerate()
        .filter(|(_, m)| m.types.as_ref().is_some_and(Vec::is_empty))
        .map(|(i, _)| i)
        .collect();
    assert!(!empty.is_empty());
    for i in &empty {
        dto.merchants[*i].types = None;
    }
    let state = NativeState::import(&dto).unwrap();
    assert_eq!(state.export(), dto);
    let view = state.observation(Seat::Zero).unwrap();
    for i in &empty {
        assert!(view.merchants[*i].types.is_none());
    }
    let (mut sampled, _, _) = PreparedRoot::new(&view, vec![])
        .unwrap()
        .sample(random())
        .unwrap();
    let saved = sampled.snapshot();
    let before = sampled.export();
    let choice = sampled.legal_choices().unwrap()[0].id.clone();
    sampled
        .apply_choice(before.context.revision, &choice)
        .unwrap();
    sampled.restore(&saved);
    assert_eq!(sampled.export(), before);
    assert_eq!(sampled.observation(Seat::Zero).unwrap(), view);
    let mut invalid = serde_json::to_value(view).unwrap();
    invalid["merchants"][empty[0]]["types"] = serde_json::Value::Null;
    assert!(
        serde_json::from_value::<brass_ai_core::state::observation::Observation>(invalid).is_err()
    );
}
