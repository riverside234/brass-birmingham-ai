use brass_ai_core::state::{
    dto::StateDto,
    initial::InitialSetup,
    simulation::{SimulationCommand as C, Simulator, MAX_STATES},
    NativeState,
};

fn fixture() -> StateDto {
    serde_json::from_str(include_str!("data/initial-state.json")).unwrap()
}
#[test]
fn explicit_setup_reproduces_initial_state_and_chance_prefix_clones() {
    let dto = fixture();
    let mut deck: Vec<_> = dto.cards.hands.iter().flatten().cloned().collect();
    deck.extend(dto.cards.discards.iter().flatten().cloned());
    deck.extend(dto.cards.deck.clone());
    assert_eq!(
        NativeState::initial(InitialSetup {
            order: dto.order,
            merchants: dto
                .merchants
                .iter()
                .map(|m| m.types.clone().unwrap())
                .collect(),
            deck
        })
        .unwrap()
        .export(),
        dto
    );
    let mut sim = Simulator::default();
    let first = sim.execute(C::New, 0).unwrap().handle.unwrap();
    let second = sim
        .execute(C::Clone { handle: first }, 0)
        .unwrap()
        .handle
        .unwrap();
    sim.execute(
        C::Step {
            handle: first,
            revision: None,
            semantic: "chance:order:2".into(),
        },
        0,
    )
    .unwrap();
    assert_eq!(
        sim.execute(C::Snapshot { handle: second }, 0)
            .unwrap()
            .snapshot
            .unwrap()
            .setup_outcomes
            .len(),
        4
    );
    let saved = sim
        .execute(C::Export { handle: first }, 0)
        .unwrap()
        .checkpoint
        .unwrap();
    let wire = serde_json::to_vec(&saved).unwrap();
    let replay = sim
        .execute(
            C::Deserialize {
                checkpoint: Box::new(serde_json::from_slice(&wire).unwrap()),
            },
            0,
        )
        .unwrap();
    assert_eq!(replay.snapshot.unwrap().setup_outcomes.len(), 3);
    sim.execute(
        C::Restore {
            handle: first,
            source: second,
        },
        0,
    )
    .unwrap();
    assert_eq!(
        sim.execute(C::Snapshot { handle: first }, 0)
            .unwrap()
            .snapshot
            .unwrap()
            .setup_outcomes
            .len(),
        4
    );
}

#[test]
fn bounded_handles_fail_atomically_and_never_alias_retired_objects() {
    let mut sim = Simulator::default();
    let first = sim
        .execute(
            C::Import {
                state: Box::new(fixture()),
            },
            1,
        )
        .unwrap()
        .handle
        .unwrap();
    for _ in 1..MAX_STATES {
        sim.execute(C::Clone { handle: first }, 1).unwrap();
    }
    assert!(sim.execute(C::Clone { handle: first }, 1).is_err());
    let before = serde_json::to_value(
        sim.execute(C::Export { handle: first }, 1)
            .unwrap()
            .checkpoint,
    )
    .unwrap();
    assert!(sim
        .execute(
            C::Step {
                handle: first,
                revision: Some(999),
                semantic: "invalid".into()
            },
            1
        )
        .is_err());
    assert_eq!(
        before,
        serde_json::to_value(
            sim.execute(C::Export { handle: first }, 1)
                .unwrap()
                .checkpoint
        )
        .unwrap()
    );
    sim.execute(C::Release { handle: first }, 1).unwrap();
    assert!(sim.execute(C::Snapshot { handle: first }, 1).is_err());
    let next = sim.execute(C::New, 1).unwrap().handle.unwrap();
    assert!(next > first);
    assert!(sim.execute(C::Snapshot { handle: next - 1 }, 1).is_err());
}
