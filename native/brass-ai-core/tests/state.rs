use brass_ai_core::state::{dto::StateDto, NativeState};
use proptest::prelude::*;

fn initial() -> StateDto {
    serde_json::from_str(include_str!("data/initial-state.json")).unwrap()
}

#[test]
fn initial_roundtrip_and_private_clone() {
    let dto = initial();
    let native = NativeState::import(&dto).unwrap();
    assert_eq!(native.export(), dto);
    assert!(native.clone_isolated());
}
#[test]
fn snapshot_survives_source_drop_and_restores_indexes() {
    let dto = initial();
    let state = NativeState::import(&dto).unwrap();
    let saved = state.snapshot();
    let indexes = state.indexes();
    drop(state);
    let mut modified = dto.clone();
    modified.players[0].cash += 7;
    let mut state = NativeState::import(&modified).unwrap();
    assert_ne!(state.export(), dto);
    state.restore(&saved);
    assert_eq!(state.export(), dto);
    assert_eq!(state.indexes(), indexes);
}
#[test]
fn three_players_and_unknown_nested_fields_are_rejected() {
    let mut value = serde_json::to_value(initial()).unwrap();
    value["players"].as_array_mut().unwrap().pop();
    assert!(serde_json::from_value::<StateDto>(value).is_err());
    let mut value = serde_json::to_value(initial()).unwrap();
    value["pending"]["secret"] = true.into();
    assert!(serde_json::from_value::<StateDto>(value).is_err());
}
proptest! {
    #[test]
    fn import_export_preserves_signed_vp_and_wide_cash(cash in 0i32..i32::MAX, vp in -1000i32..10000) {
        let mut dto=initial();dto.players[0].cash=cash;dto.players[2].vp=vp;
        let state=NativeState::import(&dto).unwrap();prop_assert_eq!(state.export(),dto);prop_assert!(state.clone_isolated());
    }
    #[test]
    fn deleting_any_physical_tile_is_rejected(seat in 0usize..4, kind in 0usize..6, item in 0usize..11) {
        let mut dto=initial();let tiles=&mut dto.players[seat].mat[kind].tiles;let index=item%tiles.len();tiles.remove(index);
        prop_assert!(NativeState::import(&dto).is_err());
    }
}
