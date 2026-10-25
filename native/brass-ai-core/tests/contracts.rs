use brass_ai_core::{
    diagnostic::DiagnosticGame,
    framing::{read_frame, write_frame, MAX_FRAME_BYTES},
    protocol::Request,
    types::{ChoiceId, Seat},
};
use proptest::prelude::*;
use std::io::{Cursor, ErrorKind};

proptest! {
    #[test]
    fn frames_round_trip(payload in prop::collection::vec(any::<u8>(), 1..8192)) {
        let mut wire = Vec::new();
        write_frame(&mut wire, &payload).unwrap();
        let mut reader = Cursor::new(&wire);
        prop_assert_eq!(read_frame(&mut reader).unwrap(), Some(payload));
        prop_assert!(read_frame(&mut reader).unwrap().is_none());
        wire.pop();
        prop_assert_eq!(read_frame(&mut Cursor::new(wire)).unwrap_err().kind(), ErrorKind::UnexpectedEof);
    }
    #[test]
    fn seat_admission(value in any::<u8>()) {
        prop_assert_eq!(Seat::try_from(value).is_ok(), value < 4);
        prop_assert_eq!(serde_json::from_str::<Seat>(&value.to_string()).is_ok(), value < 4);
    }
    #[test]
    fn choices_round_trip(value in "[a-zA-Z0-9_:/.-]{1,128}") {
        let choice = ChoiceId::try_from(value.clone()).unwrap();
        let restored: ChoiceId = serde_json::from_str(&serde_json::to_string(&choice).unwrap()).unwrap();
        prop_assert_eq!(String::from(restored), value);
    }
    #[test]
    fn native_states_are_deterministic_and_clones_are_detached(seed in any::<u64>()) {
        let game = DiagnosticGame::new(4, seed).unwrap();
        prop_assert_eq!(game.snapshot(), DiagnosticGame::new(4, seed).unwrap().snapshot());
        prop_assert!(game.clone_isolated());
    }
}

#[test]
fn framing_rejects_bad_lengths_and_partial_headers() {
    for length in [0, MAX_FRAME_BYTES as u32 + 1, u32::MAX] {
        assert_eq!(
            read_frame(&mut Cursor::new(length.to_le_bytes()))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
    }
    for length in 1..4 {
        assert_eq!(
            read_frame(&mut Cursor::new(&[1, 0, 0][..length]))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
    }
    assert!(write_frame(&mut Vec::new(), &[]).is_err());
    assert!(write_frame(&mut Vec::new(), &vec![0; MAX_FRAME_BYTES + 1]).is_err());
}

#[test]
fn external_types_and_messages_reject_unsupported_fields() {
    for value in ["", "a\nb", "\0", &"x".repeat(4097)] {
        assert!(ChoiceId::try_from(value.to_owned()).is_err());
    }
    for request in [
        r#"{"protocol":"brass-native-control/0.1.0","request_id":"a","request":{"op":"shutdown"},"hidden":1}"#,
        r#"{"protocol":"brass-native-control/0.1.0","request_id":"a","request":{"op":"diagnostic_snapshot","session":0,"hidden":1}}"#,
        r#"{"protocol":"brass-native-control/0.1.0","request_id":"a","request":{"op":"diagnostic_snapshot","session":-1}}"#,
    ] {
        assert!(serde_json::from_str::<Request>(request).is_err());
    }
    let wrong: Request = serde_json::from_str(
        r#"{"protocol":"future/1","request_id":"a","request":{"op":"shutdown"}}"#,
    )
    .unwrap();
    assert_eq!(wrong.validate().unwrap_err().code(), "unsupported_contract");
    let empty: Request = serde_json::from_str(
        r#"{"protocol":"brass-native-control/0.1.0","request_id":"","request":{"op":"shutdown"}}"#,
    )
    .unwrap();
    assert_eq!(empty.validate().unwrap_err().code(), "invalid_request");
    assert!(DiagnosticGame::new(3, 1).is_err());
}
