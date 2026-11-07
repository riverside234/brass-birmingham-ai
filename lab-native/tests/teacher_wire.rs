#![cfg(feature = "teacher")]
use serde_json::{json, Value};

#[test]
fn python_echo_preserves_remaining_budget_bits() {
    // Actual B07 failure: Python echoes this Rust f64 verbatim; the default
    // serde_json parser previously rounded its decimal to the adjacent float.
    let original = 0.025_f64 - 0.01_f64;
    assert_eq!(original.to_string(), "0.015000000000000001");
    let reply: Value = serde_json::from_slice(
        br#"{"binding":{"context":{"remaining":[0.025,0.025,0.015000000000000001,0.025]}}}"#,
    )
    .unwrap();
    let expected = json!({"context":{"remaining":[0.025,0.025,original,0.025]}});
    assert_eq!(reply["binding"], expected);
    assert_eq!(
        reply["binding"]["context"]["remaining"][2]
            .as_f64()
            .unwrap()
            .to_bits(),
        original.to_bits()
    );
}

#[test]
fn all_budget_steps_and_finite_wire_values_round_trip_exactly() {
    let mut values = Vec::new();
    for step in 0..=5 {
        let remaining = 0.025_f64 - f64::from(step) * 0.005_f64;
        values.extend([remaining, f64::from_bits(remaining.to_bits() + 1)]);
    }
    // Deterministic wide exponent/mantissa coverage, including subnormals.
    let mut bits = 0x1234_5678_9abc_def0_u64;
    for _ in 0..4096 {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        let value = f64::from_bits(bits);
        if value.is_finite() {
            values.push(value);
        }
    }
    values.extend([0., -0., f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)]);
    for value in values {
        let encoded = serde_json::to_vec(&value).unwrap();
        let decoded: f64 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            value.to_bits(),
            decoded.to_bits(),
            "{}",
            String::from_utf8(encoded).unwrap()
        );
    }
}

#[test]
fn exact_binding_comparison_still_rejects_altered_identity_or_one_bit_budget() {
    let expected = json!({"job":{"model_id":"frozen","epoch":1}, "sequence":2492,
        "root_epoch":293,"revision":293,"seat":2,"legal_ids":["pass"],
        "context":{"remaining":[0.025,0.025,0.025_f64-0.01_f64,0.025]},
        "information":"frozen-information", "search_value":{"w":0.0}});
    let wire = serde_json::to_vec(&expected).unwrap();
    let decoded: Value = serde_json::from_slice(&wire).unwrap();
    assert_eq!(expected, decoded);
    for (pointer, wrong) in [
        ("/job/model_id", json!("other")),
        ("/job/epoch", json!(0)),
        ("/sequence", json!(2491)),
        ("/root_epoch", json!(292)),
        ("/revision", json!(292)),
        ("/seat", json!(1)),
        ("/legal_ids/0", json!("other")),
        ("/information", json!("other")),
        ("/search_value/w", json!(0.1)),
        ("/context/remaining/2", json!(0.015)),
    ] {
        let mut changed = decoded.clone();
        *changed.pointer_mut(pointer).unwrap() = wrong;
        assert_ne!(changed, expected, "{pointer}");
    }
}
