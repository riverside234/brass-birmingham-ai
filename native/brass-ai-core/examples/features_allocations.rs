use brass_ai_core::{
    features::{FeatureContext, FeatureEncoder},
    protocol::BuildIdentity,
    state::{dto::StateDto, NativeState},
    types::Seat,
};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{alloc::System, hint::black_box};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn measure(mut f: impl FnMut()) -> serde_json::Value {
    let region = Region::new(GLOBAL);
    for _ in 0..100 {
        f();
    }
    let s = region.change();
    serde_json::json!({"operations":100,"allocations":s.allocations,
        "reallocations":s.reallocations,"bytes_allocated":s.bytes_allocated,"bytes_deallocated":s.bytes_deallocated})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dto: StateDto = serde_json::from_slice(&std::fs::read(
        std::env::args_os().nth(1).ok_or("fixture")?,
    )?)?;
    let state = NativeState::import(&dto)?;
    let observer = Seat::try_from(dto.context.actor as u8)?;
    let view = state.observation(observer)?;
    let choices = state.legal_choices()?;
    let ctx = FeatureContext {
        model_id: "allocation-profile".into(),
        reward_id: "g890".into(),
        profile_id: "neutral".into(),
        style: [0.; 4],
        intent: [0.; 8],
        remaining: [0.075; 4],
        max_cost: 0.075,
        stable_descriptors: None,
    };
    let mut encoder = FeatureEncoder::default();
    encoder.encode(&view, &choices, &ctx, &mut || Ok(()))?;
    let stats = encoder.tensors().unwrap().stats.clone();
    let cold = measure(|| {
        black_box(
            FeatureEncoder::default()
                .encode(&view, &choices, &ctx, &mut || Ok(()))
                .unwrap(),
        );
    });
    let warm = measure(|| {
        black_box(
            encoder
                .encode(&view, &choices, &ctx, &mut || Ok(()))
                .unwrap(),
        );
    });
    assert_eq!(warm["allocations"], 0);
    assert_eq!(warm["reallocations"], 0);
    let observation_and_choices = measure(|| {
        black_box(state.observation(observer).unwrap());
        black_box(state.legal_choices().unwrap());
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"build":BuildIdentity::current(),
        "phase":dto.context.phase,"candidates":choices.len(),"effect_cache":stats,"cold":cold,"warm":warm,
        "observation_and_choices":observation_and_choices,"note":"Cold excludes one-time static layout setup; timing measured separately with normal allocator."}))?
    );
    Ok(())
}
