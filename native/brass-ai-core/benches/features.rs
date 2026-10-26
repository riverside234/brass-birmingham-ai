use brass_ai_core::{
    features::{FeatureContext, FeatureEncoder},
    state::{dto::StateDto, NativeState},
    types::Seat,
};
use criterion::{criterion_group, criterion_main, Criterion};
use std::{hint::black_box, path::PathBuf, time::Duration};
fn bench(c: &mut Criterion) {
    let Some(dir) = std::env::var_os("BRASS_FEATURE_BENCH_DIR") else {
        return;
    };
    let mut group = c.benchmark_group("features");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for name in ["initial", "complex", "dense"] {
        let dto: StateDto = serde_json::from_slice(
            &std::fs::read(PathBuf::from(&dir).join(format!("{name}.json"))).unwrap(),
        )
        .unwrap();
        let state = NativeState::import(&dto).unwrap();
        let observer = Seat::try_from(dto.context.actor as u8).unwrap();
        let view = state.observation(observer).unwrap();
        let choices = state.legal_choices().unwrap();
        let ctx = FeatureContext {
            model_id: "timing-profile".into(),
            reward_id: "g890".into(),
            profile_id: "neutral".into(),
            style: [0.; 4],
            intent: [0.; 8],
            remaining: [0.075; 4],
            max_cost: 0.075,
            stable_descriptors: None,
        };
        let mut encoder = FeatureEncoder::default();
        group.bench_function(format!("{name}/warm"), |b| {
            b.iter(|| {
                black_box(
                    encoder
                        .encode(&view, &choices, &ctx, &mut || Ok(()))
                        .unwrap(),
                );
            })
        });
        group.bench_function(format!("{name}/cold"), |b| {
            b.iter(|| {
                black_box(
                    FeatureEncoder::default()
                        .encode(&view, &choices, &ctx, &mut || Ok(()))
                        .unwrap(),
                );
            })
        });
        group.bench_function(format!("{name}/observation-choices-encode"), |b| {
            b.iter(|| {
                let v = state.observation(observer).unwrap();
                let cs = state.legal_choices().unwrap();
                black_box(encoder.encode(&v, &cs, &ctx, &mut || Ok(())).unwrap());
            })
        });
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
