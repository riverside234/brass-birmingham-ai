use brass_ai_core::state::{dto::StateDto, NativeState};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{hint::black_box, path::PathBuf, time::Duration};

fn state(c: &mut Criterion) {
    let directory =
        PathBuf::from(std::env::var_os("BRASS_STATE_BENCH_DIR").expect("BRASS_STATE_BENCH_DIR"));
    let mut group = c.benchmark_group("semantic_state");
    for name in ["initial", "complex"] {
        let dto: StateDto = serde_json::from_slice(
            &std::fs::read(directory.join(format!("{name}.json"))).expect("benchmark fixture"),
        )
        .expect("state DTO");
        let state = NativeState::import(&dto).expect("valid fixture");
        let snapshot = state.snapshot();
        let mut destination = state.clone();
        group.bench_with_input(BenchmarkId::new("import", name), &dto, |b, d| {
            b.iter(|| black_box(NativeState::import(black_box(d)).unwrap()))
        });
        group.bench_function(BenchmarkId::new("export", name), |b| {
            b.iter(|| black_box(state.export()))
        });
        group.bench_function(BenchmarkId::new("clone", name), |b| {
            b.iter(|| black_box(state.clone()))
        });
        group.bench_function(BenchmarkId::new("restore", name), |b| {
            b.iter(|| {
                destination.restore(black_box(&snapshot));
                black_box(&destination);
            })
        });
    }
    group.finish();
}
criterion_group! {name=benches;config=Criterion::default().sample_size(20).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(1)).without_plots();targets=state}
criterion_main!(benches);
