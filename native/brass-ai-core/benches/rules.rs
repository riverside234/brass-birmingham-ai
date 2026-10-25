use brass_ai_core::state::{dto::StateDto, NativeState};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::{hint::black_box, path::PathBuf, time::Duration};

fn rules(c: &mut Criterion) {
    let directory =
        PathBuf::from(std::env::var_os("BRASS_STATE_BENCH_DIR").expect("BRASS_STATE_BENCH_DIR"));
    let mut group = c.benchmark_group("staged_rules");
    for name in ["initial", "complex"] {
        let dto: StateDto =
            serde_json::from_slice(&std::fs::read(directory.join(format!("{name}.json"))).unwrap())
                .unwrap();
        let state = NativeState::import(&dto).unwrap();
        let choices = state.legal_choices().unwrap();
        let choice = choices.first().unwrap();
        group.bench_function(BenchmarkId::new("legal_choices", name), |b| {
            b.iter(|| black_box(state.legal_choices().unwrap()))
        });
        // Setup clone is outside timing. The transaction's own clone, full legal
        // validation, mutation, history append and index rebuild are timed.
        group.bench_function(BenchmarkId::new("validated_step", name), |b| {
            b.iter_batched(
                || state.clone(),
                |mut s| {
                    s.apply_choice(dto.context.revision, black_box(&choice.id))
                        .unwrap();
                    black_box(s);
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}
criterion_group! {name=benches;config=Criterion::default().sample_size(20).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(1)).without_plots();targets=rules}
criterion_main!(benches);
