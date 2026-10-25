use brass_ai_core::diagnostic::DiagnosticGame;
use criterion::{criterion_group, criterion_main, Criterion};
use std::{hint::black_box, time::Duration};

fn scaffold(c: &mut Criterion) {
    // Initial setup only; mid-game restoration and search are later workloads.
    let mut game = DiagnosticGame::new(4, 890).expect("four-player diagnostic");
    c.bench_function("upstream_initial_state", |b| {
        b.iter(|| black_box(DiagnosticGame::new(4, black_box(890)).expect("diagnostic")))
    });
    c.bench_function("upstream_initial_state_deep_clone", |b| {
        b.iter(|| black_box(game.clone()))
    });
    c.bench_function("upstream_initial_action_types", |b| {
        b.iter(|| black_box(game.root_action_count()))
    });
    let context = game.snapshot().context;
    c.bench_function("typed_context_copy", |b| {
        b.iter(|| black_box(black_box(context)))
    });
}
criterion_group! { name = benches; config = Criterion::default().sample_size(20)
.warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(1))
.without_plots(); targets = scaffold }
criterion_main!(benches);
