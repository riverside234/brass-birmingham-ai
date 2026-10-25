use brass_ai_core::{
    state::{
        belief::{PreparedRoot, SamplingDraws},
        dto::StateDto,
        simulation::{SimulationCommand as C, Simulator, World},
        NativeState,
    },
    types::Seat,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{hint::black_box, path::PathBuf, time::Duration};

fn simulation(c: &mut Criterion) {
    let directory =
        PathBuf::from(std::env::var_os("BRASS_STATE_BENCH_DIR").expect("fixture directory"));
    let mut group = c.benchmark_group("simulation");
    for name in ["initial", "complex"] {
        let dto: StateDto =
            serde_json::from_slice(&std::fs::read(directory.join(format!("{name}.json"))).unwrap())
                .unwrap();
        let state = NativeState::import(&dto).unwrap();
        let view = state
            .observation(Seat::try_from(dto.context.actor as u8).unwrap())
            .unwrap();
        let prepared = PreparedRoot::new(&view, vec![]).unwrap();
        let random = SamplingDraws::Recorded {
            indices: vec![0; prepared.draw_count()],
        };
        let world = World::playing(state.clone());
        let choice = state.legal_choices().unwrap()[0].id.clone();
        group.bench_function(BenchmarkId::new("legal", name), |b| {
            b.iter(|| black_box(state.legal_choices().unwrap()))
        });
        group.bench_function(BenchmarkId::new("prepare", name), |b| {
            b.iter(|| black_box(PreparedRoot::new(&view, vec![]).unwrap()))
        });
        group.bench_function(BenchmarkId::new("sample", name), |b| {
            b.iter(|| black_box(prepared.sample(random.clone()).unwrap()))
        });
        group.bench_function(BenchmarkId::new("apply_rules", name), |b| {
            b.iter(|| black_box(state.after_choice(dto.context.revision, &choice).unwrap()))
        });
        group.bench_function(BenchmarkId::new("apply_with_recall", name), |b| {
            b.iter(|| black_box(world.after(Some(dto.context.revision), &choice).unwrap()))
        });
        group.bench_function(BenchmarkId::new("owned_clone", name), |b| {
            b.iter(|| black_box(state.clone()))
        });
        group.bench_function(BenchmarkId::new("diagnostic_snapshot", name), |b| {
            b.iter(|| black_box(world.snapshot().unwrap()))
        });
        let mut sim = Simulator::default();
        let handle = sim
            .execute(
                C::Import {
                    state: Box::new(dto),
                },
                0,
            )
            .unwrap()
            .handle
            .unwrap();
        group.bench_function(
            BenchmarkId::new("handle_clone_restore_release", name),
            |b| {
                b.iter(|| {
                    let clone = sim.execute(C::Clone { handle }, 0).unwrap().handle.unwrap();
                    black_box(
                        sim.execute(
                            C::Restore {
                                handle: clone,
                                source: handle,
                            },
                            0,
                        )
                        .unwrap(),
                    );
                    black_box(sim.execute(C::Release { handle: clone }, 0).unwrap());
                })
            },
        );
    }
    group.finish();
}
criterion_group! {name=benches;config=Criterion::default().sample_size(20).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(1)).without_plots();targets=simulation}
criterion_main!(benches);
