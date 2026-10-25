use brass_ai_core::{
    protocol::BuildIdentity,
    state::{
        belief::{PreparedRoot, SamplingDraws},
        dto::StateDto,
        simulation::{SimulationCommand as C, Simulator, World},
        NativeState,
    },
    types::Seat,
};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{alloc::System, hint::black_box};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn measure(mut f: impl FnMut()) -> serde_json::Value {
    f();
    let region = Region::new(GLOBAL);
    for _ in 0..100 {
        f();
    }
    let s = region.change();
    serde_json::json!({"operations":100,"allocations":s.allocations,"reallocations":s.reallocations,
        "deallocations":s.deallocations,"bytes_allocated":s.bytes_allocated,"bytes_deallocated":s.bytes_deallocated})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dto: StateDto = serde_json::from_slice(&std::fs::read(
        std::env::args_os().nth(1).ok_or("fixture")?,
    )?)?;
    let state = NativeState::import(&dto)?;
    let view = state.observation(Seat::try_from(dto.context.actor as u8)?)?;
    let prepared = PreparedRoot::new(&view, vec![])?;
    let random = SamplingDraws::Recorded {
        indices: vec![0; prepared.draw_count()],
    };
    let choice = state.legal_choices()?[0].id.clone();
    let world = World::playing(state.clone());
    let mut data = std::collections::BTreeMap::new();
    data.insert(
        "legal",
        measure(|| {
            black_box(state.legal_choices().unwrap());
        }),
    );
    data.insert(
        "prepare",
        measure(|| {
            black_box(PreparedRoot::new(&view, vec![]).unwrap());
        }),
    );
    data.insert(
        "sample",
        measure(|| {
            black_box(prepared.sample(random.clone()).unwrap());
        }),
    );
    data.insert(
        "apply_rules",
        measure(|| {
            black_box(state.after_choice(dto.context.revision, &choice).unwrap());
        }),
    );
    data.insert(
        "apply_with_recall",
        measure(|| {
            black_box(world.after(Some(dto.context.revision), &choice).unwrap());
        }),
    );
    data.insert(
        "owned_clone",
        measure(|| {
            black_box(state.clone());
        }),
    );
    data.insert(
        "diagnostic_snapshot",
        measure(|| {
            black_box(world.snapshot().unwrap());
        }),
    );
    let mut sim = Simulator::default();
    let handle = sim
        .execute(
            C::Import {
                state: Box::new(dto.clone()),
            },
            0,
        )?
        .handle
        .unwrap();
    data.insert(
        "handle_clone_restore_release",
        measure(|| {
            let clone = sim.execute(C::Clone { handle }, 0).unwrap().handle.unwrap();
            sim.execute(
                C::Restore {
                    handle: clone,
                    source: handle,
                },
                0,
            )
            .unwrap();
            sim.execute(C::Release { handle: clone }, 0).unwrap();
        }),
    );
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"build":BuildIdentity::current(),"measurements":data,
        "phase":dto.context.phase,"history_entries":dto.history.len(),"shallow_state_bytes":std::mem::size_of::<NativeState>(),
        "deep_copied_bytes":null,"note":"Allocator traffic is not bytes copied; timings use the normal allocator."})
        )?
    );
    Ok(())
}
