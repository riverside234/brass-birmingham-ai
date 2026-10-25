use brass_ai_core::{
    protocol::BuildIdentity,
    state::{dto::StateDto, NativeState},
};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{alloc::System, hint::black_box};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("state fixture path required")?;
    let dto: StateDto = serde_json::from_slice(&std::fs::read(path)?)?;
    let state = NativeState::import(&dto)?;
    drop(state.clone());
    let operations = 100;
    let region = Region::new(GLOBAL);
    for _ in 0..operations {
        drop(black_box(state.clone()));
    }
    let stats = region.change();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "format":"brass-native-state-allocation/1.0.0","build":BuildIdentity::current(),
            "operations":operations,"workload":"semantic state clone and drop, warm catalog",
            "phase":dto.context.phase,"history_entries":dto.history.len(),
            "allocations":stats.allocations,"deallocations":stats.deallocations,"reallocations":stats.reallocations,
            "bytes_allocated":stats.bytes_allocated,"bytes_deallocated":stats.bytes_deallocated,
            "shallow_state_bytes":std::mem::size_of::<NativeState>(),"deep_copied_bytes":null,
            "note":"Allocator traffic is not a count of copied bytes. Criterion uses the normal allocator."
        }))?
    );
    Ok(())
}
