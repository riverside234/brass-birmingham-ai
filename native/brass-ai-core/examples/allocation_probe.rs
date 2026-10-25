use brass_ai_core::{diagnostic::DiagnosticGame, protocol::BuildIdentity};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{alloc::System, hint::black_box};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Warm lazy immutable board definitions before measuring owned state copies.
    let game = DiagnosticGame::new(4, 890)?;
    drop(game.clone());
    let region = Region::new(GLOBAL);
    let operations = 100;
    for _ in 0..operations {
        drop(black_box(game.clone()));
    }
    let stats = region.change();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "format": "brass-native-allocation-probe/0.1.0", "build": BuildIdentity::current(),
            "workload": "upstream initial state deep clone and drop, warm definitions",
            "clone_operations": operations, "context_copy_operations": 0,
            "shallow_state_bytes": DiagnosticGame::shallow_size_bytes(),
            "deep_copied_bytes": null,
            "note": "Allocator traffic is not the number of bytes copied. Timing is measured separately without this allocator.",
            "allocations": stats.allocations, "deallocations": stats.deallocations,
            "reallocations": stats.reallocations, "bytes_allocated": stats.bytes_allocated,
            "bytes_deallocated": stats.bytes_deallocated, "bytes_reallocated": stats.bytes_reallocated
        }))?
    );
    Ok(())
}
