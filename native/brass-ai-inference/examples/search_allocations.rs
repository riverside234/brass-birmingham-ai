mod search_probe;
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
#[global_allocator]
static GLOBAL: &StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = search_probe::run(
        || Region::new(GLOBAL),
        |region| {
            let s = region.change();
            serde_json::json!({"allocations":s.allocations,"reallocations":s.reallocations,"bytes_allocated":s.bytes_allocated,"bytes_deallocated":s.bytes_deallocated,"note":"Rust allocator traffic only; excludes ORT C allocations. Timings from this instrumented executable are not performance evidence."})
        },
    )?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
