//! Optional diagnostic spans. Never compare process clock origins: correlate
//! request IDs and subtract durations contained within the caller's round trip.
use std::{
    fs::File,
    io::Write,
    sync::{Mutex, OnceLock},
    time::Instant,
};
static TRACE: OnceLock<Mutex<File>> = OnceLock::new();
static ORIGIN: OnceLock<Instant> = OnceLock::new();
pub fn enable() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        std::env::var_os("BRASS_L14_NATIVE_TRACE").ok_or("missing diagnostic output path")?;
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    TRACE
        .set(Mutex::new(file))
        .map_err(|_| "diagnostic already enabled")?;
    Ok(())
}
pub fn now() -> u64 {
    if TRACE.get().is_none() {
        return 0;
    }
    ORIGIN
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
pub fn event(id: u64, stage: &str, at: u64, bytes: usize) {
    if let Some(file) = TRACE.get() {
        if let Ok(mut file) = file.lock() {
            let _ = writeln!(
                file,
                "{}",
                serde_json::json!({"process":std::process::id(),"request":id,"stage":stage,"at_ns":at,"bytes":bytes})
            );
        }
    }
}
