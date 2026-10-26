mod search_probe;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = search_probe::run(
        std::time::Instant::now,
        |start| serde_json::json!({"wall_ns":start.elapsed().as_nanos()}),
    )?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
