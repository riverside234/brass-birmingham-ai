fn main() {
    if let Err(error) = brass_ai_worker::local::run() {
        eprintln!("local backend: {error}");
        std::process::exit(1);
    }
}
