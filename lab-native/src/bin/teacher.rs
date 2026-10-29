fn main() {
    if let Err(error) = brass_lab_collector::teacher::run() {
        eprintln!("teacher: {error}");
        std::process::exit(1);
    }
}
