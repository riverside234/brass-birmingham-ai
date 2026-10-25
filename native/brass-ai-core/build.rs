use sha2::{Digest, Sha256};
use std::{env, fs, path::Path, process::Command};
use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = env::var("CARGO_MANIFEST_DIR")?;
    let workspace = Path::new(&manifest).parent().ok_or("missing workspace")?;
    println!("cargo:rerun-if-changed={}", workspace.display());
    let mut files = Vec::new();
    for entry in WalkDir::new(workspace).into_iter().filter_entry(|e| {
        !matches!(
            e.file_name().to_str(),
            Some("target" | ".git" | "__pycache__")
        )
    }) {
        let entry = entry?;
        if entry.file_type().is_file() {
            let path = entry.into_path();
            if matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("rs" | "toml" | "lock" | "json")
            ) {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path
            .strip_prefix(workspace)?
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = fs::read(&path)?;
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    println!("cargo:rustc-env=BRASS_SOURCE_SHA256={:x}", hash.finalize());
    println!(
        "cargo:rustc-env=BRASS_LOCK_SHA256={:x}",
        Sha256::digest(fs::read(workspace.join("Cargo.lock"))?)
    );
    for name in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        println!(
            "cargo:rustc-env=BRASS_{name}={}",
            env::var(name)?.replace('\u{1f}', " ")
        );
    }
    let compiler = Command::new(env::var("RUSTC")?).arg("--version").output()?;
    if !compiler.status.success() {
        return Err("rustc version probe failed".into());
    }
    println!(
        "cargo:rustc-env=BRASS_RUSTC={}",
        String::from_utf8(compiler.stdout)?.trim()
    );
    println!("cargo:rerun-if-env-changed=RUSTC_LINKER");
    println!(
        "cargo:rustc-env=BRASS_LINKER={}",
        env::var("RUSTC_LINKER").unwrap_or_else(|_| "default".into())
    );
    Ok(())
}
