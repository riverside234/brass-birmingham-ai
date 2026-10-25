#!/usr/bin/env python3
"""Build and qualify L03 locally, without modifying the frozen L01/L02 oracle.

Uses a preinstalled, pinned project-local toolchain and Cargo cache. No network,
tool installation, Unity deployment or search/NN admission occurs in this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
TARGET = "x86_64-pc-windows-gnu"
TOOLCHAIN = "1.99.1"
LLVM = "llvm-mingw-20261020-msvcrt-x86_64"
LLVM_SHA = "52a3e3f584a74c3cde8ceb0c6960549683178e03b36818cb02b7ed109a68ff67"
FROZEN = ROOT / "artifacts/rust-backend/20261024-g890-l01-l02"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def receipt(path):
    path = Path(path)
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def native_fingerprint():
    value = hashlib.sha256()
    # Rust PathBuf ordering is case-sensitive; WindowsPath ordering is not.
    for path in sorted(NATIVE.rglob("*"), key=str):
        if not path.is_file() or path.suffix not in (".rs", ".toml", ".lock", ".json"):
            continue
        if any(part in ("target", ".git", "__pycache__") for part in path.relative_to(NATIVE).parts):
            continue
        name = path.relative_to(NATIVE).as_posix().encode("utf-8")
        data = path.read_bytes()
        value.update(struct.pack("<Q", len(name)) + name + struct.pack("<Q", len(data)) + data)
    return value.hexdigest()


def tool_environment():
    base = ROOT / "artifacts/native-toolchain"
    env = os.environ.copy()
    env["RUSTUP_HOME"] = str(base / "rustup")
    env["CARGO_HOME"] = str(base / "cargo")
    env["PATH"] = os.pathsep.join([str(base / LLVM / "bin"), str(base / "cargo/bin"), env.get("PATH", "")])
    env["CC_x86_64_pc_windows_gnu"] = str(base / LLVM / "bin/x86_64-w64-mingw32-clang.exe")
    env["AR_x86_64_pc_windows_gnu"] = str(base / LLVM / "bin/llvm-ar.exe")
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / path) for path in [
        "src", "artifacts/r2-pythondeps", "artifacts/r4-pythondeps", "artifacts/r0-pythonlibs", "artifacts/r0-validation-deps"])
    # Avoid inherited experimental flags changing the portable build silently.
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR",
                 "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS", "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER"]:
        env.pop(name, None)
    # Use Rust's matching GCC runtime/linker; LLVM supplies dlltool and the
    # benchmark-only C compiler, without substituting a different CRT at link.
    env["CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER"] = str(base / "rustup/toolchains" /
        (TOOLCHAIN + "-" + TARGET) / "lib/rustlib" / TARGET / "bin/self-contained/x86_64-w64-mingw32-gcc.exe")
    env["PROPTEST_RNG_SEED"] = "890"
    return env, base / "cargo/bin/cargo.exe"


def verify_frozen(*, reference_source_root=None):
    for entry in ["src", "artifacts/r2-pythondeps", "artifacts/r4-pythondeps", "artifacts/r0-pythonlibs", "artifacts/r0-validation-deps"]:
        sys.path.insert(0, str(ROOT / entry))
    baseline = read(FROZEN / "baseline.json")
    if reference_source_root is None:
        from brass_ai.rust_backend.baseline import verify_reference
        drift = verify_reference(baseline, ROOT)
    else:
        # Later milestones can execute an isolated, byte-identical reference
        # while unrelated training work continues in the shared checkout.
        # This never waives source verification: every executed reference file
        # must match both the original manifest and its immutable archive.
        reference_source_root = Path(reference_source_root).resolve()
        archived = baseline["reference_source"]["archive"]
        if sha(ROOT / archived["path"]) != archived["sha256"]:
            raise RuntimeError("Frozen source archive changed")
        drift = []
        with zipfile.ZipFile(ROOT / archived["path"]) as archive:
            for item in baseline["reference_source"]["files"]:
                path = (reference_source_root / item["path"]).resolve()
                if not path.is_relative_to(reference_source_root):
                    raise RuntimeError("Reference source escapes isolated root")
                if (sha(path) != item["sha256"] or
                    hashlib.sha256(archive.read(item["path"])).hexdigest() != item["sha256"]):
                    raise RuntimeError("Isolated reference source changed: " + item["path"])
                live = ROOT / item["path"]
                actual = sha(live) if live.is_file() else None
                if actual != item["sha256"]:
                    drift.append(dict(path=item["path"], frozen_sha256=item["sha256"], current_sha256=actual,
                        disposition="Isolated archived reference executed; concurrent checkout edits preserved"))
        for item in [dict(path=baseline["model"]["path"], sha256=baseline["model"]["sha256"]),
                     *baseline["model"]["bundle_files"]]:
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError("Test model/bundle changed: " + item["path"])
    checks = []
    for item in baseline["installed"]["receipt"]["files"]:
        path = ROOT / "BrassBirminghamUnity_Data/Managed" / item["name"]
        if sha(path) != item["sha256"].lower():
            raise RuntimeError("Installed fallback changed: " + str(path))
        checks.append(receipt(path))
    for item in baseline["installed"]["backup"] + [baseline["installed"][k] for k in ("stock_backup", "saves", "save_inventory")]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Frozen recovery changed: " + item["path"])
    saves = read(ROOT / baseline["installed"]["save_inventory"]["path"])
    for item in saves["files"]:
        if sha(Path(saves["root"]) / item["path"]) != item["sha256"]:
            raise RuntimeError("Original save changed: " + item["path"])
    vendor = NATIVE / "vendor/brass-upstream-rules"
    provenance = read(vendor / "UPSTREAM.json")
    if sha(FROZEN / "upstream-source.zip") != provenance["source_archive_sha256"]:
        raise RuntimeError("Upstream source archive changed")
    with zipfile.ZipFile(FROZEN / "upstream-source.zip") as archive:
        for item in provenance["source_files"]:
            path = vendor / item["path"]
            if (sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]
                or hashlib.sha256(archive.read(item["path"])).hexdigest() != item["sha256"]):
                raise RuntimeError("Imported source changed: " + str(path))
    for name, key in [("LICENSE", "license_sha256"), ("upstream-local.patch", "local_patch_sha256")]:
        if sha(vendor / name) != provenance[key]:
            raise RuntimeError("Upstream attribution changed: " + name)
    return dict(reference_file_count=len(baseline["reference_source"]["files"]),
        reference_source_digest=baseline["reference_source"]["digest"],
        baseline=receipt(FROZEN / "baseline.json"), upstream_files=len(provenance["source_files"]),
        checkpoint=baseline["model"]["sha256"], installed=checks, provenance_drift=drift,
        imported_source_byte_identical=True, recovery_hashes_valid=True)


def machine_type(path):
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise RuntimeError("Not a Windows executable")
    offset, = struct.unpack_from("<I", data, 0x3c)
    if data[offset:offset+4] != b"PE\0\0":
        raise RuntimeError("Not a PE executable")
    machine, = struct.unpack_from("<H", data, offset+4)
    if machine != 0x8664:
        raise RuntimeError("Worker is not native Windows x64")
    return "AMD64 (0x8664)"


class Qualification:
    def __init__(self, out):
        self.out = out
        self.env, self.cargo = tool_environment()
        self.commands = []

    def run(self, name, args, *, cwd=NATIVE, env=None, timeout=1200):
        log = self.out / "logs" / (name + ".txt")
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"{name}: started", flush=True)
        start = time.monotonic()
        with log.open("wb") as stream:
            completed = subprocess.run([str(x) for x in args], cwd=cwd, env=env or self.env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        entry = dict(name=name, args=[str(x) for x in args], cwd=str(cwd),
            exit_code=completed.returncode, seconds=time.monotonic()-start, log=receipt(log))
        self.commands.append(entry)
        write(self.out / "commands.json", self.commands)
        print(f"{name}: exit {completed.returncode} ({entry['seconds']:.1f}s)", flush=True)
        if completed.returncode:
            print(log.read_text(encoding="utf-8", errors="replace")[-10000:], flush=True)
            raise RuntimeError("Qualification failed: " + name)
        return log

    def cargo_run(self, name, *args):
        return self.run(name, [self.cargo, *args])

    def qualify(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "qualification.json").exists():
            raise RuntimeError("Use a new run directory; completed evidence is immutable")
        write(self.out / "progress.json", dict(status="running", milestone="L03"))
        write(self.out / "preservation-before.json", verify_frozen())
        base = ROOT / "artifacts/native-toolchain"
        if sha(base / (LLVM + ".zip")) != LLVM_SHA:
            raise RuntimeError("Toolchain archive does not match pinned LLVM-MinGW")
        self.run("rustc-version", [base / "cargo/bin/rustc.exe", "-Vv"])
        self.run("linker-version", [base / LLVM / "bin/x86_64-w64-mingw32-clang.exe", "--version"])
        self.run("rust-linker-version", [self.env["CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER"], "--version"])
        self.cargo_run("fmt", "fmt", "--all", "--", "--check")
        self.cargo_run("clippy", "clippy", "--workspace", "--all-targets", "--locked", "--offline", "--", "-D", "warnings")
        self.cargo_run("debug-tests", "test", "--workspace", "--locked", "--offline")
        # Preserve upstream modules and their tests without imposing our style on the vendor.
        self.cargo_run("upstream-tests", "test", "--manifest-path", "vendor/brass-upstream-rules/Cargo.toml", "--locked", "--offline")
        self.cargo_run("upstream-deadline-tests", "test", "--manifest-path", "vendor/brass-upstream-rules/Cargo.toml", "--features", "benchmark-deadlines", "--locked", "--offline")
        self.cargo_run("upstream-release-tests", "test", "--manifest-path", "vendor/brass-upstream-rules/Cargo.toml", "--release", "--locked", "--offline")
        self.cargo_run("release-tests", "test", "--workspace", "--release", "--locked", "--offline")
        builds = {}
        from brass_ai.native import NativeWorker
        for profile, flags, folder in [("debug", [], "debug"), ("release", ["--release"], "release"),
                                       ("profiling", ["--profile", "profiling"], "profiling")]:
            self.cargo_run(profile + "-build", "build", "-p", "brass-ai-worker", *flags, "--locked", "--offline")
            path = ROOT / "artifacts/rust-native-target" / TARGET / folder / "brass-ai-worker.exe"
            with NativeWorker(path) as worker:
                hello = worker.hello
                if hello["build"]["source_sha256"] != native_fingerprint():
                    raise RuntimeError("Worker source identity differs from current native files")
                if hello["build"]["cargo_lock_sha256"] != sha(NATIVE / "Cargo.lock"):
                    raise RuntimeError("Worker dependency lock differs")
                states = [worker.new_game(i, 890+i) for i in range(2)]
                if states[0]["owner"] == states[1]["owner"]:
                    raise RuntimeError("Sessions share an owner thread")
                if not all(worker.clone_check(i)["clone_isolated"] for i in range(2)):
                    raise RuntimeError("State clones are not isolated")
            builds[profile] = dict(binary=receipt(path), pe_machine=machine_type(path), handshake=hello)
            test_env = dict(self.env, BRASS_NATIVE_WORKER=str(path))
            self.run(profile + "-python-tests", [sys.executable, "-m", "unittest", "discover", "-s",
                str(ROOT / "tests/RustBackendNative"), "-v"], env=test_env, timeout=180)
        self.cargo_run("criterion", "bench", "-p", "brass-ai-core", "--bench", "scaffold", "--locked", "--offline")
        # Run the example separately so only its JSON goes into the measurement file.
        self.cargo_run("allocation-build", "build", "--release", "-p", "brass-ai-core", "--example", "allocation_probe", "--locked", "--offline")
        allocation = self.run("allocation", [ROOT / "artifacts/rust-native-target" / TARGET / "release/examples/allocation_probe.exe"])
        write(self.out / "allocation.json", read(allocation))
        benchmark_root = ROOT / "artifacts/rust-native-target/criterion"
        if not benchmark_root.is_dir():
            raise RuntimeError("Criterion raw output missing")
        shutil.copytree(benchmark_root, self.out / "criterion", dirs_exist_ok=True)
        metadata = self.cargo_run("metadata", "metadata", "--format-version", "1", "--locked", "--offline")
        write(self.out / "cargo-metadata.json", read(metadata))
        self.cargo_run("runtime-dependencies", "tree", "-p", "brass-ai-worker", "--edges", "normal", "--locked", "--offline")
        self.cargo_run("features", "tree", "--workspace", "--edges", "features", "--locked", "--offline")
        write(self.out / "preservation-after.json", verify_frozen())
        if read(self.out / "preservation-before.json") != read(self.out / "preservation-after.json"):
            raise RuntimeError("Reference/install identity changed during qualification")
        inputs = list(NATIVE.rglob("*")) + list((ROOT / "src/brass_ai/native").glob("*.py")) + list((ROOT / "tests/RustBackendNative").glob("*.py")) + [Path(__file__)]
        sources = [receipt(path) for path in sorted(inputs) if path.is_file() and path.suffix in (".rs", ".toml", ".lock", ".json", ".py")]
        result = dict(format="brass-native-l03-qualification/1.0.0", status="passed", milestone="L03",
            completed_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
            python=sys.version, builds=builds, commands=self.commands, sources=sources,
            toolchain=dict(rust=TOOLCHAIN, target=TARGET, llvm_archive=receipt(base / (LLVM + ".zip")),
                rustup_installer=receipt(base / "rustup-init.exe"),
                rust_linker=receipt(self.env["CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER"]),
                dlltool=receipt(base / LLVM / "bin/dlltool.exe"),
                clang=receipt(base / LLVM / "bin/clang.exe")),
            session_count=2, threads_per_session=1, installed_backend_changed=False,
            search_available=False, target_rules_qualified=False, nn_executed=False,
            matched_search_seconds=4, matched_search_run=False,
            next_milestone="L04: complete state import, export and restoration")
        write(self.out / "qualification.json", result)
        write(self.out / "progress.json", dict(status="passed", milestone="L03", next="L04"))
        print("L03 qualification passed: " + str(self.out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "artifacts/rust-backend"):
        parser.error("evidence must be under artifacts/rust-backend")
    qualification = Qualification(out)
    try:
        qualification.qualify()
    except BaseException as error:
        if not (out / "qualification.json").exists():
            write(out / "progress.json", dict(status="failed", milestone="L03", error=str(error)))
        raise


if __name__ == "__main__":
    main()
