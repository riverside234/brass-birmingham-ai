#!/usr/bin/env python3
"""Continue the qualified Rust run with policy teaching on and head Stage A."""
import importlib.util
from pathlib import Path

if __name__=='__main__':
    path=Path(__file__).with_name('continue-rust-shadow.py')
    spec=importlib.util.spec_from_file_location('rust_teacher_continuation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main(default_mode='on')
