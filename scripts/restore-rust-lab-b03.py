#!/usr/bin/env python3
"""Recheck a sealed B03 diagnostic fork; never advance or write the source run."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork',type=Path,required=True);p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--allow-cuda-topology-change',action='store_true')
    a=p.parse_args()
    from brass_ai.rust_backend.lab_continuation import DiagnosticFork,restore
    from brass_ai.recovery.retention import run_lock
    fork=DiagnosticFork(ROOT,a.fork,generation=a.expected_generation,checkpoint_sha256=a.checkpoint_sha256)
    with run_lock(fork.run):
        value=restore(fork,device=a.device,allow_cuda_topology_change=a.allow_cuda_topology_change)
        print(json.dumps(value.report,indent=2))


if __name__=='__main__':main()
