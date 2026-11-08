#!/usr/bin/env python3
"""Prepare/reuse the teacher's zero-cost validator without changing a run."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def prepare(args,evidence):
    from brass_ai.bank import read
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.learning.post_transfer_reward import eligible
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from brass_ai.rust_backend.lab_teacher_runtime import resolve as verification
    from brass_ai.rust_backend.lab_reward_runtime import SELECTOR,resolve
    from brass_ai.step5.league import write_json
    import rust_lab_b07_retry as retry
    root,run=owned(ROOT,args.run);selected=root/SELECTOR
    with run_lock(run):
        state=read(run/'state.json');validate_state(state)
        require(state['iteration']==args.expected_generation and eligible(state['signature']['config']),
            'Reward adapter preparation requires the current reviewed checkpoint/reward')
        checkpoint=state['history'][-1]
        require(pin(project_path(root,checkpoint['path']))['sha256']==checkpoint['sha256'],'Committed checkpoint changed')
        original=verification(root,teacher_build(state,prepared=False))
        if selected.is_file():
            result=resolve(root,original)
            return dict(reused=True,runtime=result['receipt'],generation=state['iteration'],training_changed=False)
    built=retry.build_teacher(evidence/'runtime',original,offline=args.offline,reward_support=True)
    with run_lock(run):
        current=read(run/'state.json')
        require(current==state and pin(project_path(root,checkpoint['path']))['sha256']==checkpoint['sha256'],
            'Run changed during reward adapter preparation')
        require(not selected.exists(),'Reward adapter selector appeared during preparation')
        write_json(selected,dict(built,original_verification_receipt=original['receipt']))
        result=resolve(root,original)
    return dict(reused=False,runtime=result['receipt'],generation=state['iteration'],training_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,required=True);p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    evidence=ROOT/'artifacts/rust-backend'/('reward-adapter-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    evidence.mkdir(parents=True)
    from brass_ai.step5.league import write_json
    report=prepare(args,evidence);write_json(evidence/'preparation.json',report)
    print('Reward adapter: '+str(report['runtime']['path'])+'; training unchanged.',flush=True)


if __name__=='__main__':main()
