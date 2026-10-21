"""Preview/apply fixed 95% entropy floors, T=2 openings and 1.5% self-play forks."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def run(root,run_path,output,*,apply=False,expected_generation=None,device='cpu'):
    import torch
    from brass_ai.bank import read
    from brass_ai.learning.checkpoint import resume_optimizer
    from brass_ai.learning.contracts import project_path,require,digest
    from brass_ai.step5.league import implementation_id,write_json
    from brass_ai.recovery import exploration as E,policy_anchor
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import load_resume,commit_transition,apply_optimizer_config
    from brass_ai.recovery.retention import owned,run_lock
    root,run=owned(root,run_path);output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json',
            'Config output must be under configs/step5/')
    with run_lock(run):
        state=read(run/'state.json');generation=state['iteration'];old=state['signature']['config']
        require(expected_generation is None or generation==expected_generation,'Unexpected committed generation')
        require(all(b in old for b in E.BLOCKS),'Entropy hotfix requires an existing exploration rollout')
        require(old['entropy_control']['contract']!=E.ENTROPY['contract'],
                'Group floors already adopted; use joint7-entropy-groups.py instead of the superseded entropy95 hotfix')
        config=validate_config(E.frozen_config(old))
        require(not output.exists() or read(output)==config,'Existing config differs; choose another --output')
        signature=dict(copy.deepcopy(state['signature']),config=config,implementation=implementation_id())
        torch.set_num_threads(config['threads'])
        with torch.random.fork_rng(devices=[]):
            model,raw,changed=load_resume(root,run,state,signature,device=device,reviewed=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            resume_optimizer(raw,optimizer,allow_cuda_topology_change=True);apply_optimizer_config(optimizer,config)
            saved=E.restore(config,state,raw['training'].get('exploration'))
            # A rating may have promoted the anchor after the last PPO checkpoint.
            # Verify its overlay without clearing it or repeating calibration.
            if 'policy_anchor' in state:policy_anchor.verify_checkpoint(state,raw)
        result=dict(applied=apply,generation=generation,games_done=state['games_done'],
            config=output.relative_to(root).as_posix(),baseline=saved['baseline'],
            floor_before=E.floors(old,saved,generation),floor_after=E.floors(config,saved,generation),
            fixed_floor_ratio=.95,temperature=2.,action_decisions_per_seat=2,
            branch_rate_per_eligible_decision=.015,branch_openings='First two action decisions per seat after the excluded fork move',
            checkpoint=state['history'][-1],exploration_state_digest=digest(saved),
            preserved='Checkpoint bytes, weights, both Adams/RNG, progress, baseline/alphas/windows, rollout age, KL-reference overlay, economic/GAE/R2 evidence and pool')
        if apply:
            if not output.exists():write_json(output,config)
            if changed:state=commit_transition(run,state,signature)
            require(state['exploration']==saved,'Entropy hotfix altered controller evidence')
            report=run/'exploration'/'hotfix-report.json'
            # Keep the first transition's old floors on idempotent reruns.
            if changed or not report.exists():write_json(report,result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-entropy95.json')
    p.add_argument('--apply',action='store_true');p.add_argument('--expected-generation',type=int)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    print(json.dumps(run(ROOT,a.run,a.output,apply=a.apply,expected_generation=a.expected_generation,
                         device=a.device),indent=2,allow_nan=False))


if __name__=='__main__':main()
