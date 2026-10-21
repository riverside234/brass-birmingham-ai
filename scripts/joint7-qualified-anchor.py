"""Review/apply the qualified KL-reference, rating-coverage and epsilon=.02 continuation."""
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
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.learning.paths import bound_reference
    from brass_ai.step5.league import implementation_id,write_json
    from brass_ai.recovery import policy_anchor,panel_gates,rating_coverage
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.resume import load_resume,commit_transition,apply_optimizer_config
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.recovery.demonstrations import completed_imitation
    from brass_ai.recovery.il_kl_probe import load as load_probe
    from brass_ai.recovery.g110_hotfix import verify_checkpoint as verify_old_limit
    root,run=owned(root,run_path);output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Config output must be under configs/step5/')
    with run_lock(run):
        state=read(run/'state.json');generation=state['iteration']
        require(expected_generation is None or generation==expected_generation,'Unexpected committed generation')
        config=validate_config(policy_anchor.configure(state['signature']['config']))
        torch.set_num_threads(config['threads'])
        require(not output.exists() or read(output)==config,'Existing config differs; choose another --output')
        signature=dict(copy.deepcopy(state['signature']),config=config,implementation=implementation_id())
        with torch.random.fork_rng(devices=[]):
            model,raw,changed=load_resume(root,run,state,signature,device=device,reviewed=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            resume_optimizer(raw,optimizer,allow_cuda_topology_change=True);apply_optimizer_config(optimizer,config)
            if 'policy_anchor' in state:policy_anchor.verify_checkpoint(state,raw)
            else:verify_old_limit(state,raw)
            origin=state['signature']['imitation_source']
            bc,actual=completed_imitation(root,origin['bc_run'])
            actual=bound_reference(root,actual,origin,('bc_run','path','manifest'))
            probe,_=load_probe(root,run,actual,bc,config['il_kl_limit'],config['update'],config['audit_workers'],
                receipt=state['il_kl_limit']['probe'])
            gates=panel_gates.prepare(root,state,config)
            preview=dict(state,ratings=dict(state['ratings'],panel_gates=gates))
            coverage=rating_coverage.prepare(root,preview,config)
            controller=state.get('policy_anchor',policy_anchor.initial(generation))
            selected,_,reason=(policy_anchor.select(root,state,controller) if gates['retired']['il-panel'] else
                (None,None,'il_panel_not_qualified'))
            if apply:
                if not output.exists():write_json(output,config)
                if changed:state=commit_transition(run,state,signature)
                state['ratings'].update(panel_gates=gates,coverage=coverage)
                write_json(run/'state.json',state)
                _,transition=policy_anchor.prepare(root,run,state,config,probe,model)
                policy_anchor.verify_checkpoint(state,raw)
                policy_anchor.console(transition)
            else:transition=None
        result=dict(applied=apply,generation=generation,games_done=state['games_done'],
            config=output.relative_to(root).as_posix(),epsilon=config['exploration'],
            proposed_reference=selected,selection_reason=reason,coverage=coverage,transition=transition,
            active_reference=(state.get('policy_anchor') or {}).get('active'),
            preserved='Original IL/probe, checkpoint bytes, both Adams/RNG, warmup/LR multiplier, economic baseline/rolling data, GAE/R2, population and progress')
        if apply:write_json(run/'policy-anchor'/'hotfix-report.json',result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-grouped-lr-critic-qualified-anchor.json')
    p.add_argument('--apply',action='store_true');p.add_argument('--expected-generation',type=int)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();print(json.dumps(run(ROOT,a.run,a.output,apply=a.apply,
        expected_generation=a.expected_generation,device=a.device),indent=2,allow_nan=False))


if __name__=='__main__':main()
