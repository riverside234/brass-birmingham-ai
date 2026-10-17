"""Three real games from existing v11.1 IL, isolated output, no search/evaluation."""
import argparse
import copy
from pathlib import Path
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,read_json,require
from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer
from brass_ai.recovery.config import latest_model_config
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.recovery.league import train
from brass_ai.recovery.resume import load_resume,apply_optimizer_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.lr_controller import KEY
from brass_ai.step5.league import write_json,implementation_id


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imitation-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(not out.exists(),'Use a new isolated output directory')
    began=time.monotonic();torch.set_num_threads(2)
    il=project_path(SOURCE_ROOT,args.imitation_run)
    protected={p:sha(p) for p in il.iterdir() if p.is_file()}
    model,origin=completed_imitation(SOURCE_ROOT,il)
    cfg=latest_model_config(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json'),origin)
    cfg['rehearsal_manifest']=origin['manifest']
    out.mkdir(parents=True);write_json(out/'production-config.json',cfg)
    cfg.update(games_per_iteration=3,threads=2,audit_workers=1)
    if 'il_kl_limit' not in cfg:cfg['rehearsal_weight']=0.
    cfg['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1)
    cfg['evaluation_schedule']['enabled']=False
    cfg['update'].update(epochs=1,effective_batch_size=512,critic_batch_size=64,microbatch_size=16,max_candidate_cells=8192)
    write_json(out/'smoke-config.json',cfg)
    result=train(SOURCE_ROOT,out/'run',cfg,iterations=1,imitation_run=il,fresh=True)
    report=result['summaries'][0];state=read_json(out/'run/state.json')
    initial,raw=load_checkpoint(out/'run/initial.pt')
    require(_same_payload(model.state_dict(),initial.state_dict()),'IL weights changed during handoff')
    require(not raw['optimizer']['state'],'IL optimizer leaked into PPO')
    require(state['games_done']==3 and state['iteration']==1,'Smoke exceeded its bound')
    require(report['composition']==dict(selfplay=2,historical=1),'Unexpected smoke table assignment')
    require(report['sampling']['learner_trajectories']==10,'Expected eight self-play and two history learner seats')
    require(report['search']['enabled'] is False and report['search']['optimizer_steps']==0,'Teacher ran while disabled')
    health=report['actor']['lr_controller']['health']
    require(health['anchor_group_budget']==.30 and health['anchor_group_min_positions']==64,
            'Grouped anchor gate settings differ')
    if 'il_kl_limit' in cfg:
        initial_probe=read_json(out/'run/il-kl/initial-report.json')
        require(initial_probe['measurement']['kl_mean']==0.,'Initial IL KL differs from zero')
        probe=report['il_kl_limit']['measurement'];groups=probe['groups']
        require(sum(g['positions'] for g in groups.values())==probe['positions'] and probe['unassigned_positions']==0,
                'Fixed probe group counts differ')
        require(report['actor']['rehearsal_active'] and report['rehearsal_schedule']['floor']==.03,
                'New rehearsal schedule was not exercised')
        require(report['by_composition']['selfplay']['health_counts']['seat_games']==8 and
                report['by_composition']['historical']['health_counts']['seat_games']==2,'Health attribution differs')
    else:
        probe=report['actor']['categorical_probe'];groups=probe['anchor_groups']
        require(probe['anchor_unassigned_positions']==0 and sum(g['positions'] for g in groups.values())==probe['positions'],
                'Ordinary replay compaction lost anchor phase metadata')
        require(all(g['positions']>0 and g['anchor_distance'] is not None for g in groups.values()),
                'Smoke did not measure all four anchor groups')
    require(all(health['anchor_groups'][name]['positions']==g['positions'] for name,g in groups.items()),
            'Controller lost group probe counts')
    require('legacy_boost' not in state,'Old v4 schedule activated')
    before=copy.deepcopy(state)
    restored,checkpoint,changed=load_resume(SOURCE_ROOT,out/'run',state,state['signature'])
    optimizer=torch.optim.AdamW(restored.parameters(),lr=cfg['learning_rate'])
    resume_optimizer(checkpoint,optimizer)
    adam=copy.deepcopy(optimizer.state_dict()['state']);control=copy.deepcopy(optimizer.param_groups[0][KEY])
    apply_optimizer_config(optimizer,cfg)
    require(not changed and _same_payload(adam,optimizer.state_dict()['state']) and optimizer.param_groups[0][KEY]==control,
            'Adam/controller resume differed')
    require(torch.equal(torch.get_rng_state(),checkpoint['rng']),'Resume RNG differed')
    if 'il_kl_limit' in cfg:
        from brass_ai.recovery.il_kl_probe import load as load_probe
        require(checkpoint['training']['il_kl_limit']==state['il_kl_limit'],'Calibrated limit checkpoint/state differ')
        rng=torch.get_rng_state().clone()
        restored_probe,_=load_probe(SOURCE_ROOT,out/'run',origin,initial,cfg['il_kl_limit'],cfg['update'],1,
            receipt=state['il_kl_limit']['probe'])
        repeated=restored_probe.measure(restored,cfg['update'])
        require(abs(repeated['kl_mean']-probe['kl_mean'])<1e-10,'Restored fixed probe measurement differs')
        require(torch.equal(rng,torch.get_rng_state()),'Fixed probe changed training RNG')
    require(state==before and read_json(out/'run/state.json')==before,'Resume changed committed state')
    require(all(sha(p)==value for p,value in protected.items()),'Source IL files changed')
    result=dict(passed=True,seconds=time.monotonic()-began,implementation=implementation_id(),
        architecture=model.config.architecture,imitation_source=origin,source_files_unchanged=True,
        composition=report['composition'],sampling=report['sampling'],controller=report['actor']['lr_controller'],
        categorical_probe=report['actor']['categorical_probe'],teacher=report['search'],
        il_kl_limit=report.get('il_kl_limit'),rehearsal_schedule=report.get('rehearsal_schedule'),
        ppo_steps=report['actor']['optimizer_steps'],ppo_steps_planned=report['actor']['optimizer_steps_planned'],
        checkpoint_adam_rng_controller_resume_exact=True,compact_history=bool((out/'run/retention-history.json').exists()),
        scope='Three CPU games, production v11.1 model and old-source completed diagnostic IL. No v4 search, teacher collection or strength evaluation; LR decisions held during warmup.')
    write_json(out/'report.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
