"""Bounded v11.1 PPO/checkpoint smoke and a 2048-position CPU guard benchmark.

Uses authored feature tensors and fresh weights in a separate output folder.
This checks computation and continuation, not game legality or playing strength.
"""
import argparse
from pathlib import Path
from statistics import median
import time

import torch

from test_mlp_v11 import inputs as model_inputs
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,require,digest
from brass_ai.learning.features import Encoded
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import lr_controller
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.learning_rate import configure
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.review_metrics import CategoricalProbe
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset,ppo_update,probabilities,policy_measure,phase_probe
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.step5.league import implementation_id,write_json
from brass_ai.bank import read,sha


def examples(model,count,*,forced=False):
    states,actions,_,mapping=model_inputs(count,32)
    behavior=Behavior(model_id(model),.08);data=[]
    target=dict(win_share=[1.,0.,0.,0.],placement=[1.,2.,3.,4.],vp=[100.,90.,80.,70.])
    reward=model.config.utility
    placements=[1.,2/3,1/3,0.]
    returns=[reward['win']*w+reward['placement']*p+reward['vp']*v
             for w,p,v in zip(target['win_share'],placements,target['vp'])]
    kinds=[dict(kind='loan'),dict(kind='pass'),dict(kind='build',industry='IronWorks'),dict(kind='build',industry='CoalMine')]
    for i in range(count):
        n=1 if forced and i%8==0 else (2,8,16,32)[i%4]
        choices=[dict(kinds[j%4]) for j in range(n)]
        row=dict(chosen=i%n,seat=i%4,intent=[0.]*8,choices=choices,
            behavior=dict(kind='model',policy_id=behavior.policy_id,probability=1.),
            view=dict(context=dict(era='CanalEra' if i%4<2 else 'RailEra',round=2 if i%2==0 else 7,phase='action')))
        seat=row['seat']
        row.update(teacher=None,_reward_id=digest(reward),_reward_return=returns,
            _reward_remaining=[reward['insolvency']['max_cost']]*4,
            _reward_trajectory=dict(win=target['win_share'][seat],placement=placements[seat],vp=target['vp'][seat],cost=0.,utility=returns[seat]),
            _insolvency_cost_to_go=0.,_insolvency_trajectory=dict(events=0,shortfall=0,cost=0.,by_era={}))
        encoded=Encoded(states[i].numpy(),actions[i,:n].numpy(),mapping[i].numpy(),tuple(str(j) for j in range(n)))
        data.append((row,target,(f'g-{i//4}',i%4),encoded))
    ds=Dataset(data,'cpu');model.eval()
    with torch.no_grad():
        for part in ds.microbatches(range(count),128,131072):
            ix,inputs,q=ds.batch(part,.08)
            p=probabilities(model(*inputs)[0],q,.08,1.)
            for j,i in enumerate(part):data[i][0]['behavior']['probability']=float(p[j,data[i][0]['chosen']])
    return behavior,data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new workspace smoke output')
    out.mkdir(parents=True);began=time.monotonic();torch.set_num_threads(2);torch.manual_seed(731)
    config=validate_config(read(SOURCE_ROOT/'configs/step5/joint7-v11-1-grouped-lr.json'))
    model=FourSeatNet(ModelConfig.preset(config['model_size']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    reward=reward_spec(config);migrate(model,optimizer,reward)
    configure(optimizer,config['learning_rate'],100,.1,'selfplay')
    lr_controller.configure(optimizer,config['ppo_lr_controller'])
    behavior,data=examples(model,48,forced=True)
    cfg=dict(config['update'],effective_batch_size=16,critic_batch_size=16,microbatch_size=8)
    print('PPO smoke: 48 authored rows, current medium v11.1 model',flush=True)
    first=ppo_update(model,optimizer,[dict(split='train')],behavior,cfg,learner_data=data,
                     reward=reward,insolvency=reward['insolvency'])
    checkpoint=out/'generation-0001.pt'
    save_checkpoint(checkpoint,model,optimizer,step=1,training=dict(mode='isolated-categorical-kl-smoke'))
    checkpoint_hash=sha(checkpoint)
    restored,raw=load_checkpoint(checkpoint)
    resumed=torch.optim.AdamW(restored.parameters(),lr=config['learning_rate'])
    resume_optimizer(raw,resumed)
    require(_same_payload(optimizer.state_dict(),resumed.state_dict()),'Adam/warmup/controller did not restore exactly')
    behavior,following=examples(model,48,forced=True)
    rng=torch.get_rng_state().clone();seconds=[]
    for net,opt in ((model,optimizer),(restored,resumed)):
        torch.set_rng_state(rng)
        seconds.append(ppo_update(net,opt,[dict(split='train')],behavior,cfg,learner_data=following,
                                  reward=reward,insolvency=reward['insolvency']))
    require(_same_payload(model.state_dict(),restored.state_dict()),'Resumed next PPO weights differ')
    require(_same_payload(optimizer.state_dict(),resumed.state_dict()),'Resumed next Adam state differs')
    require(seconds[0]['kl_guard']==seconds[1]['kl_guard'] and seconds[0]['epochs']==seconds[1]['epochs'],'Resumed guard/step decisions differ')
    require(sha(checkpoint)==checkpoint_hash,'Smoke overwrote its starting checkpoint')
    print('Checkpoint continuation: bitwise matching weights and Adam after next PPO update',flush=True)
    behavior,data=examples(model,2048);ds=Dataset(data,'cpu');model.eval()
    bench_cfg=dict(config['update'],microbatch_size=128)
    selected=phase_probe(ds,list(range(len(data))),2048)
    probe=CategoricalProbe(model,ds,selected,behavior,bench_cfg)
    # Perturb a real scorer tensor once, then measure both guards on the same
    # stationary model/probe. No optimizer or RNG draws enter timed sections.
    with torch.no_grad():model.policy_scorer[-1].weight.mul_(1.5)
    require(probe.measure_step(model)['categorical_kl']>1e-6,'Benchmark policy perturbation did not change the distribution')
    measured={name:[] for name in ('sampled_only','exact_and_sampled')}
    def run(name):
        if name=='sampled_only':return policy_measure(model,ds,selected,behavior,bench_cfg,Timings())
        with Timings().phase('ppo.kl_probe'):return probe.measure_step(model)
    for name in measured:run(name)
    print('Guard benchmark: 2048 positions, 2-32 legal candidates, CPU with 2 threads',flush=True)
    for repeat in range(3):
        for name in (list(measured) if repeat%2==0 else list(reversed(measured))):
            start=time.perf_counter();run(name);measured[name].append(time.perf_counter()-start)
    medians={name:median(values) for name,values in measured.items()}
    update_keys=('optimizer_steps','optimizer_steps_planned','optimizer_steps_fraction','kl_early_stop','kl_guard',
                 'categorical_probe','epochs','meaningful_decisions','forced_decisions','timings','learning_rate_schedule',
                 'utility_calibration','reward')
    report=dict(implementation=implementation_id(),model=model.config.to_dict(),parameters=sum(p.numel() for p in model.parameters()),
        device='cpu',threads=2,elapsed_seconds=time.monotonic()-began,checkpoint_sha256=checkpoint_hash,
        resumed_next_update_bitwise_equal=True,updates=[{k:r[k] for k in update_keys} for r in (first,seconds[0])],
        benchmark=dict(positions=2048,microbatch_size=128,min_candidates=2,max_candidates=32,repeats=3,seconds=measured,median_seconds=medians,
            exact_to_sampled_ratio=medians['exact_and_sampled']/medians['sampled_only'],measurement=probe.measure_step(model)),
        scope='Isolated CPU computation smoke with fresh medium v11.1 weights and authored Features-4 tensors. No production checkpoints, complete game collection, CUDA benchmark or playing-strength evaluation.')
    write_json(out/'report.json',report)
    print('Report:',out/'report.json',flush=True)
    print('Guard median seconds:',medians,flush=True)


if __name__=='__main__':main()
