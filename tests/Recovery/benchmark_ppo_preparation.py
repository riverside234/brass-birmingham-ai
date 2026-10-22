"""Small paired PPO update: 0 vs 4/8 preparation processes, exact state comparison."""
import argparse
import copy
from pathlib import Path
import time

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.ppo_preparation import PreparationPool
from brass_ai.recovery.updates import ppo_update
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json
from smoke_categorical_kl_guard import examples
from benchmark_single_gpu import measured


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--rows',type=int,default=2048)
    p.add_argument('--workers',type=int,nargs='+',default=[0,4,8]);p.add_argument('--repeats',type=int,default=1)
    a=p.parse_args();out=a.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated output')
    require(2048<=a.rows<=8192 and 1<=a.repeats<=3 and all(w in (0,4,8) for w in a.workers),'Bounded benchmark only')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(470)
    cfg=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only-g460.json');reward=reward_spec(cfg)
    model=UtilityNet(utility_config(ModelConfig.preset('medium'),reward)).eval()
    behavior,data=examples(model,a.rows,forced=True);initial=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    config=dict(cfg['update'],microbatch_size=512,max_candidate_cells=131072,epochs=1)
    expected=None;arms=[]
    # Warm spawned workers separately, as the trainer does during collection.
    for workers in a.workers:
        with PreparationPool(workers) as pool:
            for f in pool.ready:f.result()
            for repeat in range(a.repeats):
                net=copy.deepcopy(model).to(a.device);net.load_state_dict(initial)
                opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                torch.set_rng_state(rng);resource={}
                if torch.device(a.device).type=='cuda':torch.cuda.reset_peak_memory_stats(a.device)
                with measured(resource):
                    report=ppo_update(net,opt,[dict(split='train')],behavior,dict(config,preparation_workers=workers),
                        learner_data=data,insolvency=reward['insolvency'],reward=reward,preparation_pool=pool)
                state=(net.state_dict(),opt.state_dict(),torch.get_rng_state(),report['epochs'],report['kl_guard'])
                if expected is None:expected=copy.deepcopy(state)
                require(_same_payload(state,expected),'Process preparation changed weights/Adam/RNG/guard')
                arms.append(dict(workers=workers,repeat=repeat,resources=resource,report=report,exact_state_parity=True))
                write_json(out/'report.json',dict(passed=True,device=a.device,rows=len(data),parameters=model.parameter_count,
                    arms=arms,scope='Authored 2-32 candidate rows, one bounded epoch plus unchanged head calibration; '
                    'current medium model. Process startup warmed outside timer; CPU benchmark cannot prove GPU gains.'))
                print('PPO workers',workers,'seconds',round(resource['elapsed_seconds'],3),'steps',report['optimizer_steps'],flush=True)


if __name__=='__main__':main()
