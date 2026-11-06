"""Isolated CPU PPO before/after timing on authored tensors, not game strength."""
import argparse
import copy
from pathlib import Path
from statistics import median
import time

import torch

from brass_ai.learning.contracts import require
from brass_ai.recovery import updates,ppo_kl_control as K
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_categorical_kl_guard import GuardTests


def run(output,repeats):
    require(not output.exists(),'Use a new isolated output directory')
    require(1<=repeats<=5,'Bounded smoke supports 1-5 repeats')
    torch.set_num_threads(2)
    base_model,base_opt,behavior,rows,cfg=GuardTests().fixture()
    # Small synthetic batches can reach the cap quickly. Use an explicitly
    # reduced fixture LR so both timed paths complete the same 48-step budget.
    from brass_ai.recovery.learning_rate import configure
    base_opt=torch.optim.AdamW(base_model.parameters(),lr=6e-7,weight_decay=1e-4)
    configure(base_opt,6e-7,0,.1,'selfplay')
    # 48 nonforced authored rows / batch 2 / two epochs = 48 optimizer steps.
    data=[(copy.deepcopy(row),target,(f'{identity[0]}-{rep}',identity[1]),encoded)
        for rep in range(6) for row,target,identity,encoded in rows]
    cfg.update(epochs=2,effective_batch_size=2)
    weights=copy.deepcopy(base_model.state_dict());adam=copy.deepcopy(base_opt.state_dict())
    rng=torch.get_rng_state().clone();samples={'before':[],'after':[]};reports={};final={}
    # One untimed warmup per path followed by alternating paired repetitions.
    for repeat in range(repeats+1):
        for name in (('before','after') if repeat%2==0 else ('after','before')):
            base_model.load_state_dict(weights);base_opt.load_state_dict(copy.deepcopy(adam));torch.set_rng_state(rng)
            config=dict(cfg,kl_control=copy.deepcopy(K.DEFAULT if name=='before' else K.SPARSE_DEFAULT))
            began=time.perf_counter()
            result=updates.ppo_update(base_model,base_opt,[dict(split='train')],behavior,config,learner_data=data)
            elapsed=time.perf_counter()-began
            require(not result['kl_early_stop'] and result['optimizer_steps']==48,'Unexpected smoke early stop')
            if repeat:samples[name].append(elapsed)
            reports[name]=result
            final[name]=copy.deepcopy((base_model.state_dict(),base_opt.state_dict(),torch.get_rng_state()))
        require(_same_payload(final['before'],final['after']),'Observation cadence changed PPO weights/Adam/RNG')
    require(reports['before']['kl_guard']['probe_checks']==48 and reports['after']['kl_guard']['probe_checks']==13,
        'Unexpected probe cadence')
    require(reports['before']['kl_guard']['full_dataset_checks']==2 and reports['after']['kl_guard']['full_dataset_checks']==0,
        'Unexpected logging scan count')
    result=dict(device='cpu',threads=2,fixture_lr=6e-7,model=base_model.config.to_dict(),rows=len(data),steps=48,repeats=repeats,
        seconds=samples,median_seconds={k:median(v) for k,v in samples.items()},weights_Adam_RNG_equal=True,
        probe={k:r['categorical_probe'] for k,r in reports.items()},
        guard={k:r['kl_guard'] for k,r in reports.items()},timings={k:r['timings'] for k,r in reports.items()},
        scope='CPU correctness/timing smoke on repeated authored legal-action tensors, fixed seeds/weights and two epochs. Probe has 48 rows, not production 2048; no collection, production checkpoints, CUDA measurement or strength evaluation.')
    output.mkdir(parents=True);write_json(output/'report.json',result)
    print(result['median_seconds'],flush=True);print(output/'report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--repeats',type=int,default=3);a=p.parse_args();run(a.output,a.repeats)
