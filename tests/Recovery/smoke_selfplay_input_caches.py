"""Two bounded PPO steps and an exact checkpoint restart on fresh tensors."""
import argparse
import copy
from pathlib import Path
import time

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.learning_rate import configure
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.updates import ppo_update
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json
from smoke_categorical_kl_guard import examples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated output directory')
    out.mkdir(parents=True);began=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(617)
    config=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-entropy95.json')
    reward=reward_spec(config)
    model=UtilityNet(utility_config(ModelConfig.preset(config['model_size']),reward))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    configure(optimizer,1e-6,100,.1,'selfplay')
    behavior,data=examples(model,64,forced=True)
    path=out/'incoming.pt';save_checkpoint(path,model,optimizer,step=0)
    initial_hash=sha(path)
    cfg=dict(config['update'],epochs=1,effective_batch_size=32,microbatch_size=16,critic_batch_size=32)
    results=[];reports=[]
    for _ in range(2):
        restored,raw=load_checkpoint(path)
        opt=torch.optim.AdamW(restored.parameters(),lr=1e-6);resume_optimizer(raw,opt)
        report=ppo_update(restored,opt,[dict(split='train')],behavior,cfg,learner_data=data,
                          insolvency=reward['insolvency'],reward=reward)
        results.append((copy.deepcopy(restored.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone()))
        reports.append(report)
    require(_same_payload(results[0],results[1]),'Checkpoint restart changed weights/Adam/RNG')
    require(sha(path)==initial_hash,'Incoming checkpoint was changed')
    require(all(r['kl_guard']['probe_checks']==r['optimizer_steps'] for r in reports),'Skipped a KL guard')
    write_json(out/'report.json',dict(passed=True,seconds=time.perf_counter()-began,
        parameters=sum(p.numel() for p in model.parameters()),rows=len(data),
        optimizer_steps=reports[0]['optimizer_steps'],exact_restart=True,incoming_checkpoint_unchanged=True,
        strict_behavior_verification=True,kl_guard=reports[0]['kl_guard'],
        input_cache=reports[0]['input_cache'],sample_usage=reports[0]['sample_usage'],
        scope='Fresh current UtilityNet, 64 authored tensor rows, synthetic terminal labels; two steps repeated after exact restart, no gameplay evaluation'))
    print(read(out/'report.json'))


if __name__=='__main__':main()
