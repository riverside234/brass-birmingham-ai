"""Isolated v11.2 value progress and real-KL prefix rollback computation."""
import argparse
import copy
import time

import torch

from test_gae_value_phase import fixture,ppo
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import value_phase
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    out.mkdir(parents=True);torch.set_num_threads(2);began=time.perf_counter();identity=implementation_id()
    model,adam,cfg,(behavior,data)=fixture(preset='medium',games=8,decisions=4)
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    cfg['value_phase'].update(batch_size=16,max_epochs=1,clone_rows=128,clone_batch_size=32)
    phase=TrainingPhases(model,cfg);actor=ppo(model,adam,cfg,behavior,data,phase)
    start_model=copy.deepcopy(model.state_dict());start_adam=copy.deepcopy(phase.optimizer.state_dict())
    ppo_adam=copy.deepcopy(adam.state_dict());rng=torch.get_rng_state().clone()
    # Measure a bounded normal pass with the production 0.002 cap/LR/beta.
    original=value_phase.RawPolicyReference.measure;trace=[]
    from unittest.mock import patch
    def measure(ref,*args,**kwargs):
        result=original(ref,*args,**kwargs);trace.append(result);return result
    with patch.object(value_phase.RawPolicyReference,'measure',measure):normal=phase.fit(model)
    require(normal['optimizer_steps']>0,'No accepted normal value progress in authored smoke')
    require(normal['probe_kl']<=.002 and normal['heldout_after']['loss']<=normal['heldout_before']['loss'],
            'Normal value acceptance failed')
    # Pick a STRICTER diagnostic cap between two observed prefix KLs. No fake
    # KL or loss: rerun the same initial weights/Adam/RNG and test the transaction.
    steps=trace[:normal['attempted_steps']]
    crossing=next((i for i in range(1,len(steps)) if steps[i]>max(steps[:i])+1e-9),None)
    require(crossing is not None,'Diagnostic pass needs a rising multi-step KL prefix')
    cap=(max(steps[:crossing])+steps[crossing])/2
    model.load_state_dict(start_model);phase.optimizer.load_state_dict(start_adam);torch.set_rng_state(rng)
    phase.options['value_phase']['kl_cap']=cap
    stress=phase.fit(model)
    require(stress['partial_epochs']==1 and stress['optimizer_steps']==crossing and
            stress['attempted_steps']==crossing+1 and stress['kl_guard_tripped'],
            'Strict diagnostic cap did not retain the expected safe prefix')
    require(stress['probe_kl']<=cap and stress['heldout_after']['loss']<=stress['heldout_before']['loss'],
            'Strict prefix acceptance failed')
    require(_same_payload(ppo_adam,adam.state_dict()),'Value fitting changed PPO Adam')
    require(identity==implementation_id(),'Source changed during smoke')
    write_json(out/'report.json',dict(passed=True,implementation=identity,seconds=time.perf_counter()-began,
        parameters=model.parameter_count,normal=normal,stress=stress,normal_step_kls=steps,actor=actor,
        production_cap=.002,diagnostic_cap=cap,ppo_optimizer_unchanged=True,
        scope='Authored medium v11.2 CPU tensors, 8 games/128 rows, 1-4 legal actions, two threads. '
              'Normal 0.002 cap and stricter diagnostic cap; base value LR/beta unchanged. '
              'No production checkpoints, native-game strength or lab throughput evidence.'))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
