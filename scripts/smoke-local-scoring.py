"""Isolated eight-iteration fixture, native parity and CPU microbenchmark; no lab run."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for path in ('src','tests/Recovery','tests/FeatureEngineering','tests/R2'):sys.path.insert(0,str(ROOT/path))


def native(output):
    from brass_ai.host import RulesHost
    from brass_ai.recovery.host_runtime import local_scoring_host
    from brass_ai.learning.contracts import digest
    host_path=local_scoring_host(ROOT);runs=[]
    for seed in (153,154):
        traces=[]
        for record in (False,True):
            began=time.perf_counter();trace=[];label_time=0.
            with RulesHost(host_path) as h:
                c=h.reset(seed)
                if record:h.call('local_scoring_enable')
                for step in range(2500):
                    if c['terminal']:break
                    view=h.observe(c['actor']);choices=h.choices()
                    trace.append(digest(dict(view=view,choices=choices,hash=h.call('canonical_hash'))))
                    # Fixed deterministic choices exercise actual native scoring,
                    # while neither the labels nor a future game affect actions.
                    priority={'sell':0,'build':1,'network':2,'loan':3,'finish':0,'pass':9}
                    choice=min(choices,key=lambda x:(priority.get(x['kind'],4),x['id']))
                    c=h.apply(c,choice)
                    if step==63:
                        checkpoint=h.call('checkpoint');before=h.call('canonical_hash')
                        h.call('restore',checkpoint=checkpoint);c=h.call('context')
                        assert before==h.call('canonical_hash')
                assert c['terminal'],'Bounded native game did not terminate'
                trace.append(digest(h.call('result')))
                if record:
                    start=time.perf_counter();labels=h.call('local_scoring_targets');label_time=time.perf_counter()-start
                    (output/f'native-{seed}-labels.json').write_text(json.dumps(labels,indent=2))
            traces.append(trace);runs.append(dict(seed=seed,recording=record,seconds=time.perf_counter()-began,
                decisions=len(trace)-1,label_rpc_seconds=label_time,trace_sha256=digest(trace)))
        assert traces[0]==traces[1], 'Recorder altered gameplay/observation/RNG trace'
    return runs


def training(output):
    import torch
    from brass_ai.bank import sha
    from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
    from brass_ai.learning.batching import model_id
    from brass_ai.recovery import ppg_config,updates
    from brass_ai.recovery.actor_value import TrainingPhases,KEY
    from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT,KEY as AUX
    from brass_ai.recovery.retention import _same_payload
    from brass_ai.recovery.telemetry import memory
    from brass_ai.step5.league import write_json
    from test_ppg_eight import setup,collect
    from test_gae_value_phase import ppo,finish
    from test_local_scoring import annotate
    torch.set_num_threads(2);model,opt,cfg=setup();cfg=ppg_config.coverage_options(cfg)
    cfg['value_phase'].update(batch_size=128,acceptance_steps=16)
    cfg[AUX]=copy.deepcopy(DEFAULT)
    write_json(output/'fixture-config.json',cfg)
    run=output/'eight-iteration';run.mkdir();phase=TrainingPhases(model,cfg,root=ROOT,run=run)
    phase.local_scoring=Companion(model,cfg[AUX]);reports=[];files=[]
    for generation in range(1,9):
        before=time.perf_counter();behavior,data=collect(model,phase,generation,branches=True)
        annotate(data);annotate(phase.snapshot_data)
        report=ppo(model,opt,cfg,behavior,data,phase);value=phase.fit(model)
        finish(phase,report,generation)
        auxiliary=phase.local_scoring.report()
        checkpoint=run/f'generation-{generation:04d}.pt'
        save_checkpoint(checkpoint,model,opt,step=generation,training={KEY:phase.payload(),AUX:phase.local_scoring.payload(generation)})
        files.append(dict(path=str(checkpoint),sha256=sha(checkpoint)))
        write_json(run/'state.json',dict(iteration=generation));phase.committed(generation);phase.local_scoring.committed(generation)
        reports.append(dict(generation=generation,seconds=time.perf_counter()-before,ppo_steps=report['optimizer_steps'],
            ppo_kl=report['categorical_probe'],ppg=value,auxiliary=auxiliary,memory=memory('cpu')))
        print(json.dumps(dict(generation=generation,ppo=report['optimizer_steps'],ppg=value['optimizer_steps'],seconds=reports[-1]['seconds'])),flush=True)
        if generation in (4,8):
            expected_model=copy.deepcopy(model.state_dict());expected_adam=copy.deepcopy(opt.state_dict());expected_aux=phase.local_scoring.payload(generation)
            rng=torch.get_rng_state().clone();model,raw=load_checkpoint(checkpoint)
            opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
            phase=TrainingPhases(model,cfg,raw=raw,generation=generation,root=ROOT,run=run)
            phase.local_scoring=Companion(model,cfg[AUX],raw=raw,generation=generation)
            assert torch.equal(rng,torch.get_rng_state())
            assert _same_payload(expected_model,model.state_dict()) and _same_payload(expected_adam,opt.state_dict())
            assert _same_payload(expected_aux,phase.local_scoring.payload(generation))
    assert [r['generation'] for r in reports if r['ppg']['scheduled']]==[8]
    assert all(sha(Path(f['path']))==f['sha256'] for f in files)
    return reports


def benchmark(output):
    import torch
    from brass_ai.recovery import updates
    from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
    from test_gae_value_phase import fixture
    from test_local_scoring import annotate
    from brass_ai.recovery.telemetry import memory
    torch.set_num_threads(2);model,opt,cfg,(_,data)=fixture(preset='medium',games=4,decisions=3)
    annotate(data);ds=updates.Dataset(data,'cpu');a=Companion(model,copy.deepcopy(DEFAULT));a.age=7
    ix,inputs,_=ds.value_batch(range(32),model);times={};optimizers={}
    model.eval();initial=copy.deepcopy(model.state_dict())
    # Warm the initially-zero output projections so shared auxiliary backward
    # work is represented, without changing the core benchmark's start state.
    a.begin('ppo');_,local=a.forward(model,inputs,ds,ix);local.mean().backward();a.step('ppo')
    head_initial=copy.deepcopy(a.heads.state_dict())
    for enabled in (False,True):
        samples=[];core_steps=[];head_steps=[]
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
        for repeat in range(7):
            model.load_state_dict(initial);a.heads.load_state_dict(head_initial)
            model.zero_grad();a.begin('ppo');begin=time.perf_counter()
            if enabled:out,local=a.forward(model,inputs,ds,ix)
            else:out=updates.value_output(model,inputs,ds,ix);local=0.
            loss=updates.value_losses(out,tuple(t[ix] for t in ds.targets))[0]+(out[4]-ds.utility[ix]).square().mean(-1)+local
            loss.mean().backward()
            backward=time.perf_counter()-begin;begin=time.perf_counter();opt.step();core_time=time.perf_counter()-begin
            begin=time.perf_counter()
            if enabled:a.step('ppo')
            head_time=time.perf_counter()-begin
            if repeat>=2:samples.append(backward);core_steps.append(core_time);head_steps.append(head_time)
        times['with_heads' if enabled else 'core_only']=samples
        optimizers['with_heads' if enabled else 'core_only']=dict(core_seconds=core_steps,auxiliary_seconds=head_steps)
    return dict(core_parameters=model.parameter_count,auxiliary_parameters=sum(p.numel() for p in a.heads.parameters()),
        rows=32,threads=2,device='cpu',seconds=times,optimizers=optimizers,memory=memory('cpu'),cuda_available=torch.cuda.is_available(),
        scope='Same fixed inputs/core tensors; five warmed forward/backward and optimizer samples. Weight resets outside timed regions. No production checkpoints or claims of training-strength improvement.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--native-only',action='store_true');p.add_argument('--training-only',action='store_true');a=p.parse_args()
    output=(ROOT/a.output).resolve();assert output.is_relative_to(ROOT/'artifacts') and not output.exists()
    output.mkdir(parents=True);result={}
    if not a.training_only:result['native']=native(output)
    if not a.native_only:result.update(training=training(output),benchmark=benchmark(output))
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(str(output/'report.json'))


if __name__=='__main__':main()
