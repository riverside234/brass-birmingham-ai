"""Bounded B04.5 contracts, actual forwards/Adam, gates, IPC and rollback."""
import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.search_value import install, correction, binding
from brass_ai.recovery.search_value_config import adopt, defaults
from brass_ai.recovery.search_value_rows import targets, validate_root, ValueBuffer
from brass_ai.recovery.search_value_training import Companion, StageMachine, clustered_gate, guarded_pass
from brass_ai.recovery.teacher_phase import plan_jobs
from brass_ai.recovery.teacher_phase_config import validate
from brass_ai.recovery.inference import InferenceService
from brass_ai.rust_backend.lab_teacher_inference import Client, predict
from brass_ai.rust_backend.lab_teacher import validate_job
from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
from brass_ai.rust_backend.lab_snapshot import tree_digest
from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
from test_b025 import config, model_fixture, example
from test_b04 import wire


def value_fixture(model, split='train', generation=10, seat=0):
    c=adopt(config()); c['teacher_phase']['mode']='shadow'
    jobs,_=plan_jobs(c,dict(iteration=generation),model_id=model_id(model),export_hash='a'*64,epoch=1,ready=True,native_source=source_fingerprint())
    job=next(j for j in jobs if j['split']==split)
    job['readiness']=dict(ready=True,reward_id=job['expected']['reward_id'])
    label,_=example(model,seat=seat,job=job)
    e=label.encoded; cap=model.config.utility['insolvency']['max_cost']
    row=dict(job=job['expected'],**{k:job['expected'][k] for k in ('generation','model_id','reward_id','game_id')},
        index=1,seat=seat,split=split,phase='canal_early',profile='full',search_value=binding(0),state=e.state.tolist(),
        mapping=e.stable_to_relative.tolist(),remaining=[cap]*4,spent=[0.]*4,r=[.25]*4,v=[.5]*4,
        n=1,counts=[1,0],returns=[[0.]*4,[0.]*4],revision=0,turn_key=['CanalEra',1,seat,seat],
        target=[.7,.3,.2,.1],u=[.7,.3,.2,.1],target_lambda=1.)
    return row,job,label


class B045(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_defaults_off_legacy_recipe_unchanged_and_allseat_explicit(self):
        c=config(); before=copy.deepcopy(c); validate(c); self.assertEqual(c,before)
        new=adopt(c); validate(new)
        model=model_fixture(); row,job,_=value_fixture(model)
        validate_job(job); self.assertEqual(job['fast_search']['k'],4)
        for key,value in [('k',8),('allowance_seconds',1.),('post_search_seconds',1.)]:
            bad=copy.deepcopy(new);bad['teacher_phase']['fast_search'][key]=value
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(new);bad['teacher_phase']['search_value']['blend']['w']=.05
        with self.assertRaises(ValueError):validate(bad)
        self.assertEqual(defaults()['stage'],'off')

    def test_zero_head_preserves_rng_and_all_ordinary_outputs(self):
        model=model_fixture().eval(); row,_=example(model); inp=collate([row.encoded],'cpu')
        before=model(*inp);rng=torch.get_rng_state().clone(); weights=copy.deepcopy(model.state_dict())
        install(model); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for old,new in zip(before,model(*inp),strict=True):self.assertTrue(torch.equal(old,new))
        for k,v in weights.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        encoded=model.encode_state(inp[0]); delta=correction(model,encoded,inp[-1],None)
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        self.assertEqual(sum(p.numel() for p in model.search_value_head.parameters()),(128+4+32)*32+32+32*4+4)

    def test_sixth_output_zero_and_nonzero_preserve_main_heads_and_chunking(self):
        model=model_fixture().eval(); install(model)
        row,_=example(model); remaining=torch.tensor([[.075]*4])
        with torch.no_grad():
            for limit in (2,8192):
                legacy=predict(model,[row.encoded],'cpu',limit,remaining)
                zero=predict(model,[row.encoded],'cpu',limit,remaining,weights=[0.])
                for i in range(5):np.testing.assert_array_equal(zero[i],legacy[i])
                np.testing.assert_array_equal(zero[5],legacy[4]);np.testing.assert_array_equal(zero[-1],legacy[-1])
            model.search_value_head[-1].bias.fill_(.25)
            got=predict(model,[row.encoded],'cpu',2,remaining,weights=[.4])
            np.testing.assert_allclose(got[5],got[4]+.1,atol=1e-6)

    def test_trunk_ramp_scales_descriptor_and_state_gradients_once(self):
        model=model_fixture().eval(); install(model)
        row,_=example(model); inp=collate([row.encoded],'cpu')
        with torch.no_grad():model.search_value_head[-1].weight.fill_(.1)
        gradients=[]
        for scale in (1.,.1):
            model.zero_grad(set_to_none=True)
            correction(model,model.encode_state(inp[0]),inp[-1],None,scale=scale).sum().backward()
            gradients.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
        self.assertTrue(any(n.startswith('descriptor_encoder.') for n in gradients[0]))
        self.assertTrue(any(not n.startswith(('descriptor_encoder.','search_value_head.')) for n in gradients[0]))
        self.assertFalse(any(n.startswith(('utility_head.','outcomes.')) for n in gradients[0]))
        for name,gradient in gradients[0].items():
            torch.testing.assert_close(gradients[1][name],gradient*(1. if name.startswith('search_value_head.') else .1),rtol=1e-4,atol=1e-7)

    def test_head_only_adam_checkpoint_resume_no_trunk_or_policy_change(self):
        model=model_fixture().eval(); comp=Companion(model); row,_,_=value_fixture(model)
        before=copy.deepcopy(model.state_dict())
        comp.head_only([row],generation=10,steps=2)
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('search_value_head.')))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if not k.startswith('search_value_head.')))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'model.pt';save_checkpoint(path,model,training={'search_value':comp.snapshot()})
            restored,raw=load_checkpoint(path);other=Companion(restored,saved=raw['training']['search_value'])
            self.assertEqual(model_id(model),model_id(restored));self.assertEqual(tree_digest(comp.snapshot()),tree_digest(other.snapshot()))
            comp.head_only([row],generation=11,steps=1);other.head_only([row],generation=11,steps=1)
            self.assertEqual(model_id(model),model_id(restored))
        bad=dict(row,split='heldout')
        with self.assertRaises(ValueError):comp.head_only([bad],generation=12)

    def test_lambda_one_costs_and_stable_seats(self):
        rows=[dict(spent=[0.,0.,0.,0.],r=[.2,.4,.6,.8]),dict(spent=[.01,0.,0.,0.],r=[.4,.3,.2,.1])]
        reward=model_fixture().config.utility
        with patch('brass_ai.recovery.search_value_rows.terminal_utility',return_value=np.array([1.,.2,.3,.4])):
            g,u=targets(rows,{},[.02,0.,.01,0.],reward,1.)
            np.testing.assert_allclose(g,u,atol=1e-15);np.testing.assert_allclose(g[0],[.98,.2,.29,.4])
            g,_=targets(rows,{},[.02,0.,.01,0.],reward,.5)
            np.testing.assert_allclose(g[0],[-.01,0.,0.,0.]+.5*np.array(rows[1]['r'])+.5*u[1])
            with self.assertRaises(ValueError):targets(rows,{},[0.,0.,0.,0.],reward,1.)

    def test_root_oracle_and_reward_bound_heldout_buffer(self):
        model=model_fixture();row,job,_=value_fixture(model,seat=2);validate_root(row,job)
        for field,value in [('r',[0.]*4),('mapping',[0,1,2,3]),('seat',4),('n',2),('search_value',binding(.1))]:
            bad=dict(row,**{field:value})
            with self.assertRaises(ValueError):validate_root(bad,job)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            b=ValueBuffer(tmp,row['reward_id']);b.publish(10,'train',[row]);b.publish(10,'heldout',[dict(row,split='heldout')])
            self.assertEqual(len(b.rows(17)),1);self.assertEqual(b.rows(17,stale_ages=[7]),[])
            self.assertEqual(len(b.rows(11,'heldout',stale_ages=[1])),1)
            self.assertEqual(ValueBuffer(tmp,'other',receipts=b.receipts).rows(10),[])
            restored=ValueBuffer(tmp,row['reward_id'],receipts=b.receipts);self.assertEqual(restored.rows(10),[row])
            (Path(tmp)/b.receipts[0]['file']).write_bytes(b'bad')
            with self.assertRaises(ValueError):ValueBuffer(tmp,row['reward_id'],receipts=b.receipts)

    def test_clustered_gate_requires_full_window_and_is_reproducible(self):
        rows=[dict(split='heldout',generation=g,game_id=f'{g}-{table}',phase='canal_early',u=[1.]*4,r=[.5]*4)
              for g in range(8) for table in range(4) for _ in range(64)]
        main=np.zeros((len(rows),4));delta=np.ones_like(main)*.5
        a=clustered_gate(rows,main,delta,seed=5)
        self.assertTrue(a['passed']);self.assertEqual(a,clustered_gate(rows,main,delta,seed=5))
        self.assertFalse(clustered_gate(rows[:-1],main[:-1],delta[:-1],seed=5)['full_window'])

    def test_stages_ramp_resume_reset_guard_and_manual_no_skip(self):
        machine=StageMachine();gate=dict(passed=True,mean=.1,lower=.01,upper=.2,full_window=True)
        ordinary=dict(ev=.5,kl=.01,entropy=1.,solvency=.9)
        machine.advance(0,gate,ordinary=ordinary);self.assertEqual(machine.state['stage'],'B')
        for g in range(1,21):machine.advance(g,gate,ordinary=ordinary)
        self.assertEqual(machine.state['stage'],'head_to_head');self.assertEqual(machine.state['w'],1.)
        machine=StageMachine(saved=machine.state)
        machine.choose_lambda(.8)
        with self.assertRaises(ValueError):machine.choose_lambda(1.)
        machine.advance(21,gate,ordinary=ordinary,h2h=dict(passed=True,upper=.6,games=256))
        self.assertEqual(machine.state['stage'],'C')
        for g in range(22,32):machine.advance(g,gate,ordinary=ordinary)
        self.assertEqual(machine.state['scale'],1.)
        manual=StageMachine(saved=machine.state);manual.manual('B')
        self.assertIsNone(manual.state['full_scale_generation']);self.assertIsNone(manual.state['baseline'])
        machine.advance(32,gate,ordinary=dict(ordinary,entropy=2.));self.assertEqual(machine.state['scale'],0.)
        with self.assertRaises(ValueError):machine.manual('C')
        machine.advance(33,dict(gate,passed=False,lower=-1.,upper=-.1));self.assertEqual(machine.state['w'],0.)

    def test_guarded_pass_restores_model_adams_on_rejection_and_uses_joint_cap(self):
        model=model_fixture().eval(); comp=Companion(model);row,_,_=value_fixture(model)
        value=torch.optim.Adam([p for n,p in model.named_parameters() if not n.startswith('search_value_head.')],lr=1e-5)
        before=tree_digest(model.state_dict()),tree_digest(value.state_dict()),tree_digest(comp.snapshot())
        # Unchanged held-out loss prevents acceptance, exercising real rollback.
        with patch('brass_ai.recovery.search_value_training.predict_rows',return_value=(np.zeros((1,4)),np.zeros((1,4)))):
            report=guarded_pass(comp,[row],[dict(row,split='heldout')],value,scale=.1,measure_ordinary=lambda:0.,
                cloning_loss=lambda:next(model.parameters()).sum()*0,measure_joint_kl=lambda:0.,beta=4.,rate_for_step=lambda *_:1e-5)
        self.assertFalse(report['intervals'][0]['accepted'])
        self.assertEqual(before,(tree_digest(model.state_dict()),tree_digest(value.state_dict()),tree_digest(comp.snapshot())))
        report=guarded_pass(comp,[row],[dict(row,split='heldout')],value,scale=.1,measure_ordinary=lambda:0.,
            cloning_loss=lambda:None,measure_joint_kl=lambda:.0025,beta=4.,rate_for_step=lambda *_:1e-5,kl_cap=.003)
        self.assertEqual(report['intervals'],[])

    def test_new_value_uses_shared_transport_and_rejects_wrong_binding(self):
        model=model_fixture().eval();install(model)
        meta,payload,row=wire(model);meta['binding']['search_value']=binding(0.)
        with InferenceService(config()['execution'],'cpu') as service:
            service.register(model);endpoint=service.channels(1).acquire()
            try:
                client=Client(endpoint,meta['binding']['job'],threading.Event(),binding(0.))
                reply=client.evaluate(meta,payload)
                self.assertEqual(reply['utility'],reply['search_utility']);self.assertEqual(reply['search_value'],binding(0.))
                self.assertGreater(service.metrics['shared_requests'],0)
                bad=copy.deepcopy(meta);bad['binding']['search_value']=binding(.05)
                with self.assertRaises(ValueError):client.evaluate(bad,payload)
            finally:endpoint.close()

    def test_fast_turn_has_no_policy_validation_or_labels(self):
        model=model_fixture();row,job,_=value_fixture(model);row['profile']='fast'
        cancel=threading.Event();verifier=TurnVerifier(model,[],[job],cancel)
        import time
        meta=dict(job=job['expected'],timing='search-plus-post/1.0.0',profile='fast',remaining_ns=190_000_000,
            native_post_ns=10_000_000,search_ns=700_000_000,index=1,rows=[],value_row=row)
        with patch('brass_ai.rust_backend.lab_teacher_finalization.admit_teacher_rows',return_value=([],{})) as admit:
            # A full-profile lane may hold the model lock throughout validation.
            # Value-only turns must finish without acquiring that lock.
            with verifier.lock:
                verifier.verify(meta,time.monotonic())
            self.assertEqual(admit.call_count,0)
        cancel.set()
        with self.assertRaises(ValueError):verifier.verify(dict(meta,index=2),time.monotonic())

    def test_calibration_excludes_cold_probe_uses_three_warm_per_root(self):
        from brass_ai.rust_backend.lab_teacher import calibrate
        from unittest.mock import Mock
        model=model_fixture();_,job,_=value_fixture(model)
        # Slow cold probes must not reduce N/effective K for a whole run.
        samples=[dict(elapsed_ns=n,completed=8) for n in (8000,80,160,240,16000,160,240,320)]
        service=Mock();service.device='cpu'
        with patch('brass_ai.rust_backend.lab_teacher.Worker') as worker:
            worker.return_value.collect.side_effect=samples
            result=calibrate('binary','core',[{},{}],job,service)
        self.assertEqual(result['estimate_ns'],30)
        self.assertEqual(result['root_medians_ns'],[20,30]);self.assertEqual(len(result['samples']),8)
        self.assertEqual(sum(s['warmup'] for s in result['samples']),2)

    def test_zero_start_shadow_only_and_anneal_freezes_progress(self):
        model=model_fixture(); c=adopt(config());c['teacher_phase'].update(mode='shadow',startup='shadow-until-search-informative')
        jobs,_=plan_jobs(c,dict(iteration=1),model_id=model_id(model),export_hash='a'*64,epoch=0,ready=True)
        self.assertEqual(len(jobs),4);self.assertTrue(all(j['split']=='heldout' and j['shadow_probe'] for j in jobs))
        machine=StageMachine(startup='shadow-until-search-informative');gate=dict(passed=True,mean=.1,lower=.1,upper=.2)
        machine.advance(1,gate,annealing=True);self.assertFalse(machine.state['g0_passed'])
        machine.advance(2,gate);self.assertTrue(machine.state['g0_passed']);self.assertEqual(machine.state['stage'],'A')

    def test_g1250_current_execution_and_ordinary_config_survive_adoption(self):
        import json
        p=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g1250-20261107T041141451177Z/recorded-config.json'
        original=json.loads(p.read_text());extended=adopt(original)
        for key in ('execution','update','value_phase','local_scoring_auxiliary','insolvency','reward'):
            self.assertEqual(extended[key],original[key])
        self.assertEqual(extended['value_phase']['kl_cap'],.003)
        validate(extended)

    def test_off_phase_is_inert_and_schedule_uses_saved_blend(self):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        model=model_fixture();original=model_id(model);rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            phase=SearchValuePhase(tmp,model,defaults(),digest(model.config.utility))
            self.assertIsNone(phase.companion);self.assertEqual(model_id(model),original)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertFalse(phase.after_ppg(1,mode='off')['updated'])
        c=adopt(config());c['teacher_phase']['mode']='shadow';c['teacher_phase']['search_value']['stage']='A'
        state=dict(iteration=20,search_value=dict(stage='B',w=.25,g0_passed=True))
        jobs,_=plan_jobs(c,state,model_id=original,export_hash='a'*64,epoch=0,ready=True)
        self.assertTrue(all(j['search_value']['w']==.25 for j in jobs))


if __name__=='__main__':unittest.main()
