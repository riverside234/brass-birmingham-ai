"""Bounded AZ_db teacher fixtures; no training run or game campaign."""
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.batching import model_id,collate
from brass_ai.learning.seat_descriptors import descriptor,unknown_tensor
from brass_ai.recovery import teacher_diversity as D,teacher_labels as L
from brass_ai.recovery import domino_config as C,domino_state as S
from brass_ai.recovery.domino_auxiliary import Diversity,row_key
from brass_ai.recovery.domino_search_fit import retain,validate as validate_fit
from brass_ai.recovery.teacher_phase import plan_jobs
from brass_ai.recovery.teacher_phase_config import buffer_identity,toggle_compatible
from brass_ai.recovery.teacher_term import policy_logits,TeacherTerm
from brass_ai.recovery.teacher_buffer import TeacherBuffer
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.rust_backend.lab_intrinsic import predict as intrinsic_predict
from brass_ai.rust_backend.lab_teacher_inference import predict
from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint,CONTRACT
from test_b025 import model_fixture,example,config as legacy_config

LOG='results/joint7-v11-2-grouped-lr-critic-g1330-20261031T024940047516Z'


def config(model=None):
    value=json.loads((SOURCE_ROOT/LOG/'state.json').read_text())['signature']['config']
    if model is not None:  # Small authored models can retain a historical task head.
        value['reward']={k:v for k,v in model.config.utility.items() if k not in ('version','insolvency')}
        value['insolvency']=copy.deepcopy(model.config.utility['insolvency'])
    return D.adopt(value)


def fitted(model):
    c=C.defaults();d=Diversity(model,c,regime='fixture-task-regime')
    d.state.update(centroids=[np.full(448,z*.001).tolist() for z in range(7)],quality=[1.]*7,
                   seat_games=[40]*7,distinct_games=[20]*7)
    rows=[];data=[]
    for z in range(7):
        a=dict(strategy_id=z,receipt='context-'+str(z),training_role='learner')
        rows.append(dict(seat=0,view=dict(context=dict(revision=z,era='CanalEra',round=2)),_domino=a))
        data.append((None,None,('fixture-game-'+str(z),0)))
    ds=SimpleNamespace(rows=rows,data=data)
    d.train_keys={row_key(ds,i) for i in range(7)}
    d.target_receipts={row_key(ds,i):r['_domino']['receipt'] for i,r in enumerate(rows)}
    d.search_normalization_proven=True  # Authored target convention; no claim of complete-game evidence.
    retain(d,model,ds,set(range(7)),1)
    # Current committed centroids deliberately differ from the fitted target.
    d.state['generation']=1;d.state['centroids']=[np.full(448,z*.002).tolist() for z in range(7)]
    mid=model_id(model)
    fps=SimpleNamespace(get=lambda m,z:dict(contract='policy-style-fingerprint/1.0.0',values=[.1]*28,
        probe_hash='p'*64,context_identity=dict(model_id=m,strategy_id=z)))
    bundle,ready=D.freeze(model,d,fps,export_hash='a'*64,generation=1)
    assert ready['ready']
    return d,ds,bundle


def conditional(model):
    d,ds,bundle=fitted(model);c=config(model);c['teacher_phase']['mode']='shadow'
    jobs,_=plan_jobs(c,dict(iteration=1),model_id=model_id(model),export_hash='a'*64,epoch=0,
        ready=True,native_source=source_fingerprint(),evaluator=bundle)
    j=jobs[0];j['readiness']=dict(ready=True,reward_id=j['expected']['reward_id'])
    seat=next(s for s,a in enumerate(j['expected']['seat_contexts']) if a['strategy_id']>0)
    value,_=example(model,generation=1,seat=seat,job=j)
    row=value.row;label=row['teacher'];a=j['expected']['seat_contexts'][seat];z=a['strategy_id']
    row.update(style=a['style'],intent=a['intent'])
    encoded=PreparedEncoder().encode(row['view'],row['choices'],a['style'],a['intent'])
    row['teacher'].update(source=D.LABEL,algorithm=D.ALGORITHM,session=D.SESSION,
        base_algorithm=L.ALGORITHM,base_session=L.SESSION,seat_context=a,strategy_id=z,root_owner=seat,
        coefficient=bundle['coefficients'][z],opponent_model='root-latent-self-play',channel_counts=label['counts'],
        normalization=D.NORMALIZATION,intrinsic_prefix=0.,
        channel_watermark=label['completed_trajectories'],task_node_value=.2,intrinsic_node_value=-.4,
        task_mean_returns=[.8,0.,0.],intrinsic_mean_returns=[-.6,0.,0.])
    coef=label['coefficient'];label['node_value']=coef*.2+(1-coef)*(-.4)
    label['mean_returns']=[coef*.8+(1-coef)*(-.6),0.,0.]
    value=L.LabelRow(row,encoded);label['logits']=policy_logits(model,[value])[0].tolist()
    target,mixed=L.improved_policy(label['logits'],label['counts'],label['mean_returns'],label['node_value'])
    label.update(target=target.tolist(),v_mix=mixed)
    return value,j,bundle,d


class DiversityTeacher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.model=model_fixture().eval()

    def test_current_g1330_adoption_changes_teacher_only(self):
        state=json.loads((SOURCE_ROOT/LOG/'state.json').read_text());before=state['signature']['config'];after=config()
        self.assertEqual(state['iteration'],1330);self.assertIsNone(after['search_wall_fraction'])
        self.assertEqual(after['teacher_phase']['contract'],D.CONTRACT);self.assertEqual(after['teacher_phase']['mode'],'off')
        for k in before:
            if k!='teacher_value':self.assertEqual(after[k],before[k])
        self.assertEqual(D.adopt(after),after)
        self.assertFalse(toggle_compatible(before,after))
        toggle=copy.deepcopy(after);toggle['teacher_phase']['mode']='shadow';self.assertTrue(toggle_compatible(after,toggle))
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        target=dict(state['signature'],implementation=implementation_id())
        self.assertTrue(check_signature(state,target,True))
        self.assertTrue(check_signature(state,dict(target,config=after),True))

    def test_assignments_are_separate_iid_and_reproducible(self):
        import random
        before=random.getstate();ids=[]
        for i in range(128):
            a=D.assignments(str(i),run_seed=42,model_id='a'*64,export_hash='b'*64)
            self.assertEqual(a,D.assignments(str(i),run_seed=42,model_id='a'*64,export_hash='b'*64))
            for seat,r in enumerate(a):D.check_context(r);self.assertEqual(r['seat'],seat);ids.append(r['strategy_id'])
        self.assertEqual(set(ids),set(range(7)));self.assertEqual(before,random.getstate())
        self.assertTrue(any(ids[i]==ids[i+1] for i in range(len(ids)-1)))

    def test_fit_missing_partial_rollback_and_old_direction(self):
        d=Diversity(self.model,C.defaults(),regime='fixture')
        d.state.update(centroids=[[0.]*448 for _ in range(7)],quality=[1.]*7,seat_games=[40]*7,distinct_games=[20]*7)
        b,r=D.freeze(self.model,d,None,export_hash='a'*64,generation=0)
        self.assertIsNone(b);self.assertEqual(r['reason'],'missing_retained_intrinsic_fit')
        d,ds,b=fitted(self.model);old=copy.deepcopy(d.search_fit)
        retain(d,self.model,ds,set(),0);self.assertEqual(old,d.search_fit)
        self.assertEqual(b['directions'],old['directions']);self.assertNotEqual(b['directions'],b['current_directions'])
        snapshot=d.snapshot();d.search_fit=None;d.restore(snapshot);self.assertEqual(d.search_fit,old)
        bad=copy.deepcopy(old);bad['directions'][1][0]+=1
        with self.assertRaises(ValueError):validate_fit(bad)
        d.search_fit=copy.deepcopy(old);d.search_fit['support'].pop('6');d.search_fit['identity']=digest({k:v for k,v in d.search_fit.items() if k!='identity'})
        _,r=D.freeze(self.model,d,None,export_hash='a'*64,generation=1)
        self.assertEqual(r['missing_strategies'],[6])

    def test_conditional_label_admission_and_tampering(self):
        row,j,b,d=conditional(self.model)
        args=dict(model=self.model,expected=j['expected'],history=[dict(generation=1,model_id=model_id(self.model))],
                  admitted_sources={source_fingerprint()})
        admitted,dropped=L.admit_rows([row.row],**args);self.assertEqual(len(admitted),1)
        for key,value in [('bundle_id','wrong'),('strategy_id',0),('task_node_value',1.),('channel_counts',[2,0,0]),('normalization','wrong'),('intrinsic_prefix',float('nan')),('root_owner',(row.row['seat']+1)%4)]:
            corrupt=copy.deepcopy(row.row);corrupt['teacher'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):L.admit_rows([corrupt],**args)
        corrupt=copy.deepcopy(row.row);corrupt['style']=[0.]*4;corrupt['intent']=[0.]*8
        with self.assertRaises(ValueError):L.admit_rows([corrupt],**args)

    def test_retained_fit_on_real_cached_rows_is_batched_and_readonly(self):
        from brass_ai.recovery import updates,ppg_replay
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.domino_auxiliary import PPGCompanions
        from brass_ai.recovery.retention import _same_payload
        from test_domino import fixture,collect
        from test_domino_audit import ready
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            model,_,c=fixture();phase=TrainingPhases(model,c,root=SOURCE_ROOT,run=Path(tmp))
            d=phase.diversity=Diversity(model,c['diversity'],regime='retained-cache-fixture');ready(d)
            _,rows=collect(model,phase,1,games=42)
            phase.prepare(updates.Dataset(rows,'cpu'));dense=phase.critic_dataset
            phase.buffer.stage(dense,phase.train_ids,phase.heldout_ids,[],
                phase.collection_source,model,phase.split['selection'])
            with phase.buffer.datasets('cpu',policy=False) as (cached,_,_):
                self.assertIsInstance(cached,ppg_replay.CriticDataset)
                ids=[i for i,(key,_,_,_) in enumerate(cached.diversity_metadata(range(len(cached.data))))
                     if key in d.train_keys][:8]
                self.assertEqual(len(ids),8)
                wrapper=PPGCompanions(None,d,phase.optimizer,.5)
                for part in (ids[:4],ids[2:]):
                    wrapper.begin('ppg');wrapper.batch(cached,part);ix,inputs,_=cached.value_batch(part,model)
                    _,loss=wrapper.forward(model,inputs,cached,ix);loss.mean().backward();wrapper.step('ppg')
                    if part==ids[:4]:selected=wrapper.snapshot()
                wrapper.restore(selected)
                keys={cached.diversity_metadata([i])[0][0] for i in wrapper.retained_rows}
                before=copy.deepcopy((model.state_dict(),phase.optimizer.state_dict(),d.head.state_dict(),
                    {p:o.state_dict() for p,o in d.optimizers.items()},d.state,torch.get_rng_state()))
                # No row reconstruction or device column gather is needed for sealed fit metadata.
                with (patch.object(ppg_replay.Rows,'field',side_effect=AssertionError('hydrated row')),
                      patch.object(ppg_replay.Column,'__getitem__',side_effect=AssertionError('device gather'))):
                    retain(d,model,cached,wrapper.retained_rows,wrapper.retained_steps)
                self.assertTrue(_same_payload(before,(model.state_dict(),phase.optimizer.state_dict(),d.head.state_dict(),
                    {p:o.state_dict() for p,o in d.optimizers.items()},d.state,torch.get_rng_state())))
                receipt=copy.deepcopy(d.search_fit);validate_fit(receipt)
                self.assertEqual(receipt['retained_steps'],1)
                self.assertEqual(sum(v['rows'] for v in receipt['support'].values()),4)
                for support in receipt['support'].values():
                    for descriptor in support['input_descriptors']:
                        self.assertEqual(len(descriptor),dense.descriptors.shape[-1])
                        self.assertTrue(all(isinstance(x,float) for x in descriptor))
            dense_ids={i for i in range(len(dense.data)) if row_key(dense,i) in keys}
            retain(d,model,dense,dense_ids,wrapper.retained_steps)
            self.assertEqual(receipt,d.search_fit)

    def test_conditional_cache_roundtrip_and_dynamic_bundle_not_static_identity(self):
        row,j,b,d=conditional(self.model);c=config();identity=buffer_identity(c)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            cache=TeacherBuffer(SOURCE_ROOT,Path(tmp),identity)
            receipt=cache.publish(1,'train',[row],model_id=model_id(self.model),reward_id=row.label['reward_id'],evaluators={b['identity']:b})
            self.assertEqual(receipt['contract'],'teacher-label-buffer/2.0.0')
            reloaded=TeacherBuffer(SOURCE_ROOT,Path(tmp),identity,receipts=[receipt]).rows(2)[0][1]
            np.testing.assert_array_equal(reloaded.encoded.state,row.encoded.state)
            self.assertEqual(reloaded.label['bundle_id'],b['identity']);self.assertEqual(buffer_identity(c),identity)

    def test_readonly_intrinsic_forward_reuses_trunk_and_keeps_task_unknown(self):
        row,j,b,d=conditional(self.model);d.head[-1].bias.data.fill_(.2)
        encoded=[row.encoded,row.encoded];z=row.label['strategy_id']
        registry={b['identity']:(b,d.head)};specs=[dict(bundle_id=b['identity'],strategy_id=z)]*2
        for limit in (1,128):
            with torch.inference_mode(),patch.object(self.model,'encode_state',wraps=self.model.encode_state) as trunk:
                arrays=intrinsic_predict(self.model,encoded,'cpu',limit,None,specs,registry)
                self.assertEqual(trunk.call_count,1)
            with torch.inference_mode():reference=predict(self.model,encoded,'cpu',128,None)
            for a,x in zip(arrays[:6],reference):np.testing.assert_allclose(a,x,atol=2e-5,rtol=1e-5)
            np.testing.assert_allclose(arrays[6],.2,atol=1e-6)
        self.assertTrue(all(p.grad is None for p in d.head.parameters()))

    def test_policy_only_gradient_and_beta_archive(self):
        from brass_ai.recovery.value_preservation import TrunkGradients
        model=copy.deepcopy(self.model);row,j,b,d=conditional(model);c=config()['teacher_phase'];c['mode']='on'
        term=TeacherTerm(model,c,[(0,row)],seed=1,generation=1,state=dict(archive=[dict(beta=7.)]))
        term.apply(model,TrunkGradients(model),1)
        self.assertEqual(term.beta_used,1.)
        for module in (model.outcomes,model.utility_head,model.descriptor_encoder,d.head):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertTrue(any(p.grad is not None for n,p in model.named_parameters() if n.startswith('state_encoder')))

    def test_intrinsic_one_and_772_candidates_heterogeneous_contexts(self):
        from brass_ai.learning.features import Encoded
        from brass_ai.learning.state_layout import state_layout
        row,j,b,d=conditional(self.model);e=row.encoded
        indices=list(next(g.indices for g in state_layout().groups if g.name=='policy_context'))
        encoded=[]
        for z,n in ((1,1),(6,772)):
            state=e.state.copy();state[indices]=C.mapping()[z]['style']+C.mapping()[z]['intent']
            actions=np.tile(e.actions,(n//len(e.actions)+1,1))[:n]
            encoded.append(Encoded(state,actions,e.stable_to_relative.copy(),tuple('candidate-'+str(i) for i in range(n))))
        registry={b['identity']:(b,d.head)}
        specs=[dict(bundle_id=b['identity'],strategy_id=z) for z in (1,6)]
        for batch in (1,2):
            with torch.inference_mode():
                actual=intrinsic_predict(self.model,encoded[:batch],'cpu',128,None,specs[:batch],registry)
                expected=predict(self.model,encoded[:batch],'cpu',1000,None)
            for x,y in zip(actual[:6],expected):np.testing.assert_allclose(x,y,atol=2e-5,rtol=1e-5)
        with torch.inference_mode(),self.assertRaises(ValueError):
            intrinsic_predict(self.model,encoded[:1],'cpu',128,None,[dict(bundle_id='wrong',strategy_id=1)],registry)

    def test_diversity_readiness_survives_finish_and_objective_drift_is_diagnostic(self):
        from brass_ai.recovery.teacher_phase import TeacherPhase
        row,j,b,d=conditional(self.model);c=config();c['teacher_phase']['mode']='shadow'
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            phase=TeacherPhase(SOURCE_ROOT,Path(tmp),c,dict(iteration=1,history=[]),self.model)
            phase.receipt=dict(ready=True,reward_id=phase.reward_id)
            phase.diversity_readiness=dict(ready=False,reason='missing_retained_intrinsic_fit')
            self.assertFalse(phase.finish(self.model,{})['ready'])
        changed=copy.deepcopy(b);changed['coefficients'][row.label['strategy_id']]+=.1
        report=D.buffer_diagnostics([(1,row)],[dict(evaluators={b['identity']:b})],changed)
        self.assertAlmostEqual(report['1'][str(row.label['strategy_id'])]['coefficient_change']['median'],.1)
        self.assertEqual(row.label['coefficient'],b['coefficients'][row.label['strategy_id']])

    def test_retired_shards_survive_rollback_window(self):
        from brass_ai.step5.league import write_json
        row,j,b,d=conditional(self.model)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);cache=TeacherBuffer(SOURCE_ROOT,run,buffer_identity(config()))
            receipt=cache.publish(1,'train',[row],model_id=row.label['model_id'],reward_id=row.label['reward_id'],evaluators={b['identity']:b})
            cache.window(5)
            state=dict(iteration=5,history=[dict(generation=g) for g in range(2,6)],teacher_phase=dict(buffer=cache.payload()))
            write_json(run/'state.json',state);cache.committed(5)
            self.assertTrue((run/receipt['path']).is_file())
            state.update(iteration=10,history=[dict(generation=g) for g in range(7,11)],teacher_phase=dict(buffer=cache.payload()))
            write_json(run/'state.json',state);cache.committed(10)
            self.assertFalse((run/receipt['path']).exists())

    def test_b03_restores_live_domino_and_both_adams_without_fit(self):
        from test_b03 import continuation_fixture
        from brass_ai.bank import sha
        from brass_ai.learning.checkpoint import load_checkpoint
        from brass_ai.rust_backend import lab_continuation as continuation
        from brass_ai.recovery.retention import run_lock,_same_payload
        from brass_ai.step5.league import write_json
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            from test_domino import collect as collect_domino
            def contexts(model,phase,generation,**kwargs):
                phase.config['diversity']=C.defaults()
                phase.diversity=Diversity(model,phase.config['diversity'],regime=S.regime(phase.config))
                phase.config['update']['fixed_entropy']=copy.deepcopy(config()['update']['fixed_entropy'])
                try:return collect_domino(model,phase,generation,games=18,branches=True)
                finally:phase.config['update'].pop('fixed_entropy')
            base=Path(tmp)
            with patch('test_ppg_eight.collect',side_effect=contexts):run,state=continuation_fixture(base)
            path=SOURCE_ROOT/state['history'][-1]['path'];model,raw=load_checkpoint(path)
            c=state['signature']['config'];c['diversity']=C.defaults()
            d=Diversity(model,c['diversity'],regime=S.regime(c))
            d.state.update(generation=1,logits=[.1]*6,adam=dict(step=1,m=[.01]*6,v=[.001]*6),
                centroids=[[z*.001]*448 for z in range(7)],quality=[1.]*7,seat_games=[40]*7,distinct_games=[20]*7)
            for optimizer in d.optimizers.values():
                next(d.head.parameters()).grad=torch.ones_like(next(d.head.parameters()));optimizer.step();optimizer.zero_grad(set_to_none=True)
            d.pending=copy.deepcopy(d.state)
            raw['training']['domino']=copy.deepcopy(d.payload());raw['training']['signature']=digest(state['signature'])
            torch.save(raw,path);state['history'][-1]['sha256']=sha(path)
            state['domino_state_digest']=digest(d.state);write_json(run/'state.json',state)
            with run_lock(run):
                receipt=continuation.preflight(SOURCE_ROOT,run,generation=1,checkpoint_sha256=sha(path))
            fork=continuation.create_fork(SOURCE_ROOT,base/'fork',receipt)
            restored=continuation.restore(fork)
            self.assertTrue(restored.report['domino_restored']);self.assertIs(restored.phases.diversity,restored.diversity)
            self.assertEqual(restored.report['diversity_teacher_readiness'],'missing_retained_intrinsic_fit')
            self.assertTrue(_same_payload(restored.diversity.head.state_dict(),raw['training']['domino']['head']))
            for name,opt in restored.diversity.optimizers.items():
                self.assertTrue(_same_payload(opt.state_dict(),raw['training']['domino']['optimizers'][name]))
            self.assertEqual(restored.diversity.state,d.state);self.assertEqual(sha(path),receipt['checkpoint']['sha256'])

    @unittest.skipUnless(os.environ.get('BRASS_DOMINO_NATIVE'),'native descriptor binary not supplied')
    def test_native_descriptor_exact_in_four_phases(self):
        from brass_ai.recovery.domino_features import event,manifest
        # Use a validated real native view rather than an authored sparse board.
        state=json.loads((SOURCE_ROOT/'native/brass-ai-core/tests/data/initial-state.json').read_text())
        for era,round in [('CanalEra',1),('CanalEra',8),('RailEra',1),('RailEra',8)]:
            for kind in ('build','develop','sell','loan','network','scout','pass'):
                choice=dict(id='fixture',kind=kind)
                payload=json.dumps(dict(contract=CONTRACT,op='descriptor',state=state,era=era,round=round,choice=choice,descriptor=manifest())).encode()
                result=subprocess.run([os.environ['BRASS_DOMINO_NATIVE']],input=struct.pack('<I',len(payload))+payload,capture_output=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)
                data=result.stdout;size=struct.unpack('<I',data[4:8])[0];actual=json.loads(data[8:8+size])
                expected=event(dict(view=actual['view'],choices=[choice],chosen=0))
                self.assertEqual(actual['phase'],expected['phase']);np.testing.assert_allclose(actual['x'],expected['x'],rtol=0,atol=1e-16)

    @unittest.skipUnless(os.environ.get('BRASS_B04_CORE'),'native objective build not supplied')
    def test_native_intrinsic_probe_and_real_service(self):
        from brass_ai.rust_backend import lab_teacher as T
        from brass_ai.rust_backend.lab_intrinsic import register
        from brass_ai.recovery.inference import InferenceService
        from brass_ai.rust_backend.lab_admission import admit_teacher_rows
        d,ds,b=fitted(self.model);c=config(self.model);c['teacher_phase']['mode']='shadow'
        jobs,_=plan_jobs(c,dict(iteration=1),model_id=model_id(self.model),export_hash='a'*64,epoch=0,
            ready=True,native_source=source_fingerprint(),evaluator=b)
        state=json.loads((SOURCE_ROOT/'native/brass-ai-core/tests/data/initial-state.json').read_text())
        seat=state['context']['actor'];j=next(j for j in jobs if j['expected']['seat_contexts'][seat]['strategy_id']>0)
        j['readiness']=dict(ready=True,reward_id=j['expected']['reward_id'])
        with InferenceService(legacy_config()['execution'],'cpu') as service:
            service.register(self.model);register(service,self.model,d.head,b)
            calibration=T.calibrate(os.environ['BRASS_DOMINO_NATIVE'],os.environ['BRASS_B04_CORE'],
                [dict(state=state,rail_keys=list(range(64)))],j,service)
        self.assertEqual(len(calibration['samples']),2)
        self.assertTrue(any(s['work']['continuation_steps']>0 for s in calibration['samples']))
        for s in calibration['samples']:
            row=s['probe'];self.assertTrue(row['diagnostic']);self.assertGreater(s['intrinsic_requests'],0)
            # K1 is an explicitly marked diagnostic; verify full K16 labels via the production math oracle.
            if not s['continuation_probe']:
                rows,_=L.admit_rows([row],model=self.model,expected=j['expected'],
                    history=[dict(generation=1,model_id=model_id(self.model))],admitted_sources={source_fingerprint()})
                self.assertEqual(len(rows),1)
                with self.assertRaisesRegex(ValueError,'probes'):
                    admit_teacher_rows([row],job=j,model=self.model,history=[dict(generation=1,model_id=model_id(self.model))])

    @unittest.skipUnless(os.environ.get('BRASS_DOMINO_NATIVE'),'native descriptor binary not supplied')
    def test_native_descriptor_populated_public_state_all_seats(self):
        from brass_ai.recovery.domino_features import event,manifest
        state=json.loads((SOURCE_ROOT/'native/brass-ai-core/tests/data/initial-state.json').read_text())
        choice=dict(id='descriptor-only',kind='sell')
        def native(**value):
            raw=json.dumps(dict(contract=CONTRACT,op='descriptor',choice=choice,descriptor=manifest(),**value)).encode()
            r=subprocess.run([os.environ['BRASS_DOMINO_NATIVE']],input=struct.pack('<I',len(raw))+raw,capture_output=True,timeout=10)
            self.assertEqual(r.returncode,0,r.stderr)
            n=struct.unpack('<I',r.stdout[4:8])[0];return json.loads(r.stdout[8:8+n])
        base=native(state=state)['view']
        for seat in range(4):
            view=copy.deepcopy(base);view['seat']=seat;view['context']['actor']=seat
            player=view['players'][seat];player.update(cash=140,income=-10,spent=125)
            # Public-only authored descriptor fixture, not a rule-valid search root.
            for i,mat in enumerate(player['mat']):
                tile=copy.deepcopy(mat['tiles'][-1]);tile.update(seat=seat,flipped=i%2==0)
                view['slots'][i]['tile']=tile
                mat['tiles']=[] if i==0 else mat['tiles'][1:]
            for i,link in enumerate(view['links'][:20]):link['owner']=seat if i<18 else (seat+1)%4
            actual=native(view=view);expected=event(dict(view=view,choices=[choice],chosen=0))
            np.testing.assert_allclose(actual['x'],expected['x'],rtol=0,atol=1e-16)


if __name__=='__main__':unittest.main()
