"""Lossless packed replay, sixteen-age evidence, migration and restart."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_packing as K,dcpg_config as D,ppg_config as P,ppg_replay as R
from brass_ai.recovery import ppg_auxiliary as A,ppg_epochs,gae,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.value_buffer import row_dtype
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,stage_only


def config():
    model,opt,c=setup();return model,opt,D.sixteen_options(D.beta_options(D.configure(c)))


class Packing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_roundtrip_all_columns_duplicate_gathers_and_signed_zero(self):
        model,_,cfg=config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            result=stage_only(model,p,1)
            self.assertEqual(result['packed_generations'],1)
            self.assertLess(result['packing']['packed_array_bytes'],result['packing']['dense_array_bytes'])
            r=p.buffer.pending[-1];self.assertEqual(set(r['files']),set(K.EXTENSIONS))
            p.buffer.verify(r)
            with p.buffer.datasets('cpu') as (critic,policy,receipts):
                original=p.critic_dataset
                for i in range(len(critic.rows)):
                    row,target,identity,encoded=original.data[i]
                    got=critic.data[i]
                    self.assertEqual(got[2],identity)
                    for a,b in ((got[3].state,encoded.state),(got[3].actions,encoded.actions[:1]),
                                (got[0]['_seat_descriptors'],row['_seat_descriptors'])):
                        self.assertEqual(np.asarray(a,dtype='<f4').tobytes(),np.asarray(b,dtype='<f4').tobytes())
                    for name,key in (('utility','_reward_return'),('remaining','_reward_remaining')):
                        self.assertEqual(critic.data.column([i],name).tobytes(),np.asarray([row[key]],dtype='<f4').tobytes())
                ix=[4,1,4,0];a=critic.batch(ix)[1];b=original.value_batch(ix,model)[1]
                for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
                for i,e in enumerate(r['policy_entries']):
                    self.assertEqual(policy.encoded[i].actions.tobytes(),original.encoded[e['row']].actions.tobytes())
                expected=[i for i in p.heldout_ids if original.roles[i]=='learner' and '/branch/' not in original.data[i][2][0]]
                self.assertEqual(A.layout(critic,receipts,cfg['value_phase'],1.)[2],expected)
            self.assertFalse(p.buffer.path(1,'critic.npy').exists())
            # Arbitrary FP32 bit patterns, signed zero and all three references
            # must survive storage, including absent references and duplicates.
            raw=np.zeros(8,dtype=row_dtype());raw['mapping']=np.arange(4)
            raw['state'][:,0]=np.arange(8)/7.;raw['state'][:,1]=[0,1]*4
            raw['state'][:,2]=np.float32(-0.);raw['state'][::2,3]=np.float32(-0.)
            actions=np.zeros((9,327),dtype='<f4');fields=list(K.entity_layout()['action_fields'].values())
            for f in fields:actions[1:,f['start']]=1.
            actions[0,fields[0]['start']]=-0.;actions[:,320]=np.arange(9)/7.
            packed=K.pack(lambda ext:run/ext,raw,actions,1)
            view=K.CriticArray(lambda ext:run/ext,dict(rows=8,packing=packed))
            acts=K.ActionArray(run/'policy-actions.npy',packed)
            try:
                for name in raw.dtype.names:
                    self.assertEqual(view.column(name,slice(None)).tobytes(),raw[name].tobytes(),name)
                self.assertEqual(acts[:].tobytes(),actions.tobytes())
                self.assertEqual(acts[[7,0,7]].tobytes(),actions[[7,0,7]].tobytes())
            finally:view.close();acts.close()

    def test_game_metadata_mismatch_and_corruption_rejected(self):
        raw=np.zeros(2,dtype=row_dtype());raw['mapping']=np.arange(4)
        raw['descriptors'][1,0,0]=1.
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            with self.assertRaisesRegex(ValueError,'metadata changed'):
                K.pack(lambda ext:Path(tmp)/ext,raw,np.zeros((1,327),dtype='<f4'),1)
            model,_,cfg=config();run=Path(tmp)/'run';p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            stage_only(model,p,1);r=p.buffer.pending[-1]
            p.buffer.path(1,'state-bits.npy').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'corrupt'):p.buffer.verify(r)


class Schedule(unittest.TestCase):
    def test_g620_source_config_helper_and_unrelated_change_rejection(self):
        from brass_ai.recovery.config import validate_config,latest_model_config,insolvency_040_cap100_config
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        exported=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0620-20261023T133150568578Z'
        state=read(exported/'state.json');old=state['signature']['config'];new=insolvency_040_cap100_config(D.sixteen_options(old))
        validate_config(new)
        self.assertEqual(latest_model_config(old,old['imitation_checkpoint'])['value_phase'],D.fast_options(new)['value_phase'])
        target=dict(state['signature'],config=new,implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        with self.assertRaises(ValueError):check_signature(state,target)
        bad=copy.deepcopy(target);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        spec=importlib.util.spec_from_file_location('sixteen_helper',SOURCE_ROOT/'scripts/configure-ppg-sixteen.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0620.pt';checkpoint.write_bytes(b'explicit helper hash fixture')
            s=copy.deepcopy(state);s['history'][-1]=dict(s['history'][-1],path='artifacts/run/generation-0620.pt',sha256=sha(checkpoint))
            write_json(run/'state.json',s);before={p.name:sha(p) for p in run.iterdir()}
            out='configs/step5/test-sixteen.json'
            r=helper.generate(root,'artifacts/run',out,620)
            self.assertEqual(r['first_phase'],636);self.assertEqual(read(root/out),new)
            self.assertEqual(r['insolvency'],dict(event_cost=.04,max_cost=.10))
            self.assertTrue(r['retire_old_reward_cache_on_startup'])
            self.assertEqual(helper.generate(root,'artifacts/run',out,620),r)
            self.assertEqual(before,{p.name:sha(p) for p in run.iterdir()})
            with self.assertRaises(ValueError):helper.generate(root,'artifacts/run',out,619)

    def test_exact_config_and_clock_boundaries(self):
        _,_,c=config();v=c['value_phase'];P.validate(v)
        old=D.initial(600)
        for g in range(601,621):old=P.advance(old,g,P.due(old))
        s=D.migrate_sixteen_clock(old,620);self.assertEqual(s['phases'],old['phases'])
        seen=[]
        for g in range(621,654):
            self.assertEqual(P.due(s),P.due(json.loads(json.dumps(s))))
            due=P.due(s)
            if due:seen.append(g)
            s=P.advance(s,g,due)
        self.assertEqual(seen,[636,652])
        self.assertEqual(P.next_phase(P.initial(0,v)),16)
        self.assertIsNone(v['kl_cap']);self.assertEqual(v['beta_control'],D.FIXED_BETA_CONTROL)
        from brass_ai.recovery.ppg_selection import MINIMUM_DEFAULT,RELAXED_DEFAULT
        self.assertEqual(v['selection'],MINIMUM_DEFAULT)
        for key,value in (('buffer_iterations',8),('validation_iterations',8),('phase_interval',8),('cache_format','unknown'),
                          ('kl_cap',.002),('kl_control','hard-cap'),('beta_control',D.BETA_CONTROL),('selection',RELAXED_DEFAULT)):
            with self.assertRaises(ValueError):P.validate(dict(v,**{key:value}))
        original=D.beta_options(D.configure(setup()[2]));changed=D.sixteen_options(original)
        self.assertEqual({k for k in changed if changed[k]!=original[k]},{'value_phase'})
        self.assertEqual(D.sixteen_options(changed),changed)
        self.assertEqual(D.beta_options(changed),changed)


class PenaltyOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_fixed_checkpoint_beta_does_not_respond_to_KL(self):
        c=config()[2]['value_phase']
        for beta in (1.,4.,32.):
            for kl in (0.,.0005,.001,.002,.01,1.):
                r=dict(beta_clone_final=beta,scheduled=True,optimizer_steps=16,
                    kl_guard_tripped=False,beta_retries=0,probe_kl=kl,age0_probe_kl=kl)
                actual,receipt=D.beta_after_phase(r,c)
                self.assertEqual(actual,beta);self.assertEqual(receipt['reason'],'fixed_current_beta')
                self.assertIsNone(receipt['hard_cap'])
        with self.assertRaises(ValueError):D.beta_after_phase(dict(beta_clone_final=float('nan')),c)

    def fit(self,selected_step,neutral=False):
        model,ppo_opt,cfg=config();cfg['value_phase']['batch_size']=16
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            for g in range(1,17):
                stage_only(model,p,g)
                if g<16:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            adam_before=copy.deepcopy(ppo_opt.state_dict());saved={};checks=[]
            real_step=p.optimizer.step;real_probe=A.Reference.probes
            def steps():return max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            def step(*args,**kwargs):
                result=real_step(*args,**kwargs)
                if steps()==selected_step:saved.update(weights=copy.deepcopy(model.state_dict()),adam=copy.deepcopy(p.optimizer.state_dict()))
                return result
            def losses(net,ds,ids,update):
                n=steps();v=1. if n==0 else 1.1 if n>selected_step else 1.-n*.001
                if neutral and n==selected_step+16:
                    games=sorted({ds.data[i][2][0] for i in ids})
                    shift={g:(.02 if j%2 else -.02) for j,g in enumerate(games)}
                    return {i:1.-selected_step*.001+.0001+shift[ds.data[i][2][0]] for i in ids}
                return {i:v for i in ids}
            def probes(ref,net):
                result=real_probe(ref,net);checks.append(steps())
                # Both overall and current-age movement exceed the old cap.
                # Retention must depend only on held-out loss, never this value.
                result.update(overall=.004,age0=.006);return result
            with patch.object(p.optimizer,'step',step),patch.object(A,'losses',losses),patch.object(A.Reference,'probes',probes):
                r=A.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                    cfg['value_phase'],cfg['update'],beta=4.,snapshot_weight=1.,dropout=None,run_phase=True,
                    ppo_work=p.ppo_work,epoch_cap=5,ppo_peak=D.rate_receipt(6e-6,16,48))
            self.assertEqual(r['optimizer_steps'],selected_step);self.assertEqual(r['attempted_steps'],selected_step+(32 if neutral else 16))
            self.assertEqual(checks,[32,64,selected_step] if neutral else [32,selected_step])
            self.assertEqual(r['early_stop_reason'],'validation_patience')
            if neutral:
                interval=next(i for i in r['interval_reports'] if i['budget_step']==selected_step+16)
                self.assertEqual(interval['outcome'],'neutral');self.assertFalse(interval['retained_as_minimum_loss'])
            self.assertEqual(r['beta_mode'],'fixed-checkpoint');self.assertEqual(r['beta_clone_final'],4.)
            self.assertEqual(r['beta_retries'],0);self.assertEqual(r['rejected_unverified_selections'],0)
            self.assertFalse(r['kl_guard_tripped']);self.assertFalse(r['policy_guard_enabled']);self.assertIsNone(r['kl_cap'])
            self.assertGreater(r['clone_presentations'],0);self.assertEqual(r['probe_kl'],.004);self.assertEqual(r['age0_probe_kl'],.006)
            self.assertTrue(_same_payload(saved['weights'],model.state_dict()))
            self.assertTrue(_same_payload(saved['adam'],p.optimizer.state_dict()))
            self.assertTrue(_same_payload(adam_before,ppo_opt.state_dict()))
            self.assertEqual(D.beta_after_phase(r,cfg['value_phase'])[0],4.)

    def test_high_KL_does_not_stop_and_heldout_rejection_still_restores_both_states(self):self.fit(32)

    def test_loss_selected_16_step_state_needs_no_KL_permission(self):self.fit(16)

    def test_neutral_interval_can_continue_but_cannot_replace_minimum(self):self.fit(32,neutral=True)


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_sixteen_ages_holdout_full_policy_and_post_commit_expiry(self):
        model,_,cfg=config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            for g in range(1,18):
                stage_only(model,p,g)
                if g==16:
                    with p.buffer.datasets('cpu') as (critic,policy,receipts):
                        train,held,validation,ages,support=A.layout(critic,receipts,cfg['value_phase'],1.)
                        self.assertEqual(set(ages[i] for i in validation),set(range(16)))
                        self.assertEqual(len(A.by_game(critic,{i:0. for i in validation})),16*6)
                        self.assertTrue(all('/branch/' not in critic.data[i][2][0] and critic.roles[i]=='learner' for i in validation))
                        order,_,_=A.epoch_order(train,cfg['value_phase'])
                        self.assertEqual(set(order),{i for v in train.values() for i in v})
                        self.assertEqual(len(order),len(set(order)))
                        ref=A.Reference(model,policy,receipts,cfg['value_phase'],cfg['update'])
                        self.assertEqual(len(ref.clone_ids),16*24);self.assertEqual(len(ref.probe_ids),24+15*12)
                        self.assertFalse(set(ref.clone_ids)&set(ref.probe_ids))
                        self.assertLess(ref.probes(model)['overall'],1e-6)
                if g==17:
                    self.assertTrue(p.buffer.path(1,'meta.npy').exists())
                    with self.assertRaisesRegex(ValueError,'commit'):p.buffer.committed(g)
                write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            self.assertFalse(p.buffer.path(1,'meta.npy').exists())
            q=R.Buffer(SOURCE_ROOT,run,cfg,generation=17,receipts=p.buffer.receipts,origin=0)
            self.assertEqual(len(q.receipts),16)

    def test_packed_local_scoring_labels_and_cleanup_ownership(self):
        from test_local_scoring import annotate
        from test_ppg_eight import collect
        from brass_ai.recovery import updates,value_buffer
        from brass_ai.recovery.local_scoring_auxiliary import DEFAULT
        model,_,cfg=config();cfg['local_scoring_auxiliary']=copy.deepcopy(DEFAULT)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            _,data=collect(model,p,1,branches=True);annotate(data);annotate(p.snapshot_data)
            ds=updates.Dataset(data,'cpu');p.prepare(ds)
            actor=[i for i in range(len(data)) if len(data[i][3].choice_ids)>1]
            p.after_ppo(model,actor,actor[:4],{},critic_steps=1,critic_rows=len(p.train_ids))
            p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,actor,p.collection_source,model,p.split['selection'])
            p.buffer.verify(p.buffer.pending[-1])
            with p.buffer.datasets('cpu') as (critic,policy,receipts):
                self.assertIn('local-scoring.json',receipts[0]['files'])
                for i in range(len(critic.rows)):
                    self.assertEqual(critic.rows[i]['_local_scoring'],p.critic_dataset.rows[i]['_local_scoring'])
            unknown=p.buffer.directory/'unknown.bin';unknown.write_bytes(b'keep')
            before={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            with self.assertRaisesRegex(ValueError,'Unknown'):value_buffer._retire_owned(SOURCE_ROOT,run)
            self.assertEqual(before,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            unknown.unlink();result=value_buffer._retire_owned(SOURCE_ROOT,run)
            self.assertGreater(result['deleted_bytes'],0)
            self.assertEqual([f.name for f in p.buffer.directory.iterdir()],['owner.json'])

    def test_adoption_preserves_old_cache_beta_optimizers_and_evidence(self):
        model,opt,old=setup();old=D.beta_options(D.configure(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            for g in range(1,11):
                stage_only(model,p,g);write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            p.state=gae.initial(old['advantage'],10)
            p.critic_state['generation']=10
            from brass_ai.recovery import snapshot_r2
            p.critic_state['weight_recovery']=snapshot_r2.initial(10,p.critic_state['snapshot_weight'])
            p.schedule=D.initial(0)
            for g in range(1,11):p.schedule=P.advance(p.schedule,g,P.due(p.schedule))
            p.epoch_control=dict(ppg_epochs.initial(8,old['value_phase']),phases=1)
            p.beta=4.;p.trips=[False,True]
            parameter=next(model.parameters());parameter.grad=torch.ones_like(parameter)
            p.optimizer.step();p.optimizer.zero_grad(set_to_none=True)
            p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            p.pending_schedule=p.schedule;p.pending_epoch_control=p.epoch_control
            saved=p.payload();p.buffer.pending=None
            raw=dict(training={KEY:saved});weights=copy.deepcopy(model.state_dict())
            rng=torch.get_rng_state().clone();hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            cfg=D.sixteen_options(old)
            q=TrainingPhases(model,cfg,raw=raw,generation=10,root=SOURCE_ROOT,run=run)
            self.assertEqual(P.next_phase(q.schedule),26);self.assertEqual(q.cache_origin,2)
            self.assertEqual(q.beta,4.);self.assertEqual(q.trips,p.trips)
            self.assertTrue(_same_payload(q.optimizer.state_dict(),p.optimizer.state_dict()))
            self.assertEqual(q.state,p.state);self.assertEqual(q.critic_state,p.critic_state)
            self.assertTrue(_same_payload(weights,model.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            self.assertEqual(q.buffer.receipts,p.buffer.receipts)
            q.buffer.pending=copy.deepcopy(q.buffer.receipts)
            q.pending_schedule=q.schedule;q.pending_epoch_control=q.epoch_control
            migrated=q.payload();q.buffer.pending=None
            again=TrainingPhases(model,cfg,raw=dict(training={KEY:migrated}),generation=10,root=SOURCE_ROOT,run=run)
            self.assertEqual(again.schedule,q.schedule);self.assertIsNone(again.rolling_migration)
            stage_only(model,again,11)
            with again.buffer.datasets('cpu') as (critic,policy,receipts):
                self.assertEqual(len(receipts),9);self.assertIn('packing',receipts[0])
                self.assertNotIn('packing',receipts[1])
                ids=[0,len(critic.data.parts[0]),0]
                self.assertEqual(critic.batch(ids)[1][0].shape[0],3)


if __name__=='__main__':unittest.main()
