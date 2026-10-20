"""Critic-only opponent rows, v11.2 upgrade, categorical metadata and EV gates."""
import copy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_mlp_v11 import inputs
from test_gae_value_phase import fixture,examples,ppo,finish,config
from test_value_buffer import commit,source
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.model import FourSeatNet,ModelConfig,LEGACY_ARCHITECTURE
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.learning.seat_descriptors import (unknown_numpy,unknown_tensor,descriptor,relative,validate,
    critic_inputs,drop_game,INDUSTRIES)
from brass_ai.recovery import critic_data,gae,value_phase,value_buffer,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.critic_upgrade import upgrade_model
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec


def style():return [1/8]*16+[1/6]*12


def mixed_fixture(*,prefix='critic-fixture-',preset='small'):
    model,opt,cfg,(behavior,data)=fixture(preset=preset,games=9,decisions=4)
    learner,snapshot,excluded=partition(data,prefix)
    cfg['critic_data']['rollout']['mode']='snapshots'
    return model,opt,cfg,behavior,learner,snapshot,excluded


def partition(data,prefix):
    games=list(dict.fromkeys(d[2][0] for d in data));learner=[];snapshot=[];excluded=[]
    for row,target,(game,seat),encoded in data:
        g=games.index(game);kind=('selfplay','historical','legacy')[g%3]
        names=['learner']*4 if kind=='selfplay' else ['learner','learner','snapshot','il'] if kind=='historical' else ['learner','learner','rule_bot','rule_bot']
        stable=[descriptor(role,0 if role in ('learner','rule_bot') else 3,style()) for role in names]
        name=next(prefix+str(g)+'-'+str(i) for i in range(10000) if value_phase.heldout(prefix+str(g)+'-'+str(i),.05)==(g<3))
        r=dict(row,_critic_role=names[seat],_table_kind=kind,_seat_descriptors=relative(stable,encoded.stable_to_relative),_descriptor_generation=3)
        if names[seat]!='learner':r['behavior']=dict(row['behavior'],policy_id='frozen-'+names[seat])
        (learner if names[seat]=='learner' else snapshot if names[seat]=='snapshot' else excluded).append((r,target,(name,seat),encoded))
    return learner,snapshot,excluded


class Descriptors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_zero_upgrade_bit_identical_policy_and_all_outcome_heads(self):
        for preset in ('small','medium'):
            old=FourSeatNet(replace(ModelConfig.preset(preset),architecture=LEGACY_ARCHITECTURE)).eval()
            rng=torch.get_rng_state().clone();new=upgrade_model(old).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(new.parameter_count-old.parameter_count,6272)
            for batch in (1,2):
                x=inputs(batch,9);d=torch.tensor(np.stack([[descriptor(r,4,style()) for r in ('snapshot','il','rule_bot','learner')]]*batch))
                with torch.no_grad():
                    for a,b in zip(old(*x),new(*x,descriptors=d)):self.assertTrue(torch.equal(a,b))
            with self.assertRaisesRegex(ValueError,'11.2'):ModelConfig.from_dict(old.config.to_dict())

    def test_descriptors_cannot_change_policy_but_critic_gradients_flow(self):
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(config())))
        x=inputs(2,7);d=unknown_tensor(x[0]);d[:,:,4]=0;d[:,:,0]=1
        with torch.no_grad():
            model.outcomes[0].weight[:,-32:].normal_(std=.01)
            model.utility_head[0].weight[:,-32:].normal_(std=.01);model.utility_head[2].weight.normal_(std=.01)
        a=model(*x,descriptors=d);b=model(*x)
        self.assertTrue(torch.equal(a[0],b[0]));self.assertFalse(torch.equal(a[3],b[3]));self.assertFalse(torch.equal(a[4],b[4]))
        (a[3].square().mean()+a[4].square().mean()).backward()
        self.assertGreater(float(model.descriptor_encoder[0].weight.grad.abs().sum()),0)
        self.assertIsNone(model.policy_scorer[0].weight.grad)

    def test_relative_mapping_roles_lag_and_exact_game_dropout(self):
        stable=[descriptor(r,0 if r=='learner' else 256,style()) for r in ('learner','snapshot','il','rule_bot')]
        self.assertEqual(stable[0][5],0);self.assertEqual(stable[1][5],1)
        for observer in range(4):
            mapping=[(s-observer)%4 for s in range(4)];r=relative(stable,mapping)
            for s in range(4):np.testing.assert_array_equal(r[mapping[s]],stable[s])
        model,opt,cfg,b,data,snap,_=mixed_fixture();ds=updates.Dataset(data+snap,'cpu')
        rng=torch.get_rng_state().clone();drop=(7,1,.25,'value');ids=list(range(len(ds.data)))
        a=critic_inputs(ds,ids,dropout=drop);b=critic_inputs(ds,ids,dropout=drop)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for i in ids:
            expected=unknown_tensor(ds.descriptors)[i] if drop_game(ds.data[i][2][0],7,1,.25,'value') else ds.descriptors[i]
            self.assertTrue(torch.equal(a[i],expected))
        self.assertTrue(torch.equal(critic_inputs(ds,ids),ds.descriptors))
        self.assertNotEqual([drop_game(str(i),7,1,.25,'value') for i in range(100)],
                            [drop_game(str(i),7,2,.25,'value') for i in range(100)])
        for bad in (np.zeros((4,34)),np.ones((4,34)),np.full((4,34),np.nan)):
            with self.assertRaises(ValueError):validate(bad)

    def test_fingerprint_raw_families_conditional_industries_and_v4_choices(self):
        rows=[]
        for era in ('CanalEra','RailEra'):
            rows.append(dict(era=era,phase='action',kinds=['build']*6+['loan','pass'],industries=list(INDUSTRIES)+[None,None],chosen=2))
        probe=SimpleNamespace(rows=rows,metadata={'probe_hash':'fixed'},arrays={'offsets':np.array([0,8,16])})
        fp=critic_data.fingerprint(probe,np.array([1/8]*16));bot=critic_data.fingerprint(probe,rule_bot=True)
        for v in (fp['values'],bot['values']):
            for a,b in ((0,8),(8,16),(16,22),(22,28)):self.assertAlmostEqual(sum(v[a:b]),1,places=6)
        self.assertEqual(bot['values'][0],1);self.assertEqual(bot['values'][18],1)
        self.assertAlmostEqual(fp['values'][0],.75);self.assertEqual(fp['action_positions'],[1,1])
        probe.rows[1]['phase']='continuation'
        self.assertEqual(critic_data.fingerprint(probe,np.array([1/8]*16))['values'][8:16],[0]*8)

    def test_table_rotation_uses_physical_seats_and_generation_lag(self):
        from brass_ai.recovery.execution import neural_spec
        fp=dict(values=style());members=[dict(model_id='snap',generation=4,fingerprint=fp),dict(model_id='il',generation=0,fingerprint=fp)]
        policies=[neural_spec('current',.08,'mixed'),neural_spec('snap'),neural_spec('il'),dict(kind='legacy',policy_id='bot')]
        for rotation in range(4):
            table=critic_data.planned_table(policies,rotation,'current',9,members,fp,fp,
                learner_policy_id=policies[0]['policy_id'])
            self.assertEqual(table['seats'][rotation]['role'],'learner')
            self.assertEqual(table['seats'][(rotation+1)%4]['role'],'snapshot')
            self.assertAlmostEqual(table['stable'][(rotation+1)%4][5],np.log1p(5)/np.log1p(256),places=6)

    def test_same_weights_frozen_opponent_is_not_a_learner(self):
        from brass_ai.recovery.execution import neural_spec
        fp=dict(values=style());learner=neural_spec('current',.08,'mixed')
        frozen=neural_spec('current',.05,'neutral')
        for born,role in ((0,'il'),(9,'snapshot')):
            members=[dict(model_id='current',generation=born,fingerprint=fp)]
            table=critic_data.planned_table([learner,frozen,learner,frozen],0,'current',9,members,fp,fp,
                learner_policy_id=learner['policy_id'])
            self.assertEqual([s['role'] for s in table['seats']],['learner',role,'learner',role])
            game=dict(opponents=[learner['policy_id'],frozen['policy_id']]*2,records=[])
            critic_data.check_table(table,game,learner['policy_id'])


class CriticTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_ingress_keeps_snapshot_metadata_and_excludes_il_and_rule_rows(self):
        from brass_ai.recovery.ingress import _prepare_inner
        from brass_ai.recovery.execution import neural_spec
        from test_insolvency_recovery import row
        from fixtures import result
        from brass_ai.learning.features import Encoded
        class Game(dict):pass
        fp=dict(values=style());learner=neural_spec('current',.08,'mixed')
        snap=neural_spec('snap',.05,'neutral');il=neural_spec('il',.05,'neutral')
        bot=dict(kind='legacy',policy_id='bot')
        members=[dict(model_id='snap',generation=3,fingerprint=fp),dict(model_id='il',generation=0,fingerprint=fp)]
        x=inputs(1,1);encoded=Encoded(x[0][0].numpy(),x[1][0].numpy(),np.arange(4),('a',))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'record';path.write_bytes(b'checked-test-replay')
            for kind,policies,expected in (('historical',[learner,learner,snap,il],['learner','learner','snapshot']),
                    ('legacy',[learner,learner,bot,bot],['learner','learner'])):
                records=[]
                for seat,p in enumerate(policies):
                    records.append(dict(row(seat),step=seat,style=[0.]*4,intent=[0.]*8,teacher=None,
                        behavior=dict(kind='external' if p is bot else 'model',policy_id=p['policy_id'],probability=1.)))
                game=Game(game_id=kind,setup_group=kind,split='train',records=records,result=result(),
                    opponents=[p['policy_id'] for p in policies])
                game.table_kind=kind
                game.critic_table=critic_data.planned_table(policies,0,'current',5,members,fp,fp,
                    learner_policy_id=learner['policy_id'])
                spec=reward_spec(config())
                task=([dict(path=str(path),sha256=sha(path),game_id=kind)],learner['policy_id'],{kind:kind},spec['insolvency'],spec)
                with patch('brass_ai.recovery.ingress.read_shards',return_value=[game]), \
                        patch('brass_ai.recovery.ingress.summarize',return_value={}), \
                        patch('brass_ai.recovery.updates.PreparedEncoder.encode',return_value=encoded):
                    _,data,_,_=_prepare_inner(task)
                self.assertEqual([r[0]['_critic_role'] for r in data],expected)
                for r,_,_,_ in data:
                    np.testing.assert_array_equal(r['_seat_descriptors'],game.critic_table['stable'])
                    self.assertEqual(r['_descriptor_generation'],5)

    def test_snapshot_rows_critic_only_holdout_excluded_and_true_baselines(self):
        model,opt,cfg,b,data,snap,excluded=mixed_fixture();phases=TrainingPhases(model,cfg);phases.snapshot_data=snap
        seen=[];original=updates.value_output
        def output(model,inputs,ds,indices,**kwargs):
            seen.append((list(ds.roles),[int(i) for i in indices],kwargs.get('dropout')))
            return original(model,inputs,ds,indices,**kwargs)
        with patch('brass_ai.recovery.updates.value_output',side_effect=output):report=ppo(model,opt,cfg,b,data,phases)
        self.assertEqual(report['learner_decisions'],len(data));self.assertGreater(report['critic_rows']['snapshot_distinct_used'],0)
        self.assertTrue(all(i<len(data) for i in phases.actor_ids));self.assertTrue(set(phases.heldout_ids).isdisjoint(phases.train_ids))
        self.assertTrue(all(r not in ('il','rule_bot') for r in phases.critic_dataset.roles))
        self.assertTrue(any(d is None and len(roles)==len(data) for roles,_,d in seen))
        self.assertTrue(any(d is not None and len(roles)>len(data) for roles,_,d in seen))
        for roles,ids,d in seen:
            if d is not None:self.assertTrue(set(ids).isdisjoint(phases.heldout_ids))
        self.assertEqual(report['gae']['lambda_by_table'],dict(selfplay=1.,historical=1.,legacy=1.))
        value=phases.fit(model)
        self.assertIn('descriptor_gain',value)
        for kind in ('selfplay','historical','legacy'):
            self.assertIn('table:'+kind+'/role:learner',value['heldout_after']['groups'])
        self.assertIn('role:snapshot',value['heldout_after']['groups'])
        # A forged route must fail before any PPO actor step.
        from brass_ai.recovery.league import Behavior
        from brass_ai.learning.batching import model_id
        with self.assertRaisesRegex(ValueError,'External or stale'):ppo(model,opt,cfg,Behavior(model_id(model),.08),data+snap,None)

    def test_phase1_then_snapshot_weight_gate_is_committed_and_health_gated(self):
        c=copy.deepcopy(critic_data.DEFAULT);s=critic_data.initial(c)
        report=dict(optimizer_steps=1,heldout_after=dict(groups={'role:learner':dict(explained_variance=.6)}),descriptor_gain={'learner_ev_gain':.01})
        s,r=critic_data.advance(s,c,generation=1,report=report,healthy=False,fingerprint_now=dict(values=style()))
        self.assertFalse(s['snapshots_active'])
        s,r=critic_data.advance(s,c,generation=2,report=report,healthy=True,fingerprint_now=dict(values=style()))
        self.assertTrue(s['snapshots_active']);self.assertEqual(r['decision'],'enable_snapshots');self.assertEqual(r['fingerprint_mean_absolute_drift'],0.)
        report['heldout_after']['groups']['role:learner']['explained_variance']=.579
        s,r=critic_data.advance(s,c,generation=3,report=report,healthy=True,fingerprint_now=None,ppo_snapshot_rows=8)
        self.assertEqual(s['snapshot_weight'],.5);self.assertEqual(r['weight_used'],1.)
        model,opt,cfg,b,data,snap,_=mixed_fixture();cfg['critic_data']['rollout']['mode']='auto'
        phase=TrainingPhases(model,cfg);phase.snapshot_data=snap;phase.prepare(updates.Dataset(data,'cpu'))
        self.assertTrue(all(phase.critic_dataset.roles[i]=='learner' for i in phase.train_ids))

    def test_history_v4_ev_streaks_missing_reset_and_exact_resume(self):
        c=copy.deepcopy(gae.DEFAULT);c['gate']['mode']='fixed';c['lambda']=.95;s=gae.initial(c)
        # Archived contract remains readable; new window behavior is covered
        # in test_health_ev_window. Continuation upgrades this block explicitly.
        c['table_gate']=copy.deepcopy(gae.ARCHIVED_TABLE_GATE);s=gae.initial(c)
        def advance(s,h=.56,v=.56):
            groups={'table:selfplay/role:learner':dict(explained_variance=.6),
                'table:historical/role:learner':dict(explained_variance=h),'table:legacy/role:learner':dict(explained_variance=v)}
            return gae.advance(s,c,generation=s['generation']+1,value_report=dict(optimizer_steps=1,heldout_after=dict(groups=groups)),health_ok=True,rating=dict(ok=True))
        for _ in range(9):s,_=advance(s)
        a,ar=advance(s);b,br=advance(copy.deepcopy(s));self.assertEqual((a,ar),(b,br))
        self.assertTrue(all(a['table_enabled'].values()));self.assertEqual(ar['table_gate']['legacy']['lambda_used'],1.)
        self.assertEqual(ar['table_gate']['legacy']['lambda_next'],.95)
        independent,ir=advance(copy.deepcopy(s),h=.54,v=.56)
        self.assertEqual(independent['table_enabled'],dict(historical=False,legacy=True))
        self.assertEqual(ir['table_gate']['historical']['lambda_next'],1.)
        kept,_=advance(independent,h=None,v=None)
        self.assertTrue(kept['table_enabled']['legacy'])
        self.assertEqual(kept['table_origins']['legacy'],10)
        s,_=advance(s,h=None,v=.54);self.assertEqual(s['table_streaks'],dict(historical=0,legacy=0))
        for _ in range(9):s,_=advance(s)
        self.assertFalse(any(s['table_enabled'].values()))
        s,_=advance(s);self.assertTrue(all(s['table_enabled'].values()))

    def test_gae_never_uses_mixed_lambda_before_gate(self):
        rows=[(dict(_table_kind=kind,view=dict(context=dict(era='RailEra',phase='action',round=5))),None,(kind,0),None)
              for kind in ('selfplay','selfplay','historical','historical','legacy','legacy')]
        target=torch.tensor([1.,1.,1.,1.,1.,1.]);mc=target-torch.tensor([.2,.3,.2,.3,.2,.3])
        actual,r=gae.compute(rows,list(range(6)),mc,target,.95,[.98,.95,.9],dict(selfplay=.95,historical=1.,legacy=1.))
        self.assertFalse(torch.equal(actual[:2],mc[:2]));self.assertTrue(torch.equal(actual[2:],mc[2:]))

    def test_snapshot_costs_and_official_targets_include_its_own_future_debt(self):
        from test_insolvency_recovery import row
        from fixtures import result
        from brass_ai.learning.reward import terminal_utility
        from brass_ai.learning.replay import terminal_targets
        from brass_ai.learning.features import Encoded
        rows=[row(2),row(2,phase='debt',kind='liquidate',debts={2:5}),row(2,era='RailEra',round=8)]
        for i,r in enumerate(rows):r.update(behavior=dict(kind='model',policy_id='snapshot'),style=[0.]*4,intent=[0.]*8,teacher=None,step=i)
        game=dict(game_id='snapshot-game',result=result(),records=rows)
        x=inputs(1,1);e=Encoded(x[0][0].numpy(),x[1][0].numpy(),np.array([2,3,0,1]),('a',))
        spec=reward_spec(config())
        with patch('brass_ai.recovery.updates.PreparedEncoder.encode',return_value=e):data=updates.prepared([game],insolvency=spec['insolvency'],reward=spec)
        terminal=terminal_utility(result(),spec)
        self.assertEqual(data[0][1],terminal_targets(result()));self.assertEqual(data[0][2],('snapshot-game',2))
        self.assertAlmostEqual(data[0][0]['_insolvency_cost_to_go'],.0075)
        self.assertAlmostEqual(data[0][0]['_reward_return'][2],terminal[2]-.0075)
        self.assertAlmostEqual(data[1][0]['_reward_return'][2],terminal[2])
        self.assertAlmostEqual(data[1][0]['_reward_remaining'][2],.0675)

    def test_disk_descriptors_snapshot_roles_and_resume_exact_next_update(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp)/'run';other_run=Path(tmp)/'restored'
            model,opt,cfg,b,data,snap,_=mixed_fixture();cfg['value_phase']['buffer_iterations']=2
            phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run);phase.snapshot_data=snap
            phase.collection_source=dict(source(b),snapshot_rows=True)
            report=ppo(model,opt,cfg,b,data,phase);phase.fit(model);finish(phase,report,1)
            path=run/'generation-1.pt';save_checkpoint(path,model,opt,step=1,training={KEY:phase.payload()});commit(run,phase,1)
            import shutil
            shutil.copytree(run,other_run)
            restored,raw=load_checkpoint(path);adam=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,adam)
            resumed=TrainingPhases(restored,cfg,raw=raw,generation=1,root=SOURCE_ROOT,run=other_run)
            b,raw_data=examples(model,games=9,decisions=4,prefix='second-');data,snap,_=partition(raw_data,'second-')
            rng=torch.get_rng_state().clone();results=[]
            for net,optimizer,p in ((model,opt,phase),(restored,adam,resumed)):
                torch.set_rng_state(rng);p.snapshot_data=snap;p.collection_source=dict(source(b),snapshot_rows=True)
                r=ppo(net,optimizer,cfg,b,data,p);v=p.fit(net);finish(p,r,2)
                results.append(torch.get_rng_state().clone())
                self.assertGreater(v['previous_train_rows'],0)
                with p.buffer.previous() as old:
                    self.assertTrue(np.any(old.array['role']==1));self.assertEqual(old.array['descriptors'].shape[1:],(4,34))
                commit(p.buffer.run,p,2);self.assertFalse(p.buffer.path(1,'npy').exists())
            self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()));self.assertTrue(_same_payload(opt.state_dict(),adam.state_dict()))
            self.assertTrue(_same_payload(phase.optimizer.state_dict(),resumed.optimizer.state_dict()));self.assertEqual(phase.state,resumed.state)
            self.assertEqual(phase.critic_state,resumed.critic_state);self.assertTrue(torch.equal(*results))


if __name__=='__main__':unittest.main()
