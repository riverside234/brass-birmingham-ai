"""Three-decision tempered openings, matched PPO and loss16 resume regressions."""
import copy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch,Mock

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import exploration as E,opening_objective as O,ppg_g210 as G,updates,value_phase as V,gae,snapshot_r2
from brass_ai.recovery.training_revision_g1120 import configure
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.review_metrics import CategoricalProbe,categorical_kl
from brass_ai.recovery.branching import eligible,continue_branch,validate_opening_rows
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import write_json,implementation_id
from brass_ai.step5.policy import NeuralPolicy,exploration_distribution
from test_opening_ppo import opening_rows
from test_gae_value_phase import fixture
from test_ppg_eight import collect as original_collect,stage_only
from test_value_buffer import source
from test_training_g1050 import smoke_config as prior_config
from test_ppg_g210 import prepare

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1120-20261026T012745632742Z'


def smoke_config():
    model,opt,c=prior_config();c=G.loss16_options(c)
    c['update']['opening_objective']=copy.deepcopy(O.T13)
    return model,opt,c


def collect_revised(model,phase,generation,*,branches=False):
    _,data=original_collect(model,phase,generation,branches=branches)
    behavior,data=opening_rows(model,data,.05,opening=E.THREE_OPENING,objective=O.T13)
    phase.collection_source=dict(source(behavior),snapshot_rows=True)
    return behavior,data


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_recipe_fresh_defaults_and_validation(self):
        old=read(EXPORT/'recorded-config.json');new=validate_config(configure(old))
        self.assertEqual(configure(new),new)
        self.assertEqual({k for k in new if old[k]!=new[k]},{'opening','branching','update','value_phase'})
        self.assertEqual(new['update'],dict(old['update'],opening_objective=O.T13))
        self.assertEqual(new['value_phase'],dict(old['value_phase'],contract=G.LOSS16_CONTRACT,acceptance_steps=16))
        self.assertEqual(new['opening'],E.THREE_OPENING);self.assertEqual(new['branching'],E.THREE_BRANCH)
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        from brass_ai.recovery.training_revision_g1320 import configure as reward_config
        from brass_ai.recovery.training_revision_g1322 import configure as current_config
        expected=current_config(reward_config(G.kl32_options(new)))
        from brass_ai.recovery.training_revision_g1324 import configure as latest_recipe
        expected=latest_recipe(expected)
        for k in ('opening','branching','value_phase'):self.assertEqual(fresh[k],expected[k])
        for block,key,value in [('opening','action_decisions_per_seat',2),('branching','skip_opening_decisions',2),
                                ('value_phase','acceptance_steps',8),('value_phase','beta_clone_min',1.)]:
            bad=copy.deepcopy(new);bad[block][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(new);bad['update']['opening_objective']=copy.deepcopy(O.T15)
        with self.assertRaises(ValueError):validate_config(bad)

    def test_actual_sampling_three_actions_per_seat_including_forced(self):
        p=NeuralPolicy(None,exploration=.05,focus='neutral',evaluator=SimpleNamespace(model_id='model'),
            exploration_version=3,opening=E.THREE_OPENING)
        rngs=[random.Random(i) for i in range(4)];records=[]
        phases=('action','continuation','action','debt','action','action')
        for n,phase in enumerate(phases):
            for seat,rng in enumerate(rngs):
                choices=[dict(kind='loan')] if n==2 else [dict(kind='loan'),dict(kind='pass')]
                raw=np.array([1.] if n==2 else [.7,.3]);view=dict(context=dict(phase=phase))
                chosen,prob,_=p.sample_training(view,choices,rng,raw,[0]*8)
                b=dict(p.last_behavior[rng],kind='model',policy_id=p.policy_id,probability=prob)
                records.append(dict(seat=seat,view=view,behavior=b))
                tempered=raw**(1/b['temperature']);tempered/=tempered.sum()
                expected=.95*tempered+.05*exploration_distribution(choices,[0]*8,3,phase)
                self.assertAlmostEqual(prob,expected[chosen],places=7)
        validate_opening_rows(records,E.THREE_OPENING)
        for seat in range(4):
            rows=[r['behavior'] for r in records if r['seat']==seat]
            self.assertEqual([b['temperature'] for b in rows],[1.3,1.,1.3,1.,1.3,1.])
            self.assertEqual([b['action_decision'] for b in rows],[0,None,1,None,2,3])
        with self.assertRaises(ValueError):validate_opening_rows(records,E.REVISED_OPENING)

    def test_three_decoupled_rows_per_parent_and_branch_and_complete_kl_reporting(self):
        m,_,cfg,(_,data)=fixture(games=3,decisions=4)
        parent=[d for d in data if d[2]==data[1][2]]  # Seat 1 has four nonforced action decisions.
        branch=[(dict(r,_branch=True),t,(g+'/branch/test',s),e) for r,t,(g,s),e in parent]
        b,rows=opening_rows(m,parent+branch,.05,opening=E.THREE_OPENING,objective=O.T13)
        ds=updates.Dataset(rows,'cpu');ds.opening_reference=O.Reference(ds,b,O.T13)
        self.assertEqual(ds.opening_reference.mask.tolist(),[True,True,True,False]*2)
        actual=ds.old.clone();updates.policy_measure(m,ds,range(len(rows)),b,cfg['update'],Timings(),verify=True)
        self.assertTrue(ds.opening_reference.frozen);self.assertTrue(torch.equal(actual,ds.old))
        report=ds.opening_reference.report();self.assertEqual(report['opening_rows'],6);self.assertEqual(report['branch_opening_rows'],3)
        probe=CategoricalProbe(m,ds,list(range(len(rows))),b,dict(cfg['update'],opening_objective=O.T13))
        class Shift(torch.nn.Module):
            def forward(self,*inputs):
                out=list(m(*inputs));out[0]=out[0]+torch.arange(out[0].shape[1])*.4
                return out
        result=probe.measure(Shift());self.assertEqual(result['opening_positions'],6)
        expected=0.
        with torch.no_grad():
            for index in range(len(probe.parts)):
                _,old,ix,inputs,q=probe._batch(index);mask=probe.opening_masks[index]
                expected+=float(categorical_kl(probe.raw_parts[index][mask],Shift()(*inputs)[0].softmax(-1)[mask]))
        self.assertAlmostEqual(result['opening_temperature1_categorical_kl'],expected/6,places=6)
        self.assertNotAlmostEqual(result['categorical_kl'],result['behavior_categorical_kl'],places=6)
        for objective in (O.DEFAULT,O.T15):
            with self.assertRaises(ValueError):O.Reference(ds,b,objective)
        bad=copy.deepcopy(rows);bad[0][0]['behavior']['temperature']=1.5
        with self.assertRaises(ValueError):updates.verify_behavior_metadata(updates.Dataset(bad,'cpu'),b)
        bad=copy.deepcopy(rows);bad[2][0]['behavior']['action_decision']=3
        with self.assertRaises(ValueError):O.Reference(updates.Dataset(bad,'cpu'),b,O.T13)
        bad=copy.deepcopy(rows);bad[0][0][O.ROW_KEY]=O.T15
        with self.assertRaises(ValueError):O.Reference(updates.Dataset(bad,'cpu'),b,O.T13)

    def test_t15_opening_probe_regression(self):
        m,_,c,(_,data)=fixture(games=3,decisions=4)
        b,data=opening_rows(m,data,.05,opening=E.REVISED_OPENING,objective=O.T15)
        ds=updates.Dataset(data,'cpu');probe=CategoricalProbe(m,ds,list(range(len(data))),b,dict(c['update'],opening_objective=O.T15))
        self.assertGreater(probe.opening_positions,0)
        self.assertAlmostEqual(probe.measure(m)['opening_temperature1_categorical_kl'],0.,places=6)

    def test_fork_skips_three_and_resets_all_four_seats(self):
        context=dict(phase='action',era='CanalEra');view=dict(context=context,deck_count=1,own_hand=[1])
        choices=[dict(id=0),dict(id=1)]
        self.assertEqual([eligible(view,choices,n,E.THREE_BRANCH['skip_opening_decisions']) for n in range(4)],[False,False,False,True])
        self.assertTrue(eligible(view,choices,2))  # Explicit old recipe remains reproducible.
        row=dict(step=25,seat=2,view=view,choices=choices,chosen=0,style=[0.]*4,intent=[0.]*8,
            behavior=dict(probability=.5,component='policy',temperature=1.,action_decision=3))
        meta=dict(contract=E.THREE_BRANCH['contract'],seed='12345678',decision=25,chosen=1,marginal_probability=.3)
        packet=dict(row=row,metadata=meta,counts=[3,4,4,2],contexts=[{'s':i} for i in range(4)])
        assigned=[SimpleNamespace(contexts={},action_counts={},opening=E.THREE_OPENING) for _ in range(4)]
        host=Mock();host.call.return_value=context;host.apply.return_value={'terminal':True}
        parent=dict(game_id='parent',setup_group='setup',split='train',opponents=['model']*4)
        def turns(host,job,policies,timing,cap,context,game,rngs,encoded,encoder,lease):
            self.assertNotIn('branching',job)
            for seat,p in enumerate(policies):
                self.assertEqual(p.action_counts[rngs[seat]],0);self.assertEqual(p.opening,E.THREE_OPENING)
            self.assertEqual(game['records'][0]['behavior']['component'],'branch_move')
            return game
        with patch('brass_ai.recovery.execution._run_turns',side_effect=turns),patch('brass_ai.recovery.replay.CheckedGame'):
            continue_branch(host,dict(branching=E.THREE_BRANCH,provenance={}),parent,packet,assigned,None,2000,Mock(),None)
        self.assertEqual(packet['counts'],[3,4,4,2])

    def test_loss16_adoption_preserves_beta_cache_adams_rng_and_gates(self):
        m,opt,old=prior_config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            for optimizer in (opt,p.optimizer):
                optimizer.zero_grad(set_to_none=True);next(m.parameters()).grad=torch.zeros_like(next(m.parameters()));optimizer.step()
            p.beta=2.;p.trips=[True,False];p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            raw=dict(training={KEY:p.payload()});new=G.loss16_options(old)
            before=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            q=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.beta,2.);self.assertEqual(q.trips,p.trips);self.assertEqual(q.buffer.receipts,p.buffer.receipts)
            self.assertIsNone(q.startup(1));self.assertIsNone(q.g210_migration)
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(before,(m.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            q.buffer.pending=copy.deepcopy(q.buffer.receipts)
            resumed=TrainingPhases(m,new,raw=dict(training={KEY:q.payload()}),generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(resumed.beta,2.);self.assertIsNone(resumed.ppg_schedule_migration)

    def test_real_phase_checks_loss_and_kl_at16_and_restores_matching_adam(self):
        m,opt,c=smoke_config();c['value_phase']['batch_size']=2
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1)
            metric=V.metrics;captured={}
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1.-.001*step
                    captured.setdefault(step,copy.deepcopy((m.state_dict(),p.optimizer.state_dict())))
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.001,.003,.001]):r=p.fit(m)
            self.assertEqual(r['attempted_steps'],32);self.assertEqual(r['optimizer_steps'],16)
            self.assertEqual([x['steps'] for x in r['interval_reports']],[16,16])
            self.assertEqual([x['step'] for x in r['kl_checks']],[16,32])
            self.assertTrue(_same_payload(captured[16],(m.state_dict(),p.optimizer.state_dict())))

    def test_helper_is_generation_bound_narrow_and_read_only(self):
        spec=importlib.util.spec_from_file_location('g1120_helper',SOURCE_ROOT/'scripts/configure-training-g1120.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);state=read(EXPORT/'state.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated-checkpoint-stand-in')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            before=(sha(run/'state.json'),sha(checkpoint));output='configs/step5/g1120.json'
            report=helper.generate(root,run,output,1120)
            self.assertEqual(report,helper.generate(root,run,output,1120))
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)));self.assertFalse(report['beta_reset'])
            signature=dict(state['signature'],implementation=implementation_id(),config=configure(state['signature']['config']))
            self.assertTrue(check_signature(state,signature,reviewed=True))
            signature['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=True)
            with self.assertRaises(ValueError):helper.generate(root,run,'configs/step5/stale.json',1119)


if __name__=='__main__':unittest.main()
