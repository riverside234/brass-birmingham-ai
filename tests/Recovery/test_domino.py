"""DOMiNO math, immutable assignments, masks, rollback and full-resume contracts."""
import copy
from collections import Counter
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.batching import model_id
from brass_ai.learning.features import Encoded
from brass_ai.recovery import domino_config as C,domino_state as S,domino_sampling as A,domino_features as D
from brass_ai.recovery.domino_auxiliary import Diversity,PPGCompanions,KEY as DKEY
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery import updates,value_preservation
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.policy import context_for
from brass_ai.step5.league import write_json
from test_gae_value_phase import examples,finish
from test_critic_descriptors import partition
from test_opening_ppo import opening_rows
from test_training_g1324 import smoke_config,configure,old_recipe
from test_value_buffer import source,commit


def fixture():
    model,opt,c=smoke_config();c=configure(c);c['diversity']=C.defaults()
    c['update'].update(epochs=1,effective_batch_size=4096,critic_batch_size=512,microbatch_size=64,preparation_workers=0)
    c['value_phase'].update(batch_size=1024,max_epochs=2,clone_rows=64,clone_batch_size=32)
    return model,opt,c


def descriptor_row(seat=0,phase=0,kind='loan'):
    from test_insolvency_recovery import row
    r=row(seat,round=(2,7,3,8)[phase],era='CanalEra' if phase<2 else 'RailEra',kind=kind)
    r['view'].update(seat=seat,links=[])
    for p in r['view']['players']:
        p.update(spent=8.,mat=[dict(type=i,tiles=[dict(level=2,seat=p['seat'],type=i)]) for i in C.NAMES[1:]])
    return r


def collect(model,phases,generation,*,games=126,branches=True):
    _,rows=examples(model,games=games,decisions=4,prefix=f'domino{generation}-')
    learner,snapshot,_=partition(rows,f'domino{generation}-')
    game_index={g:i for i,g in enumerate(dict.fromkeys(d[2][0] for d in rows))}
    # Partition gives authored new game identities. Sampling stays per whole seat.
    games_index={g:i for i,g in enumerate(dict.fromkeys(d[2][0] for d in learner+snapshot))}
    mid=model_id(model)
    for data,role in ((learner,'learner'),(snapshot,'frozen')):
        counts={}
        for i,(row,target,identity,e) in enumerate(data):
            g,seat=identity;index=games_index[g]+(generation-1)*games
            n=counts.get(identity,0);counts[identity]=n+1
            row['view']['context'].update(revision=4*n+seat,era='CanalEra' if n<2 else 'RailEra',round=(2,7,3,8)[n])
            spec=dict(kind='neural',model_id=mid,policy_id=row['behavior']['policy_id'])
            planned=A.assignments([spec]*4,0,seed=732,index=index,learner_id=spec['policy_id'] if role=='learner' else 'other',
                current=mid,checkpoints={mid:'1'*64})['seats'][seat]
            row['_domino']=planned;style,intent=context_for(planned['context']);row.update(style=style,intent=intent)
            state=e.state.copy();state[-12:]=style+intent
            data[i]=(row,target,identity,Encoded(state,e.actions,e.stable_to_relative,e.choice_ids))
            if role=='learner':row['_domino_event']=D.event(descriptor_row(seat,n,row['choices'][row['chosen']]['kind']))
    b,learner=opening_rows(model,learner,phases.config['exploration'],opening=phases.config['opening'],objective=phases.config['update']['opening_objective'])
    b.entropy_alphas=copy.deepcopy(phases.config['update']['fixed_entropy']['weights'])
    # Bind the same final behavior family the collector uses.
    for row,_,_,_ in learner:
        a=row['_domino'];a['legacy_policy_id']=b.policy_id
        a['behavior_family_id']=digest(dict(policy_id=b.policy_id,context_contract=C.CONTEXT,context_encoding=C.ENCODING))
        a['seat_policy_id']=digest(dict(behavior_family_id=a['behavior_family_id'],strategy_id=a['strategy_id']))
        a['receipt']=digest({k:v for k,v in a.items() if k!='receipt'})
    if branches:
        row,t,(g,s),e=learner[-1];r=copy.deepcopy(row)
        r.update(_branch=True,_parent_game_id=g)
        first=next(x[0] for x in learner if x[2]==(g,s) and '_reward_trajectory' in x[0])
        r['_reward_trajectory']=dict(first['_reward_trajectory'],cost=r['_insolvency_cost_to_go'],utility=r['_reward_return'][s])
        r['_insolvency_trajectory']=dict(events=0,shortfall=0,cost=0.,by_era={})
        r.pop('_domino_event',None)
        learner.append((r,t,(g+'/branch/authored',s),e))
    phases.snapshot_data=snapshot;phases.collection_source=dict(source(b),snapshot_rows=True)
    return b,learner


class Mathematics(unittest.TestCase):
    def test_strict_config_manifest_and_exact_contexts(self):
        C.validate(C.defaults());self.assertEqual(D.validate_manifest(),D.manifest())
        self.assertEqual([x['name'] for x in C.mapping()],list(C.NAMES))
        self.assertEqual(len(C.NAMES),7)
        for change in (dict(strategy_count=True),dict(quality_ratio=float('nan')),dict(foo=1),
            dict(contexts=list(reversed(C.mapping()))),dict(descriptor_manifest_sha256='0'*64),dict(reference_strategy=1)):
            with self.assertRaises(ValueError):C.validate(dict(C.defaults(),**change))

    def test_iid_repeatable_rng_isolated_no_modulo_quotas(self):
        random.seed(91);before=random.getstate();draws=[A.strategy(9,g,s) for g in range(10000) for s in range(4)]
        self.assertEqual(random.getstate(),before);counts=Counter(draws)
        for z in range(7):self.assertLess(abs(counts[z]/len(draws)-1/7),.008)
        self.assertEqual(draws[:400],[A.strategy(9,g,s) for g in range(100) for s in range(4)])
        self.assertTrue(any(len(set(draws[g:g+4]))<4 for g in range(0,400,4)))

    def test_nearest_ties_zero_and_local_finite_difference(self):
        x=np.arange(7,dtype=float)[:,None]*np.array([[1.,2.]])
        d,j,dist=S.directions(x);self.assertEqual(j.tolist(),[1,0,1,2,3,4,5]);self.assertTrue(np.isinf(dist.diagonal()).all())
        phi=np.array([.2,.7]);h=1e-6;i=4
        f=lambda y:.5*np.square(y-x[j[i]]).sum()
        self.assertAlmostEqual(float(phi@d[i]),(f(x[i]+h*phi)-f(x[i]-h*phi))/(2*h),places=7)
        zero,n,_=S.directions(np.zeros((7,2)));self.assertFalse(zero.any());self.assertEqual(n.tolist(),[1,0,0,0,0,0,0])
        self.assertEqual(d[0].tolist(),[0.,0.])

    def test_dual_sign_reference_support_and_commit_once(self):
        c=C.defaults();s=S.initial(c,0,'r')
        self.assertEqual(S.coefficients(s,c).tolist(),[1.]*7)
        s.update(centroids=[np.zeros(4*D.width()).tolist() for _ in range(7)],quality=[1.,.8,1.2,1.,1.,1.,1.],seat_games=[32]*7,distinct_games=[16]*7)
        n=S.advance(s,c,[],1)
        self.assertGreater(n['logits'][0],0);self.assertLess(n['logits'][1],0);self.assertEqual(S.coefficients(n,c)[0],1.)
        self.assertEqual(s['adam']['step'],0);self.assertEqual(n['adam']['step'],1)
        with self.assertRaises(ValueError):S.advance(n,c,[],1)
        for ref in (0.,-1.,None):
            a=copy.deepcopy(s);a['quality'][0]=ref;b=S.advance(a,c,[],1)
            self.assertEqual(b['logits'],a['logits']);self.assertEqual(b['adam'],a['adam'])
        a=copy.deepcopy(s);a['quality'][0]=float('nan')
        with self.assertRaises(ValueError):S.advance(a,c,[],1)

    def test_features_perspective_irrelevant_labels_phases_normalization(self):
        a=descriptor_row(0,0);b=descriptor_row(3,0)
        b.update(strategy_id=6,model_id='irrelevant',seed=823)
        np.testing.assert_array_equal(D.event(a)['x'],D.event(b)['x'])
        one=D.event(a);late=D.event(descriptor_row(0,3,'pass'))
        x,n=D.occupancy([one,late]);y,m=D.occupancy([one,one,None,late])
        np.testing.assert_allclose(x,y);self.assertEqual(n,[1,0,0,1]);self.assertEqual(m,[2,0,0,1])
        changed,_=D.occupancy([dict(one,phase=3),dict(late,phase=0)])
        self.assertFalse(np.array_equal(x,changed))
        nonaction=copy.deepcopy(a);nonaction['view']['context']['phase']='payment';self.assertIsNone(D.event(nonaction))


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_head_init_preserves_weights_adam_rng_and_old_export(self):
        m,o,c=fixture();before=copy.deepcopy((m.state_dict(),o.state_dict(),torch.get_rng_state()))
        d=Diversity(m,c['diversity'],regime='r')
        self.assertTrue(_same_payload(before,(m.state_dict(),o.state_dict(),torch.get_rng_state())))
        self.assertTrue(torch.equal(d.head[-1].weight,torch.zeros_like(d.head[-1].weight)))
        self.assertFalse(any('domino' in k for k in m.state_dict()))
        with self.assertRaisesRegex(ValueError,'Missing DOMiNO'):Diversity(m,c['diversity'],generation=1)

    def test_current_only_targets_preservation_trunk_and_joint_rollback(self):
        m,o,c=fixture()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));d=p.diversity=Diversity(m,c['diversity'],regime='r')
            b,rows=collect(m,p,1);ds=updates.Dataset(rows,'cpu');p.prepare(ds)
            self.assertEqual(d.report['readiness'],'centroids_incomplete')
            self.assertFalse(set(d.train_keys)&d.heldout_keys)
            self.assertEqual(d.state['generation'],0);self.assertGreater(d.pending['seat_games'][0],32)
            self.assertNotIn(next(k for k in (tuple((x[2][0],x[0]['seat'],x[0]['view']['context']['revision'])) for x in rows) if '/branch/' in k[0]),d.targets)
            d.state=d.pending;d.pending=None
            self.assertTrue(d.active)
            # Freeze nonzero intrinsic outputs and pull the head off them: preservation reaches the trunk.
            with torch.no_grad():d.head[-1].weight.fill_(.1)
            f=value_preservation.FrozenValues(m,p.critic_dataset,c['update'],diversity=d)
            ids=p.learner_train_ids[:4];ix,inputs,_=ds.value_batch(ids,m)
            with torch.no_grad():d.head[-1].bias.add_(.02)
            loss=f.losses(m,inputs,ix)['intrinsic'].sum();loss.backward()
            self.assertGreater(sum(float(x.grad.abs().sum()) for n,x in m.named_parameters() if n.startswith('state_encoder') and x.grad is not None),0)
            saved=d.snapshot();d.begin('ppg');d.prediction(m,inputs,ds,ix).square().mean().backward();d.step('ppg',1e-5)
            self.assertFalse(_same_payload(saved,d.snapshot()));d.restore(saved);self.assertTrue(_same_payload(saved,d.snapshot()))
            actor=[i for i in range(len(rows)) if len(rows[i][3].choice_ids)>1]
            task=torch.ones(len(rows));baseline=f.values['intrinsic'][:len(rows)]
            mixed=d.combine(task,ds,actor,baseline)
            self.assertEqual(float(mixed[-1]),1.) # branch stays task-only
            for i in actor:
                if rows[i][0]['_domino']['strategy_id']==0:self.assertEqual(float(mixed[i]),1.)

    def test_new_config_only_diversity_and_off_no_mutation(self):
        from brass_ai.recovery.config import validate_config
        old=configure(old_recipe());new=validate_config(C.configure(old))
        self.assertEqual({k:v for k,v in new.items() if k!='diversity'},old)
        off=dict(new,diversity=C.defaults('off'));self.assertFalse(C.enabled(off));validate_config(off)


if __name__=='__main__':unittest.main()
