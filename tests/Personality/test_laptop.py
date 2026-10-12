"""Focused P6 interfaces; authored fixtures never qualify difficulty or release."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from personality_fixtures import parent
from test_difficulty import named,own_goal
from brass_ai.personality.model import import_parent
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.personality.search.contracts import Profile
from brass_ai.personality.search.intent import IntentState
from brass_ai.personality.laptop import sampling,contracts,export,install,benchmark
from brass_ai.personality.laptop.runtime import completed_paths
from brass_ai.personality.laptop.server import public_error
from types import SimpleNamespace


def package():
    r=named(16,8);return dict(id='fixture-package',roster=r,fixture=True)


class LaptopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_sixteen_by_eight_uniform_member_bytes_and_fixed_stream(self):
        p=package();g=p['roster']['archetypes'][3];seen=set()
        for seed in range(256):
            v=sampling.choose(p,'game',str(seed),2,g['id'],'medium');seen.add(v['member'])
            self.assertEqual(v,sampling.choose(p,'game',str(seed),2,g['id'],'medium'))
            self.assertIn(v['member'],g['members'])
        self.assertEqual(seen,set(g['members']))

    def test_seat_streams_independent_of_sampling_order_and_tier(self):
        p=package();group=p['roster']['archetypes'][0]['id']
        a=[sampling.choose(p,'game','seed',s,group,'easy') for s in range(4)]
        b=[sampling.choose(p,'game','seed',s,group,'hard') for s in reversed(range(4))]
        self.assertEqual([x['member'] for x in a],[x['member'] for x in reversed(b)])
        self.assertEqual(len({x['rng_seed'] for x in a}),4)
        self.assertNotEqual(a[0],sampling.choose(p,'game2','seed',0,group,'easy'))

    def test_save_restores_rng_intent_variant_and_recall_exactly(self):
        p=package();assign=[dict(seat=s,archetype='random',tier='hard') for s in range(4)]
        save=sampling.new_save(p,'game','seed',assign);s=save['seats'][0]
        rng=random.Random();rng.setstate(sampling.tuples(json.loads(json.dumps(s['rng']))));rng.random();s['rng']=rng.getstate()
        member=next(m for m in p['roster']['members'] if m['id']==s['selection']['member'])
        intent=IntentState(0,Profile.parse(member['profile']));intent.goal=own_goal()['goal'];intent.key=('CanalEra',2,1,0)
        s['intent']=intent.snapshot();s['recall']=[dict(revision=1,hand=['location:Stafford'])]
        save=contracts.seal({k:v for k,v in save.items() if k!='id'});wire=json.loads(json.dumps(save))
        restored=sampling.restore(p,wire,'game');self.assertEqual(restored,wire)
        resumed=random.Random();resumed.setstate(sampling.tuples(restored['seats'][0]['rng']));self.assertEqual(resumed.random(),rng.random())

    def test_corrupt_foreign_or_rerolled_save_rejected(self):
        p=package();s=sampling.new_save(p,'g','s',[dict(seat=0,archetype='random',tier='easy')])
        with self.assertRaises(ValueError):sampling.restore(p,s,'different')
        bad=copy.deepcopy(s);bad['seats'][0]['selection']['member']='foreign'
        with self.assertRaises(ValueError):sampling.restore(p,bad,'g')
        bad=contracts.seal({k:v for k,v in bad.items() if k!='id'})
        with self.assertRaisesRegex(ValueError,'never reroll'):sampling.restore(p,bad,'g')

    def test_nested_manifest_is_pinned_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as d:
            root=Path(d);contracts.write(root/'nested/manifest.json',dict(v=1));contracts.write(root/'manifest.json',dict(root=True))
            self.assertEqual([x['path'] for x in contracts.inventory(root)],['nested/manifest.json'])
            for path in ('../escape','C:/escape','nested\\escape',''):
                with self.assertRaises(ValueError):contracts.child(root,path)

    def test_named_export_all_heads_dynamic_masks_and_seat_rotations(self):
        torch.manual_seed(37);model,_=import_parent(parent())
        with torch.no_grad():
            for name,p in model.named_parameters():
                if not name.startswith('core.'):p.uniform_(-.03,.03)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as d:
            m=export.export(model,Path(d)/'actor.onnx');graph=export.Graph(Path(d)/'actor.onnx')
            self.assertTrue(export.check_backend(graph)['passed'])
            self.assertEqual(m['outputs'],export.BASE_OUTPUTS+export.GOAL_OUTPUTS)
            self.assertLess(max(m['parity']['errors'].values()),2e-4)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_mask_parity_catches_illegal_probability(self):
        a=dict(policy_logits=np.array([[1.,-np.inf]],np.float32))
        with self.assertRaises(ValueError):export.compare(a,dict(policy_logits=np.array([[1.,0.]],np.float32)),np.array([[True,False]]))

    def test_game_error_does_not_disclose_rejected_private_input(self):
        result=public_error(ValueError('own_hand: location:Stafford; sampled opponent hand'))
        self.assertNotIn('Stafford',result);self.assertNotIn('hand',result)
        self.assertIn('no move published',result)

    def test_only_one_complete_ordered_incumbent_is_reused(self):
        s=SimpleNamespace(workers=[],last_continuation={'root':'pass|card','a':'finish|||||','b':'loan|card','c':'finish|||||'})
        v=dict(seat=0,context=dict(actions_remaining=2));c=[dict(id='pass|card')]
        self.assertEqual(completed_paths(s,v,c,0),[['pass|card','finish|||||'],['loan|card','finish|||||']])
        s.last_continuation.pop('c');self.assertIsNone(completed_paths(s,v,c,0))
        s.last_continuation['c']='finish|||||';self.assertIsNone(completed_paths(s,v,[dict(id='different')],0))

    def test_real_release_rejects_fixture_before_any_writes(self):
        with patch.object(install,'validate',side_effect=ValueError('Fixture package cannot be installed or released')):
            with self.assertRaisesRegex(ValueError,'Fixture'):install.certify('fixture','build','bench','review','out')

    def test_fixture_installer_cannot_target_actual_game(self):
        from brass_ai.learning.contracts import SOURCE_ROOT
        with self.assertRaisesRegex(ValueError,'isolated'):install._sandbox(SOURCE_ROOT)

    def test_install_paths_are_short_and_checked_before_copy(self):
        m=dict(id='a'*64,files=[dict(path='runtime/artifacts/r4-host/System.ComponentModel.DataAnnotations.dll')])
        path=install._bundle_path(SOURCE_ROOT,m)
        self.assertEqual(path.name,'a'*16)
        with patch.object(install.os,'name','nt'):
            with self.assertRaisesRegex(ValueError,'Windows game runtime limit'):
                install._bundle_path(SOURCE_ROOT/('x'*180),m)

    def test_saved_deadline_and_public_text_validation(self):
        p=package();s=sampling.new_save(p,'g','s',[dict(seat=0,archetype='random',tier='easy')])
        s['pending']=dict(key=[0,'CanalEra',2,1,0],remaining_seconds=13)
        s=contracts.seal({k:v for k,v in s.items() if k!='id'})
        with self.assertRaisesRegex(ValueError,'whole-turn allowance'):sampling.restore(p,s,'g')
        s['pending']=None;s['seats'][0]['ui']=dict(private_goal='opponent hand')
        s=contracts.seal({k:v for k,v in s.items() if k!='id'})
        with self.assertRaisesRegex(ValueError,'public display'):sampling.restore(p,s,'g')

    def test_backend_recommendation_uses_completed_work_not_memory(self):
        rows=lambda seconds,n:[dict(status='ok',seconds=seconds,metrics=dict(completed=n))]
        r=dict(fixture=True,results=[dict(backend='ort-cpu',available=True,records=rows(2,20)),
            dict(backend='openvino-gpu',available=True,records=rows(2,8)),dict(backend='unavailable',available=False)])
        self.assertEqual(benchmark.recommendation(r)['backend'],'ort-cpu')
        self.assertIn('Fixture',benchmark.recommendation(r)['status'])
        r['results'][0]['records'].append(dict(status='cancelled',interrupt='cancel',seconds=.1,metrics=dict(completed=0)))
        self.assertEqual(benchmark.recommendation(r)['backend'],'ort-cpu')

if __name__=='__main__':unittest.main()
