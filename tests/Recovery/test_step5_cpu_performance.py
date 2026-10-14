"""Faster input work retains detached snapshots and strict history validation."""
import copy
import time
import unittest
from collections import UserDict
from unittest.mock import patch

from fixtures import position
from brass_ai.learning.batching import Request, _snapshot_input
from brass_ai.learning.contracts import digest
from brass_ai.recovery import inputs
from brass_ai.runtime_contracts import validate


class InputPerformanceTests(unittest.TestCase):
    def history(self):
        view, choices = position()
        view['history'] = [dict(seat=i%4, era='CanalEra',
                                choice=dict(choices[i%3], id=str(i))) for i in range(96)]
        return view, choices

    def test_appending_history_reuses_complete_prefix_chunks(self):
        view, choices = self.history(); boundary = inputs.InputBoundary()
        boundary.check(view, choices)
        view['history'].append(dict(seat=0, era='RailEra', choice=choices[0]))
        visited=[]
        def hashed(value):
            if isinstance(value,dict) and set(value)=={'seat','era','choice'}:visited.append(value)
            return digest(value)
        with patch.object(inputs,'digest',side_effect=hashed):
            actual=boundary.check(view,choices)
        self.assertEqual(visited,[view['history'][-1]])
        self.assertEqual(actual,digest(dict(view=view,choices=choices)))

    def test_mutations_at_every_chunk_boundary_match_original_validation(self):
        view,choices=self.history();boundary=inputs.InputBoundary();boundary.check(view,choices)
        for index in (0,31,32,63,64,95):
            for mutate in (lambda e:e.update(seat=4),lambda e:e.update(hidden_hand=['secret']),
                           lambda e:e['choice'].update(kind='invalid'),lambda e:e.pop('era')):
                bad=copy.deepcopy(view);mutate(bad['history'][index])
                with self.subTest(index=index),self.assertRaises(Exception):validate('observation',bad)
                with self.subTest(index=index),self.assertRaises(Exception):boundary.check(bad,choices)
            changed=copy.deepcopy(view);changed['history'][index]['era']='RailEra'
            validate('observation',changed)
            self.assertEqual(boundary.check(changed,choices),digest(dict(view=changed,choices=choices)))

    def test_history_reordering_and_small_cache_eviction_are_content_checked(self):
        view,choices=self.history();boundary=inputs.InputBoundary(frames=1,events=2,parts=3)
        for history in (view['history'],list(reversed(view['history'])),view['history'][:33],[]):
            view['history']=history
            self.assertEqual(boundary.check(view,choices),digest(dict(view=view,choices=choices)))
            self.assertLessEqual(len(boundary.events),2);self.assertLessEqual(len(boundary.parts),3)
        view['history']=[dict(seat=True,era='CanalEra',choice=choices[0])]
        with self.assertRaises(Exception):boundary.check(view,choices)

    def test_request_snapshot_matches_deepcopy_and_detaches_nested_inputs(self):
        view,choices=self.history();before=copy.deepcopy((view,choices))
        request=Request('fixture','a'*64,view,choices,time.monotonic()+30)
        self.assertEqual((request.view,request.choices),before)
        expected=digest(dict(view=before[0],choices=before[1],style=(0.,)*4,intent=(0.,)*8,model='a'*64))
        self.assertEqual(request.input_digest,expected)
        view['history'][32]['choice']['id']='changed';view['players'][0]['mat'][0]['tiles'].clear()
        choices[0]['card']='changed'
        self.assertEqual((request.view,request.choices),before)
        self.assertEqual(request.digest(),expected)
        request.view['history'][0]['choice']['id']='snapshot mutation'
        self.assertNotEqual(request.digest(),request.input_digest)

    def test_copy_preserves_aliases_cycles_and_non_json_types(self):
        shared={'values':[True,None,1,1.0,'\N{SNOWMAN}']}
        value={'one':shared,'two':shared,'tuple':(shared,), 'custom':UserDict({'items':[1]})}
        result=_snapshot_input(value)
        self.assertEqual(result,copy.deepcopy(value));self.assertIs(result['one'],result['two'])
        self.assertIs(result['tuple'][0],result['one']);self.assertIsNot(result['one'],shared)
        self.assertIsInstance(result['custom'],UserDict);self.assertIsNot(result['custom'],value['custom'])
        cycle=[];cycle.append(cycle);cloned=_snapshot_input(cycle)
        self.assertIs(cloned[0],cloned);self.assertIsNot(cloned,cycle)

    def test_snapshot_does_not_normalize_invalid_container_types(self):
        view,choices=position();view['own_hand']=tuple(view['own_hand'])
        request=Request('fixture','a'*64,view,choices,time.monotonic()+30)
        self.assertIsInstance(request.view['own_hand'],tuple)
        with self.assertRaises(Exception):inputs.InputBoundary().check(request.view,request.choices)


if __name__=='__main__':unittest.main()
