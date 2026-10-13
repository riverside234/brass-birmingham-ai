"""Cached player mats retain the complete permitted-observation schema."""
import copy
import unittest
from unittest.mock import patch

from fixtures import position
from brass_ai.learning.contracts import digest
from brass_ai.recovery.inputs import InputBoundary
from brass_ai.runtime_contracts import validate, validator


class BoundaryPerformanceTests(unittest.TestCase):
    def test_player_counters_reuse_validated_mats_without_mutating_schema(self):
        view,choices=position();boundary=InputBoundary()
        schema=copy.deepcopy(validator('observation').schema)
        self.assertEqual(boundary.check(view,choices),digest(dict(view=view,choices=choices)))
        count=boundary.checked_parts
        cls=type(validator('observation'));descend=cls.descend;revisited=[]
        def tracked(checker,instance,schema,*args,**kwargs):
            if schema=={'$ref':'#/$defs/tile'}:revisited.append(instance)
            return descend(checker,instance,schema,*args,**kwargs)
        with patch.object(cls,'descend',tracked):
            for i in range(1,21):
                view['players'][0]['cash']+=1
                boundary.check(view,choices)
                # Only the newly changed player is novel. None of its tile stacks
                # should be traversed again because a scalar counter changed.
                self.assertEqual(boundary.checked_parts,count+i)
        self.assertEqual(revisited,[],'A counter change traversed unchanged tile schemas')
        self.assertEqual(validator('observation').schema,schema)

    def test_nested_changes_match_the_original_schema_after_cache_hits(self):
        view,choices=position();boundary=InputBoundary();boundary.check(view,choices)
        mutations=[
            lambda p:p.update(cash=-1),
            lambda p:p.update(cash=True),
            lambda p:p.update(extra_private_field=[]),
            lambda p:p.pop('mat'),
            lambda p:p.update(mat={}),
            lambda p:p['mat'].pop(),
            lambda p:p['mat'].append(copy.deepcopy(p['mat'][0])),
            lambda p:p['mat'].__setitem__(0,None),
            lambda p:p['mat'][0].update(type='unknown'),
            lambda p:p['mat'][0].update(extra=1),
            lambda p:p['mat'][0].pop('tiles'),
            lambda p:p['mat'][0].update(tiles=None),
            lambda p:p['mat'][0]['tiles'].append({}),
            lambda p:p['mat'][0]['tiles'][0].update(level=9),
            lambda p:p['mat'][0]['tiles'][0].update(level=True),
            lambda p:p['mat'][0]['tiles'][0].update(seat=4),
            lambda p:p['mat'][0]['tiles'][0].update(flipped=1),
            lambda p:p['mat'][0]['tiles'][0].update(coal=-1),
            lambda p:p['mat'][0]['tiles'][0].update(unpermitted=1),
            lambda p:p['mat'][0]['tiles'][0].pop('cost'),
        ]
        for i,mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                changed=copy.deepcopy(view);changed['players'][0]['cash']+=1
                mutate(changed['players'][0])
                with self.assertRaises(Exception):validate('observation',changed)
                with self.assertRaises(Exception):boundary.check(changed,choices)
                self.assertEqual(boundary.check(view,choices),digest(dict(view=view,choices=choices)))

    def test_valid_mat_updates_and_eviction_still_cross_all_constraints(self):
        view,choices=position();boundary=InputBoundary(frames=1,events=2,parts=3)
        for i in range(10):
            view['players'][0]['cash']+=1
            tile=view['players'][0]['mat'][0]['tiles'][0]
            tile['resource']=i;tile['flipped']=bool(i%2)
            validate('observation',view);validate('choices',choices)
            self.assertEqual(boundary.check(view,choices),digest(dict(view=view,choices=choices)))
            self.assertLessEqual(len(boundary.parts),3)
        tile['resource']=-1
        with self.assertRaises(Exception):boundary.check(view,choices)
        tile['resource']=0
        boundary.check(view,choices)
        view['players'][0]['mat'][0]['tiles'].clear()
        validate('observation',view);boundary.check(view,choices)


if __name__=='__main__':unittest.main()
