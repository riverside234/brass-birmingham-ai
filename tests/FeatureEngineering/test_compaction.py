"""FC2: one semantic pass (ten positions) and one tiny model/debug batch."""
import copy
import json
import time
import unittest
from dataclasses import replace

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT,FEATURE_SCHEMA,FEATURE_VERSION,read_json
from brass_ai.learning.features import Encoder,check_encoded
from brass_ai.learning.batching import collate
from brass_ai.learning.model import FourSeatNet,ModelConfig,ResidualBlock,dense_multiply_adds
from brass_ai.learning.state_layout import state_layout,build_state_layout
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.personality.model import PersonalityNet,PersonalityConfig
from brass_ai.personality.laptop.export import validate_metadata
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.utility import reward_spec
from brass_ai.feature_engineering.contracts import (Measure,STATE_FIELDS,ACTION_FIELDS,STATE_INPUTS,ACTION_INPUTS,
    STATE_OMISSIONS,ACTION_OMISSIONS,tensorize,specification)
from brass_ai.feature_engineering.context import DerivedContextBuilder
from brass_ai.feature_engineering.industry import tile_properties
from brass_ai.feature_engineering.rules import BoardFacts
from fixtures import rotate
from fe_fixtures import benchmark_positions,Factory


def positions():
    keep={'canal-opening','develop-continuation','sale-payment','free-development-bonus',
          'rail-action','rail-second-payment','pending-debt','partial-debt'}
    rows=[r for r in benchmark_positions() if r['name'] in keep]
    # Source-authored optional money bonus: either owned beer or merchant beer.
    f=Factory()
    try:
        view=f.view();f.put(view,'Worcester:0','CottonMill');f.put(view,'SouthWest:0','Brewery',resource=1)
        for link in view['links']:
            if link['era']=='Canal':link['owner']=view['seat']
        next(m for m in view['merchants'] if m['id']=='Warrington:0').update(types=['CottonMill'],beer=1)
        view,choices=f.load(view)
        rows.append(dict(name='optional-money-proceeds',view=view,choices=choices))
        view=copy.deepcopy(next(r['view'] for r in rows if r['name']=='rail-action'))
        view['context'].update(round=8,actions_remaining=1)
        view,choices=f.load(view)
        rows.append(dict(name='final-rail',view=view,choices=choices))
    finally:f.close()
    return rows


class CompactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.rows=positions()

    def test_semantics_and_encoder_boundary(self):
        began=time.monotonic();reference=Encoder();fast=PreparedEncoder(cache_size=2)
        layout=state_layout();schema=reference.schema
        self.assertEqual(schema['derived'],json.loads(json.dumps(specification())))
        self.assertEqual(layout,build_state_layout(schema))
        self.assertEqual(sorted(layout.permutation),list(range(reference.state_dim)))
        self.assertEqual(tuple(len(g.indices) for g in layout.groups),(31,218,109,1314,1247,197,12))
        self.assertEqual((reference.state_dim,reference.action_dim),(3128,327))
        self.assertEqual((len(STATE_INPUTS),len(ACTION_INPUTS)),(195,41))
        self.assertEqual([Measure.exact(0).compact_tensor(100),Measure().compact_tensor(100),Measure.absent().compact_tensor(100)],
                         [(0.,1.,1.),(0.,0.,1.),(0.,0.,0.)])
        self.assertEqual(Measure.exact(-5).compact_tensor(100),(-.05,1.,1.))
        for value in (Measure.interval(1,3),Measure.interval(None,2),Measure.interval(-2,None)):
            with self.assertRaisesRegex(ValueError,'cannot discard'):value.compact_tensor(100)
        segments={s.name:s for s in layout.segments};all_kinds=set();bounded=set();candidate_count=0
        for row in self.rows:
            view,choices=row['view'],row['choices'];original=copy.deepcopy(row)
            all_kinds.update(c['kind'] for c in choices);candidate_count+=len(choices)
            style=[.1,.2,-.1,0];intent=[0,.1,.2,.3,.4,.5,.6,.7]
            a=reference.encode(view,choices,style,intent);b=fast.encode(view,choices,style,intent)
            np.testing.assert_array_equal(a.state,b.state);np.testing.assert_array_equal(a.actions,b.actions)
            self.assertEqual(a.choice_ids,tuple(c['id'] for c in choices));self.assertIs(fast.encode(view,choices,style,intent),b)
            np.testing.assert_array_equal(a.stable_to_relative,[(s-view['seat'])%4 for s in range(4)])
            np.testing.assert_array_equal(a.state[-12:],np.asarray(style+intent,np.float32))
            np.testing.assert_array_equal(a.actions[:,:184],np.asarray([reference.choice_features(c) for c in choices],np.float32))
            rotated=fast.encode(rotate(view,1),choices,style,intent)
            np.testing.assert_array_equal(a.state,rotated.state);np.testing.assert_array_equal(a.actions,rotated.actions)
            context=reference.derived.build(view,choices);facts=context.state
            self.assertEqual(set(facts),{f.name for f in STATE_FIELDS})
            self.assertEqual(len(facts),326)
            for field in STATE_INPUTS:
                s=segments[field.name]
                np.testing.assert_array_equal(a.state[s.start:s.stop],np.asarray(field.tensor(facts[field.fact]),np.float32))
            for seat in range(4):
                self.assertEqual(facts[f'finance.{seat}.settlement'],facts['finance.0.settlement'])
                self.assertEqual(facts[f'finance.{seat}.public_debt'].value,view['players'][(view['seat']+seat)%4]['debt'])
            self.assertEqual(facts['order.0.spending_difference'],Measure.exact(0))
            self.assertEqual(facts['opportunities.main_action_available'],Measure.exact(view['context']['phase']=='action'))
            for kind in schema['kinds']:
                targets=facts[f'opportunities.{kind}.targets'];available=facts[f'opportunities.{kind}.available']
                self.assertEqual(available,Measure.exact(targets.value>0) if targets.known else Measure() if targets.applicable else Measure.absent())
            for mat in view['players'][view['seat']]['mat']:
                kind=mat['type'];head=mat['tiles'][0] if mat['tiles'] else None
                prefix=f'development.{kind}.current.'
                self.assertEqual(facts[prefix+'present'],Measure.exact(head is not None))
                for name,value in tile_properties(head).items():self.assertEqual(facts[prefix+name],value)
                for horizon in ('current','next_era'):
                    prefix=f'development.{kind}.{horizon}_target.';present=facts[prefix+'present'];reachable=facts[prefix+'reachable']
                    self.assertEqual(facts[prefix+'already_unlocked'],Measure.exact(0) if present.applicable else Measure.absent())
                    self.assertEqual(facts[prefix+'blocked'],Measure.exact(present.value and not reachable.value) if present.applicable else Measure.absent())
                    self.assertEqual(facts[prefix+'survives_canal'],Measure.exact(1) if present.value else Measure.absent())
                    for era in ('canal','rail'):
                        target=facts[prefix+era+'_eligible'];current=facts[f'development.{kind}.current.{era}_eligible']
                        expected=Measure.exact(target.value-current.value) if target.known else Measure.absent()
                        self.assertEqual(facts[prefix+'delta_'+era+'_eligible'],expected)
            for index,(choice,effect) in enumerate(zip(choices,context.actions)):
                self.assertEqual(set(effect),{f.name for f in ACTION_FIELDS})
                offset=184
                for field in ACTION_INPUTS:
                    np.testing.assert_array_equal(a.actions[index,offset:offset+len(field.components)],np.asarray(field.tensor(effect[field.fact]),np.float32))
                    offset+=len(field.components)
                    if not effect[field.fact].known and (effect[field.fact].lower_known or effect[field.fact].upper_known):bounded.add(field.fact)
                for kind in ('coal','iron','beer'):self.assertEqual(effect[kind+'_used'],Measure.exact(choice['kind']==kind))
                for seat in range(1,4):self.assertEqual(effect[f'effects.{seat}.immediate_vp'],Measure.exact(0))
                develop=choice['kind'] in ('develop','bonus_develop')
                self.assertEqual(effect['develop.removed_tiles'],Measure.exact(1) if develop else Measure.absent())
                self.assertEqual(effect['develop.free'],Measure.exact(choice['kind']=='bonus_develop') if develop else Measure.absent())
                level=effect['develop.result.level']
                self.assertEqual(effect['develop.result.survives_canal'],Measure.exact(level.value>1) if level.known else Measure.absent())
                for prefix in ('','next_era_'):
                    after=effect[f'develop.{prefix}removals_after'];before=effect[f'develop.{prefix}removals_before']
                    self.assertEqual(before,Measure.exact(after.value+1) if after.known else after)
            for slot in view['slots']:
                s=segments['slot.'+slot['id']+'.owner'];self.assertEqual(a.state[s.start:s.stop].sum(),int(slot.get('tile') is not None))
            self.assertEqual(row,original)
        self.assertIn('portion_cash_proceeds',bounded);self.assertIn('portion_cash_after',bounded)
        opening=next(r for r in self.rows if r['name']=='canal-opening')
        context=DerivedContextBuilder().build(opening['view'],opening['choices'])
        self.assertFalse(context.state['realization.card_supported_actions'].known)
        # An unconnected legal-region quote has a genuine one-sided bound.
        quote=BoardFacts(opening['view']).resource_cost(1,0,('Worcester',))
        self.assertTrue(quote.lower_known);self.assertFalse(quote.upper_known)
        action=dict(context.actions[0]);action['portion_cash_spent']=quote
        tensorize(action,ACTION_FIELDS)  # Retains, never compacts, this cost bound.
        action['cash_spent']=quote
        with self.assertRaisesRegex(ValueError,'cannot discard'):tensorize(action,ACTION_FIELDS)
        for missing in (dict(list(context.state.items())[1:]),dict(context.state,unexpected=Measure.exact(1))):
            with self.assertRaisesRegex(ValueError,'Incomplete'):tensorize(missing,STATE_FIELDS)
        # Values omitted from neural input remain part of supported-profile admission.
        for field in ('slot','merchant','hidden','tile'):
            row=copy.deepcopy(next(r for r in self.rows if r['name']=='rail-action'));view=row['view']
            if field=='slot':view['slots'][0]['types']=['IronWorks']
            elif field=='merchant':view['merchants'][0]['bonus']='Money'
            elif field=='hidden':view['players'][0]['hidden_discards']=1
            else:next(s['tile'] for s in view['slots'] if s.get('tile'))['coal']+=1
            for encoder in (reference,fast):
                with self.assertRaisesRegex(ValueError,'Compacted features require'):encoder.encode(view,row['choices'])
        stale=replace(a,feature_version='permitted-features/2.0.0')
        with self.assertRaisesRegex(ValueError,'mixed-version'):check_encoded(stale)
        print(dict(check='semantic/encoder',positions=len(self.rows),candidate_rows=candidate_count,kinds=sorted(all_kinds),
                   genuine_bounded_actions=sorted(bounded),seconds=time.monotonic()-began))

    def test_tiny_model_batch_and_admission(self):
        encoder=PreparedEncoder();rows=[encoder.encode(r['view'],r['choices']) for r in self.rows[:2]]
        inputs=collate(rows);model=FourSeatNet();count=[]
        hook=model.state_encoder.register_forward_hook(lambda *_:count.append(1))
        outputs=model(*inputs);hook.remove();self.assertEqual(len(count),1)
        logits,*heads=outputs;mask=inputs[2]
        self.assertTrue(torch.isfinite(logits[mask]).all());self.assertTrue(torch.isneginf(logits[~mask]).all())
        self.assertTrue(all(torch.isfinite(h).all() and h.shape==(2,4) for h in heads))
        self.assertTrue(torch.allclose(heads[0].sum(1),torch.ones(2)))
        loss=logits[mask].square().mean()+sum(h.square().mean() for h in heads);loss.backward()
        for p in model.state_encoder.projections:self.assertGreater(p[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.state_encoder.fusion[0].weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(model.state_encoder.fusion[0].in_features,556)
        self.assertEqual((model.action_encoder[0].in_features,model.action_encoder[0].out_features),(327,128))
        self.assertEqual(sum(isinstance(m,ResidualBlock) for m in model.modules()),6)
        for block in model.state_encoder.backbone[:-1]:self.assertEqual(block.branch[-1].weight.count_nonzero(),0)
        reward=reward_spec(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json'))
        utility=UtilityNet(utility_config(model.config,reward));actor=PersonalityNet(PersonalityConfig(utility.config.to_dict()))
        remaining=torch.full((2,4),reward['insolvency']['max_cost'])
        with torch.no_grad():
            u=utility(*inputs,remaining=remaining)
            from brass_ai.personality.contracts import CONTEXT_DIM
            p=actor(*inputs,remaining=remaining,personality=torch.zeros(2,CONTEXT_DIM))
        self.assertTrue(torch.isfinite(u[-1]).all());self.assertTrue(torch.isfinite(p['remaining_utility_4']).all())
        self.assertEqual([m.parameter_count for m in (model,utility,actor)],[1904445,1914465,1980298])
        self.assertEqual(dense_multiply_adds(model)['fixed_per_position'],1797760)
        self.assertEqual(dense_multiply_adds(model)['per_padded_candidate'],97280)
        # Metadata-only mismatches; no old architecture/checkpoint is constructed.
        for key,value in (('architecture','brass-four-seat-mlp-resnet/7.0.0'),('state_dim',5489),
                          ('action_dim',548),('group_layout','f'*64),('projection_widths',[64,32,48,160,192,48])):
            bad=dict(model.config.to_dict(),**{key:value})
            with self.assertRaises(ValueError):ModelConfig.from_dict(bad)
            with self.assertRaises(ValueError):validate_metadata(dict(config=dict(utility.config.to_dict(),**{key:value})))
        altered=copy.deepcopy(encoder.schema)
        altered['derived']['state'][0]['group']='board_resources'
        self.assertNotEqual(state_layout().identity,build_state_layout(altered).identity)
        print(dict(check='model',batch_positions=2,padded_candidates=inputs[1].shape[1],base=model.parameter_count,utility=utility.parameter_count,personality=actor.parameter_count))


if __name__=='__main__':unittest.main()
