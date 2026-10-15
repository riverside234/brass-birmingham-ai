import copy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
from jsonschema.exceptions import ValidationError

from fe_fixtures import Factory,benchmark_positions
from fixtures import rotate
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.feature_engineering.context import DerivedContextBuilder
from brass_ai.feature_engineering.contracts import STATE_FIELDS,ACTION_FIELDS
from brass_ai.feature_engineering.effects import action_effects
from brass_ai.feature_engineering.finance import financial,provisional_order
from brass_ai.feature_engineering.rules import BoardFacts,market_buy,market_sell,income_level


class NativeEffectsTests(unittest.TestCase):
    def setUp(self):self.f=Factory();self.addCleanup(self.f.close)

    def compare(self,kind,**fields):
        before=self.f.view();choice=next(c for c in self.f.host.choices() if c['kind']==kind and all(c.get(k)==v for k,v in fields.items()))
        effect=action_effects(BoardFacts(before),choice);actor=before['seat']
        self.f.context=self.f.host.apply(self.f.context,choice);after=self.f.host.observe(actor)
        # These probes stop before automatic turn/round advancement.
        spent=after['players'][actor]['spent']-before['players'][actor]['spent']
        proceeds=after['players'][actor]['cash']-before['players'][actor]['cash']+spent
        self.assertEqual(effect['cash_spent'].value,spent,(kind,fields))
        self.assertEqual(effect['cash_proceeds'].value,proceeds,(kind,fields))
        self.assertEqual(effect['cash_after'].value,after['players'][actor]['cash'])
        old_tiles={s['tile']['id']:s['tile'] for s in before['slots'] if s.get('tile')}
        for relative in range(4):
            seat=(actor+relative)%4;b=before['players'][seat];a=after['players'][seat]
            self.assertEqual(effect[f'effects.{relative}.income_marker_delta'].value,a['income_marker']-b['income_marker'],(kind,seat))
            self.assertEqual(effect[f'effects.{relative}.income_delta'].value,a['income']-b['income'],(kind,seat))
            self.assertEqual(effect[f'effects.{relative}.immediate_vp'].value,a['vp']-b['vp'])
            flipped=sum(s['tile']['seat']==seat and s['tile']['flipped'] and not old_tiles.get(s['tile']['id'],{}).get('flipped',False) for s in after['slots'] if s.get('tile'))
            self.assertEqual(effect[f'effects.{relative}.flips'].value,flipped)
            secured=lambda v:sum(s['tile']['vp'] for s in v['slots'] if s.get('tile',{}).get('seat')==seat and s['tile']['flipped'])
            self.assertEqual(effect[f'effects.{relative}.secured_era_vp_delta'].value,secured(after)-secured(before))
        return before,effect,after

    def test_loan_nonlinear_marker_and_actual_income(self):
        v=self.f.view();v['players'][v['seat']].update(cash=20,income_marker=34);self.f.load(v)
        before,e,after=self.compare('loan')
        self.assertEqual(e['cash_proceeds'].value,30);self.assertEqual(e['effects.0.income_delta'].value,-3)
        self.assertNotEqual(e['effects.0.income_marker_delta'].value,-3)

    def test_iron_market_quotes_and_second_develop_do_not_repeat_first_payment(self):
        for stock in (0,1,8,10):
            v=self.f.reset();v['iron_market']=stock;v['players'][v['seat']]['cash']=40;self.f.load(v)
            _,first,_=self.compare('develop',industry='CottonMill')
            self.assertEqual(first['cash_spent'].value,0);self.assertEqual(first['iron_used'].value,0)
            self.assertEqual(first['portion_cash_spent'].value,market_buy('iron',stock))
            self.compare('iron',target='market')
            _,second,_=self.compare('develop',industry='CottonMill')
            self.assertEqual(second['portion_cash_spent'].value,market_buy('iron',max(0,stock-1)))
            self.assertEqual(second['develop.removed_tiles'].value,1)

    def test_coal_build_gross_cost_proceeds_flip_and_deferred_vp(self):
        v=self.f.view();v['coal_market']=0;v['players'][v['seat']]['cash']=30
        self.f.hand(v,['location:Coventry']);self.f.connect(v,'CanalBirminghamCoventry');self.f.connect(v,'CanalBirminghamOxford');self.f.load(v)
        _,e,_=self.compare('build',target='Coventry:1',industry='CoalMine')
        self.assertEqual((e['cash_spent'].value,e['cash_proceeds'].value),(5,market_sell('coal',0,2)))
        self.assertEqual(e['effects.0.flips'].value,1);self.assertEqual(e['effects.0.immediate_vp'].value,0)
        self.assertEqual(e['effects.0.secured_era_vp_delta'].value,1)

    def test_last_resource_payment_flips_provider_and_triggers_build_resale(self):
        v=self.f.view();seat=v['seat'];v['players'][seat]['cash']=40;v['iron_market']=0
        self.f.hand(v,['location:Coalbrookdale'])
        self.f.put(v,'Coalbrookdale:2','CoalMine',seat=(seat+1)%4,resource=1)
        mat=next(m for m in v['players'][seat]['mat'] if m['type']=='IronWorks');mat['tiles']=[t for t in mat['tiles'] if t['level']>=4]
        self.f.load(v);_,root,_=self.compare('build',target='Coalbrookdale:1',industry='IronWorks')
        self.assertEqual(root['cash_proceeds'].value,0);self.assertEqual(root['portion_cash_proceeds'].value,24)
        _,payment,_=self.compare('coal',target='Coalbrookdale:2')
        self.assertEqual(payment['cash_spent'].value,0);self.assertEqual(payment['cash_proceeds'].value,24)
        self.assertEqual(payment['portion_cash_spent'].value,0)
        self.assertEqual(payment['effects.0.flips'].value,1);self.assertEqual(payment['effects.1.flips'].value,1)

    def test_second_rail_is_ten_plus_its_own_coal_no_first_link_recharge(self):
        v=self.f.view();self.f.rail(v);v['players'][v['seat']]['cash']=40
        self.f.put(v,'Birmingham:1','Manufacturer',level=2,flipped=True)
        self.f.put(v,'Coventry:1','CoalMine',level=2,seat=(v['seat']+1)%4,resource=1)
        self.f.put(v,'Dudley:0','CoalMine',level=3,seat=(v['seat']+1)%4,resource=3)
        self.f.put(v,'Central:0','Brewery',level=2,resource=2)
        self.f.load(v);self.compare('network',target='RailBirminghamCoventry');self.compare('coal',target='Coventry:1')
        _,e,_=self.compare('network',target='RailBirminghamDudley')
        self.assertEqual(e['cash_spent'].value,10);self.assertEqual(e['portion_cash_spent'].value,10)
        self.assertEqual((e['portion_coal'].value,e['portion_beer'].value),(1,1))
        self.compare('beer',target='Central:0');self.compare('coal',target='Dudley:0')

    def test_multi_coal_part_quote_matches_actual_completion_and_overbuild_exposure(self):
        v=self.f.view();self.f.rail(v);seat=v['seat'];v['players'][seat]['cash']=100;v['coal_market']=1
        self.f.hand(v,['location:Coventry']);self.f.put(v,'Coventry:0','Pottery',level=2,flipped=True)
        self.f.put(v,'Coventry:1','CoalMine',level=2,seat=(seat+1)%4,resource=1)
        self.f.connect(v,'RailBirminghamCoventry');self.f.connect(v,'RailBirminghamOxford');self.f.load(v)
        before,e,_=self.compare('build',target='Coventry:0',industry='Pottery')
        self.assertEqual(e['portion_cash_spent'].value,29);self.assertEqual(e['effects.0.secured_era_vp_delta'].value,-1)
        self.compare('coal',target='Coventry:1');self.compare('coal',target='market')
        after=self.f.host.observe(seat)
        self.assertEqual(after['players'][seat]['spent']-before['players'][seat]['spent'],e['portion_cash_spent'].value)
        self.assertEqual(after['players'][seat]['cash'],e['portion_cash_after'].value)

    def test_merchant_bonus_free_develop_and_sale_completion(self):
        v=self.f.view();self.f.put(v,'Worcester:0','CottonMill');self.f.connect(v,'CanalWorcesterGloucester')
        for m in v['merchants']:
            if m['id'].startswith('Gloucester'):m.update(types=['CottonMill'],beer=1)
        self.f.load(v);self.compare('sell',target='Worcester:0',merchant='Gloucester:0')
        _,e,_=self.compare('beer',target='merchant:Gloucester:0')
        self.assertEqual(e['merchant_free_develop'].value,1);self.assertEqual(e['effects.0.flips'].value,1)
        _,e,_=self.compare('bonus_develop',industry='Brewery')
        self.assertEqual((e['iron_used'].value,e['portion_iron'].value,e['portion_cash_spent'].value),(0,0,0))
        self.assertEqual(e['develop.free'].value,1)

    def test_vp_money_and_income_merchant_bonuses(self):
        for region,bonus,expected in (('Oxford','Income',2),('Warrington','Money',5),('Shrewsbury','VP',4),('Nottingham','VP',3)):
            v=self.f.reset();self.f.put(v,'Worcester:0','CottonMill')
            for link in v['links']:
                if link['era']=='Canal':link['owner']=v['seat']
            merchant=next(m for m in v['merchants'] if m['id'].startswith(region+':'));merchant.update(types=['CottonMill'],beer=1)
            self.f.load(v);self.compare('sell',target='Worcester:0',merchant=merchant['id'])
            self.compare('beer',target='merchant:'+merchant['id'])

    def test_optional_money_merchant_is_bounded_not_assumed(self):
        v=self.f.view();self.f.put(v,'Worcester:0','CottonMill');self.f.put(v,'Central:0','Brewery',resource=1)
        for link in v['links']:
            if link['era']=='Canal':link['owner']=v['seat']
        merchant=next(m for m in v['merchants'] if m['id'].startswith('Warrington:'));merchant.update(types=['CottonMill'],beer=1)
        self.f.load(v);before,e,after=self.compare('sell',target='Worcester:0',merchant=merchant['id'])
        self.assertFalse(e['portion_cash_proceeds'].known);self.assertEqual((e['portion_cash_proceeds'].lower,e['portion_cash_proceeds'].upper),(0,5))
        self.assertFalse(e['portion_projected_cash'].known)
        self.compare('beer',target='Central:0')

    def test_two_beer_sale_retains_forced_future_merchant_proceeds(self):
        v=self.f.view();self.f.put(v,'Coventry:0','Pottery',level=3);self.f.put(v,'Central:0','Brewery',resource=1)
        for link in v['links']:
            if link['era']=='Canal':link['owner']=v['seat']
        merchant=next(m for m in v['merchants'] if m['id'].startswith('Warrington:'));merchant.update(types=['Pottery'],beer=1)
        self.f.load(v);before,root,_=self.compare('sell',target='Coventry:0',merchant=merchant['id'])
        self.assertEqual(root['portion_cash_proceeds'].value,5)
        _,first,_=self.compare('beer',target='Central:0')
        self.assertEqual(first['cash_proceeds'].value,0);self.assertEqual(first['portion_cash_proceeds'].value,5)
        self.compare('beer',target='merchant:'+merchant['id']);after=self.f.host.observe(before['seat'])
        self.assertEqual(after['players'][before['seat']]['cash'],root['portion_cash_after'].value)
        self.assertEqual(after['players'][before['seat']]['cash'],first['portion_cash_after'].value)
        for effects in (root,first):
            projected=after['players'][before['seat']]['cash']+after['players'][before['seat']]['income']
            self.assertLessEqual(effects['portion_projected_cash'].lower,projected)
            self.assertGreaterEqual(effects['portion_projected_cash'].upper,projected)

    def test_partial_debt_liquidation_and_no_era_points_double_count(self):
        v=self.f.view();debtor=(v['seat']+1)%4;v['context'].update(turn=4,actions_remaining=1)
        v['players'][debtor].update(cash=0,income_marker=0,vp=12)
        self.f.put(v,'Dudley:0','CoalMine',seat=debtor,resource=2);self.f.put(v,'Dudley:1','IronWorks',seat=debtor,resource=4)
        self.f.load(v);self.f.choose('pass');self.f.choose('finish')
        _,e,_=self.compare('liquidate',target='Dudley:0')
        self.assertEqual(e['cash_proceeds'].value,2);self.assertEqual(e['debt_after'].value,8)
        self.assertFalse(e['projected_cash'].known)

    def test_settlement_and_stable_order_match_native_next_round(self):
        v=self.f.view();actor=v['seat'];v['context'].update(round=2,turn=4,actions_remaining=1)
        v['order']=[s for s in v['order'] if s!=actor]+[actor]
        for i,p in enumerate(v['players']):p.update(cash=30,income_marker=7+2*i,spent=[4,1,1,3][i])
        self.f.load(v);self.f.choose('pass');before=self.f.view();finance=financial(before);order=provisional_order(before)
        self.f.choose('finish');after=self.f.host.observe(actor)
        self.assertEqual(after['context']['round'],3)
        for relative in range(4):
            seat=(actor+relative)%4
            self.assertEqual(finance[f'finance.{relative}.projected_cash'].value,after['players'][seat]['cash'])
            self.assertEqual(order[f'order.{relative}.position'].value,after['order'].index(seat))


class ContextCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.positions=benchmark_positions()

    def test_all_six_groups_are_complete_finite_and_choice_aligned(self):
        builder=DerivedContextBuilder()
        for row in self.positions:
            before=copy.deepcopy(row);out=builder.build(row['view'],row['choices'])
            self.assertEqual(out.choice_ids,tuple(c['id'] for c in row['choices']))
            self.assertEqual(len(out.state),len(STATE_FIELDS));self.assertEqual(len(out.actions),len(row['choices']))
            self.assertTrue(np.isfinite(out.state_tensor()).all());self.assertTrue(np.isfinite(out.action_tensors()).all())
            self.assertEqual(row,before)
            self.assertIs(builder.build(row['view'],row['choices']),out)
            uncached=DerivedContextBuilder(cache_size=0).build(row['view'],row['choices'])
            self.assertEqual(out.state,uncached.state);self.assertEqual(out.actions,uncached.actions)
        self.assertEqual(builder.report()['hits'],len(self.positions))

    def test_identical_semantics_different_cash_market_pending_and_observer_do_not_alias(self):
        row=self.positions[0];view=copy.deepcopy(row['view']);choices=copy.deepcopy(row['choices']);builder=DerivedContextBuilder()
        first=builder.build(view,choices)
        view['players'][view['seat']]['cash']+=10
        second=builder.build(view,choices);self.assertNotEqual(first.identity,second.identity)
        index=next(i for i,c in enumerate(choices) if c['kind']=='loan')
        self.assertEqual(second.actions[index]['cash_after'].value-first.actions[index]['cash_after'].value,10)
        view['iron_market']=0;third=builder.build(view,choices);self.assertNotEqual(second.identity,third.identity)
        rotated=builder.build(rotate(view,1),choices);self.assertNotEqual(rotated.identity,third.identity)
        self.assertEqual(rotated.state,third.state)
        with self.assertRaises(TypeError):second.actions[index]['cash_after']=None

    def test_revision_pending_beer_and_stack_changes_invalidate_context(self):
        row=next(r for r in self.positions if r['name']=='rail-second-payment')
        view=copy.deepcopy(row['view']);choices=row['choices'];builder=DerivedContextBuilder()
        original=builder.build(view,choices)
        view['context']['revision']+=1;revision=builder.build(view,choices)
        self.assertNotEqual(original.identity,revision.identity);self.assertEqual(original.actions,revision.actions)
        view['pending']['beer']+=1;pending=builder.build(view,choices)
        self.assertNotEqual(revision.identity,pending.identity)
        self.assertEqual(pending.actions[0]['portion_beer'].value,revision.actions[0]['portion_beer'].value+1)
        next(s for s in view['slots'] if s['id']=='Central:0')['tile']['resource']=1
        beer=builder.build(view,choices);self.assertNotEqual(beer.identity,pending.identity)
        index=next(i for i,c in enumerate(choices) if c['target']=='Central:0')
        self.assertEqual(beer.actions[index]['effects.0.flips'].value,1)
        mat=next(m for m in view['players'][view['seat']]['mat'] if m['type']=='CottonMill');mat['tiles'].pop(0)
        stack=builder.build(view,choices);self.assertNotEqual(stack.identity,beer.identity)
        key='development.CottonMill.current_target.removals'
        self.assertEqual(stack.state[key].value,beer.state[key].value-1)

    def test_cache_is_bounded_static_and_dynamic_separate(self):
        builder=DerivedContextBuilder(cache_size=2,static_cache_size=3)
        for row in self.positions:builder.build(row['view'],row['choices'])
        self.assertEqual(len(builder.cache),2);self.assertLessEqual(len(builder.static),3)
        self.assertLess(builder.dynamic_calculations,sum(len(r['choices']) for r in self.positions))

    def test_context_and_private_input_boundary_and_deadline(self):
        row=self.positions[0];v=copy.deepcopy(row['view']);c=row['choices'];builder=DerivedContextBuilder()
        for extra in ('checkpoint','opponent_hands','deck'):
            bad=copy.deepcopy(v);bad[extra]={}
            with self.assertRaises(Exception):builder.build(bad,c)
        bad=copy.deepcopy(v);bad['seat']=(bad['seat']+1)%4
        with self.assertRaises(Exception):builder.build(bad,c)
        bad=copy.deepcopy(v);bad['players'][0]['mat'].pop()
        with self.assertRaises(ValidationError):builder.build(bad,c)
        with self.assertRaisesRegex(ValueError,'deadline'):builder.build(v,c,deadline=time.monotonic()-1)
        first=builder.build(v,c);other=builder.build(v,c,style=[1,0,0,0]);self.assertNotEqual(first.identity,other.identity)
        self.assertEqual(first.state,other.state)

    def test_history_is_validated_but_not_reserialized_for_derived_identity(self):
        from brass_ai.learning.contracts import digest
        row=self.positions[0];builder=DerivedContextBuilder()
        with patch('brass_ai.feature_engineering.context.digest',wraps=digest) as hashed:
            builder.build(row['view'],row['choices'])
        self.assertNotIn('view',hashed.call_args.args[0])
        self.assertEqual(len(hashed.call_args.args[0]['input']),64)


if __name__=='__main__':unittest.main()
