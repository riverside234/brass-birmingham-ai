"""Native and encoded checks for the three required existing-feature semantics."""
import copy
import unittest
import numpy as np

from brass_ai.feature_engineering.contracts import ACTION_INPUTS,STATE_INPUTS
from brass_ai.feature_engineering.context import DerivedContextBuilder
from brass_ai.recovery.encoding import PreparedEncoder
from fixtures import rotate
from fe_fixtures import Factory


class ExistingFeatureSemanticsTests(unittest.TestCase):
    def setUp(self):self.f=Factory();self.addCleanup(self.f.close)

    def encoded_action(self,view,choices,index,name):
        encoder=PreparedEncoder();row=encoder.encode(view,choices)
        i=next(i for i,f in enumerate(ACTION_INPUTS) if f.name==name);field=ACTION_INPUTS[i]
        return row.actions[index,encoder.base_action_dim+sum(len(f.components) for f in ACTION_INPUTS[:i])]*field.scale

    def encoded_state(self,view,choices,name):
        encoder=PreparedEncoder();row=encoder.encode(view,choices)
        i=next(i for i,f in enumerate(STATE_INPUTS) if f.name==name);field=STATE_INPUTS[i]
        return row.state[encoder.base_state_dim-12+sum(len(f.components) for f in STATE_INPUTS[:i])]*field.scale

    def test_last_resource_flips_credit_actual_owner_in_native_and_encoded_seats(self):
        for other in (False,True):
            v=self.f.reset();actor=v['seat'];owner=(actor+2)%4 if other else actor
            self.f.put(v,'Dudley:1','IronWorks',seat=owner,resource=1)
            v['players'][owner]['income_marker']=9;self.f.load(v)
            self.f.choose('develop',industry='CottonMill');v,c=self.f.position()
            index=next(i for i,x in enumerate(c) if x['kind']=='iron' and x['target']=='Dudley:1')
            context=DerivedContextBuilder().build(v,c);effect=context.actions[index]
            self.f.context=self.f.host.apply(self.f.context,c[index]);after=self.f.host.observe(actor)
            for rel in range(4):
                seat=(actor+rel)%4;before=v['players'][seat];p=after['players'][seat]
                expected=dict(flips=int(seat==owner),income_marker_delta=p['income_marker']-before['income_marker'],
                    income_delta=p['income']-before['income'],immediate_vp=0,secured_era_vp_delta=3 if seat==owner else 0)
                for name,value in expected.items():
                    key=f'effects.{rel}.{name}'
                    self.assertEqual(effect[key].value,value)
                    if any(f.name==key for f in ACTION_INPUTS):
                        self.assertAlmostEqual(self.encoded_action(v,c,index,key),value,places=5)
            for shift in range(4):
                rotated=rotate(v,shift)
                np.testing.assert_array_equal(PreparedEncoder().encode(v,c).actions,PreparedEncoder().encode(rotated,c).actions)

    def test_equal_coal_totals_different_connectivity_change_this_build_quote(self):
        quotes=[]
        for connected in (False,True):
            v=self.f.reset();self.f.rail(v);actor=v['seat'];v['players'][actor]['cash']=100
            self.f.hand(v,['location:Coventry']);self.f.put(v,'Coventry:0','Pottery',level=2,flipped=True)
            self.f.put(v,'Dudley:0','CoalMine',level=2,seat=(actor+1)%4,resource=2)
            self.f.connect(v,'RailBirminghamCoventry');self.f.connect(v,'RailBirminghamOxford')
            if connected:self.f.connect(v,'RailBirminghamDudley')
            self.f.load(v);v,c=self.f.position()
            i=next(i for i,x in enumerate(c) if x['kind']=='build' and x['target']=='Coventry:0' and x['industry']=='Pottery')
            quote=self.encoded_action(v,c,i,'portion_cash_spent');before=v['players'][actor]['spent']
            self.f.context=self.f.host.apply(self.f.context,c[i])
            while self.f.context['phase']=='payment':self.f.choose('coal')
            self.assertAlmostEqual(quote,self.f.view()['players'][actor]['spent']-before,places=4);quotes.append(quote)
        self.assertGreater(quotes[0],quotes[1])

    def sale_position(self,*,connected=True,merchant_beer=False,compatible=True):
        v=self.f.reset();actor=v['seat'];self.f.put(v,'Worcester:0','CottonMill');self.f.put(v,'Worcester:1','CottonMill')
        if not merchant_beer:self.f.put(v,'Central:0','Brewery',seat=(actor+1)%4,resource=1)
        if connected:
            for link in v['links']:
                if link['era']=='Canal':link['owner']=actor
        else:self.f.connect(v,'CanalWorcesterGloucester')
        for m in v['merchants']:m.update(beer=0,types=['Pottery'])
        m=next(m for m in v['merchants'] if m['id']=='Gloucester:0')
        m.update(beer=int(merchant_beer),types=['CottonMill'] if compatible else ['Pottery'])
        self.f.load(v);return self.f.position()

    def test_rail_link_coal_quote_and_second_link_beer_follow_this_link_rules(self):
        quotes=[]
        for connected in (False,True):
            v=self.f.reset();self.f.rail(v);actor=v['seat'];v['players'][actor]['cash']=100
            self.f.put(v,'Birmingham:1','Manufacturer',level=2,flipped=True)
            self.f.put(v,'Dudley:0','CoalMine',level=2,seat=(actor+1)%4,resource=1)
            self.f.connect(v,'RailBirminghamOxford')
            if connected:self.f.connect(v,'RailBirminghamDudley')
            self.f.load(v);v,c=self.f.position()
            i=next(i for i,x in enumerate(c) if x['kind']=='network' and x['target']=='RailBirminghamCoventry')
            quotes.append(self.encoded_action(v,c,i,'portion_cash_spent'))
        self.assertGreater(quotes[0],quotes[1])
        for own in (False,True):
            v=self.f.reset();self.f.rail(v);actor=v['seat'];v['players'][actor]['cash']=100
            self.f.put(v,'Birmingham:1','Manufacturer',level=2,flipped=True)
            self.f.put(v,'Coventry:1','CoalMine',level=2,seat=(actor+1)%4,resource=2)
            self.f.put(v,'Central:0','Brewery',level=2,seat=actor if own else (actor+1)%4,resource=1)
            self.f.load(v);self.f.choose('network',target='RailBirminghamCoventry');self.f.choose('coal',target='Coventry:1')
            v,c=self.f.position()
            second=[i for i,x in enumerate(c) if x['kind']=='network' and x['target']=='RailBirminghamDudley']
            self.assertEqual(bool(second),own)
            if second:self.assertEqual(self.encoded_action(v,c,second[0],'portion_beer'),1)

    def test_selected_sale_requires_accessible_or_eligible_merchant_beer(self):
        for connected,merchant_beer,compatible,expected in ((False,False,True,0),(True,False,True,2),
                (True,True,False,0),(True,True,True,2)):
            v,c=self.sale_position(connected=connected,merchant_beer=merchant_beer,compatible=compatible)
            legal={x['target'] for x in c if x['kind']=='sell'}
            self.assertEqual(len(legal),expected)
            self.assertAlmostEqual(self.encoded_state(v,c,'realization.individually_saleable_count'),expected)

    def test_two_individual_sales_do_not_claim_joint_beer_feasibility(self):
        v,c=self.sale_position()
        self.assertEqual(self.encoded_state(v,c,'realization.individually_saleable_count'),2)
        self.f.choose('sell',target='Worcester:0',merchant='Gloucester:0');self.f.choose('beer',target='Central:0')
        v,c=self.f.position();self.assertFalse(any(x['kind']=='sell' for x in c))
        self.assertEqual(self.encoded_state(v,c,'realization.saleable_count'),1)
        self.assertEqual(self.encoded_state(v,c,'realization.individually_saleable_count'),0)

    def test_identical_stack_target_keeps_finance_time_and_sale_constraints_distinguishable(self):
        early,choices=self.sale_position();late=copy.deepcopy(early);actor=late['seat']
        early['players'][actor]['cash']=100;late['players'][actor]['cash']=2
        late['context'].update(round=8,actions_remaining=1)
        for link in late['links']:link.pop('owner',None)
        next(l for l in late['links'] if l['id']=='CanalWorcesterGloucester')['owner']=actor
        self.f.load(early);early,early_choices=self.f.position()
        self.f.load(late);late,late_choices=self.f.position()
        a=DerivedContextBuilder().build(early,early_choices);b=DerivedContextBuilder().build(late,late_choices)
        for name in a.state:
            if name.startswith('development.CottonMill.'):self.assertEqual(a.state[name],b.state[name])
        self.assertTrue(a.state['development.CottonMill.current_target.reachable'].value)
        self.assertGreater(a.state['realization.own_action_slots'].value,b.state['realization.own_action_slots'].value)
        self.assertGreater(a.state['finance.0.projected_cash'].value,b.state['finance.0.projected_cash'].value)
        self.assertEqual(a.state['realization.individually_saleable_count'].value,2)
        self.assertEqual(b.state['realization.individually_saleable_count'].value,0)
        for name in ('realization.own_action_slots','finance.0.projected_cash','realization.individually_saleable_count'):
            self.assertNotEqual(self.encoded_state(early,early_choices,name),self.encoded_state(late,late_choices,name))


if __name__=='__main__':unittest.main()
