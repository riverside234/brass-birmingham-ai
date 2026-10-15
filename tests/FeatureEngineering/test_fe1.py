import copy
import unittest
from fixtures import position,rotate
from brass_ai.feature_engineering.contracts import Measure
from brass_ai.feature_engineering.finance import financial,provisional_order,opportunities


class FinanceOrderOpportunityTests(unittest.TestCase):
    def test_signed_projection_and_relative_seats(self):
        v,_=position(2);v['players'][3].update(cash=2,income=-7)
        f=financial(v)
        self.assertEqual(f['finance.1.negative_payment'].value,7)
        self.assertEqual(f['finance.1.projected_cash'].value,-5)
        self.assertEqual(f['finance.1.shortfall'].value,5)
        self.assertEqual(f,financial(rotate(v,1)))

    def test_debt_unknown_is_not_zero_or_past_settlement(self):
        v,_=position();v['context']['phase']='debt';v['players'][0].update(cash=0,income=-10,debt=10)
        f=financial(v);self.assertFalse(f['finance.0.projected_cash'].known)
        self.assertTrue(f['finance.0.projected_cash'].applicable);self.assertEqual(f['finance.0.public_debt'].value,10)
        self.assertEqual(f['finance.0.negative_payment'].value,10)
        v['context'].update(phase='action',round=2);v['players'][0].update(cash=3,income=-2,debt=0)
        self.assertEqual(financial(v)['finance.0.projected_cash'].value,1)

    def test_final_rail_and_canal_are_different(self):
        v,_=position();v['context'].update(era='RailEra',round=8)
        self.assertFalse(financial(v)['finance.0.projected_cash'].applicable)
        self.assertEqual(financial(v)['finance.0.settlement'].value,0)
        self.assertFalse(provisional_order(v)['order.0.position'].applicable)
        v['context']['era']='CanalEra'
        self.assertTrue(financial(v)['finance.0.projected_cash'].known)
        self.assertTrue(provisional_order(v)['order.0.position'].applicable)

    def test_order_uses_current_order_ties_and_signed_differences(self):
        v,_=position(2);v['order']=[2,0,3,1]
        for p in v['players']:p['spent']=4
        v['players'][1]['spent']=1
        f=provisional_order(v)
        self.assertEqual([f[f'order.{r}.position'].value for r in range(4)],[1,3,2,0])
        self.assertEqual(f['order.3.spending_difference'].value,3)
        self.assertEqual(f,provisional_order(rotate(v,3)))

    def test_dedup_keeps_material_targets_and_scout_multisets(self):
        v,c=position();c+= [dict(id='b1',kind='build',target='Birmingham:0',industry='CottonMill',card='a'),dict(id='b2',kind='build',target='Birmingham:0',industry='CottonMill',card='b'),dict(id='b3',kind='build',target='Birmingham:1',industry='CottonMill',card='a'),dict(id='s1',kind='scout',card='a',cards=['a','b']),dict(id='s2',kind='scout',card='b',cards=['a','a']),dict(id='s3',kind='scout',card='a',cards=['b','b'])]
        original=copy.deepcopy(c);f=opportunities(v,c)
        self.assertEqual(f['opportunities.build.targets'].value,2)
        self.assertEqual(f['opportunities.scout.targets'].value,2)
        self.assertEqual(f['opportunities.pass.targets'].value,1)
        self.assertEqual(c,original)

    def test_payment_absence_does_not_claim_no_main_opportunity(self):
        v,_=position();v['context']['phase']='payment';v['pending']['kind']='build'
        f=opportunities(v,[dict(kind='coal',target='market')])
        self.assertFalse(f['opportunities.build.available'].applicable)
        self.assertTrue(f['opportunities.coal.available'].known)
        self.assertEqual(f['opportunities.coal.targets'].value,1)

    def test_measure_exact_unknown_bounded_inapplicable_are_distinct(self):
        self.assertEqual(Measure.exact(0).tensor(100),(0.,1.,1.,0.,1.,0.,1.))
        self.assertNotEqual(Measure().tensor(1),Measure.absent().tensor(1))
        self.assertEqual(Measure.interval(-5,None).lower,-5)
        self.assertFalse(Measure.interval(-5,None).upper_known)
        with self.assertRaises(ValueError):Measure.interval(2,1)


if __name__=='__main__':unittest.main()
