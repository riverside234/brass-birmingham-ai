"""H3/H4 checks through the actual rules worker, not a replacement simulator."""
import copy
import unittest
from collections import Counter
from test_legacy import LegacyTests, COMPOSITION


class AdvancedTests(LegacyTests):
    def setUp(self):
        super().setUp()
        self.profile=self.w.profile('balanced-s2-v2')
        self.profile.update(work_limit=120,reply_candidates=3,reply_scenarios=1,reply_confirmation=1,reply_work=2048)

    def scenario(self,v):
        cards=v['own_hand'][:v['context']['actions_remaining']]
        actions=[[f'pass|||{c}||','finish|||||'] for c in cards]
        return self.w.call('scenario',observation=v,plan={'actions':actions},sampling_seed=173)

    def final_hand(self,v,canal):
        seat=v['seat'];cards=v['own_hand'][:2]
        pool=Counter(COMPOSITION);pool.subtract(cards);remaining=list(pool.elements())
        v['own_hand']=cards;v['deck_count']=0
        for p in v['players']:
            count=(13 if p['seat']==seat else 15) if canal else (14 if p['seat']==seat else 16)
            p.update(hand_count=2 if p['seat']==seat else 0,discards=remaining[:count],hidden_discards=1 if canal else 0,known_wild=[],cash=50,income_marker=10,income=0)
            del remaining[:count]

    def test_s2_matched_pilot_and_disjoint_fresh_samples(self):
        d=self.plan();r=d['replies']
        self.assertEqual('confirmed',r['status'])
        self.assertEqual(1,r['pilot_batches']);self.assertEqual(1,r['fresh_batches'])
        self.assertFalse(set(r['pilot_seeds'])&set(r['fresh_seeds']))
        self.assertTrue(all(len(e['seats'])==4 for e in r['estimates']))
        self.assertTrue(all('variance' not in e for e in r['estimates']))
        self.commit(d['selected']['actions'][0])

    def test_root_and_rollout_privacy_all_four_seats(self):
        self.profile.update(reply_candidates=2,work_limit=40)
        for _ in range(4):
            view=self.view();a=self.plan()
            self.load(view,97531);b=self.plan()
            self.assertEqual(a['selected'],b['selected']);self.assertEqual(a['replies'],b['replies'])
            choices=self.w.call('choices')
            for style in ('balanced','cotton','supplier','rail-tempo'):
                x=self.w.call('rollout_choice',observation=view,choices=choices,style=style)
                self.load(view,86420)
                y=self.w.call('rollout_choice',observation=self.view(),choices=self.w.call('choices'),style=style)
                self.assertEqual(x,y)
            self.commit([next(c['id'] for c in choices if c['kind']=='pass'),'finish|||||'])

    def test_scheduler_zero_and_six_intervening_turns(self):
        v=self.view();self.rail(v);seat=v['seat']
        v['context'].update(round=2,turn=4)
        v['order']=[s for s in range(4) if s!=seat]+[seat]
        for p in v['players']:p['spent']=0 if p['seat']==seat else 100
        self.assertEqual([0],self.scenario(v)['telemetry']['intervening_turns'])
        v['context']['turn']=1;v['order']=[seat]+[s for s in range(4) if s!=seat]
        for p in v['players']:p['spent']=100 if p['seat']==seat else 0
        result=self.scenario(v)
        self.assertEqual([6],result['telemetry']['intervening_turns'])

    def test_reply_resolves_canal_transition_and_terminal(self):
        v=self.view();seat=v['seat'];v['context'].update(round=8,turn=4,actions_remaining=2)
        v['order']=[s for s in range(4) if s!=seat]+[seat]
        for p in v['players']:p['spent']=0 if p['seat']==seat else 100
        self.final_hand(v,True)
        self.assertEqual(['RailEra'],self.scenario(v)['telemetry']['leaf_eras'])
        self.rail(v);v['context'].update(round=8,turn=4,actions_remaining=2);self.final_hand(v,False)
        r=self.scenario(v)
        self.assertEqual([0],r['telemetry']['intervening_turns'])
        self.assertEqual(4,len(r['values']))
        self.assertTrue(all(0<=x<=1.15 for x in r['values']))

    def test_exhausted_reply_budget_preserves_full_s1_incumbent(self):
        self.profile['reply_work']=1
        v=self.view();v['context'].update(round=2,actions_remaining=2);self.load(v)
        d=self.plan();self.assertNotEqual('confirmed',d['replies']['status'])
        self.assertEqual(2,len(d['selected']['actions']))
        for action in d['selected']['actions']:self.commit(action)

    def intent_fixture(self):
        v=self.view();v['context'].update(round=3,actions_remaining=2)
        v['players'][v['seat']]['cash']=50
        self.hand(v,['location:Worcester','location:Coalbrookdale','location:Birmingham','location:Coventry'])
        self.connect(v,'CanalWorcesterGloucester');self.connect(v,'CanalCoalbrookdaleShrewsbury')
        borders={m['id'].split(':')[0] for m in v['merchants']}
        for city in ('Birmingham','Coventry'):
            queue=[(city,[])];seen={city}
            for current,path in queue:
                if current in borders:
                    for link in path:link['owner']=v['seat']
                    break
                for link in v['links']:
                    if not link['id'].startswith('Canal') or current not in link['regions']:continue
                    for region in link['regions']:
                        if region not in seen:seen.add(region);queue.append((region,path+[link]))
        v['players'][v['seat']]['link_tiles']=14-sum(l.get('owner')==v['seat'] for l in v['links'])
        for m in v['merchants']:m['types']=['CottonMill','Manufacturer','Pottery'];m['beer']=1
        v['iron_market']=0
        return v

    def test_specialist_intent_entry_retention_and_recovery(self):
        for name,focus in [('cotton','CottonMill'),('goods-beer','Manufacturer'),('supplier','IronWorks'),('pottery-flex','Pottery')]:
            v=self.intent_fixture();p=self.w.profile(name+'-v2')
            a=self.w.call('intent',observation=v,profile=p);self.assertEqual(focus,a['focus'])
            v['context']['revision']+=1
            b=self.w.call('intent',observation=v,profile=p,intent=a)
            self.assertEqual(a['target'],b['target']);self.assertIn('retained',b['reason'])
            v['own_hand']=[]
            c=self.w.call('intent',observation=v,profile=p,intent=b)
            self.assertEqual('balanced',c['focus']);self.assertIn('abandoned',c['reason'])
            self.assertIn('cash_budget',a);self.assertIn('required_cards',a)

    def test_intent_cannot_cross_seats_profiles_or_future_revision(self):
        v=self.intent_fixture();p=self.w.profile('cotton-v2')
        a=self.w.call('intent',observation=v,profile=p)
        for key,value in [('seat',(v['seat']+1)%4),('profile','another-policy'),('revision',a['revision']+1)]:
            invalid=copy.deepcopy(a);invalid[key]=value
            with self.assertRaises(RuntimeError):self.w.call('intent',observation=v,profile=p,intent=invalid)

    def test_contested_beer_repairs_or_abandons_sale_intent(self):
        v=self.intent_fixture();p=self.w.profile('cotton-v2')
        self.put(v,'Worcester:0','CottonMill',level=2)
        for merchant in v['merchants']:merchant['beer']=0
        intent=self.w.call('intent',observation=v,profile=p)
        self.assertEqual('secure_own_beer',intent['milestone'])
        self.assertEqual('Brewery',intent['focus']);self.assertTrue(intent['required_cards'])
        v['players'][v['seat']]['cash']=0
        abandoned=self.w.call('intent',observation=v,profile=p,intent=intent)
        self.assertEqual('balanced',abandoned['focus']);self.assertIn('no affordable',abandoned['reason'])
        for merchant in v['merchants']:merchant['types']=[]
        self.assertEqual('balanced',self.w.call('intent',observation=v,profile=p)['focus'])

    def test_specialists_select_only_credible_complete_turns(self):
        v=self.intent_fixture();self.load(v)
        for style in ('cotton','goods-beer','supplier','pottery-flex'):
            self.profile=self.w.profile(style+'-v2');self.profile['work_limit']=140
            d=self.plan();self.assertEqual(2,len(d['selected']['actions']))
            best=max(p['score'] for p in d['alternatives'])
            self.assertGreaterEqual(d['selected']['score'],best-self.profile['credible_window']-1e-8)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(AdvancedTests(n) for n in AdvancedTests.__dict__ if n.startswith('test_'))
