"""R3 behavioral/contract tests against the actual pinned four-player rules."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from brass_ai.legacy import LegacyWorker
from brass_ai.beliefs import sample_world_setup

COMPOSITION = json.loads((ROOT / 'configs/contracts/v0.2.0/card-composition.json').read_text())


class LegacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.w = LegacyWorker(ROOT)
    @classmethod
    def tearDownClass(cls): cls.w.close()

    def setUp(self):
        self.ctx = self.w.call('reset', seed=1301)
        self.profile = self.w.profile()
        self.profile.update(work_limit=180, milliseconds=1000)

    def view(self): return self.w.call('observe', seat=self.ctx['actor'])
    def load(self, v, seed=2):
        self.ctx = self.w.call('reset', seed=1, setup=sample_world_setup(v, COMPOSITION, seed))
    def plan(self, v=None, **kw):
        return self.w.call('plan', observation=v or self.view(), choices=self.w.call('choices'),
                           profile=self.profile, fixed_work=True, **kw)
    def commit(self, path):
        self.ctx = self.w.call('commit', game_id=self.ctx['game_id'], revision=self.ctx['revision'], choices=path)
    def hand(self, v, cards):
        v['deck_count'] += len(v['own_hand']) - len(cards)
        v['players'][v['seat']]['hand_count'] = len(cards)
        v['own_hand'] = sorted(cards)
    def put(self, v, slot, kind, *, seat=None, level=1, resource=None, flipped=False):
        seat = v['seat'] if seat is None else seat
        mat = next(m for m in v['players'][seat]['mat'] if m['type'] == kind)
        i = next(i for i, t in enumerate(mat['tiles']) if t['level'] == level)
        tile = mat['tiles'][i]
        del mat['tiles'][:i+1]
        tile['flipped'] = flipped
        if resource is not None: tile['resource'] = resource
        next(s for s in v['slots'] if s['id'] == slot)['tile'] = tile
        return tile
    def rail(self, v):
        v['context'].update(era='RailEra', round=2, actions_remaining=2)
        for p in v['players']:
            v['deck_count'] += p['hidden_discards']
            p['hidden_discards'] = 0
    def connect(self, v, link):
        next(l for l in v['links'] if l['id'] == link)['owner'] = v['seat']

    def test_first_canal_turn_has_one_complete_action(self):
        actor=self.ctx['actor']
        d = self.plan()
        self.assertEqual(1, len(d['selected']['actions']))
        self.commit(d['selected']['actions'][0])
        self.assertNotEqual(self.ctx['actor'],actor)

    def test_pure_repeatability_and_hidden_world_invariance_all_seats(self):
        for seat in range(4):
            v = self.view()
            # Obtain each acting seat via a complete legal pass.
            own = v['seat']
            before = self.w.call('hash')
            a = self.plan()
            self.assertEqual(before, self.w.call('hash'))
            b = self.plan()
            self.assertEqual(a['selected'], b['selected'])
            self.load(v, 743)
            c = self.plan()
            self.assertEqual(a['selected'], c['selected'])
            self.assertEqual(a['nodes'], c['nodes'])
            choices = self.w.call('choices')
            self.commit([next(c['id'] for c in choices if c['kind'] == 'pass'), 'finish|||||'])
            self.assertNotEqual(own, self.ctx['actor'])

    def test_two_action_plan_revalidates_and_replays(self):
        v = self.view(); v['context'].update(round=2, actions_remaining=2); self.load(v)
        cp = self.w.call('checkpoint')
        d = self.plan(); self.assertEqual(2, len(d['selected']['actions']))
        for a in d['selected']['actions']: self.commit(a)
        final = self.w.call('hash')
        self.ctx = self.w.call('restore', checkpoint=cp)
        for a in d['selected']['actions']: self.commit(a)
        self.assertEqual(final, self.w.call('hash'))

    def test_stale_duplicate_and_partial_action_are_rejected_without_mutation(self):
        before = self.w.call('hash')
        p = self.plan()['selected']['actions'][0]
        with self.assertRaises(RuntimeError): self.commit(p[:-1])
        self.assertEqual(before, self.w.call('hash'))
        old = self.ctx.copy(); self.commit(p)
        after = self.w.call('hash')
        with self.assertRaises(RuntimeError):
            self.w.call('commit', game_id=old['game_id'], revision=old['revision'], choices=p)
        self.assertEqual(after, self.w.call('hash'))
        with self.assertRaises(RuntimeError):
            self.w.call('apply',game_id='previous-game',revision=self.ctx['revision'],choice=self.w.call('choices')[0]['id'])
        self.assertEqual(after,self.w.call('hash'))

    def test_zero_budget_still_returns_valid_full_turn_incumbent(self):
        v=self.view();v['context'].update(round=2,actions_remaining=2);self.load(v)
        self.profile.update(milliseconds=0,work_limit=1)
        d=self.w.call('plan',observation=self.view(),choices=self.w.call('choices'),profile=self.profile)
        self.assertEqual(2,len(d['selected']['actions']))
        for a in d['selected']['actions']:self.commit(a)
        self.assertTrue(d['overrun'])

    def test_non_four_player_and_stale_legal_contract_fail_closed(self):
        v=self.view();v['players'].pop()
        with self.assertRaises(RuntimeError):self.plan(v)
        with self.assertRaises(RuntimeError):
            self.w.call('plan',observation=self.view(),choices=[],profile=self.profile)

    def test_terminal_uses_official_four_seat_tie_break_not_cash_conversion(self):
        v=self.view();v['context']['terminal']=True
        for i,p in enumerate(v['players']):p.update(vp=100,income=5,cash=20 if i<2 else 10)
        d=self.w.call('evaluate',observation=v,seat=0)
        self.assertEqual(0.5,d['win_share']);self.assertEqual(1,d['rank'])
        self.assertEqual(0,d['finance']);self.assertEqual(0,d['industries'])
        self.assertEqual(3,self.w.call('evaluate',observation=v,seat=2)['rank'])

    def test_final_rail_turn_cash_has_no_invented_vp(self):
        v=self.view();self.rail(v);v['context'].update(round=8,actions_remaining=0)
        a=self.w.call('evaluate',observation=v,seat=v['seat'])
        v['players'][v['seat']]['cash']=1000
        b=self.w.call('evaluate',observation=v,seat=v['seat'])
        self.assertEqual(a['score'],b['score'])

    def test_known_final_turn_prefers_winning_tie_break_over_a_loan(self):
        from collections import Counter
        v=self.view();self.rail(v);seat=v['seat']
        v['context'].update(round=8,turn=4,actions_remaining=1)
        v['order']=[s for s in range(4) if s!=seat]+[seat]
        card=v['own_hand'][0];v['own_hand']=[card];v['deck_count']=0
        pool=Counter(COMPOSITION);pool[card]-=1;cards=sorted(pool.elements())
        for p in v['players']:
            count=15 if p['seat']==seat else 16
            p.update(vp=20,bonus_vp=0 if p['seat']==seat else 4,
                     cash=1 if p['seat']==seat else 0,income_marker=10,income=0,
                     hand_count=1 if p['seat']==seat else 0,discards=cards[:count],known_wild=[])
            del cards[:count]
        self.load(v);d=self.plan();action=d['selected']['actions'][0]
        self.assertTrue(action[0].startswith('pass|'),action)
        self.assertEqual(1,d['selected']['ledger']['win_share'])
        self.commit(action);self.assertTrue(self.ctx['terminal'])
        actual=self.w.call('evaluate',observation=self.w.call('observe',seat=seat),seat=seat)
        self.assertEqual(d['selected']['ledger']['win_share'],actual['win_share'])
        self.assertEqual(d['selected']['ledger']['banked'],actual['banked'])

    def test_merchant_bonus_summary_is_not_scored_twice(self):
        v=self.view();seat=v['seat'];v['players'][seat].update(vp=14,bonus_vp=4)
        a=self.w.call('evaluate',observation=v,seat=seat)
        self.assertEqual(14,a['banked'])
        v['players'][seat]['bonus_vp']=0
        self.assertEqual(a,self.w.call('evaluate',observation=v,seat=seat))

    def test_canal_banked_points_count_once(self):
        v=self.view();self.rail(v)
        self.put(v,'Birmingham:1','Manufacturer',level=2,flipped=True)
        a=self.w.call('evaluate',observation=v,seat=v['seat'])
        v['players'][v['seat']]['vp']+=20
        b=self.w.call('evaluate',observation=v,seat=v['seat'])
        self.assertAlmostEqual(20,b['score']-a['score'])

    def test_border_link_points_are_independent_of_merchant_tiles(self):
        v=self.view();self.connect(v,'CanalCoalbrookdaleShrewsbury')
        for m in v['merchants']:
            if m['id'].startswith('Shrewsbury:'):m.pop('types',None);m['beer']=0
        self.assertEqual(2,self.w.call('evaluate',observation=v,seat=v['seat'])['links'])

    def test_supply_saturation_reduces_expected_completion(self):
        v=self.view();self.put(v,'Coalbrookdale:1','IronWorks',resource=4)
        a=self.w.call('evaluate',observation=v,seat=v['seat'])
        for seat,slot in zip([s for s in range(4) if s!=v['seat']],['Dudley:1','Birmingham:2','BurtonOnTrent:1']):
            self.put(v,slot,'IronWorks',seat=seat,resource=8)
        b=self.w.call('evaluate',observation=v,seat=v['seat'])
        self.assertLess(b['industries'],a['industries'])

    def test_loan_enables_build_and_preserves_location_card(self):
        v=self.view();v['context'].update(round=2,actions_remaining=2)
        v['players'][v['seat']]['cash']=0
        self.put(v,'Coalbrookdale:2','CoalMine',seat=(v['seat']+1)%4,resource=2)
        self.connect(v,'CanalCoalbrookdaleShrewsbury')
        v['iron_market']=0
        self.hand(v,['location:Coalbrookdale','location:Redditch'])
        self.load(v);self.profile['work_limit']=360
        d=self.plan();actions=d['selected']['actions']
        self.assertTrue(actions[0][0].startswith('loan|'),actions)
        self.assertTrue(actions[1][0].startswith('build|'),actions)
        for a in actions:self.commit(a)

    def test_rail_second_link_uses_legal_brewery_not_merchant(self):
        v=self.view();self.rail(v);v['players'][v['seat']]['cash']=45
        self.put(v,'Birmingham:1','Manufacturer',level=2,flipped=True)
        self.put(v,'BurtonOnTrent:0','Brewery',level=2,resource=2)
        self.put(v,'Coventry:1','CoalMine',seat=(v['seat']+1)%4,level=2,resource=4)
        self.connect(v,'RailBirminghamCoventry');self.load(v)
        self.profile.update(work_limit=360);d=self.plan()
        double_links=0
        for p in [d['selected']]+d['alternatives']:
            for action in p['actions']:
                if sum(x.startswith('network|') for x in action)==2:
                    double_links+=1
                    self.assertFalse(any(x.startswith('beer|merchant:') for x in action))
        self.assertGreater(double_links,0,'Fixture must actually exercise a double rail plan')
        for a in d['selected']['actions']:self.commit(a)

    def test_develop_then_build_uses_the_retained_location_card(self):
        v=self.view();v['context'].update(round=2,actions_remaining=2)
        v['players'][v['seat']]['cash']=30
        # One of the three level-I cotton tiles has already been used. Two
        # developments can now unlock level II within this ordinary action.
        self.put(v,'Worcester:0','CottonMill',flipped=True)
        v['players'][v['seat']]['income_marker']=15
        for m in v['players'][v['seat']]['mat']:
            if m['type']!='CottonMill':m['tiles']=[]
        self.hand(v,['location:Kidderminster','location:Nuneaton'])
        self.connect(v,'CanalWorcesterSouthWest');self.connect(v,'CanalWorcesterGloucester')
        for m in v['merchants']:
            if m['id'].startswith('Gloucester:'):m['types']=['CottonMill'];m['beer']=1
        self.load(v);self.profile['work_limit']=600
        d=self.plan();actions=d['selected']['actions']
        self.assertTrue(actions[0][0].startswith('develop|'),actions)
        self.assertTrue(actions[1][0].startswith('build|'),actions)
        for a in actions:self.commit(a)
        view=self.w.call('observe',seat=v['seat'])
        self.assertTrue(any(s.get('tile',{}).get('seat')==v['seat'] and s['tile']['level']>=2 for s in view['slots']))

    def test_scout_enables_build_when_all_current_location_cards_are_blocked(self):
        v=self.view();v['context'].update(round=2,actions_remaining=2)
        self.hand(v,['location:Redditch','location:Nuneaton','location:Tamworth'])
        other=(v['seat']+1)%4
        for slot in v['slots']:
            if slot['id'].split(':')[0] not in ('Redditch','Nuneaton','Tamworth'):continue
            kind=slot['types'][0]
            mat=next(m for m in v['players'][other]['mat'] if m['type']==kind)
            self.put(v,slot['id'],kind,seat=other,level=mat['tiles'][0]['level'])
        self.load(v);self.profile['work_limit']=600
        self.assertNotIn('build',{c['kind'] for c in self.w.call('choices')})
        d=self.plan()
        plans=list(d['family_best'].values())
        self.assertTrue(any(p['actions'][0][0].startswith('scout|') and p['actions'][1][0].startswith('build|') for p in plans),[p['actions'] for p in plans])
        for a in d['selected']['actions']:self.commit(a)

if __name__=='__main__':unittest.main()
