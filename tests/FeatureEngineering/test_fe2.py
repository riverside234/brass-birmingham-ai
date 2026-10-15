import copy
import unittest
from fixtures import position
from brass_ai.feature_engineering.contracts import INDUSTRIES
from brass_ai.feature_engineering.industry import development,development_step,realization,own_action_slots
from brass_ai.feature_engineering.rules import BoardFacts
from fe_fixtures import Factory


class DevelopmentRealizationTests(unittest.TestCase):
    def setUp(self):self.factory=Factory();self.addCleanup(self.factory.close)

    def test_repeated_copies_progress_without_premature_unlock(self):
        view=self.factory.view();board=BoardFacts(view);key='development.CottonMill.current_target.'
        f=development(board);self.assertEqual(f[key+'removals'].value,3)
        step=development_step(board,dict(kind='develop',industry='CottonMill'))
        self.assertEqual((step['develop.removals_before'].value,step['develop.removals_after'].value),(3,2))
        self.assertEqual(step['develop.immediately_unlocks'].value,0)
        self.factory.choose('develop',industry='CottonMill')
        self.assertEqual(development(BoardFacts(self.factory.view()))[key+'removals'].value,2)

    def test_blockers_cannot_be_skipped_and_missing_targets_are_distinct(self):
        board=BoardFacts(self.factory.view());f=development(board)
        prefix='development.Pottery.current_target.'
        self.assertEqual(f[prefix+'present'].value,1);self.assertEqual(f[prefix+'blocked'].value,1)
        self.assertFalse(f[prefix+'removals'].known)
        board.mats['CottonMill']=[];f=development(board)
        self.assertEqual(f['development.CottonMill.current.present'].value,0)
        self.assertFalse(f['development.CottonMill.current_target.level'].applicable)

    def test_current_and_next_era_targets_and_eligibility(self):
        board=BoardFacts(self.factory.view());stack=board.mats['Brewery']
        board.mats['Brewery']=stack[-3:];f=development(board)
        self.assertEqual(f['development.Brewery.current_target.present'].value,0)
        self.assertEqual(f['development.Brewery.next_era_target.level'].value,4)
        self.assertEqual(f['development.Brewery.next_era_target.removals'].value,2)
        self.assertEqual(f['development.Brewery.current.already_unlocked'].value,1)
        board.view['context']['era']='RailEra';f=development(board)
        self.assertFalse(f['development.Brewery.next_era_target.present'].applicable)
        self.assertEqual(f['development.Brewery.current_target.level'].value,4)

    def test_single_removal_unlock_and_free_development_cost_semantics(self):
        board=BoardFacts(self.factory.view());board.mats['CoalMine']=board.mats['CoalMine'][:2]
        f=development_step(board,dict(kind='bonus_develop',industry='CoalMine'))
        self.assertEqual(f['develop.free'].value,1);self.assertEqual(f['develop.removed_tiles'].value,1)
        self.assertEqual(f['develop.immediately_unlocks'].value,1);self.assertEqual(f['develop.result.level'].value,2)
        self.assertEqual(f['develop.removals_after'].value,0)

    def test_shared_beer_is_individual_not_joint_and_works_outside_sell_ui(self):
        v=self.factory.view();self.factory.put(v,'Worcester:0','CottonMill');self.factory.put(v,'Worcester:1','CottonMill')
        self.factory.connect(v,'CanalWorcesterGloucester')
        for m in v['merchants']:
            if m['id'].startswith('Gloucester'):m.update(types=['CottonMill'],beer=int(m['id'].endswith(':0')))
        self.factory.load(v);view=self.factory.view();view['context']['phase']='payment';view['pending']['kind']='develop'
        f=realization(BoardFacts(view))
        self.assertEqual(f['realization.individually_saleable_count'].value,2)
        self.assertEqual(f['realization.beer_needed'].value,2)
        self.assertEqual(f['realization.secured_industry_vp'].value,0)

    def test_resource_industries_and_canal_exposure_are_separate(self):
        v=self.factory.view();self.factory.put(v,'Worcester:0','CottonMill');self.factory.put(v,'Dudley:0','CoalMine')
        self.factory.put(v,'Dudley:1','IronWorks',flipped=True)
        f=realization(BoardFacts(v))
        self.assertEqual(f['realization.saleable_count'].value,1)
        self.assertEqual(f['realization.unflipped_level_one_count'].value,2)
        self.assertEqual(f['realization.secured_industry_vp'].value,3)
        self.assertEqual(f['realization.potential_sale_vp'].value,5)

    def test_opening_partial_turn_cards_and_debt_action_counts(self):
        v=self.factory.view();self.assertEqual(own_action_slots(v)[0],15)
        self.factory.choose('develop',industry='CottonMill');v=self.factory.view();v['deck_count']=0
        slots,cards,active=own_action_slots(v)
        self.assertEqual(slots,15);self.assertTrue(active);self.assertEqual(cards.value,len(v['own_hand'])+1)
        v['context'].update(round=8,actions_remaining=2);v['own_hand']=v['own_hand'][:1]
        self.assertEqual(own_action_slots(v)[1].value,2)
        v['context']['phase']='debt';self.assertEqual(own_action_slots(v)[0],0)


if __name__=='__main__':unittest.main()
