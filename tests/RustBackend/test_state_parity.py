import copy
import unittest

from brass_ai.rust_backend.contracts import chance_progress,comparison,first_difference
from support import root_fixture


class ComparatorTests(unittest.TestCase):
    def setUp(self):self.fixture=root_fixture();self.original=self.fixture['projection']

    def check_change(self,mutate,expected_path):
        actual=copy.deepcopy(self.original);mutate(actual)
        result=comparison(self.original,actual,expected_source='frozen-csharp',actual_source='deliberately-mutated')
        self.assertFalse(result['passed']);self.assertEqual(result['first_difference']['path'],expected_path)
        self.assertEqual(result['expected_source'],'frozen-csharp');self.assertEqual(result['actual_source'],'deliberately-mutated')
        self.assertNotEqual(result['expected_digest'],result['actual_digest'])

    def test_changed_state_detected(self):
        self.check_change(lambda p:p['context'].__setitem__('actions_remaining',2),'$.context.actions_remaining')

    def test_legal_choice_identity_detected(self):
        self.check_change(lambda p:p['choices'][0].__setitem__('id','illegal'),'$.choices[0].id')

    def test_legal_choice_order_is_not_normalized(self):
        changed=copy.deepcopy(self.original);changed['choices'].reverse()
        self.assertTrue(first_difference(self.original,changed)['path'].startswith('$.choices[0].'))

    def test_masked_view_detected(self):
        self.check_change(lambda p:p['views'][0]['players'][0].__setitem__('cash',999),'$.views[0].players[0].cash')

    def test_hidden_deck_order_detected(self):
        self.check_change(lambda p:p['cards']['deck'].__setitem__(0,'wrong-card'),'$.cards.deck[0]')

    def test_returns_and_chance_are_not_ignored(self):
        for field in ('utility','terminal_utility','game_utility'):
            expected={field:[1.,0.,0.,0.]};actual={field:[0.,1.,0.,0.]}
            self.assertEqual(first_difference(expected,actual)['path'],f'$.{field}[0]')
        self.assertEqual(first_difference({'probability':.25},{'probability':.5})['path'],'$.probability')

    def test_pending_rail_deal_keeps_picked_order(self):
        chance=dict(remaining=62)
        a=chance_progress(chance,[{'choice':'finish'},{'chance':'a'},{'chance':'b'}])
        b=chance_progress(chance,[{'choice':'finish'},{'chance':'b'},{'chance':'a'}])
        self.assertEqual(a['needed'],64);self.assertEqual(a['picked'],['a','b'])
        self.assertEqual(first_difference(a,b)['path'],'$.picked[0]')
        self.assertIsNone(chance_progress(None,[{'chance':'a'}]))

    def test_inline_inference_has_no_executor_and_rejects_other_owner(self):
        from unittest.mock import patch
        import threading
        from brass_ai.rust_backend.benchmark import InlineEvaluator
        from brass_ai.learning.batching import Evaluator
        evaluator=object.__new__(InlineEvaluator);evaluator.owner=threading.get_ident();evaluator._executor=None;evaluator.calls=0
        with patch.object(Evaluator,'_predict',return_value=['same-forward-path']) as forward:
            self.assertEqual(evaluator.predict(['request']),['same-forward-path']);forward.assert_called_once_with(['request'])
            self.assertIsNone(evaluator._executor)
            evaluator.owner=-1
            with self.assertRaises(ValueError):evaluator.predict(['request'])

    def test_root_merge_keeps_reference_prior_tiebreak(self):
        from brass_ai.rust_backend.benchmark import select_root
        choices=[dict(id='first'),dict(id='second'),dict(id='third')]
        stats=dict(root={c['id']:dict(visits=2) for c in choices})
        self.assertEqual(select_root(stats,choices,[.1,.7,.2]),1)
        stats['root']['third']['visits']=3
        self.assertEqual(select_root(stats,choices,[.1,.7,.2]),2)
        self.assertEqual(select_root(dict(root={}),choices,[1.,1.,1.]),0)

    def test_nonfinite_and_missing_fields_rejected(self):
        self.assertIsNotNone(first_difference(1.,float('nan')))
        self.assertIsNotNone(first_difference(1,True))
        self.assertEqual(first_difference({'a':None},{})['path'],'$.a')

    def test_exact_identity_passes(self):
        self.assertTrue(comparison(self.original,copy.deepcopy(self.original),expected_source='a',actual_source='b')['passed'])


if __name__=='__main__':unittest.main()
