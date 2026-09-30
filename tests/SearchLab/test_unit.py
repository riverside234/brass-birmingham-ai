import copy
from pathlib import Path
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from brass_ai.searchlab.s3 import TurnNode,TreeConfig,history_key
from brass_ai.searchlab.spiel import action_id,TYPE,INFO
from brass_ai.searchlab.experiment import paired_summary,validate_config


class SearchUnitTests(unittest.TestCase):
    def test_combined_r4_cache_respects_private_cards_and_ordered_choices(self):
        from brass_ai.searchlab.ismcts import PermittedEvaluator
        policy=SimpleNamespace(evaluator=SimpleNamespace(model_id='frozen'),exploration=.1,
            temperature=1.,contexts={},predict=Mock(return_value=[(np.array([.4,.6]),
                {'win_share':[.1,.2,.3,.4]})]))
        state=SimpleNamespace(current_player=lambda:0,views=[{'own_hand':['Pottery']},None,None,None],
            legal=[{'id':'a'},{'id':'b'}])
        cache=PermittedEvaluator(policy,([0]*4,[0]*8),random.Random(5))
        cache.prior(state);self.assertEqual(cache.evaluate(state),[.1,.2,.3,.4])
        self.assertEqual(policy.predict.call_count,1)
        state.views[0]['own_hand']=['Brewery'];cache.evaluate(state)
        self.assertEqual(policy.predict.call_count,2)
        state.legal.reverse();cache.prior(state)
        self.assertEqual(policy.predict.call_count,3)

    def test_current_shared_execution_config_rejects_ignored_aliases(self):
        from brass_ai.learning.contracts import SOURCE_ROOT,read_json
        config=read_json(SOURCE_ROOT/'configs/searchlab/experiment.json')
        validate_config(config)
        with self.assertRaisesRegex(ValueError,'aliases differ'):
            validate_config(dict(config,batch_size=1))

    def test_four_seat_backup_uses_actor_utility(self):
        node=TurnNode([['a'],['b']])
        node.backup(0,[.7,.1,.1,.1]);node.backup(1,[.1,.1,.7,.1])
        self.assertEqual(node.choose(0,1,True),0)
        self.assertEqual(node.choose(2,1,True),1)
        self.assertEqual(node.sums.tolist(),[[.7,.1,.1,.1],[.1,.1,.7,.1]])
        with self.assertRaises(ValueError):node.backup(0,[.5,.5])

    def test_private_observations_and_recall_split_future_nodes(self):
        view=dict(own_hand=['Pottery'],history=[dict(choice='loan')])
        other=copy.deepcopy(view);other['own_hand']=['Brewery']
        self.assertNotEqual(history_key(view,[['loan']]),history_key(other,[['loan']]))
        self.assertNotEqual(history_key(view,[['loan']]),history_key(view,[['pass']]))

    def test_unvisited_root_plan_survives_pruning(self):
        node=TurnNode([['loan','build'],['build'],['pottery']])
        for expected in range(3):
            self.assertEqual(node.choose(0,1),expected);node.backup(expected,[0,0,0,1])

    def test_action_ids_stable_bounded_and_order_independent(self):
        choices=['loan|||||','build|x|Pottery|card||','chance:card:location:Birmingham']
        self.assertEqual({c:action_id(c) for c in choices},{c:action_id(c) for c in reversed(choices)})
        self.assertTrue(all(0<=action_id(c)<2**31-1 for c in choices))
        self.assertEqual(len(set(map(action_id,choices))),3)

    def test_four_player_constant_sum_explicit_chance_metadata(self):
        import pyspiel
        self.assertEqual(TYPE.min_num_players,4);self.assertEqual(TYPE.max_num_players,4)
        self.assertEqual(TYPE.chance_mode,pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
        self.assertEqual(TYPE.utility,pyspiel.GameType.Utility.CONSTANT_SUM)
        self.assertEqual(INFO.utility_sum,1)

    def test_tree_budget_validation(self):
        for config in [TreeConfig(simulations=0),TreeConfig(future_turns=0),TreeConfig(future_turns=3)]:
            with self.assertRaises(ValueError):config.validate()
        TreeConfig(future_turns=2).validate()

    def results(self,count=30):
        return [dict(variant=v,seed=s,seat=p,win=float(v=='s3'),rank=1 if v=='s3' else 4,
            vp=150 if v=='s3' else 80,over_budget=0,max_turn_seconds=.8,
            pottery=dict(builds=dict(Pottery=1),legal_build_opportunities=dict(Pottery=10)))
            for s in range(count) for p in range(4) for v in ['s2','s3']]

    def test_promotion_requires_paired_whole_setups_and_advantage(self):
        data=self.results()
        self.assertTrue(paired_summary(data,['s2','s3'],30,'validation')['s3']['eligible'])
        with self.assertRaises(ValueError):paired_summary(data[:-1],['s2','s3'],30,'validation')
        self.assertFalse(paired_summary(self.results(1),['s2','s3'],1,'validation')['s3']['eligible'])

    def test_test_partition_timing_or_pottery_failure_blocks_promotion(self):
        data=self.results()
        self.assertFalse(paired_summary(data,['s2','s3'],30,'test')['s3']['eligible'])
        data[0]['over_budget']=1
        self.assertFalse(paired_summary(data,['s2','s3'],30,'validation')['s3']['eligible'])
        data=self.results()
        for row in data:row['pottery']['builds']['Pottery']=0
        self.assertFalse(paired_summary(data,['s2','s3'],30,'validation')['s3']['eligible'])

    def test_no_mean_advantage_blocks_promotion(self):
        data=self.results()
        for row in data:row['win']=.25
        self.assertFalse(paired_summary(data,['s2','s3'],30,'validation')['s3']['eligible'])

    def test_invalid_or_weakened_experiment_settings_rejected(self):
        import json
        config=json.loads((Path(__file__).resolve().parents[2]/'configs/searchlab/experiment.json').read_text())
        validate_config(config)
        for key,value in [('players',2),('promotion_min_setups',1),('seconds',0),('validation_reserve_fraction',1)]:
            invalid=dict(config);invalid[key]=value
            with self.assertRaises(ValueError):validate_config(invalid)


if __name__=='__main__':unittest.main()
