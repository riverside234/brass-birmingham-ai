"""Regression checks for P0/P1 review; no real trained model is required."""
import unittest

from personality_fixtures import episode
from brass_ai.learning.contracts import digest
from brass_ai.learning.reward import specification
from brass_ai.personality.contracts import context
from brass_ai.personality.traces import examples


class AttributionReviewTests(unittest.TestCase):
    def recorded(self):
        game=episode(rounds=2)
        lid,pid='learner-fixture','parent-fixture'
        policy='personality-pilot/'+digest(dict(model=lid,profile='neutral',exploration=.05))
        parent='personality-parent/'+pid
        game['opponents']=[policy,parent,parent,parent]
        attribution=[]
        for row in game['records']:
            learner=row['seat']==0
            row['behavior'].update(kind='model',policy_id=policy if learner else parent,temperature=1.)
            attribution.append(dict(step=row['step'],role='learner' if learner else 'incumbent',
                model_id=lid if learner else pid,context=context() if learner else None))
        meta=dict(spec=dict(model_id=lid,parent_id=pid),contexts={game['game_id']:attribution})
        return game,meta

    def test_matching_recorded_model_actions_are_on_policy(self):
        game,meta=self.recorded()
        data=examples([game],specification(),metadata=meta,learner_only=True)
        self.assertTrue(data)
        self.assertTrue(all(x['on_policy'] for x in data))

    def test_external_action_cannot_become_ppo_through_matching_id(self):
        game,meta=self.recorded()
        game['records'][0]['behavior']['kind']='external'
        with self.assertRaisesRegex(ValueError,'unsearched model behavior'):
            examples([game],specification(),metadata=meta,learner_only=True)

    def test_actual_sampling_temperature_must_match_recipe(self):
        game,meta=self.recorded()
        game['records'][0]['behavior']['temperature']=.5
        with self.assertRaisesRegex(ValueError,'temperature 1'):
            examples([game],specification(),metadata=meta,learner_only=True)

    def test_opponent_rows_are_checked_before_learner_filter(self):
        game,meta=self.recorded()
        next(r for r in game['records'] if r['seat']==1)['behavior']['kind']='external'
        with self.assertRaisesRegex(ValueError,'unsearched model behavior'):
            examples([game],specification(),metadata=meta,learner_only=True)


if __name__=='__main__':unittest.main()
