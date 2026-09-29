"""Synthetic admission fixtures, never match or playing-strength evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from brass_ai.bank import digest
from brass_ai.qualification import summarize, write_new


class EconomicAdmissionTests(unittest.TestCase):
    def report(self, terminal_vp):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);name='balanced-v3'
            profiles={name:dict(id='fixture-balanced-v3',tactic='balanced')}
            payload=dict(player_count=4,wire='0.2.0',rules='steam-brass-patch-1.1.0',
                         source_files={},adapter_files=[],profiles=profiles)
            bank='bank-'+digest(payload)[:20]
            write_new(root/'training/opponents/bank'/bank/'manifest.json',dict(id=bank,payload=payload))
            jobs=[dict(id=stage,stage=stage,seed=1,table=[name,None,None,None],target=0)
                  for stage in ('guarded','mixed','identical')]
            body=dict(bank=bank,jobs=jobs,bootstrap_seed=731,bootstrap_samples=100,equal_budget_pair=[],
                admission=dict(min_guarded_games=1,min_mixed_games=1,min_identical_games=1,
                    maximum_economic_stalls=0,minimum_guarded_mean_rank=2.5,
                    held_out_family='pottery-flex',general_hard_min_setup_groups=30))
            spec=dict(id='qualification-'+digest(body)[:20],payload=body)
            for job in jobs:
                seats=[dict(seat=s,vp=90-s*10,cash=10,income=5,rank=s+1,utility=int(s==0)) for s in range(4)]
                if job['stage']=='mixed':seats[0].update(vp=terminal_vp,cash=0,income=-9,rank=4,utility=0)
                match=dict(completed=True,seed=1,player_count=4,turns=64,actions=124,trace=[],turn_reports=[],
                    table_profiles=['fixture-balanced-v3']+['stock-hard-guarded-1.1.0']*3,
                    result=dict(objective='win-share/1.0.0',seats=seats))
                write_new(root/'matches'/(job['id']+'.json'),dict(job=job,spec_id=spec['id'],match=match))
            return summarize(root,spec,root/'matches',dict(passed=True,bank=bank))

    def test_legal_completion_does_not_admit_zero_vp_bankruptcy(self):
        report=self.report(0);profile=report['profiles']['balanced-v3']
        self.assertEqual(0,report['failed_games'])
        self.assertTrue(profile['reliable_in_observed_games'])
        self.assertEqual(1,len(profile['economic_stalls']))
        self.assertFalse(profile['niche_qualified'])
        self.assertFalse(profile['training_eligible'])

    def test_spending_down_at_game_end_is_not_automatically_a_stall(self):
        profile=self.report(20)['profiles']['balanced-v3']
        self.assertEqual([],profile['economic_stalls'])
        self.assertTrue(profile['niche_qualified'])


if __name__=='__main__':unittest.main()
