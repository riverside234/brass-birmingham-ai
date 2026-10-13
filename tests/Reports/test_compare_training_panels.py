import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('compare_panels', ROOT/'scripts/compare-training-panels.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows = [dict(seed=s, seat=seat, win=.25, rank=2.5, vp=50.) for s in (101, 103) for seat in range(4)]
    return dict(format='step5-evaluation-report/1.0.0', failed_games=0,
        spec=dict(implementation='reviewed', config=dict(setups=2, use_search=False), seeds=[101, 103],
            lineups=['recent-neural'], entries=[dict(sha256='candidate', model_id='candidate'), dict(sha256='IL', model_id='IL')]),
        results={'recent-neural': dict(games=8, setups=2, win_share=.25, mean_rank=2.5, mean_vp=50., outcomes=rows)})


class CompareTests(unittest.TestCase):
    def test_constant_paired_effect_and_shared_wins(self):
        a, b = fixture(), fixture()
        for row in b['results']['recent-neural']['outcomes']:
            row.update(win=.5, rank=2., vp=60.)
        b['results']['recent-neural'].update(win_share=.5, mean_rank=2., mean_vp=60.)
        result = module.compare(a, b, draws=1000)
        wins = result['panels']['recent-neural']['metrics']['win']
        self.assertEqual(wins['delta'], .25)
        self.assertEqual(wins['paired_setup_bootstrap_95'], [.25, .25])
        self.assertFalse(result['promotion_eligible'])

    def test_resamples_whole_setups_not_individual_seats(self):
        a, b = fixture(), fixture()
        rows = b['results']['recent-neural']['outcomes']
        for row in rows:
            row['win'] = .5 if row['seed'] == 101 else 0.
        result = module.compare(a, b, draws=1000)
        wins = result['panels']['recent-neural']['metrics']['win']
        self.assertEqual(wins['paired_setup_bootstrap_95'], [-.25, .25])
        self.assertEqual(wins['delta'], 0.)
        self.assertEqual(module.compare(a, b, draws=1000), result)

    def test_mismatched_evidence_and_incomplete_outcomes_rejected(self):
        a = fixture()
        for field in ('implementation', 'config', 'seeds', 'lineups', 'entries'):
            b = copy.deepcopy(a)
            b['spec'][field] = ({'setups': 2, 'use_search': True} if field == 'config' else
                [dict(sha256='candidate', model_id='candidate'), dict(sha256='other', model_id='other')] if field == 'entries' else 'changed')
            with self.subTest(field=field), self.assertRaises(ValueError):module.compare(a, b, draws=1000)
        for mutation in ('missing', 'duplicate', 'failed', 'summary', 'nan'):
            b = copy.deepcopy(a);panel = b['results']['recent-neural']
            if mutation == 'missing':panel['outcomes'].pop()
            elif mutation == 'duplicate':panel['outcomes'][-1] = panel['outcomes'][0]
            elif mutation == 'failed':b['failed_games'] = 1
            elif mutation == 'summary':panel['win_share'] = .8
            else:panel['outcomes'][0]['win'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):module.compare(a, b, draws=1000)
