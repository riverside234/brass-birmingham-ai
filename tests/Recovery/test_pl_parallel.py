"""Deterministic clustered refits, process parity, convergence and paired CIs."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from brass_ai.recovery import plackett_luce as P,policy_anchor
from brass_ai.recovery.quality_config import RATING
from test_strength_rating import simulated,AGENTS


class Parallel(unittest.TestCase):
    def test_resampling_retains_original_rng_and_cross_slot_clusters(self):
        games=simulated(setups=7)
        games+= [dict(g,slot=20,game_id=g['game_id']+'-later') for g in games]
        games=P.encode(games,AGENTS)[0];c=dict(RATING,bootstrap=20)
        setups,cluster,counts=P.bootstrap_draws(games,c);rng=np.random.default_rng(c['seed'])
        for draw in counts:
            np.testing.assert_array_equal(draw,np.bincount(rng.integers(len(setups),size=len(setups)),minlength=len(setups)))
            for setup in setups:
                self.assertEqual(len({int(draw[cluster[i]]) for i,g in enumerate(games) if g['setup_id']==setup}),1)
        _,_,again=P.bootstrap_draws(games,c)
        np.testing.assert_array_equal(counts,again)

    def test_cold_warm_serial_parallel_and_paired_promotion_parity(self):
        games=simulated(setups=24);c=dict(RATING,bootstrap=24)
        cold,cs=P.estimate(games,AGENTS,c)
        serial,ss=P.estimate(games,AGENTS,c,compute=P.COMPUTE,parallel=False)
        parallel,ps=P.estimate(list(reversed(games)),list(reversed(AGENTS)),c,compute=P.COMPUTE)
        np.testing.assert_allclose(ps,ss,rtol=0,atol=1e-12)
        np.testing.assert_allclose(ps,cs,rtol=0,atol=2e-5)
        self.assertEqual(cold['fit'],serial['fit']);self.assertEqual(serial['ratings'],parallel['ratings'])
        self.assertTrue(parallel['bootstrap_execution']['converged'])
        self.assertEqual(parallel['ratings'][P.IL]['theta'],0.)
        np.testing.assert_array_equal(ps[:,0],0.)
        self.assertAlmostEqual(sum(parallel['seat_offsets']),0.)
        for a,b in zip(cold['differences'],parallel['differences']):
            np.testing.assert_allclose(a['points_delta_ci95'],b['points_delta_ci95'],atol=.005,rtol=0)
            self.assertEqual((a['a_stronger'],a['b_stronger']),(b['a_stronger'],b['b_stronger']))
            evidence=policy_anchor.paired_difference(parallel,ps,list(parallel['ratings']),b['b'],b['a'],c['elo_scale'])
            self.assertEqual(evidence['points_delta_ci95'],b['points_delta_ci95'])
        from test_policy_anchor import evaluated
        from brass_ai.learning.contracts import SOURCE_ROOT
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);state,_,_,_,_=evaluated(root,generations=(10,20))
            for reference in (None,'generation-000010'):
                controller=policy_anchor.initial(20)
                if reference:
                    controller['active']=dict(agent=reference,checkpoint=state['ratings']['agents'][reference]['checkpoint'])
                selected=[]
                for report,draws in ((cold,cs),(serial,ss),(parallel,ps)):
                    with patch.object(policy_anchor,'load_fit',return_value=(report,draws,list(report['ratings']))):
                        choice,_,reason=policy_anchor.select(root,state,controller)
                    selected.append((choice['agent'] if choice else None,reason))
                self.assertEqual(selected[0],selected[1]);self.assertEqual(selected[1],selected[2])

    def test_warm_start_validated_and_failed_fit_retries_cold(self):
        _,agents,ordered,orders=P.encode(simulated(setups=4),AGENTS)
        n=len(agents)
        with self.assertRaises(ValueError):P.fit_encoded(ordered,orders,n,RATING,start=np.zeros(1))
        with self.assertRaises(ValueError):P.fit_encoded(ordered,orders,n,RATING,start=np.full(n+2,np.nan))
        expected=(np.zeros(n),np.zeros(4),dict(converged=True,iterations=1,gradient_max=0.))
        with patch.object(P,'fit_encoded',side_effect=[ValueError('Plackett-Luce fit did not converge'),expected]) as fit:
            _,diagnostic=P._bootstrap_refit(np.ones(4,dtype=int),(ordered,orders,n,RATING,np.repeat(np.arange(4),4),np.zeros(n+2)))
        self.assertTrue(diagnostic['cold_retry']);self.assertIn('start',fit.call_args_list[0].kwargs)
        self.assertNotIn('start',fit.call_args_list[1].kwargs)
        with patch.object(P,'fit_encoded',side_effect=ValueError('Plackett-Luce fit did not converge')):
            with self.assertRaises(ValueError):P._bootstrap_refit(np.ones(4,dtype=int),(ordered,orders,n,RATING,np.repeat(np.arange(4),4),np.zeros(n+2)))

    def test_execution_contract_and_spawn_environment_restoration(self):
        for n in (4,6,8):P.validate_compute(dict(P.COMPUTE,workers=n))
        for change in (dict(workers=9),dict(workers=True),dict(threads_per_worker=2),dict(warm_start=False)):
            with self.assertRaises(ValueError):P.validate_compute(dict(P.COMPUTE,**change))
        before=copy.deepcopy(dict(os.environ))
        with self.assertRaises(RuntimeError):
            with P.single_thread_children():
                self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'],'1')
                self.assertEqual(os.environ['OMP_NUM_THREADS'],'1')
                raise RuntimeError('test cleanup')
        self.assertEqual(dict(os.environ),before)


if __name__=='__main__':unittest.main()
