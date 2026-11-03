"""Short task-only refresh checks; no production games or training updates."""
import copy
import json
import threading
import time
import unittest
from unittest.mock import patch

import torch
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.batching import model_id
from brass_ai.recovery import gae
from brass_ai.recovery.teacher_phase_config import defaults, validate, TIMING
from brass_ai.rust_backend.lab_recipe import phase_payload
from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
from brass_ai.rust_backend.lab_transfer import RETIRED
from test_b025 import model_fixture, example
from test_b04 import job

LOG = SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0370-20261102T221517838215Z'


class Refresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = model_fixture().eval()
        cls.state = json.loads((LOG/'state.json').read_text())
        cls.config = json.loads((SOURCE_ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g370.json').read_text())

    def fixture(self):
        j = job(self.model)
        row,_ = example(self.model,job=j,round=j['reservations']['0']['canal_early'])
        stop = threading.Event()
        v = TurnVerifier(self.model,[dict(generation=10,model_id=model_id(self.model))],[j],stop)
        meta = dict(job=j['expected'],index=1,timing=TIMING,remaining_ns=900_000_000,
                    native_post_ns=50_000_000,search_ns=4_000_000_000,rows=[row.row])
        return j,v,meta,stop

    def test_retired_modules_absent_and_no_intrinsic_profile(self):
        self.assertTrue(all(not (SOURCE_ROOT/p).exists() for p in RETIRED))
        from brass_ai.recovery.teacher_phase_config import adopt
        c = adopt(self.config); self.assertEqual(c['teacher_phase']['search']['post_search_seconds'],1.)
        self.assertEqual(c['teacher_phase']['search']['allowance_seconds'],4.)
        self.assertNotIn('diversity',c['teacher_phase'])
        c['diversity']['mode']='apply'
        with self.assertRaises(ValueError): validate(c)

    def test_g370_config_and_checkpoint_distinguished(self):
        from brass_ai.recovery.training_revision_g370 import configure
        from brass_ai.recovery.config import validate_config
        self.assertEqual(self.config,validate_config(configure(self.state['signature']['config'])))
        self.assertEqual(self.state['history'][-1]['sha256'],'198af418247e6559eb4e2413acb02afd54e7baa5e446a30fb6ce1442c37dfcdc')
        self.assertEqual(self.config['diversity']['mode'],'off')
        self.assertEqual(self.config['advantage']['lambda'],.95)
        self.assertEqual(self.config['branching']['rate_per_decision'],.005)
        self.assertEqual(self.config['value_phase']['beta_clone_min'],1.)

    def test_only_reviewed_g370_pending_phase_changes_are_projected(self):
        old = self.state['signature']['config']
        saved = dict(options={k:copy.deepcopy(old[k]) for k in ('advantage','value_phase','critic_data','seat_descriptors')},
            gae=gae.initial(old['advantage'],370),beta_clone=.5,adoption=None,
            optimizer={'moment':torch.tensor([4.])},rolling_buffer=[{'generation':369}],kl_trips=[False,True])
        raw = {'training':{'actor_value_phases':saved}}; before = copy.deepcopy(raw)
        after, receipt = phase_payload(raw,self.config,370)
        self.assertEqual(after['beta_clone'],1.); self.assertEqual(after['gae']['active_lambda'],.95)
        self.assertEqual(after['rolling_buffer'],saved['rolling_buffer'])
        self.assertTrue(torch.equal(after['optimizer']['moment'],saved['optimizer']['moment']))
        from brass_ai.recovery.retention import _same_payload
        self.assertTrue(_same_payload(before,raw))
        self.assertEqual(set(receipt['changes']),{'gae','ppg'})
        self.assertIsNone(phase_payload({'training':{'actor_value_phases':after}},self.config,370)[1])
        bad=copy.deepcopy(self.config);bad['seat_descriptors']['dropout_games']=.99
        with self.assertRaises(ValueError): phase_payload(raw,bad,370)

    def test_once_only_verification_and_tamper_rejected(self):
        j,v,meta,stop = self.fixture()
        # Empty later searched turns are valid; no invented evidence.
        for i in range(1,17):
            m=dict(meta,index=i,rows=meta['rows'] if i==1 else [])
            self.assertEqual(v.verify(m,time.monotonic())['status'],'ok')
        result=dict(job_id=j['expected']['job_id'],rows=copy.deepcopy(meta['rows']),metrics={'reserved_turns':16})
        bad=copy.deepcopy(result);bad['rows'][0]['teacher']['target'][0] += .1
        with self.assertRaises(ValueError): v.take(bad,j,self.model)
        with patch('brass_ai.rust_backend.lab_teacher_finalization.admit_teacher_rows',side_effect=AssertionError('duplicate validation')):
            rows,rejected=v.take(result,j,self.model)
        self.assertEqual(len(rows),1); self.assertEqual(len(result['metrics']['verification']),16)
        with self.assertRaises(ValueError): v.take(result,j,self.model)

    def test_expiry_cancellation_stale_and_unbounded_allowance(self):
        for mode in ('expired','cancelled','stale','duplicate','wide'):
            j,v,meta,stop=self.fixture();start=time.monotonic()
            if mode=='expired':start-=2
            elif mode=='cancelled':stop.set()
            elif mode=='stale':meta['job']=dict(meta['job'],epoch=999)
            elif mode=='duplicate':meta['index']=2
            else:meta['remaining_ns']=1_000_000_001
            with self.subTest(mode=mode),self.assertRaises(ValueError):v.verify(meta,start)

    def test_late_verification_cannot_acknowledge_or_publish(self):
        j,v,meta,stop=self.fixture()
        real=time.monotonic(); times=iter([real,real,real,real+2])
        with patch('brass_ai.rust_backend.lab_teacher_finalization.time.monotonic',side_effect=lambda:next(times)):
            with self.assertRaisesRegex(ValueError,'deadline'):v.verify(meta,real)
        self.assertEqual(v.cache[j['expected']['job_id']]['accepted'],[])


if __name__=='__main__':unittest.main()
