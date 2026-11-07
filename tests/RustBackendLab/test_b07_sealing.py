"""Immutable teacher handoff, strict fallback and per-table deadline regressions."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery.admitted_inputs import FrozenDict
from brass_ai.recovery.search_value_config import adopt
from brass_ai.recovery.search_value_phase import SearchValuePhase
from brass_ai.recovery.teacher_buffer import TeacherBuffer
from brass_ai.recovery.teacher_phase_config import TIMING
from brass_ai.rust_backend import lab_teacher_finalization as F
from test_b025 import config,example,model_fixture
from test_b045 import value_fixture


class Sealing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def stream(self):
        model=model_fixture().eval();_,job,_=value_fixture(model)
        label,_=example(model,job=job,round=job['reservations']['0']['canal_early'])
        original=copy.deepcopy(label.row);expected=copy.deepcopy(original)
        v=F.TurnVerifier(model,[dict(generation=10,model_id=model_id(model))],[job],threading.Event())
        rows=[]
        for i in range(1,65):
            meta=dict(job=job['expected'],timing=TIMING,index=i,profile='full' if i==1 else 'fast',
                      remaining_ns=950_000_000 if i==1 else 490_000_000,
                      native_post_ns=10_000_000,search_ns=800_000_000,rows=[original] if i==1 else [])
            v.verify(meta,time.monotonic());rows.extend(meta['rows'])
        result=dict(job_id=job['expected']['job_id'],rows=rows,metrics=dict(searched_turns=64,reserved_turns=16))
        return model,job,v,result,original,expected

    def test_live_handoff_reuses_only_exact_immutable_rows(self):
        model,job,v,result,original,expected=self.stream()
        self.assertIs(type(result['rows'][0]),FrozenDict)
        original['view']['players'][0]['cash']+=1
        self.assertEqual(result['rows'][0],expected)
        with self.assertRaises(TypeError):result['rows'][0]['teacher']['counts'][0]=999
        with patch.object(F,'digest',side_effect=AssertionError('Live immutable path reserialized')):
            selected,_=v.take(result,job,model)
        self.assertIs(copy.deepcopy(selected[0].row),selected[0].row)
        with self.assertRaises(ValueError):v.take(result,job,model)

    def test_fast_allowance_is_half_second_and_still_bounded(self):
        from brass_ai.recovery.search_value_config import fast_profile
        self.assertEqual(fast_profile({})['post_search_seconds'],.5)
        model,job,_,_,_,_=self.stream()
        v=F.TurnVerifier(model,[],[job],threading.Event())
        meta=dict(job=job['expected'],timing=TIMING,index=1,profile='fast',
                  remaining_ns=500_000_001,native_post_ns=0,search_ns=800_000_000,rows=[])
        with self.assertRaisesRegex(ValueError,'Invalid post-search'):v.verify(meta,time.monotonic())
        meta['remaining_ns']=490_000_001;meta['native_post_ns']=10_000_000
        with self.assertRaisesRegex(ValueError,'Invalid post-search'):v.verify(meta,time.monotonic())

    def test_plain_reload_uses_strict_canonical_fallback_and_rejects_type_drift(self):
        model,job,v,result,_,_=self.stream()
        result=json.loads(json.dumps(result))
        with patch.object(F,'digest',wraps=digest) as hashed:v.take(result,job,model)
        self.assertEqual(hashed.call_count,2)
        model,job,v,result,_,_=self.stream();result=json.loads(json.dumps(result))
        # Python equality regards 0 == False. The serialized fallback must not.
        field=result['rows'][0]['teacher'];field['epoch']=float(field['epoch'])
        with self.assertRaisesRegex(ValueError,'Changed/unverified'):v.take(result,job,model)

    def test_freezing_is_inside_deadline_and_cancel_still_retires_handoff(self):
        model,job,v,result,_,_=self.stream();v.cancel.set()
        with self.assertRaisesRegex(ValueError,'Retired/changed'):v.take(result,job,model)
        v=F.TurnVerifier(model,[],[job],threading.Event());clock=[0.]
        from brass_ai.recovery.admitted_inputs import _freeze
        def slow(value):clock[0]=2.;return _freeze(value)
        meta=dict(job=job['expected'],timing=TIMING,index=1,profile='full',remaining_ns=900_000_000,
                  native_post_ns=10_000_000,search_ns=800_000_000,rows=[{'fixture':1}])
        with patch.object(F,'admit_teacher_rows',return_value=([],{})),patch('time.monotonic',side_effect=lambda:clock[0]),\
             patch('brass_ai.recovery.admitted_inputs._freeze',side_effect=slow):
            with self.assertRaisesRegex(ValueError,'deadline exceeded'):v.verify(meta,0.)
        self.assertEqual(v.cache[job['expected']['job_id']]['last'],0)

    def test_publication_features_are_detached_and_cold_rows_identical(self):
        model,job,v,result,_,expected=self.stream();rows,_=v.take(result,job,model)
        before=copy.deepcopy(rows[0].encoded)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            b=TeacherBuffer(SOURCE_ROOT,Path(tmp),'fixture')
            b.publish(10,'train',rows,model_id=model_id(model),reward_id=job['expected']['reward_id'])
            rows[0].encoded.state.setflags(write=True);rows[0].encoded.state[0]+=1
            self.assertEqual(b.rows(10)[0][1].row,expected)
            np.testing.assert_array_equal(b.rows(10)[0][1].encoded.state,before.state)
            cold=TeacherBuffer(SOURCE_ROOT,Path(tmp),'fixture',receipts=b.pending)
            self.assertEqual(cold.rows(10)[0][1].row,expected)
            np.testing.assert_array_equal(cold.rows(10)[0][1].encoded.state,before.state)

    def test_final_cache_publication_crossing_deadline_cannot_be_taken(self):
        model,job,v,result,_,_=self.stream();key=job['expected']['job_id']
        v.cache[key]['last']=63
        meta=dict(job=job['expected'],timing=TIMING,index=64,profile='fast',remaining_ns=490_000_000,
                  native_post_ns=10_000_000,search_ns=800_000_000,rows=[])
        with patch('time.monotonic',side_effect=[0.]*6+[.5]):
            with self.assertRaisesRegex(ValueError,'deadline exceeded'):v.verify(meta,0.)
        self.assertNotIn(key,v.cache)
        with self.assertRaisesRegex(ValueError,'Changed/unverified'):v.take(result,job,model)

    def test_frozen_label_ages_save_and_reload_with_strict_checkpoint_loader(self):
        from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
        model,job,_,result,_,_=self.stream()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            options=adopt(config())['teacher_phase']['search_value']
            value_dir=Path(tmp)/'values'
            phase=SearchValuePhase(value_dir,model,options,job['expected']['reward_id'])
            phase.label_ages=[dict(generation=10,reward_id=job['expected']['reward_id'],rows=result['rows'])]
            payload=phase.payload()
            def plain(v):
                if isinstance(v,dict):
                    self.assertIs(type(v),dict)
                    for x in v.values():plain(x)
                elif isinstance(v,list):
                    self.assertIs(type(v),list)
                    for x in v:plain(x)
            plain(payload['label_ages'])
            self.assertEqual(digest(payload['label_ages']),digest(phase.label_ages))
            path=Path(tmp)/'shadow.pt'
            save_checkpoint(path,model,step=10,training={'search_value':payload})
            restored,saved=load_checkpoint(path)  # weights_only=True, no allowlist
            cold=SearchValuePhase(value_dir,restored,options,job['expected']['reward_id'],saved=saved['training']['search_value'])
            self.assertEqual(cold.payload()['label_ages'],payload['label_ages'])
            payload['label_ages'][0]['rows'][0]['teacher']['epoch']+=1
            self.assertNotEqual(digest(payload['label_ages']),digest(phase.label_ages))
            # Keep the production directory-integrity check that caught the old
            # fixture placing shadow.pt in the shard directory.
            value_dir.mkdir(exist_ok=True);(value_dir/'unexpected.pt').write_bytes(b'foreign')
            with self.assertRaisesRegex(ValueError,'Unknown/uncommitted'):
                SearchValuePhase(value_dir,restored,options,job['expected']['reward_id'],saved=saved['training']['search_value'])

    def test_each_table_pays_own_seal_and_no_cap_is_relaxed(self):
        model=model_fixture();row,job,_=value_fixture(model,split='heldout')
        second=copy.deepcopy(job);second['expected'].update(job_id='second',game_id='second-game')
        tables=[dict(job_id=j['expected']['job_id'],value_rows=[],result={},final_spent=[0.]*4,
                    metrics=dict(turns=[dict(profile='fast',post_search_ns=20_000_000)])) for j in (job,second)]
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            phase=SearchValuePhase(tmp,model,adopt(config())['teacher_phase']['search_value'],row['reward_id'])
            with patch('brass_ai.recovery.search_value_phase.complete_game',side_effect=lambda *a,**k:[dict(row)]),\
                 patch('brass_ai.learning.reward.terminal_utility',return_value=np.zeros(4)),patch.object(phase.buffer,'publish') as publish:
                spent={job['expected']['job_id']:.10,'second':.01}
                with patch('time.monotonic',side_effect=[0.,0.,.05,.05,.17,.17]):phase.admit(tables,[job,second],10,post_spent=spent)
                self.assertEqual(publish.call_count,2)
                for bad in ({},dict(spent,second=-1),dict(spent,second=float('nan')),dict(spent,second=True)):
                    with self.assertRaisesRegex(ValueError,'post-search accounting'):phase.admit(tables,[job,second],10,post_spent=bad)
                with patch('time.monotonic',side_effect=[0.,0.,.39]):
                    with self.assertRaisesRegex(ValueError,'limit=0.5s'):phase.admit(tables[:1],[job],10,post_spent=.10)


if __name__=='__main__':unittest.main()
