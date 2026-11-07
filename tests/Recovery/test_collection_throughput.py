"""Admitted input reuse, complete parallel rating validation and safe cleanup."""
import copy
import json
import os
import pickle
import random
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import numpy as np

from fixtures import position,game
from brass_ai.bank import sha,read
from brass_ai.learning.contracts import SOURCE_ROOT,digest,read_json,CONFIG_DIR,FEATURE_SCHEMA
from brass_ai.learning.board_topology import topology
from brass_ai.learning.batching import Request,accept_response
from brass_ai.recovery import inputs,ratings,retention,rating_ingress
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.admitted_inputs import request as admitted_request
from brass_ai.recovery.replay import write_shard,cache_receipt
from brass_ai.recovery.retention_digests import PlanDigests,METADATA_REUSE
from brass_ai.recovery.config import RETENTION,validate_config
from brass_ai.recovery.throughput_config import configure
from brass_ai.step5.league import seeds_for,write_json


def correct_topology(view):
    board=topology()
    profile=read_json(CONFIG_DIR/FEATURE_SCHEMA)['static_profile']
    for slot in view['slots']:slot['types']=list(profile['slot_types'][slot['id']])
    for merchant in view['merchants']:merchant['bonus']=profile['merchant_bonus'][merchant['id']]
    for link in view['links']:
        link['regions']=list(board['link_endpoints'][link['id']]);link['era']=board['link_eras'][link['id']]
    return view


def position_with_history(count=96):
    view,choices=position();correct_topology(view)
    view['history']=[dict(seat=i%4,era='CanalEra',choice=dict(choices[i%3],id=str(i))) for i in range(count)]
    return view,choices


def replay_fixture(directory,count=4):
    seeds,_=seeds_for(913,count,'validation');spec=dict(generation=10,config=dict(contract='fixture',seed=32),lineups=[])
    jobs=[];records=[]
    for index,seed in enumerate(seeds):
        g=game();g.update(game_id=f'game-{index}',split='validation',setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed)))
        for row in g['records']:correct_topology(row['view'])
        path=directory/f'game-{index}.jsonl.gz'
        write_shard(path,[g],dict(purpose='evaluation'))
        records.append(dict(job=index,game_id=g['game_id'],path=str(path),sha256=sha(path),features=cache_receipt(path)))
        jobs.append(dict(seed=seed,rotation=index%4,policies=[dict(policy_id='fixture')]*4))
        spec['lineups'].append(dict(seed=seed,seat=index%4,kind='rating',agents=['candidate','il','old1','old2']))
    return records,jobs,spec


class AdmittedInputs(unittest.TestCase):
    def test_policy_sampling_opening_counters_and_rng_are_identical(self):
        from test_inference_deadline_race import Queue
        from brass_ai.step5.policy import NeuralPolicy
        class AdmittedQueue(Queue):request_factory=staticmethod(admitted_request)
        outcomes=[]
        for fast in (False,True):
            view,choices=position_with_history()
            if fast:view,choices=inputs.BOUNDARY.admit(view,choices)
            rng=random.Random(17)
            policy=NeuralPolicy(None,evaluator=AdmittedQueue() if fast else Queue(),focus='neutral',
                exploration=.05,exploration_version=3,opening=dict(temperature=1.3,action_decisions_per_seat=3))
            rows=[]
            for _ in range(5):
                rows.append((policy.select_batch([(view,choices,rng)]),copy.deepcopy(policy.last_behavior[rng])))
            outcomes.append((rows,rng.getstate(),policy.action_counts[rng]))
        self.assertEqual(*outcomes)

    def test_snapshot_encoding_request_hash_and_rng_behavior_match(self):
        view,choices=position_with_history();before=copy.deepcopy((view,choices))
        frozen,fc=inputs.BOUNDARY.admit(view,choices)
        a=Request('a','f'*64,view,choices,time.monotonic()+60)
        b=admitted_request('a','f'*64,frozen,fc,a.deadline)
        self.assertEqual(a.input_digest,b.input_digest);self.assertIs(b.view,frozen)
        enc=PreparedEncoder();raw=enc.encode(view,choices);fast=enc.encode(frozen,fc)
        for name in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(raw,name),getattr(fast,name))
        self.assertEqual(raw.choice_ids,fast.choice_ids)
        with patch.object(inputs,'digest',side_effect=AssertionError('Rehashed admitted frame')):
            self.assertEqual(inputs.BOUNDARY.check(frozen,fc),digest(dict(view=view,choices=choices)))
            self.assertEqual(inputs.input_digest(frozen,fc),digest(dict(view=view,choices=choices)))
            self.assertIs(enc.encode(frozen,fc,validated=True),fast)
        view['players'][0]['cash']+=1;choices[0]['id']='changed'
        self.assertEqual((frozen,fc),before);self.assertEqual(b.digest(),a.input_digest)

    def test_request_digest_reuse_keeps_model_context_and_ipc_checks(self):
        view,choices=inputs.BOUNDARY.admit(*position_with_history())
        r=admitted_request('x','a'*64,view,choices,time.monotonic()+30,remaining=(.01,)*4)
        with patch.object(Request,'digest',side_effect=AssertionError('Rehashed immutable request')):
            self.assertEqual(r.digest(),r.input_digest)
        # Even unsupported low-level frozen-field replacement must invalidate
        # the cache; normal field/nested-container mutation is rejected outright.
        object.__setattr__(r,'expected_model','b'*64)
        self.assertNotEqual(r.digest(),r.input_digest)
        decoded=pickle.loads(pickle.dumps(r))
        self.assertIs(type(decoded.view),dict);before=decoded.digest()
        decoded.view['players'][0]['cash']+=1;self.assertNotEqual(decoded.digest(),before)

    def test_all_nested_mutations_rejected_and_ipc_loses_receipt(self):
        view,choices=position_with_history();v,c=inputs.BOUNDARY.admit(view,choices)
        operations=[lambda:v.update(extra=True),lambda:v['players'][0].update(cash=0),
            lambda:v['history'].append({}),lambda:c.reverse(),lambda:c[0].pop('id'),
            lambda:setattr(v,'_admission',None),lambda:v['players'].__setitem__(0,{}),
            lambda:v.__ior__({'extra':1}),lambda:c.__imul__(2)]
        for operation in operations:
            with self.assertRaises(TypeError):operation()
        copied=pickle.loads(pickle.dumps((v,c)))
        self.assertEqual(copied,(view,choices));self.assertIs(type(copied[0]),dict)
        self.assertIs(type(copied[0]['history']),list)
        from brass_ai.recovery.admitted_inputs import admitted_digest
        self.assertIsNone(admitted_digest(*copied));self.assertIsNone(admitted_digest(v,list(c)))
        copied[0]['players'][0]['cash']=999
        self.assertNotEqual(copied[0],v)

    def test_invalid_input_is_not_blessed_and_mutable_changes_are_detected(self):
        view,choices=position_with_history();inputs.BOUNDARY.check(view,choices)
        view['history'][31]['seat']=5
        with self.assertRaises(Exception):inputs.BOUNDARY.admit(view,choices)
        view,choices=position_with_history();old=inputs.input_digest(view,choices)
        choices[0]['id']='changed';self.assertNotEqual(inputs.input_digest(view,choices),old)


class RatingValidation(unittest.TestCase):
    def test_parallel_completion_order_keeps_exact_serial_rows_and_checks_receipts(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            records,jobs,spec=replay_fixture(Path(tmp))
            serial=[rating_ingress.validate_shard(([r],jobs,spec))[0][0][1] for r in records]
            with ThreadPoolExecutor(max_workers=4) as pool:
                stream=rating_ingress.RatingIngress(pool,jobs,spec)
                for record in reversed(records):stream.submit([record])
                rows,metrics=stream.finish(records)
            self.assertEqual(rows,serial);self.assertEqual(metrics['shards'],4)
            bad=copy.deepcopy(jobs);bad[0]['policies'][0]['policy_id']='foreign'
            with self.assertRaisesRegex(ValueError,'seat/setup'):rating_ingress.validate_shard(([records[0]],bad,spec))
            records[0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'replay changed'):rating_ingress.validate_shard(([records[0]],jobs,spec))

    def test_duplicate_or_unpublished_shards_rejected(self):
        from concurrent.futures import Future
        class Pool:
            def submit(self,*args):
                future=Future();future.set_result(([],dict(worker_seconds=0.,worker_cpu_seconds=0.)));return future
        stream=rating_ingress.RatingIngress(Pool(),[{}],{})
        record=dict(path='fixture',job=0)
        stream.submit([record])
        with self.assertRaisesRegex(ValueError,'Repeated'):stream.submit([record])
        with self.assertRaisesRegex(ValueError,'receipts differ'):stream.finish([])


class CleanupDigests(unittest.TestCase):
    def test_same_stat_stamp_cannot_reuse_digest_after_same_size_write(self):
        from brass_ai.recovery import retention_digests as module
        first=(1,2,4,100,100,0o100644,1)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'fixture';path.write_bytes(b'abcd')
            # Model the lab's coarse timestamps deterministically, without sleeps
            # or weakening the real-file test that caught the regression.
            with patch.object(module,'stamp',return_value=first):
                cache=PlanDigests();old=cache.digest(path)
                self.assertEqual(cache.digest(path),old)
                path.write_bytes(b'efgh')
                self.assertNotEqual(cache.digest(path),old)
                self.assertEqual(cache.digest(path),sha(path))
                self.assertEqual(cache.hits,0);self.assertEqual(cache.read_bytes,16)
        self.assertFalse(METADATA_REUSE)

    def test_reuse_only_unchanged_local_file_and_detect_same_size_edit(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'fixture';path.write_bytes(b'abcd');cache=PlanDigests()
            first=cache.digest(path);self.assertEqual(cache.digest(path),first);self.assertEqual(cache.hits,int(METADATA_REUSE))
            path.write_bytes(b'efgh');self.assertNotEqual(cache.digest(path),first)
            fresh=PlanDigests();self.assertEqual(fresh.digest(path),sha(path));self.assertEqual(fresh.hits,0)

    def test_file_changed_during_hash_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'fixture';path.write_bytes(b'abcd')
            def changed(p):
                value=sha(p);p.write_bytes(b'changed-length');return value
            with patch('brass_ai.recovery.retention_digests.sha',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'while hashing'):PlanDigests().digest(path)

    def test_prune_rehashes_before_move_and_final_delete(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            path=run/'orphan.pt';path.write_bytes(b'cleanup fixture')
            def plan(root,run,config,*,_digests):
                return dict(format='retention-plan/1.0.0',run=str(run),created=0,generation=0,
                    entries=[dict(path=path.name,sha256=_digests.digest(path),bytes=path.stat().st_size,reason='fixture')],
                    reclaimable_bytes=path.stat().st_size,protected=[])
            with patch.object(retention,'plan',side_effect=plan),patch.object(retention,'sha',wraps=sha) as final_hash:
                result=retention.prune(root,run,dict(RETENTION,quarantine_hours=0),apply=True)
            self.assertEqual(result['hash_reuse']['reuse_hits'],int(METADATA_REUSE))
            self.assertTrue(any('.retention-trash' in str(c.args[0]) for c in final_hash.call_args_list))
            self.assertEqual(result['permanently_deleted_files'],1);self.assertFalse(path.exists())

    def test_prune_rejects_same_size_changed_file_before_quarantine(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            path=run/'orphan.pt';path.write_bytes(b'abcd')
            def plan(root,run,config,*,_digests):
                expected=_digests.digest(path)
                path.write_bytes(b'efgh')
                return dict(format='retention-plan/1.0.0',run=str(run),created=0,generation=0,
                    entries=[dict(path=path.name,sha256=expected,bytes=4,reason='fixture')],
                    reclaimable_bytes=4,protected=[])
            with patch.object(retention,'plan',side_effect=plan),\
                    patch('brass_ai.recovery.retention_digests.stamp',return_value=(1,2,4,100,100,0o100644,1)):
                with self.assertRaisesRegex(ValueError,'changed during cleanup'):
                    retention.prune(root,run,dict(RETENTION,quarantine_hours=0),apply=True)
            self.assertEqual(path.read_bytes(),b'efgh')
            self.assertFalse((run/'.retention-trash').exists())


class Configuration(unittest.TestCase):
    def test_only_compute_workers_change_and_cpu_limits_hold(self):
        base=read(SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0921-20261105T213224468369Z/recorded-config.json')
        new=configure(base);validate_config(new)
        self.assertEqual(new['rating_compute']['workers'],16)
        restored=copy.deepcopy(new);restored['rating_compute']=base['rating_compute'];self.assertEqual(restored,base)
        self.assertEqual(configure(new),new)
        bad=copy.deepcopy(base);bad['audit_workers']=8
        with self.assertRaisesRegex(ValueError,'CPU slots'):configure(bad)


if __name__=='__main__':unittest.main()
