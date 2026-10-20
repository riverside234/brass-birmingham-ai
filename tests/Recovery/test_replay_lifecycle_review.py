"""Failure cleanup and shared admission for overlapped replay preparation."""
from contextlib import contextmanager
import copy
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.replay import split_for
from brass_ai.recovery.execution import ActorPool, _collection_recipe
from brass_ai.recovery.ingress import reader_pool, _prepare
from brass_ai.recovery.replay import write_shard
from brass_ai.step5.league import seeds_for


class ReplayLifecycleReviewTests(unittest.TestCase):
    def pool(self):
        pool=ActorPool.__new__(ActorPool)
        pool.service=Mock()
        pool.service.error=None
        pool.processes=[]
        pool.jobs=Mock();pool.results=Mock()
        pool.lanes=2
        return pool

    def test_collection_failure_aborts_shared_admission_before_unwinding(self):
        pool=self.pool()
        with patch.object(pool,'_collect',side_effect=ValueError('bad durable shard')):
            with self.assertRaisesRegex(ValueError,'bad durable shard'):
                pool.collect([dict(index=0)])
        pool.service.cpu.abort.assert_called_once_with()

    def test_failed_close_aborts_before_terminating_actors(self):
        pool=self.pool();events=[]
        process=Mock();process.is_alive.side_effect=[True,False]
        process.terminate.side_effect=lambda:events.append('terminate')
        pool.processes=[process]
        pool.service.cpu.abort.side_effect=lambda:events.append('abort')
        pool.close(failed=True)
        self.assertEqual(events,['abort','terminate'])

    def test_collection_surfaces_original_inference_error_after_admission_abort(self):
        pool=self.pool()
        pool.service.error=RuntimeError('Invalid inference utility budget')
        secondary=ValueError('CPU admission aborted')
        with patch.object(pool,'_collect',side_effect=secondary):
            with self.assertRaisesRegex(ValueError,'Inference service failed: Invalid inference utility budget') as raised:
                pool.collect([dict(index=0)])
        self.assertIs(raised.exception.__cause__,secondary)
        pool.service.cpu.abort.assert_called_once_with()

    def test_user_interrupt_is_not_replaced_by_service_error(self):
        pool=self.pool()
        pool.service.error=RuntimeError('actor exited')
        with patch.object(pool,'_collect',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):pool.collect([dict(index=0)])
        pool.service.cpu.abort.assert_called_once_with()

    def test_successful_collection_keeps_other_pools_alive(self):
        pool=self.pool();records=[dict(job=0)]
        with patch.object(pool,'_collect',return_value=records):
            self.assertEqual(pool.collect([dict(index=0)]),records)
        pool.service.cpu.abort.assert_not_called()

    def test_single_inline_reader_uses_and_restores_shared_admission(self):
        import brass_ai.recovery.ingress as ingress
        events=[]
        class Admission:
            @contextmanager
            def acquire(self,workers,timings):
                events.append(('acquire',workers))
                try:yield
                finally:events.append(('release',workers))
        previous=ingress._admission
        with patch('brass_ai.recovery.ingress._prepare_inner',side_effect=lambda task:events.append(task) or 17):
            with reader_pool(1,admission=Admission()) as reader:
                self.assertIsNone(reader)
                self.assertEqual(_prepare('prepared'),17)
        self.assertIs(ingress._admission,previous)
        self.assertEqual(events,[('acquire',1),'prepared',('release',1)])

    def test_inline_reader_restores_admission_after_validation_failure(self):
        import brass_ai.recovery.ingress as ingress
        previous=ingress._admission
        with self.assertRaisesRegex(ValueError,'validation failed'):
            with reader_pool(1,admission=object()):
                raise ValueError('validation failed')
        self.assertIs(ingress._admission,previous)

    def test_partial_collection_reuses_durable_shard_and_publishes_complete_membership(self):
        from review_fixture import current_game as game
        pool=self.pool();pool.width=pool.lanes=1;pool.heavy=False
        pool.config=dict(shard_games=2,feature_cache=False)
        pool.results=queue.Queue();pool.start=Mock()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            directory=Path(temp);seeds=seeds_for(17100000,2)[0]
            jobs=[dict(index=i,seed=seed,path=str(directory/f'actor-{i}.jsonl.gz'),policies=[dict(kind='random')]*4,
                provenance=dict(purpose='training',seed=seed,policy='frozen-behavior')) for i,seed in enumerate(seeds)]
            recipe=_collection_recipe(jobs,pool.config)
            identity=digest(recipe);prefix=directory/('games-'+identity[:16])
            def publish(job,path):
                episode=copy.deepcopy(game());episode['game_id']='partial-'+str(job['index'])
                episode['setup_group']=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
                episode['split']=split_for(episode['setup_group'])
                write_shard(path,[episode],dict(purpose='training',collection=identity,recipe=recipe,
                    jobs=[dict(index=job['index'],seed=job['seed'],provenance=job['provenance'])]),cache=False)
                return dict(job=job['index'],game_id=episode['game_id'],path=str(path),sha256=sha(path),cached=False,features=None)
            original=Path(str(prefix)+'-part-0000.jsonl.gz');publish(jobs[0],original)
            original_hash=sha(original);packets=[];callbacks=[]
            def enqueue(packet):
                packets.append(packet)
                self.assertEqual(packet['jobs'],[jobs[1]])
                path=Path(packet['prefix']+f'-part-{packet["part_start"]:04d}.jsonl.gz')
                pool.results.put(dict(records=[publish(jobs[1],path)]))
            pool.jobs.put.side_effect=enqueue
            result=pool.collect(jobs,on_records=lambda records:callbacks.append(records))
            self.assertEqual([r['job'] for r in result],[0,1])
            self.assertEqual([r['cached'] for r in result],[True,False])
            self.assertEqual([[r['job'] for r in group] for group in callbacks],[[0],[1]])
            self.assertEqual(packets[0]['part_start'],1)
            self.assertEqual(sha(original),original_hash)
            self.assertEqual(len(list(directory.glob('*.jsonl.gz'))),2)
            pool.service.cpu.abort.assert_not_called()


if __name__=='__main__':unittest.main()
