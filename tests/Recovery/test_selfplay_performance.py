"""Self-play ingress parity, bounded CUDA staging and committed-run cleanup."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.config import RETENTION,UPDATE_V21,validate_config
from brass_ai.recovery.diagnostics import summarize
from brass_ai.recovery.ingress import load_collection,reader_pool
from brass_ai.recovery.league import schedule_evaluations
from brass_ai.recovery.replay import write_shard,read_shards,cache_receipt
from brass_ai.recovery.retention import prune,purge,plan
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset,prepared,ppo_update,policy_measure
from brass_ai.step5.league import write_json
import test_revision as revision


class SelfplayPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_parallel_ingress_preserves_ppo_and_diagnostics(self):
        model,policy,game=revision.RevisionTests().onpolicy({1})
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            records=[];kinds={game['setup_group']:'selfplay'}
            for i in range(2):
                episode=copy.deepcopy(game);episode['game_id']=f'game-{i}'
                path=Path(temp)/f'{i}.jsonl.gz';write_shard(path,[episode],dict(purpose='training'))
                records.append(dict(path=str(path),job=i,game_id=episode['game_id'],sha256=sha(path),features=cache_receipt(path)))
            original=read_shards([r['path'] for r in records],partition='train',receipts={r['path']:r['features'] for r in records})
            with reader_pool(2) as pool:
                headers,data,metrics=load_collection(records,policy.policy_id,kinds,executor=pool,workers=2)
            self.assertEqual(metrics,summarize(original,{policy.policy_id},kinds))
            self.assertNotIn('records',headers[0]);self.assertNotIn('history',data[0][0]['view'])
            expected=prepared(original)
            for a,b in zip(expected,data):
                self.assertEqual(a[1:3],b[1:3])
                np.testing.assert_array_equal(a[3].state,b[3].state)
                np.testing.assert_array_equal(a[3].actions,b[3].actions)
            left,right=copy.deepcopy(model),copy.deepcopy(model)
            config=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2,critic_batch_size=4,epochs=1)
            torch.manual_seed(187)
            a=ppo_update(left,torch.optim.AdamW(left.parameters(),lr=1e-6),original,policy,config)
            torch.manual_seed(187)
            b=ppo_update(right,torch.optim.AdamW(right.parameters(),lr=1e-6),headers,policy,config,learner_data=data)
            self.assertEqual(a['epochs'],b['epochs'])
            self.assertTrue(all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items()))
            with self.assertRaisesRegex(ValueError,'Duplicate game'):
                load_collection(records+[dict(records[0],job=3)],policy.policy_id,kinds)
            Path(records[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'replay changed'):
                load_collection(records,policy.policy_id,kinds)

    def test_prefetched_exploration_and_full_likelihood_check(self):
        model,policy,game=revision.RevisionTests().onpolicy({1});ds=Dataset(prepared([game]),'cpu')
        parts=[[2,0],[1],[3]];cfg=dict(UPDATE_V21,microbatch_size=1)
        with patch.object(torch.Tensor,'pin_memory',lambda self:self),ThreadPoolExecutor(max_workers=2) as pool:
            for part,ix,inputs,q in ds.policy_batches(iter(parts),policy.exploration,pool):
                expected_ix,expected,expected_q=ds.batch(part,policy.exploration)
                self.assertTrue(torch.equal(ix,expected_ix) and torch.equal(q,expected_q))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(inputs,expected)))
            self.assertAlmostEqual(policy_measure(model,ds,range(4),policy,cfg,Timings(),True,executor=pool),0)
            ds.old[-1]*=.5
            with self.assertRaisesRegex(ValueError,'Wrong/stale'):
                policy_measure(model,ds,range(4),policy,cfg,Timings(),True,executor=pool)

    def test_many_commits_keep_few_files_and_protect_dependencies(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);cfg=dict(RETENTION,replay_rounds=2,quarantine_hours=0)
            initial=run/'initial.pt';initial.write_bytes(b'anchor')
            history=[dict(path=initial.relative_to(SOURCE_ROOT).as_posix(),model_id='initial')]
            pinned=run/'generation-9991.pt'
            write_json(run/'pins.json',dict(paths=[pinned.relative_to(SOURCE_ROOT).as_posix()]))
            # Cross the four-digit filename boundary; ordering must be numeric.
            for i in range(9991,10003):
                checkpoint=run/f'generation-{i:04d}.pt';checkpoint.write_bytes(str(i).encode())
                history.append(dict(path=checkpoint.relative_to(SOURCE_ROOT).as_posix(),model_id=str(i)))
                round_dir=run/f'iteration-{i-1:04d}';round_dir.mkdir()
                (round_dir/'games.jsonl.gz').write_bytes(b'replay')
                write_json(round_dir/'report.json',dict(format='step5-recovery-report/2.2.0',iteration=i-1,actor={'loss':i}))
                state=dict(format='step5-recovery-league/2.2.0',iteration=i,history=history,resume=history[-2:],
                    evaluation_anchor=history[0],population=[],signature=dict(config=dict(retention=cfg)))
                write_json(run/'state.json',state)
                prune(SOURCE_ROOT,run,cfg,apply=True)
            self.assertTrue(pinned.exists() and initial.exists())
            self.assertEqual(len(list(run.glob('generation-*.pt'))),3)
            self.assertEqual(len(list(run.glob('iteration-*'))),2)
            self.assertEqual({p.name for p in run.glob('iteration-*')},{'iteration-10000','iteration-10001'})
            self.assertEqual(len(read(run/'retention-history.json')),10)
            self.assertFalse(list((run/'.retention-trash').rglob('*')))
            self.assertLessEqual(len(list((run/'.retention').glob('*.json'))),2)
            before=set(run.rglob('*'));prune(SOURCE_ROOT,run,cfg,apply=True)
            self.assertEqual(before,set(run.rglob('*')))  # no empty cleanup receipts

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
    def test_cuda_ppo_with_prefetched_actor_critic_and_rehearsal(self):
        model,policy,game=revision.RevisionTests().onpolicy({1})
        data=prepared([game]);model.to('cuda')
        config=dict(UPDATE_V21,effective_batch_size=2,microbatch_size=1,critic_batch_size=2,epochs=1)
        report=ppo_update(model,torch.optim.AdamW(model.parameters(),lr=1e-6),[game],policy,config,
                          rehearsal=data,rehearsal_weight=.1)
        torch.cuda.synchronize()
        self.assertGreater(report['optimizer_steps'],0)
        self.assertGreater(report['rehearsal_steps'],0)
        self.assertGreater(report['memory']['cuda_allocated_bytes'],0)

    def test_quarantine_new_pin_and_interrupted_delete(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);path=run/'old.pt';path.write_bytes(b'old');receipt=run/'.retention/cleanup-1.json'
            quarantine=run/'.retention-trash/1/old.pt';quarantine.parent.mkdir(parents=True)
            entry=dict(path='old.pt',quarantine=quarantine.relative_to(run).as_posix(),bytes=3,sha256=sha(path),moved=True)
            path.replace(quarantine)
            write_json(receipt,dict(created=0,quarantine_hours=0,generation=0,status='quarantined',entries=[entry]))
            write_json(run/'pins.json',dict(paths=[path.relative_to(SOURCE_ROOT).as_posix()]))
            self.assertEqual(purge(SOURCE_ROOT,run)['rescued_dependencies'],1)
            self.assertEqual(path.read_bytes(),b'old')
            # A crash after unlink but before journal completion is recoverable.
            entry.update(path='unused.pt',quarantine='.retention-trash/2/unused.pt',deleting=True)
            write_json(receipt,dict(created=0,quarantine_hours=0,generation=0,status='quarantined',entries=[entry]))
            purge(SOURCE_ROOT,run)
            self.assertTrue(read(receipt)['entries'][0]['deleted'])

    def test_small_schedule_and_completed_development_cleanup(self):
        cfg=validate_config(read(SOURCE_ROOT/'configs/step5/lab.json'));schedule=cfg['evaluation_schedule']
        self.assertEqual((schedule['monitor_every_games'],schedule['monitor_setups']*4),(6400,4))
        self.assertEqual((schedule['broader_every_games'],schedule['broader_setups']*8),(25600,8))
        candidate=dict(model_id='latest',path='current.pt')
        state=dict(signature=dict(config=cfg),games_done=6399,evaluations=[],history=[candidate],
                   population=[dict(model_id='a',path='a.pt'),dict(model_id='b',path='b.pt')],evaluation_anchor=dict(model_id='a',path='a.pt'))
        with patch('brass_ai.recovery.evaluation.evaluate') as evaluate:
            schedule_evaluations(SOURCE_ROOT,Path('unused'),state,'cpu',None);evaluate.assert_not_called()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp)
            write_json(run/'state.json',dict(format='step5-recovery-league/2.2.0',iteration=4,history=[],signature={'config':{}},population=[]))
            for i in range(4):
                directory=run/'evaluations'/f'monitor-{i:04d}';directory.mkdir(parents=True)
                spec=dict(config=dict(panel='development',setups=1))
                write_json(directory/'spec.json',spec)
                write_json(directory/'report.json',dict(format='step5-evaluation-report/1.0.0',spec=spec,games=4))
                (directory/'game.jsonl.gz').write_bytes(b'example')
            formal=run/'evaluations/formal-0001';formal.mkdir()
            write_json(formal/'spec.json',dict(config=dict(panel='full',setups=30)))
            write_json(formal/'report.json',dict(spec=dict(config=dict(panel='full')),promotion_eligible=False))
            prune(SOURCE_ROOT,run,dict(RETENTION,quarantine_hours=0),apply=True)
            self.assertEqual(len(list((run/'evaluations').glob('monitor-*'))),2)
            self.assertTrue((formal/'report.json').exists())

    def test_new_manifest_pin_rescues_its_quarantined_dependency(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);source=run/'replay.jsonl.gz';source.write_bytes(b'replay')
            manifest=run/'dataset.json';write_json(manifest,dict(path=source.relative_to(SOURCE_ROOT).as_posix()))
            entries=[]
            # Dependency comes first in the journal: a newly pinned manifest
            # must protect it before any deletion, even with both in trash.
            for path in (source,manifest):
                destination=run/'.retention-trash/1'/path.name;destination.parent.mkdir(parents=True,exist_ok=True)
                entries.append(dict(path=path.name,quarantine=destination.relative_to(run).as_posix(),
                                    bytes=path.stat().st_size,sha256=sha(path),moved=True))
                path.replace(destination)
            write_json(run/'.retention/cleanup-1.json',dict(created=0,quarantine_hours=0,generation=0,status='quarantined',entries=entries))
            write_json(run/'pins.json',dict(paths=[manifest.relative_to(SOURCE_ROOT).as_posix()]))
            self.assertEqual(purge(SOURCE_ROOT,run)['rescued_dependencies'],2)
            self.assertEqual(source.read_bytes(),b'replay')


if __name__=='__main__':unittest.main()
