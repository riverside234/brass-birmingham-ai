"""Regression checks for the pre-lab audit and CUDA synchronization fixes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery.demonstrations import _audit_path, audit
from brass_ai.recovery.replay import write_shard
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.updates import Dataset,prepared,supervised_epoch


class LabReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_verification_entry_point_is_safe_for_spawn_imports(self):
        import runpy
        with patch('argparse.ArgumentParser.parse_args',side_effect=AssertionError('Spawn imported the CLI')):
            module=runpy.run_path(str(SOURCE_ROOT/'scripts/verify-step5.py'),run_name='__mp_main__')
        self.assertTrue(callable(module['main']))

    def test_serial_audit_commits_only_complete_shards_and_deduplicates(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            directory=Path(temp)
            entry=dict(path='valid.jsonl.gz',game_id='valid',semantic_sha256='same-game',setup_group='group',split='train')
            batches=[([], [dict(path='bad.jsonl.gz',reason='broken')]),([entry],[]),([entry],[])]
            with patch('brass_ai.recovery.demonstrations._audit_path',side_effect=batches):
                result=audit(SOURCE_ROOT,[directory/(str(i)+'.jsonl.gz') for i in range(3)],directory/'manifest.json',workers=1)
            self.assertEqual(result['entries'],[entry])
            self.assertEqual(len(result['excluded']),2)
            self.assertEqual(result['partitions'],dict(train=1))

    def test_corrupt_shards_are_excluded_instead_of_crashing_parallel_audit(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            for name,data in [('corrupt',b'not a gzip file'),('truncated',b'\x1f\x8b'),('empty',b'')]:
                path=Path(temp)/(name+'.jsonl.gz');path.write_bytes(data)
                accepted,excluded=_audit_path((str(SOURCE_ROOT),str(path),['fixture'],None))
                self.assertEqual(accepted,[]);self.assertEqual(len(excluded),1)

    def test_neural_rows_never_enter_a_heuristic_audit_manifest(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'mixed.jsonl.gz';episode=game()
            for row in episode['records']:row['behavior']['kind']='external'
            episode['records'][1]['behavior']['kind']='model'
            write_shard(path,[episode],dict(purpose='training',seed=7),cache=False)
            with patch('brass_ai.recovery.demonstrations.RulesHost',side_effect=AssertionError('Do not replay forbidden data')):
                accepted,excluded=_audit_path((str(SOURCE_ROOT),str(path),['fixture'],None))
            self.assertEqual(accepted,[])
            self.assertIn('prohibits neural behavior',excluded[0]['reason'])

    def test_bc_eligibility_needs_no_per_row_device_scalar_reads(self):
        episode=game()
        for row in episode['records'][:2]:row['choices']=row['choices'][:1]
        model=FourSeatNet(ModelConfig.preset('small'))
        dataset=Dataset(prepared([episode]),'cpu',policy=False)
        config=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=1)
        expected=supervised_epoch(model,None,dataset,config,training=False)
        mask=dataset.forced
        class VectorReadsOnly:
            def __getitem__(self,index):
                if not torch.is_tensor(index):raise AssertionError('Per-row CUDA synchronization')
                return mask[index]
        dataset.forced=VectorReadsOnly()
        actual=supervised_epoch(model,None,dataset,config,training=False)
        self.assertEqual(actual['meaningful_decisions'],2)
        self.assertEqual(actual['loss'],expected['loss'])
        self.assertEqual(actual['natural_policy_loss'],expected['natural_policy_loss'])


if __name__=='__main__':unittest.main()
