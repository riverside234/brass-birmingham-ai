"""Parallel BC ingress, bounded batches, progress and the explicit fresh handoff."""
import contextlib
import copy
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixtures import game
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.replay import split_for
from brass_ai.recovery import demonstrations as demos
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.progress import Progress
from brass_ai.recovery.replay import write_shard,cache_receipt
from brass_ai.recovery.updates import Dataset,prepared,supervised_epoch
from brass_ai.step5.league import seeds_for,write_json


class ImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture_manifest(self,directory):
        teacher=next(iter(read(SOURCE_ROOT/'training/opponents/pool.json')['training']))
        entries=[]
        for i,partition in enumerate(('train','train','validation')):
            seed=seeds_for(75000+i*100,1,partition)[0][0];episode=game()
            episode.update(game_id=f'fixture-{i}',setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed)),opponents=[teacher]*4)
            episode['split']=split_for(episode['setup_group'])
            for row in episode['records']:row['behavior'].update(kind='external',policy_id=teacher)
            path=directory/f'{i}.jsonl.gz';write_shard(path,[episode],dict(purpose='training',seed=seed))
            entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),game_id=episode['game_id'],
                setup_group=episode['setup_group'],split=partition,transitions_verified=True,features=cache_receipt(path)))
        manifest=directory/'manifest.json'
        write_json(manifest,dict(format='bc-dataset/1.0.0',compatibility=compatibility(),label_kind='qualified-v3-choice/1.0.0',
            context='neutral-predecision/1.0.0',teachers=[teacher],entries=entries))
        return manifest

    def test_parallel_compact_load_preserves_targets_features_and_split(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            manifest=self.fixture_manifest(Path(temp))
            serial,_=demos.load_dataset(SOURCE_ROOT,manifest,'train')
            parallel,_=demos.load_dataset(SOURCE_ROOT,manifest,'train',workers=2,compact=True)
            self.assertEqual(len(parallel),8)
            for a,b in zip(serial,parallel):
                self.assertEqual(a[1:3],b[1:3]);self.assertEqual(a[0]['chosen'],b[0]['chosen'])
                self.assertNotIn('own_hand',b[0]['view'])
                for field in ('state','actions','stable_to_relative'):
                    np.testing.assert_array_equal(getattr(a[3],field),getattr(b[3],field))
            model=FourSeatNet(ModelConfig.preset('small'));cfg=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2)
            left=supervised_epoch(model,None,serial,cfg,training=False)
            right=supervised_epoch(model,None,parallel,cfg,training=False)
            for key in ('loss','natural_policy_loss','by_teacher_era_phase_action','meaningful_top1'):
                self.assertEqual(left[key],right[key])
            path=SOURCE_ROOT/read(manifest)['entries'][0]['path'];path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):
                demos.load_dataset(SOURCE_ROOT,manifest,'train',workers=2,compact=True)

    def test_progress_milestones_and_prefetch_order(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            progress=Progress('fixture',7)
            for i in range(1,8):progress(i)
        for percent in (20,40,60,80,100):self.assertEqual(output.getvalue().count(f': {percent}%'),1)
        dataset=Dataset(prepared([game()]),'cpu',policy=False);parts=[[2,0],[3],[1]]
        # Exercise the queued path on CPU without requiring a CUDA installation.
        with patch.object(torch.Tensor,'pin_memory',lambda self:self),ThreadPoolExecutor(max_workers=2) as pool:
            actual=list(dataset.supervised_batches(iter(parts),pool))
        self.assertEqual([p for p,_,_ in actual],parts)
        for part,ix,inputs in actual:
            expected_ix,expected,_=dataset.batch(part)
            self.assertTrue(torch.equal(ix,expected_ix))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(inputs,expected)))

    def test_completed_bc_handoff_rejects_unfinished_or_changed_candidate(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            directory=Path(temp);manifest=self.fixture_manifest(directory);run=directory/'bc'
            train=game();validation=copy.deepcopy(train);validation['game_id']='validation'
            rows={'train':prepared([train],neutral=True),'validation':prepared([validation],neutral=True)}
            def load(root,path,partition,**kwargs):return rows[partition],{}
            with patch.object(demos,'load_dataset',side_effect=load),contextlib.redirect_stdout(io.StringIO()):
                report=demos.train_bc(SOURCE_ROOT,run,manifest,None,dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2),
                    epochs=1,min_epochs=1,patience=1)
            model,origin=demos.completed_imitation(SOURCE_ROOT,run)
            self.assertEqual(origin['sha256'],report['candidate_sha256'])
            self.assertTrue((run/'initial.pt').exists());self.assertEqual(model.parameter_count,1840109)
            state=read(run/'state.json');bad=copy.deepcopy(state);bad['epoch']=0;write_json(run/'state.json',bad)
            with self.assertRaisesRegex(ValueError,'not completed'):demos.completed_imitation(SOURCE_ROOT,run)
            write_json(run/'state.json',state)
            path=SOURCE_ROOT/origin['path'];path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):demos.completed_imitation(SOURCE_ROOT,run)

    def test_new_dataset_continuation_keeps_parent_and_resets_optimizer(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            directory=Path(temp);manifest=self.fixture_manifest(directory)
            second=directory/'new-manifest.json';write_json(second,dict(read(manifest),dataset_revision=2))
            parent,child=directory/'round-1',directory/'round-2'
            train=game();validation=copy.deepcopy(train);validation['game_id']='validation'
            rows={'train':prepared([train],neutral=True),'validation':prepared([validation],neutral=True)}
            cfg=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2)
            with patch.object(demos,'load_dataset',side_effect=lambda root,path,split,**kwargs:(rows[split],{})),contextlib.redirect_stdout(io.StringIO()):
                demos.train_bc(SOURCE_ROOT,parent,manifest,None,cfg,epochs=1,min_epochs=1,patience=1)
                parent_sha=sha(parent/'best.pt')
                demos.train_bc(SOURCE_ROOT,child,second,None,cfg,epochs=1,min_epochs=1,patience=1,from_imitation_run=parent)
                # Same-dataset resume is permitted; changing an existing run's
                # manifest is not a silent continuation or validation reset.
                demos.train_bc(SOURCE_ROOT,child,second,None,cfg,epochs=1,min_epochs=1,patience=1,from_imitation_run=parent)
                with self.assertRaisesRegex(ValueError,'settings/source changed'):
                    demos.train_bc(SOURCE_ROOT,child,manifest,None,cfg,epochs=1,min_epochs=1,patience=1,from_imitation_run=parent)
            _,origin=demos.completed_imitation(SOURCE_ROOT,child)
            self.assertEqual(origin['manifest_sha256'],sha(second))
            self.assertEqual(sha(parent/'best.pt'),parent_sha)
            previous=torch.load(parent/'best.pt',map_location='cpu',weights_only=True)
            initial=torch.load(child/'initial.pt',map_location='cpu',weights_only=True)
            self.assertEqual(previous['model_id'],initial['model_id']);self.assertEqual(initial['optimizer']['state'],{})
            self.assertEqual(read(child/'state.json')['signature']['parent_imitation']['sha256'],parent_sha)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
    def test_cuda_prefetch_trains_on_gpu(self):
        model=FourSeatNet(ModelConfig.preset('small')).to('cuda')
        data=Dataset(prepared([game()]),'cuda',policy=False)
        cfg=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2)
        report=supervised_epoch(model,torch.optim.AdamW(model.parameters(),lr=1e-4),data,cfg)
        self.assertEqual(report['optimizer_steps'],1)
        self.assertGreater(report['memory']['cuda_allocated_bytes'],0)


if __name__=='__main__':unittest.main()
