"""Exact IL extension, immutable recipe admission, cache reuse and completion."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery import demonstrations as demos, imitation_extension as extension
from brass_ai.recovery import prepared_store as store
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.learning_rate import KEY
from brass_ai.step5.cli import parser
from brass_ai.step5.league import implementation_id, write_json
import test_bounded_imitation as bounded_fixtures


class ExtensionAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        signature=dict(epochs=10,manifest_path='data.json',manifest_sha256='a'*64,
                       implementation=implementation_id(),patience=3,min_epochs=3,lr=6e-5,
                       architecture={'version':11.1},update={'precision':'fp32'})
        state=dict(signature=signature,epoch=10,stopped=False,bad_epochs=0,best_loss=1.7378614,
                   last=dict(path='epoch-0010.pt',sha256='b'*64),best={'path':'epoch-0010.pt'},
                   history=[{'epoch':i} for i in range(1,11)],parent=None)
        return state,copy.deepcopy(signature)

    def test_total_twenty_idempotent_and_all_old_state_preserved(self):
        state,requested=self.fixture()
        state.update(bad_epochs=2,best={'path':'epoch-0008.pt'})
        before=copy.deepcopy(state)
        result=extension.resume_state(SOURCE_ROOT,state,requested,20)
        self.assertEqual(state,before)
        self.assertEqual({k:v for k,v in result.items() if k!='epoch_extensions'},before)
        self.assertEqual(extension.epoch_limit(result),20)
        self.assertEqual(len(result['epoch_extensions']),1)
        self.assertEqual(extension.resume_state(SOURCE_ROOT,result,requested,20),result)
        self.assertEqual(extension.resume_state(SOURCE_ROOT,result,requested,None),result)
        self.assertIsNone(extension.checkpoint_extension(result,10))
        self.assertEqual(extension.checkpoint_extension(result,11),digest(result['epoch_extensions']))
        args=parser().parse_args(['imitate','--epochs','10','--extend-to-epochs','20','--patience','3'])
        self.assertEqual((args.epochs,args.extend_to_epochs,args.patience),(10,20,3))

    def test_unknown_source_changed_recipe_and_invalid_caps_rejected(self):
        state,requested=self.fixture()
        for field,value in [('epochs',20),('patience',2),('lr',3e-5),
                            ('manifest_path','other.json'),('manifest_sha256','f'*64),
                            ('architecture',{'version':11}),('update',{'precision':'bf16'})]:
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'settings/source changed'):
                extension.resume_state(SOURCE_ROOT,state,dict(requested,**{field:value}),20)
        old=copy.deepcopy(state);old['signature']['implementation']='f'*64
        with self.assertRaisesRegex(ValueError,'pinned source review'):
            extension.resume_state(SOURCE_ROOT,old,requested,20)
        for cap in (0,-1,True,10,9,20.0):
            with self.subTest(cap=cap),self.assertRaises(ValueError):
                extension.resume_state(SOURCE_ROOT,state,requested,cap)
        with self.assertRaisesRegex(ValueError,'Finish the existing'):
            extension.resume_state(SOURCE_ROOT,dict(state,epoch=9),requested,20)
        with self.assertRaisesRegex(ValueError,'stopped by patience'):
            extension.resume_state(SOURCE_ROOT,dict(state,stopped=True),requested,20)

    def test_exact_review_and_cache_aliases_require_current_source(self):
        state,requested=self.fixture();review=extension.source_review()
        self.assertIsNotNone(review)
        for previous in review['from_implementations']:
            old=copy.deepcopy(state);old['signature']['implementation']=previous
            self.assertEqual(extension.epoch_limit(extension.resume_state(SOURCE_ROOT,old,requested,20)),20)
            with self.assertRaisesRegex(ValueError,'settings/source changed'):
                extension.resume_state(SOURCE_ROOT,old,requested,None)
        with patch.object(extension,'implementation_id',return_value='f'*64):
            self.assertIsNone(extension.source_review())
            self.assertEqual(extension.equivalent_preparations(store.FORMAT,'validator'),[])

    def test_completed_extension_uses_handoff_review_and_replay_effective_limit(self):
        from latest_model_fixture import imitation_fixture
        from test_il_replay import helper
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
            run,_,_=imitation_fixture(Path(folder))
            state=read(run/'state.json');state['last']=copy.deepcopy(state['best'])
            extended=extension.resume_state(SOURCE_ROOT,state,copy.deepcopy(state['signature']),2)
            write_json(run/'state.json',extended)
            # Neither consumer may admit the old completed-epoch report now.
            with self.assertRaises(ValueError):demos.completed_imitation(SOURCE_ROOT,run)
            with self.assertRaises(ValueError):helper.dataset_source(SOURCE_ROOT,run)
            extended['epoch']=2;write_json(run/'state.json',extended)
            report=read(run/'report.json');report.update(epochs=2,epoch_limit=2,epoch_extensions=extended['epoch_extensions'])
            write_json(run/'report.json',report)
            self.assertEqual(helper.dataset_source(SOURCE_ROOT,run)[0]['name'],run.name)
            current=implementation_id();production=demos.read
            def handoff_review(path):
                data=production(path)
                if Path(path).name=='imitation-handoff-sources.json':
                    data['compatible_sources'].append(dict(implementation=current,
                        learning_compatibility=extension.source_review()['learning_compatibility']))
                return data
            # A future weights-only handoff review is independent of permission
            # to resume Adam across that future source change.
            with patch.object(demos,'implementation_id',return_value='9'*64):
                with self.assertRaisesRegex(ValueError,'reviewed compatible source'):
                    demos.completed_imitation(SOURCE_ROOT,run)
                with patch.object(demos,'read',side_effect=handoff_review),patch.object(extension,'source_allowed',side_effect=AssertionError('not optimizer continuation')):
                    _,origin=demos.completed_imitation(SOURCE_ROOT,run)
            self.assertEqual(origin['sha256'],state['best']['sha256'])


class ExactExtensionTests(unittest.TestCase):
    # Native-board fixtures avoid obsolete topology shortcuts. Tests run only
    # 8 train / 4 validation rows, inside disposable artifact directories.
    setUpClass=classmethod(bounded_fixtures.BoundedImitationTests.setUpClass.__func__)
    setUp=bounded_fixtures.BoundedImitationTests.setUp

    def train(self,run,epochs=1,extend=None,**overrides):
        cfg=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2)
        settings=dict(epochs=epochs,min_epochs=1,patience=3,workers=1,
                      warmup_steps=2,extend_to_epochs=extend)
        settings.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return demos.train_bc(SOURCE_ROOT,run,self.manifest,None,cfg,**settings)

    def equal_nested(self,a,b):
        if isinstance(a,torch.Tensor):
            self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.equal_nested(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.equal_nested(x,y)
        else:self.assertEqual(a,b)

    def test_real_update_matches_uninterrupted_after_committed_interruption(self):
        full=self.directory/'full';split=self.directory/'split'
        self.train(full,epochs=3)
        with patch.object(demos,'implementation_id',return_value=extension.source_review()['from_implementations'][0]):
            initial=self.train(split)
        first_state=read(split/'state.json')
        first_sha=sha(split/'epoch-0001.pt')
        production=demos.supervised_epoch
        def interrupted(*args,**kwargs):
            if kwargs.get('progress','').startswith('imitation epoch 3/3 train'):
                raise RuntimeError('injected after epoch 2 commit')
            return production(*args,**kwargs)
        with patch.object(demos,'supervised_epoch',interrupted):
            with self.assertRaisesRegex(RuntimeError,'epoch 2 commit'):self.train(split,extend=3)
        state=read(split/'state.json')
        self.assertEqual(state['epoch'],2)
        self.assertEqual(state['signature'],first_state['signature'])
        self.assertEqual(state['history'][0],initial['history'][0])
        with self.assertRaises(ValueError):demos.completed_imitation(SOURCE_ROOT,split)
        result=self.train(split,extend=3)
        self.assertEqual(result['epochs'],3);self.assertEqual(result['epoch_limit'],3)
        self.assertEqual(len(read(split/'state.json')['epoch_extensions']),1)
        self.assertEqual(sha(split/'epoch-0001.pt'),first_sha)
        for part in result['preparation'].values():self.assertEqual(part['cache_hits'],part['shards'])
        left=torch.load(full/'last.pt',map_location='cpu',weights_only=True)
        right=torch.load(split/'last.pt',map_location='cpu',weights_only=True)
        for key in ('state_dict','optimizer','rng','cuda_rng','step'):
            self.equal_nested(left[key],right[key])
        warmup=right['optimizer']['param_groups'][0][KEY]
        self.assertEqual(warmup['completed_steps'],6)
        self.assertEqual(right['optimizer']['param_groups'][0]['lr'],6e-5)
        _,origin=demos.completed_imitation(SOURCE_ROOT,split)
        self.assertEqual(origin['sha256'],result['candidate_sha256'])
        before={p.name:sha(p) for p in split.glob('*.pt')};before_state=read(split/'state.json')
        repeated=self.train(split,extend=3)
        self.assertEqual(repeated['history'],result['history'])
        self.assertEqual(before_state,read(split/'state.json'))
        self.assertEqual(before,{p.name:sha(p) for p in split.glob('*.pt')})

    def test_original_best_and_patience_survive_extension_and_handoff(self):
        run=self.directory/'plateau';production=demos.supervised_epoch
        def increasing(*args,**kwargs):
            result=production(*args,**kwargs)
            if kwargs.get('training') is False:
                epoch=int(kwargs['progress'].split()[2].split('/')[0])
                result[0]['loss']=float(epoch)
            return result
        with patch.object(demos,'supervised_epoch',increasing):
            first=self.train(run);best_sha=sha(run/'best.pt')
            result=self.train(run,extend=20)
        self.assertEqual(result['epoch_limit'],20)
        self.assertEqual(result['epochs'],4)  # Three bad epochs after original best.
        state=read(run/'state.json');self.assertTrue(state['stopped'])
        self.assertEqual(state['bad_epochs'],3);self.assertEqual(state['best_loss'],1.)
        self.assertEqual(state['history'][0],first['history'][0]);self.assertEqual(sha(run/'best.pt'),best_sha)
        _,origin=demos.completed_imitation(SOURCE_ROOT,run)
        self.assertEqual(origin['sha256'],best_sha)
        with self.assertRaisesRegex(ValueError,'stopped by patience'):self.train(run,extend=21)

    def test_rejections_happen_before_loading_or_changing_checkpoints(self):
        run=self.directory/'guard';self.train(run)
        before={p.name:sha(p) for p in run.glob('*') if p.is_file()}
        with patch.object(demos,'load_dataset',side_effect=AssertionError('must reject before preparation')):
            for args in (dict(epochs=20),dict(extend=20,patience=2),dict(extend=20,lr=1e-4)):
                with self.assertRaisesRegex(ValueError,'settings/source changed'):self.train(run,**args)
            with self.assertRaisesRegex(ValueError,'existing IL run'):self.train(self.directory/'missing',extend=20)
        self.assertEqual(before,{p.name:sha(p) for p in run.glob('*') if p.is_file()})

    def test_reviewed_old_prepared_files_are_reused_but_still_check_raw_hashes(self):
        from brass_ai.recovery.replay import cache_version
        legacy=extension.equivalent_preparations(store.FORMAT,cache_version())
        self.assertTrue(legacy)
        directory=self.directory/'legacy-prepared';directory.mkdir()
        key=store.task_key(self.tasks[0],legacy[0])
        receipt,reused=store._prepare_job((self.tasks[0],str(directory),key))
        self.assertFalse(reused);write_json(store.paths(directory,key)['receipt'],receipt)
        before={p.name:sha(p) for p in directory.iterdir()}
        rows=store.prepare(self.tasks[:1],directory,1,'train')
        try:self.assertEqual(rows.report['cache_hits'],1)
        finally:rows.close()
        self.assertEqual(before,{p.name:sha(p) for p in directory.iterdir()})
        raw=Path(self.tasks[0][0]);raw.write_bytes(raw.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'source changed'):
            store.prepare(self.tasks[:1],directory,1,'train')


if __name__=='__main__':unittest.main()
