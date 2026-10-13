"""Policy-only teacher updates cannot reuse a pre-update utility fit receipt."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fixtures import game
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.teacher_collection import collect_due
from brass_ai.recovery.utility import calibration_ready,calibrate
from brass_ai.recovery.updates import Dataset,prepared,distill_update
from brass_ai.search.ismcts import visit_teacher
import test_combined_reward as reward_fixtures
import test_selfplay_70_review as collection_fixtures


class TeacherReadinessReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        model,opt=reward_fixtures.ModelTests().model()
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility),scope='authored test receipt')
        identity=model_id(model);episode=game();episode['split']='train'
        for row,(era,round_) in zip(episode['records'],[('CanalEra',1),('CanalEra',5),('RailEra',1),('RailEra',7)]):
            row['view']['context'].update(era=era,round=round_)
            stats=dict(root={c['id']:dict(visits=n) for c,n in zip(row['choices'],[9,1,1])},expansion=[True]*3)
            row['teacher']=visit_teacher(row['choices'],stats,[.8,.1,.1],model=identity,observer=row['seat'],config=reward_fixtures.config())
        cfg=dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2)
        quality=dict(min_samples=4,min_games=1,min_samples_per_phase=1,batch_size=4,max_reuse=1)
        return model,opt,episode,cfg,quality

    def test_actual_distillation_invalidates_fit_but_preserves_heads_and_checkpoint_layout(self):
        model,opt,episode,cfg,quality=self.fixture()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp)
            save_checkpoint(directory/'before.pt',model,opt,step=70)
            report=distill_update(model,opt,[episode],cfg,steps=1,quality=quality)
            self.assertTrue(report['updated']);self.assertTrue(report['utility_calibration_invalidated'])
            self.assertFalse(calibration_ready(model))
            self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('state_encoder.')))
            for k,v in before.items():
                if k.startswith(('outcomes.','utility_head.')):torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
            save_checkpoint(directory/'after.pt',model,opt,step=71)
            old,old_raw=load_checkpoint(directory/'before.pt');new,new_raw=load_checkpoint(directory/'after.pt')
            self.assertEqual(old_raw['format'],new_raw['format'])
            self.assertEqual(old_raw['model_config'],new_raw['model_config'])
            self.assertEqual(set(old_raw['state_dict']),set(new_raw['state_dict']))
            self.assertEqual(old_raw['optimizer']['param_groups'],new_raw['optimizer']['param_groups'])
            self.assertTrue(calibration_ready(old));self.assertFalse(calibration_ready(new))

    def test_insufficient_collection_preserves_readiness_weights_and_optimizer(self):
        model,opt,episode,cfg,quality=self.fixture();episode['records'][0]['teacher']=None
        old=copy.deepcopy(model.utility_calibration);identity=model_id(model);steps=len(opt.state)
        report=distill_update(model,opt,[episode],cfg,steps=1,quality=quality)
        self.assertFalse(report['updated']);self.assertEqual(model.utility_calibration,old)
        self.assertEqual(model_id(model),identity);self.assertEqual(len(opt.state),steps)

    def test_real_ordinary_refit_after_distillation_changes_only_the_utility_head(self):
        model,opt,episode,cfg,quality=self.fixture()
        distill_update(model,opt,[episode],cfg,steps=1,quality=quality)
        self.assertFalse(calibration_ready(model))
        ordinary=game();ordinary['game_id']='separate-fresh-ordinary-game'
        self.assertTrue(all(row['teacher'] is None for row in ordinary['records']))
        data=prepared([ordinary],insolvency=model.config.utility['insolvency'],reward=model.config.utility)
        dataset=Dataset(data,'cpu',policy=False)
        before={k:v.clone() for k,v in model.state_dict().items()};rates=[g['lr'] for g in opt.param_groups]
        indices,inputs,_=dataset.batch(list(range(len(data))));model.eval()
        with torch.inference_mode():outputs=[v.clone() for v in model(*inputs,remaining=dataset.remaining[indices])[:4]]
        receipt=calibrate(model,opt,dataset,cfg)
        self.assertTrue(receipt['required']);self.assertTrue(calibration_ready(model))
        self.assertEqual(receipt['receipt']['games'],1);self.assertEqual(receipt['receipt']['steps'],3)
        self.assertEqual([g['lr'] for g in opt.param_groups],rates)
        for k,v in before.items():
            if not k.startswith('utility_head.'):torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('utility_head.')))
        with torch.inference_mode():after=model(*inputs,remaining=dataset.remaining[indices])
        for old,new in zip(outputs,after[:4]):torch.testing.assert_close(old,new,rtol=0,atol=0)

    def test_second_due_collection_defers_before_seeds_time_or_schedule_are_spent(self):
        cfg,state,service,heavy,teaching,update,jobs=collection_fixtures.TeacherCollectionTests().fixture()
        model=SimpleNamespace(config=SimpleNamespace(utility=cfg['search']['reward']),
            utility_calibration=dict(ready=True,reward_id=digest(cfg['search']['reward'])))
        def trained(*args,**kwargs):
            model.utility_calibration=dict(model.utility_calibration,ready=False)
            return copy.deepcopy(update)
        with patch('brass_ai.recovery.teacher_collection.rollout_opponents',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.read_shards',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.collection_report',return_value=teaching), \
             patch('brass_ai.recovery.teacher_collection.distill_update',side_effect=trained) as train, \
             patch('brass_ai.recovery.teacher_collection.time.monotonic',side_effect=[0.,200.]):
            report,next_due,cursor,spent=collect_due(SOURCE_ROOT,SOURCE_ROOT/'artifacts/mock',SOURCE_ROOT/'artifacts/mock/iteration-0070',
                cfg,state,{},model,None,service,heavy,done=960,cursor=18000000,ordinary_seconds=10000.,ordinary_game_seconds=1.)
        from brass_ai.step5.league import seeds_for
        expected=18000000
        for _ in range(cfg['search_games']):_,expected=seeds_for(expected,1)
        self.assertEqual((len(jobs),train.call_count,report['collection_count']),(5,1,1))
        self.assertEqual((next_due,cursor,spent),(960,expected,210.))
        self.assertEqual((report['budget']['due_collections'],report['budget']['max_collections']),(2,2))
        self.assertTrue(report['budget']['deferred']);self.assertIn('Utility calibration invalidated',report['reason'])
        self.assertEqual(state['next_search_games'],800)

    def test_bounded_ordinary_calibration_keeps_two_collections_and_charges_its_time(self):
        cfg,state,service,heavy,teaching,update,jobs=collection_fixtures.TeacherCollectionTests().fixture()
        model=SimpleNamespace(config=SimpleNamespace(utility=cfg['search']['reward']),
            utility_calibration=dict(ready=True,reward_id=digest(cfg['search']['reward'])))
        clock=[0.];old_collect=heavy.collect
        def collect(jobs):clock[0]+=40.;return old_collect(jobs)
        heavy.collect=collect
        def trained(*args,**kwargs):
            clock[0]+=10.;model.utility_calibration=dict(model.utility_calibration,ready=False)
            return copy.deepcopy(update)
        def calibrate_ordinary():
            self.assertFalse(calibration_ready(model));clock[0]+=25.
            model.utility_calibration=dict(model.utility_calibration,ready=True)
            return dict(required=True,receipt=dict(model.utility_calibration,scope='fresh ordinary fixture'))
        with patch('brass_ai.recovery.teacher_collection.rollout_opponents',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.read_shards',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.collection_report',return_value=teaching), \
             patch('brass_ai.recovery.teacher_collection.distill_update',side_effect=trained), \
             patch('brass_ai.recovery.teacher_collection.time.monotonic',side_effect=lambda:clock[0]), \
             patch.object(SimpleNamespace(run=calibrate_ordinary),'run',side_effect=calibrate_ordinary) as callback:
            report,next_due,_,spent=collect_due(SOURCE_ROOT,SOURCE_ROOT/'artifacts/mock',SOURCE_ROOT/'artifacts/mock/iteration-0070',
                cfg,state,{},model,None,service,heavy,done=960,cursor=18000000,ordinary_seconds=10000.,ordinary_game_seconds=1.,
                recalibrate=callback)
        self.assertEqual((len(jobs),callback.call_count,report['collection_count'],next_due),(10,2,2,1120))
        self.assertEqual(spent,160.);self.assertFalse(report['budget']['deferred']);self.assertTrue(calibration_ready(model))
        self.assertTrue(all(b['post_distill_calibration']['receipt']['ready'] for b in report['collections']))
        self.assertEqual([b['post_distill_calibration_seconds'] for b in report['collections']],[25.,25.])

    def test_failed_calibration_fit_remains_unready_and_defers_remaining_collection(self):
        cfg,state,service,heavy,teaching,update,jobs=collection_fixtures.TeacherCollectionTests().fixture()
        model=SimpleNamespace(config=SimpleNamespace(utility=cfg['search']['reward']),
            utility_calibration=dict(ready=True,reward_id=digest(cfg['search']['reward'])))
        def trained(*args,**kwargs):
            model.utility_calibration=dict(model.utility_calibration,ready=False)
            return copy.deepcopy(update)
        def failed_fit():
            self.assertFalse(calibration_ready(model))
            return dict(required=True,receipt=dict(model.utility_calibration,scope='authored failed ordinary fit'))
        with patch('brass_ai.recovery.teacher_collection.rollout_opponents',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.read_shards',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.collection_report',return_value=teaching), \
             patch('brass_ai.recovery.teacher_collection.distill_update',side_effect=trained) as train, \
             patch('brass_ai.recovery.teacher_collection.time.monotonic',side_effect=[0.,175.,200.,200.]), \
             patch.object(SimpleNamespace(run=failed_fit),'run',side_effect=failed_fit) as callback:
            report,next_due,_,spent=collect_due(SOURCE_ROOT,SOURCE_ROOT/'artifacts/mock',SOURCE_ROOT/'artifacts/mock/iteration-0070',
                cfg,state,{},model,None,service,heavy,done=960,cursor=18000000,ordinary_seconds=10000.,ordinary_game_seconds=1.,
                recalibrate=callback)
        self.assertEqual((len(jobs),train.call_count,callback.call_count,next_due,spent),(5,1,1,960,210.))
        self.assertFalse(calibration_ready(model));self.assertTrue(report['budget']['deferred'])
        self.assertFalse(report['collections'][0]['post_distill_calibration']['receipt']['ready'])
        self.assertEqual(report['collections'][0]['post_distill_calibration_seconds'],25.)


if __name__=='__main__':unittest.main()
