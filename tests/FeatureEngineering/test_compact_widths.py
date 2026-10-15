"""Width-only v9 admission and teacher-only worker allocation; no game searches."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import (SOURCE_ROOT,CONFIG_DIR,compatibility,digest,
    read_json,assert_compatible,assert_heuristic_data_compatible)
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.state_layout import state_layout
from brass_ai.recovery.config import (TEACHER_TIME_BUDGET,latest_model_config,
    execution_config,search_workers_per_turn,validate_config)
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.execution import ActorPool,make_policy
from brass_ai.recovery.replay import write_shard,read_shards,cache_receipt,load_cache
from brass_ai.recovery.teacher_schedule import estimate_collection,CONTRACT
from brass_ai.step5.evaluation import evaluation_config
from fixtures import game,rotate
from fe_fixtures import Factory


class CompactWidthTests(unittest.TestCase):
    def test_feature_contract_and_all_input_indices_stay_identical(self):
        schema=read_json(CONFIG_DIR/'features-4.0.0.json')
        self.assertEqual(digest(schema),'59facdd72a386b6046abbb875daf3e1051b92e580ff55d0ab37250dd0c1dfb6a')
        old=read_json(CONFIG_DIR/'state-groups-4.0.0.json');new=state_layout().specification()
        self.assertEqual(new['segments'],old['segments'])
        for before,after in zip(old['groups'],new['groups']):
            self.assertEqual({k:v for k,v in before.items() if k!='projection_width'},
                             {k:v for k,v in after.items() if k!='projection_width'})
        self.assertEqual(new['feature_schema_sha256'],old['feature_schema_sha256'])
        self.assertEqual((new['state_dim'],new['raw_state_dim'],new['fusion_inputs']),(3128,2539,620))
        self.assertNotEqual(new['layout_sha256'],old['layout_sha256'])
        # Identical inputs do not make v8 checkpoints valid for wider weights.
        bad=dict(ModelConfig.preset().to_dict(),architecture='brass-four-seat-mlp-resnet/8.0.0',
                 group_layout=old['layout_sha256'],projection_widths=[g['projection_width'] for g in old['groups'][:-1]])
        with self.assertRaisesRegex(ValueError,'fresh version-9'):ModelConfig.from_dict(bad)

    def test_existing_heuristic_data_admitted_without_relabeling_neural_replay(self):
        old=dict(compatibility(),architecture_sha256='a'*64)
        assert_heuristic_data_compatible(old)
        with self.assertRaisesRegex(ValueError,'Incompatible'):assert_compatible(old)
        for field in ('features_sha256','encoder_sha256','derived_source_sha256','rules'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                assert_heuristic_data_compatible(dict(old,**{field:'b'*64}))
        episode=game()
        fixture=Factory()
        try:view,choices=fixture.position()
        finally:fixture.close()
        for row in episode['records']:
            row['behavior']['kind']='external'
            row['view']=rotate(view,(row['seat']-view['seat'])%4)
            row['choices']=copy.deepcopy(choices)
            row['view']['context']['revision']=row['step']
            row['chosen']=0
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'audited.jsonl.gz'
            with patch('brass_ai.recovery.replay.compatibility',return_value=old):
                originals=write_shard(path,[episode],dict(purpose='training'))
                receipt=cache_receipt(path)
            files=list(Path(temp).iterdir());before={p.name:sha(p) for p in files}
            self.assertIsNone(load_cache(path,[episode],receipt))
            loaded=read_shards([path],receipts={str(path):receipt},heuristic_only=True)
            self.assertEqual(loaded[0]['records'],episode['records'])
            encoder=PreparedEncoder()
            for a,b in zip(originals[0]['records'],loaded[0]['records']):
                first=originals[0].encoded(a,encoder);second=loaded[0].encoded(b,encoder)
                np.testing.assert_array_equal(first.state,second.state)
                np.testing.assert_array_equal(first.actions,second.actions)
            self.assertEqual(before,{p.name:sha(p) for p in files})
            with self.assertRaisesRegex(ValueError,'Incompatible'):read_shards([path])
            for row in episode['records']:row['behavior']['kind']='model'
            with patch('brass_ai.recovery.replay.compatibility',return_value=old):
                write_shard(Path(temp)/'neural.jsonl.gz',[episode],dict(purpose='training'))
            with self.assertRaisesRegex(ValueError,'prohibits neural'):
                read_shards([Path(temp)/'neural.jsonl.gz'],heuristic_only=True)


class TeacherWidthTests(unittest.TestCase):
    def config(self):return read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json')

    def test_current_recipe_generator_clocks_and_cpu_validation(self):
        config=self.config();validate_config(config)
        self.assertEqual(config['teacher_work_budget'],TEACHER_TIME_BUDGET)
        self.assertEqual(TEACHER_TIME_BUDGET['seconds_per_phase'],
                         dict(canal_early=9.,canal_late=12.,rail_early=12.,rail_late=12.))
        self.assertEqual(latest_model_config(config,config['imitation_checkpoint']),config)
        self.assertEqual(search_workers_per_turn(config['execution'],teacher=True),8)
        self.assertEqual(search_workers_per_turn(config['execution']),4)
        self.assertEqual(config['legacy_budget']['search_workers'],4)
        self.assertEqual(estimate_collection(config,0.),900.)
        self.assertIsNone(config['search_wall_fraction'])
        self.assertEqual((config['search_games'],config['search_every_games'],config['distill_steps']),(5,160,8))
        evaluation=evaluation_config(config)
        self.assertEqual(search_workers_per_turn(evaluation['execution']),4)
        self.assertEqual(evaluation['nn_work_budget']['seconds_per_turn'],12.)
        for bad in (0,True,17,2.5):
            with self.subTest(workers=bad),self.assertRaises(ValueError):
                execution_config(dict(config['execution'],teacher_search_workers_per_turn=bad))
        with self.assertRaisesRegex(ValueError,'CPU admission'):
            execution_config(dict(config['execution'],cpu_budget=11,cpu_reserve=4))
        archived=copy.deepcopy(config['execution']);archived.pop('teacher_search_workers_per_turn')
        self.assertEqual(search_workers_per_turn(archived,teacher=True),4)

    def test_five_teacher_actors_reserve_forty_search_channels_plus_five_coordinators(self):
        config=self.config();pool=ActorPool.__new__(ActorPool)
        pool.width=config['execution']['heavy_processes'];pool.lanes=1;pool.heavy=True
        pool.config=config['execution'];pool.processes=[];pool.root=SOURCE_ROOT
        pool.host=config['host'];pool.mono=None;pool.jobs=object();pool.results=object()
        pool.service=Mock();pool.mp=Mock()
        pool.mp.Process.side_effect=lambda **kw:Mock()
        with patch('brass_ai.recovery.host_runtime.training_host'):
            pool.start(config['search_games'])
        self.assertEqual(len(pool.processes),5)
        pool.service.channels.assert_called_once_with(45)
        for call in pool.mp.Process.call_args_list:self.assertEqual(call.kwargs['args'][8],9)

    def test_teacher_and_evaluation_factories_use_different_forests_on_shared_channels(self):
        config=self.config();endpoints=list(range(9));forests=[]
        class Policy:
            def __init__(self,*args,**kwargs):
                self.config=kwargs['config'];self.game=SimpleNamespace(host=object())
                self.simulation_games=[]
        def forest(policy,workers,role,factory):
            policy.replicas=[factory(i) for i in range(workers)]
            policy.config=replace(policy.config,workers=workers);forests.append(workers)
            return policy
        with patch('brass_ai.recovery.execution.MaskedSearchPolicy',Policy), \
             patch('brass_ai.recovery.execution.NeuralPolicy',return_value=object()), \
             patch('brass_ai.recovery.execution.RemoteEvaluator',return_value=object()), \
             patch('brass_ai.recovery.execution.instrument_host'), \
             patch('brass_ai.recovery.parallel.parallelize',side_effect=forest), \
             patch('brass_ai.recovery.teacher_schedule.training_teacher',side_effect=lambda p,**kw:p), \
             patch('brass_ai.recovery.teacher_schedule.work_search',side_effect=lambda p,*args:p):
            spec=dict(kind='masked_ismcts',model_id='a'*64,search=config['search'])
            teacher=make_policy(dict(spec,teacher_schedule=CONTRACT,teacher_work_budget=TEACHER_TIME_BUDGET),
                                SOURCE_ROOT,endpoints,config['execution'],None)
            evaluation=make_policy(spec,SOURCE_ROOT,endpoints,config['execution'],None)
        self.assertEqual(forests,[8,4])
        self.assertEqual((len(teacher.replicas),teacher.config.seconds),(8,12.))
        self.assertEqual((len(evaluation.replicas),evaluation.config.seconds),(4,12.))

    def test_worker_capacity_is_reported_in_compact_history(self):
        from test_selfplay_70_review import TeacherCollectionTests
        from brass_ai.recovery.history import compact
        fixture=TeacherCollectionTests();original=fixture.fixture
        def configured():
            cfg,*rest=original();cfg.update(teacher_work_budget=copy.deepcopy(TEACHER_TIME_BUDGET))
            cfg['execution']['teacher_search_workers_per_turn']=8
            return cfg,*rest
        fixture.fixture=configured
        (report,*_),jobs,calls=fixture.run_fixture()
        budget=report['budget']
        self.assertEqual((budget['search_workers_per_game'],budget['parallel_game_capacity'],budget['total_search_worker_capacity']),(8,5,40))
        self.assertEqual(compact(dict(search=report))['search']['budget'],budget)
        self.assertEqual(len(jobs),10);self.assertEqual([c.kwargs['steps'] for c in calls],[8,8])


if __name__=='__main__':unittest.main()
