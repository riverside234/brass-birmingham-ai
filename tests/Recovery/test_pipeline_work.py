"""Active Step 5 routes use shared clocks; archived work configs remain readable."""
import copy
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.config import (TEACHER_WORK_BUDGET,LEGACY_OPPONENT_WORK_BUDGET,
    NN_WORK_BUDGET,TEACHER_TIME_BUDGET,TEACHER_PHASE_TIME_BUDGET_V3,LEGACY_TIME_BUDGET,NN_TIME_BUDGET,
    fixed_work_config,validate_config,nn_work_budget_config,latest_model_recipe)
from brass_ai.step5.evaluation import evaluation_config,validate_evaluation_config


class PipelineWorkTests(unittest.TestCase):
    def test_all_active_training_and_evaluation_presets(self):
        folder=SOURCE_ROOT/'configs/step5'
        for p in [*folder.glob('lab*.json'),folder/'smoke.json',folder/'recovery-smoke.json',folder/'joint7-latest.json']:
            with self.subTest(preset=p.name):
                cfg=read(p);validate_config(cfg)
                self.assertEqual(cfg['teacher_work_budget'],TEACHER_TIME_BUDGET if latest_model_recipe(cfg) else TEACHER_PHASE_TIME_BUDGET_V3)
                self.assertEqual(cfg['legacy_opponent_work_budget'],LEGACY_TIME_BUDGET)
                self.assertEqual(fixed_work_config(cfg),cfg)
        for name in ('evaluation.json','development-evaluation.json','broader-evaluation.json'):
            cfg=read(folder/name);validate_evaluation_config(cfg)
            self.assertEqual(cfg['nn_work_budget'],NN_TIME_BUDGET)
            self.assertEqual(cfg['legacy_opponent_work_budget'],LEGACY_TIME_BUDGET)

    def test_old_custom_config_upgrades_without_mutation_or_hyperparameter_changes(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        cfg.pop('teacher_work_budget');cfg.pop('legacy_opponent_work_budget');original=copy.deepcopy(cfg)
        new=fixed_work_config(cfg)
        self.assertEqual(cfg,original)
        for key in cfg:self.assertEqual(new[key],cfg[key])
        self.assertEqual(new['teacher_work_budget'],TEACHER_PHASE_TIME_BUDGET_V3)
        self.assertEqual(new['legacy_opponent_work_budget'],LEGACY_TIME_BUDGET)
        new['teacher_work_budget']=copy.deepcopy(TEACHER_WORK_BUDGET)
        new['legacy_opponent_work_budget']=copy.deepcopy(LEGACY_OPPONENT_WORK_BUDGET)
        for key in ('teacher_work_budget','legacy_opponent_work_budget'):
            new[key]['contract']=new[key]['contract'].replace('/1.1.0','/1.0.0')
            new[key].pop('watchdog_seconds')
        upgraded=fixed_work_config(new)
        self.assertEqual(upgraded,fixed_work_config(original))

    def test_evaluation_derived_from_training_uses_shared_clocks(self):
        train=read(SOURCE_ROOT/'configs/step5/lab.json')
        for searched in (False,True):
            cfg=evaluation_config(train,use_search=searched)
            self.assertEqual(cfg['use_search'],searched)
            self.assertEqual(cfg['nn_work_budget'],NN_TIME_BUDGET)
            self.assertEqual(cfg['legacy_opponent_work_budget'],LEGACY_TIME_BUDGET)
        old=read(SOURCE_ROOT/'configs/step5/evaluation.json')
        old.pop('nn_work_budget');old.pop('legacy_opponent_work_budget')
        cfg=evaluation_config(old,use_search=True)
        self.assertEqual(cfg['nn_work_budget'],NN_TIME_BUDGET)
        self.assertEqual(cfg['legacy_opponent_work_budget'],LEGACY_TIME_BUDGET)

    def test_watchdog_and_counts_are_bounded_separately(self):
        for field,value in [('contract',None),('simulations_per_turn',False),('simulations_per_turn',0),
                            ('watchdog_seconds',1),('watchdog_seconds',float('inf'))]:
            cfg=dict(NN_WORK_BUDGET);cfg[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):nn_work_budget_config(cfg)
        cfg=read(SOURCE_ROOT/'configs/step5/evaluation.json')
        cfg['nn_work_budget']=dict(NN_WORK_BUDGET)
        cfg['nn_work_budget']['simulations_per_turn']=1
        with self.assertRaisesRegex(ValueError,'cannot support workers'):validate_evaluation_config(cfg)
        cfg=read(SOURCE_ROOT/'configs/step5/evaluation.json')
        cfg['legacy_opponent_work_budget']=dict(LEGACY_OPPONENT_WORK_BUDGET)
        cfg['legacy_opponent_work_budget']['simulations_per_turn']=17
        with self.assertRaisesRegex(ValueError,'matched four-attempt'):validate_evaluation_config(cfg)

    def test_legacy_identity_allows_eval_without_admitting_holdout_to_training(self):
        from brass_ai.recovery.legacy_search import legacy_spec,teachers
        pool=read(SOURCE_ROOT/'training/opponents/pool.json')
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json');budget=cfg['legacy_budget']
        parent=sorted(pool['training'])[0];held=sorted(pool['evaluation'])[0]
        old=legacy_spec(SOURCE_ROOT,parent,budget)
        searched=legacy_spec(SOURCE_ROOT,parent,budget,work_budget=LEGACY_OPPONENT_WORK_BUDGET)
        fast=legacy_spec(SOURCE_ROOT,parent,budget,search_free=True,work_budget=LEGACY_OPPONENT_WORK_BUDGET)
        self.assertEqual(len({old['policy_id'],searched['policy_id'],fast['policy_id']}),3)
        self.assertIn(old['policy_id'],teachers(SOURCE_ROOT,budget))
        self.assertIn(searched['policy_id'],teachers(SOURCE_ROOT,budget,LEGACY_OPPONENT_WORK_BUDGET))
        self.assertEqual(legacy_spec(SOURCE_ROOT,held,budget,'evaluation',work_budget=LEGACY_OPPONENT_WORK_BUDGET)['purpose'],'evaluation')
        with self.assertRaises(ValueError):legacy_spec(SOURCE_ROOT,held,budget,'training',work_budget=LEGACY_OPPONENT_WORK_BUDGET)

    def test_gpu_selection_is_forwarded_as_runtime_placement_not_learning_config(self):
        import torch
        from brass_ai.step5 import cli
        original=read(SOURCE_ROOT/'configs/step5/joint7-latest.json')
        for count in (2,6):
            with self.subTest(devices=count),patch.object(torch.cuda,'is_available',return_value=True),\
                 patch.object(torch.cuda,'device_count',return_value=count),patch.object(cli,'require_preflight'),\
                 patch.object(cli,'train',return_value=dict(completed_iteration=1,summaries=[])) as train,\
                 contextlib.redirect_stdout(io.StringIO()):
                cli.main(['--root',str(SOURCE_ROOT),'--device','cuda','--inference-devices','auto',
                    'train','--run','artifacts/runtime-gpu-test','--resume-reviewed-source'])
                self.assertEqual(train.call_args.kwargs['inference_devices'],['cuda:'+str(i) for i in range(count)])
                self.assertEqual(train.call_args.args[2],original)
                self.assertTrue(train.call_args.kwargs['resume_reviewed_source'])
                self.assertFalse(train.call_args.kwargs['fresh'])


if __name__=='__main__':unittest.main()
