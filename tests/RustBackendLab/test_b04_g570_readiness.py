"""Pinned user-authorized null readiness, immutable checkpoints and receipt reuse."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.rust_backend import lab_b04_smoke as B, lab_prerequisite_reuse as R
from brass_ai.rust_backend.lab_contract import pin, SOURCE
from brass_ai.recovery.teacher_phase import TeacherPhase
from test_b025 import config, model_fixture


def load(name):
    with patch.object(sys, 'path', [str(SOURCE_ROOT/'scripts'), *sys.path]):
        spec=importlib.util.spec_from_file_location(name.replace('-', '_'), SOURCE_ROOT/'scripts'/name)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class NullReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def fixture(self):
        c=config();reward=c['teacher_phase']['search']['reward']
        checkpoint=dict(path=B.G570_RUN+'/generation-0570.pt',sha256=B.G570_CHECKPOINT,
                        model_id=B.G570_MODEL,generation=570)
        state=dict(iteration=570,history=[dict(path='retired.pt',generation=569),checkpoint])
        raw=dict(step=570,model_id=B.G570_MODEL,model_config={'utility':reward},
                 training=dict(utility_calibration=None,fresh_run=B.G570_RUN))
        return state,raw,checkpoint,digest(reward)

    def test_exact_g570_null_is_ready_without_reading_retired_history_or_mutation(self):
        state,raw,checkpoint,reward=self.fixture();before=copy.deepcopy((state,raw))
        with patch.object(B.torch,'load',side_effect=AssertionError('No historical scan')):
            receipt,ready,evidence=B.latest_readiness(SOURCE_ROOT,state,raw,reward,checkpoint=checkpoint)
        self.assertIsNone(receipt);self.assertTrue(ready)
        self.assertEqual(evidence['kind'],'user-authorized-g570-null-utility')
        self.assertFalse(evidence['calibration_measured']);self.assertEqual(evidence['checkpoint_sha256'],B.G570_CHECKPOINT)
        self.assertEqual((state,raw),before)

    def test_null_on_other_or_unpinned_checkpoints_still_defers(self):
        state,raw,checkpoint,reward=self.fixture()
        for selected in (None,dict(checkpoint,sha256='a'*64)):
            self.assertFalse(B.latest_readiness(SOURCE_ROOT,state,raw,reward,checkpoint=selected)[1])

    def test_identity_and_reward_changes_cannot_use_assumption(self):
        for change in ('generation','model','run','reward','missing_head','history_hash','history_path'):
            state,raw,checkpoint,reward=self.fixture()
            if change=='generation':raw['step']=571
            elif change=='model':raw['model_id']='foreign'
            elif change=='run':raw['training']['fresh_run']='another-run'
            elif change=='reward':reward='foreign'
            elif change=='missing_head':raw['model_config']['utility']=None
            elif change=='history_hash':state['history'][-1]['sha256']='a'*64
            else:state['history'][-1]['path']='another/checkpoint.pt'
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'assumption'):
                B.latest_readiness(SOURCE_ROOT,state,raw,reward,checkpoint={'sha256':B.G570_CHECKPOINT})

    def test_explicit_false_and_reward_mismatch_are_not_overridden(self):
        for where in ('current','saved'):
            for value in (dict(ready=False),dict(ready=True,reward_id='other')):
                state,raw,checkpoint,reward=self.fixture()
                receipt={'reward_id':reward,**value}
                if where=='current':raw['training']['utility_calibration']=receipt
                else:state['teacher_phase']={'readiness':receipt}
                actual,ready,_=B.latest_readiness(SOURCE_ROOT,state,raw,reward,checkpoint=checkpoint)
                self.assertEqual(actual,receipt);self.assertFalse(ready)

    def test_shadow_lifecycle_retains_assumption_without_changing_model_calibration(self):
        state,raw,checkpoint,reward=self.fixture();receipt,ready,evidence=B.latest_readiness(SOURCE_ROOT,state,raw,reward,checkpoint=checkpoint)
        model=model_fixture();model.utility_calibration=None
        c=config();c['teacher_phase']['mode']='shadow';before=copy.deepcopy(state)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            phase=TeacherPhase(SOURCE_ROOT,Path(temp),c,state,model)
            phase.receipt=dict(ready=True,reward_id=reward,basis=evidence);phase.ready=ready
            jobs=phase.begin(state,model,export_hash=B.G570_CHECKPOINT,epoch=0,native_source=B.source_fingerprint())
            self.assertEqual(len(jobs),32)
            self.assertTrue(all(j['readiness']['basis']==evidence for j in jobs))
            self.assertIsNone(model.utility_calibration);self.assertEqual(state,before)
            self.assertEqual(phase.receipt['basis']['scope'],'B04 shadow qualification only')


class Reuse(unittest.TestCase):
    def test_b04_entry_reuses_b03_but_still_checks_its_evidence_and_state(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);snapshot=dict(state_file={'sha256':'state'},checkpoint={'sha256':'model'})
            lab=dict(passed=True,new_games=0,optimizer_steps=0)
            for name in ('preflight','restoration','replay','diagnostic_fork'):
                path=root/(name+'.json')
                path.write_text(json.dumps({'snapshot':snapshot} if name=='preflight' else {}),encoding='utf-8')
                lab[name]=pin(path)
            report=dict(contract='brass-lab-b03/1.0.0',passed=True,method='search_v1',v205_enabled=False,
                lab_qualification='Passed for this exact committed checkpoint',transfer={'implementation':'old'},lab=lab)
            path=root/'b03.json';path.write_text(json.dumps(report),encoding='utf-8')
            proof=dict(b03_implementation='old',to_implementation='new',receipts={'b03':pin(path)})
            with patch.object(R,'reviewed',return_value=proof),patch.object(B,'b02_entry',return_value=(snapshot,None,None,None)):
                self.assertEqual(B.entry('b02',path,root,'cuda:0'),snapshot)
                (root/'replay.json').write_text('{"changed":true}',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Changed B03 replay'):B.entry('b02',path,root,'cuda:0')
                (root/'replay.json').write_text('{}',encoding='utf-8')
                with patch.object(B,'b02_entry',return_value=(dict(snapshot,state_file={'sha256':'advanced'}),None,None,None)):
                    with self.assertRaisesRegex(ValueError,'another committed state'):B.entry('b02',path,root,'cuda:0')

    def test_b03_reuse_requires_exact_original_receipt_and_implementation(self):
        receipt=dict(path='/original',bytes=1,sha256='a'*64)
        proof=dict(b03_implementation='old',to_implementation='new',receipts={'b03':receipt})
        with patch.object(R,'reviewed',return_value=proof),patch.object(R,'pin',return_value=receipt):
            self.assertEqual(R.verify_b03('/original','old')['original_b03'],receipt)
            with self.assertRaises(ValueError):R.verify_b03('/original','foreign')
        with patch.object(R,'reviewed',return_value=proof),patch.object(R,'pin',return_value=dict(receipt,sha256='b'*64)):
            with self.assertRaises(ValueError):R.verify_b03('/original','old')

    def test_reuse_native_build_preserves_binary_and_rejects_tampering(self):
        tool=load('qualify-rust-lab-b04.py')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp)
            for name in ('binary','bootstrap','dependency','lock','b02','b03'):(root/name).write_bytes(name.encode())
            (root/'overlay').write_text(json.dumps(dict(core_source='core')),encoding='utf-8')
            build=dict(contract='brass-lab-b04-build/1.0.0',base_source=SOURCE,producer=B.source_fingerprint(),
                implementation='old',core_source='core',binary=pin(root/'binary'),bootstrap=dict(binary=pin(root/'bootstrap'),
                dependencies=[pin(root/'dependency')]),overlay=pin(root/'overlay'),lock=pin(root/'lock'),
                b02=pin(root/'b02'),b03=pin(root/'b03'))
            path=root/'build.json';path.write_text(json.dumps(build),encoding='utf-8')
            proof=dict(b04_build=pin(path),b03_implementation='old')
            with patch.object(R,'reviewed',return_value=proof):
                reused=tool.reuse_build(path,root/'b02',root/'b03')
                self.assertEqual(reused['binary'],build['binary']);self.assertEqual(reused['reused_from'],pin(path))
                self.assertEqual(json.loads(path.read_bytes()),build)
                for name in ('binary','bootstrap','dependency','lock','overlay','b02','b03'):
                    original=(root/name).read_bytes();(root/name).write_bytes(b'tampered')
                    try:
                        with self.subTest(name=name),self.assertRaises(ValueError):tool.reuse_build(path,root/'b02',root/'b03')
                    finally:(root/name).write_bytes(original)
                with self.assertRaises(ValueError):tool.reuse_build(root/'binary',root/'b02',root/'b03')

    def test_runner_restarts_at_b04_with_both_original_receipts_and_build(self):
        runner=load('run-rust-lab-qualification.py')
        args=SimpleNamespace(stage='all',start_at='b04',device='cuda:0',fixtures_only=False,run=Path('run'),
            expected_generation=570,checkpoint_sha256=B.G570_CHECKPOINT,b02_workers=60,
            allow_cuda_topology_change=False,b04_build=Path('old-build.json'))
        receipts={'b02':Path('old-b02.json'),'b03':Path('old-b03.json')}
        self.assertEqual(runner.selected_stages(args,receipts),('b04',))
        cmd=runner.command('b04',args,Path('out'),receipts)
        self.assertEqual(cmd[cmd.index('--reuse-build')+1],'old-build.json')
        self.assertEqual(cmd[cmd.index('--b03')+1],'old-b03.json')
        self.assertNotIn('--allow-cuda-topology-change',cmd)


if __name__=='__main__':unittest.main()
