"""Utility widening: old identities, learned outputs, both Adams and real resume."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,UtilityConfig,utility_config
from brass_ai.learning.utility_width import widen,optimizer_state
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.search_value import ordinary_parameters,install
from brass_ai.recovery import utility_width_upgrade as U
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json


def recipe():
    from test_g1190_restart import committed,B
    state,config=committed()
    return B.seed_recipe(config,state,share_target=.05,revision=B.SEED_CHARGED)


def learned(size='small'):
    torch.manual_seed(1190)
    model=UtilityNet(utility_config(ModelConfig.preset(size),reward_spec(recipe())))
    with torch.no_grad():
        for p in model.utility_head.parameters():p.uniform_(-.05,.05)
    model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility))
    return model


def inputs(model):
    generator=torch.Generator().manual_seed(14)
    context=torch.randn(12,model.config.hidden,generator=generator)
    mapping=torch.tensor([[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]*3)
    heads=(torch.softmax(torch.randn(12,4,generator=generator),-1),torch.tensor([[1.,2.,3.,4.]]*12),
           torch.tensor([[0.,90.,180.,240.]]*12))
    return context,mapping,heads,torch.tensor([[.025,.01,.0025,0.]]*12),torch.randn(12,4,32,generator=generator)


class Head(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_legacy_config_and_identity_and_nonzero_prediction_preservation(self):
        model=learned();old_config=model.config.to_dict();identity=model_id(model)
        self.assertNotIn('utility_hidden',old_config)
        self.assertEqual(UtilityConfig.from_dict(old_config),model.config)
        args=inputs(model);expected=model.predict_utility(*args).detach();before=dict(model.named_parameters())
        rng=torch.get_rng_state();ready=model.utility_calibration
        widen(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIs(model.utility_calibration,ready)
        self.assertEqual(model.config.utility,old_config['utility'])
        torch.testing.assert_close(model.predict_utility(*args),expected,rtol=1e-6,atol=1e-6)
        for name,p in model.named_parameters():
            if not name.startswith('utility_head.'):self.assertIs(p,before[name])
        self.assertNotEqual(identity,model_id(model))
        self.assertEqual(model.config.to_dict()['utility_hidden'],256)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            path=Path(d)/'wide.pt';save_checkpoint(path,model,step=1529)
            loaded,raw=load_checkpoint(path)
            self.assertEqual((loaded.config.utility_hidden,raw['step']),(256,1529))
            self.assertEqual(model_id(model),model_id(loaded))
            torch.testing.assert_close(loaded.predict_utility(*args),expected,rtol=1e-6,atol=1e-6)

    def test_original_g1190_checkpoint_loads_with_its_original_identity(self):
        path=ROOT/'test_nn/joint7-v11-2-g1290-restart-g1290-ae347ae9-g1190-f228f4dd7f1a-1af78fc4bcc8/checkpoint.pt'
        model,raw=load_checkpoint(path)
        self.assertEqual(model.config.utility_hidden,32)
        self.assertEqual(model_id(model),raw['model_id'])
        self.assertEqual(model.config.to_dict(),raw['model_config'])

    def test_both_adams_keep_moments_and_added_neurons_learn(self):
        for optimizer_type in (torch.optim.AdamW,torch.optim.Adam):
            with self.subTest(optimizer=optimizer_type.__name__):
                model=learned();install(model,hidden=256)
                opt=optimizer_type(ordinary_parameters(model),lr=1e-3)
                args=inputs(model)
                model.predict_utility(*args).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
                old=opt.state_dict();shapes=widen(model)
                new=optimizer_state(old,model,shapes)
                self.assertIs(new['param_groups'],old['param_groups'])
                for i,(name,p) in zip(new['param_groups'][0]['params'],
                        ((n,p) for n,p in model.named_parameters() if not n.startswith('search_value_head.')),strict=True):
                    if i not in old['state']:continue
                    for key,value in old['state'][i].items():
                        actual=new['state'][i][key]
                        if key=='step' or name not in shapes or shapes[name]==tuple(p.shape):
                            self.assertIs(actual,value)
                        else:
                            torch.testing.assert_close(actual[tuple(slice(0,n) for n in shapes[name])],value,rtol=0,atol=0)
                            extra=actual[32:] if name!='utility_head.2.weight' else actual[:,32:]
                            self.assertEqual(torch.count_nonzero(extra).item(),0)
                restored=optimizer_type(ordinary_parameters(model),lr=1e-3);restored.load_state_dict(new)
                for _ in range(2):
                    restored.zero_grad(set_to_none=True)
                    (model.predict_utility(*args)-2).square().mean().backward()
                    restored.step()
                self.assertGreater(model.utility_head[2].weight[:,32:].abs().sum().item(),0)
                self.assertGreater(model.utility_head[0].weight.grad[32:].abs().sum().item(),0)

    def test_medium_model_stays_under_budget_and_exports_cpu_onnx(self):
        from brass_ai.rust_backend.lab_cpu_models import export_bundle
        model=learned('medium');install(model,hidden=256);widen(model)
        self.assertEqual(model.parameter_count,6194060)
        self.assertLess(model.parameter_count,model.config.parameter_limit)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            result=export_bundle(model,Path(d)/'bundle')
            self.assertTrue(result['parity']['passed'])
            self.assertEqual(result['outputs'][-1],'search_delta_4')
            self.assertLess(max(result['parity']['errors'].values()),2e-4)


class Migration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        from test_g1190_restart import Build,B
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        from brass_ai.recovery.search_value_iteration import CONTRACT,stamp
        from brass_ai.recovery.local_scoring_auxiliary import KEY
        from brass_ai.recovery.teacher_gradient_qualification import finish
        from brass_ai.rust_backend.lab_teacher_activation import configure as teacher_on,CONTRACT as ON
        builder=Build();builder.setUp();self.addCleanup(builder.doCleanups)
        builder.build(B.SEED_CHARGED,share_target=.05)
        run=builder.run;state=read(run/'state.json');config=state['signature']['config']
        model,raw=load_checkpoint(ROOT/state['history'][-1]['path'])
        config=teacher_on(config,.05,revise_stages=True)
        config['teacher_phase']['skip_utility_qualification']=True
        state['signature']=dict(state['signature'],config=config)
        entry=state['history'][-1]
        term=SimpleNamespace(config=config['teacher_phase'],qualification=None,mode='shadow',generation=1491,
            beta=1.,calibrated=False,steps=[dict(raw_share=.2,cosine=.5,actor_norm=1.,teacher_norm=.2)],
            qualification_context=dict(tables=dict(train=32,heldout=4),model_id=entry['model_id'],export_hash=entry['sha256'],
                labels_by_phase={name:1 for name in ('canal_early','canal_late','rail_early','rail_late')}))
        weight={};finish(term,weight);q=weight['gradient_qualification']
        state['native_backend']=dict(state['native_backend'],config_sha256=digest(config),
            teacher=dict(contract=ON,tables=dict(train=32,held_out=4),gradient_qualification=q['sha256'],
                beta=q['beta'],qualified_generation=q['qualified_generation'],generation=1492,
                stage_schedule=digest(config['teacher_phase']['search_value']['schedule']),ppg_beta_floor=.25))
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility))
        phase=SearchValuePhase(run/'search-values',model,config['teacher_phase']['search_value'],digest(model.config.utility))
        phase.machine.state.update(generation=1528,stage_a_iterations=37)
        saved=dict(phase.payload(),contract=CONTRACT,generation=1529,reward_id=digest(model.config.utility))
        teacher=dict(state['teacher_phase'],generation=1529,readiness=model.utility_calibration,weight=weight)
        state.update(iteration=1529,teacher_phase=teacher,search_value=stamp(saved))
        state['exploration']=dict(state['exploration'],generation=1529,
            entropy_schedule=dict(state['exploration']['entropy_schedule'],last_generation=1529))
        training=dict(raw['training'],signature=digest(state['signature']),native_backend=state['native_backend'],
            teacher_phase=teacher,search_value=saved,exploration=state['exploration'],
            actor_value_phases=B.ppg_state(config,model,1529,None),
            **{KEY:B.rollback_companion(raw['training'][KEY],1529,1491)})
        opt=torch.optim.AdamW(ordinary_parameters(model),lr=1e-4);opt.load_state_dict(raw['optimizer'])
        path=run/'generation-1529.pt';save_checkpoint(path,model,opt,step=1529,training=training)
        entry=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),model_id=model_id(model),generation=1529)
        state['history']=[*state['history'],entry];state['resume']=[entry]
        write_json(run/'state.json',state)
        for name in ('iteration-1529/teacher-shadow/partial.bin','generation-1530.pt.tmp',
                     'teacher-labels/generation-001529-train.jsonl.gz','search-values/001529-train-partial.json.gz',
                     '.ppg-eight-cache/generation-001530.critic.npy.partial'):
            p=run/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(b'interrupted work')
        # Only unavailable lab binaries are mocked. The resume, checkpoint,
        # optimizer, teacher planning, phase restore and beta checks are real.
        backend=state['native_backend']
        for name,replacement in (
            ('check_build',dict()),
            ('checked_pin',dict(passed=True,implementation=backend['implementation'])),
            ('pin',backend['rollback'])):
            p=mock.patch('brass_ai.rust_backend.lab_backend.'+name,return_value=replacement)
            p.start();self.addCleanup(p.stop)
        return run,state

    def test_committed_1529_migration_resume_beta_and_retry(self):
        from brass_ai.recovery.resume import load_resume
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from brass_ai.recovery.teacher_phase import TeacherPhase
        from brass_ai.recovery.teacher_gradient_qualification import finish
        from brass_ai.rust_backend.lab_teacher_activation import configure,authorize
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        run,before=self.fixture();old_path=ROOT/before['history'][-1]['path'];old_hash=sha(old_path)
        _,old=load_checkpoint(old_path)
        after=U.prepare(ROOT,run,expected_generation=1529);config=after['signature']['config']
        self.assertEqual(after['iteration'],1529);self.assertEqual(sha(old_path),old_hash)
        for key in ('quality_pool','ratings','search_value','exploration','teacher_seed_cursor','games_done'):
            self.assertEqual(before[key],after[key],key)
        self.assertEqual(config['teacher_phase']['search_value']['schedule']['stage_a_iterations'],100)
        self.assertEqual(config['teacher_phase']['term']['rows_per_step'],256)
        self.assertEqual(config['insolvency'],dict(event_cost=.0025,max_cost=.025))
        self.assertEqual(config['teacher_phase']['term']['share_target'],.05)
        model,raw,changed=load_resume(ROOT,run,after,after['signature'])
        self.assertFalse(changed);self.assertTrue(model.utility_calibration['ready'])
        self.assertTrue(torch.equal(raw['rng'],old['rng']))
        opt=torch.optim.AdamW(ordinary_parameters(model),lr=1e-4);resume_optimizer(raw,opt,restore_rng=False)
        TrainingPhases(model,config,raw=raw,generation=1529,root=ROOT,run=run)
        Companion(model,config['local_scoring_auxiliary'],raw=raw,generation=1529)
        teacher=TeacherPhase(ROOT,run,config,after,model,raw)
        self.assertTrue(teacher.ready);self.assertEqual(teacher.weight,{})
        machine=teacher.search_value.phase.machine
        self.assertEqual((machine.state['stage_a_iterations'],machine.stage_a),(37,100))
        jobs=teacher.begin(after,model,export_hash=after['history'][-1]['sha256'],epoch=1529)
        self.assertEqual(sum(j['split']=='train' for j in jobs),32)
        self.assertEqual(sum(j['split']=='heldout' for j in jobs),4)
        term=SimpleNamespace(config=config['teacher_phase'],qualification=None,mode='shadow',generation=1529,
            beta=1.,calibrated=False,steps=[dict(raw_share=.2,cosine=.5,actor_norm=1.,teacher_norm=.2)],
            qualification_context=dict(tables=dict(train=32,heldout=4),model_id=raw['model_id'],export_hash=after['history'][-1]['sha256'],
                labels_by_phase={name:1 for name in ('canal_early','canal_late','rail_early','rail_late')}))
        weight={};finish(term,weight);self.assertTrue(weight['gradient_qualification']['passed'])
        measured=dict(after,iteration=1530,teacher_phase=dict(after['teacher_phase'],weight=weight),
            exploration=dict(after['exploration'],generation=1530,
                entropy_schedule=dict(after['exploration']['entropy_schedule'],last_generation=1530)))
        on=configure(config,.05,revise_stages=True)
        self.assertEqual(authorize(measured,on)['beta'],.25)
        self.assertTrue(signature_transition(after['signature'],dict(after['signature'],config=on)))
        for g in range(1530,1592):machine.advance(g,None)
        self.assertEqual((machine.state['stage'],machine.state['stage_a_iterations']),('A',99))
        machine.advance(1592,None);self.assertEqual(machine.state['stage'],'B')
        checkpoint=after['history'][-1]
        again=U.prepare(ROOT,run,expected_generation=1529)
        self.assertEqual(again,after);self.assertEqual(sha(ROOT/checkpoint['path']),checkpoint['sha256'])
        archive=run/'configuration-history/utility256-1529'
        for name in read(archive/'pending.json')['paths']:
            self.assertFalse((run/name).exists());self.assertTrue((archive/'pending'/name).exists())

    def test_interrupted_migration_retries_without_losing_pending_work(self):
        run,before=self.fixture();real_write=U.write_json
        def fail_commit(path,value):
            if Path(path)==run/'state.json':raise KeyboardInterrupt()
            return real_write(path,value)
        with mock.patch.object(U,'write_json',side_effect=fail_commit),self.assertRaises(KeyboardInterrupt):
            U.prepare(ROOT,run,expected_generation=1529)
        self.assertEqual(read(run/'state.json'),before)
        after=U.prepare(ROOT,run,expected_generation=1529)
        self.assertEqual(after['signature']['config']['utility_hidden'],256)


class Command(unittest.TestCase):
    def test_prepared_wide_recipe_keeps_cpu_runtime(self):
        from brass_ai.recovery.collector_runtime import configured,from_config,cpu_block,recipe as signed_recipe
        spec=importlib.util.spec_from_file_location('shadow_wide',ROOT/'scripts/continue-rust-shadow.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=U.configure(recipe());runtime=dict(from_config(config),native_inference=cpu_block())
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            root=Path(d);run=root/'artifacts/step5/joint7-v11-2-g1190-restart-3';run.mkdir(parents=True)
            checkpoint=run/'generation-1529-utility256.pt';checkpoint.write_bytes(b'checkpoint already verified by migration')
            state=dict(iteration=1529,signature=dict(config=config),history=[dict(generation=1529,
                path=checkpoint.relative_to(root).as_posix(),sha256=sha(checkpoint))])
            state['signature']['config']=configured(config,runtime)
            write_json(run/'state.json',state)
            # This checkout's long Windows root exceeds MAX_PATH with the lab's
            # descriptive filename. Exercise recipe publication at a short test
            # path and separately check Linux's filename-component limit.
            def publish(path,value):write_json(root/'prepared.json',value)
            with mock.patch('brass_ai.rust_backend.lab_backend.validate_state'),\
                 mock.patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value=dict(core_source='test')),\
                 mock.patch('brass_ai.rust_backend.lab_cpu_runtime.resolve'),\
                 mock.patch('brass_ai.step5.league.write_json',side_effect=publish) as publication:
                _,path,remaining=module.prepare(root,run,1529,133,.05,mode='shadow',revision='g1491',skip_utility_qualification=True)
            publication.assert_called_once();self.assertEqual(publication.call_args.args[0],path)
            actual=read(root/'prepared.json')
            self.assertEqual(remaining,133)
            self.assertEqual(signed_recipe(actual),config)
            self.assertEqual(from_config(actual),runtime)
            self.assertLessEqual(len(path.name.encode())+4,255)  # atomic .tmp file too

    def test_upgrade_then_one_beta_iteration_then_continue(self):
        spec=importlib.util.spec_from_file_location('continue_wide',ROOT/'scripts/continue-rust-g1491.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        shared=module.shared();before=dict(iteration=1529,signature=dict(config=recipe()))
        upgraded=dict(before,signature=dict(config=U.configure(before['signature']['config'])),teacher_phase=dict(weight={}))
        def continuation(args,**kw):
            self.assertTrue(args.skip_utility_qualification)
            return dict(upgraded,iteration=1530,teacher_phase=dict(weight=dict(gradient_qualification=dict(passed=True))))
        argv=['continue','--run','artifacts/step5/joint7-v11-2-g1190-restart-3','--expected-generation','1529',
              '--iterations','133','--share-target','.05','--skip-utility-qualification','--upgrade-utility-head']
        with mock.patch('sys.argv',argv),mock.patch.object(module,'shared',return_value=shared),\
             mock.patch.object(module,'state_of',return_value=before),mock.patch.object(module,'ensure_cpu'),\
             mock.patch.object(U,'prepare',return_value=upgraded) as prepare,\
             mock.patch.object(shared,'continuation',side_effect=continuation) as call:
            module.main(share_target_required=True)
        self.assertEqual(prepare.call_args.kwargs,dict(expected_generation=1529))
        self.assertEqual([c.kwargs for c in call.call_args_list],[
            dict(mode='shadow',expected=1529,iterations=1,revision='g1491'),
            dict(mode='on',expected=1530,iterations=132,revise_stages=True)])


if __name__=='__main__':unittest.main()
