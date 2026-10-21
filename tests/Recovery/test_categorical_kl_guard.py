"""Exact PPO stopping, sampled fallback, and checkpoint/source continuity."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_ppo_lr_controller as lr_fixtures
from test_mlp_v11 import inputs as model_inputs
from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.features import Encoded
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import updates,lr_controller
from brass_ai.recovery.history import compact
from brass_ai.recovery.learning_rate import KEY as WARMUP_KEY
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition,apply_optimizer_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.review_metrics import CategoricalProbe,PPO_KL_GUARD
from brass_ai.recovery.source_reviews import expand_source_reviews
from brass_ai.recovery.telemetry import Timings
from brass_ai.step5.league import implementation_id,write_json


PREVIOUS='0c96c4a3ae6ce4d93ac1c1861fe9603b83143c3a844c094919579585cd6548ac'
GUARD_IMPLEMENTATION='33b925447d6e6dc0e8ec0b2a928f5ffba50738ea93abafab83101f3a1b491439'


class Logits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(architecture=None)
        self.logits=torch.nn.Parameter(torch.log(torch.tensor([.2,.4,.4])))
        self.calls=0

    def forward(self,states,actions,legal,mapping):
        self.calls+=1
        return (self.logits[:legal.shape[1]].expand(len(states),-1).masked_fill(~legal,-torch.inf),)


def probe_fixture(lengths=(3,3,3,3),microbatch=2):
    states,actions,_,mapping=model_inputs(len(lengths),3)
    rows=[]
    for i,length in enumerate(lengths):
        choices=[dict(kind='loan'),dict(kind='pass'),dict(kind='pass')][:length]
        row=dict(choices=choices,chosen=0,seat=i%4,intent=[0.]*8,
            behavior=dict(probability=1/length),view=dict(context=dict(era='CanalEra',phase='action')))
        encoded=Encoded(states[i].numpy(),actions[i,:length].numpy(),mapping[i].numpy(),tuple(str(j) for j in range(length)))
        rows.append((row,dict(win_share=[1.,0.,0.,0.],placement=[1.,2.,3.,4.],vp=[100.,90.,80.,70.]),('g',i%4),encoded))
    dataset=updates.Dataset(rows,'cpu')
    model=Logits().eval();behavior=SimpleNamespace(exploration=.08,temperature=1.)
    cfg=dict(lr_fixtures.config()['update'],microbatch_size=microbatch,bucket=False)
    with torch.no_grad():
        for part in dataset.microbatches(range(len(dataset.rows)),microbatch,cfg['max_candidate_cells'],False):
            ix,inputs,q=dataset.batch(part,.08)
            p=updates.probabilities(model(*inputs)[0],q,.08,1.)
            dataset.old[ix]=p[:,0]
    indices=[i for i,length in enumerate(lengths) if length>1]
    return model,dataset,behavior,cfg,CategoricalProbe(model,dataset,indices,behavior,cfg)


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_unchosen_shift_is_detected_and_old_snapshot_stays_fixed(self):
        model,ds,behavior,cfg,probe=probe_fixture()
        cached=[before.clone() for _,before in probe.parts]
        rng=torch.get_rng_state().clone()
        self.assertAlmostEqual(probe.measure_step(model)['categorical_kl'],0.,places=7)
        with torch.no_grad():model.logits.copy_(torch.log(torch.tensor([.2,.79,.01])))
        before_calls=model.calls
        measured=probe.measure_step(model)
        self.assertEqual(model.calls-before_calls,len(probe.parts))
        self.assertAlmostEqual(measured['sampled_kl'],0.,places=7)
        expected=0.
        for part,old in probe.parts:
            _,inputs,q=ds.batch(part,.08)
            new=updates.probabilities(model(*inputs)[0],q,.08,1.).detach().double()
            expected+=float((old.double()*(old.double().log()-new.log())).sum())
        self.assertAlmostEqual(measured['categorical_kl'],expected/probe.positions,places=6)
        self.assertGreater(measured['categorical_kl'],.1)
        self.assertEqual(measured,probe.measure_step(model))
        self.assertAlmostEqual(measured['categorical_kl'],probe.measure(model)['categorical_kl'],places=6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(model.logits.grad)
        for old,(_,current) in zip(cached,probe.parts):self.assertTrue(torch.equal(old,current))

    def test_variable_masks_forced_filter_and_weighted_mean(self):
        model,ds,behavior,cfg,probe=probe_fixture((3,2,1,3),microbatch=2)
        self.assertEqual(probe.positions,3)
        with torch.no_grad():model.logits.add_(torch.tensor([.1,-.2,.3]))
        measured=probe.measure_step(model)
        self.assertAlmostEqual(measured['categorical_kl'],probe.measure(model)['categorical_kl'],places=6)
        expected=updates.policy_measure(model,ds,[0,1,3],behavior,cfg,Timings())
        self.assertAlmostEqual(measured['sampled_kl'],expected,places=6)
        self.assertEqual(probe.parts[0][1][1,-1],0.)
        self.assertTrue(all(torch.allclose(old.sum(-1),torch.ones(len(part))) for part,old in probe.parts))

    def test_nonfinite_new_or_cached_old_distribution_is_rejected(self):
        for source in ('new','old','sampled'):
            model,ds,_,_,probe=probe_fixture()
            with torch.no_grad():
                if source=='new':model.logits[1]=float('nan')
                elif source=='old':probe.parts[0][1][0,1]=float('inf')
                else:ds.old[0]=float('nan')
            with self.subTest(source=source),self.assertRaisesRegex(ValueError,'Nonfinite PPO KL probe'):
                probe.measure_step(model)

    def test_prefetched_batches_keep_snapshot_alignment(self):
        model,_,_,_,probe=probe_fixture((3,2,1,3),microbatch=2)
        with torch.no_grad():model.logits.add_(torch.tensor([.1,-.2,.3]))
        expected=probe.measure_step(model)
        with patch.object(torch.Tensor,'pin_memory',lambda self:self),ThreadPoolExecutor(max_workers=2) as pool:
            actual=probe.measure_step(model,executor=pool)
        self.assertEqual(actual,expected)

    def test_empty_probe_does_not_forward_or_change_rng(self):
        model,_,_,_,probe=probe_fixture((1,1,1,1))
        before=model.calls;rng=torch.get_rng_state().clone()
        self.assertEqual(probe.measure_step(model),dict(positions=0,categorical_kl=0.,sampled_kl=0.))
        self.assertEqual(model.calls,before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        torch.manual_seed(361)
        model=FourSeatNet(ModelConfig.preset('small'))
        helper=lr_fixtures.OptimizerIntegration()
        behavior,data=helper.data(model);optimizer=helper.optimizer(model)
        cfg=dict(lr_fixtures.config()['update'],effective_batch_size=2,microbatch_size=2,critic_batch_size=2)
        return model,optimizer,behavior,data,cfg

    def run_update(self,exact=None,sampled=None,full=None):
        model,optimizer,behavior,data,cfg=self.fixture()
        real=CategoricalProbe.measure_step;real_full=updates.policy_measure
        probes=[]
        def measure(probe,*args,**kwargs):
            probes.append(probe)
            result=real(probe,*args,**kwargs)
            if exact is not None:result['categorical_kl']=exact
            if sampled is not None:result['sampled_kl']=sampled
            return result
        def secondary(*args,**kwargs):
            value=real_full(*args,**kwargs)
            verify=kwargs.get('verify',args[6] if len(args)>6 else False)
            return value if verify or full is None else full
        with patch.object(CategoricalProbe,'measure_step',measure),patch.object(updates,'policy_measure',side_effect=secondary):
            report=updates.ppo_update(model,optimizer,[dict(split='train')],behavior,cfg,learner_data=data)
        return report,optimizer,probes

    def test_exact_stops_first_step_when_sampled_probe_is_zero(self):
        report,optimizer,probes=self.run_update(exact=.02,sampled=0.)
        self.assertEqual(report['optimizer_steps'],1);self.assertEqual(report['optimizer_steps_planned'],8)
        self.assertEqual(report['optimizer_steps_fraction'],1/8)
        self.assertTrue(report['kl_early_stop'])
        guard=report['kl_guard']
        self.assertEqual(guard['contract'],PPO_KL_GUARD)
        self.assertEqual((guard['stop_reason'],guard['stop_step'],guard['probe_checks']),('categorical_probe',1,1))
        self.assertEqual(report['epochs'][0]['probe_categorical_kl'],.02)
        self.assertEqual(report['epochs'][0]['probe_kl'],0.)
        self.assertEqual(optimizer.param_groups[0][WARMUP_KEY]['completed_steps'],1)
        history=compact(dict(actor=report))['actor']
        self.assertEqual(history['kl_guard'],guard)
        self.assertEqual(history['epochs'][0]['probe_categorical_kl'],.02)
        digest(history)

    def test_sampled_probe_does_not_trigger_primary_stop_and_boundary_is_strict(self):
        report,_,probes=self.run_update(exact=.01,sampled=.02)
        self.assertFalse(report['kl_early_stop']);self.assertEqual(report['optimizer_steps'],8)
        self.assertEqual(len(probes),8);self.assertTrue(all(p is probes[0] for p in probes))
        self.assertIsNone(report['kl_guard']['stop_reason'])
        self.assertEqual(report['kl_guard']['full_dataset_checks'],2)

    def test_full_dataset_sampled_guard_still_stops_at_epoch_boundary(self):
        report,_,_=self.run_update(exact=.001,sampled=0.,full=.02)
        self.assertEqual(report['optimizer_steps'],4)
        self.assertTrue(report['kl_early_stop'])
        self.assertEqual(report['kl_guard']['stop_reason'],'sampled_full_dataset')
        self.assertEqual(report['kl_guard']['stop_step'],4)

    def test_nonfinite_full_dataset_check_cannot_pass_silently(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite full-dataset PPO sampled KL'):
            self.run_update(exact=.001,full=float('nan'))

    def test_nonfinite_policy_after_step_prevents_another_optimizer_step(self):
        model,optimizer,behavior,data,cfg=self.fixture()
        real=updates.scheduled_step;steps=[]
        def poison(opt,**kwargs):
            real(opt,**kwargs);steps.append(1)
            with torch.no_grad():next(model.parameters()).fill_(float('nan'))
        with patch.object(updates,'scheduled_step',side_effect=poison),self.assertRaisesRegex(ValueError,'Nonfinite PPO KL probe'):
            updates.ppo_update(model,optimizer,[dict(split='train')],behavior,cfg,learner_data=data)
        self.assertEqual(len(steps),1)

    def test_all_forced_rows_keep_critic_updates_without_an_actor_probe(self):
        model,optimizer,behavior,data,cfg=self.fixture()
        forced=[]
        for row,target,identity,encoded in data:
            row=dict(row,chosen=0,choices=row['choices'][:1],behavior=dict(row['behavior'],probability=1.))
            encoded=Encoded(encoded.state,encoded.actions[:1],encoded.stable_to_relative,encoded.choice_ids[:1])
            forced.append((row,target,identity,encoded))
        report=updates.ppo_update(model,optimizer,[dict(split='train')],behavior,cfg,learner_data=forced)
        self.assertEqual(report['meaningful_decisions'],0)
        self.assertEqual(report['optimizer_steps'],8)
        self.assertEqual(report['kl_guard']['probe_positions'],0)
        self.assertFalse(report['kl_early_stop'])

    def test_lr_feedback_uses_actual_stop_and_step_fraction(self):
        report,optimizer,_=self.run_update(exact=.02,sampled=0.)
        # Only external game health/collection identity is irrelevant here.
        with patch.object(lr_controller,'collection_signature',return_value='a'*64),patch.object(
                lr_controller,'game_health',return_value=dict(selfplay_vp_per_seat=100.,zero_vp_fraction=0.)):
            result=lr_controller.finish_iteration(optimizer,report,lr_fixtures.config(),{},[],None,{},'anchor')
        self.assertEqual(result['decision'],'down')
        self.assertEqual(result['early_stop_reason'],'categorical_probe')
        self.assertEqual((result['steps_done'],result['steps_planned']),(1,8))
        self.assertAlmostEqual(result['multiplier_after'],1/1.5)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        temporary=tempfile.TemporaryDirectory(prefix='categorical-kl-resume-',dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(temporary.cleanup);run=Path(temporary.name).resolve()
        cfg=lr_fixtures.config();cfg['model_size']='small'
        cfg['learning_rate_warmup']['selfplay_steps']=0
        previous=dict(config=cfg,compatibility=compatibility(),
            implementation='e0dc28ad559ddcd2d6b0eff95d1dac61c4ede949f70bf341a16d08e2ca71dca3',legacy_pool={},teachers={})
        # The archived guard source predates v11.2. Exercise the current-layout
        # performance-only route without approving an old architecture.
        target=dict(previous,implementation=implementation_id())
        torch.manual_seed(761);model=FourSeatNet(ModelConfig.preset('small'))
        helper=lr_fixtures.OptimizerIntegration();optimizer=helper.optimizer(model)
        # Nonempty Adam moments and a partly accumulated LR-controller window.
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean() for p in model.parameters()).backward();updates.scheduled_step(optimizer,ppo=True)
        control=optimizer.param_groups[0][lr_controller.KEY]
        for _ in range(2):control,_=lr_controller.decide(control,lr_fixtures.measurements())
        optimizer.param_groups[0][lr_controller.KEY]=control
        path=run/'generation-0002.pt'
        identity=save_checkpoint(path,model,optimizer,step=2,training=dict(mode='step5-population-ppo',
            fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(previous)))
        entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=2)
        state=dict(format='step5-recovery-league/2.2.0',signature=previous,iteration=2,games_done=640,
            next_seed=18510640,context_counter=2048,search_seed_cursor=19000000,next_search_games=800,
            ordinary_seconds=20.,search_seconds=0.,history=[entry],population=[entry],evaluation_anchor=entry)
        write_json(run/'state.json',state)
        return run,state,target,model,optimizer

    def test_reviewed_resume_preserves_state_and_next_update(self):
        run,state,target,model,optimizer=self.fixture()
        checkpoint=SOURCE_ROOT/state['history'][-1]['path'];before=checkpoint.read_bytes()
        restored,raw,changed=load_resume(SOURCE_ROOT,run,state,target,reviewed=True)
        self.assertTrue(changed)
        opt=torch.optim.AdamW(restored.parameters(),lr=1.)
        resume_optimizer(raw,opt);apply_optimizer_config(opt,target['config'])
        self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()))
        self.assertTrue(_same_payload(optimizer.state_dict(),opt.state_dict()))
        self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
        committed=commit_transition(run,state,target)
        for key,value in state.items():
            if key!='signature':self.assertEqual(committed[key],value)
        self.assertEqual(checkpoint.read_bytes(),before)
        _,again,changed=load_resume(SOURCE_ROOT,run,committed,target)
        self.assertFalse(changed);self.assertEqual(again['model_id'],raw['model_id'])
        helper=lr_fixtures.OptimizerIntegration();behavior,data=helper.data(model)
        cfg=dict(target['config']['update'],effective_batch_size=2,critic_batch_size=2,microbatch_size=2)
        rng=torch.get_rng_state().clone();reports=[]
        for net,adam in ((model,optimizer),(restored,opt)):
            torch.set_rng_state(rng)
            reports.append(updates.ppo_update(net,adam,[dict(split='train')],behavior,cfg,learner_data=data))
        self.assertEqual(reports[0]['kl_guard'],reports[1]['kl_guard'])
        self.assertEqual(reports[0]['epochs'],reports[1]['epochs'])
        self.assertTrue(_same_payload(model.state_dict(),restored.state_dict()))
        self.assertTrue(_same_payload(optimizer.state_dict(),opt.state_dict()))

    def test_review_and_pending_collection_checks_remain_strict(self):
        run,state,target,_,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,target)
        for mutation in ('source','config'):
            bad=copy.deepcopy(target)
            if mutation=='source':bad['implementation']='0'*64
            else:bad['config']['exploration']=.07
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):check_signature(state,bad,True)
        pending=run/'iteration-0002';pending.mkdir();(pending/'collection.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):load_resume(SOURCE_ROOT,run,state,target,reviewed=True)

    def test_prior_source_reviews_are_unchanged_and_il_source_is_admitted(self):
        entries=expand_source_reviews(read(SOURCE_ROOT/'configs/learning/selfplay-resume-sources.json'))['transitions']
        self.assertEqual(digest(entries[:2158]),'6cb5973d40e01cd2e10bfe64770da4b4cfaca55be1561f8ee4f29c3dafc45d0e')
        current=[e for e in entries if e['to_implementation']==GUARD_IMPLEMENTATION]
        self.assertEqual(len(current),1);self.assertEqual(current[0]['from_implementation'],PREVIOUS)
        self.assertEqual(current[0]['learning_compatibility'],'b3b944dbc492bc1483c41bdb5a9d566f13e2b2a8b7988dbeec8fcb9ebb372674')
        self.assertFalse(current[0].get('config_changes'))
        implementation=implementation_id()
        compatible_source='e0dc28ad559ddcd2d6b0eff95d1dac61c4ede949f70bf341a16d08e2ca71dca3'
        self.assertTrue(any(e['to_implementation']==implementation and e['from_implementation']==compatible_source
                            and e['learning_compatibility']==digest(compatibility()) for e in entries))
        handoff=read(SOURCE_ROOT/'configs/learning/imitation-handoff-sources.json')['compatible_sources']
        # New completed-IL receipts may be prepended (bounded rehearsal does
        # this). The historical entries and their order must still be exact.
        start=next(i for i,e in enumerate(handoff) if e['implementation']=='86fd379b6fe6fb4da8e2b977d9c6ef98aca870d0cd35cb6a6c0f9b2080f2916a')
        self.assertEqual(digest(handoff[start:start+47]),'506f7a6e45e3a2e139bdf37972f6c2ca29bc3e926a3cfa726a3e9c1eeddbc225')
        self.assertTrue(any(e['implementation']==compatible_source and e['learning_compatibility']==digest(compatibility()) for e in handoff))


if __name__=='__main__':unittest.main()
