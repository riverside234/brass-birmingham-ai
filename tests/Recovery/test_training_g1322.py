"""Exact g1322 recipe, real behavior/entropy and fresh-cache continuation."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.reward import CostLedger
from brass_ai.recovery import exploration as E,opening_objective as O,updates
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.training_revision_g1322 import configure,CURRENT
from brass_ai.recovery.training_revision_g1320 import configure as prior_recipe
from brass_ai.recovery.utility import reward_spec,reviewed_reward_upgrade
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.review_metrics import CategoricalProbe
from brass_ai.recovery.branching import validate_opening_rows
from brass_ai.step5.league import write_json,implementation_id
from brass_ai.step5.policy import NeuralPolicy,exploration_distribution
from test_gae_value_phase import fixture
from test_opening_ppo import opening_rows
from test_exploration_revision import measured
from test_ppg_eight import collect as original_collect
from test_value_buffer import source
import test_training_g1320 as prior_tests
import test_training_g1120 as opening_tests
from test_training_g1120_kl32 import smoke_config as base_fixture

PREVIOUS_SOURCE='83ebfadaaccf862ebd06fb78cecb4b34362ad977781adf67f118851df244f015'


def old_recipe():return prior_recipe(read(prior_tests.EXPORT/'recorded-config.json'))


def smoke_config():
    model,opt,c=base_fixture();c=prior_recipe(c);recipe=old_recipe()
    model.config=replace(model.config,utility=reward_spec(c))
    for key in (*E.BLOCKS,'exploration','selfplay_recipe','il_kl_limit'):c[key]=copy.deepcopy(recipe[key])
    return model,opt,c


def collect_revised(model,phase,generation,*,branches=False):
    _,data=original_collect(model,phase,generation,branches=branches)
    c=phase.config
    behavior,data=opening_rows(model,data,c['exploration'],opening=c['opening'],objective=c['update']['opening_objective'])
    behavior.entropy_alphas=copy.deepcopy(c['update']['fixed_entropy']['weights']) if 'fixed_entropy' in c['update'] else dict.fromkeys(E.GROUPS,.005)
    phase.collection_source=dict(source(behavior),snapshot_rows=True)
    return behavior,data


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_recipe_defaults_and_rejection(self):
        old=old_recipe();original=copy.deepcopy(old);new=validate_config(configure(old))
        self.assertEqual(old,original);self.assertEqual(configure(new),new)
        self.assertEqual({k for k in new if old[k]!=new[k]},
            {'insolvency','search','opening','branching','exploration','update'})
        self.assertEqual(new['insolvency'],CURRENT)
        self.assertEqual(new['search'],dict(old['search'],reward=reward_spec(new)))
        self.assertEqual(new['update'],dict(old['update'],opening_objective=O.T2_THREE,fixed_entropy=E.FIXED_ENTROPY))
        self.assertEqual(new['branching'],dict(old['branching'],contract=E.THREE_T2_BRANCH['contract']))
        self.assertEqual(new['opening'],dict(old['opening'],contract=E.THREE_T2_OPENING['contract'],temperature=2.))
        self.assertEqual(new['value_phase'],old['value_phase']);self.assertEqual(new['entropy_control'],old['entropy_control'])
        self.assertEqual(validate_config(old),old)
        for origin in (old,new):
            fresh=latest_model_config(origin,origin['imitation_checkpoint'])
            from brass_ai.recovery.training_revision_g1324 import configure as latest_recipe
            expected=latest_recipe(new)
            for k in ('insolvency','opening','branching','exploration','value_phase'):
                self.assertEqual(fresh[k],expected[k],k)
            self.assertEqual(fresh['update']['fixed_entropy'],E.FIXED_ENTROPY_010)
        for block,key,value in [('opening','action_decisions_per_seat',2),('branching','skip_opening_decisions',2),
            ('entropy_control','mode','apply'),('update','opening_objective',O.DEFAULT),('update','entropy_weight',.015)]:
            bad=copy.deepcopy(new);bad[block][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        for mutate in (lambda c:c.update(exploration=.05),lambda c:c['update'].pop('fixed_entropy'),
            lambda c:c['update']['fixed_entropy']['weights'].update(canal_other=.015)):
            bad=copy.deepcopy(new);mutate(bad)
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(old);bad['insolvency']['max_cost']=.10
        with self.assertRaises(ValueError):configure(bad)

    def test_reward_migration_and_ten_event_cap(self):
        old=old_recipe();new=configure(old);previous,current=reward_spec(old),reward_spec(new)
        self.assertTrue(reviewed_reward_upgrade(previous,current));self.assertFalse(reviewed_reward_upgrade(current,previous))
        for costs in (dict(event_cost=.005,max_cost=.075),dict(event_cost=.01,max_cost=.05)):
            self.assertFalse(reviewed_reward_upgrade(previous,dict(current,insolvency=costs)))
        ledger=CostLedger(current)
        for n in range(1,14):
            ledger.charge(0);self.assertAlmostEqual(float(ledger.spent[0]),min(n*.005,.05))

    def test_fixed_entropy_gradients_reporting_and_baseline_preservation(self):
        c=configure(old_recipe());state=prior_tests.exported_state()
        restored=E.restore(c,state,state['exploration']);before=copy.deepcopy(restored)
        settings=E.effective(c,restored,state['iteration'])
        self.assertEqual(settings['alphas'],E.FIXED_ENTROPY['weights']);self.assertTrue(settings['raw_entropy'])
        advanced,report=E.advance(restored,c,measured(.01),generation=state['iteration']+1,warmup_done=True,healthy=True)
        self.assertEqual(restored,before);self.assertEqual(advanced['baseline'],before['baseline'])
        for g in E.GROUPS:
            self.assertEqual(advanced['groups'][g]['alpha'],before['groups'][g]['alpha'])
            self.assertEqual(report['groups'][g]['effective_alpha'],E.FIXED_ENTROPY['weights'][g])
        self.assertEqual(report['fixed_entropy'],E.FIXED_ENTROPY)
        contexts=[dict(era=era,phase=phase) for era in ('CanalEra','RailEra') for phase in ('action','debt')]
        ds=SimpleNamespace(rows=[dict(view=dict(context=c)) for c in contexts])
        behavior=SimpleNamespace(raw_entropy=True,entropy_alphas=settings['alphas'])
        logits=torch.tensor([[2.,0.,float('-inf')]]*4,requires_grad=True)
        h,term=updates.entropy_term(logits,torch.ones_like(logits)/3,ds,list(range(4)),behavior,c['update'])
        torch.testing.assert_close(term/h,torch.tensor([.015,.005,.015,.005]))
        term.sum().backward();self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[0],3*logits.grad[1]);self.assertEqual(float(logits.grad[:,2].abs().sum()),0.)
        behavior.entropy_alphas=dict.fromkeys(E.GROUPS,.005)
        with self.assertRaisesRegex(ValueError,'matching raw-policy'):updates.entropy_term(logits,logits,ds,list(range(4)),behavior,c['update'])

    def test_real_sampling_three_t2_actions_and_epsilon(self):
        p=NeuralPolicy(None,exploration=.08,focus='neutral',evaluator=SimpleNamespace(model_id='model'),
            exploration_version=3,opening=E.THREE_T2_OPENING)
        old=NeuralPolicy(None,exploration=.05,focus='neutral',evaluator=SimpleNamespace(model_id='model'),
            exploration_version=3,opening=E.THREE_OPENING)
        self.assertNotEqual(p.policy_id,old.policy_id)
        records=[];rngs=[random.Random(i) for i in range(4)]
        for n,phase in enumerate(('action','continuation','action','debt','action','action')):
            for seat,rng in enumerate(rngs):
                choices=[dict(kind='loan')] if n==2 else [dict(kind='loan'),dict(kind='pass')]
                raw=np.array([1.] if n==2 else [.7,.3]);view=dict(context=dict(phase=phase))
                chosen,prob,_=p.sample_training(view,choices,rng,raw,[0]*8)
                b=dict(p.last_behavior[rng],kind='model',policy_id=p.policy_id,probability=prob)
                records.append(dict(seat=seat,view=view,behavior=b))
                tempered=raw**(1/b['temperature']);tempered/=tempered.sum()
                expected=.92*tempered+.08*exploration_distribution(choices,[0]*8,3,phase)
                self.assertAlmostEqual(prob,expected[chosen],places=7)
        validate_opening_rows(records,E.THREE_T2_OPENING)
        for seat in range(4):
            self.assertEqual([r['behavior']['temperature'] for r in records if r['seat']==seat],[2.,1.,2.,1.,2.,1.])
        with self.assertRaises(ValueError):validate_opening_rows(records,E.THREE_OPENING)

    def test_decoupled_three_rows_actual_behavior_and_stale_rejection(self):
        m,_,c,(_,data)=fixture(games=3,decisions=4)
        parent=[d for d in data if d[2]==data[1][2]]
        branch=[(dict(r,_branch=True),t,(g+'/branch/test',s),e) for r,t,(g,s),e in parent]
        b,rows=opening_rows(m,parent+branch,.08,opening=E.THREE_T2_OPENING,objective=O.T2_THREE)
        ds=updates.Dataset(rows,'cpu');ds.opening_reference=O.Reference(ds,b,O.T2_THREE)
        self.assertEqual(ds.opening_reference.mask.tolist(),[True,True,True,False]*2)
        actual=ds.old.clone();updates.policy_measure(m,ds,range(len(rows)),b,c['update'],Timings(),verify=True)
        self.assertTrue(ds.opening_reference.frozen);self.assertTrue(torch.equal(actual,ds.old))
        probe=CategoricalProbe(m,ds,list(range(len(rows))),b,dict(c['update'],opening_objective=O.T2_THREE))
        self.assertEqual(probe.opening_positions,6)
        self.assertAlmostEqual(probe.measure(m)['opening_temperature1_categorical_kl'],0.,places=6)
        for objective in (O.T13,O.DEFAULT):
            with self.assertRaises(ValueError):O.Reference(ds,b,objective)
        for field,value in [('temperature',1.3),('exploration',.05)]:
            bad=copy.deepcopy(rows);bad[0][0]['behavior'][field]=value
            with self.assertRaises(ValueError):updates.verify_behavior_metadata(updates.Dataset(bad,'cpu'),b)
        old_behavior,_=opening_rows(m,parent+branch,.05,opening=E.THREE_OPENING,objective=O.T13)
        with self.assertRaises(ValueError):updates.verify_behavior_metadata(ds,old_behavior)

    def test_postfork_resets_three_decisions_with_new_contract(self):
        # Reuse the full fork/reset regression with the new versioned pair.
        with patch.object(E,'THREE_OPENING',E.THREE_T2_OPENING),patch.object(E,'THREE_BRANCH',E.THREE_T2_BRANCH):
            opening_tests.Revision().test_fork_skips_three_and_resets_all_four_seats()

    def test_fresh_cache_and_optimizer_restart_guards(self):
        # Same audited cleanup/optimizer/RNG test, now .01/.075 -> .005/.05.
        with patch.object(prior_tests,'prior_config',smoke_config),patch.object(prior_tests,'configure',configure):
            prior_tests.Revision().test_refresh_is_after_commit_retry_safe_and_preserves_training_state()

    def test_helper_from_simulated_1322_read_only_and_strict(self):
        spec=importlib.util.spec_from_file_location('g1322_helper',SOURCE_ROOT/'scripts/configure-training-g1322.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        # No g1322 export is supplied. Model its stated two g1320-recipe commits;
        # the real helper verifies the actual lab state and checkpoint hash.
        state=prior_tests.exported_state();state.update(iteration=1322,games_done=423040)
        state['signature'].update(config=old_recipe(),implementation=PREVIOUS_SOURCE)
        state['history'][-1].update(generation=1322,path='artifacts/step5/joint7-v11-2-grouped-lr-critic/generation-1322.pt')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated fixture checkpoint')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            output='configs/step5/g1322.json';before=(sha(run/'state.json'),sha(checkpoint))
            report=helper.generate(root,run,output,1322)
            self.assertEqual(report,helper.generate(root,run,output,1322));self.assertFalse(report['run_modified'])
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            signature=dict(state['signature'],implementation=implementation_id(),config=configure(state['signature']['config']))
            self.assertTrue(check_signature(state,signature,reviewed=True))
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=False)
            bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
            with self.assertRaisesRegex(ValueError,'Stale expected generation'):helper.generate(root,run,output,1320)
            state['signature']=signature;write_json(run/'state.json',state)
            self.assertEqual(report,helper.generate(root,run,output,1322))
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint missing/changed'):helper.generate(root,run,output,1322)


if __name__=='__main__':unittest.main()
