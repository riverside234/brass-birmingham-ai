"""Narrow reviewed g1340 migration, fresh target caches and behavior correctness."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.reward import CostLedger
from brass_ai.recovery import resume,updates,domino_state as D
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.training_revision_g1340 import configure,recipe,CURRENT,MIGRATION
from brass_ai.recovery.utility import reward_spec,reviewed_reward_upgrade
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.domino_auxiliary import Diversity
from brass_ai.step5.league import implementation_id,write_json
from test_context_evaluation import recipe as previous_recipe,RUN
import test_training_g1320 as cache_tests
from test_training_g1322 import smoke_config as previous_smoke
from test_training_g1322 import configure as g1322
from test_domino import fixture,collect


def old_smoke():
    m,o,c=previous_smoke();c=g1322(c);m.config=replace(m.config,utility=reward_spec(c))
    return m,o,c


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_idempotent_revision_and_fresh_defaults(self):
        old=previous_recipe();before=copy.deepcopy(old);new=validate_config(configure(old))
        self.assertEqual(old,before);self.assertEqual(configure(new),new)
        self.assertEqual({k for k in new if new[k]!=old.get(k)},
            {'insolvency','search','exploration','context_evaluation'})
        self.assertEqual(new['insolvency'],CURRENT);self.assertEqual(new['exploration'],.05)
        self.assertEqual(new['search'],dict(old['search'],reward=reward_spec(new)))
        self.assertEqual(new['value_phase'],old['value_phase'])
        self.assertEqual(new['update']['fixed_entropy']['weights'],dict.fromkeys(('canal_action','canal_other','rail_action','rail_other'),.10))
        self.assertEqual(validate_config(old),old)
        for origin in (old,new):
            fresh=latest_model_config(origin,origin['imitation_checkpoint'])
            self.assertEqual(fresh['insolvency'],dict(event_cost=.02,max_cost=.075));self.assertEqual(fresh['exploration'],.05)
            self.assertEqual(latest_model_config(fresh,fresh['imitation_checkpoint']),fresh)
        for change in (dict(exploration=.06),dict(insolvency=dict(event_cost=.01,max_cost=.05))):
            with self.assertRaises(ValueError):configure(dict(old,**change))
        bad=copy.deepcopy(old);bad['search']['reward']['insolvency']['event_cost']=.01
        with self.assertRaisesRegex(ValueError,'Search/learner'):configure(bad)

    def test_new_rewards_and_existing_cache_adoption_preserve_optimizer_rng(self):
        old=previous_recipe();new=configure(old)
        self.assertTrue(reviewed_reward_upgrade(reward_spec(old),reward_spec(new)))
        bad=reward_spec(new);bad['vp']+=.01
        self.assertFalse(reviewed_reward_upgrade(reward_spec(old),bad))
        ledger=CostLedger(reward_spec(new))
        for n in range(1,13):
            ledger.charge(0);self.assertAlmostEqual(float(ledger.spent[0]),min(n*.01,.075))
        with patch.object(cache_tests,'prior_config',old_smoke),patch.object(cache_tests,'configure',recipe):
            cache_tests.Revision().test_refresh_is_after_commit_retry_safe_and_preserves_training_state()

    def test_reward_regime_keeps_duals_but_archives_incompatible_quality(self):
        old=previous_recipe();new=configure(old);s=D.initial(old['diversity'],1326,D.regime(old))
        s.update(generation=1340,quality=[.4]*7,seat_games=[12]*7,distinct_games=[12]*7,
            logits=[.1]*6,adam=dict(step=3,m=[.1]*6,v=[.2]*6))
        before=copy.deepcopy(s);out=D.change_regime(s,new['diversity'],D.regime(new))
        self.assertEqual(s,before);self.assertEqual(out['adam'],s['adam']);self.assertEqual(out['logits'],s['logits'])
        self.assertEqual(out['quality'],[None]*7);self.assertEqual(out['archive'][-1]['quality'],s['quality'])
        self.assertEqual(D.change_regime(out,new['diversity'],D.regime(new)),out)

    def test_helper_from_1340_is_read_only_generation_bound_and_reviewed(self):
        spec=importlib.util.spec_from_file_location('g1340_helper',SOURCE_ROOT/'scripts/configure-training-g1340.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state=read(SOURCE_ROOT/RUN/'state.json');state.update(iteration=1340)
        state['history'][-1].update(generation=1340,path='artifacts/step5/demo/generation-1340.pt')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/demo';run.mkdir(parents=True)
            cp=run/'generation-1340.pt';cp.write_bytes(b'isolated fixture')
            state['history'][-1]['sha256']=sha(cp);write_json(run/'state.json',state)
            before=(sha(run/'state.json'),sha(cp));output='configs/step5/g1340.json'
            result=helper.generate(root,run,output,1340)
            self.assertEqual(result['next_context_generation'],1390);self.assertFalse(result['run_modified'])
            self.assertEqual(result,helper.generate(root,run,output,1340))
            self.assertEqual(before,(sha(run/'state.json'),sha(cp)))
            sig=dict(state['signature'],config=configure(state['signature']['config']),implementation=implementation_id())
            self.assertTrue(resume.check_signature(state,sig,reviewed=True))
            with self.assertRaises(ValueError):resume.check_signature(state,sig,reviewed=False)
            bad=copy.deepcopy(sig);bad['config']['learning_rate']*=2
            with self.assertRaises(ValueError):resume.check_signature(state,bad,reviewed=True)
            self.assertFalse(resume.evaluation_receipt_preserved(state,sig,dict(training={'signature':digest(state['signature'])}),{}))
            with self.assertRaisesRegex(ValueError,'Stale expected generation'):helper.generate(root,run,output,1330)
            state['signature']=sig;write_json(run/'state.json',state)
            self.assertEqual(result,helper.generate(root,run,output,1340))
            cp.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint missing/changed'):helper.generate(root,run,output,1340)

    def test_isolated_ppo_ppg_and_incompatible_epsilon_rejection(self):
        m,opt,c=fixture();c=recipe(c);m.config=replace(m.config,utility=reward_spec(c))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));p.diversity=Diversity(m,c['diversity'],regime='g1340-smoke')
            b,rows=collect(m,p,1,games=42)
            self.assertEqual(b.exploration,.05)
            bad=copy.deepcopy(rows);bad[0][0]['behavior']['exploration']=.08
            with self.assertRaises(ValueError):updates.verify_behavior_metadata(updates.Dataset(bad,'cpu'),b)
            reward=reward_spec(c)
            r=updates.ppo_update(m,opt,[dict(split='train')],b,c['update'],learner_data=rows,
                insolvency=reward['insolvency'],reward=reward,phases=p)
            self.assertEqual(r['optimizer_steps'],1)
            value=p.fit(m);self.assertLessEqual(value['probe_kl'],.002+1e-7)
            print('g1340 isolated PPO/PPG smoke:',dict(ppo_steps=r['optimizer_steps'],ppg_probe_kl=value['probe_kl'],exploration=b.exploration),flush=True)


if __name__=='__main__':unittest.main()
