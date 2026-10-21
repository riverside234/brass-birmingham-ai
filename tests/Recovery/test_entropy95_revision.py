"""Frozen entropy floors, fresh branch openings and exact g320 continuation."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import exploration as E,branching
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json
from test_exploration_revision import measured

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0320-20261021T052141479749Z'


class Entropy95(unittest.TestCase):
    def setUp(self):
        self.state=read(EXPORT/'state.json');self.old=self.state['signature']['config']
        self.cfg=E.frozen_config(self.old)

    def test_g320_floors_and_long_term_freeze_without_rebaselining(self):
        before=copy.deepcopy(self.state);saved=copy.deepcopy(self.state['exploration'])
        restored=E.restore(self.cfg,self.state,saved)
        for generation in (320,321,1000,100000):
            self.assertEqual(E.floors(self.cfg,restored,generation),
                             {g:.95*v for g,v in saved['baseline'].items()})
        self.assertLess(E.floors(self.old,saved,320)['canal_action'],.8*saved['baseline']['canal_action'])
        self.assertEqual(restored,saved);self.assertEqual(self.state,before)
        self.assertEqual((restored['origin'],restored['armed_at']),(185,197))
        self.assertEqual(E.effective(self.cfg,restored,320)['branching'],E.BRANCH)

    def test_restart_controller_decision_is_identical_and_health_still_blocks(self):
        s=self.state['exploration'];measurement=measured(.1)
        a,r=E.advance(s,self.cfg,measurement,generation=321,warmup_done=True,healthy=True)
        b,_=E.advance(json.loads(json.dumps(s)),self.cfg,measurement,generation=321,warmup_done=True,healthy=True)
        self.assertEqual(a,b);self.assertEqual(a['baseline'],s['baseline'])
        held,report=E.advance(s,self.cfg,measurement,generation=321,warmup_done=True,healthy=False)
        for g in E.GROUPS:
            self.assertGreater(a['groups'][g]['alpha'],s['groups'][g]['alpha'])
            self.assertEqual(held['groups'][g]['alpha'],s['groups'][g]['alpha'])
            self.assertEqual(report['groups'][g]['reason'],'unhealthy')
        with self.assertRaisesRegex(ValueError,'skipped/repeated'):
            E.advance(a,self.cfg,measurement,generation=321,warmup_done=True,healthy=True)

    def test_signed_legacy_recipe_and_only_three_blocks_change(self):
        validate_config(self.old);validate_config(self.cfg)
        self.assertEqual({k for k in self.cfg if self.cfg[k]!=self.old[k]},
                         {'entropy_control','opening','branching'})
        self.assertEqual(E.frozen_config(self.cfg),self.cfg)
        self.assertEqual(E.continuation_config(self.old),self.old)
        pre_adoption={k:v for k,v in self.old.items() if k not in E.BLOCKS}
        self.assertEqual(E.frozen_config(pre_adoption),pre_adoption)
        for key,value in (('floor_start_ratio',.96),('floor_decay_per_iteration',.999),('floor_min_ratio',.9)):
            bad=copy.deepcopy(self.cfg);bad['entropy_control'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        for key in ('opening','branching','entropy_control'):
            bad=copy.deepcopy(self.cfg);bad[key]=copy.deepcopy(self.old[key])
            with self.assertRaises(ValueError):validate_config(bad)

    def test_reviewed_resume_needs_explicit_approval_and_no_other_recipe_edits(self):
        signature=dict(self.state['signature'],config=self.cfg,implementation=implementation_id())
        self.assertTrue(check_signature(self.state,signature,reviewed=True))
        with self.assertRaises(ValueError):check_signature(self.state,signature)
        for key in ('exploration','learning_rate'):
            bad=copy.deepcopy(signature);bad['config'][key]*=1.1
            with self.assertRaises(ValueError):check_signature(self.state,bad,reviewed=True)

    def test_policy_and_collection_identity_change_and_old_temperatures_rejected(self):
        from brass_ai.recovery.execution import neural_spec
        from brass_ai.recovery.updates import verify_behavior_metadata
        old=neural_spec('m',.02,'mixed',version=3,opening=self.old['opening'])
        new=neural_spec('m',.02,'mixed',version=3,opening=self.cfg['opening'])
        self.assertNotEqual(old['policy_id'],new['policy_id'])
        b=dict(kind='model',policy_id=new['policy_id'],probability=.5,temperature=1.5,
               exploration=.02,exploration_version=3,component='policy',action_decision=0)
        row=dict(view=dict(context=dict(phase='action')),behavior=b)
        behavior=SimpleNamespace(exploration=.02,temperature=1.,exploration_version=3,opening=E.OPENING)
        with self.assertRaises(ValueError):verify_behavior_metadata(SimpleNamespace(rows=[row]),behavior)
        b['temperature']=2.;verify_behavior_metadata(SimpleNamespace(rows=[row]),behavior)


class BranchOpenings(unittest.TestCase):
    def test_counts_restart_for_all_seats_only_for_new_contract(self):
        for cfg in (E.LEGACY_BRANCH,E.BRANCH):
            context=dict(phase='action',era='CanalEra')
            row=dict(step=25,seat=2,view=dict(context=context),choices=[{'id':0},{'id':1}],chosen=0,
                     style=[0.]*4,intent=[0.]*8,teacher=None,seconds=0.,
                     behavior=dict(probability=.5,component='policy',temperature=1.,action_decision=3))
            meta=dict(contract=cfg['contract'],seed='12345678',decision=25,chosen=1,marginal_probability=.3)
            packet=dict(row=row,metadata=meta,counts=[3,4,4,2],contexts=[{'s':i} for i in range(4)])
            assigned=[SimpleNamespace(contexts={},action_counts={},opening=E.OPENING) for _ in range(4)]
            host=Mock();host.call.return_value=context;host.apply.return_value={'terminal':True}
            parent=dict(game_id='parent',setup_group='setup',split='train',opponents=['model']*4)
            def turns(host,job,policies,timing,cap,context,game,rngs,encoded,encoder,lease):
                self.assertNotIn('branching',job)
                for seat,p in enumerate(policies):
                    self.assertEqual(p.action_counts[rngs[seat]],0 if cfg==E.BRANCH else packet['counts'][seat])
                    self.assertEqual(p.contexts[rngs[seat]],{'s':seat})
                return game
            with patch('brass_ai.recovery.execution._run_turns',side_effect=turns):
                result=branching.continue_branch(host,dict(branching=cfg),parent,packet,assigned,None,2000,Mock(),None)
            self.assertEqual(result['records'][0]['behavior']['component'],'branch_move')
            self.assertEqual(packet['counts'],[3,4,4,2])

    def test_per_seat_counter_admission_including_intervening_subdecisions(self):
        rows=[]
        for n in range(3):
            for seat in (2,0,3,1):
                rows.append(dict(seat=seat,view=dict(context=dict(phase='bonus')),
                                 behavior=dict(action_decision=None,temperature=1.)))
                rows.append(dict(seat=seat,view=dict(context=dict(phase='action')),
                                 behavior=dict(action_decision=n,temperature=2. if n<2 else 1.)))
        branching.validate_opening_rows(rows)
        for field,value in (('action_decision',5),('temperature',1.)):
            bad=copy.deepcopy(rows);bad[1]['behavior'][field]=value
            with self.assertRaisesRegex(ValueError,'Branch opening'):branching.validate_opening_rows(bad)

    def test_old_fork_seed_preserved_and_new_contract_is_distinct(self):
        self.assertEqual(branching.seed_for('p',3),digest(dict(contract=E.LEGACY_BRANCH['contract'],parent='p',decision=3)))
        self.assertNotEqual(branching.seed_for('p',3),branching.seed_for('p',3,E.BRANCH['contract']))


class HotfixHelper(unittest.TestCase):
    def test_isolated_apply_preview_idempotence_and_state_preservation(self):
        from brass_ai.recovery import resume,policy_anchor,learning_rate,lr_controller
        from brass_ai.recovery.retention import _same_payload
        spec=importlib.util.spec_from_file_location('entropy95_helper',SOURCE_ROOT/'scripts/joint7-entropy95.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state=read(EXPORT/'state.json');cfg=state['signature']['config']
        model=torch.nn.Linear(2,2);adam=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
        learning_rate.configure(adam,cfg['learning_rate'],100,.1,'selfplay')
        lr_controller.configure(adam,cfg['ppo_lr_controller'])
        adam.param_groups[0][lr_controller.KEY]['multiplier']=1.5625
        model(torch.ones(1,2)).sum().backward();adam.step();adam.zero_grad()
        raw=dict(optimizer=adam.state_dict(),rng=torch.get_rng_state().clone(),cuda_rng=[],
                 training=dict(exploration=copy.deepcopy(state['exploration'])))
        before=copy.deepcopy(raw);weights=copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';output='configs/step5/entropy95.json'
            write_json(run/'state.json',state);original=(run/'state.json').read_bytes()
            def restore(root,run,s,target,**kwargs):
                return model,raw,resume.check_signature(s,target,reviewed=kwargs['reviewed'])
            # The g320 weights are remote. Fixture substitutes only their loader
            # and anchor verification; signature/commit/Adam/RNG handling is real.
            with patch.object(resume,'load_resume',side_effect=restore),\
                 patch.object(policy_anchor,'verify_checkpoint') as anchor:
                rng=torch.get_rng_state().clone()
                preview=helper.run(root,run,output,expected_generation=320)
                self.assertEqual(original,(run/'state.json').read_bytes());self.assertFalse((root/output).exists())
                result=helper.run(root,run,output,apply=True,expected_generation=320)
                after=read(run/'state.json');report=(run/'exploration/hotfix-report.json').read_bytes()
                self.assertEqual(result['floor_after'],preview['floor_after'])
                for key,value in state.items():
                    if key not in ('signature','source_transitions'):self.assertEqual(after[key],value,key)
                self.assertIsNotNone(after['policy_anchor']['pending'])
                self.assertEqual(len(after['source_transitions']),len(state['source_transitions'])+1)
                payload=(run/'state.json').read_bytes()
                helper.run(root,run,output,apply=True,expected_generation=320)
                self.assertEqual(payload,(run/'state.json').read_bytes())
                self.assertEqual(report,(run/'exploration/hotfix-report.json').read_bytes())
                self.assertEqual(anchor.call_count,3)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(_same_payload(before,raw));self.assertTrue(_same_payload(weights,model.state_dict()))
                with self.assertRaisesRegex(ValueError,'Unexpected committed generation'):
                    helper.run(root,run,output,apply=True,expected_generation=319)


if __name__=='__main__':unittest.main()
