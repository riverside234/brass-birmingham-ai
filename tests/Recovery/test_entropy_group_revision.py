"""g410 continuation: group floors, lower fork rate and preserved committed state."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import exploration as E
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0410-20261021T151332842603Z'


def helper(name):
    spec=importlib.util.spec_from_file_location(name,SOURCE_ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class GroupFloors(unittest.TestCase):
    def setUp(self):
        self.state=read(EXPORT/'state.json');self.old=self.state['signature']['config']
        self.cfg=E.group_floor_config(self.old)

    def test_exact_two_block_migration_and_fresh_defaults(self):
        original=copy.deepcopy(self.old);validate_config(self.old);validate_config(self.cfg)
        self.assertEqual({k for k in self.cfg if self.cfg[k]!=self.old[k]},{'entropy_control','branching'})
        self.assertEqual(self.old,original);self.assertEqual(E.group_floor_config(self.cfg),self.cfg)
        fresh=latest_model_config(self.old,self.old['imitation_checkpoint'])
        self.assertEqual(fresh['entropy_control'],E.ANCHOR_ENTROPY)
        self.assertEqual(fresh['branching'],self.cfg['branching'])
        self.assertEqual(self.cfg['opening'],self.old['opening'])
        self.assertEqual(self.cfg['branching']['rate_per_decision'],.005)
        self.assertTrue(self.cfg['branching']['reset_opening_at_fork'])
        self.assertEqual(self.cfg['branching']['max_per_game'],1)

    def test_group_floors_never_decay_or_reset_saved_baselines(self):
        before=copy.deepcopy(self.state);s=self.state['exploration']
        restored=E.restore(self.cfg,self.state,copy.deepcopy(s))
        for generation in (410,411,1000,100000):
            got=E.floors(self.cfg,restored,generation)
            self.assertEqual(got,{g:v*(.95 if g.endswith('_action') else .9) for g,v in s['baseline'].items()})
            old=E.floors(self.old,s,generation)
            for g in ('canal_action','rail_action'):self.assertEqual(got[g],old[g])
        self.assertEqual(restored,s);self.assertEqual(self.state,before)

    def test_controller_uses_other_floors_and_keeps_health_restart_guards(self):
        s=copy.deepcopy(self.state['exploration'])
        for entry in s['groups'].values():entry.update(alpha=.02,window=[])
        values=dict(groups={g:dict(raw_entropy=.94*b,normalized_entropy=.2) for g,b in s['baseline'].items()})
        for generation in (411,412,413):
            restored=json.loads(json.dumps(s))
            a,r=E.advance(s,self.cfg,values,generation=generation,warmup_done=True,healthy=True)
            self.assertEqual((a,r),E.advance(restored,self.cfg,values,generation=generation,warmup_done=True,healthy=True))
            s=a
        for g,v in r['groups'].items():self.assertEqual(v['decision'],'raise' if g.endswith('_action') else 'relax')
        self.assertEqual(r['floor_ratio_by_group'],self.cfg['entropy_control']['floor_ratio_by_group'])
        saved=copy.deepcopy(s)
        held,report=E.advance(s,self.cfg,values,generation=414,warmup_done=True,healthy=False)
        self.assertEqual(s,saved)
        for g,v in held['groups'].items():self.assertEqual(v['alpha'],s['groups'][g]['alpha'])
        summary=compact(dict(format='step5-iteration-report/1.0.0',entropy_control=report))
        self.assertEqual(summary['entropy_control'],report)

    def test_strict_ratios_and_whole_recipe_validation(self):
        for change in (lambda c:c['entropy_control']['floor_ratio_by_group'].__setitem__('canal_other',.91),
                       lambda c:c['entropy_control']['floor_ratio_by_group'].pop('rail_action'),
                       lambda c:c['entropy_control']['floor_ratio_by_group'].__setitem__('surprise',.9),
                       lambda c:c['entropy_control'].__setitem__('floor_decay_per_iteration',.999),
                       lambda c:c['branching'].__setitem__('rate_per_decision',.015),
                       lambda c:c.__setitem__('entropy_control',copy.deepcopy(self.old['entropy_control']))):
            bad=copy.deepcopy(self.cfg);change(bad)
            with self.assertRaises(ValueError):validate_config(bad)

    def test_old_signed_frozen_recipe_and_branch_contract_keep_their_meaning(self):
        self.assertEqual(E.frozen_config(self.old),self.old)
        self.assertEqual(E.continuation_config(self.old),self.old)
        self.assertEqual(E.HIGH_BRANCH['rate_per_decision'],.015)
        self.assertEqual(E.HIGH_BRANCH['contract'],E.BRANCH['contract'])
        pre={k:v for k,v in self.old.items() if k not in E.BLOCKS}
        self.assertEqual(E.group_floor_config(pre),pre)

    def test_reviewed_continuation_needs_exact_changes(self):
        target=dict(self.state['signature'],config=self.cfg,implementation=implementation_id())
        self.assertTrue(check_signature(self.state,target,reviewed=True))
        with self.assertRaises(ValueError):check_signature(self.state,target)
        for field in ('learning_rate','exploration'):
            bad=copy.deepcopy(target);bad['config'][field]*=1.1
            with self.assertRaises(ValueError):check_signature(self.state,bad,reviewed=True)

    def test_branch_rate_changes_collection_and_controller_regime_identity(self):
        from brass_ai.recovery.execution import _collection_recipe
        from brass_ai.recovery.lr_controller import collection_signature
        from brass_ai.recovery import snapshot_r2
        job=dict(index=0,policies=[],branching=self.old['branching'],provenance=dict(kind='selfplay'))
        new=dict(job,branching=self.cfg['branching'])
        self.assertNotEqual(_collection_recipe([job],self.cfg['execution']),_collection_recipe([new],self.cfg['execution']))
        sampling=dict(by_composition=dict(selfplay=dict(tables=160,learner_trajectories=640)))
        regimes=[]
        for c in (self.old,self.cfg):
            e=E.effective(c,self.state['exploration'],410)
            regimes.append({k:e[k] for k in ('version','opening','raw_entropy','branching')})
        keys=[collection_signature(c,dict(sampling,exploration_regime=e)) for c,e in zip((self.old,self.cfg),regimes)]
        self.assertNotEqual(*keys)
        self.assertNotEqual(snapshot_r2.regime({}, {},regimes[0]),snapshot_r2.regime({}, {},regimes[1]))


class Hotfix(unittest.TestCase):
    def test_preview_apply_and_idempotent_rerun_preserve_all_training_state(self):
        from brass_ai.recovery import resume,policy_anchor,learning_rate,lr_controller
        from brass_ai.recovery.retention import _same_payload
        tool=helper('joint7-entropy-groups');state=read(EXPORT/'state.json');cfg=state['signature']['config']
        torch.set_num_threads(2)
        model=torch.nn.Linear(2,2);adam=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
        learning_rate.configure(adam,cfg['learning_rate'],100,.1,'selfplay')
        lr_controller.configure(adam,cfg['ppo_lr_controller'])
        adam.param_groups[0][lr_controller.KEY]['multiplier']=1.0416666666666667
        model(torch.ones(1,2)).sum().backward();adam.step();adam.zero_grad()
        raw=dict(optimizer=adam.state_dict(),rng=torch.get_rng_state().clone(),cuda_rng=[],
                 training=dict(exploration=copy.deepcopy(state['exploration'])))
        before=copy.deepcopy(raw);weights=copy.deepcopy(model.state_dict())
        # g410 full checkpoints are on the lab machine. Substitute the loader
        # only; source review, Adam/RNG restoration and atomic commit are real.
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';out='configs/step5/groups.json'
            write_json(run/'state.json',state);original=(run/'state.json').read_bytes()
            def restore(root,run,s,target,**kwargs):
                return model,raw,resume.check_signature(s,target,reviewed=kwargs['reviewed'])
            with patch.object(resume,'load_resume',side_effect=restore),patch.object(policy_anchor,'verify_checkpoint'):
                rng=torch.get_rng_state().clone()
                preview=tool.run(root,run,out,expected_generation=410)
                self.assertFalse((root/out).exists());self.assertEqual(original,(run/'state.json').read_bytes())
                result=tool.run(root,run,out,apply=True,expected_generation=410)
                self.assertEqual(result['floor_after'],preview['floor_after'])
                after=read(run/'state.json')
                for k,v in state.items():
                    if k not in ('signature','source_transitions'):self.assertEqual(after[k],v,k)
                self.assertEqual(len(after['source_transitions']),len(state['source_transitions'])+1)
                payload=(run/'state.json').read_bytes();report=run/'exploration/group-floors-hotfix-report.json'
                receipt=report.read_bytes()
                tool.run(root,run,out,apply=True,expected_generation=410)
                self.assertEqual((run/'state.json').read_bytes(),payload);self.assertEqual(report.read_bytes(),receipt)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(_same_payload(before,raw));self.assertTrue(_same_payload(weights,model.state_dict()))
                with self.assertRaisesRegex(ValueError,'Unexpected committed generation'):
                    tool.run(root,run,out,apply=True,expected_generation=409)

    def test_failure_before_commit_preserves_state_and_writes_no_config(self):
        from brass_ai.recovery import resume
        tool=helper('joint7-entropy-groups');state=read(EXPORT/'state.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';out='configs/step5/groups.json'
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes()
            with patch.object(resume,'load_resume',side_effect=ValueError('Uncommitted iteration')):
                with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):tool.run(root,run,out,apply=True)
            self.assertEqual((run/'state.json').read_bytes(),before)
            self.assertFalse((root/out).exists())

    def test_superseded_helper_cannot_restore_uniform_floors_and_high_branch_rate(self):
        state=read(EXPORT/'state.json');state['signature']['config']=E.group_floor_config(state['signature']['config'])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';write_json(run/'state.json',state)
            with self.assertRaisesRegex(ValueError,'superseded'):
                helper('joint7-entropy95').run(root,run,'configs/step5/old.json',apply=True)


if __name__=='__main__':unittest.main()
