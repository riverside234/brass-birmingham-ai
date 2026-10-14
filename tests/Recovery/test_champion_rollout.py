"""Staged frozen-opponent exposure, strict reward resume and retained evidence."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery import champion_mix as mix
from brass_ai.recovery.config import CHAMPION_OPPONENTS,CHAMPION_MIX_CONFIG_MIGRATION,champion_mix_config,validate_config,table_rotation
from brass_ai.recovery.execution import neural_spec
from brass_ai.recovery.history import compact
from brass_ai.recovery.league import require_start_generation
from brass_ai.recovery.retention import plan,_same_payload
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.recovery.utility import reward_spec,migrate,calibration_ready
from brass_ai.step5.league import write_json,implementation_id


def entry(generation):return dict(generation=generation,model_id=str(generation),path=f'population/{generation}.pt',sha256='test')


def fixture():
    config=read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json')
    state=dict(signature=dict(config=config),games_done=51200,iteration=160,champion=entry(160),
        evaluation_anchor=entry(0),population=[entry(i) for i in range(161)])
    mix.promoted(state,entry(140))
    return state,config


class RolloutTests(unittest.TestCase):
    def test_boundaries_use_persisted_ordinary_games(self):
        state,cfg=fixture()
        for elapsed,stage,count in [(0,'one_champion',1),(1599,'one_champion',1),
            (1600,'balanced_transition',None),(3199,'balanced_transition',None),(3200,'two_champions',2),(99999,'two_champions',2)]:
            state['games_done']=51200+elapsed
            entries,receipt=mix.opponents(state,cfg,random.Random(10))
            self.assertEqual(receipt['stage'],stage)
            if count:self.assertEqual(receipt['champion_opponents'],count)
            self.assertIn(state['evaluation_anchor'],entries)
        cfg['games_per_iteration']=16
        self.assertEqual(mix.status(state,cfg)['stage_games'],1600)

    def test_transition_balanced_deterministic_and_survives_odd_commit(self):
        state,cfg=fixture();state['games_done']+=1600
        def batch(start,n):return [mix.opponents(state,cfg,random.Random(i),transition_index=i)[1] for i in range(start,start+n)]
        all_rows=batch(0,224)
        self.assertEqual(sum(r['champion_opponents']==2 for r in all_rows),112)
        for i in range(0,224,2):self.assertEqual(sorted(r['champion_opponents'] for r in all_rows[i:i+2]),[1,2])
        first=batch(0,71);state['champion_mix']['transition_tables']=71
        state=json.loads(json.dumps(state))
        self.assertEqual(first+batch(state['champion_mix']['transition_tables'],153),all_rows)
        self.assertEqual(all_rows,batch(0,224))
        # Balanced selection is not permanently coupled to a seat parity.
        self.assertGreater(len({i%2 for i,r in enumerate(all_rows) if r['champion_opponents']==2}),1)

    def test_hard_limit_previous_anchor_and_learner_only_after_rotation(self):
        state,cfg=fixture();learner=neural_spec('160',.05,'mixed')
        for elapsed in (0,1600,3200,99999):
            state['games_done']=51200+elapsed
            for index in range(100):
                entries,r=mix.opponents(state,cfg,random.Random(index),transition_index=index,current_id='learner')
                self.assertLessEqual(sum(e['model_id']=='160' for e in entries),2)
                self.assertEqual(entries[-1]['model_id'],'0')
                if r['champion_opponents']==1:self.assertEqual(entries[1]['model_id'],'140')
                policies=mix.policies(learner,entries);rotation=table_rotation(index)
                rotated=[policies[(s-rotation)%4] for s in range(4)]
                self.assertEqual(rotated.count(learner),1)
        cfg['champion_opponents']['max_champion_opponents']=1
        self.assertEqual(mix.opponents(state,cfg,random.Random(0))[1]['champion_opponents'],1)
        self.assertEqual(mix.status(state,cfg)['stage'],'one_champion_override')

    def test_limited_population_and_initial_identity_never_bypass_cap(self):
        state,cfg=fixture();state['champion_mix']['previous']=entry(0)
        state['population']=[entry(0),entry(2),entry(160)]
        entries,r=mix.opponents(state,cfg,random.Random(0))
        self.assertEqual([e['generation'] for e in entries],[160,2,0])
        state['population']=[entry(160)]
        self.assertEqual([e['generation'] for e in mix.opponents(state,cfg,random.Random(0))[0]],[160,0,0])
        state['champion']=entry(0);mix.promoted(state,entry(0))
        entries,r=mix.opponents(state,cfg,random.Random(0))
        self.assertEqual(r['champion_opponents'],1);self.assertEqual(r['bootstrap_legacy_slots'],2)
        learner=neural_spec('learner',.05,'mixed');fallback=[dict(policy_id='legacy-a'),dict(policy_id='legacy-b')]
        self.assertEqual(mix.policies(learner,entries,fallback)[-2:],fallback)
        with self.assertRaisesRegex(ValueError,'two frozen legacy'):mix.policies(learner,entries)

    def test_feature_introduction_and_promotions_preserve_anchor(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=SOURCE_ROOT;run=Path(tmp);state,cfg=fixture();state['history']=state['population']
            for e in state['population']:
                path=run/f"weights-{e['model_id']}.pt";path.write_bytes(e['model_id'].encode())
                e.update(path=path.relative_to(root).as_posix(),sha256=sha(path))
            state['evaluation_anchor']=state['population'][0];state['champion']=state['population'][160]
            state['champion_gate']=dict(accepted_gate=dict(incumbent_generation=140))
            state.pop('champion_mix');state['games_done']=57280
            with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,existing:e):
                self.assertTrue(mix.initialize(root,run,state))
            self.assertEqual(mix.status(state,cfg)['elapsed_ordinary_games'],0)
            self.assertEqual(state['champion_mix']['previous']['generation'],140)
            state['champion_mix']['transition_tables']=73;state['games_done']+=1800
            before=copy.deepcopy(state['champion_mix'])
            self.assertFalse(mix.initialize(root,run,state));self.assertEqual(before,state['champion_mix'])
            previous=state['champion'];state['champion']=entry(180);mix.promoted(state,previous)
            self.assertEqual(state['champion_mix']['anchor'],state['evaluation_anchor'])
            self.assertEqual(state['champion_mix']['origin_games'],59080)
            self.assertEqual(state['champion_mix']['transition_tables'],0)

    def test_retention_and_compact_history_keep_opponents_and_measurements(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);state,cfg=fixture();(run/'population').mkdir()
            for key in ('anchor','previous'):
                e=state['champion_mix'][key];p=run/'population'/f"weights-{key}.pt";p.write_bytes(key.encode())
                e.update(path=p.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(p))
            orphan=run/'population/weights-orphan.pt';orphan.write_bytes(b'orphan')
            state.update(format='step5-recovery-league/2.2.0',population=[],history=[],champion=None)
            write_json(run/'state.json',state)
            retired=[r['path'] for r in plan(SOURCE_ROOT,run)['entries']]
            self.assertIn('population/weights-orphan.pt',retired)
            for key in ('anchor','previous'):self.assertNotIn(f'population/weights-{key}.pt',retired)
        state,cfg=fixture();rows=[mix.opponents(state,cfg,random.Random(i))[1] for i in range(10)]
        summary=mix.summarize(state,cfg,rows)
        self.assertEqual(compact(dict(selfplay_opponents=summary))['selfplay_opponents'],summary)

    def test_generation_guard_fails_before_training(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);write_json(run/'state.json',dict(iteration=179))
            with self.assertRaisesRegex(ValueError,'160'):require_start_generation(run,160)
            write_json(run/'state.json',dict(iteration=160));require_start_generation(run,160)

    def test_defaults_preserve_teacher_gate_and_composition(self):
        cfg=read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json');validate_config(cfg)
        self.assertEqual(cfg['champion_opponents'],CHAMPION_OPPONENTS)
        self.assertEqual(cfg['curriculum'][-1],dict(until=None,selfplay=70,historical=25,legacy=5))
        self.assertEqual(cfg['teacher_work_budget']['seconds_per_phase'],dict(canal_early=12.,canal_late=16.,rail_early=20.,rail_late=20.))
        self.assertEqual((cfg['search_games'],cfg['search_every_games'],cfg['distill_steps']),(5,160,8))
        self.assertIsNone(cfg['search_wall_fraction']);self.assertEqual(cfg['insolvency'],dict(event_cost=.005,max_cost=.075))
        self.assertEqual(cfg['search']['reward'],reward_spec(cfg))
        self.assertEqual(champion_mix_config(cfg),cfg)
        cfg['champion_opponents']['max_champion_opponents']=3
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_manual_one_seat_option_requires_exact_review_and_keeps_clock(self):
        state,cfg=fixture();state['format']='step5-recovery-league/2.2.0'
        state['signature'].update(compatibility=compatibility(),implementation=implementation_id())
        state['games_done']+=4000;before=copy.deepcopy(state['champion_mix'])
        one=copy.deepcopy(state['signature']);one['config']['champion_opponents']['max_champion_opponents']=1
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,one)
        self.assertTrue(check_signature(state,one,reviewed=True))
        self.assertEqual(mix.status(state,one['config'])['stage'],'one_champion_override')
        self.assertEqual(state['champion_mix'],before)
        unsafe=copy.deepcopy(one);unsafe['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,unsafe,reviewed=True)
        state['signature']=one;two=copy.deepcopy(one);two['config']['champion_opponents']['max_champion_opponents']=2
        self.assertTrue(check_signature(state,two,reviewed=True))


class RewardResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_cap_upgrade_preserves_tensors_adam_rng_and_requires_recalibration(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json');cfg['model_size']='small'
            oldcfg=copy.deepcopy(cfg);oldcfg.pop('champion_opponents')
            oldcfg['insolvency']['max_cost']=.025;oldcfg['search']['reward']['insolvency']['max_cost']=.025
            old=dict(config=oldcfg,compatibility=compatibility(),implementation='old',legacy_pool={},teachers={})
            new=dict(old,config=cfg,implementation=implementation_id())
            model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(oldcfg)))
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
            sum(p.square().sum() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad()
            model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility))
            path=run/'generation-0160.pt';identity=save_checkpoint(path,model,optimizer,step=160,
                training=dict(signature=digest(old),fresh_run=run.relative_to(SOURCE_ROOT).as_posix()))
            e=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),model_id=identity,sha256=sha(path),generation=160)
            state=dict(format='step5-recovery-league/2.2.0',iteration=160,games_done=51200,signature=old,
                history=[e],population=[e],evaluation_anchor=e,champion=e,next_search_games=51360)
            approvals=dict(format='selfplay-resume-sources/1.0.0',transitions=[dict(from_implementation='old',
                to_implementation=new['implementation'],learning_compatibility=digest(new['compatibility']),
                config_migrations=[CHAMPION_MIX_CONFIG_MIGRATION])])
            with patch('brass_ai.recovery.resume.read',return_value=approvals):
                with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):load_resume(SOURCE_ROOT,run,state,new)
                restored,raw,changed=load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
                self.assertTrue(changed)
                opt=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,opt)
                tensors=copy.deepcopy(restored.state_dict());moments=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
                receipt=migrate(restored,opt,reward_spec(cfg))
                self.assertEqual(receipt['version'],'insolvency-cap-extension/1.0.0')
                self.assertTrue(_same_payload(tensors,restored.state_dict()));self.assertTrue(_same_payload(moments,opt.state_dict()))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(calibration_ready(restored))
                committed=commit_transition(run,state,new)
                again,_,changed=load_resume(SOURCE_ROOT,run,committed,new)
                self.assertFalse(changed);self.assertIsNotNone(migrate(again,opt,reward_spec(cfg)))
                bad=copy.deepcopy(new);bad['config']['reward']['vp']=.002
                with self.assertRaises(ValueError):load_resume(SOURCE_ROOT,run,state,bad,reviewed=True)
            self.assertEqual(sha(path),e['sha256'])
            with self.assertRaisesRegex(ValueError,'explicit reward migration'):
                migrate(model,optimizer,reward_spec(cfg))


if __name__=='__main__':unittest.main()
