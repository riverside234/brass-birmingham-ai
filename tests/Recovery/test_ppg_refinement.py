"""Smaller value-only steps, stratified leakage barriers and g110 continuation."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import critic_holdout as holdout,value_phase,value_buffer,updates,gae,ev_evidence
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config,insolvency_event_config
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json
from test_gae_value_phase import config,fixture,examples,ppo
from test_critic_descriptors import mixed_fixture
from test_value_buffer import source


def rows(counts=(160,144,16)):
    return [(dict(_table_kind=kind,_critic_role='learner' if seat<2 else 'snapshot'),None,
             (f'{kind}-{game}',seat),None)
            for kind,count in zip(holdout.TABLES,counts) for game in range(count) for seat in range(4)]


class Selection(unittest.TestCase):
    def test_320_games_exact_quotas_order_rng_and_all_seats(self):
        data=rows();rng=torch.get_rng_state().clone();python_rng=random.getstate()
        train,held,report=value_phase.split(data,.05,holdout.DEFAULT)
        self.assertEqual({k:r['heldout_games'] for k,r in report['by_table'].items()},dict(selfplay=8,historical=7,legacy=2))
        self.assertEqual(report['heldout_games'],17);self.assertEqual(report['train_games'],303)
        self.assertEqual(len(held),68);self.assertEqual(len(train),1212)
        reverse=value_phase.split(list(reversed(data)),.05,holdout.DEFAULT)[2]
        self.assertEqual(report,reverse)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(random.getstate(),python_rng)
        self.assertTrue({data[i][2][0] for i in held}.isdisjoint(data[i][2][0] for i in train))
        for game in report['selection']['heldout_games']:
            self.assertEqual({data[i][2][1] for i in held if data[i][2][0]==game},{0,1,2,3})

    def test_small_strata_leave_training_game_and_report_support(self):
        train,held,r=value_phase.split(rows((1,2,3)),.05,holdout.DEFAULT)
        self.assertEqual([r['by_table'][k]['heldout_games'] for k in holdout.TABLES],[0,1,2])
        self.assertEqual(r['by_table']['selfplay']['support'],'singleton_training_only')
        self.assertEqual(r['by_table']['historical']['support'],'limited_population')
        self.assertEqual(value_phase.split([],.05,holdout.DEFAULT)[2]['heldout_games'],0)

    def test_invalid_identity_recipe_and_tampered_selection_rejected(self):
        data=rows((2,2,2));bad=copy.deepcopy(data);bad[1][0]['_table_kind']='legacy'
        with self.assertRaisesRegex(ValueError,'conflicting'):value_phase.split(bad,.05,holdout.DEFAULT)
        bad=copy.deepcopy(data);bad[0][0]['_table_kind']='unknown'
        with self.assertRaises(ValueError):value_phase.split(bad,.05,holdout.DEFAULT)
        for value in (True,0,9):
            with self.assertRaises(ValueError):holdout.validate(dict(holdout.DEFAULT,minimum_games_per_table=value))
        s=value_phase.split(data,.05,holdout.DEFAULT)[2]['selection'];s['heldout_games']=[]
        with self.assertRaisesRegex(ValueError,'selection differs'):holdout.validate_selection(s)


class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_fresh_defaults_and_only_explicit_config_upgrade(self):
        old=config();new=value_phase.refined_options(old)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'value_phase'})
        self.assertEqual(old['value_phase']['learning_rate'],6e-6)
        self.assertEqual(new['value_phase']['learning_rate'],3e-6)
        self.assertEqual(new['value_phase']['kl_cap'],.002)
        self.assertEqual(new,value_phase.refined_options(new));validate_config(old);validate_config(new)
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh,new)
        self.assertEqual(read(SOURCE_ROOT/'configs/step5/joint7-latest.json')['value_phase'],value_phase.DEFAULT)
        for update in (dict(learning_rate=7e-6),dict(holdout=dict(holdout.DEFAULT,extra=True))):
            bad=copy.deepcopy(new);bad['value_phase'].update(update)
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(old);bad['value_phase']['holdout']=holdout.DEFAULT
        with self.assertRaises(ValueError):validate_config(bad)

    def test_exported_g110_and_prior_hotfix_config_are_reviewed(self):
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0110-20261019T233943412893Z/state.json')
        new=value_phase.refined_options(insolvency_event_config(state['signature']['config']))
        self.assertEqual(new,read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-g110-ppg.json'))
        signature=dict(state['signature'],config=new,implementation=implementation_id())
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        self.assertTrue(check_signature(state,signature,reviewed=True))
        previous=copy.deepcopy(state);previous['signature'].update(
            implementation='3a0c2da9e512b810ad4fa4ea3583ea16dd5e09252cfb651a2dc42da5a0d6c702',
            config=insolvency_event_config(state['signature']['config']))
        self.assertTrue(check_signature(previous,signature,reviewed=True))
        bad=copy.deepcopy(signature);bad['config']['value_phase']['kl_cap']=.003
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)

    def test_g110_value_adam_weights_rng_and_active_gate_state_preserved(self):
        model,opt,old,_=fixture();phase=TrainingPhases(model,old)
        p=next(model.parameters());p.grad=torch.ones_like(p);phase.optimizer.step();phase.optimizer.zero_grad(set_to_none=True)
        phase.state=gae.initial(old['advantage'],generation=110)
        phase.state.update(active_lambda=.98,value_iterations=50,ev_streak=4,healthy_streak=6,stage_origin=81,
            table_enabled=dict(historical=True,legacy=True),table_origins=dict(historical=70,legacy=78),
            table_streaks=dict(historical=10,legacy=10))
        evidence=ev_evidence.initial(108)
        for g in (109,110):evidence,_=ev_evidence.advance(evidence,old['advantage']['table_gate'],generation=g,metrics={},healthy=True,value_updated=True)
        phase.state['table_evidence']=evidence
        phase.critic_state.update(generation=110,snapshots_active=True,snapshot_weight=.5,learner_ev_baseline=.447,
            activation_generation=60)
        phase.critic_state['weight_recovery'].update(origin=85,healthy_streak=2)
        raw=dict(training={KEY:phase.payload()});saved=copy.deepcopy(raw);new=value_phase.refined_options(old)
        weights=copy.deepcopy(model.state_dict());ppo_state=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        resumed=TrainingPhases(model,new,raw=raw,generation=110)
        self.assertTrue(_same_payload(raw,saved));self.assertTrue(_same_payload(model.state_dict(),weights))
        self.assertTrue(_same_payload(opt.state_dict(),ppo_state));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        expected=copy.deepcopy(phase.optimizer.state_dict());expected['param_groups'][0]['lr']=3e-6
        self.assertTrue(expected['state']);self.assertTrue(_same_payload(resumed.optimizer.state_dict(),expected))
        for key in ('generation','active_lambda','value_iterations','stage_origin','table_enabled','table_origins'):
            self.assertEqual(resumed.state[key],phase.state[key])
        self.assertEqual(resumed.state['healthy_streak'],0);self.assertEqual(resumed.state['table_evidence'],ev_evidence.initial(110))
        self.assertEqual(resumed.critic_state['snapshot_weight'],.5);self.assertEqual(resumed.critic_state['learner_ev_baseline'],.447)
        self.assertEqual(resumed.critic_state['weight_recovery']['healthy_streak'],0)
        second=TrainingPhases(model,new,raw=dict(training={KEY:resumed.payload()}),generation=110)
        self.assertIsNone(second.refinement_migration)
        self.assertTrue(_same_payload(resumed.payload(),second.payload()))

    def test_new_split_excluded_from_ppo_critic_calibration_and_buffer(self):
        model,opt,cfg,behavior,data,snap,_=mixed_fixture();cfg=value_phase.refined_options(cfg)
        phase=TrainingPhases(model,cfg);phase.snapshot_data=snap;seen=[]
        real=updates.Dataset.value_batches
        def batches(ds,parts,*args,**kwargs):
            parts=list(parts);seen.extend(ds.data[i][2][0] for part in parts for i in part)
            yield from real(ds,parts,*args,**kwargs)
        with patch.object(updates.Dataset,'value_batches',batches):report=ppo(model,opt,cfg,behavior,data,phase)
        held=set(phase.split['selection']['heldout_games'])
        self.assertEqual(len(held),6);self.assertTrue(set(seen).isdisjoint(held))
        self.assertTrue(any(phase.dataset.data[i][2][0] in held for i in phase.actor_ids))
        self.assertEqual(report['meaningful_decisions'],sum(len(d[0]['choices'])>1 for d in data))
        ppo_before=copy.deepcopy(opt.state_dict());value=phase.fit(model)
        self.assertTrue(_same_payload(ppo_before,opt.state_dict()));self.assertEqual(value['learning_rate'],3e-6)
        self.assertLessEqual(value['heldout_after']['loss'],value['heldout_before']['loss'])
        self.assertLessEqual(value['probe_kl'],.002)

    def test_earlier_reviewed_EV_and_weight_migrations_compose_with_refinement(self):
        from test_snapshot_weight_recovery import active
        model,_,cfg,_=fixture();old_cfg=copy.deepcopy(cfg)
        old_cfg['advantage']['table_gate']=copy.deepcopy(gae.ARCHIVED_TABLE_GATE)
        old=TrainingPhases(model,old_cfg);old.state['generation']=85
        _,old.critic_state=active();saved=old.payload();saved['contract']='actor-value-phases/3.0.0'
        saved['critic_data'].pop('weight_recovery');before=copy.deepcopy(saved)
        new=TrainingPhases(model,value_phase.refined_options(cfg),raw=dict(training={KEY:saved}),generation=85)
        self.assertEqual(new.contract,'actor-value-phases/6.0.0')
        self.assertEqual(new.critic_state['snapshot_weight'],.5)
        self.assertEqual(new.state['table_evidence']['origin'],85)
        self.assertEqual(new.optimizer.param_groups[0]['lr'],3e-6)
        self.assertTrue(_same_payload(saved,before))

    def test_stratified_cache_rebuild_tamper_leakage_and_commit_cleanup(self):
        model,_,cfg,b,data,snap,_=mixed_fixture();cfg=value_phase.refined_options(cfg)
        cfg['value_phase']['buffer_iterations']=2;ds=updates.Dataset(data+snap,'cpu',policy=False)
        train,held,_=value_phase.split(ds.data,.05,cfg['value_phase']['holdout'])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp)/'run';buf=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=0)
            provenance=dict(source(b),snapshot_rows=True)
            with self.assertRaisesRegex(ValueError,'Held-out row'):buf.stage(ds,held,provenance,model)
            buf.stage(ds,train,provenance,model);receipt=copy.deepcopy(buf.pending)
            self.assertEqual(receipt['contract'],value_buffer.STRATIFIED_CONTRACT)
            write_json(run/'state.json',dict(iteration=1));buf.committed(1);before=sha(buf.path(1,'npy'))
            with buf.previous() as disk:self.assertEqual(len(disk),len(train))
            held_only=updates.Dataset([ds.data[i] for i in held],'cpu',policy=False)
            with self.assertRaisesRegex(ValueError,'overlap'):buf.stage(held_only,[],provenance,model)
            buf.path(1,'npy').unlink()
            with patch.object(value_buffer.DiskBuffer,'_rebuild_rows',lambda _:iter(ds.data[i] for i in train)):
                rebuilt=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=1,receipt=receipt)
            self.assertTrue(rebuilt.rebuilt);self.assertEqual(sha(rebuilt.path(1,'npy')),before)
            bad=copy.deepcopy(receipt);bad['holdout_selection']['heldout_games']=[]
            with self.assertRaisesRegex(ValueError,'selection differs'):
                value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=1,receipt=bad)
            bad=copy.deepcopy(receipt);bad['game_ids'].append(receipt['holdout_selection']['heldout_games'][0])
            with self.assertRaisesRegex(ValueError,'Held-out'):
                value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=1,receipt=bad)

    def test_real_rebuild_selection_and_role_order_from_admitted_ingress(self):
        model,_,cfg,b,data,snap,_=mixed_fixture();cfg=value_phase.refined_options(cfg)
        cfg['value_phase']['buffer_iterations']=2;ds=updates.Dataset(data+snap,'cpu',policy=False)
        train,held,_=value_phase.split(ds.data,.05,cfg['value_phase']['holdout'])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp)/'run';buf=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=0)
            provenance=dict(source(b),snapshot_rows=True,records=[dict(path='iteration-0000/shard.jsonl.gz')])
            buf.stage(ds,train,provenance,model);receipt=copy.deepcopy(buf.pending)
            write_json(run/'state.json',dict(iteration=1));buf.committed(1);before=sha(buf.path(1,'npy'))
            buf.path(1,'npy').unlink()
            with patch('brass_ai.recovery.ingress.load_collection',return_value=([],data+snap,{})) as ingress:
                resumed=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=1,receipt=receipt)
            self.assertTrue(resumed.rebuilt);self.assertEqual(ingress.call_count,2)
            self.assertEqual(sha(resumed.path(1,'npy')),before)
            with resumed.previous() as disk:
                self.assertEqual([disk[i][2] for i in range(len(disk))],[ds.data[i][2] for i in train])

    def test_g110_helper_refinement_and_limit_override_are_idempotent(self):
        from brass_ai.recovery import g110_hotfix as fix
        from test_selfplay110_review import hotfix_state
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            state=hotfix_state();checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'original')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            raw=dict(step=110,model_id=fix.MODEL,training=dict(il_kl_limit=state['il_kl_limit']))
            with patch('brass_ai.recovery.resume.load_resume',return_value=(None,raw,True)):
                fix.run(root,run,'configs/step5/ppg.json',apply=True,refine_ppg=True)
                saved=read(run/'state.json')
                repeat=fix.run(root,run,'configs/step5/ppg.json',apply=True,refine_ppg=True)
            self.assertTrue(repeat['already_applied']);self.assertEqual(saved,read(run/'state.json'))
            self.assertEqual(saved['il_kl_limit']['limit'],.2288818359375)
            self.assertEqual(read(root/'configs/step5/ppg.json')['value_phase']['learning_rate'],3e-6)
            self.assertEqual(checkpoint.read_bytes(),b'original')

    def test_selection_upgrade_excludes_old_cache_once_without_reward_change(self):
        model,_,old,(b,data)=fixture();old['value_phase']['buffer_iterations']=2;new=value_phase.refined_options(old)
        ds=updates.Dataset(data,'cpu');train,_,_=value_phase.split(data,.05)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp)/'run';buf=value_buffer.DiskBuffer(SOURCE_ROOT,run,old,generation=0)
            buf.stage(ds,train,source(b),model);receipt=copy.deepcopy(buf.pending)
            write_json(run/'state.json',dict(iteration=1));buf.committed(1)
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                value_buffer.DiskBuffer(SOURCE_ROOT,run,new,generation=1,receipt=receipt)
            resumed=value_buffer.DiskBuffer(SOURCE_ROOT,run,new,generation=1,receipt=receipt,previous_value=old['value_phase'])
            self.assertIsNone(resumed.receipt);self.assertIsNone(resumed.reward_transition)
            self.assertEqual(resumed.holdout_transition['excluded_previous_rows'],len(train));self.assertTrue(buf.path(1,'npy').exists())
            b,data=examples(model,prefix='new-stratified-');ds=updates.Dataset(data,'cpu')
            train,_,_=value_phase.split(data,.05,new['value_phase']['holdout'])
            report=resumed.stage(ds,train,source(b),model)
            self.assertEqual(report['previous_rows'],0);self.assertEqual(report['excluded_cache_bytes'],receipt['bytes'])
            write_json(run/'state.json',dict(iteration=2));resumed.committed(2)
            self.assertFalse(buf.path(1,'npy').exists());self.assertIsNone(resumed.holdout_transition)
            restored=value_buffer.DiskBuffer(SOURCE_ROOT,run,new,generation=2,receipt=resumed.receipt)
            with restored.previous() as disk:self.assertEqual(len(disk),len(train))


if __name__=='__main__':unittest.main()
