"""Fresh g1290 weights, neutral collection, bounded imports and PPO/PPG isolation."""
import copy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.features import Encoded
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery import fresh_selfplay as F,neutral_training as N,exploration as E,branching as B
from brass_ai.recovery import quality_pool as Q,pfsp_pool as P,il_kl_probe as probes,il_kl_limit
from brass_ai.recovery import critic_data,updates
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.utility import reward_spec
from brass_ai.recovery.retention import _same_payload,update_population
from brass_ai.step5.league import write_json,implementation_id

REPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1360-20261101T011723464975Z/state.json'


def recipe():return copy.deepcopy(read(REPORT)['signature']['config'])


def source_fixture(root):
    from test_diagnostics_cleanup import probe_fixture
    c=recipe();c['model_size']='small';source=root/'artifacts/source';source.mkdir(parents=True)
    il=FourSeatNet(ModelConfig.preset('small'))
    path=source/'population'/('weights-'+model_id(il)+'.pt');save_checkpoint(path,il)
    anchor=dict(path=path.relative_to(root).as_posix(),sha256=sha(path),model_id=model_id(il),generation=0,weights_only=True)
    val=root/'validation.jsonl';val.write_bytes(b'authored probe source')
    manifest=root/'manifest.json';write_json(manifest,dict(entries=[dict(path='validation.jsonl',sha256=sha(val),split='validation')]))
    origin=dict(bc_run='artifacts/il',path=anchor['path'],sha256=anchor['sha256'],model_id=anchor['model_id'],
        manifest='manifest.json',manifest_sha256=sha(manifest),bc_signature='b'*64)
    c['imitation_checkpoint']=copy.deepcopy(origin);c['rehearsal_manifest']='manifest.json'
    probe=probe_fixture(8);probe.config=c['il_kl_limit']
    probe.metadata.update(format=probes.FORMAT,source=probes.source_binding(root,origin),config=c['il_kl_limit'])
    probe.arrays['reference']=probe.probabilities(il,c['update'])
    probe.metadata['probe_hash']=probes.semantic_hash(probe.metadata,probe.arrays)
    directory=source/'il-kl';directory.mkdir();cache=directory/'probe.npz'
    with cache.open('wb') as f:np.savez(f,**probe.arrays)
    probe.metadata['cache']=dict(path=cache.relative_to(root).as_posix(),sha256=sha(cache))
    write_json(directory/'manifest.json',probe.metadata)
    receipt=dict(path=(directory/'manifest.json').relative_to(root).as_posix(),sha256=sha(directory/'manifest.json'),probe_hash=probe.metadata['probe_hash'])
    pool=Q.initial(anchor,c['opponent_pool'],dict(probe.metadata['cache'],probe_hash=probe.metadata['probe_hash']))
    model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(c)))
    seed=None
    for generation in (855,1290,1330,1335):
        with torch.no_grad():next(model.parameters()).add_(.001)
        path=source/'population'/('weights-'+model_id(model)+'.pt')
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility))
        save_checkpoint(path,model,step=generation,training=dict(mode='step5-population-ppo',fresh_run='artifacts/source'))
        e=dict(path=path.relative_to(root).as_posix(),sha256=sha(path),model_id=model_id(model),generation=generation,weights_only=True)
        if generation==1290:seed=e;continue
        values=probe.probabilities(model,c['update'])
        e.update(probe=Q.cache_candidate(root,source,e,values,probe),admitted_iteration=generation,
            admission_health=dict(ok=True),**P.prior(c['opponent_pool']))
        e.update(S=7.,C=30.,alarm_streak=2)
        pool['members'].append(e)
    pool.update(generation=1360,games_done=435200);P.seal(pool)
    signature=dict(config=c,compatibility=compatibility(),implementation=implementation_id(),imitation_source=origin)
    last=source/'generation-1360.pt'
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'])
    next(model.parameters()).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    save_checkpoint(last,model,opt,step=1360,training=dict(signature=digest(signature),fresh_run='artifacts/source',
        quality_pool_digest=Q.state_digest(pool),il_kl_limit=dict(probe=receipt),domino=dict(never_import=True)))
    latest=dict(path=last.relative_to(root).as_posix(),sha256=sha(last),model_id=model_id(model),generation=1360)
    # The original g1290 full checkpoint has been retired; only its pinned slim copy remains.
    retired=dict(seed,path='artifacts/source/generation-1290.pt',sha256='a'*64);retired.pop('weights_only')
    state=dict(format='step5-recovery-league/2.2.0',signature=signature,iteration=1360,games_done=435200,next_seed=19000000,
        history=[anchor,retired,latest],population=[Q.reference(m) for m in pool['members']],quality_pool=pool,
        policy_anchor=dict(active=dict(checkpoint=seed)),il_kl_limit=dict(probe=receipt),evaluation_anchor=anchor)
    write_json(source/'state.json',state)
    return source,state,il,origin,seed


class FreshNeutral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_recipe_defaults_legacy_unchanged_and_cutoff_inventory(self):
        old=recipe();before=copy.deepcopy(old);c=validate_config(N.configure(old))
        self.assertEqual(old,before);self.assertEqual(N.configure(c),c)
        self.assertEqual(c['update']['fixed_entropy'],E.FIXED_ENTROPY_005)
        self.assertEqual(c['exploration'],.05);self.assertEqual(c['insolvency'],dict(event_cost=.02,max_cost=.075))
        self.assertEqual(c['opening'],E.THREE_OPENING);self.assertEqual(c['branching'],E.NEUTRAL_BRANCH)
        self.assertEqual(c['value_phase'],old['value_phase']);self.assertEqual(c['ppo_lr_controller'],old['ppo_lr_controller'])
        self.assertEqual(c['diversity']['mode'],'off');self.assertIsNone(c['diversity']['adoption'])
        self.assertEqual(c['context_evaluation']['mode'],'off');self.assertIsNone(c['local_scoring_auxiliary']['adoption'])
        self.assertEqual(validate_config(old),old)
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh['update']['fixed_entropy'],E.FIXED_ENTROPY_005);self.assertTrue(N.enabled(fresh))
        self.assertEqual(latest_model_config(fresh,fresh['imitation_checkpoint']),fresh)
        from brass_ai.recovery.teacher_phase_config import defaults as teacher_defaults
        diverse_teacher=copy.deepcopy(old);diverse_teacher['teacher_phase']=teacher_defaults(reward_spec(old),diversity=True)
        rolled=validate_config(N.configure(diverse_teacher))
        self.assertNotIn('diversity',rolled['teacher_phase']);self.assertEqual(rolled['teacher_phase']['mode'],'off')
        entry,members=F.selected(read(REPORT),1290,1330)
        self.assertEqual(entry['generation'],1290);self.assertEqual(len(members),19)
        self.assertEqual(max(m['generation'] for m in members),1330)
        for mutate in (lambda x:x['branching']['move_distribution'].update(temperature_1=.7),
            lambda x:x['opening'].update(temperature=2.),lambda x:x['diversity'].update(mode='apply'),
            lambda x:x['update']['fixed_entropy']['weights'].update(canal_action=.1)):
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):validate_config(bad)

    def test_branch_components_marginal_fallback_and_postfork_three_moves(self):
        raw=np.array([.6,.3,.1]);q=np.array([.1,.2,.7]);contract=E.NEUTRAL_BRANCH['contract']
        choices=[]
        for vector in (raw,np.sqrt(raw),q):
            v=vector.copy();v[0]=0;choices.append(v/v.sum())
        mixed=.6*choices[0]+.3*choices[1]+.1*choices[2]
        rng=Mock();rng.choices.side_effect=[[1],[2]]
        r=B.alternative(raw,q,0,rng,contract=contract)
        self.assertEqual(rng.choices.call_args_list[0].kwargs['weights'],[.6,.3,.1])
        self.assertAlmostEqual(r['marginal_probability'],mixed[2]);self.assertEqual(r['component'],'temperature_2')
        for seed in range(20):
            self.assertEqual(B.alternative(raw,q,0,random.Random(seed),contract=contract),B.alternative(raw,q,0,random.Random(seed),contract=contract))
        r=B.alternative(raw,np.array([1.,0.,0.]),0,random.Random(7),contract=contract)
        self.assertNotEqual(r['component'],'exploration_q');self.assertAlmostEqual(r['component_probability'],2/3 if r['component']=='temperature_1' else 1/3)
        from test_training_g1120 import Revision
        with patch.object(E,'THREE_BRANCH',E.NEUTRAL_BRANCH):Revision().test_fork_skips_three_and_resets_all_four_seats()

    def test_neutral_rows_reject_focused_replay_and_parent_branches(self):
        proof=dict(seat_contexts=N.CONTEXTS)
        row=dict(behavior=dict(kind='model'),style=[0.]*4,intent=[0.]*8)
        N.check_rows(dict(records=[row]),proof)
        wrong=copy.deepcopy(row);wrong['intent'][0]=1
        with self.assertRaisesRegex(ValueError,'Non-neutral'):N.check_rows(dict(records=[wrong]),proof)
        with self.assertRaisesRegex(ValueError,'DOMiNO'):N.check_rows(dict(records=[row]),dict(proof,diversity_assignment={}))

    def test_native_admission_binds_neutral_contract_and_rejects_focused_rows(self):
        from test_b03 import family_fixture
        from brass_ai.rust_backend import lab_admission as A
        from brass_ai.rust_backend.lab_collection import frozen_id
        f,_,c=family_fixture();c['seat_contexts']=copy.deepcopy(N.CONTEXTS)
        job=f['frozen_job'];job['provenance']['seat_contexts']=copy.deepcopy(N.CONTEXTS)
        f['job_id']=frozen_id(job);f['parent']['native_receipt']['job']=f['job_id']
        f['sha256']=digest({k:v for k,v in f.items() if k!='sha256'})
        a=A.binding(job,f['producer'],generation=10,model_id='a'*64,config=c)
        game=A.convert(f,a);N.check_rows(game,job['provenance'])
        with self.assertRaisesRegex(ValueError,'Seat-context'):A.binding(job,f['producer'],generation=10,model_id='a'*64,config={k:v for k,v in c.items() if k!='seat_contexts'})
        bad=copy.deepcopy(f);bad['parent']['game']['records'][0]['intent'][0]=1.
        bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaises(ValueError):A.convert(bad,a)

    def test_seed_bundle_fresh_start_resume_retention_and_source_unchanged(self):
        from brass_ai.recovery import league,legacy_search,ratings
        from brass_ai.recovery.learning_rate import KEY as WARMUP
        from brass_ai.recovery.lr_controller import KEY as LR
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);source,state,il,origin,seed=source_fixture(root)
            before={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
            run=root/'artifacts/neutral';output='configs/step5/neutral.json'
            result=F.prepare(root,source,run,output)
            self.assertEqual(result['imported_generations'],[0,855,1330]);self.assertEqual(result['excluded_generations'],[1335])
            self.assertEqual(F.prepare(root,source,run,output),result)
            c=read(root/output);m=F.load_manifest(root,run,c)
            rng=torch.get_rng_state().clone();model,receipt=F.seed_model(root,run,c,m,'cpu')
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            old,_=load_checkpoint(root/seed['path'])
            self.assertTrue(_same_payload(old.state_dict(),model.state_dict()));self.assertIsNone(model.utility_calibration)
            self.assertEqual(model.config.utility,reward_spec(c))
            probe,_=probes.load(root,run,origin,il,c['il_kl_limit'],c['update'],1,receipt=m['probe'])
            pool,anchor=F.initialize_pool(root,run,c,m,probe)
            self.assertEqual([e['generation'] for e in pool['members']],[0,-476,-1])
            self.assertEqual(pool['games_done'],0);self.assertEqual(pool['generation'],0)
            self.assertTrue(all((e['S'],e['C'],e['alarm_streak'])==(5.,10.,0) for e in pool['members']))
            self.assertEqual(anchor['model_id'],origin['model_id']);self.assertNotEqual(anchor['model_id'],model_id(model))
            self.check_planner(root,run,c,pool,model_id(model),anchor)
            write_json(root/'training/opponents/pool.json',dict(training=[f'bot{i}' for i in range(5)]))
            class Initialized(Exception):pass
            with (patch.object(league,'admission'),patch.object(legacy_search,'legacy_spec',side_effect=lambda root,p,*a,**k:dict(kind='legacy',policy_id=p,search_free=True)),
                  patch.object(ratings,'check_dependencies'),patch('brass_ai.recovery.demonstrations.completed_imitation',return_value=(il,origin)),
                  patch('brass_ai.recovery.actor_value.TrainingPhases',side_effect=Initialized),
                  patch('brass_ai.recovery.domino_auxiliary.Diversity',side_effect=AssertionError('DOMiNO constructed'))):
                for fresh in (True,False):
                    with self.assertRaises(Initialized):league.train(root,run,c,iterations=1,imitation_run='artifacts/il',fresh=fresh)
            saved=read(run/'state.json');raw=torch.load(run/'initial.pt',weights_only=True)
            self.assertEqual((saved['iteration'],saved['games_done'],raw['step']),(0,0,0));self.assertEqual(raw['optimizer']['state'],{})
            self.assertEqual(raw['optimizer']['param_groups'][0][WARMUP]['completed_steps'],0)
            self.assertEqual(raw['optimizer']['param_groups'][0][LR]['multiplier'],1.)
            self.assertEqual(raw['training']['parent'],seed['model_id']);self.assertNotIn('domino',raw['training'])
            self.assertNotIn('actor_value_phases',raw['training']);self.assertNotIn('policy_anchor',saved)
            self.assertEqual(saved['evaluation_anchor']['model_id'],origin['model_id'])
            from brass_ai.recovery.plackett_luce import IL
            self.assertEqual(saved['ratings']['agents'][IL]['checkpoint']['model_id'],origin['model_id'])
            self.assertTrue(_same_payload(raw['state_dict'],dict(old.state_dict())))
            update_population(root,run,saved,c['population'])
            self.assertEqual(len(saved['population']),3)
            self.assertEqual(before,{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()})
            from brass_ai.recovery.retention import references
            member=saved['quality_pool']['members'][1];path=root/member['path']
            self.assertIn(path,references(root,run))
            retired=copy.deepcopy(saved)
            retired['quality_pool']['members']=[e for e in retired['quality_pool']['members'] if e['model_id']!=member['model_id']]
            retired['population']=[e for e in retired['population'] if e['model_id']!=member['model_id']]
            write_json(run/'state.json',retired)
            pins=references(root,run)
            self.assertNotIn(path,pins)  # Immutable seed inventory must not pin retired imports forever.
            self.assertNotIn(root/member['probe']['path'],pins)
            self.assertIn(root/m['seed_checkpoint']['path'],pins)
            write_json(run/'state.json',saved)
            with self.assertRaisesRegex(ValueError,'progress'):F.prepare(root,source,run,output)
            with self.assertRaisesRegex(ValueError,'escapes owned run|another run'):F.load_manifest(root,root/'artifacts/other',c)
            bad=copy.deepcopy(pool);bad['members'][1]['imported_from']['source_generation']=1335
            with self.assertRaisesRegex(ValueError,'imported PFSP'):Q.validate_state(bad,c['opponent_pool'])
            print('Fresh-seed initialization smoke:',dict(seed_generation=1290,initial_generation=0,imported=[0,855,1330],
                fresh_adam=True,IL_preserved=True,restart_verified=True,source_unchanged=True),flush=True)

    def check_planner(self,root,run,c,pool,current,anchor):
        from brass_ai.recovery.ordinary import plan_collection
        from brass_ai.recovery.execution import neural_spec
        from brass_ai.recovery.replay import CheckedGame
        from brass_ai.recovery.panel_gates import initial
        state=dict(iteration=0,games_done=0,next_seed=c['seed'],context_counter=0,quality_pool=pool,
            population=[Q.reference(m) for m in pool['members']],history=[dict(model_id=current,sha256='c'*64,generation=0)],
            evaluation_anchor=anchor,ratings=dict(panel_gates=initial(0)))
        service=Mock();service.models={m['model_id']:None for m in pool['members']}
        bots={f'bot{i}':dict(kind='legacy',policy_id=f'bot{i}',search_free=True) for i in range(5)}
        # Test the complete planner without depending on a compiled native host.
        config=copy.deepcopy(c);config['local_scoring_auxiliary']['mode']='off'
        before=copy.deepcopy(state);settings=E.effective(config,E.initial(config,0),0)
        plans=[plan_collection(root,run,state,config,dict(config=config),service,current,list(bots),bots,bots,None,exploration=settings) for _ in range(2)]
        self.assertEqual(plans[0],plans[1]);self.assertEqual(state,before)
        p=plans[0];self.assertEqual(p['composition'],dict(selfplay=160,historical=144,legacy=16));self.assertEqual(p['context_counter'],0)
        learner=neural_spec(current,.05,'mixed',version=3,opening=E.THREE_OPENING)
        for j in p['jobs']:
            self.assertNotIn('diversity_assignment',j['provenance']);self.assertEqual(set(j['focuses'].values()),{'neutral'})
            self.assertEqual(len(j['focuses']),2 if j['provenance']['kind']=='legacy' else 4)
            ps=j['policies'];rotation=j['rotation'];table=j['provenance']['critic_table']
            game=CheckedGame(opponents=[ps[(s-rotation)%4]['policy_id'] for s in range(4)],records=[])
            critic_data.check_table(table,game,learner['policy_id'])
            self.assertTrue(np.all(np.asarray(table['stable'])[:,6:]==0))

    def test_isolated_neutral_ppo_ppg_smoke(self):
        from test_training_g1324 import smoke_config
        from test_gae_value_phase import examples
        from test_critic_descriptors import partition
        from test_opening_ppo import opening_rows
        from test_value_buffer import source
        model,opt,c=smoke_config();c=N.configure(c);model.config=replace(model.config,utility=reward_spec(c))
        c['update'].update(epochs=1,effective_batch_size=4096,critic_batch_size=512,microbatch_size=64,preparation_workers=0)
        c['value_phase'].update(batch_size=1024,clone_rows=64,clone_batch_size=32)
        _,rows=examples(model,games=42,decisions=4,prefix='neutral-smoke-')
        learner,snapshot,_=partition(rows,'neutral-smoke-')
        for data in (learner,snapshot):
            for i,(r,t,identity,e) in enumerate(data):
                r.update(style=[0.]*4,intent=[0.]*8)
                state=e.state.copy();state[-12:]=0
                data[i]=(r,t,identity,Encoded(state,e.actions,e.stable_to_relative,e.choice_ids))
        b,learner=opening_rows(model,learner,.05,opening=c['opening'],objective=c['update']['opening_objective'])
        b.entropy_alphas=copy.deepcopy(E.FIXED_ENTROPY_005['weights'])
        with (tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp,
              patch.object(probes.Probe,'measure',side_effect=AssertionError('AnchorKL measured')),
              patch('brass_ai.recovery.domino_auxiliary.Diversity',side_effect=AssertionError('DOMiNO constructed'))):
            p=TrainingPhases(model,c,root=SOURCE_ROOT,run=Path(tmp));p.diversity=None
            p.snapshot_data=snapshot;p.collection_source=dict(source(b),snapshot_rows=True)
            result=updates.ppo_update(model,opt,[dict(split='train')],b,c['update'],learner_data=learner,
                insolvency=c['insolvency'],reward=reward_spec(c),phases=p)
            value=p.fit(model)
            self.assertEqual(result['optimizer_steps'],1);self.assertGreater(value['optimizer_steps'],0)
            self.assertIsNone(p.diversity);self.assertEqual(c['update']['fixed_entropy']['weights'],dict.fromkeys(E.GROUPS,.005))
            print('Neutral PPO/PPG smoke:',dict(ppo_steps=result['optimizer_steps'],ppg_steps=value['optimizer_steps'],
                ppo_kl=result['categorical_probe']['categorical_kl'],ppg_kl=value['probe_kl'],authored_games=42),flush=True)


if __name__=='__main__':unittest.main()
