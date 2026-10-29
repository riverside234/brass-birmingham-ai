"""Short authored B03 admission/continuation fixtures; no games or training run."""
import copy
import gzip
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, digest
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.batching import model_id
from brass_ai.learning.replay import split_for
from brass_ai.recovery import critic_data, gae
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.execution import neural_spec
from brass_ai.recovery.retention import _same_payload, run_lock
from brass_ai.recovery.replay import check_game, write_shard, read_shards
from brass_ai.recovery.updates import prepared
from brass_ai.step5.league import write_json, implementation_id, seeds_for
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.rust_backend import lab_admission as A, lab_continuation as C
from brass_ai.rust_backend.lab_contract import SOURCE
from brass_ai.rust_backend.lab_collection import frozen_id
from brass_ai.rust_backend.lab_families import FORMAT
from fixtures import position, result
from test_b025 import current_board, config, model_fixture

PREVIOUS = '6abba7672db47057df6090907c9397eb70ecad52b691653ed552bf250725e96d'


def scoring(game, native=True):
    from brass_ai.learning.board_topology import topology
    board = topology(); vp = [s['vp'] for s in game['result']['seats']]; eras = []
    for era in ('CanalEra', 'RailEra'):
        if not any(r['view']['context']['era'] == era for r in game['records']): continue
        eras.append(dict(era=era, before=vp,
            slots=[dict(id=n, owner=0, vp=0, valid=True, instance=None) for n in board['slot_order']],
            links=[dict(id=n, owner=0, vp=0, valid=board['link_eras'][n] == era.replace('Era', '')) for n in board['link_order']],
            audit=dict(industry=[0]*4, network=[0]*4, bonus=[0]*4, bonus_since_recording=[0]*4,
                       outside_adjustment=[0]*4, outside_scoring=[0]*4, after=vp)))
    return dict(contract='local-scoring-targets/1.0.0', rules='steam-brass-patch-1.1.0',
        producer='rust-local-scoring/1.0.0' if native else 'local-scoring-host/1.0.0',
        start_revision=game['records'][0]['view']['context']['revision'], initial_vp=vp, final_vp=vp, eras=eras)


def family_fixture(*, kind='selfplay', seed=0, branch=False, generation=10):
    c = config(); c['local_scoring_auxiliary']['adoption'] = None
    # Same weights, different policy identity in the snapshot seat.
    learner = neural_spec('a'*64, c['exploration'], 'neutral', version=3, opening=c['opening'])
    policies = [copy.deepcopy(learner) for _ in range(4)]
    if kind != 'selfplay':
        policies[2] = neural_spec('a'*64, c['exploration'], 'Brewery', version=3, opening=c['opening'])
        policies[3] = neural_spec('b'*64, c['exploration'], 'neutral', version=3, opening=c['opening'])
    if kind=='legacy':policies[3]=dict(kind='legacy',policy_id='fixture-v4',search_free=True)
    seed = seeds_for(seed, 1)[0][0]
    group = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed))
    producer = dict(build_sha256='b'*64, b01_sha256='c'*64, signature='d'*64, profile='search_v1', v205_enabled=False)
    fp = dict(values=[0.]*28)
    members = [dict(model_id='a'*64, generation=5, fingerprint=fp), dict(model_id='b'*64, generation=0, fingerprint=fp)]
    table = critic_data.planned_table(policies, 0, 'a'*64, generation, members, fp, fp, learner_policy_id=learner['policy_id'])
    from brass_ai.recovery.local_scoring_targets import identity
    proof = dict(purpose='training', kind=kind, spec=producer['signature'],
        learner_behavior=dict(model_id='a'*64, policy_id=learner['policy_id']), critic_table=table,
        ppo_opening_objective=c['update']['opening_objective'],
        local_scoring=dict(identity(), producer=dict(runtime_sha256='e'*64, host_sha256='f'*64)))
    job = dict(index=seed, seed=seed, sampling_seed=57, stable_game_id='fixture-'+str(seed),
               policies=policies, provenance=proof)
    if branch:
        job['branching'] = copy.deepcopy(c['branching']); proof['exploration'] = dict(branching=job['branching'])
    samplers = [NeuralPolicy(None, evaluator=SimpleNamespace(model_id=p['model_id']), focus=p['focus'],
        exploration=p['exploration'], exploration_version=3, opening=p['opening']) if p['kind']=='neural' else None for p in policies]

    def rows(start, count, rngs, samplers):
        values=[]; evidence=[]
        for index in range(count):
            seat=index%4; p=samplers[seat]; rng=rngs[seat]
            view, choices=position(seat); current_board(view)
            view['context'].update(revision=start+index, round=3, era='CanalEra' if index<count//2 else 'RailEra')
            for player in view['players']:player['hidden_discards']=int(view['context']['era']=='CanalEra')
            if index==1:
                view['context']['phase']='continuation';view['pending'].update(kind='develop',parts=1)
                choices=[dict(id='finish',kind='finish')]
            raw=np.array([1.] if len(choices)==1 else [.6,.25,.15])
            if p is None:
                spec=policies[seat]
                values.append(dict(step=index,seat=seat,view=view,choices=choices,chosen=0,style=[0.]*4,intent=[0.]*8,
                    teacher=None,seconds=0.,behavior=dict(kind='external',policy_id=spec['policy_id'],probability=1.,temperature=0.)))
                evidence.append(dict(kind='ordinary',model_id=None,policy_id=spec['policy_id'],raw_probabilities=None,semantic_id=choices[0]['id']))
                continue
            style,intent=p.context_features(view,rng)
            chosen,probability,_=p.sample_training(view,choices,rng,raw,intent)
            row=dict(step=index,seat=seat,view=view,choices=choices,chosen=chosen,style=style,intent=intent,
                teacher=None,seconds=0.,behavior=dict(kind='model',policy_id=p.policy_id,probability=probability,**p.last_behavior[rng]))
            values.append(row);evidence.append(dict(kind='ordinary',model_id=policies[seat]['model_id'],policy_id=p.policy_id,
                raw_probabilities=raw.tolist(),semantic_id=choices[chosen]['id']))
        return values,evidence

    rngs=[random.Random(digest(dict(sampling_seed=57,game=0,seat=s))) for s in range(4)]
    r,e=rows(0,24 if branch else 8,rngs,samplers)
    game=dict(game_id=job['stable_game_id'],setup_group=group,players=4,split=split_for(group),
        opponents=[p['policy_id'] for p in policies],result=result(),records=r)

    def part(game,evidence):
        if kind=='legacy':
            from brass_ai.recovery.local_scoring_targets import bind,FIELD
            game=copy.deepcopy(game);game[FIELD]=bind(scoring(game,False),game,proof['local_scoring']['producer'])
            return dict(game=game,evidence=evidence,scoring=None,branch_expected=False,timings={},
                        producer=dict(contract='brass-reference-collection/1.0.0',runtime=dict(sha256='e'*64)))
        score=scoring(game)
        return dict(game=game,evidence=evidence,scoring=score,branch_expected=False,diagnostic_branch_probe=False,
            origin_counts=[0]*4,timings={},producer=dict(contract='brass-lab-collection/1.0.0',core=dict(source_sha256=SOURCE)),
            native_receipt=dict(contract='brass-lab-collection/1.0.0',job=frozen_id(job),rows=len(game['records']),result=game['result'],scoring=score))
    parent=part(game,e); child=None
    if branch:
        from brass_ai.recovery.branching import alternative, seed_for
        from brass_ai.step5.policy import exploration_distribution
        index=16; row=copy.deepcopy(r[index]); contract=job['branching']['contract']; child_seed=seed_for(game['game_id'],index,contract)
        raw=np.asarray(e[index]['raw_probabilities']);q=exploration_distribution(row['choices'],row['intent'],3,'action')
        meta=dict(contract=contract,parent_game_id=game['game_id'],decision=index,played=row['chosen'],seed=child_seed,
                  prior_events=[0]*4,raw_probabilities=raw.tolist(),**alternative(raw,q,row['chosen'],random.Random(child_seed),contract=contract))
        row.update(step=0,chosen=meta['chosen']);row['behavior'].update(component='branch_move',probability=meta['marginal_probability'])
        samplers=[NeuralPolicy(None,evaluator=SimpleNamespace(model_id=p['model_id']),focus=p['focus'],exploration=p['exploration'],
                    exploration_version=3,opening=p['opening']) for p in policies]
        cr,ce=rows(index+1,8,[random.Random(digest(dict(branch=child_seed,seat=s))) for s in range(4)],samplers)
        for j,value in enumerate(cr):value['step']=j+1
        # Continue in the actual fork era; receipt/targets cannot inherit Canal maps.
        for value in cr:
            value['view']['context']['era']=row['view']['context']['era']
            for player in value['view']['players']:player['hidden_discards']=int(row['view']['context']['era']=='CanalEra')
        child_game=dict(game,game_id=game['game_id']+'/branch/'+child_seed,records=[row,*cr],branch=meta)
        child=part(child_game,[dict(kind='excluded-fork',metadata=copy.deepcopy(meta)),*ce])
        parent.update(branch_expected=True,diagnostic_branch_probe=True)
    family=dict(format=FORMAT,job_id=frozen_id(job),frozen_job=job,producer=producer,parent=parent,child=child,training_admitted=False)
    family['sha256']=digest(family)
    admission=A.binding(job,producer,generation=generation,model_id='a'*64,config=c,scope='diagnostic' if branch else 'ordinary')
    return family,admission,c


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def temporary(self):
        temp=tempfile.TemporaryDirectory(prefix='b03-',dir=SOURCE_ROOT/'artifacts');self.addCleanup(temp.cleanup)
        return Path(temp.name)


class Admission(Base):
    def test_native_reference_targets_features_and_head_metadata_match(self):
        f,a,c=family_fixture(kind='historical');g=A.convert(f,a)
        self.assertEqual([x['role'] for x in g.critic_table['seats']],['learner','learner','snapshot','il'])
        from brass_ai.recovery.local_scoring_targets import bind
        legacy=copy.deepcopy(dict(g));legacy['local_scoring_targets']=bind(scoring(legacy,False),legacy,dict(runtime_sha256='e'*64,host_sha256='f'*64))
        old=check_game(legacy);old.critic_table=g.critic_table;old.table_kind=g.table_kind;old.ppo_opening_objective=g.ppo_opening_objective
        from brass_ai.recovery.utility import reward_spec
        selected=lambda r:r['seat'] in (0,1,2)
        x=prepared([g],selected,insolvency=c['insolvency'],reward=reward_spec(c))
        y=prepared([old],selected,insolvency=c['insolvency'],reward=reward_spec(c))
        self.assertEqual(len(x),6)
        for (r,t,i,e),(s,u,j,v) in zip(x,y):
            self.assertEqual((t,i),(u,j))
            np.testing.assert_equal({k:v for k,v in r.items() if k.startswith('_') and k!='_local_scoring'},
                                    {k:v for k,v in s.items() if k.startswith('_') and k!='_local_scoring'})
            for key in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(e,key),getattr(v,key))
        self.assertTrue(any(len(e.choice_ids)==1 for _,_,_,e in x))
        self.assertEqual(g['local_scoring_targets']['producer']['contract'],A.SCORER)

    def test_publication_receipt_retry_cache_rebuild_and_ingress(self):
        f,a,c=family_fixture();p=self.temporary()/'replay.gz';r=A.publish(p,f,a)
        self.assertEqual(r,A.publish(p,f,a))
        games=read_shards([p],partition='train',native_admissions={str(p):r['native_admission']})
        self.assertEqual(len(games),1)
        from brass_ai.recovery.ingress import load_collection
        from brass_ai.recovery.utility import reward_spec
        _,rows,_=load_collection([r],a['job']['provenance']['learner_behavior']['policy_id'],{games[0]['setup_group']:'selfplay'},
                                insolvency=c['insolvency'],reward=reward_spec(c))
        self.assertEqual(len(rows),8);self.assertTrue(all('_local_scoring' in x[0] for x in rows))
        with self.assertRaises(ValueError):read_shards([p])
        with self.assertRaises(ValueError):read_shards([p],heuristic_only=True,native_admissions={str(p):r['native_admission']})
        changed=copy.deepcopy(f);changed['parent']['game']['records'][0]['seconds']=1.;changed['sha256']=digest({k:v for k,v in changed.items() if k!='sha256'})
        with self.assertRaises(ValueError):A.publish(p,changed,a)

    def test_stale_foreign_teacher_and_method_tampering_rejected(self):
        f,a,c=family_fixture()
        mutations=[lambda x:x['parent']['game']['records'][0]['behavior'].update(probability=.001),
            lambda x:x['parent']['evidence'][0].update(model_id='foreign'),
            lambda x:x['parent']['game']['records'][0].update(teacher=dict(source='search-v1-improved-policy/1.0.0')),
            lambda x:x['parent']['producer']['core'].update(source_sha256='f'*64),
            lambda x:x['frozen_job'].update(seed=999),lambda x:x.update(training_admitted=True),
            lambda x:x['parent'].update(branch_expected=True)]
        for mutate in mutations:
            x=copy.deepcopy(f);mutate(x);x['sha256']=digest({k:v for k,v in x.items() if k!='sha256'})
            with self.subTest(mutate=mutate),self.assertRaises((ValueError,KeyError)):A.convert(x,a)
        for method in ('search_v2','search_v2_no_switch','search_v2_focus_only',None):
            producer=dict(a['producer'],profile=method)
            with self.assertRaises(ValueError):A.binding(a['job'],producer,generation=10,model_id='a'*64,config=c)
        with self.assertRaises(ValueError):A.binding(a['job'],dict(a['producer'],v205_enabled=True),generation=10,model_id='a'*64,config=c)

    def test_diagnostic_branch_keeps_family_holdout_and_excludes_fork_move(self):
        f,a,c=family_fixture(branch=True)
        with self.assertRaisesRegex(ValueError,'Diagnostic'):A.convert(f,a)
        g=A.convert(f,a,allow_diagnostic=True)
        from brass_ai.recovery.branching import expand
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.recovery.critic_holdout import select
        games=expand([g]);data=prepared(games,insolvency=c['insolvency'],reward=reward_spec(c))
        from brass_ai.rust_backend.lab_b03_smoke import compare_preparation
        compared=compare_preparation(g,f,c,a['job']['provenance']['learner_behavior']['policy_id'])
        self.assertEqual(len(compared),len(data))
        self.assertEqual(len(data),len(g['records'])+len(g['branches'][0]['records'])-1)
        self.assertFalse(any(r['behavior'].get('component')=='branch_move' for r,_,_,_ in data))
        selection=select({x['game_id']:'selfplay' for x in games},.05,c['value_phase']['holdout'])
        self.assertEqual(g['game_id'] in selection['heldout_games'],g['branches'][0]['game_id'] in selection['heldout_games'])
        self.assertEqual(g['branches'][0]['local_scoring_targets']['receipt']['eras'][0]['era'],'RailEra')
        fake=copy.deepcopy(a);fake['scope']='ordinary';fake['sha256']=digest({k:v for k,v in fake.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'Diagnostic branch'):A.convert(f,fake)

    def test_native_cannot_be_relabelled_legacy_or_mutate_checked_rows(self):
        f,a,c=family_fixture();g=A.convert(f,a);out=self.temporary()
        with self.assertRaisesRegex(ValueError,'separate native'):write_shard(out/'bad.gz',[g],dict(purpose='training'))
        with gzip.open(out/'foreign.gz','wt') as stream:
            stream.write(json.dumps(dict(format='four-seat-replay/1.0.0',compatibility=compatibility(),provenance={}))+'\n')
            stream.write(json.dumps(g)+'\n')
        with self.assertRaisesRegex(ValueError,'relabeled'):read_shards([out/'foreign.gz'])
        g['records'][0]['behavior']['probability']=.2
        with self.assertRaises(ValueError):prepared([g])

    def test_duplicate_shards_and_incomplete_family_are_rejected(self):
        f,a,_=family_fixture();out=self.temporary();x=A.publish(out/'a.gz',f,a);y=A.publish(out/'b.gz',f,a)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            read_shards([x['path'],y['path']],native_admissions={r['path']:r['native_admission'] for r in (x,y)})
        with gzip.open(out/'a.gz','at') as stream:stream.write('{}\n')
        with self.assertRaises(ValueError):A.read(out/'a.gz',x['native_admission'])

    def test_reference_table_retains_recorder_and_excludes_rule_bot_critic(self):
        f,a,c=family_fixture(kind='legacy');g=A.convert(f,a)
        from brass_ai.rust_backend.lab_b03_smoke import prepared_rows
        data=prepared_rows([g],c,a['job']['provenance']['learner_behavior']['policy_id'])
        self.assertEqual([s['role'] for s in g.critic_table['seats']],['learner','learner','snapshot','rule_bot'])
        self.assertEqual(len(data),6);self.assertNotIn(3,{r['seat'] for r,_,_,_ in data})
        self.assertEqual(g['local_scoring_targets']['receipt']['producer'],'local-scoring-host/1.0.0')

    def test_native_derived_cache_requires_receipt_and_rebuilds_after_source_change(self):
        f,a,_=family_fixture();p=self.temporary()/'cached.gz';r=A.publish(p,f,a)
        with patch.object(PreparedEncoder,'encode',side_effect=AssertionError('Trusted cache should reuse arrays')):
            A.read(p,r['native_admission'])
        old=copy.deepcopy(r['native_admission']);old['features']['validator']='old-source'
        self.assertEqual(A.read(p,old)['game_id'],f['parent']['game']['game_id'])
        cache=p.with_name(p.name+'.features-v4.npz');cache.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'receipt changed'):A.read(p,r['native_admission'])

    def test_teacher_rows_keep_their_separate_contract(self):
        from test_b025 import example
        m=model_fixture();row,expected=example(m)
        job=dict(kind='teacher_phase',ordinary_training=False,search=config()['teacher_phase']['search'],expected=expected)
        rows,dropped=A.admit_teacher_rows([row.row],job=job,model=m,history=[dict(generation=10,model_id=model_id(m))])
        self.assertEqual(len(rows),1);self.assertFalse(any(dropped.values()))
        job['ordinary_training']=True
        with self.assertRaises(ValueError):A.admit_teacher_rows([row.row],job=job,model=m,history=[])


def continuation_fixture(base):
    """Real small learner, four Adam states, one critic age and a fixed IL probe."""
    from brass_ai.recovery.actor_value import TrainingPhases
    from brass_ai.recovery.local_scoring_auxiliary import Companion
    from brass_ai.recovery.il_kl_probe import FORMAT as PROBE, semantic_hash, source_binding, group_for
    from brass_ai.recovery.il_kl_limit import initial_state
    from brass_ai.recovery import exploration, quality_pool, pfsp_pool
    from test_ppg_eight import stage_only
    run=base/'joint7';run.mkdir();host=base/'host';host.mkdir();write_json(host/'runtime.json',dict(fixture=True))
    c=config();c['host']=host.relative_to(SOURCE_ROOT).as_posix()
    c['local_scoring_auxiliary'].update(adoption=None,mode='off')
    model=model_fixture();optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    il=base/'il';il.mkdir();checkpoint=il/'candidate.pt'
    old_id=save_checkpoint(checkpoint,model,optimizer)
    validation=il/'validation.json';write_json(validation,dict(authored=True))
    manifest=il/'manifest.json';write_json(manifest,dict(entries=[dict(split='validation',path=validation.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(validation))]))
    origin=dict(bc_run=il.relative_to(SOURCE_ROOT).as_posix(),path=checkpoint.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(checkpoint),
        model_id=old_id,manifest=manifest.relative_to(SOURCE_ROOT).as_posix(),manifest_sha256=sha(manifest),bc_signature='a'*64)
    c['imitation_checkpoint']=copy.deepcopy(origin)
    signature=dict(config=c,implementation=PREVIOUS,compatibility=compatibility(),legacy_pool={},teachers={},imitation_source=origin)
    probe_dir=run/'il-kl';probe_dir.mkdir();view,choices=position();current_board(view);encoded=PreparedEncoder().encode(view,choices)
    arrays=dict(states=encoded.state[None,:],actions=encoded.actions,seats=encoded.stable_to_relative[None,:],
                offsets=np.array([0,3]),reference=np.array([.6,.25,.15]))
    probe_cache=probe_dir/'probe.npz';np.savez(probe_cache,**arrays)
    meta=dict(format=PROBE,config=c['il_kl_limit'],source=source_binding(SOURCE_ROOT,origin),rows=[dict(index=0,identity=['fixture',0],
        era='CanalEra',phase='action',group=group_for(view['context']),kinds=[x['kind'] for x in choices],
        industries=[None]*3,chosen=0,choice_ids=list(encoded.choice_ids))],
        cache=dict(path=probe_cache.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(probe_cache)))
    meta['probe_hash']=semantic_hash(meta,arrays);write_json(probe_dir/'manifest.json',meta)
    probe=dict(path=(probe_dir/'manifest.json').relative_to(SOURCE_ROOT).as_posix(),sha256=sha(probe_dir/'manifest.json'),probe_hash=meta['probe_hash'])
    limit=initial_state(c['il_kl_limit'],probe);limit['last_generation']=1
    phases=TrainingPhases(model,c,root=SOURCE_ROOT,run=run)
    stage_only(model,phases,1);write_json(run/'state.json',dict(iteration=1));phases.buffer.committed(1)
    phases.state=gae.initial(c['advantage'],1);phases.critic_state['generation']=1
    phases.beta=2.;phases.trips=[True,False];phases.buffer.pending=copy.deepcopy(phases.buffer.receipts)
    companion=Companion(model,c['local_scoring_auxiliary'])
    for opt,parameters in [(optimizer,list(model.parameters())),(phases.optimizer,list(model.parameters())),
            *[(opt,list(companion.heads.parameters())) for opt in companion.optimizers.values()]]:
        parameters[0].grad=torch.ones_like(parameters[0]);opt.step();opt.zero_grad(set_to_none=True)
    companion.retained_steps=dict(ppo=1,ppg=1)
    exp=exploration.initial(c,1)
    anchor=dict(path=origin['path'],sha256=origin['sha256'],model_id=old_id,generation=0)
    pool=quality_pool.initial(anchor,c['opponent_pool'],dict(probe_hash=meta['probe_hash']))
    pool.update(generation=1,games_done=320);pfsp_pool.seal(pool)
    model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility))
    path=run/'generation-0001.pt'
    mid=save_checkpoint(path,model,optimizer,step=1,training=dict(mode='step5-population-ppo',
        fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(signature),actor_value_phases=phases.payload(),
        local_scoring_auxiliary=companion.payload(1),il_kl_limit=limit,exploration=exp,quality_pool_digest=quality_pool.state_digest(pool)))
    entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=mid,generation=1)
    state=dict(format='step5-recovery-league/2.2.0',signature=signature,iteration=1,games_done=320,next_seed=4921,
        context_counter=42,history=[entry],population=[entry],evaluation_anchor=entry,quality_pool=pool,
        next_search_games=12345,exploration=exp,il_kl_limit=limit,source_transitions=[],
        search_seconds=0.,ordinary_seconds=15.,origin=dict(initialization='completed-il',imitation=origin))
    write_json(run/'state.json',state)
    return run,state


class Continuation(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp=tempfile.TemporaryDirectory(prefix='b03-state-',dir=SOURCE_ROOT/'artifacts')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base=Path(cls.temp.name);cls.source_run,cls.state=continuation_fixture(cls.base)

    def preflight(self):
        with run_lock(self.source_run):
            return C.preflight(SOURCE_ROOT,self.source_run,generation=1,checkpoint_sha256=self.state['history'][-1]['sha256'])

    def fork(self):
        root=self.temporary();return C.create_fork(SOURCE_ROOT,root/'fork',self.preflight())

    def test_full_restore_preserves_model_all_adams_rng_cache_and_recipe(self):
        before={p.relative_to(self.source_run).as_posix():sha(p) for p in self.source_run.rglob('*') if p.is_file()}
        fork=self.fork();restored=C.restore(fork)
        self.assertEqual(restored.state,self.state)
        self.assertEqual(restored.report['optimizers'],['ppo','value','local-scoring-ppo','local-scoring-ppg'])
        self.assertEqual(restored.report['preserved_ppg_ages'],[1])
        self.assertEqual((restored.phases.beta,restored.phases.trips,restored.phases.cache_origin),(2.,[True,False],0))
        self.assertEqual(restored.report['optimizer_steps'],0);self.assertFalse(restored.model.utility_calibration['ready'])
        self.assertEqual(len(restored.probe.rows),1)
        self.assertEqual(before,{p.relative_to(self.source_run).as_posix():sha(p) for p in self.source_run.rglob('*') if p.is_file()})
        second=C.restore(C.DiagnosticFork(SOURCE_ROOT,fork.directory,generation=1,checkpoint_sha256=self.state['history'][-1]['sha256']))
        self.assertTrue(_same_payload(restored.optimizer.state_dict(),second.optimizer.state_dict()))
        self.assertTrue(_same_payload(restored.phases.optimizer.state_dict(),second.phases.optimizer.state_dict()))
        self.assertTrue(_same_payload(restored.companion.payload(1),second.companion.payload(1)))

    def test_expected_generation_hash_foreign_source_and_partial_work_rejected(self):
        for generation,value in ((2,self.state['history'][-1]['sha256']),(1,'0'*64)):
            with self.assertRaisesRegex(ValueError,'Expected generation'):
                C.preflight(SOURCE_ROOT,self.source_run,generation=generation,checkpoint_sha256=value)
        p=self.source_run/'generation-0002.pt.tmp';p.write_bytes(b'uncommitted')
        try:
            with self.assertRaisesRegex(ValueError,'pending'):self.preflight()
        finally:p.unlink()
        fork=self.fork()
        with self.assertRaisesRegex(ValueError,'binding'):
            C.DiagnosticFork(SOURCE_ROOT,fork.directory,generation=2,checkpoint_sha256=self.state['history'][-1]['sha256'])

    def test_plain_copy_cannot_resume_and_mapped_copy_cannot_train_implicitly(self):
        from brass_ai.recovery.resume import load_resume
        from brass_ai.recovery.league import train
        fork=self.fork()
        with self.assertRaisesRegex(ValueError,'escapes'):
            load_resume(SOURCE_ROOT,fork.run,self.state,fork.receipt['target_signature'],reviewed=True)
        with self.assertRaisesRegex(ValueError,'explicit restoration'):
            train(SOURCE_ROOT,fork.run,self.state['signature']['config'])
        bad=copy.deepcopy(fork.receipt['target_signature']);bad['config']['learning_rate']*=.8
        with self.assertRaises(ValueError):load_resume(SOURCE_ROOT,fork.run,self.state,bad,reviewed=True,diagnostic_fork=fork)

    def test_committed_cache_missing_tampered_and_unknown_files(self):
        unknown=self.source_run/'keep-user-file.txt';unknown.write_text('owned by user')
        try:
            fork=self.fork();self.assertFalse((fork.run/unknown.name).exists());self.assertEqual(unknown.read_text(),'owned by user')
        finally:unknown.unlink()
        file=next((fork.run/'.ppg-eight-cache').glob('*.critic.npy'));file.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'diagnostic dependency'):
            C.DiagnosticFork(SOURCE_ROOT,fork.directory,generation=1,checkpoint_sha256=self.state['history'][-1]['sha256'])

    def test_probe_inventory_excludes_preparation_scratch_and_requires_sources(self):
        scratch=self.source_run/'il-kl'/'.prepared';scratch.mkdir();unknown=scratch/'unknown.bin';unknown.write_bytes(b'disposable')
        try:
            receipt=self.preflight()
            self.assertNotIn('il-kl/.prepared/unknown.bin',receipt['dependencies']['files'])
            validation=self.base/'il'/'validation.json'
            self.assertIn(validation.relative_to(SOURCE_ROOT).as_posix(),receipt['dependencies']['external'])
            before=validation.read_bytes();validation.write_bytes(b'changed')
            try:
                with self.assertRaisesRegex(ValueError,'dependency'):self.preflight()
            finally:validation.write_bytes(before)
        finally:unknown.unlink();scratch.rmdir()

    def test_production_source_commit_crash_chain_is_reviewed(self):
        from brass_ai.recovery.resume import load_resume,commit_transition
        # A separate owned fixture avoids changing the shared test origin.
        root=self.temporary();run,state=continuation_fixture(root);target=C.target_signature(state)
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,target,reviewed=True)
        self.assertTrue(changed)
        torch.optim.AdamW(model.parameters()).load_state_dict(raw['optimizer'])
        updated=commit_transition(run,state,target)
        _,_,changed=load_resume(SOURCE_ROOT,run,updated,target)
        self.assertFalse(changed)
        forged=copy.deepcopy(updated);forged['signature']['implementation']='0'*64
        forged['source_transitions'][0]['to_signature']=digest(forged['signature'])
        with self.assertRaisesRegex(ValueError,'not been reviewed'):
            load_resume(SOURCE_ROOT,run,forged,forged['signature'])
        bad=copy.deepcopy(updated);bad['source_transitions'][0]['from_signature']['implementation']='0'*64
        with self.assertRaisesRegex(ValueError,'bound'):load_resume(SOURCE_ROOT,run,bad,target)
        raw['optimizer']=None;path=run/'generation-0001.pt';torch.save(raw,path)
        bad=copy.deepcopy(state);bad['history'][-1]['sha256']=sha(path);bad['population']=[bad['history'][-1]];bad['evaluation_anchor']=bad['history'][-1]
        write_json(run/'state.json',bad)
        with self.assertRaisesRegex(ValueError,'optimizer'):
            C.preflight(SOURCE_ROOT,run,generation=1,checkpoint_sha256=sha(path))

    def test_reference_and_native_critic_ages_coexist_without_relabelling(self):
        fork=self.fork();restored=C.restore(fork);phase=restored.phases
        original=copy.deepcopy(phase.buffer.receipts);games=[];records=[];seed=0;cfg=restored.state['signature']['config']
        for _ in range(9):
            f,a,_=family_fixture(seed=seed,generation=1);seed=a['job']['seed']+1
            path=fork.run/'iteration-0001'/('actor-'+str(seed)+'.gz')
            record=A.publish(path,f,a);records.append(record);games.append(A.read(path,record['native_admission']))
        from brass_ai.rust_backend.lab_b03_smoke import prepared_rows
        from brass_ai.recovery.updates import Dataset
        learner=records[0]['native_admission']['admission']['job']['provenance']['learner_behavior']['policy_id']
        data=prepared_rows(games,cfg,learner);dataset=Dataset(data,'cpu');phase.prepare(dataset)
        source=phase.buffer.source(records,learner,{g['setup_group']:'selfplay' for g in games},'a'*64)
        actor=[i for i,x in enumerate(data) if len(x[3].choice_ids)>1]
        phase.buffer.stage(phase.critic_dataset,phase.train_ids,phase.heldout_ids,actor,source,restored.model,phase.split['selection'])
        self.assertEqual(phase.buffer.pending[0],original[0]);self.assertEqual([r['generation'] for r in phase.buffer.pending],[1,2])
        self.assertTrue(all('native_admission' in r for r in phase.buffer.pending[1]['source']['records']))
        with phase.buffer.datasets('cpu',policy=False) as (ds,_,receipts):
            self.assertEqual(len(receipts),2);self.assertEqual(len(ds.data),sum(r['rows'] for r in receipts))


class FreshIL(Base):
    def test_completed_il_is_separate_and_rejects_incomplete_or_foreign_checkpoint(self):
        run=self.temporary();model=model_fixture();manifest=run/'manifest.json';write_json(manifest,dict(fixture=True))
        sig=dict(initialization='random',implementation=implementation_id(),epochs=1,architecture=model.config.to_dict(),
                 manifest_path=manifest.relative_to(SOURCE_ROOT).as_posix(),manifest_sha256=sha(manifest))
        path=run/'candidate.pt';mid=save_checkpoint(path,model,step=1,training=dict(mode='step5-v3-bc',initialization='random',parent=None,
            fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(sig),manifest_sha256=sha(manifest)))
        state=dict(signature=sig,epoch=1,stopped=False,best=dict(path=path.name,sha256=sha(path)))
        report=dict(format='bc-training-report/1.0.0',signature=sig,epochs=1,candidate_sha256=sha(path))
        write_json(run/'state.json',state);write_json(run/'report.json',report)
        value=C.check_completed_il(SOURCE_ROOT,run);self.assertEqual(value['origin']['model_id'],mid)
        state['epoch']=0;report['epochs']=0;write_json(run/'state.json',state);write_json(run/'report.json',report)
        with self.assertRaisesRegex(ValueError,'not completed'):C.check_completed_il(SOURCE_ROOT,run)
        with self.assertRaises(ValueError):C.preflight(SOURCE_ROOT,run,generation=1,checkpoint_sha256=sha(path))


class Qualification(Base):
    def test_sealed_family_smoke_reuses_all_three_tables_and_branch(self):
        from brass_ai.rust_backend import lab_b03_smoke as B
        from brass_ai.rust_backend.lab_contract import pin
        directory=self.temporary();jobs=[];files=[]
        for index,kind in enumerate(('selfplay','historical','legacy')):
            f,a,c=family_fixture(kind=kind,seed=100*index,branch=kind=='selfplay')
            jobs.append(a['job']);path=directory/(kind+'.gz')
            with gzip.open(path,'wt') as stream:json.dump(f,stream)
            files.append(pin(path))
        snapshot=dict(state=dict(signature=dict(config=c)),generation=10,model_id=a['model_id'])
        report=B.run(snapshot,jobs,files,a['producer'],directory/'replay')
        self.assertTrue(report['passed']);self.assertEqual(report['new_games_collected'],0)
        self.assertEqual({r['kind'] for r in report['families']},{'selfplay','historical','legacy'})
        self.assertTrue(report['families'][0]['branch'])
        for record in report['families']:
            with self.assertRaisesRegex(ValueError,'Diagnostic'):
                A.read(record['receipt']['path'],record['receipt']['native_admission'])

    def test_entry_rejects_pending_stale_and_unsealed_b02_evidence(self):
        from brass_ai.rust_backend import lab_b03_smoke as B
        from brass_ai.rust_backend.lab_contract import pin
        directory=self.temporary();f,a,_=family_fixture()
        def sealed(name,value):
            path=directory/name;write_json(path,value);return pin(path)
        binary=directory/'binary';binary.write_bytes(b'fixture')
        b01=sealed('b01.json',{})
        build=dict(shared_core=SOURCE,b01=b01,search_method='search_v1',trainer_implementation=implementation_id(),
                   collector=pin(binary),bootstrap=dict(binary=pin(binary),dependencies=[]))
        smoke=dict(passed=True,branch_exit_qualified=True,missing_table_kinds=[],
            frozen_jobs=sealed('jobs.json',dict(jobs=[a['job']])),families=[sealed('family.json',f)])
        receipt=dict(contract='brass-lab-b02/2.0.0',passed=True,milestone='B02',method='search_v1',
                     v205_enabled=False,search_enabled=False,training_admitted=False,b01=b01,
                     build=sealed('build.json',build),smoke=sealed('smoke.json',smoke))
        path=directory/'qualification.json'
        snapshot=dict(signature_sha256=a['producer']['signature'])
        module=SimpleNamespace(entry=lambda *args:(snapshot,None))
        spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None))
        with patch.object(B.importlib.util,'spec_from_file_location',return_value=spec), \
                patch.object(B.importlib.util,'module_from_spec',return_value=module):
            write_json(path,receipt);self.assertEqual(B.entry(path,directory,'cuda:0')[0],snapshot)
            for patch_value in (dict(passed=False),dict(method='search_v2'),dict(v205_enabled=True)):
                write_json(path,dict(receipt,**patch_value))
                with self.assertRaisesRegex(ValueError,'Passing'):B.entry(path,directory,'cuda:0')
            write_json(path,receipt);sealed('jobs.json',{})
            with self.assertRaisesRegex(ValueError,'Changed'):B.entry(path,directory,'cuda:0')
            smoke.pop('families');receipt['smoke']=sealed('smoke.json',smoke);write_json(path,receipt)
            with self.assertRaisesRegex(ValueError,'sealed admission'):B.entry(path,directory,'cuda:0')
            build['trainer_implementation']='0'*64;receipt['build']=sealed('build.json',build);write_json(path,receipt)
            with self.assertRaisesRegex(ValueError,'exact trainer'):B.entry(path,directory,'cuda:0')


if __name__=='__main__':unittest.main()
