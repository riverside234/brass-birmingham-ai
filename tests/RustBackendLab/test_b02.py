"""Short development checks. No game, native process, GPU or lab qualification."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery.execution import neural_spec
from brass_ai.recovery.exploration import REVISED_OPENING,REVISED_BRANCH,THREE_OPENING,THREE_BRANCH
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.rust_backend.lab_collection import frozen_id,route
from brass_ai.rust_backend.lab_families import FamilyStore,validate_family,validate_part,collect_families,FORMAT
from brass_ai.rust_backend.lab_transport import DecisionEncoder


def job(index=0):
    spec=neural_spec('frozen',.05,'neutral',version=3,opening=REVISED_OPENING)
    return dict(index=index,seed=91+index,sampling_seed=31,policies=[copy.deepcopy(spec) for _ in range(4)],
                provenance=dict(kind='selfplay'),path=f'ignored-{index}',stable_game_id=f'job-{index}')


def part(name='main',branch=False):
    return dict(game=dict(game_id=name),branch_expected=branch)


class Contract(unittest.TestCase):
    def test_current_search_v1_b01_receipt_is_accepted_without_lab_work(self):
        from brass_ai.rust_backend.lab_contract import B01_CONTRACT,METHOD,SOURCE,extract_new,pin
        from brass_ai.learning.contracts import read_json
        spec=importlib.util.spec_from_file_location('b02_runner',SOURCE_ROOT/'scripts/qualify-rust-lab-b02.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            temp=Path(temp)
            extract_new(SOURCE_ROOT/'distribution/rust-lab-b01/b01-search-v1-inputs.zip',temp/'inputs')
            def sealed(name,value):
                path=temp/name;path.write_text(json.dumps(value),encoding='utf-8');return pin(path)
            binary=temp/'worker';binary.write_bytes(b'mocked binary')
            state={'state_file':{'path':str(temp/'state.json')}}
            value=dict(contract=B01_CONTRACT,method=METHOD,milestone='B01',passed=True,v205_enabled=False,
                profile=sealed('profile.json',read_json(SOURCE_ROOT/'configs/search/lab-search-v1.json')),
                lab_state=sealed('state-receipt.json',state),local_entry=sealed('entry.json',{}),
                linux_build=sealed('build.json',dict(binary=pin(binary),handshake=dict(build=dict(source_sha256=SOURCE)))),
                python_sources=sealed('sources.json',[]),commands=sealed('commands.json',[]),
                gpu=[sealed('gpu.json',dict(passed=True,device='cuda:0'))])
            path=temp/'qualification.json';path.write_text(json.dumps(value),encoding='utf-8')
            with patch.object(m,'native_fingerprint',return_value=SOURCE),patch('brass_ai.rust_backend.lab_snapshot.verify_unchanged'):
                self.assertEqual(m.entry(path,temp,'cuda:0')[0],state)
            (temp/'profile.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Changed B01 profile'):m.entry(path,temp,'cuda:0')

    def test_output_relocation_does_not_change_job_but_any_execution_input_does(self):
        j=job();self.assertEqual(frozen_id(j),frozen_id(dict(j,path='relocated')))
        for key,value in [('seed',100),('sampling_seed',1),('rotation',1),('focuses',{'0':'Brewery'})]:
            self.assertNotEqual(frozen_id(j),frozen_id(dict(j,**{key:value})))

    def test_reference_route_never_silently_substitutes_unsupported_seat(self):
        j=job();before=copy.deepcopy(j);self.assertEqual(route(j),('rust',None));self.assertEqual(j,before)
        j['policies'][2]=dict(kind='lab-legacy',policy_id='exact-v4',search_free=True)
        self.assertEqual(route(j)[0],'reference');self.assertEqual(j['policies'][2]['policy_id'],'exact-v4')
        j['policies'][2]=dict(kind='masked_ismcts',policy_id='search')
        self.assertEqual(route(j)[0],'reference')

    def test_branch_contract_is_frozen_and_supports_current_opening_revision(self):
        for branch in (REVISED_BRANCH,THREE_BRANCH):
            j=dict(job(),branching=copy.deepcopy(branch));self.assertEqual(route(j)[0],'rust')
            j['branching']['max_per_game']=2
            with self.assertRaises(ValueError):route(j)

    def test_b01_pending_is_not_qualification(self):
        spec=importlib.util.spec_from_file_location('b02_runner',SOURCE_ROOT/'scripts/qualify-rust-lab-b02.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'qualification.json'
            for value in ({'status':'Pending'},{'contract':'brass-lab-b01/1.0.0','passed':False},
                          {'contract':'brass-lab-b02/1.0.0','passed':True},
                          {'contract':'brass-lab-b01/1.0.0','passed':True,'milestone':'B01','v205_enabled':False,'method':'search_v2_focus_only'},
                          {'contract':'brass-lab-b01/2.0.0','passed':True,'milestone':'B01','v205_enabled':False,'method':'search_v2_focus_only'},
                          {'contract':'brass-lab-b01/2.0.0','passed':True,'milestone':'B01','v205_enabled':True,'method':'search_v1'}):
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'B01 qualification'):m.entry(path,temp,'cuda:0')

    def test_missing_table_or_branch_coverage_cannot_pass(self):
        from brass_ai.rust_backend.lab_b02_smoke import require_coverage
        require_coverage({'selfplay','historical','legacy'},1)
        with self.assertRaisesRegex(ValueError,'table coverage'):require_coverage({'selfplay'},1)
        with self.assertRaisesRegex(ValueError,'branch family'):require_coverage({'selfplay','historical','legacy'},0)


class Behavior(unittest.TestCase):
    def make(self,opening):
        j=job()
        for spec in j['policies']:spec.update(opening=opening)
        evaluator=type('Evaluator',(),{'model_id':'frozen'})()
        p=NeuralPolicy(None,evaluator=evaluator,exploration=.05,focus='neutral',exploration_version=3,opening=opening)
        for spec in j['policies']:spec['policy_id']=p.policy_id
        rng=random.Random(digest(dict(sampling_seed=j['sampling_seed'],game=0,seat=0)));rows=[];evidence=[]
        choices=[dict(id='loan',kind='loan'),dict(id='pass',kind='pass')]
        for step,phase in enumerate(('action','continuation','action','action','action')):
            legal=choices if phase=='action' else [dict(id='finish',kind='finish')]
            view=dict(seat=0,context=dict(phase=phase));raw=np.array([.8,.2]) if len(legal)>1 else np.ones(1)
            selected,probability,_=p.sample_training(view,legal,rng,raw,[0.]*8)
            rows.append(dict(step=step,seat=0,view=view,choices=legal,chosen=selected,style=[0.]*4,intent=[0.]*8,
                behavior=dict(kind='model',policy_id=p.policy_id,probability=probability,**p.last_behavior[rng])))
            evidence.append(dict(raw_probabilities=raw.tolist(),model_id='frozen',policy_id=p.policy_id,semantic_id=legal[selected]['id']))
        game=dict(records=rows,game_id=j['stable_game_id'],setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=j['seed'])),
                  opponents=[s['policy_id'] for s in j['policies']])
        return j,dict(game=game,evidence=evidence,producer=dict(contract='brass-reference-collection/1.0.0'))

    def test_real_sampler_behavior_and_intermediate_counter_parity(self):
        for opening in (REVISED_OPENING,THREE_OPENING):
            j,p=self.make(opening)
            with patch('brass_ai.rust_backend.lab_families.check_game',side_effect=lambda g:g):validate_part(p,j)
            self.assertEqual([r['behavior']['action_decision'] for r in p['game']['records']],[0,None,1,2,3])

    def test_wrong_denominator_counter_model_policy_or_legal_id_is_rejected(self):
        j,original=self.make(REVISED_OPENING)
        mutations=[lambda p:p['game']['records'][0]['behavior'].update(probability=.5),
                   lambda p:p['game']['records'][2]['behavior'].update(action_decision=2),
                   lambda p:p['evidence'][0].update(model_id='same-weights-other-model'),
                   lambda p:p['evidence'][0].update(policy_id='snapshot-not-learner'),
                   lambda p:p['evidence'][0].update(semantic_id='illegal'),
                   lambda p:p['game'].update(game_id='foreign'),
                   lambda p:p['game'].update(setup_group='foreign'),
                   lambda p:p['game']['records'][0]['behavior'].update(component='policy' if p['game']['records'][0]['behavior']['component']=='exploration' else 'exploration'),
                   lambda p:p['evidence'][0].update(checkpoint={'hidden':'forbidden'})]
        for mutate in mutations:
            p=copy.deepcopy(original);mutate(p)
            with patch('brass_ai.rust_backend.lab_families.check_game',side_effect=lambda g:g),self.assertRaises(ValueError):validate_part(p,j)

    def test_forced_probability_one_is_still_a_record(self):
        j,p=self.make(REVISED_OPENING)
        self.assertEqual(p['game']['records'][1]['behavior']['probability'],1.)
        p['game']['records'][1]['behavior']['probability']=.99
        with patch('brass_ai.rust_backend.lab_families.check_game',side_effect=lambda g:g),self.assertRaises(ValueError):validate_part(p,j)

    def test_cancellation_skips_native_encoding(self):
        event=threading.Event();event.set()
        from brass_ai.learning.encoding_budget import EncodingStopped
        encoder=DecisionEncoder(None,'model')
        with self.assertRaises(EncodingStopped):encoder.encode({},[],cancelled=event)
        with self.assertRaises(EncodingStopped):encoder.encode({},[],deadline=time.monotonic()-1)


class Publication(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.store=FamilyStore(self.temp.name,{'core':'frozen'})
        self.admit=patch('brass_ai.rust_backend.lab_families.validate_part',side_effect=lambda p,*a,**k:p['game'])
        self.admit.start();self.addCleanup(self.admit.stop)
        self.child=patch('brass_ai.rust_backend.lab_families.validate_child')
        self.child.start();self.addCleanup(self.child.stop)

    def test_incomplete_parent_cannot_publish(self):
        with self.assertRaises(ValueError):self.store.publish(job(),part(branch=True),None)
        self.assertIsNone(self.store.load(job()))

    def test_atomic_ack_retry_and_changed_duplicate_rejection(self):
        j=job();value=self.store.publish(j,part(),None)
        self.assertEqual(value,self.store.load(j));self.assertEqual(value,self.store.publish(j,part(),None))
        with self.assertRaises(ValueError):self.store.publish(j,part('different'),None)
        self.assertFalse(list(Path(self.temp.name).glob('*.tmp')))

    def test_source_job_seal_or_training_admission_cannot_change(self):
        j=job();v=self.store.publish(j,part(),None)
        for key,x in [('job_id','foreign'),('producer',{}),('training_admitted',True),('sha256','0'*64)]:
            changed=dict(v,**{key:x})
            if key!='sha256':changed['sha256']=digest({k:x for k,x in changed.items() if k!='sha256'})
            with self.assertRaises(ValueError):validate_family(changed,j,self.store.producer)

    def test_stale_partial_file_is_never_a_complete_family(self):
        self.store.path(job()).with_suffix('.tmp').write_bytes(b'partial')
        self.assertIsNone(self.store.load(job()))

    def test_one_lane_schedules_child_without_waiting_parent_and_reuses_complete_family(self):
        jobs=[dict(job(0),branching=REVISED_BRANCH),job(1)];order=[]
        def collect(j,continuation,submit):
            role='child' if continuation else 'parent';order.append((j['index'],role))
            if j['index']==0 and continuation is None:
                submit(dict(row={'view':{'context':{'era':'CanalEra','round':3}}}))
                return part(branch=True)
            return part(role)
        result,report=collect_families(jobs,self.store,collect,collect,lanes=1)
        self.assertEqual(order,[(0,'parent'),(0,'child'),(1,'parent')]);self.assertEqual(len(result),2)
        self.assertLessEqual(report['peak_unfinished_families'],2)
        def forbidden(*args):raise AssertionError('retry executed job')
        _,cached=collect_families(jobs,self.store,forbidden,forbidden,lanes=1)
        self.assertEqual(cached['counts']['cached'],2)

    def test_child_can_finish_first_and_join_exactly_once(self):
        entered=threading.Event();ended=threading.Event()
        def collect(j,continuation,submit):
            if continuation:
                entered.set();ended.set();return part('child')
            submit(dict(row={'view':{'context':{'era':'RailEra','round':4}}}))
            self.assertTrue(ended.wait(2));return part(branch=True)
        result,report=collect_families([dict(job(),branching=REVISED_BRANCH)],self.store,collect,collect,lanes=2)
        self.assertTrue(entered.is_set());self.assertEqual(len(result),1)

    def test_canceled_or_failed_work_does_not_publish(self):
        stop=threading.Event();stop.set()
        with self.assertRaisesRegex(ValueError,'canceled'):collect_families([job()],self.store,None,None,lanes=1,cancel=stop)
        def fail(*args):raise RuntimeError('device failed')
        with self.assertRaises(RuntimeError):collect_families([job()],self.store,fail,fail,lanes=1)
        self.assertIsNone(self.store.load(job()))

    def test_cancellation_after_worker_completion_still_prevents_publication(self):
        stop=threading.Event()
        def collect(*args):stop.set();return part()
        with self.assertRaisesRegex(ValueError,'canceled'):
            collect_families([job()],self.store,collect,collect,lanes=1,cancel=stop)
        self.assertIsNone(self.store.load(job()))

    def test_duplicate_jobs_and_fork_messages_rejected(self):
        with self.assertRaises(ValueError):collect_families([job(),job()],self.store,None,None,lanes=1)
        def collect(j,continuation,submit):
            if continuation:return part('child')
            packet=dict(row={'view':{'context':{'era':'CanalEra','round':2}}})
            submit(packet);submit(packet);return part(branch=True)
        with self.assertRaisesRegex(ValueError,'Duplicate'):collect_families([dict(job(),branching=REVISED_BRANCH)],self.store,collect,collect,lanes=1)

    def test_parent_cannot_deny_a_submitted_branch_and_publish_alone(self):
        def collect(j,continuation,submit):
            submit(dict(row={'view':{'context':{'era':'CanalEra','round':2}}}))
            return part(branch=False)
        with self.assertRaisesRegex(ValueError,'Parent/fork'):
            collect_families([dict(job(),branching=REVISED_BRANCH)],self.store,collect,collect,lanes=1)
        self.assertFalse(list(Path(self.temp.name).glob('*.family.json.gz')))

if __name__=='__main__':unittest.main()
