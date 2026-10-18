import copy
from concurrent.futures import ThreadPoolExecutor
import json
import random
from threading import Event
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from fixtures import position
from personality_fixtures import parent
from brass_ai.learning.contracts import SOURCE_ROOT,digest,read_json
from brass_ai.learning.reward import specification
from brass_ai.learning.reward import CostLedger
from brass_ai.personality.model import import_parent
from brass_ai.personality.contracts import context,GOALS,FAMILIES,FACTORS
from brass_ai.personality.search.contracts import Profile,search_config,identity
from brass_ai.personality.search.inference import PersonalityRequest,SharedInference,PersonalityRemote,PersonalityPolicy
from brass_ai.personality.search.intent import biased_prior,candidate_order,alignment,IntentState
from brass_ai.personality.search.engine import teacher,validate_personality_teacher,RootMasked,SearchSession
from brass_ai.personality.search.teachers import due,CADENCE,WARMUP,WALL_FRACTION
from brass_ai.recovery.teacher_schedule import reserved_rounds,PHASES,counters
from brass_ai.search.ismcts import MaskedISMCTS,Node
from brass_ai.step5.policy import exploration_distribution


def goal():
    return dict(schema=GOALS,family='industry_sequence',industry='CottonMill',region='unspecified',horizon=4,
        confidence=.85,age=0,created_turn='authored-goal',revision_reason='fixture')


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_zero_bias_exact_and_bounded_support(self):
        _,choices=position();raw=np.array([.001,.499,.5]);base=.9*raw+.1*exploration_distribution(choices)
        g=np.array([1.,0.,-.5])
        zero=biased_prior(base,raw,choices,g,Profile(alpha=0.),1.)
        np.testing.assert_array_equal(zero,base)
        self.assertEqual(candidate_order(choices,base,zero,16,Profile(alpha=0.)),candidate_order(choices,base,base,16,Profile()))
        p=biased_prior(base,raw,choices,g,Profile(alpha=.35),.8)
        self.assertTrue((p>0).all());self.assertAlmostEqual(p.sum(),1.)
        # Removing the unchanged grouped exploration recovers a bounded neural
        # odds tilt, not action rewards or a legality-mask change.
        q=(p-.1*exploration_distribution(choices))/.9
        self.assertLessEqual((q[0]/q[2])/(raw[0]/raw[2]),np.exp(.35*.8*1.5))

    def test_unbiased_candidates_and_industry_protection(self):
        choices=[dict(id=f'cotton-{i}',kind='build',industry='CottonMill') for i in range(30)]
        choices+=[dict(id='pottery',kind='build',industry='Pottery'),dict(id='develop',kind='develop',industry='Pottery'),dict(id='loan',kind='loan')]
        base=np.arange(len(choices),0,-1,dtype=float);base/=base.sum();biased=base[::-1]
        order,protected=candidate_order(choices,base,biased,16,Profile())
        self.assertEqual(order[:4],[0,1,2,3]);self.assertTrue({30,31,32}<=set(protected));self.assertEqual(set(order),set(range(len(choices))))

    def test_intent_development_progress_restore_and_seat_privacy(self):
        view,choices=position();state=IntentState(0,Profile());state.before_turn(view);state.goal=goal()
        state.commit({},view,dict(bias_ready=False))
        changed=copy.deepcopy(view);changed['context']['round']=2
        next(m for m in changed['players'][0]['mat'] if m['type']=='CottonMill')['tiles'][0]['level']+=1
        ctx=state.before_turn(changed)
        self.assertEqual(state.stall,0);self.assertEqual(ctx['saved_intent']['revision_reason'],'productive-progress')
        saved=json.loads(json.dumps(state.snapshot()));restored=IntentState(0,Profile());restored.restore(saved)
        self.assertEqual(restored.snapshot(),state.snapshot())
        with self.assertRaisesRegex(ValueError,'another seat'):restored.before_turn(position(1)[0])
        changed['context']['era']='RailEra';changed['context']['round']=1
        self.assertIsNone(restored.before_turn(changed)['saved_intent'])

    def test_alignment_only_at_action_root(self):
        view,choices=position();g=goal()
        choices=[dict(id='d',kind='develop',industry='CottonMill'),dict(id='c',kind='coal',target='market')]
        self.assertGreater(alignment(view,choices,g)[0],0)
        view['context']['phase']='coal';np.testing.assert_array_equal(alignment(view,choices,g),[0,0])

    def test_root_hook_never_biases_opponents_or_continuations(self):
        view,choices=position();bot=object.__new__(RootMasked);bot.base=object();bot.root=0;bot.root_key='with-budget'
        bot.config=search_config(specification());bot.tree={bot.root_key:Node()}
        state=type('State',(),dict(legal=choices))();p=np.array([.3,.3,.4])
        bot.scope=dict(information='root',inject=True,alignment=np.array([1.,0.,0.]),profile=Profile(),confidence=.9)
        def request(*args,**kwargs):
            if False:yield None
            return p,dict(probabilities=p)
        def finish(generator):
            try:next(generator)
            except StopIteration as done:return done.value
        with patch.object(MaskedISMCTS,'request',request),patch('brass_ai.personality.search.engine.node_key',return_value='root'):
            changed,_=finish(bot.request(bot.base,state,random.Random(1)));self.assertFalse(np.array_equal(changed,p))
            actual,_=finish(bot.request(object(),state,random.Random(1)));np.testing.assert_array_equal(actual,p)
        with patch.object(MaskedISMCTS,'request',request),patch('brass_ai.personality.search.engine.node_key',return_value='payment'):
            actual,_=finish(bot.request(bot.base,state,random.Random(1)));np.testing.assert_array_equal(actual,p)

    def test_central_queue_named_heads_context_cancel_and_expiry(self):
        from fe_fixtures import Factory
        from fixtures import rotate
        factory=Factory()
        try:view,choices=factory.position()
        finally:factory.close()
        model,_=import_parent(parent());execution=read_json(SOURCE_ROOT/'configs/step5/lab.json')['execution']
        service=SharedInference(execution);ident=service.register(model);channels=service.channels(4)
        remotes=[PersonalityRemote(channels.acquire(),ident) for _ in range(4)]
        for remote in remotes:self.addCleanup(remote.endpoint.close)
        def request(i,deadline=None,cancel=None):
            return PersonalityRequest(str(i),ident,rotate(view,i),choices,deadline or time.monotonic()+5,
                cancelled=cancel,remaining=(.025,)*4,personality=context('networker' if i else 'neutral'))
        with service:
            requests=[request(i) for i in range(4)]
            with ThreadPoolExecutor(4) as pool:responses=list(pool.map(lambda x:remotes[x].predict([requests[x]])[0],range(4)))
            for i,r in enumerate(responses):
                self.assertEqual(r['status'],'ok');self.assertEqual(r['context_id'],digest(requests[i].personality));self.assertEqual(len(r['utility']),4)
            event=Event();event.set();self.assertEqual(remotes[0].predict([request(0,cancel=event)])[0]['status'],'cancelled')
            self.assertEqual(remotes[1].predict([request(1,deadline=time.monotonic()-1)])[0]['status'],'expired')
        self.assertTrue(service.report()['central_batching']);self.assertEqual(service.report()['positions'],4)

    def teacher_fixture(self):
        view,choices=position();cfg=search_config(specification());model=digest('fixture-model')
        trace=[('root-info',choices[0]['id'])];signature=digest(trace)
        stats=dict(root={c['id']:dict(visits=n,returns=[1.]*4,available=3) for c,n in zip(choices,[2,1,0]) if n},
            expansion=[True]*3,plans={signature:dict(signature=signature,first=choices[0]['id'],trace=trace,visits=2,returns=[1.]*4)})
        scope=dict(context=context(),goal=goal(),profile=Profile(),inject=True,information='root-info',view_hash=digest(view),id='context',calibration_id='calibration')
        label=teacher(choices,stats,[.4,.3,.3],[.45,.3,.25],model=model,seat=0,config=cfg,scope=scope,remaining=[.025]*4,selected=0)
        row=dict(view=view,choices=choices,seat=0,chosen=0,teacher=None,behavior=dict(kind='external'))
        return label,row,model,cfg

    def test_teacher_rejects_old_objective_context_and_model(self):
        label,row,model,cfg=self.teacher_fixture();validate_personality_teacher(label,row,model,cfg.reward)
        with self.assertRaisesRegex(ValueError,'model/reward'):validate_personality_teacher(label,row,'old',cfg.reward)
        changed=copy.deepcopy(label);changed['personality']['profile']='supplier'
        with self.assertRaisesRegex(ValueError,'context changed'):validate_personality_teacher(changed,row,model,cfg.reward)
        changed=copy.deepcopy(label);changed['selected_plan_support']=1;changed['fingerprint']=digest({k:v for k,v in changed.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'Insufficient'):validate_personality_teacher(changed,row,model,cfg.reward)

    def test_reserved_rounds_warmup_cadence_and_failed_time_budget(self):
        a=[reserved_rounds(123,456,s) for s in range(4)]
        self.assertEqual(a,[reserved_rounds(123,456,s) for s in range(4)]);self.assertGreater(len({digest(x) for x in a}),1)
        for rounds in a:
            for name,era,lo,hi in PHASES:self.assertTrue(lo<=rounds[name]<=hi)
        self.assertEqual((WARMUP,CADENCE,WALL_FRACTION),(640,160,.9))
        state=dict(ordinary_games=639,next_games=640,latest_iteration='one',iteration_attempts={},ordinary_seconds=1000.,search_seconds=0.)
        self.assertEqual(due(state,12),'not_due');state['ordinary_games']=640;self.assertIsNone(due(state,12))
        state['search_seconds']=9000.;self.assertEqual(due(state,12),'wall_time_deferred')
        state['search_seconds']=0;state['iteration_attempts']['one']=2;self.assertEqual(due(state,12),'iteration_collection_limit')

    def test_one_deadline_after_admission_both_actions_workers_and_cancel(self):
        from concurrent.futures import Future
        from threading import RLock
        from brass_ai.recovery.telemetry import Timings
        session=object.__new__(SearchSession);session.config=search_config(specification());session.profiles=[Profile()]*4
        session.search_enabled=True
        session.calibration=dict(bias_ready=False);session.model_id=digest('model');session.cancelled=Event();session.lock=RLock()
        session.timings=Timings();clock=[100.];admitted=[];deadlines=[]
        class Lease:
            def update(self,key,n):clock[0]+=2.;admitted.append(n)
            def close(self):pass
        class Pool:
            def submit(self,f,i):
                future=Future()
                try:future.set_result(f(i))
                except BaseException as e:future.set_exception(e)
                return future
        class Policy:
            request_deadline=None
            utility_remaining={}
            def set_context(self,value):self.personality=value
            def predict(self,inputs):return [(np.array([.4,.3,.3]),dict(probabilities=[.4,.3,.3])) for _ in inputs]
        class FakeWorker:
            def __init__(self):self.base=Policy()
            def decide(self,view,choices,rng,deadline,*_):
                deadlines.append(deadline);clock[0]+=.2;trace=[(self.scope['information'],choices[0]['id'])];sig=digest(trace)
                return dict(context_id=self.scope['id'],completed=3,interrupted=0,opponent_nodes=0,max_pending=0,opponent_choices=1,
                    batching=dict(attempted=3,deadline_stops=0,deadline_remaining_seconds=0.),
                    expansion=[True]*3,root={c['id']:dict(visits=n,returns=[1.]*4,available=3) for c,n in zip(choices,[2,1,0]) if n},
                    plans={sig:dict(signature=sig,first=choices[0]['id'],trace=trace,visits=2,returns=[1.]*4)})
        session.lease=Lease();session.pool=Pool();session.workers=[FakeWorker() for _ in range(4)];session.base=Policy();session.game=None
        session.turn=None;session.telemetry=dict(decisions=0,searches=0,completed=0,interrupted=0,fallbacks=0,seconds=0.,prior_seconds=0.,overruns=0.,
            attempted=0,deadline_stops=0,deadline_unused_worker_seconds=0.)
        session.begin(42,1);view,choices=position();ledger=CostLedger(specification());ledger.observe(view);rng=random.Random(17)
        with patch('time.monotonic',side_effect=lambda:clock[0]),patch('brass_ai.recovery.teacher_schedule.TeacherTurns.incumbent',return_value={}):
            _,label,_=session.select(view,choices,rng,ledger);self.assertIsNotNone(label);self.assertEqual(session.turn['deadline'],114.)
            view=copy.deepcopy(view);view['context']['revision']+=1;view['context']['actions_remaining']=1
            session.select(view,choices,rng,ledger)
            self.assertEqual(len(deadlines),8);self.assertTrue(all(x==112.8 for x in deadlines));self.assertEqual(admitted,[4])
            self.assertEqual(session.telemetry['attempted'],24);self.assertEqual(session.telemetry['deadline_stops'],0)
            session.cancelled.set();_,label,_=session.select(view,choices,rng,ledger);self.assertIsNone(label);self.assertEqual(len(deadlines),8)
            session.finish_turn();saved=json.loads(json.dumps(session.snapshot()));session.restore(saved);self.assertEqual(session.snapshot(),saved)

    def test_forced_root_does_not_create_intent_from_payment_predictions(self):
        from threading import RLock
        session=object.__new__(SearchSession);session.config=search_config(specification());session.profiles=[Profile()]*4
        session.search_enabled=False;session.calibration=dict(bias_ready=False);session.cancelled=Event();session.lock=RLock()
        session.lease=SimpleNamespace(update=lambda *args:None,close=lambda:None);session.workers=[];session.turn=None
        class Policy:
            request_deadline=None
            utility_remaining={}
            def set_context(self,value):pass
            def predict(self,inputs):
                return [(np.full(len(c),1/len(c)),{}) for _,c,_ in inputs]
        session.base=Policy();session.telemetry=dict(decisions=0)
        session.begin(42,1);view,choices=position();ledger=CostLedger(specification());ledger.observe(view);rng=random.Random(1)
        with patch.object(session.intents[0],'commit',return_value=None) as propose:
            session.select(view,choices[:1],rng,ledger)
            payment=copy.deepcopy(view);payment['context']['phase']='iron';payment['context']['revision']+=1
            session.select(payment,choices,rng,ledger)
            propose.assert_not_called();self.assertTrue(session.turn['proposed'])
            view['context']['revision']+=2;view['context']['actions_remaining']=1
            session.select(view,choices,rng,ledger)
            propose.assert_not_called()
            view['context']['round']+=1;view['context']['actions_remaining']=2
            session.select(view,choices,rng,ledger)
            propose.assert_called_once();self.assertTrue(session.turn['proposed'])
        session.finish_turn()


if __name__=='__main__':unittest.main()
