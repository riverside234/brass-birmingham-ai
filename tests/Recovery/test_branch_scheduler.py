import copy
import queue
import threading
import unittest
from unittest.mock import Mock

from brass_ai.recovery.branch_scheduler import ReadyWork,FamilyMailbox,report
from brass_ai.recovery.execution import ActorPool
from brass_ai.recovery.history import compact
from brass_ai.recovery.telemetry import iteration_utilization


def packet(index,kind='selfplay',branch=True):
    job=dict(index=index,provenance=dict(kind=kind))
    if branch:job['branching']={'rate_per_decision':.005}
    return dict(schedule_token='test',jobs=[job])


def fork(index,era,round):
    return dict(packet(index),continuation=dict(row=dict(view=dict(context=dict(era=era,round=round)))))


class BranchSchedulingTests(unittest.TestCase):
    def test_ready_long_branches_precede_pending_parents_without_changing_jobs(self):
        jobs=[packet(3,'legacy',False),packet(2,'historical',False),packet(1),packet(0)]
        original=copy.deepcopy(jobs)
        ready=ReadyWork(jobs,{'legacy':4,'historical':2})
        self.assertEqual(ready.pop()['jobs'][0]['index'],0)
        ready.add_branch(fork(8,'RailEra',7));ready.add_branch(fork(9,'CanalEra',3))
        self.assertEqual([ready.pop()['jobs'][0]['index'] for _ in range(5)],[9,8,1,3,2])
        self.assertFalse(ready);self.assertEqual(jobs,original)

    def test_family_joins_in_either_order_without_waiting_on_a_worker(self):
        inbox=queue.Queue();results=queue.Queue();stop=threading.Event();published=[]
        mailbox=FamilyMailbox(inbox,results,stop,lambda *a:published.append(a))
        try:
            a=packet(1);b=packet(2)
            mailbox.parent(['parent-a'],a)
            self.assertEqual(published,[])
            mailbox.child(dict(key=('test',2),payload=b'b'))
            mailbox.parent(['parent-b'],b)
            mailbox.child(dict(key=('test',1),payload=b'a'))
            self.assertEqual([p[0][0] for p in published],['parent-b','parent-a'])
            self.assertTrue(results.empty())
        finally:mailbox.close()

    def test_duplicate_child_rejected_and_missing_family_fails_closed(self):
        inbox=queue.Queue();results=queue.Queue();stop=threading.Event()
        mailbox=FamilyMailbox(inbox,results,stop,Mock())
        mailbox.child(dict(key=('test',0),payload=b'one'))
        with self.assertRaisesRegex(ValueError,'Duplicate'):mailbox.child(dict(key=('test',0),payload=b'two'))
        with self.assertRaisesRegex(ValueError,'Unpublished'):mailbox.close()

    def test_collection_budget_and_tail_report_are_not_cpu_utilization(self):
        events=[dict(kind='parent',job=0,started=0,finished=10),
                dict(kind='parent',job=1,started=0,finished=5),
                dict(kind='branch',job=0,queued=2,started=5,finished=12)]
        r=report(events,0,13,2)
        self.assertEqual(r['branch_worker_seconds_per_worker'],3.5)
        self.assertEqual(r['branch_tail_seconds'],2)
        self.assertEqual(r['publication_tail_seconds'],1)
        self.assertEqual(r['branch_queue_wait_mean_seconds'],3)
        self.assertEqual(r['branches_started_before_parent_finished'],1)
        self.assertEqual(r['parent_branch_overlap_seconds'],5)
        self.assertAlmostEqual(r['work_utilization'],22/24)
        self.assertIn('not CPU busy time',r['scope'])
        summary=compact(dict(exploration_metrics={'branching':{'scheduling':r}}))
        self.assertEqual(summary['exploration_metrics']['branching']['scheduling'],r)

    def test_branch_cpu_and_phase_work_are_included_once(self):
        record=dict(metrics={'host':{'seconds':3}},actor_python_cpu_seconds=2,
            branch_execution=dict(metrics={'host':{'seconds':1}},actor_python_cpu_seconds=.5))
        r=iteration_utilization({}, {},[record],4,1,
            dict(processes=1,hosts_per_process=1,cpu_budget=2,cpu_reserve=0))
        self.assertEqual(r['actor_phase_seconds']['host'],4)
        self.assertEqual(r['actor_python_cpu_seconds'],2.5)
        self.assertEqual(r['actor_cpu_expected_work_items'],2)

    def test_coordinator_dispatches_fork_before_next_parent_and_drains_events(self):
        """A branch arrives while its parent is running; a different lane frees."""
        pool=ActorPool.__new__(ActorPool)
        pool.width=2;pool.lanes=1;pool.heavy=False;pool.processes=[Mock(),Mock()]
        pool.service=Mock();pool.service.error=None
        pool.jobs=Mock();pool.results=queue.Queue()
        pending=[packet(0),packet(1),packet(2)]
        dispatched=[];tokens=[];seq=iter(range(50));branch_dispatched=threading.Event()
        def put(p):
            i=p['jobs'][0]['index'];branch='continuation' in p
            token=p['schedule_token'];tokens.append(token);dispatched.append(('branch' if branch else 'parent',i))
            if branch:branch_dispatched.set()
            t=next(seq)
            e=dict(token=token,job=i,kind='branch' if branch else 'parent',
                table_kind='selfplay',started=t,finished=t+1,queued=t)
            if not branch and i==0:
                f=dict(p,continuation=dict(row=dict(view=dict(context=dict(era='CanalEra',round=3)))))
                pool.results.put(dict(fork=f))
                # Keep parent's completion pending until the branch starts.
                pool.delayed=e
            else:
                pool.results.put(dict(work_complete=e))
                if branch:
                    pool.results.put(dict(work_complete=pool.delayed))
                    pool.results.put(dict(records=[dict(job=0)]))
                else:pool.results.put(dict(records=[dict(job=i)]))
        pool.jobs.put.side_effect=put
        received=[]
        def blocked_reader(rs):
            self.assertTrue(branch_dispatched.wait(timeout=2),'Reader backpressure blocked branch dispatch')
            received.extend(rs)
        result=pool._collect_branches([p['jobs'][0] for p in pending],pending,[],blocked_reader)
        self.assertEqual(dispatched,[('parent',0),('parent',1),('branch',0),('parent',2)])
        self.assertEqual([r['job'] for r in result],[0,1,2])
        self.assertTrue(pool.results.empty());self.assertEqual(len(received),3)
        self.assertEqual(pool.last_collection_report['branch_games'],1)
        self.assertEqual(len(set(tokens)),1)

    def test_failed_reader_submission_aborts_before_join(self):
        pool=ActorPool.__new__(ActorPool);pool.service=Mock();pool.last_collection_report={}
        def dispatch(jobs,pending,records,callback):
            callback([dict(job=0)])
            return records
        pool._dispatch_branches=dispatch
        with self.assertRaisesRegex(ValueError,'reader failed'):
            pool._collect_branches([],[],[],Mock(side_effect=ValueError('reader failed')))
        pool.service.cpu.abort.assert_called_once()

    def test_coordinator_rejects_foreign_completion(self):
        pool=ActorPool.__new__(ActorPool);pool.lanes=1;pool.heavy=False;pool.processes=[Mock()]
        pool.service=Mock();pool.jobs=Mock();pool.results=queue.Queue()
        pool.results.put(dict(work_complete=dict(token='foreign',kind='parent',job=0)))
        with self.assertRaisesRegex(ValueError,'foreign'):
            pool._collect_branches([packet(0)['jobs'][0]],[packet(0)],[],None)


if __name__=='__main__':unittest.main()
