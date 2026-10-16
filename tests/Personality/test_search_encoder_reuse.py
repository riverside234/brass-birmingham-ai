"""Deployment forests share immutable features, never predictions or builders."""
import copy
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.encoding_budget import EncodingStopped
from brass_ai.learning.features import Encoder
from brass_ai.learning.reward import specification
from brass_ai.personality.search.contracts import Profile, search_config
from brass_ai.personality.search.engine import SearchSession
from brass_ai.recovery.inference import Endpoint
from fe_fixtures import Factory


class Channels:
    def __init__(self): self.endpoints=[]
    def acquire(self):
        endpoint=Endpoint(Mock(),Mock(),len(self.endpoints))
        self.endpoints.append(endpoint)
        return endpoint


def worker(root,config,base,opponents,mono,cancelled,index):
    return SimpleNamespace(base=base,rollouts=opponents,game=None,close=lambda:None)


class SearchEncoderReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        factory=Factory()
        try: cls.view,cls.choices=factory.position()
        finally: factory.close()

    def session(self,workers=3,enabled=True):
        channels=Channels()
        with patch('brass_ai.personality.search.engine.Worker',side_effect=worker):
            session=SearchSession(SOURCE_ROOT,SimpleNamespace(cpu=None),channels,'actor', ['parent'],
                search_config(specification(),workers=workers),[Profile(alpha=0.)]*4,{},search_enabled=enabled)
        self.addCleanup(session.close)
        self.addCleanup(lambda:[endpoint.close() for endpoint in channels.endpoints])
        return session,channels

    def test_coordinator_workers_and_parents_share_completed_features_only(self):
        session,channels=self.session()
        clients=[session.base.evaluator]
        for w in session.workers: clients.extend([w.base.evaluator,*[p.evaluator for p in w.rollouts]])
        self.assertEqual(len(channels.endpoints),7)
        self.assertEqual(session.encoding_cache.capacity,7*256)
        self.assertEqual(len({id(c.encoder) for c in clients}),7)
        self.assertEqual(len({id(c.encoder.derived) for c in clients}),7)
        self.assertEqual(len({id(c.endpoint) for c in clients}),7)
        expected=Encoder().encode(self.view,self.choices,[0.]*4,[0.]*8)
        first=None
        for client in clients:
            got=client.encoder.encode(copy.deepcopy(self.view),copy.deepcopy(self.choices),[0.]*4,[0.]*8)
            if first is None:first=got
            self.assertIs(got,first)
            for field in ('state','actions','stable_to_relative'):
                np.testing.assert_array_equal(getattr(got,field),getattr(expected,field))
                self.assertFalse(getattr(got,field).flags.writeable)
        self.assertEqual(session.encoding_cache.report()['warm_hit'],6)
        self.assertEqual(session.encoding_cache.report()['miss'],1)
        another,_=self.session()
        self.assertIsNot(another.encoding_cache,session.encoding_cache)

    def test_context_mutations_and_stopped_requests_never_reuse_stale_features(self):
        session,_=self.session(2)
        a=session.base.evaluator.encoder;b=session.workers[0].base.evaluator.encoder
        saved=a.encode(self.view,self.choices)
        changed=copy.deepcopy(self.view);changed['players'][changed['seat']]['cash']+=1
        self.assertIsNot(b.encode(changed,self.choices),saved)
        self.assertIsNot(b.encode(self.view,self.choices,[.5,0,0,0]),saved)
        self.assertIsNot(b.encode(self.view,list(reversed(self.choices))),saved)
        stop=Event();stop.set()
        with self.assertRaises(EncodingStopped):b.encode(self.view,self.choices,cancelled=stop)
        with self.assertRaises(EncodingStopped):b.encode(self.view,self.choices,deadline=time.monotonic()-1)
        self.assertIs(b.encode(self.view,self.choices),saved)

    def test_direct_mode_keeps_only_one_lane_and_one_bounded_cache(self):
        session,channels=self.session(enabled=False)
        self.assertEqual(len(channels.endpoints),1)
        self.assertFalse(session.workers)
        self.assertEqual(session.encoding_cache.capacity,256)


if __name__=='__main__':unittest.main()
