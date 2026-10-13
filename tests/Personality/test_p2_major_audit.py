"""Bounded P2 admission and calibration regressions; authored evidence only."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from personality_fixtures import parent
import test_search
from brass_ai.bank import sha
from brass_ai.learning.batching import model_id as base_id
from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json
from brass_ai.learning.replay import split_for
from brass_ai.learning.reward import specification
from brass_ai.personality import checkpoint
from brass_ai.personality.contracts import context
from brass_ai.personality.model import import_parent
from brass_ai.personality.search import collection, readiness
from brass_ai.personality.search.contracts import READY, Profile, controller, identity, search_config
from brass_ai.personality.search.inference import SharedInference
from brass_ai.step5.league import write_json


class P2MajorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary_root = SOURCE_ROOT/'artifacts/p0p6-audit-20261013/tmp'
        cls.temporary_root.mkdir(parents=True, exist_ok=True)

    def collection_inputs(self):
        incumbent = parent(); model, _ = import_parent(incumbent)
        cfg = search_config(specification())
        ready = dict(id='authored-ready', model=dict(model_id=checkpoint.model_id(model)),
                     reward=cfg.reward, goal_calibration=dict(bias_ready=False))
        execution = dict(read_json(SOURCE_ROOT/'configs/step5/lab.json')['execution'], shared_slot_bytes=0)
        handoff = dict(signature=dict(config=dict(execution=execution)))
        return (model, [incumbent], ready, handoff, digest([base_id(incumbent)])), cfg

    def test_partial_collection_channel_allocation_closes_service(self):
        loaded, cfg = self.collection_inputs(); execution = loaded[3]['signature']['config']['execution']
        service = SharedInference(execution); allocate = service._channel; count = 0

        def fail():
            nonlocal count
            count += 1
            if count == 1:
                return allocate()
            # Drain the first queued handle before injecting failure, so the
            # Windows feeder is not still serializing it during teardown.
            service.leases[-1].get(timeout=5)[0].close()
            raise OSError('injected P2 partial channel allocation')

        try:
            with tempfile.TemporaryDirectory(dir=self.temporary_root) as tmp, \
                    patch.object(collection, 'load', return_value=loaded), \
                    patch.object(collection, 'SharedInference', return_value=service), \
                    patch.object(service, '_channel', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'partial channel allocation'):
                    collection.collect(SOURCE_ROOT, Path(tmp), Path(tmp)/'collection', [dict(index=0)], cfg, tables=1)
            self.assertTrue(service.connections, 'The first real pipe must have been allocated')
            self.assertTrue(service.closed)
            self.assertTrue(all(c.closed for c in service.connections.values()))
        finally:
            service.__exit__()

    def test_opponent_endpoint_failure_closes_constructed_search_session(self):
        loaded, cfg = self.collection_inputs(); execution = loaded[3]['signature']['config']['execution']
        service = SharedInference(execution); session = Mock(); lane = Mock()
        lane.acquire.side_effect = OSError('injected opponent endpoint failure')
        try:
            with tempfile.TemporaryDirectory(dir=self.temporary_root) as tmp, \
                    patch.object(collection, 'load', return_value=loaded), \
                    patch.object(collection, 'SharedInference', return_value=service), \
                    patch.object(service, 'channels', return_value=lane), \
                    patch.object(collection, 'SearchSession', return_value=session):
                with self.assertRaisesRegex(OSError, 'opponent endpoint'):
                    collection.collect(SOURCE_ROOT, Path(tmp), Path(tmp)/'collection', [dict(index=0)], cfg, tables=1)
            session.close.assert_called_once_with()
            self.assertTrue(service.closed)
        finally:
            service.__exit__()

    def test_collection_rejects_teacher_from_a_different_executed_context(self):
        label, row, ident, cfg = test_search.SearchTests().teacher_fixture()
        row['step'] = 0; seed = 42
        setup = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed)); profile = Profile()
        candidate = controller(ident, 'parent', profile, cfg, 'pool')['candidate_id']
        job = dict(index=0, seed=seed, partition=split_for(setup), neural_seats=[0],
                   profiles=[profile.to_dict()]*4, controllers=[candidate]*4)
        spec = dict(jobs=[job], model_id=ident, parent_incumbent_id='parent', pool_id='pool',
                    decision_mode='searched', search=cfg.to_dict())
        game = dict(game_id='authored-game', setup_group=setup, split=job['partition'], records=[row])
        recorded = dict(step=0, model_id=ident, candidate_id=candidate, context=context(), goal=label['goal'], teacher=label)
        side = dict(game_id=game['game_id'], spec_id=digest(spec), job=job, records=[recorded])
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as tmp:
            directory = Path(tmp); replay = directory/'replay'; replay.write_bytes(b'authored replay boundary')
            side_path = directory/'context.json'; write_json(side_path, side)
            item = dict(path=replay.name, context=side_path.name, sha256=sha(replay), context_sha256=sha(side_path),
                        game_id=game['game_id'], job=0)
            manifest = dict(spec=spec, spec_id=digest(spec), records=[item])
            with patch.object(collection, 'read_shards', return_value=[game]):
                self.assertEqual(len(list(collection.iter_pairs(directory, manifest))), 1)
                for field in ('personality', 'goal', 'profile', 'search'):
                    with self.subTest(field=field):
                        changed = copy.deepcopy(side)
                        target = changed['records'][0]['teacher']
                        if field == 'personality':
                            target[field] = context('networker')
                        elif field == 'goal':
                            target[field] = copy.deepcopy(target[field])
                            target[field]['age'] = 1
                        elif field == 'profile':
                            target[field]['alpha'] = .3
                        else:
                            target['target']['search']['seconds'] = 5.
                            t = target['target']
                            t['fingerprint'] = digest(dict(model=t['model_id'], config=t['search'],
                                contract='masked-ismcts/'+t['source'].rsplit('/', 1)[1], source=t['source_fingerprint']))
                        target['fingerprint'] = digest({k:v for k,v in target.items() if k!='fingerprint'})
                        write_json(side_path, changed); item['context_sha256'] = sha(side_path)
                        with self.assertRaisesRegex(ValueError, 'Teacher.*(context|recipe)'):
                            list(collection.iter_pairs(directory, manifest))

    def test_readiness_calibration_must_match_saved_model_training_receipt(self):
        incumbent = parent(); model, _ = import_parent(incumbent)
        goal = dict(temperature=1., brier=.01, ece=.01, labels=72, setups=2,
                    bias_ready=True, scope='authored calibration fixture')
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as tmp:
            directory = Path(tmp); bundle = directory/'handoff'; bundle.mkdir()
            write_json(bundle/'manifest.json', dict(fixture=True))
            ident = checkpoint.save(directory/'model.pt', model, training=dict(phase='p2-readiness', goal_calibration=goal))
            receipt = dict(format=READY, contract=identity(), model=dict(path='model.pt', sha256=sha(directory/'model.pt'), model_id=ident),
                           data=[], handoff=bundle.relative_to(SOURCE_ROOT).as_posix(), handoff_sha256=sha(bundle/'manifest.json'),
                           reward=model.core.config.utility, utility_ready=True, goal_calibration=goal)
            receipt['id'] = digest(receipt); write_json(directory/'ready.json', receipt)
            with patch.object(readiness.handoff, 'load', return_value=(dict(reward=specification(), references=[]), incumbent)):
                self.assertEqual(readiness.load(SOURCE_ROOT, directory)[2]['goal_calibration'], goal)
                receipt['goal_calibration'] = dict(goal, temperature=.5)
                receipt['id'] = digest({k:v for k,v in receipt.items() if k!='id'}); write_json(directory/'ready.json', receipt)
                with self.assertRaisesRegex(ValueError, 'Goal calibration.*model'):
                    readiness.load(SOURCE_ROOT, directory)

    def test_masked_out_setup_cannot_qualify_goal_calibration(self):
        labels = [dict(confidence=[[0.]*3]*6, confidence_mask=[[True]*3]*6) for _ in range(4)]
        labels.append(dict(confidence=[[0.]*3]*6, confidence_mask=[[False]*3]*6))
        result = readiness.confidence_calibration(np.full((5, 6, 3), -10.), labels, ['observed']*4+['masked-out'])
        self.assertEqual(result['labels'], 72)
        self.assertEqual(result['setups'], 1)
        self.assertFalse(result['bias_ready'])

    def test_goal_calibration_rejects_invalid_numerics_and_unsupported_readiness(self):
        valid = dict(temperature=1., brier=.01, ece=.01, labels=72, setups=2, bias_ready=True, scope='authored')
        self.assertEqual(readiness.validate_goal_calibration(valid), valid)
        for values in (dict(temperature=0.), dict(temperature=-1.), dict(temperature=float('nan')),
                       dict(ece=float('inf')), dict(brier=-.01), dict(labels=63), dict(setups=1),
                       dict(bias_ready=1), dict(labels=True), dict(ece=.11)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                readiness.validate_goal_calibration(dict(valid, **values))
        # A probe with no observed confidence labels stays valid but unready.
        labels = [dict(confidence=[[0.]*3]*6, confidence_mask=[[False]*3]*6)]
        empty = readiness.confidence_calibration(np.zeros((1, 6, 3)), labels, ['masked'])
        self.assertEqual((empty['labels'], empty['setups'], empty['ece'], empty['bias_ready']), (0, 0, None, False))


if __name__ == '__main__':
    unittest.main()
