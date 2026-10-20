"""Rolling 100-iteration rating reports preserve cumulative evidence and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ratings as r,retention,plackett_luce as pl
from brass_ai.recovery.config import RETENTION
from brass_ai.recovery.quality_config import RATING
from brass_ai.step5.league import write_json
import test_strength_rating as fixtures


class RatingRetention(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.run=self.root/'artifacts/run'
        self.c=dict(RATING,setups=2,panel_setups=1,bootstrap=4)
        self.cfg=dict(rating=self.c,host='unused',execution={},retention=dict(RETENTION,quarantine_hours=0))
        self.state=fixtures.Slots().state(self.root,self.c);self.state['signature']=dict(config=self.cfg)
        for item in (
            patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,old=None:e),
            patch.object(r,'collect_slot',side_effect=lambda root,directory,spec,*args:fixtures.fixture_rows(spec)),
            patch.object(r,'plot_report',side_effect=lambda path,*args:path.write_text('<svg/>'))):
            item.start();self.addCleanup(item.stop)

    def advance(self,generation):
        if generation!=10:
            path=self.root/f'candidate-{generation}.pt';path.write_bytes(str(generation).encode())
            self.state['history'].append(dict(generation=generation,path=path.name,sha256=sha(path),model_id=f'model-{generation}'))
        self.state['iteration']=generation
        return r.schedule(self.root,self.run,self.state,self.cfg)

    def through(self,generation):
        for g in range(10,generation+1,10):self.advance(g)

    def cleanup(self):return retention.prune(self.root,self.run,self.cfg['retention'],apply=True)

    def test_100_iteration_boundary_removes_full_slots_but_keeps_exact_cumulative_fit(self):
        self.through(100)
        self.assertNotIn('archive',self.state['ratings'])
        self.state['iteration']=109;write_json(self.run/'state.json',self.state)
        self.assertIsNone(r.schedule(self.root,self.run,self.state,self.cfg))
        self.assertEqual(len(self.state['ratings']['slots']),10)
        self.cfg['retention']['enabled']=False
        fitted=self.advance(110)
        before_games=r.read_games(self.root,self.state['ratings']);before_panels=r.panel_history(self.root,self.state['ratings'])
        self.cfg['retention']['enabled']=True
        write_json(self.run/'state.json',self.state)
        r.compact_history(self.root,self.run,self.state)
        self.assertEqual([s['generation'] for s in self.state['ratings']['slots']],list(range(20,111,10)))
        decision=self.cleanup()
        self.assertFalse((self.run/'ratings/slots/generation-000010').exists())
        self.assertEqual(len(list((self.run/'ratings/slots').glob('generation-*'))),10)
        self.assertEqual(r.read_games(self.root,self.state['ratings']),before_games)
        self.assertEqual(r.panel_history(self.root,self.state['ratings']),before_panels)
        r.validate_state(self.root,self.state['ratings'],self.c)
        actual,samples=pl.estimate([g for g in before_games if g['kind']=='rating'],self.state['ratings']['agents'],self.c)
        self.assertEqual(actual['ratings'],fitted['ratings'])
        with np.load(self.root/self.state['ratings']['slots'][-1]['bootstrap']['path']) as saved:
            np.testing.assert_array_equal(samples,saved['theta'])
        self.assertEqual(actual['games'],88)
        self.assertGreater(decision['permanently_deleted_files'],0)
        self.assertTrue(all((self.root/a['checkpoint']['path']).exists() for a in self.state['ratings']['agents'].values() if a['kind']=='model'))

    def test_resume_skips_archived_slots_future_fit_and_archive_supersession(self):
        self.through(110);self.cleanup()
        original_archive=self.root/self.state['ratings']['archive']['path']
        self.state=read(self.run/'state.json')
        with patch.object(r,'collect_slot',side_effect=AssertionError('Must not recollect committed evidence')):
            self.assertIsNone(r.schedule(self.root,self.run,self.state,self.cfg))
        report=self.advance(120);self.cleanup()
        self.assertEqual(report['games'],96)
        self.assertEqual([s['generation'] for s in r.archived_history(self.root,self.state['ratings'])['slots']],[10,20])
        self.assertFalse(original_archive.exists())
        self.assertEqual(len(list((self.run/'ratings').glob('history-*.json'))),1)
        r.validate_state(self.root,self.state['ratings'],self.c)
        self.assertEqual(len(list((self.run/'ratings/slots').glob('generation-*'))),10)

    def test_interrupted_archive_publication_preserves_old_state_and_source_files(self):
        self.cfg['retention']['enabled']=False;self.through(110)
        before=copy.deepcopy(self.state);state_bytes=(self.run/'state.json').read_bytes()
        source={p:p.read_bytes() for p in (self.run/'ratings/slots').rglob('*') if p.is_file()}
        with patch.object(r,'write_json',side_effect=OSError('interrupted state commit')):
            with self.assertRaisesRegex(OSError,'interrupted'):r.compact_history(self.root,self.run,self.state)
        self.assertEqual(self.state,before);self.assertEqual((self.run/'state.json').read_bytes(),state_bytes)
        self.assertTrue(all(p.read_bytes()==raw for p,raw in source.items()))
        r.validate_state(self.root,self.state['ratings'],self.c)
        self.cleanup()
        self.assertEqual(list((self.run/'ratings').glob('history-*.json')),[])
        r.compact_history(self.root,self.run,self.state);self.cleanup()
        self.assertEqual(len(self.state['ratings']['slots']),10)

    def test_tampered_archive_is_rejected_before_cleanup_or_fit(self):
        self.through(110);self.cleanup()
        path=self.root/self.state['ratings']['archive']['path'];path.write_bytes(path.read_bytes()+b' ')
        for operation in (
            lambda:r.validate_state(self.root,self.state['ratings'],self.c),
            lambda:r.read_games(self.root,self.state['ratings']),
            lambda:retention.plan(self.root,self.run,self.cfg['retention'])):
            with self.assertRaisesRegex(ValueError,'Archived rating evidence changed'):operation()

    def test_explicit_pins_keep_required_old_files_and_unknown_files_stay_local(self):
        self.through(110)
        old=self.run/'ratings/slots/generation-000010'
        pinned=old/'bootstrap.npz';unknown=old/'my-notes.txt';unknown.write_text('keep')
        write_json(self.run/'pins.json',dict(required=pinned.relative_to(self.root).as_posix()))
        self.cleanup()
        self.assertTrue(pinned.exists());self.assertTrue(unknown.exists())
        self.assertFalse((old/'games.jsonl').exists())
        self.assertEqual(len(r.read_games(self.root,self.state['ratings'])),176)

    def test_compaction_rejects_stale_state_and_receipts_outside_the_run(self):
        self.cfg['retention']['enabled']=False;self.through(110)
        self.state['iteration']=111
        with self.assertRaisesRegex(ValueError,'Run advanced'):r.compact_history(self.root,self.run,self.state)
        self.state=read(self.run/'state.json')
        outside=self.root/'outside.json';outside.write_text('{}')
        self.state['ratings']['slots'][0]['spec']=dict(path='outside.json',sha256=sha(outside))
        write_json(self.run/'state.json',self.state)
        with self.assertRaisesRegex(ValueError,'escapes owned run'):r.compact_history(self.root,self.run,self.state)
        self.assertEqual(outside.read_text(),'{}')

    def test_live_GAE_rating_tuples_compact_and_resume_without_recollecting(self):
        self.through(100)
        # gae.rating_evidence returns pairs as tuples in the live trainer.
        self.state['recent_training']=[dict(actor=dict(gae=dict(rating=dict(
            points=[(80,346.08),(90,372.29),(100,341.52)]))))]
        self.assertNotEqual(self.state,json.loads(json.dumps(self.state)))
        self.advance(110)
        self.assertEqual([s['generation'] for s in self.state['ratings']['slots']],list(range(20,111,10)))
        saved=read(self.run/'state.json')
        with patch.object(r,'collect_slot',side_effect=AssertionError('Committed rating must be reused')):
            self.assertIsNone(r.schedule(self.root,self.run,saved,self.cfg))
        # Real changes, even within the same generation, must still fail.
        self.cfg['retention']['enabled']=False;self.advance(120)
        self.state['recent_training'][0]['actor']['gae']['rating']['points'][0]=(80,999.)
        with self.assertRaisesRegex(ValueError,'Run advanced'):
            r.compact_history(self.root,self.run,self.state)


if __name__=='__main__':unittest.main()
