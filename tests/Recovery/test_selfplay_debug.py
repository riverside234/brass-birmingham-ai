"""Interrupted rating slots across reviewed fixes preserve evidence and resume."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import ratings as r,retention
from brass_ai.recovery.config import RETENTION
from brass_ai.recovery.quality_config import RATING
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id,write_json
import test_strength_rating as fixtures


EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0540-20261022T144917069131Z'


class InterruptedRating(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.run=self.root/'artifacts/run'
        self.c=dict(RATING,setups=2,panel_setups=1,bootstrap=4)
        self.cfg=dict(rating=self.c,host='unused',execution={},retention=dict(RETENTION,quarantine_hours=0))
        self.state=fixtures.Slots().state(self.root,self.c)
        old=dict(read(EXPORT/'state.json')['signature'],config=copy.deepcopy(self.cfg))
        current=dict(old,implementation=implementation_id())
        self.state.update(signature=current,source_transitions=[dict(from_signature=old,to_signature=digest(current))])
        self.old=old
        for item in (
            patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,old=None:e),
            patch.object(r,'collect_slot',side_effect=lambda root,directory,spec,*args:fixtures.fixture_rows(spec)),
            patch.object(r,'plot_report',side_effect=lambda path,*args:path.write_text('<svg/>'))):
            item.start();self.addCleanup(item.stop)

    def pending(self,implementation=None):
        directory=self.run/'ratings/slots/generation-000010'
        spec=r.slot_spec(self.root,self.run,self.state['ratings'],self.state['history'][-1],self.cfg)
        spec['implementation']=implementation or self.old['implementation']
        r.write_once(directory/'spec.json',spec)
        (directory/'game-000000.jsonl.gz').write_bytes(b'old partial evidence must survive')
        return directory,{p.name:p.read_bytes() for p in directory.iterdir()}

    def test_reviewed_fix_restarts_pending_slot_in_separate_namespace(self):
        old,before=self.pending()
        fitted=r.schedule(self.root,self.run,self.state,self.cfg)
        slot=self.state['ratings']['slots'][0]
        selected=(self.root/slot['spec']['path']).parent
        self.assertEqual(selected.name,old.name+'-source-'+implementation_id()[:16])
        self.assertEqual(fitted['games'],8)
        self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
        with patch.object(r,'collect_slot',side_effect=AssertionError('Committed evidence recollected')):
            self.assertIsNone(r.schedule(self.root,self.run,self.state,self.cfg))
        r.validate_state(self.root,self.state['ratings'],self.c)

    def test_unknown_source_cannot_bypass_immutable_spec(self):
        old,before=self.pending('not-a-reviewed-source')
        with self.assertRaisesRegex(ValueError,'reviewed transition history'):
            r.schedule(self.root,self.run,self.state,self.cfg)
        self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)

    def test_same_source_changed_recipe_still_fails(self):
        old,_=self.pending(implementation_id())
        spec=read(old/'spec.json');spec['temperature']=.7
        (old/'spec.json').unlink();r.write_once(old/'spec.json',spec)
        with self.assertRaisesRegex(ValueError,'Immutable rating file changed'):
            r.schedule(self.root,self.run,self.state,self.cfg)

    def test_source_slot_compaction_and_retention_follow_actual_receipts(self):
        old,before=self.pending();r.schedule(self.root,self.run,self.state,self.cfg)
        slot=copy.deepcopy(self.state['ratings']['slots'][0]);directory=(self.root/slot['spec']['path']).parent
        unknown=directory/'my-notes.txt';unknown.write_text('keep')
        pinned=directory/'bootstrap.npz'
        write_json(self.run/'pins.json',dict(required=pinned.relative_to(self.root).as_posix()))
        path=self.root/'candidate-20.pt';path.write_bytes(b'next')
        self.state['history'].append(dict(generation=20,path=path.name,sha256=sha(path),model_id='model-20'))
        self.state['iteration']=20;r.schedule(self.root,self.run,self.state,self.cfg)
        self.state['iteration']=110;write_json(self.run/'state.json',self.state)
        r.compact_history(self.root,self.run,self.state)
        self.assertEqual(len(r.read_games(self.root,self.state['ratings'])),32)
        self.assertEqual(r.archived_history(self.root,self.state['ratings'])['slots'][0]['directory'],
            directory.relative_to(self.run).as_posix())
        retention.prune(self.root,self.run,self.cfg['retention'],apply=True)
        self.assertFalse((directory/'report.json').exists())
        self.assertTrue(pinned.exists());self.assertEqual(unknown.read_text(),'keep')
        self.assertEqual({p.name:p.read_bytes() for p in old.iterdir()},before)
        self.assertEqual(len(r.read_games(self.root,self.state['ratings'])),32)

    def test_archived_slot_paths_are_relative_and_confined_to_their_generation(self):
        canonical=self.run/'ratings/slots/generation-000010'
        self.assertEqual(r.archived_slot_directory(self.run,dict(generation=10)),canonical)
        for path in ('ratings/slots/generation-000020','ratings/slots/my-notes',
                     'ratings/slots/generation-000010/nested',
                     'ratings/slots/generation-000010-source-unknown',str(canonical)):
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'Invalid archived rating'):
                r.archived_slot_directory(self.run,dict(generation=10,directory=path))

    def test_actual_g540_and_already_adopted_one_se_source_resume_without_recipe_changes(self):
        state=read(EXPORT/'state.json');before=copy.deepcopy(state)
        config=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-one-se-g540.json')
        for source,old in ((state['signature']['implementation'],state['signature']['config']),
                           ('056449b7e5177379e545cb499da91a870d22ebe033fa6827ba3b5beac214d534',config)):
            prior=dict(state,signature=dict(state['signature'],implementation=source,config=old))
            target=dict(prior['signature'],implementation=implementation_id(),config=copy.deepcopy(config))
            self.assertTrue(check_signature(prior,target,reviewed=True))
            with self.assertRaises(ValueError):check_signature(prior,target)
            target['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(prior,target,reviewed=True)
        self.assertEqual(state,before)


if __name__=='__main__':unittest.main()
