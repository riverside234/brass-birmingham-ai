"""Compact rating evidence transfers without admitting any full replay."""
import copy
import json
import unittest

from test_native_transfer import NativeTransfer,ROOT,transfer
from brass_ai.recovery import ratings
from brass_ai.step5.league import seeds_for


class RatingTransfer(unittest.TestCase):
    def setUp(self):
        self.case=NativeTransfer();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.root,self.run=self.case.root,self.case.run
        self.configure('generation-000001')

    def configure(self,name):
        self.prefix='artifacts/step5/selfplay/ratings/slots/'+name
        config=dict(contract='rating-fixture/1',seed=411,setups=1,every_iterations=1,enabled=True)
        self.seed=seeds_for(config['seed'],1,'validation')[0][0]
        self.spec=dict(format='plackett-luce-slot/1.0.0',generation=1,
            lineups=[dict(kind='rating',seed=self.seed,seat=s,agents=['new','generation-0','v4','old']) for s in range(4)])
        self.rows=[dict(game_id='game-'+str(s),setup_id='same-setup',setup_seed=self.seed,slot=1,
            candidate_seat=s,kind='rating',agents=[self.spec['lineups'][s]['agents'][(p-s)%4] for p in range(4)],
            order=[0,1,2,3],official_placement=[1,2,3,4],vp=[200,150,100,50],win_share=[1.,0.,0.,0.],
            spec_digest=transfer.digest(self.spec)) for s in range(4)]
        slot=dict(generation=1)
        for key,name,value in [('spec','spec.json',self.spec),('report','report.json',dict(fit='already completed')),
                               ('panels','panels.json',{}),('bootstrap','bootstrap.npz',b'previous bootstrap bytes')]:
            p=self.case.put(self.prefix+'/'+name,value);slot[key]=self.case.pin(p)
        archive=self.case.put('artifacts/step5/selfplay/ratings/history.json',dict(format=ratings.ARCHIVE_FORMAT,slots=[],games=[]))
        self.rating=dict(contract=config['contract'],config=config,seeds=[self.seed],slots=[slot],agents={'generation-0':{},'v4':{}},
            latest_report=slot['report'],archive=self.case.pin(archive))
        self.case.state['ratings']=self.rating
        self.save_rows()

    def save_rows(self,*,repin=True):
        data=''.join(json.dumps(r,sort_keys=True,allow_nan=True)+'\n' for r in self.rows).encode()
        p=self.case.put(self.prefix+'/games.jsonl',data)
        if repin:self.rating['slots'][0]['records']=self.case.pin(p)
        self.case.put('artifacts/step5/selfplay/state.json',self.case.state)
        return p

    def test_export_and_relocated_restore_preserve_outcomes_and_rating_continuation(self):
        before=(self.run/'state.json').read_bytes();baseline=ratings.read_games(self.root,self.rating)
        preview=transfer.export(self.root,self.run,self.case.bundle,dry_run=True,archive_format='zip')
        self.assertTrue(preview['dry_run'])
        self.case.export();m=transfer.inspect_bundle(self.case.bundle);names={r['path'] for r in m['files']}
        self.assertIn(self.prefix+'/games.jsonl',names)
        self.assertNotIn('artifacts/step5/selfplay/iteration-0001/games.jsonl.gz',names)
        self.assertFalse(any('.prepared' in n or '/teacher-labels/' in n or '/search-values/' in n for n in names))
        destination=self.case.area/'new-machine';self.assertTrue(transfer.restore(destination,self.case.bundle,relocate=True)['verified'])
        restored=transfer.read(destination/'artifacts/step5/selfplay/state.json')
        self.assertEqual((destination/'artifacts/step5/selfplay/state.json').read_bytes(),before)
        ratings.validate_state(destination,restored['ratings'],self.rating['config'])
        self.assertEqual(ratings.read_games(destination,restored['ratings']),baseline)
        self.assertEqual((destination/self.prefix/'games.jsonl').read_bytes(),(self.root/self.prefix/'games.jsonl').read_bytes())

    def test_committed_source_suffixed_slot_paths_are_supported(self):
        self.configure('generation-000001-source-'+'a'*16)
        self.case.export();names={r['path'] for r in transfer.inspect_bundle(self.case.bundle)['files']}
        self.assertIn(self.prefix+'/games.jsonl',names)

    def test_unrelated_jsonl_or_disguised_full_replay_is_rejected(self):
        raw=self.case.put('artifacts/step5/selfplay/iteration-0001/forbidden.jsonl',b'full replay')
        self.case.put('artifacts/step5/selfplay/pins.json',dict(paths=[raw.relative_to(self.root).as_posix()]))
        with self.assertRaisesRegex(ValueError,'Replay/preparation data'):self.case.export()
        (self.run/'pins.json').unlink()
        self.rows[0]['records']=[dict(view='full action replay must not be exported')];self.save_rows()
        with self.assertRaisesRegex(ValueError,'unsupported/replay fields'):self.case.export()
        self.assertFalse(self.case.bundle.exists())

    def test_missing_or_changed_summary_still_fails_hash_and_dependency_checks(self):
        p=self.save_rows()
        # Valid JSON with changed whitespace must still fail the original pin.
        p.write_bytes(b' '+p.read_bytes())
        with self.assertRaisesRegex(ValueError,'Dependency hash differs'):self.case.export()
        self.save_rows();p.unlink()
        with self.assertRaisesRegex(ValueError,'Missing committed rating outcome'):self.case.export()

    def test_incomplete_duplicate_nonfinite_or_foreign_outcomes_cannot_be_approved(self):
        original=copy.deepcopy(self.rows)
        for name in ('missing','duplicate','nonfinite','foreign'):
            with self.subTest(name=name):
                self.rows=copy.deepcopy(original)
                if name=='missing':self.rows.pop()
                elif name=='duplicate':self.rows[1]=copy.deepcopy(self.rows[0])
                elif name=='nonfinite':self.rows[0]['vp'][0]=float('nan')
                else:self.rows[0]['spec_digest']='f'*64
                self.save_rows()
                with self.assertRaises(ValueError):self.case.export()
                self.assertFalse(self.case.bundle.exists())

    def test_only_exact_owned_committed_slot_record_can_use_the_exception(self):
        self.rating['slots'][0]['records']=self.case.pin(self.case.put('artifacts/step5/selfplay/ratings/not-a-slot/games.jsonl',b'{}\n'))
        self.case.put('artifacts/step5/selfplay/state.json',self.case.state)
        with self.assertRaisesRegex(ValueError,'Invalid committed rating summary path'):self.case.export()
        self.assertFalse(self.case.bundle.exists())

    def test_transport_repair_keeps_training_fingerprint_and_existing_reward_adapter(self):
        from brass_ai.learning.source_identity import implementation_id
        review=transfer.read(ROOT/'configs/learning/b07-replay-location-review.json')
        self.assertEqual(implementation_id(ROOT),review['execution_implementation'])


if __name__=='__main__':unittest.main()
