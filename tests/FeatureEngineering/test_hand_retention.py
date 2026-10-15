"""Feature sidecars follow source pins and existing evaluation retention quotas."""
from pathlib import Path
import tempfile
import unittest

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import retention
from brass_ai.recovery.config import RETENTION
from brass_ai.step5.league import write_json


class FeatureSidecarRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'artifacts/run';self.run.mkdir(parents=True)
        self.cfg=dict(RETENTION,evaluation_samples=1,replay_rounds=2)

    def files(self,directory):
        directory.mkdir(parents=True,exist_ok=True);created=[]
        for name in ('a.jsonl.gz','b.jsonl.gz'):
            raw=directory/name;raw.write_bytes(b'raw fixture');created.append(raw)
            for version in ('','-v2','-v3'):
                for ext in ('npz','json'):
                    p=directory/(name+'.features'+version+'.'+ext);p.write_bytes(b'cache fixture');created.append(p)
        return created

    def league(self):
        write_json(self.run/'state.json',dict(format='step5-recovery-league/2.2.0',iteration=3,
            history=[],population=[],signature=dict(config=dict(retention=self.cfg))))

    def test_raw_pin_preserves_v1_v2_v3_sidecars_from_expired_iteration(self):
        self.league();files=self.files(self.run/'iteration-0000')
        for n in range(4):write_json(self.run/f'iteration-{n:04d}/report.json',{})
        write_json(self.run/'pins.json',dict(raw='artifacts/run/iteration-0000/b.jsonl.gz'))
        plan=retention.plan(self.root,self.run,self.cfg);deleted={e['path'] for e in plan['entries']}
        self.assertFalse(any(p.relative_to(self.run).as_posix() in deleted for p in files if p.name.startswith('b.')))
        self.assertTrue(any(p.relative_to(self.run).as_posix() in deleted for p in files if p.name.startswith('a.')))

    def test_retired_development_game_retires_every_known_feature_sidecar(self):
        files=self.files(self.run)
        write_json(self.run/'report.json',dict(format='step5-evaluation-report/1.0.0',
            spec=dict(config=dict(panel='development')),promotion_eligible=False))
        plan=retention.plan(self.root,self.run,self.cfg);deleted={e['path'] for e in plan['entries']}
        self.assertTrue(all(p.relative_to(self.run).as_posix() in deleted for p in files if p.name.startswith('b.')))
        self.assertFalse(any(p.relative_to(self.run).as_posix() in deleted for p in files if p.name.startswith('a.')))

    def test_retired_champion_game_retires_every_known_feature_sidecar(self):
        self.league();directory=self.run/'evaluations/champion-0000';files=self.files(directory)
        write_json(directory/'report.json',dict(format='four-player-champion/1.0.0'))
        plan=retention.plan(self.root,self.run,self.cfg);deleted={e['path'] for e in plan['entries']}
        self.assertTrue(all(p.relative_to(self.run).as_posix() in deleted for p in files if p.name.startswith('b.')))


if __name__=='__main__':unittest.main()
