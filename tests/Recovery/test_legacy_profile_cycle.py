"""All five v4 training variants, absolute table ordinals and completed-seat reports."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import quality_pool
from brass_ai.recovery.curriculum import tables_before,training_table
from brass_ai.recovery.ordinary import plan_collection,sampling_report
from brass_ai.recovery.execution import neural_spec
from brass_ai.recovery.history import compact


class LegacyProfiles(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.config=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic.json')
        self.pool=read(SOURCE_ROOT/'training/opponents/pool.json')
        self.profiles=sorted(self.pool['training'])
        self.assertEqual(len(self.profiles),5)
        path=self.root/'weights.pt';path.write_bytes(b'planning only')
        self.fp=dict(values=[1/8]*16+[1/6]*12)
        entry=dict(model_id='IL',generation=0,path=path.relative_to(SOURCE_ROOT).as_posix(),
                   sha256=sha(path),fingerprint=self.fp)
        pool=quality_pool.initial(entry,self.config['opponent_pool'],dict(path='probe',sha256='a'*64,probe_hash='probe'))
        pool['members'][0]['fingerprint']=self.fp
        self.state=dict(iteration=0,games_done=0,next_seed=self.config['seed'],context_counter=0,
            population=[entry],history=[entry],evaluation_anchor=entry,quality_pool=pool)
        self.fast={p:dict(kind='lab-legacy',policy_id='fast:'+p,search_free=True) for p in self.profiles}
        self.slow={p:dict(kind='lab-legacy',policy_id='searched:'+p,search_free=False) for p in self.profiles}

    def plan(self,state=None,config=None,profiles=None):
        service=Mock();service.models={'IL':None}
        return plan_collection(SOURCE_ROOT,self.root,state or self.state,config or self.config,
            dict(config=self.config),service,'current',profiles or self.profiles,self.slow,self.fast,None,
            fingerprints=dict(learner=self.fp,rule_bot=self.fp))

    def test_320_games_use_all_five_and_keep_pottery_heldout(self):
        before=copy.deepcopy(self.state);rng=random.getstate();plan=self.plan()
        self.assertEqual(self.state,before);self.assertEqual(random.getstate(),rng)
        self.assertEqual(plan['composition'],dict(selfplay=160,historical=144,legacy=16))
        counts=Counter();old=Counter();ordinals=[]
        for job in plan['jobs']:
            if job['provenance']['kind']!='legacy':continue
            physical=[job['policies'][(s-job['rotation'])%4] for s in range(4)]
            receipt=job['provenance']['legacy_profiles'];ordinals.append(receipt['table_ordinal'])
            self.assertEqual(len(receipt['seats']),2);self.assertEqual(len(set(receipt['seats'].values())),2)
            for seat,profile in receipt['seats'].items():
                self.assertEqual(physical[int(seat)],self.fast[profile]);counts[profile]+=1
            index=job['provenance']['ordinary_game_index']
            old.update(self.profiles[(index*3+s)%5] for s in range(2))
        self.assertEqual(ordinals,list(range(16)));self.assertEqual(set(counts),set(self.profiles))
        self.assertEqual(sorted(counts.values()),[6,6,6,7,7])
        self.assertEqual({p.split('/')[-1].split('-v3')[0] for p in old},{'cotton','goods-beer'})
        self.assertTrue(all('pottery-flex' not in p for p in counts))
        self.assertTrue(all('pottery-flex' in p for p in self.pool['evaluation']))
        with self.assertRaisesRegex(ValueError,'evaluation-only'):
            self.plan(profiles=self.profiles+list(self.pool['evaluation']))

    def test_restart_and_irregular_chunks_preserve_profile_order_and_physical_seats(self):
        full=self.plan();self.assertEqual(full,self.plan(json.loads(json.dumps(self.state))))
        state=copy.deepcopy(self.state);jobs=[]
        for size in (13,109,1,197):
            cfg=dict(self.config,games_per_iteration=size);part=self.plan(state,cfg);jobs.extend(part['jobs'])
            state.update(games_done=state['games_done']+size,next_seed=part['cursor'],context_counter=part['context_counter'])
        for a,b in zip(full['jobs'],jobs):
            for field in ('seed','policies','rotation','sampling_seed','focuses','provenance'):
                self.assertEqual(a[field],b[field])
        resumed=dict(state,games_done=320);next_plan=self.plan(resumed)
        legacy=next(j for j in next_plan['jobs'] if j['provenance']['kind']=='legacy')
        self.assertEqual(legacy['provenance']['legacy_profiles']['table_ordinal'],16)
        # 30 legacy tables cover all six learner pairs and five variant offsets.
        expanded=self.plan(config=dict(self.config,games_per_iteration=600));seats={p:Counter() for p in self.profiles}
        for job in expanded['jobs']:
            for s,p in job['provenance'].get('legacy_profiles',{}).get('seats',{}).items():seats[p][s]+=1
        self.assertTrue(all(sorted(c.values())==[3]*4 for c in seats.values()))

    def test_ordinal_matches_brute_force_at_curriculum_and_boost_boundaries(self):
        cfg=copy.deepcopy(self.config)
        cfg.pop('selfplay_recipe')
        cfg['curriculum']=[dict(until=37,selfplay=60,historical=30,legacy=10),
                           dict(until=73,selfplay=70,historical=25,legacy=5),
                           dict(until=None,selfplay=50,historical=45,legacy=5)]
        cfg['legacy_boost']=dict(stages=[dict(games=31,selfplay=40,historical=20,legacy=40),
                                         dict(games=18,selfplay=50,historical=30,legacy=20)])
        for window in (None,dict(start_games=11,until_games=60)):
            for n in (0,1,11,12,19,20,21,36,37,38,42,59,60,61,72,73,74,127,320):
                expected=Counter(training_table(i,cfg,window)[0] for i in range(n))
                for kind in ('selfplay','historical','legacy'):
                    self.assertEqual(tables_before(n,kind,cfg,window),expected[kind])

    def test_completed_seat_counts_survive_compaction(self):
        plan=self.plan();games=[];data=[];learner=neural_spec('current',.08,'mixed')['policy_id']
        for n,job in enumerate(plan['jobs']):
            setup=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
            policies=[job['policies'][(s-job['rotation'])%4]['policy_id'] for s in range(4)]
            game=dict(game_id=str(n),setup_group=setup,opponents=policies);games.append(game)
            data.extend((dict(behavior=dict(policy_id=learner,kind='model'),choices=[{},{}]),None,(str(n),s),None)
                        for s,p in enumerate(policies) if p==learner)
        report=sampling_report(games,data,learner,plan['kind_by_group'],self.config,
            legacy_profiles={spec['policy_id']:p for p,spec in self.fast.items()})
        usage=report['legacy_profiles'];self.assertEqual(usage['opponent_seats'],32)
        self.assertEqual(sorted(p['seats'] for p in usage['profiles'].values()),[6,6,6,7,7])
        for profile in usage['profiles'].values():self.assertEqual(profile['seats'],sum(profile['seat_counts']))
        self.assertEqual(compact(dict(sampling=report))['sampling']['legacy_profiles'],usage)


if __name__=='__main__':unittest.main()
