"""P6 package/certificate integrity regressions; no installed game mutations."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_difficulty import named
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.personality.difficulty import contracts as difficulty,ui
from brass_ai.personality.laptop import contracts,export,install,package
from brass_ai.personality.search.contracts import READY,identity as search_identity


class PackageAuditTests(unittest.TestCase):
    def fixture(self,directory):
        roster=named(16,8);references=['parent-a','parent-b']
        roster['pool_id']=digest(references)
        profiles=difficulty.default_profiles();reward=roster['search']['reward']
        calibration=dict(temperature=1.,brier=.2,ece=.05,labels=64,setups=2,bias_ready=True,scope='Authored fixture only')
        ready=contracts.seal(dict(format=READY,contract=search_identity(),
            model=dict(model_id=roster['shared_model']['model_id']),reward=reward,
            utility_ready=True,goal_calibration=calibration))
        roster['shared_model']['ready_id']=ready['id']
        roster=contracts.seal({k:v for k,v in roster.items() if k!='id'})
        compact=dict(id=roster['id'],archetypes=[{k:g[k] for k in ('id','name','description','members')} for g in roster['archetypes']],
                     members=[{k:m[k] for k in ('id','archetype_id','profile')} for m in roster['members']])
        graphs=[dict(path=f'models/{i}.onnx',model_id=ident,actor=i==0)
                for i,ident in enumerate([roster['shared_model']['model_id'],*references])]
        files=[dict(path=g['path'],sha256=str(i)*64,bytes=1) for i,g in enumerate(graphs)]
        manifest=dict(format=contracts.PACKAGE,contract=contracts.identity(),fixture=True,release_qualified=False,
            roster=compact,p4_id=roster['id'],profiles=profiles,reward=reward,pool_id=roster['pool_id'],models=graphs,
            calibration=calibration,qualified_sources=False,files=files,
            searches={t:difficulty.resolve_search(roster,profiles,t).to_dict() for t in difficulty.TIERS})
        documents={'manifest.json':contracts.seal(manifest),'roster.json':roster,'qualification.json':None,
            'readiness.json':ready,'profiles.json':profiles,'public-menu.json':ui.menu(roster,fixture=True)}
        for g,f in zip(graphs,files):
            documents[str(Path(g['path']).with_suffix('.json')).replace('\\','/')]=dict(
                format=contracts.EXPORT,actor=g['actor'],model_id=g['model_id'],sha256=f['sha256'],
                compatibility=compatibility(),precision='float32',opset=17,
                inputs=export.BASE_INPUTS+(['personality'] if g['actor'] else []),
                outputs=export.BASE_OUTPUTS+(export.GOAL_OUTPUTS if g['actor'] else []),
                config=dict(utility=reward),reward_id=digest(reward),parity=dict(passed=True))
        return documents,files

    def test_reference_model_order_and_roster_pool_are_bound(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);original,files=self.fixture(directory)
            def validate(documents):
                with patch.object(package,'read',side_effect=lambda path:documents[Path(path).relative_to(directory).as_posix()]),\
                     patch.object(package,'inventory',return_value=files):
                    return package.validate(directory)
            self.assertEqual(validate(original)['pool_id'],original['roster.json']['pool_id'])
            for change in ('order','substitution','pool'):
                documents=copy.deepcopy(original);m=documents['manifest.json']
                if change=='order':m['models'][1:]=reversed(m['models'][1:])
                elif change=='substitution':
                    m['models'][1]['model_id']='unqualified-parent'
                    documents['models/1.json']['model_id']='unqualified-parent'
                else:m['pool_id']='unqualified-pool'
                documents['manifest.json']=contracts.seal({k:v for k,v in m.items() if k!='id'})
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'rollout pool'):
                    validate(documents)


class CertificationAuditTests(unittest.TestCase):
    def evidence(self):
        rows=[dict(tier=tier,era=era,seed=seed,status='ok',interrupt=None,
            available_kinds=['build','develop','sell','network'],
            metrics=dict(deadline_resets=0,turn_overrun_seconds=0.,completed=1,play_now=False,cancelled=False))
            for tier in difficulty.TIERS for era in ('CanalEra','RailEra') for seed in (1,2)]
        for control,status in (('cancel','cancelled'),('play-now','ok')):
            rows.append(dict(tier='easy',era='CanalEra',seed=3,status=status,interrupt=control,
                metrics=dict(play_now=True,cancelled=control=='cancel',deadline_resets=0,turn_overrun_seconds=0.)))
        report=dict(format=contracts.BENCHMARK,package_id='package',fixture=False,unity_running=True,
            environment=dict(processor='Intel Core Ultra 5 135U'),results=[dict(backend='ort-cpu',available=True,records=rows)])
        review=dict(format='personality-laptop-acceptance/1.0.0',package_id='package',build_id='build',backend='ort-cpu',
            target='Intel Core Ultra 5 135U',reviewer='Synthetic fixture only',checks={k:True for k in (
                'four_player_game','sixteen_styles','save_load_pending_action','history_rewind','play_now','menu_cancel',
                'resource_restoration','old_save_unchanged','same_profile_strength_review')})
        return report,review

    def certify(self,report,review,output):
        with patch.object(install,'validate',return_value=dict(id='package',p4_id='p4',p5_id='p5')),\
             patch.object(install,'build_check',return_value=dict(id='build')),\
             patch.object(install,'read',side_effect=[contracts.seal(report),review]),\
             patch.object(install,'sha',return_value='fixture-sha'):
            return install.certify('package','build','benchmark','review',output)

    def test_scheduled_but_unobserved_controls_cannot_certify(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            report,review=self.evidence()
            self.assertTrue(self.certify(report,review,Path(tmp)/'positive.json')['qualified'])
            for control,field in (('play-now','play_now'),('cancel','cancelled')):
                damaged=copy.deepcopy(report)
                next(r for r in damaged['results'][0]['records'] if r['interrupt']==control)['metrics'][field]=False
                output=Path(tmp)/(control+'.json')
                with self.subTest(control=control):
                    with self.assertRaisesRegex(ValueError,'cancellation/Play Now'):
                        self.certify(damaged,review,output)
                    self.assertFalse(output.exists())

    def test_play_now_must_preserve_admitted_clock(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            report,review=self.evidence()
            for field,value in (('deadline_resets',1),('turn_overrun_seconds',1.)):
                damaged=copy.deepcopy(report)
                next(r for r in damaged['results'][0]['records'] if r['interrupt']=='play-now')['metrics'][field]=value
                output=Path(tmp)/(field+'.json')
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError,'Deadline/failure'):
                        self.certify(damaged,review,output)
                    self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
