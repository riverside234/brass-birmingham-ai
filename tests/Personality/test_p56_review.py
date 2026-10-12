"""Authored regression evidence only; nothing here qualifies a real controller."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_difficulty import named,panel,record
from test_laptop import package
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest,compatibility
from brass_ai.personality.difficulty import contracts as c,ui,panels,collection,evaluation
from brass_ai.personality.laptop import contracts,export,install,sampling


class EvidenceTests(unittest.TestCase):
    def test_incomplete_summary_and_empty_gate_set_do_not_enable_tiers(self):
        v=named(16,8)
        r=dict(p5_qualified=True,fixture=False,roster_id=v['id'],profiles=c.default_profiles())
        self.assertFalse(any(x['enabled'] for x in ui.menu(v,r)['choices']))
        r.update(format=c.REPORT,contract=c.identity(),checks={},gates=asdict(c.Gates()),release_qualified=False)
        r=c.sealed(r)
        with self.assertRaisesRegex(ValueError,'Missing'):c.validate_report_summary(r,v)
        self.assertFalse(c.qualified_report(r,v))

    def test_resealed_report_cannot_change_results_from_pinned_evidence(self):
        v=named();p=panel(v);r=record(p['jobs'][0]);r['failure']=dict(kind='stock',type='Fixture',message='failed')
        r['result']=None;r['baseline_failures']=[r['failure']]
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            d=Path(tmp);contracts.write(d/'manifest.json',p)
            def chunks(*args):return p,v,iter([r]),{0},[]
            with patch.object(collection,'read_chunks',side_effect=chunks),patch.object(evaluation,'public_packet',return_value={}):
                original=evaluation.check(SOURCE_ROOT,d,[],d/'report')
                self.assertEqual(evaluation.load_report(SOURCE_ROOT,d/'report'),original)
                original['checks']['no_failed_games']=True
                contracts.write(d/'report/report.json',c.sealed({k:x for k,x in original.items() if k!='id'}))
                with self.assertRaisesRegex(ValueError,'pinned evidence'):evaluation.load_report(SOURCE_ROOT,d/'report')

    def test_release_follows_calibration_dependency_closure(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            d=Path(tmp);contracts.write(d/'report.json',c.sealed(dict(format=c.REPORT,contract=c.identity(),stage='calibration')))
            with patch.object(evaluation,'load_report',side_effect=ValueError('Missing calibration chunk')) as loader:
                with self.assertRaisesRegex(ValueError,'Missing calibration'):panels.calibration_report(SOURCE_ROOT,d)
                loader.assert_called_once_with(SOURCE_ROOT,d)

    def test_stock_failure_cannot_pass_even_if_failure_field_missing(self):
        v=named();p=panel(v);r=record(p['jobs'][0]);r['baseline_failures']=[dict(type='failed')]
        checks=evaluation.evidence_checks(v,p,[r],{0},dict(passed=False),{})
        self.assertFalse(checks['no_failed_games'])


class ExportAndSaveTests(unittest.TestCase):
    def test_matching_nan_or_infinity_is_not_parity(self):
        for name in ('policy_logits','remaining_utility_4','win_share_4','goal_family','confidence_logits'):
            for bad in (np.nan,np.inf,-np.inf):
                values={'policy_logits':np.ones((1,2)),name:np.full((1,2),bad)}
                with self.assertRaisesRegex(ValueError,'Nonfinite'):export.compare(values,values,np.ones((1,2),bool))
        values=dict(policy_logits=np.array([[1.,-np.inf]]),remaining_utility_4=np.array([[1.2,-.1,.3,2.5]]))
        self.assertEqual(export.compare(values,values,np.array([[True,False]]))['remaining_utility_4'],0.)

    def test_export_metadata_checks_seat_mapping_and_named_outputs(self):
        reward=dict(example='fixture')
        m=dict(format=contracts.EXPORT,actor=True,compatibility=compatibility(),precision='float32',opset=17,
               inputs=export.BASE_INPUTS+['personality'],outputs=export.BASE_OUTPUTS+export.GOAL_OUTPUTS,
               reward_id=digest(reward),config=dict(utility=reward))
        export.validate_metadata(m)
        for change in ('seat','output','reward'):
            bad=copy.deepcopy(m)
            if change=='seat':bad['compatibility']['seat_order']='relative'
            elif change=='output':bad['outputs'][1],bad['outputs'][2]=bad['outputs'][2],bad['outputs'][1]
            else:bad['reward_id']='old'
            with self.assertRaisesRegex(ValueError,'Incompatible'):export.validate_metadata(bad)

    def test_malformed_or_over_budget_pending_save_rejected_without_reset(self):
        p=package();save=sampling.new_save(p,'g','s',[dict(seat=0,archetype='random',tier='easy')])
        for key,seconds in (([0,'CanalEra',1,1,0],3.),([0,'wrong',1,1,0],1.),([0,'RailEra',1,0,0],1.)):
            value=copy.deepcopy(save);value['pending']=dict(key=key,remaining_seconds=seconds)
            value=contracts.seal({k:v for k,v in value.items() if k!='id'})
            with self.assertRaisesRegex(ValueError,'whole-turn'):sampling.restore(p,value,'g')
        old=copy.deepcopy(save);old['format']='personality-runtime-save/0.9.0'
        old=contracts.seal({k:v for k,v in old.items() if k!='id'})
        with self.assertRaisesRegex(ValueError,'format'):sampling.restore(p,old,'g')


class InstallTests(unittest.TestCase):
    def test_certificate_cannot_use_benchmark_from_different_cpu(self):
        report=contracts.seal(dict(format=contracts.BENCHMARK,package_id='fixture-id',fixture=False,
            unity_running=True,environment=dict(processor='different CPU')))
        with patch.object(install,'validate',return_value=dict(id='fixture-id')),patch.object(install,'build_check',return_value={}),\
             patch.object(install,'read',side_effect=[report,{}]):
            with self.assertRaisesRegex(ValueError,'target laptop'):install.certify('package','build','benchmark','review','out')

    def test_interpreter_preflight_uses_game_module_path_and_fails_closed(self):
        with patch.object(install.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='missing pyspiel')) as call:
            with self.assertRaisesRegex(ValueError,'dependencies'):install.preflight(Path('fixture'),Path('python'),'ort-cpu',fixture=True)
        kwargs=call.call_args.kwargs
        self.assertEqual(kwargs['env']['PYTHONPATH'],str(Path('fixture/runtime/src')))
        self.assertEqual(kwargs['timeout'],60)

    def test_failed_second_rollback_replacement_restores_current_install_and_can_retry(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            g=Path(tmp);(g/'.p6-fixture-sandbox').touch()
            managed=g/'BrassBirminghamUnity_Data/Managed';managed.mkdir(parents=True)
            backup=g/'BrassPersonality/backups/fixture';backup.mkdir(parents=True)
            pointer=g/'BrassPersonality/active.json';contracts.write(pointer,dict(current=True))
            for n in install.OWNED:
                (managed/n).write_bytes(('new-'+n).encode());(backup/n).write_bytes(('old-'+n).encode())
            j=contracts.seal(dict(format=install.INSTALL,game=str(g),fixture=True,status='installed',
                backup=backup.relative_to(g).as_posix(),previous_pointer_sha256=None,pointer_sha256=sha(pointer),
                previous_sha256=sha(backup/'BrassLegacyAI.dll'),installed_sha256=sha(managed/'BrassLegacyAI.dll'),
                previous_files={n:sha(backup/n) for n in install.OWNED},installed_files={n:sha(managed/n) for n in install.OWNED}))
            path=backup/'journal.json';contracts.write(path,j)
            before={p:p.read_bytes() for p in [pointer,*[managed/n for n in install.OWNED]]}
            replace=install.os.replace
            def fail(source,dest):
                if str(source).endswith('.p6-rollback') and Path(dest).name=='BrassPatch.dll':raise OSError('injected second replacement failure')
                return replace(source,dest)
            with patch.object(install,'_closed'),patch.object(install.os,'replace',side_effect=fail):
                with self.assertRaisesRegex(OSError,'injected'):install.rollback(path,apply=True,sandbox_fixture=True)
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            with patch.object(install,'_closed'):self.assertTrue(install.rollback(path,apply=True,sandbox_fixture=True)['rolled_back'])
            self.assertFalse(pointer.exists())
            for n in install.OWNED:self.assertEqual((managed/n).read_bytes(),(backup/n).read_bytes())


if __name__=='__main__':unittest.main()
