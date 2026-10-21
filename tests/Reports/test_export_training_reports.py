"""Report selection, byte preservation and consistent publication; no training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('export_reports', ROOT / 'scripts/export-training-reports.py')
reports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reports)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='reports-test-', dir=ROOT / 'artifacts')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = 'artifacts/step5/example'
        self.state = dict(format='step5-recovery-league/2.2.0', iteration=60, games_done=19200,
            signature=dict(implementation='recorded-source', config=dict(learning_rate=1e-5)),
            origin=dict(bc_run='artifacts/step5/il'),
            evaluations=[dict(report=self.run + '/evaluations/monitor-0003/report.json')],
            champion_gate=dict(recent=[]))
        self.put(self.run + '/state.json', reports.encode(self.state).replace(b'\n', b'\r\n'))
        self.put(self.run + '/retention-history.json', {'iteration-0057/report.json': {'summary': {'iteration': 57}}})
        for iteration in (57, 58, 59, 60):
            self.put(f'{self.run}/iteration-{iteration:04d}/report.json',
                     dict(iteration=iteration, timings=dict(iteration_seconds=1)))

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else reports.encode(value))
        return path

    def export(self, **kwargs):
        return reports.export(self.root, self.run, **kwargs)

    def test_committed_temperature_comparison_and_evidence_are_exported(self):
        directory=self.run+'/ratings/temperature/generation-000050/'
        spec=self.put(directory+'spec.json',dict(temperatures=[1.,.7]))
        games=self.put(directory+'games.jsonl',b'{"setup":1}\n')
        reference=lambda p:dict(path=p.relative_to(self.root).as_posix(),sha256=reports.checksum(p.read_bytes()))
        report=self.put(directory+'report.json',dict(spec=reference(spec),records=reference(games)))
        self.state['temperature_ratings']=[dict(generation=50,report=reference(report))]
        self.put(self.run+'/state.json',self.state)
        output,_=self.export()
        for path in (spec,games,report):
            self.assertEqual((output/path.relative_to(self.root/self.run)).read_bytes(),path.read_bytes())

    def test_entropy_hotfix_report_is_exported_unchanged(self):
        data=dict(applied=True,generation=60,fixed_floor_ratio=.95,
                  floor_before={'canal_action':.8},floor_after={'canal_action':.95},
                  branch_rate_per_eligible_decision=.015)
        path=self.put(self.run+'/exploration/hotfix-report.json',data)
        output,_=self.export()
        self.assertEqual((output/'exploration/hotfix-report.json').read_bytes(),path.read_bytes())

    def test_active_policy_anchor_cache_and_fit_are_exported_without_dense_arrays(self):
        reference=lambda p:dict(path=p.relative_to(self.root).as_posix(),sha256=reports.checksum(p.read_bytes()))
        dense=self.put(self.run+'/policy-anchor/probabilities/current.npz',b'large probabilities')
        cache=self.put(self.run+'/policy-anchor/probabilities/current.json',dict(probabilities=reference(dense)))
        folder=self.run+'/ratings/slots/generation-000030/'
        fit={key:reference(self.put(folder+key+'.json',dict(generation=30))) for key in ('report','spec')}
        fit['bootstrap']=reference(self.put(folder+'bootstrap.npz',b'large bootstrap'))
        hotfix=self.put(self.run+'/policy-anchor/hotfix-report.json',dict(applied=True))
        self.state['policy_anchor']=dict(active=dict(cache=reference(cache),fit=fit))
        self.put(self.run+'/state.json',self.state)
        output,_=self.export()
        for p in (cache,hotfix,*[self.root/fit[k]['path'] for k in ('report','spec')]):
            self.assertEqual((output/p.relative_to(self.root/self.run)).read_bytes(),p.read_bytes())
        self.assertFalse(any(p.suffix=='.npz' for p in output.rglob('*')))

    def test_collects_latest_two_all_evaluations_and_il_ancestry_preserving_bytes(self):
        self.put(self.run + '/best.json', {'generation': 0})
        self.put(self.run + '/generation-0060.pt', b'large checkpoint excluded')
        self.put(self.run + '/iteration-0059/games.jsonl.gz', b'replay excluded')
        self.put(self.run + '/iteration-0059/games.jsonl.gz.features.json', {'excluded': True})
        self.put(self.run + '/iteration-0059/teacher/receipt.json', {'qualified': 10})
        self.put(self.run + '/iteration-0059/profile-collection/report.json', {'excluded': True})
        for panel in ('champion-00019200', 'monitor-0003', 'broader-0001'):
            self.put(f'{self.run}/evaluations/{panel}/report.json', {'panel': panel})
            self.put(f'{self.run}/evaluations/{panel}/spec.json', {'seeds': [1, 2]})
        self.put('artifacts/step5/il/state.json', {'signature': {'parent_imitation': {'bc_run': 'artifacts/step5/parent'}}})
        self.put('artifacts/step5/il/report.json', {'epochs': 6})
        self.put('artifacts/step5/parent/state.json', {'signature': {}})
        self.put('artifacts/step5/parent/report.json', {'epochs': 5})
        before = {p: p.read_bytes() for p in (self.root / 'artifacts').rglob('*') if p.is_file()}
        output, manifest = self.export()
        self.assertTrue(output.name.startswith('example-g0060-'))
        self.assertEqual(manifest['latest_iterations'], [58, 59])
        self.assertEqual(manifest['missing'], [])
        self.assertTrue((output / 'iteration-0058/report.json').is_file())
        self.assertTrue((output / 'iteration-0059/teacher/receipt.json').is_file())
        self.assertTrue((output / 'evaluations/broader-0001/spec.json').is_file())
        self.assertTrue((output / 'lineage/artifacts/step5/parent/report.json').is_file())
        self.assertFalse((output / 'iteration-0057').exists())
        self.assertFalse((output / 'iteration-0060').exists())
        self.assertFalse(any(p.suffix in ('.pt', '.gz', '.npz') for p in output.rglob('*')))
        self.assertFalse(any('features' in p.name or 'profile-collection' in p.parts for p in output.rglob('*')))
        for entry in manifest['files']:
            raw = (output / entry['path']).read_bytes()
            self.assertEqual(raw, before[self.root / entry['source']])
            self.assertEqual(reports.checksum(raw), entry['sha256'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_completed_il_critic_upgrade_receipt_and_original_reports_are_exported(self):
        self.put(self.run+'/evaluations/monitor-0003/report.json',{})
        self.put(self.run+'/evaluations/monitor-0003/spec.json',{})
        self.put('artifacts/step5/il/upgrade.json', dict(contract='completed-il-critic-upgrade/1.0.0',
            parent=dict(bc_run='artifacts/step5/parent')))
        self.put('artifacts/step5/il/candidate.pt',b'weights excluded')
        self.put('artifacts/step5/parent/state.json',dict(signature={}))
        self.put('artifacts/step5/parent/report.json',dict(epochs=20))
        output,manifest=self.export()
        self.assertEqual(manifest['missing'],[])
        self.assertTrue((output/'lineage/artifacts/step5/il/upgrade.json').exists())
        self.assertTrue((output/'lineage/artifacts/step5/parent/report.json').exists())
        self.assertFalse(any(p.suffix=='.pt' for p in output.rglob('*')))
        self.assertEqual(json.loads((output / 'recorded-config.json').read_text()), self.state['signature']['config'])

    def test_missing_reports_are_explicit_not_reconstructed_from_summaries(self):
        (self.root / self.run / 'iteration-0058/report.json').unlink()
        output, manifest = self.export(output='results/snapshot')
        missing = {x['source'] for x in manifest['missing']}
        self.assertIn(self.run + '/iteration-0058/report.json', missing)
        self.assertIn(self.run + '/evaluations/monitor-0003/report.json', missing)
        self.assertIn('artifacts/step5/il/report.json', missing)
        self.assertFalse((output / 'iteration-0058/report.json').exists())

    def test_interrupted_cleanup_includes_older_uncompacted_report_without_pending_iteration(self):
        self.put(self.run+'/retention-history.json',{})
        original=(self.root/self.run/'iteration-0057/report.json').read_bytes()
        output,manifest=self.export()
        self.assertEqual(manifest['latest_iterations'],[58,59])
        self.assertEqual(manifest['uncompacted_iterations'],[57])
        self.assertEqual((output/'iteration-0057/report.json').read_bytes(),original)
        self.assertFalse((output/'iteration-0060').exists())

    def test_never_overwrites_existing_output_or_accepts_escaped_paths(self):
        sentinel = self.put('results/existing/keep.json', b'unchanged')
        for output in ('results/existing', 'artifacts/export', '../outside', 'results/../../outside'):
            with self.assertRaises(ValueError):
                self.export(output=output)
        self.assertEqual(sentinel.read_bytes(), b'unchanged')

    def test_detects_changed_source_and_does_not_publish_partial_bundle(self):
        original_encode = reports.encode

        def concurrent_write(value):
            if value.get('format') == reports.FORMAT:
                updated = dict(self.state, iteration=61)
                (self.root / self.run / 'state.json').write_bytes(original_encode(updated))
            return original_encode(value)

        with patch.object(reports, 'encode', side_effect=concurrent_write):
            with self.assertRaisesRegex(ValueError, 'changed during export'):
                self.export(output='results/snapshot')
        self.assertFalse((self.root / 'results/snapshot').exists())
        self.assertEqual(list((self.root / 'results').iterdir()), [])

    def test_more_iterations_external_panels_and_receipt_mismatch_are_recorded(self):
        self.state['evaluations'][0]['report_sha256'] = 'old-receipt'
        self.put(self.run + '/state.json', self.state)
        self.put(self.run + '/evaluations/monitor-0003/report.json', {'pruned': True})
        self.put('artifacts/comparison/report.json', {'candidate': 'generation60'})
        output, manifest = self.export(latest=3, extras=['artifacts/comparison'])
        self.assertTrue((output / 'iteration-0057/report.json').exists())
        self.assertTrue((output / 'related/artifacts/comparison/report.json').exists())
        self.assertEqual(len(manifest['warnings']), 1)
        entry = next(x for x in manifest['files'] if x['path'] == 'evaluations/monitor-0003/report.json')
        self.assertFalse(entry['matches_recorded_sha256'])

    def test_absent_run_is_not_created(self):
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            reports.export(self.root, 'artifacts/missing')
        self.assertFalse((self.root / 'artifacts/missing').exists())

    def test_compacted_rating_history_is_copied_with_exact_bytes_and_checksum(self):
        raw=reports.encode(dict(format='plackett-luce-history/1.0.0',slots=[dict(generation=10)],games=[]))
        path=self.put(self.run+'/ratings/history-'+reports.checksum(raw)+'.json',raw)
        self.state['ratings']=dict(slots=[],archive=dict(path=path.relative_to(self.root).as_posix(),sha256=reports.checksum(raw)))
        self.put(self.run+'/state.json',self.state)
        output,manifest=self.export()
        self.assertEqual((output/path.relative_to(self.root/self.run)).read_bytes(),raw)
        self.assertTrue(any(e['source']==path.relative_to(self.root).as_posix() for e in manifest['files']))
        path.write_bytes(raw+b' ')
        with self.assertRaisesRegex(ValueError,'Archived rating evidence changed'):
            self.export(output='results/tampered-archive')

    def modern_run(self, generation):
        self.run = 'artifacts/step5/joint7-v11-1-grouped-lr'
        il_run = 'artifacts/step5/features4-v11-1-batch7-il'
        controller = dict(contract='ppo-categorical-kl-lr/1.1.0', mode='apply',
            kl=.006, median_kl=.005, steps_done=8, steps_planned=8,
            multiplier_before=1.0, multiplier_after=1.0, decision='hold',
            reason='target_band', effective_lr=6e-6,
            health=dict(anchor_distance=.12, anchor_budget=.15,
                anchor_group_budget=.30, anchor_group_min_positions=64,
                anchor_groups={name:dict(positions=512, anchor_distance=.12, status='ok')
                    for name in ('canal_early', 'canal_late', 'rail_early', 'rail_late')}))
        self.state = dict(format='step5-recovery-league/2.2.0', iteration=generation,
            games_done=320 * generation, origin=dict(bc_run=il_run), evaluations=[],
            signature=dict(implementation='v11-1-grouped-source', config=dict(
                selfplay_recipe='latest-model-selfplay/2.0.0', teacher_enabled=False,
                exploration=.08, population=dict(recent=4, anchors=12),
                curriculum=[dict(until=None, selfplay=60, historical=30, legacy=10)],
                ppo_lr_controller=dict(contract=controller['contract'], mode='apply',
                    anchor_kl_budget=.15, anchor_group_kl_budget=.30, anchor_group_min_positions=64))))
        self.put(self.run + '/state.json', self.state)
        self.put(il_run + '/state.json', {'signature': {}})
        self.put(il_run + '/report.json', {'epochs': 10, 'complete': True})
        for iteration in range(max(0, generation - 2), generation):
            self.put(f'{self.run}/iteration-{iteration:04d}/report.json', dict(
                iteration=iteration, actor=dict(lr_controller=controller,
                    sample_usage=dict(actor_distinct_used=100, actor_unused_eligible=5)),
                sampling=dict(learner_trajectories=1024, by_composition=dict(
                    selfplay=dict(tables=192, learner_trajectories=768),
                    historical=dict(tables=96, learner_trajectories=192),
                    legacy=dict(tables=32, learner_trajectories=64))),
                search=dict(enabled=False, collection_count=0, optimizer_steps=0),
                timings=dict(iteration_seconds=100)))
        return controller

    def test_default_cli_exports_new_run_and_keeps_completed_il_name(self):
        controller = self.modern_run(generation=1)
        # An explicitly null gate must also work for a gate-disabled run.
        self.state['champion_gate'] = None
        self.put(self.run + '/state.json', self.state)
        before = {p: p.read_bytes() for p in (self.root / 'artifacts').rglob('*') if p.is_file()}
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts/export-training-reports.py'),
            '--root', str(self.root)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        outputs = list((self.root / 'results').iterdir())
        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self.assertTrue(output.name.startswith('joint7-v11-1-grouped-lr-g0001-'))
        manifest = json.loads((output / 'export-manifest.json').read_text())
        self.assertEqual(manifest['run'], self.run)
        self.assertEqual(manifest['latest_iterations'], [0])
        self.assertEqual(manifest['missing'], [dict(source=self.run + '/retention-history.json',
            reason='Retained iteration trends')])
        self.assertEqual(manifest['warnings'], [])
        report = json.loads((output / 'iteration-0000/report.json').read_text())
        self.assertEqual(report['actor']['lr_controller'], controller)
        self.assertFalse(report['search']['enabled'])
        il_report = output / 'lineage/artifacts/step5/features4-v11-1-batch7-il/report.json'
        self.assertEqual(il_report.read_bytes(), before[self.root / self.state['origin']['bc_run'] / 'report.json'])
        self.assertFalse((output / 'evaluations').exists())
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_full_and_compact_grouped_lr_fields_are_preserved(self):
        controller = self.modern_run(generation=3)
        history = {'iteration-0000/report.json': dict(sha256='retired-full-report',
            summary=dict(summary_version=5, actor=dict(lr_controller=controller),
                search=dict(enabled=False), sampling=dict(learner_trajectories=1024)))}
        self.put(self.run + '/retention-history.json', history)
        self.put(self.run + '/iteration-0003/report.json', {'iteration': 3, 'pending': True})
        output, manifest = self.export()
        self.assertEqual(manifest['latest_iterations'], [1, 2])
        self.assertEqual(manifest['missing'], [])
        self.assertFalse((output / 'iteration-0003').exists())
        self.assertEqual(json.loads((output / 'retention-history.json').read_text()), history)
        for entry in manifest['files']:
            self.assertEqual((output / entry['path']).read_bytes(), (self.root / entry['source']).read_bytes())
        config = json.loads((output / 'recorded-config.json').read_text())
        self.assertEqual(config, self.state['signature']['config'])
        self.assertEqual(config['ppo_lr_controller']['anchor_kl_budget'], .15)
        self.assertEqual(config['ppo_lr_controller']['anchor_group_kl_budget'], .30)
        report = json.loads((output / 'iteration-0002/report.json').read_text())
        self.assertEqual(report['actor']['lr_controller'], controller)
        self.assertEqual(report['actor']['sample_usage']['actor_unused_eligible'], 5)
        self.assertEqual(report['sampling']['by_composition']['historical']['learner_trajectories'], 192)

    def test_windows_publication_retries_transient_handle_error(self):
        stage = self.put('stage/report.json', {'original': True}).parent
        output = self.root / 'published'
        error = PermissionError('temporary Windows handle')
        error.winerror = 32
        original_rename = Path.rename
        def rename(path, destination):
            if mocked_rename.call_count == 1:
                raise error
            return original_rename(path, destination)
        with patch.object(reports, 'os') as system, patch.object(reports.time, 'sleep') as sleep:
            system.name = 'nt'
            with patch.object(Path, 'rename', autospec=True, side_effect=rename) as mocked_rename:
                reports.publish(stage, output)
                self.assertEqual(mocked_rename.call_count, 2)
            sleep.assert_called_once_with(.05)
        self.assertEqual(json.loads((output / 'report.json').read_text()), {'original': True})
        self.assertFalse(stage.exists())

    def test_fixed_il_probe_reports_are_included_without_validation_tensors(self):
        self.modern_run(generation=1)
        self.put(self.run+'/il-kl/manifest.json',dict(format='fixed-il-kl-probe/1.0.0',probe_hash='fixed',
            cache=dict(path=self.run+'/il-kl/probe.npz')))
        self.put(self.run+'/il-kl/initial-report.json',dict(generation=0,measurement=dict(kl_mean=0.)))
        self.put(self.run+'/il-kl/probe.npz',b'dense validation tensors excluded')
        self.put(self.run+'/il-kl/.prepared/shard.json',dict(excluded=True))
        output,manifest=self.export()
        self.assertTrue((output/'il-kl/manifest.json').is_file())
        self.assertTrue((output/'il-kl/initial-report.json').is_file())
        self.assertFalse((output/'il-kl/probe.npz').exists())
        self.assertFalse((output/'il-kl/.prepared').exists())
        self.assertEqual(len([e for e in manifest['files'] if e['path'].startswith('il-kl/')]),2)

    def test_windows_publication_stops_if_output_appears_during_retry(self):
        stage = self.put('stage/report.json', {}).parent
        output = self.root / 'published'
        error = PermissionError('temporary Windows handle')
        error.winerror = 5
        def competing_export(path, destination):
            self.put('published/keep.json', {'sentinel': True})
            raise error
        with patch.object(reports, 'os') as system, patch.object(reports.time, 'sleep'):
            system.name = 'nt'
            with patch.object(Path, 'rename', autospec=True, side_effect=competing_export) as rename:
                with self.assertRaisesRegex(ValueError, 'Output appeared'):
                    reports.publish(stage, output)
                rename.assert_called_once()
        self.assertTrue((stage / 'report.json').is_file())
        self.assertEqual(json.loads((output / 'keep.json').read_text()), {'sentinel': True})

    def test_windows_publication_retry_is_bounded(self):
        stage = self.put('stage/report.json', {}).parent
        output = self.root / 'published'
        error = PermissionError('persistent Windows access error')
        error.winerror = 5
        with patch.object(reports, 'os') as system, patch.object(reports.time, 'sleep') as sleep:
            system.name = 'nt'
            with patch.object(Path, 'rename', autospec=True, side_effect=error) as rename:
                with self.assertRaisesRegex(PermissionError, 'persistent'):
                    reports.publish(stage, output)
                self.assertEqual(rename.call_count, 5)
            self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), .75)
        self.assertFalse(output.exists())
        self.assertTrue((stage / 'report.json').is_file())

    def test_rating_outcomes_panels_and_curves_preserve_bytes_without_dense_caches(self):
        directory=self.run+'/ratings/slots/generation-000060'
        paths={
            'records':self.put(directory+'/games.jsonl',b'{"game_id":"held-out","slot":60}\n'),
            'panels':self.put(directory+'/panels.json',dict(qualified=None,reason='Review required')),
            'report':self.put(directory+'/report.json',dict(ratings={'generation-0':dict(points=0)})),
            'spec':self.put(directory+'/spec.json',dict(exploration=0)),
        }
        curve=self.put(directory+'/progress.svg',b'<svg xmlns="http://www.w3.org/2000/svg"/>\n')
        self.put(directory+'/bootstrap.npz',b'large dense bootstrap array')
        self.put(directory+'/game-000001.jsonl.gz',b'full replay')
        self.put(self.run+'/opponent-pool/probes/member.npz',b'large probability cache')
        self.state['ratings']=dict(slots=[dict(generation=60,**{
            key:dict(path=p.relative_to(self.root).as_posix(),sha256=reports.checksum(p.read_bytes()))
            for key,p in paths.items()})])
        self.state['quality_pool']=dict(members=[dict(q=-.2,p=.4)],max_members=24)
        self.put(self.run+'/state.json',self.state)
        output,manifest=self.export()
        for path in [*paths.values(),curve]:
            self.assertEqual((output/path.relative_to(self.root/self.run)).read_bytes(),path.read_bytes())
        self.assertFalse(any(p.suffix in ('.npz','.gz','.pt') for p in output.rglob('*')))
        self.assertEqual(json.loads((output/'state.json').read_text())['quality_pool'],self.state['quality_pool'])
        paths['records'].write_bytes(b'{"changed":true}\n')
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.export()

    def advance_run(self, generation):
        self.state.update(iteration=generation, games_done=generation*320)
        self.put(self.run+'/state.json', self.state)
        self.put(f'{self.run}/iteration-{generation-1:04d}/report.json',
                 dict(iteration=generation-1, timings=dict(iteration_seconds=1)))

    def rating_slot(self, generation):
        directory=f'{self.run}/ratings/slots/generation-{generation:06d}'
        paths={key:self.put(directory+'/'+name, raw) for key,name,raw in (
            ('records','games.jsonl',b'{"heldout":true}\n'),
            ('panels','panels.json',dict(qualified=None)),
            ('report','report.json',dict(generation=generation)),
            ('spec','spec.json',dict(generation=generation)))}
        self.put(directory+'/progress.svg',b'<svg xmlns="http://www.w3.org/2000/svg"/>\n')
        return dict(generation=generation, **{
            key:dict(path=path.relative_to(self.root).as_posix(),sha256=reports.checksum(path.read_bytes()))
            for key,path in paths.items()})

    def active_anchor_at_latest_rating(self):
        self.advance_run(320)
        slot=self.rating_slot(320)
        cache=self.put(self.run+'/policy-anchor/probabilities/current.json',dict(model_id='anchor'))
        self.state['ratings']=dict(slots=[slot])
        self.state['policy_anchor']=dict(active=dict(
            cache=dict(path=cache.relative_to(self.root).as_posix(),sha256=reports.checksum(cache.read_bytes())),
            fit={key:slot[key] for key in ('report','spec')}))
        self.put(self.run+'/state.json',self.state)
        return slot

    def test_latest_rating_also_used_by_anchor_deduplicates_and_incrementally_reuses(self):
        slot=self.active_anchor_at_latest_rating()
        original={key:(self.root/slot[key]['path']).read_bytes() for key in ('report','spec')}
        output,manifest=self.export(output='results/anchor-first')
        for key,raw in original.items():
            name=Path(slot[key]['path']).relative_to(self.run).as_posix()
            self.assertEqual((output/name).read_bytes(),raw)
            self.assertEqual(sum(e['path']==name for e in manifest['files']),1)
        completed=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/export-training-reports.py'),
            '--root',str(self.root),'--run',self.run,'--output','results/anchor-continued','--continue'],
            capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        continued=self.root/'results/anchor-continued'
        manifest=json.loads((continued/'export-manifest.json').read_text())
        self.assertEqual(manifest['rating_slot_generation'],320)
        for key,raw in original.items():
            name=Path(slot[key]['path']).relative_to(self.run).as_posix()
            self.assertFalse((continued/name).exists())
            reused=[e for e in manifest['reused_files'] if e['path']==name]
            self.assertEqual(len(reused),1)
            self.assertEqual(reused[0]['previous_export'],'results/anchor-first')
            self.assertEqual((self.root/reused[0]['previous_export']/name).read_bytes(),raw)
            self.assertEqual((self.root/slot[key]['path']).read_bytes(),raw)

    def test_duplicate_anchor_rating_with_changed_bytes_still_aborts_publication(self):
        slot=self.active_anchor_at_latest_rating();target=self.root/slot['report']['path']
        original=target.read_bytes();changed=original.replace(b'320',b'321')
        self.assertNotEqual(original,changed);self.assertEqual(len(original),len(changed))
        original_files=reports.report_files
        def change_between_selections(root,directory,**kwargs):
            for path in original_files(root,directory,**kwargs):
                if path==target:target.write_bytes(changed)
                yield path
        with patch.object(reports,'report_files',side_effect=change_between_selections):
            with self.assertRaisesRegex(ValueError,'Conflicting export destination: ratings/slots/generation-000320/report.json'):
                self.export(output='results/anchor-changed')
        self.assertFalse((self.root/'results/anchor-changed').exists())

    def test_only_latest_committed_rating_slot_is_copied(self):
        old=self.rating_slot(10);latest=self.rating_slot(50)
        self.rating_slot(60)  # On disk but not yet committed in rating state.
        self.state['ratings']=dict(slots=[latest,old])
        self.put(self.run+'/state.json',self.state)
        self.put(self.run+'/ratings/slots/generation-000010/report.json',b'old evidence not selected')
        output,manifest=self.export()
        self.assertEqual(manifest['rating_slot_generation'],50)
        self.assertEqual([p.name for p in (output/'ratings/slots').iterdir()],['generation-000050'])
        self.assertEqual(len(list((output/'ratings/slots/generation-000050').iterdir())),5)
        path=self.root/latest['records']['path'];path.write_bytes(b'{"changed":true}\n')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            self.export()

    def test_uncommitted_rating_directory_without_state_slot_is_excluded(self):
        self.rating_slot(60)
        output,manifest=self.export()
        self.assertIsNone(manifest['rating_slot_generation'])
        self.assertFalse((output/'ratings/slots').exists())

    def test_retired_panel_trigger_slot_exported_until_archived(self):
        old=self.rating_slot(10);latest=self.rating_slot(50);self.rating_slot(60)
        self.state['ratings']=dict(slots=[old,latest],panel_gates=dict(retired={
            'v4-panel':dict(generation=10),'il-panel':dict(generation=10)}))
        self.put(self.run+'/state.json',self.state)
        output,manifest=self.export()
        self.assertEqual(manifest['rating_slot_generation'],50)
        self.assertEqual({p.name for p in (output/'ratings/slots').iterdir()},
                         {'generation-000010','generation-000050'})
        for key in ('panels','spec','records'):
            path=self.root/old[key]['path']
            self.assertEqual((output/path.relative_to(self.root/self.run)).read_bytes(),path.read_bytes())

    def test_continue_reuses_legacy_exports_but_copies_new_and_changed_reports(self):
        original,_=self.export(output='results/original')
        manifest_path=original/'export-manifest.json'
        legacy=json.loads(manifest_path.read_text());legacy['format']='selfplay-report-export/1.0.0'
        manifest_path.write_bytes(reports.encode(legacy))
        self.advance_run(61)
        output,manifest=self.export(output='results/next',continue_export=True)
        self.assertTrue(manifest['incremental'])
        self.assertFalse((output/'iteration-0059/report.json').exists())
        self.assertTrue((output/'iteration-0060/report.json').exists())
        self.assertTrue((output/'state.json').exists())
        reused=next(e for e in manifest['reused_files'] if e['path']=='iteration-0059/report.json')
        self.assertEqual(reused['previous_export'],'results/original')
        for entry in manifest['reused_files']:
            path=self.root/entry['previous_export']/entry['previous_path']
            self.assertEqual(reports.checksum(path.read_bytes()),entry['sha256'])
        changed=dict(iteration=59,timings=dict(iteration_seconds=2),finalized=True)
        self.put(self.run+'/iteration-0059/report.json',changed)
        output,manifest=self.export(output='results/finalized',continue_export=True)
        self.assertEqual(json.loads((output/'iteration-0059/report.json').read_text()),changed)
        self.assertNotIn('iteration-0059/report.json',{e['path'] for e in manifest['reused_files']})

    def test_continue_first_export_and_foreign_or_future_bundles_do_not_skip_files(self):
        first,baseline=self.export(output='results/first',continue_export=True)
        self.assertEqual(baseline['reused_files'],[])
        path=first/'export-manifest.json';foreign=dict(baseline,run='artifacts/step5/another-run')
        path.write_bytes(reports.encode(foreign))
        _,manifest=self.export(output='results/foreign-check',continue_export=True)
        self.assertEqual(manifest['reused_files'],[])
        self.assertEqual(len(manifest['files']),len(baseline['files']))
        for previous in (first,self.root/'results/foreign-check'):
            path=previous/'export-manifest.json';future=dict(baseline,committed_generation=61)
            path.write_bytes(reports.encode(future))
        _,manifest=self.export(output='results/future-check',continue_export=True)
        self.assertEqual(manifest['reused_files'],[])

    def test_continue_validates_actual_old_bytes_and_keeps_references_flat(self):
        original,_=self.export(output='results/first')
        (original/'iteration-0059/report.json').write_bytes(b'corrupted old export')
        second,manifest=self.export(output='results/second',continue_export=True)
        self.assertTrue((second/'iteration-0059/report.json').exists())
        self.assertTrue(any('Previous report unavailable or changed' in w for w in manifest['warnings']))
        self.advance_run(61)
        _,manifest=self.export(output='results/third',continue_export=True)
        report=next(e for e in manifest['reused_files'] if e['path']=='iteration-0059/report.json')
        self.assertEqual(report['previous_export'],'results/second')
        history=next(e for e in manifest['reused_files'] if e['path']=='retention-history.json')
        self.assertEqual(history['previous_export'],'results/first')

    def test_continue_aborts_if_reused_evidence_changes_during_publication(self):
        original,_=self.export(output='results/original')
        original_encode=reports.encode
        def concurrent_write(value):
            if value.get('format')==reports.FORMAT:
                (original/'iteration-0059/report.json').write_bytes(b'changed during export')
            return original_encode(value)
        with patch.object(reports,'encode',side_effect=concurrent_write):
            with self.assertRaisesRegex(ValueError,'Previous report changed during export'):
                self.export(output='results/incomplete',continue_export=True)
        self.assertFalse((self.root/'results/incomplete').exists())

    def test_continue_cli_reuses_unchanged_latest_rating_evidence(self):
        slot=self.rating_slot(60);self.state['ratings']=dict(slots=[slot])
        self.put(self.run+'/state.json',self.state)
        self.export(output='results/previous')
        self.advance_run(61)
        completed=subprocess.run([sys.executable,str(ROOT/'scripts/export-training-reports.py'),
            '--root',str(self.root),'--run',self.run,'--output','results/continued','--continue'],
            capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        output=self.root/'results/continued'
        manifest=json.loads((output/'export-manifest.json').read_text())
        self.assertFalse((output/'ratings/slots').exists())
        self.assertEqual(len([e for e in manifest['reused_files'] if e['path'].startswith('ratings/slots/')]),5)
        self.assertEqual(manifest['rating_slot_generation'],60)
        self.assertIn('Reused',completed.stdout)

    def test_real_iteration60_exports_can_be_bundled_without_models_or_torch(self):
        # Reconstruct only report paths in a temporary fixture, never the real run.
        source = ROOT / 'results/selfplay-joint'
        state = json.loads((source / 'state.json').read_text())
        self.run = 'artifacts/step5/v4-selfplay-joint'
        for filename in ('state.json', 'retention-history.json'):
            self.put(self.run + '/' + filename, (source / filename).read_bytes())
        self.put(self.run + '/iteration-0059/report.json', (source / 'iter59.json').read_bytes())
        self.put(state['evaluations'][-1]['report'], (source / 'monitor.json').read_bytes())
        self.put(state['champion_gate']['recent'][-1]['report'], (source / 'champion.json').read_bytes())
        output, manifest = self.export(output='results/real-evidence-smoke')
        self.assertEqual(manifest['committed_generation'], 60)
        self.assertEqual(len(manifest['files']), 5)
        self.assertEqual((output / 'iteration-0059/report.json').read_bytes(), (source / 'iter59.json').read_bytes())
        self.assertIn(self.run + '/iteration-0058/report.json', {m['source'] for m in manifest['missing']})


if __name__ == '__main__':
    unittest.main()
