"""Restart, publication, retention and evaluation-only source boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.step5.league import write_json
from brass_ai.recovery import ratings, resume
from brass_ai.recovery import context_evaluation as L,context_evaluation_config as C
from brass_ai.recovery import context_evaluation_collect as E,context_evaluation_spec as S
from test_context_evaluation import recipe,fixture_spec,fixture_rows


def panel(root):
    run=root/'artifacts/step5/demo';run.mkdir(parents=True)
    config=C.configure(recipe());state=dict(iteration=1280,ratings={'slots':[]},
        optimizer=dict(moment=31),rng=[1,2,3],population=['same'],domino={'unchanged':True})
    L.prepare(root,run,state,config);state['iteration']=1330
    spec=fixture_spec(1);parent=spec['parent'];directory=L.directory_for(run,1330)
    main=run/'ratings/slots/generation-001330';ratings.write_once(main/'spec.json',parent)
    with patch.object(L,'specification_for',return_value=spec):L.seal(root,run,state,config,parent,main/'spec.json')
    rows=fixture_rows(spec);jobs={(0,j['index']):j for j in ratings.jobs_for(parent,directory)}
    jobs.update({(j['provenance']['context_id'],j['provenance']['parent_index']):j for j in S.jobs_for(spec,directory/'raw')})
    for r in rows:
        job=jobs[r['context_id'],r['parent_index']]
        r.update(runtime=spec['runtime'],implementation=spec['implementation'],
            candidate=parent['agents'][parent['candidate']]['checkpoint'],policies=job['policies'],
            opponents=[job['policies'][(s-job['rotation'])%4]['policy_id'] for s in range(4)],
            sampling_seed=job['sampling_seed'],rng_index=0)
        if r['context_id']==0:E.save_summary(directory,r)
    records=[]
    for r in rows[:4]:
        s=r['seat'];vp=[0]*4;vp[s]=r['vp'];win=[0]*4;win[s]=r['win_share'];place=[4]*4;place[s]=r['placement']
        records.append(dict(kind='rating',setup_seed=r['setup_seed'],candidate_seat=s,game_id=r['game_id'],setup_id=r['setup_id'],
            vp=vp,win_share=win,official_placement=place,spec_digest=digest(parent)))
    ratings.append_games(main/'games.jsonl',records)
    state['ratings']['slots']=[dict(generation=1330,spec=ratings.receipt(root,main/'spec.json'),records=ratings.receipt(root,main/'games.jsonl'))]
    write_json(run/'state.json',state)
    return run,state,config,spec,directory,rows


def collector(rows):
    def collect(root,directory,spec,**kw):
        existing=E.load_summaries(directory,spec)
        for row in rows[4:]:E.save_summary(directory,row)
        return dict(collection_seconds=.2,extraction_seconds=.1,unique_games=24,
            reused_completed_jobs=len(existing)-4,new_completed_jobs=28-len(existing),inference={})
    return collect


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_interrupt_resumes_sealed_checkpoint_state_and_no_duplicate_summaries(self):
        run,state,c,spec,directory,rows=panel(self.root);before=copy.deepcopy({k:v for k,v in state.items() if k!='context_evaluation'})
        def interrupted(root,directory,spec,**kw):
            for row in rows[4:8]:E.save_summary(directory,row)
            raise KeyboardInterrupt
        with patch.object(L,'validate_runtime'),patch.object(L,'collect',side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):L.run_pending(self.root,run,state,c)
        self.assertEqual(state['context_evaluation']['pending']['generation'],1330)
        support=read(directory/'status.json')['support']
        self.assertEqual((support['validated_games'],support['new_completed'],support['neutral_reused']),(8,4,4))
        self.assertEqual(sum(support['per_context'].values()),8)
        sealed=sha(directory/'spec.json');files={p:sha(p) for p in (directory/'evidence').glob('*.json')}
        state=read(run/'state.json')
        with patch.object(L,'validate_runtime'),patch.object(L,'collect',side_effect=collector(rows)) as collect,\
             patch('brass_ai.recovery.plackett_luce.estimate',side_effect=AssertionError('no extra PL fit')):
            result=L.run_pending(self.root,run,state,c)
        self.assertEqual(result['status'],'complete');self.assertEqual(collect.call_count,1)
        self.assertEqual(state['context_evaluation']['next_due_generation'],1380);self.assertIsNone(state['context_evaluation']['pending'])
        self.assertEqual(before,{k:v for k,v in state.items() if k!='context_evaluation'})
        self.assertEqual(sealed,sha(directory/'spec.json'));self.assertTrue(all(sha(p)==h for p,h in files.items()))
        self.assertEqual(len(E.load_summaries(directory,spec)),28)
        with patch.object(L,'collect',side_effect=AssertionError('no duplicate jobs')):L.run_pending(self.root,run,state,c)

    def test_corrupt_neutral_or_source_is_permanent_and_never_replaced(self):
        run,state,c,spec,directory,rows=panel(self.root)
        path=next((directory/'evidence').glob('*.receipt.json'));path.write_text('{}',encoding='utf-8')
        with patch.object(L,'validate_runtime'),patch.object(L,'collect') as collect:
            result=L.run_pending(self.root,run,state,c)
        self.assertEqual(result['status'],'failed');collect.assert_not_called()
        self.assertIsNone(state['context_evaluation']['pending']);self.assertEqual(state['context_evaluation']['panels'][-1]['status'],'failed')

    def test_transient_failures_bounded_and_no_training_state_reset(self):
        run,state,c,spec,directory,rows=panel(self.root);before=copy.deepcopy({k:v for k,v in state.items() if k!='context_evaluation'})
        with patch.object(L,'validate_runtime'),patch.object(L,'collect',side_effect=RuntimeError('transient actor exit')) as collect:
            for i in range(3):result=L.run_pending(self.root,run,state,c)
        self.assertEqual(result['status'],'failed');self.assertEqual(collect.call_count,3)
        self.assertEqual(before,{k:v for k,v in state.items() if k!='context_evaluation'})
        self.assertEqual(len(read(directory/'status.json')['attempts']),3)

    def test_failed_render_retry_never_collects_and_alias_keeps_generation(self):
        run,state,c,spec,directory,rows=panel(self.root)
        with patch.object(L,'validate_runtime'),patch.object(L,'collect',side_effect=collector(rows)),\
             patch('brass_ai.recovery.context_evaluation_plot.render',side_effect=ValueError('renderer unavailable')):
            result=L.run_pending(self.root,run,state,c)
        self.assertEqual(result['status'],'complete');self.assertEqual(read(directory/'visualization-failed.json')['status'],'failed')
        self.assertIsNone(state['context_evaluation']['latest_visualization']);report=sha(directory/'report.json')
        with patch.object(L,'collect',side_effect=AssertionError('no games on render retry')):L.run_pending(self.root,run,state,c)
        self.assertEqual(sha(directory/'report.json'),report)
        self.assertEqual(state['context_evaluation']['latest_visualization']['generation'],1330)
        self.assertEqual((directory/'progress.svg').read_bytes(),(run/'ratings/contexts/progress.svg').read_bytes())

    def test_retention_owns_only_expired_raw_complete_panels_and_protects_pending(self):
        run=self.root/'artifacts/run';run.mkdir(parents=True);panels=[]
        for g in (50,100,150,200):
            directory=L.directory_for(run,g);directory.mkdir(parents=True);ratings.write_once(directory/'report.json',{'generation':g})
            raw=directory/'raw';raw.mkdir();(raw/'a.jsonl.gz').write_bytes(b'raw')
            (directory/'progress.svg').write_text('compact',encoding='utf-8');(raw/'unowned.txt').write_text('keep',encoding='utf-8')
            if g<200:panels.append(dict(generation=g,status='complete',report=ratings.receipt(self.root,directory/'report.json')))
        state=dict(context_evaluation=dict(config=C.defaults(),panels=panels,pending=dict(generation=200)))
        result=L.prune(self.root,run,state);self.assertEqual(result['files'],1)
        self.assertFalse((L.directory_for(run,50)/'raw/a.jsonl.gz').exists())
        for g in (100,150,200):self.assertTrue((L.directory_for(run,g)/'raw/a.jsonl.gz').exists())
        self.assertTrue((L.directory_for(run,50)/'progress.svg').exists());self.assertTrue((L.directory_for(run,50)/'raw/unowned.txt').exists())

    def test_export_verifies_compact_evidence_and_svg_alias_and_omits_raw(self):
        run,state,c,spec,directory,rows=panel(self.root)
        with patch.object(L,'validate_runtime'),patch.object(L,'collect',side_effect=collector(rows)):L.run_pending(self.root,run,state,c)
        spec_module=importlib.util.spec_from_file_location('context_export',SOURCE_ROOT/'scripts/export-training-reports.py')
        m=importlib.util.module_from_spec(spec_module);spec_module.loader.exec_module(m)
        files=list(m.context_files(self.root,run,state));names={p.name for p,_ in files}
        self.assertTrue({'report.json','spec.json','games.jsonl','summaries.jsonl','progress.svg','report.md','status.json'}<=names)
        self.assertFalse(any('raw' in p.parts for p,_ in files))
        (run/'ratings/contexts/progress.svg').write_text('wrong generation',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'receipt'):list(m.context_files(self.root,run,state))

    def test_final_cost_includes_render_and_retry_time_and_keeps_missing_intervals_missing(self):
        run,state,c,spec,directory,rows=panel(self.root)
        state['context_evaluation']['timing'].update(normal_seconds=1000.,last_generation=1329)
        with patch.object(L,'validate_runtime'),patch.object(L,'collect',side_effect=collector(rows)):L.run_pending(self.root,run,state,c)
        L.record_iteration_time(self.root,run,state,dict(iteration=1329,timings=dict(iteration_seconds=105.,context_evaluation_seconds=5.)))
        cost=read(directory/'cost.json');self.assertAlmostEqual(cost['normal_trainer_seconds'],1100.)
        self.assertGreater(cost['additional_seconds'],0.);self.assertGreater(state['context_evaluation']['timing']['added_seconds'],cost['cumulative_added_seconds'])
        self.assertTrue(read(directory/'visualization-final.json')['cost'])
        L.record_iteration_time(self.root,run,state,dict(iteration=1331,timings=dict(iteration_seconds=100.)))
        self.assertFalse(state['context_evaluation']['timing']['complete'])

    def test_final_render_failure_preserves_previous_svg_and_retries_are_bounded(self):
        run,state,c,spec,directory,rows=panel(self.root)
        with patch.object(L,'validate_runtime'),patch.object(L,'collect',side_effect=collector(rows)):L.run_pending(self.root,run,state,c)
        old=copy.deepcopy(state['context_evaluation']['latest_visualization']);svg=(directory/'progress.svg').read_bytes()
        state['context_evaluation']['timing'].update(normal_seconds=1000.,last_generation=1329)
        with patch('brass_ai.recovery.context_evaluation_plot.render',side_effect=RuntimeError('render failed')) as render:
            L.record_iteration_time(self.root,run,state,dict(iteration=1329,timings=dict(iteration_seconds=100.)))
            for _ in range(5):L.run_pending(self.root,run,state,c)
            self.assertEqual(render.call_count,3)
        self.assertEqual(state['context_evaluation']['latest_visualization'],old)
        self.assertEqual((directory/'progress.svg').read_bytes(),svg)
        self.assertEqual((run/'ratings/contexts/progress.svg').read_bytes(),svg)
        entry=state['context_evaluation']['panels'][-1]
        self.assertEqual(entry['render_failures'],3)
        self.assertEqual(read(run/'ratings/contexts/index.json')['panels'],state['context_evaluation']['panels'])

    def test_neutral_extraction_timer_survives_interrupt_and_detects_hard_kill(self):
        run,state,c,spec,directory,rows=panel(self.root)
        sink=L.neutral_sink(self.root,run,state,spec['parent'])
        self.assertTrue(read(directory/'status.json')['neutral_running'])
        # Simulate an unknown process-kill tail before the finish hook.
        restarted=L.neutral_sink(self.root,run,state,spec['parent'])
        restarted.finish();restarted.finish()
        status=read(directory/'status.json')
        self.assertFalse(status['neutral_running']);self.assertFalse(status['timing_complete'])

    def test_only_pending_context_panel_pins_its_frozen_checkpoints(self):
        from brass_ai.recovery.retention import references
        run=self.root/'artifacts/step5/demo';run.mkdir(parents=True)
        checkpoint=run/'population/frozen.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'model')
        directory=L.directory_for(run,50);path=directory/'spec.json'
        spec=dict(contract=C.CONTRACT,checkpoint=dict(path=checkpoint.relative_to(self.root).as_posix()))
        write_json(path,spec)
        pending=dict(spec=ratings.receipt(self.root,path),dependencies=[spec['checkpoint']])
        state=dict(format='step5-recovery-league/2.2.0',context_evaluation=dict(pending=pending,panels=[]))
        write_json(run/'state.json',state)
        self.assertIn(checkpoint.resolve(),references(self.root,run))
        state['context_evaluation'].update(pending=None,panels=[dict(generation=50,status='failed',spec=pending['spec'])])
        write_json(run/'state.json',state)
        self.assertNotIn(checkpoint.resolve(),references(self.root,run))


class ResumeBoundary(unittest.TestCase):
    def test_inapplicable_recipe_review_does_not_hide_a_valid_exact_path(self):
        old=dict(config=dict(learning_rate=.1),implementation='old',compatibility={'fixed':True})
        new=dict(old,config=dict(learning_rate=.2),implementation='new')
        state=dict(format='step5-recovery-league/2.2.0',signature=old,iteration=1340)
        base=dict(from_implementation='old',to_implementation='new',learning_compatibility=digest(old['compatibility']))
        approvals=dict(format='selfplay-resume-sources/1.0.0',transitions=[
            dict(base,config_migrations=['training-recipe-g1340/1.0.0']),
            dict(base,config_changes={'learning_rate':{'from':.1,'to':.2}})])
        with patch.object(resume,'read',return_value=approvals):
            self.assertTrue(resume.check_signature(state,new,reviewed=True))
            with self.assertRaises(ValueError):resume.check_signature(state,dict(new,config=dict(learning_rate=.3)),reviewed=True)
            approvals['transitions'][0]['config_migrations']=['unregistered']
            with self.assertRaisesRegex(ValueError,'Unknown reviewed'):resume.check_signature(state,new,reviewed=True)

    def test_reviewed_config_source_transition_and_calibration_are_independent_of_training(self):
        c=recipe();old=dict(config=c,implementation='old',compatibility={'fixed':True});new=dict(old,config=C.configure(c),implementation='new')
        state=dict(format='step5-recovery-league/2.2.0',signature=old,iteration=1330)
        approval=dict(format='selfplay-resume-sources/1.0.0',transitions=[dict(from_implementation='old',to_implementation='new',
            learning_compatibility=digest(old['compatibility']),context_evaluation_changes=True,preserve_evaluation_calibration=True)])
        with patch.object(resume,'read',return_value=approval):
            self.assertTrue(resume.check_signature(state,new,reviewed=True))
            self.assertTrue(resume.evaluation_receipt_preserved(state,new,dict(training={'signature':digest(old)}),{}))
            with self.assertRaises(ValueError):resume.check_signature(state,new,reviewed=False)
            bad=copy.deepcopy(new);bad['config']['exploration']*=2
            with self.assertRaises(ValueError):resume.check_signature(state,bad,reviewed=True)
            self.assertFalse(resume.evaluation_receipt_preserved(state,bad,dict(training={'signature':digest(old)}),{}))
            approval['transitions'][0].pop('preserve_evaluation_calibration')
            self.assertFalse(resume.evaluation_receipt_preserved(state,new,dict(training={'signature':digest(old)}),{}))
        self.assertNotIn('context_evaluation',state)


if __name__=='__main__':unittest.main()
