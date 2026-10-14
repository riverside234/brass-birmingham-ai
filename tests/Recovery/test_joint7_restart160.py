"""Exact rollback bytes, fail-closed restore and audited counter reconstruction."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.batching import model_id
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.retention import run_lock,slim_checkpoint,_same_payload,update_population
from brass_ai.recovery.resume import load_resume
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json,implementation_id

sys.path.insert(0,str(SOURCE_ROOT/'scripts'))
from joint7_restart160 import reconstruct,resolve_models,reduce_population,publish,main,EXPORT,TARGET_SHA,REVIEWED_MISSING_GENERATIONS


class RestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_export_reconstructs160_and_cross_checks170(self):
        state,history,receipt=reconstruct(SOURCE_ROOT/EXPORT)
        self.assertEqual(state['iteration'],160);self.assertEqual(state['games_done'],51200)
        self.assertEqual(state['next_seed'],18575575);self.assertEqual(state['next_search_games'],51360)
        self.assertEqual(state['history'][-1]['sha256'],TARGET_SHA)
        self.assertEqual(len(state['population']),64)
        self.assertTrue(all(e['generation']<=160 for e in state['population']))
        self.assertLess(receipt['counter_max_error'],1e-6)
        self.assertEqual(state['signature']['config']['insolvency']['max_cost'],.025)
        self.assertNotIn('champion_mix',state)
        self.assertTrue(all(v['summary']['games_done']<=51200 for v in history.values()))

    def fixture(self,*,population=None):
        tmp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(tmp.cleanup)
        parent=Path(tmp.name);run=parent/'league';run.mkdir();backup=parent/'backup';backup.mkdir()
        cfg=read(SOURCE_ROOT/'configs/step5/recovery-smoke.json');cfg['model_size']='small'
        if population is not None:cfg['population']=population
        signature=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(cfg)))
        opt=torch.optim.AdamW(model.parameters());entries=[]
        for gen in (0,160):
            sum(p.square().mean() for p in model.parameters()).backward();opt.step();opt.zero_grad()
            path=run/f'generation-{gen:04d}.pt'
            identity=save_checkpoint(path,model,opt,step=gen,training=dict(signature=digest(signature),fresh_run=run.relative_to(SOURCE_ROOT).as_posix()))
            entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=gen))
        slim=slim_checkpoint(SOURCE_ROOT,run,entries[-1])
        state=dict(format='step5-recovery-league/2.2.0',signature=signature,iteration=160,games_done=51200,
            previous_games=0,next_seed=18575575,next_search_games=51360,context_counter=51200,
            ordinary_seconds=105472.,search_seconds=70394.,history=entries,population=[entries[0],slim],
            evaluation_anchor=entries[0],champion=slim,champion_gate=dict(accepted_games=51200,recent=[],accepted_gate=None),evaluations=[])
        later=copy.deepcopy(state);later.update(iteration=179,games_done=57280)
        write_json(run/'state.json',later);future=run/'iteration-0160';future.mkdir();(future/'report.json').write_bytes(b'future evidence')
        receipt=dict(source_state_digest=digest(later),checkpoint=entries[-1],target_generation=160)
        return run,backup,state,receipt

    def test_full_optimizer_missing_is_not_replaced_by_best_weights(self):
        run,backup,state,receipt=self.fixture();full=SOURCE_ROOT/state['history'][-1]['path'];saved=full.read_bytes()
        full.unlink()
        files,missing=resolve_models(SOURCE_ROOT,run,state)
        self.assertIn(state['history'][-1]['path'],missing)
        (backup/full.name).write_bytes(b'wrong checkpoint')
        _,missing=resolve_models(SOURCE_ROOT,run,state,checkpoint_backup=backup/full.name)
        self.assertIn(state['history'][-1]['path'],missing)
        (backup/full.name).write_bytes(saved)
        files,missing=resolve_models(SOURCE_ROOT,run,state,checkpoint_backup=backup/full.name)
        self.assertFalse(missing)

    def test_swap_preserves_original_bytes_adam_rng_lineage_and_future_evidence(self):
        run,backup,state,receipt=self.fixture();full=SOURCE_ROOT/state['history'][-1]['path'];saved=full.read_bytes()
        before=read(run/'state.json');old_state_bytes=(run/'state.json').read_bytes()
        files,missing=resolve_models(SOURCE_ROOT,run,state);self.assertFalse(missing)
        with run_lock(run):result=publish(SOURCE_ROOT,run,state,{},files,receipt)
        archive=SOURCE_ROOT/result['archive']
        self.assertEqual((archive/'state.json').read_bytes(),old_state_bytes)
        self.assertEqual((archive/'iteration-0160/report.json').read_bytes(),b'future evidence')
        self.assertFalse((run/'iteration-0160').exists());self.assertFalse((archive/'.active-run.json').exists())
        self.assertEqual(full.read_bytes(),saved);self.assertEqual((archive/full.name).read_bytes(),saved)
        restored=read(run/'state.json');self.assertEqual(restored['iteration'],160)
        model,raw,changed=load_resume(SOURCE_ROOT,run,restored,state['signature'])
        self.assertFalse(changed)
        _,old=load_checkpoint(archive/full.name);self.assertTrue(_same_payload(raw,old))
        opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
        self.assertTrue(_same_payload(raw['optimizer'],opt.state_dict()));self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
        self.assertEqual(restored['history'],state['history']);self.assertEqual(restored['population'],state['population'])
        self.assertEqual(read(run/'best.json')['checkpoint'],state['champion'])

    def test_live_progress_change_cannot_be_overwritten(self):
        run,backup,state,receipt=self.fixture();files,missing=resolve_models(SOURCE_ROOT,run,state)
        later=read(run/'state.json');later['iteration']=180;write_json(run/'state.json',later)
        with run_lock(run):
            with self.assertRaisesRegex(ValueError,'advanced'):publish(SOURCE_ROOT,run,state,{},files,receipt)
        self.assertEqual(read(run/'state.json')['iteration'],180)


class ReducedPopulationTests(unittest.TestCase):
    fixture=RestartTests.fixture

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def audited_state(self):
        state,_,_=reconstruct(SOURCE_ROOT/EXPORT)
        missing=[e['path'] for e in state['population'] if e['generation'] in REVIEWED_MISSING_GENERATIONS]
        self.assertEqual(len(missing),17)
        return state,missing

    def test_explicit_reduction_keeps_history_counters_and_capacity(self):
        state,missing=self.audited_state();before=copy.deepcopy(state)
        receipt=reduce_population(state,missing)
        self.assertEqual((receipt['original_count'],receipt['retained_count'],receipt['target_capacity']),(64,47,64))
        self.assertEqual(receipt['distinct_models_to_full'],17)
        self.assertEqual([e['generation'] for e in receipt['dropped']],
            [23,24,33,34,75,115,146,147,148,149,150,151,153,154,155,156,157])
        self.assertEqual({k:v for k,v in state.items() if k!='population'},
                         {k:v for k,v in before.items() if k!='population'})
        self.assertEqual(state['signature']['config']['population'],dict(recent=16,anchors=48))
        self.assertIn(0,receipt['retained_generations']);self.assertIn(160,receipt['retained_generations'])
        restored=copy.deepcopy(before);receipt=reduce_population(restored,missing[:5])
        self.assertEqual(receipt['retained_count'],59) # Backups can recover some of the reported models.

    def test_required_or_unreviewed_missing_models_cannot_be_dropped(self):
        state,missing=self.audited_state()
        critical=[state['history'][-1],state['evaluation_anchor'],state['champion']]
        additional=next(e for e in state['population'] if e['generation']==4)
        for path in [e['path'] for e in critical]+[additional['path'],'unknown.pt']:
            with self.subTest(path=path):
                changed=copy.deepcopy(state)
                with self.assertRaises(ValueError):reduce_population(changed,missing+[path])
                self.assertEqual(changed,state)
        for change in ('future','capacity','generation'):
            changed=copy.deepcopy(state)
            if change=='future':changed['population'][1]['generation']=161
            elif change=='capacity':changed['signature']['config']['population']['anchors']=31
            else:changed['iteration']=179
            before=copy.deepcopy(changed)
            with self.assertRaises(ValueError):reduce_population(changed,missing)
            self.assertEqual(changed,before)

    def test_cli_requires_explicit_reduction_and_apply_independently(self):
        run,_,_,receipt=self.fixture();state,missing=self.audited_state()
        receipt['run']=run.relative_to(SOURCE_ROOT).as_posix()
        before=(run/'state.json').read_bytes()
        for flags in ([],['--apply'],['--allow-reduced-population'],['--allow-reduced-population','--apply']):
            with self.subTest(flags=flags),redirect_stdout(io.StringIO()), \
                 patch('sys.argv',['joint7_restart160.py','--root',str(SOURCE_ROOT),'--run',receipt['run'],*flags]), \
                 patch('joint7_restart160.reconstruct',return_value=(copy.deepcopy(state),{},copy.deepcopy(receipt))), \
                 patch('joint7_restart160.resolve_models',return_value=({},missing)), \
                 patch('joint7_restart160.publish',side_effect=lambda root,run,s,h,f,r:dict(r,applied=True)) as publish_mock:
                if flags==['--apply']:
                    with self.assertRaisesRegex(ValueError,'missing files'):main()
                else:self.assertEqual(main(),0 if '--allow-reduced-population' in flags else 2)
                self.assertEqual(publish_mock.call_count,int('--apply' in flags and '--allow-reduced-population' in flags))
                if publish_mock.called:
                    restored=publish_mock.call_args.args[2];report=publish_mock.call_args.args[5]
                    self.assertEqual(len(restored['population']),47)
                    self.assertEqual((report['ready'],report['blocking_missing'],report['missing']),(True,[],missing))
            self.assertEqual((run/'state.json').read_bytes(),before)

    def test_real_47_model_restore_preserves_learner_bytes_adam_rng_and_receipt(self):
        run,backup,state,receipt=self.fixture(population=dict(recent=16,anchors=48));template,missing=self.audited_state()
        full=SOURCE_ROOT/state['history'][-1]['path'];saved=full.read_bytes()
        model,raw=load_checkpoint(full);training=raw['training']
        entries={e['generation']:e for e in state['population']}
        for original in template['population']:
            generation=original['generation']
            if generation in entries:continue
            if generation in REVIEWED_MISSING_GENERATIONS:
                identity=digest(dict(missing=generation));checksum=None
                path=run/'population'/('weights-'+identity+'.pt')
            else:
                with torch.no_grad():next(model.parameters()).reshape(-1)[0].add_(.001)
                identity=model_id(model);path=run/'population'/('weights-'+identity+'.pt')
                save_checkpoint(path,model,step=generation,training=training);checksum=sha(path)
            entries[generation]=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=checksum,
                model_id=identity,generation=generation,weights_only=True)
        state['population']=[entries[g] for g in sorted(entries)]
        state['history']=sorted([e for g,e in entries.items() if g!=160]+[state['history'][-1]],key=lambda e:e['generation'])
        files,missing=resolve_models(SOURCE_ROOT,run,state)
        self.assertEqual(len(missing),17);self.assertEqual(len(state['population']),64) # Default stays strict.
        receipt['population_recovery']=reduce_population(state,missing)
        with run_lock(run):result=publish(SOURCE_ROOT,run,state,{},files,receipt)
        restored=read(run/'state.json');self.assertEqual(len(restored['population']),47)
        self.assertEqual(full.read_bytes(),saved)
        self.assertEqual((SOURCE_ROOT/result['archive']/full.name).read_bytes(),saved)
        self.assertEqual(restored['restart_origin']['population_recovery'],receipt['population_recovery'])
        self.assertEqual(read(run/'restart160.json')['population_recovery'],receipt['population_recovery'])
        learner,raw,changed=load_resume(SOURCE_ROOT,run,restored,state['signature']);self.assertFalse(changed)
        optimizer=torch.optim.AdamW(learner.parameters());resume_optimizer(raw,optimizer)
        self.assertTrue(_same_payload(raw['optimizer'],optimizer.state_dict()))
        self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()))
        self.assertEqual(restored['games_done'],51200)
        self.assertFalse((run/'iteration-0160').exists())
        self.assertTrue((SOURCE_ROOT/result['archive']/'iteration-0160/report.json').is_file())
        self.assertTrue(all((SOURCE_ROOT/e['path']).is_file() for e in restored['population']))

    def test_committed_pool_refills_from47_to64_across_restart(self):
        state,missing=self.audited_state();reduce_population(state,missing)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);original_run=Path('artifacts/step5/v4-selfplay-joint')
            # Real selection/commit code with tiny file markers; checkpoint payload
            # integrity is exercised separately by the 47-model restoration test.
            for e in state['history']+state['population']+[state['evaluation_anchor'],state['champion']]:
                e['path']=(run/Path(e['path']).relative_to(original_run)).relative_to(SOURCE_ROOT).as_posix()
            for e in state['population']+[state['history'][-1],state['evaluation_anchor']]:
                path=SOURCE_ROOT/e['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            anchor=state['evaluation_anchor']['model_id'];counts=[47]
            with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,existing=None:copy.deepcopy(existing or e)):
                for generation in range(161,179):
                    path=run/f'generation-{generation:04d}.pt';path.write_bytes(b'fixture')
                    e=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),
                        model_id=digest(dict(new_generation=generation)),generation=generation)
                    state['history'].append(e);state['iteration']=generation
                    state['games_done']+=320
                    update_population(SOURCE_ROOT,run,state,state['signature']['config']['population'])
                    counts.append(len(state['population']))
                    self.assertIn(anchor,{p['model_id'] for p in state['population']})
                    self.assertFalse(REVIEWED_MISSING_GENERATIONS.intersection(p['generation'] for p in state['population']))
                    self.assertTrue(all(p['generation']<=generation for p in state['population']))
                    self.assertEqual(len({p['model_id'] for p in state['population']}),len(state['population']))
                    write_json(run/'state.json',state);state=read(run/'state.json')
            self.assertEqual(counts,list(range(47,65))+[64])
            self.assertEqual(counts[5],52);self.assertEqual(counts[17],64)
            self.assertEqual(state['signature']['config']['population'],dict(recent=16,anchors=48))


if __name__=='__main__':unittest.main()
