"""P5 authored fixtures prove interfaces/integrity, never playing strength."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fixtures import result as official_result
from test_deployment import candidate
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, digest, compatibility
from brass_ai.learning.replay import terminal_targets
from brass_ai.personality.contracts import GOALS, source_id
from brass_ai.personality.deployment import roster
from brass_ai.personality.deployment.contracts import identity as p4_identity
from brass_ai.personality.search.contracts import Profile, identity as p2_identity
from brass_ai.personality.repertoire.contracts import identity as p3_identity
from brass_ai.step5.league import implementation_id, write_json
from brass_ai.personality.difficulty import contracts as c, panels, collection, evaluation, ui


def named(groups=2, members=2):
    value = candidate(groups,members)
    for i,g in enumerate(value['archetypes']):
        g['name'] = f'Fixture style {i+1}'; g['description'] = 'Authored interface fixture, not a learned personality.'
    return value


def panel(value, n=2):
    profiles=c.default_profiles()
    return dict(id='fixture-panel',roster_id=value['id'],fixture=True,stage='calibration',profiles=profiles,
                gates=asdict(c.Gates()),seeds=list(range(n)),hardware_label='fixture-cpu',tables=1,
                jobs=panels.jobs_for(value,list(range(n)),2,profiles,repetitions=1),
                baseline=dict(identity=c.STOCK),partition='validation',forbidden_groups=[])


def record(job, win=True):
    seat=job['seat'] if job['seat'] is not None else job['higher_seats'][0]
    scores=[(50+s,10,1) for s in range(4)]; scores[seat]=(100 if win else 10,10,1)
    return dict(job=job,failure=None,result=official_result(scores),fixture=True,fixture_direct=False,
                both_eras=True,turns=[],environment=dict(device='fixture-cpu'),telemetry={},seconds=.1,
                baseline_failures=[],behavior={},intent_display=[])


def own_goal(family='industry_sequence', confidence=.9, seat=0):
    goal=dict(schema=GOALS,family=family,industry='CottonMill' if family=='industry_sequence' else 'none',
              region='unspecified',horizon=4,confidence=confidence,age=0,created_turn='fixture',revision_reason='fixture')
    return dict(seat=seat,profile=Profile().to_dict(),goal=goal)


class DifficultyTests(unittest.TestCase):
    def test_profiles_preserve_model_reward_prior_and_training_settings(self):
        v=named();p=c.default_profiles();before=copy.deepcopy(v)
        self.assertEqual(p,json.loads((SOURCE_ROOT/'configs/personality/difficulty-1.0.0.json').read_text()))
        for tier,seconds in zip(c.TIERS,(2.,5.,12.)):
            search=c.resolve_search(v,p,tier).to_dict()
            self.assertEqual(search,dict(v['search'],seconds=seconds,workers=2,cpu_role='laptop'))
        self.assertEqual(v,before)
        p['near_best_tolerance']=.2
        with self.assertRaisesRegex(ValueError,'Unsupported'):c.validate_profiles(p)

    def test_independent_p5_identity_preserves_p0_p4_and_step5(self):
        def upstream():return (implementation_id(),source_id(),p2_identity.__wrapped__(),p3_identity(),p4_identity.__wrapped__(),compatibility())
        original=Path.read_text; before=upstream(); own=c.identity.__wrapped__()
        def changed(path,*a,**kw):
            value=original(path,*a,**kw)
            return value+'\n# fixture P5 change\n' if path.parent.name=='difficulty' and path.suffix=='.py' else value
        with patch.object(Path,'read_text',changed):
            self.assertEqual(before,upstream());self.assertNotEqual(own,c.identity.__wrapped__())

    def test_full_panel_covers_all_members_mixtures_tiers_and_direct_seats(self):
        v=named(16,8);p=panel(v);jobs=p['jobs']
        self.assertEqual(jobs,panels.jobs_for(v,[0,1],2,p['profiles'],repetitions=1))
        for m in v['members']:
            for tier in c.TIERS:
                own=[j for j in jobs if j['lineup']=='frozen-neural' and j['tier']==tier and j['arm']['id']==m['id']]
                self.assertEqual(len(own),8);self.assertEqual({j['seat'] for j in own},set(range(4)))
            baseline=[j for j in jobs if j['lineup']=='original-hard' and j['arm']['id']==m['id']]
            self.assertEqual(len(baseline),8)
            self.assertTrue(all(sum(s['kind']=='stock' for s in j['table'])==3 for j in baseline))
        direct=[j for j in jobs if j['lineup']=='adjacent-direct']
        self.assertEqual(len(direct),2*16*6*2)
        self.assertTrue(all(len(j['higher_seats'])==2 for j in direct))
        self.assertEqual(len({j['index'] for j in jobs}),len(jobs))

    def test_mixture_draws_are_stable_seat_specific_and_matched_across_tiers(self):
        v=named(2,8);p=panel(v,16);jobs=p['jobs']
        for group in v['archetypes']:
            part=[j for j in jobs if j['lineup']=='frozen-neural' and j['arm']['kind']=='mixture' and j['arm']['id']==group['id']]
            self.assertEqual({j['table'][j['seat']]['member'] for j in part},set(group['members']))
            for setup in p['seeds']:
                for seat in range(4):
                    matched=[j for j in part if j['seed']==setup and j['seat']==seat]
                    self.assertEqual(len({j['table'][seat]['member'] for j in matched}),1)
                    self.assertEqual(len({j['sampling_seed'] for j in matched}),1)
        self.assertEqual(jobs,panels.jobs_for(v,p['seeds'],2,p['profiles'],repetitions=1))

    def test_official_ties_and_tiebreakers_not_raw_vp(self):
        p=panel(named());job=next(j for j in p['jobs'] if j['seat']==0 and j['lineup']=='original-hard')
        r=record(job);r['result']=official_result([(100,10,1),(100,10,1),(90,20,99),(80,0,0)])
        row=evaluation.subject_rows([r])[0]
        self.assertEqual(row['win'],.5);self.assertEqual(row['rank'],1)
        r['result']=official_result([(100,9,99),(100,10,1),(90,20,99),(80,0,0)])
        self.assertEqual(evaluation.subject_rows([r])[0]['win'],0)

    def test_rotations_cluster_before_intervals_and_multiplicity_widens_them(self):
        rows=[dict(setup=str(i),win=(i%7)/7) for i in range(40)]
        grouped=evaluation.grouped(rows*4,'win');self.assertEqual(len(grouped),40)
        a=evaluation.interval(grouped.values(),c.Gates(),1)
        b=evaluation.interval(grouped.values(),c.Gates(),580)
        self.assertLess(b['lower'],a['lower']);self.assertGreater(b['upper'],a['upper'])
        self.assertIsNone(evaluation.interval([.5]*4,c.Gates(),580)['lower'])
        # A few identical fixture outcomes never imply certainty/equivalence.
        constant=evaluation.interval([0.]*32,c.Gates(),580,width=2)
        self.assertLess(constant['lower'],-.05)

    def test_weak_easy_member_cannot_hide_behind_archetype_or_vp(self):
        v=named();p=panel(v,40);weak=v['members'][0]['id'];records=[]
        for j in p['jobs']:
            if j['lineup']=='original-hard':records.append(record(j,j['arm']['id']!=weak))
        report=evaluation.automatic(v,p,records)
        self.assertFalse(report['members'][weak]['easy_original_hard']['passed'])
        self.assertTrue(report['members'][v['members'][1]['id']]['easy_original_hard']['passed'])
        self.assertFalse(report['checks']['every_easy_member'])
        self.assertTrue(report['checks']['every_easy_mixture'])
        self.assertFalse(report['checks']['same_tier_consistency'])
        self.assertFalse(report['checks']['tier_ordering'])

    def test_invalid_tier_hint_and_search_profiles_rejected(self):
        for key,value in [('workers',8),('cpu_role','unknown'),('hint_mode','private-debug')]:
            p=c.default_profiles();p[key]=value
            with self.assertRaises(ValueError):c.validate_profiles(p)
        p=c.default_profiles();p['tiers']['hard']['seconds']=24
        with self.assertRaisesRegex(ValueError,'2/5/12'):c.validate_profiles(p)

    def test_preview_or_fixture_cannot_enter_real_artifact_gate(self):
        v=named()
        with patch.object(roster,'validate_sources',return_value=2),patch.object(panels,'load_ready',return_value=(None,None,dict(scope='Fixture'),dict(origin=dict(fixture=True)),None)):
            self.assertEqual(panels.source_check(SOURCE_ROOT,v,True),2)
            with self.assertRaisesRegex(ValueError,'Fixture'):panels.source_check(SOURCE_ROOT,v,False)
        with patch.object(roster,'validate_sources',return_value=2),patch.object(panels,'load_ready',return_value=(None,None,dict(scope='Readiness'),dict(origin={}),None)):
            with self.assertRaisesRegex(ValueError,'qualified real'):panels.source_check(SOURCE_ROOT,v,False)

    def test_menu_is_16_public_choices_and_cannot_claim_fixture_qualification(self):
        v=named(16,8);m=ui.menu(v,fixture=True)
        self.assertEqual(len(m['choices']),16);self.assertTrue(all(not g['enabled'] and g['tiers']==[] for g in m['choices']))
        text=json.dumps(m)
        for member in v['members']:
            self.assertNotIn(member['id'],text);self.assertNotIn(member['candidate_id'],text)
        self.assertNotIn('centroid',text);self.assertNotIn('model_id',text)
        self.assertFalse(m['release_qualified'])
        with self.assertRaisesRegex(ValueError,'not qualified'):ui.selection(m,v['archetypes'][0]['id'],'easy')
        self.assertTrue(ui.selection(m,v['archetypes'][0]['id'],'easy',preview=True)['preview'])
        with self.assertRaisesRegex(ValueError,'16 styles'):ui.menu(named())
        fake=dict(p5_qualified=True,fixture=True,roster_id=v['id'],profiles=c.default_profiles())
        self.assertFalse(ui.menu(v,fake)['choices'][0]['enabled'])

    def test_hint_override_does_not_inherit_measured_difficulty(self):
        v=named(16,8);menu=ui.menu(v,fixture=True)
        # An explicitly authored UI DTO exercises selection only, not a fake
        # four-field qualification report that could accidentally enable release.
        for choice in menu['choices']:choice.update(enabled=True,tiers=list(c.TIERS))
        menu['qualified_hint_mode']='public'
        with self.assertRaisesRegex(ValueError,'hint mode'):ui.selection(menu,v['archetypes'][0]['id'],'hard','intent-hints')

    def test_public_intent_never_reads_private_goal_or_raw_view(self):
        a=ui.IntentDisplay(0);b=ui.IntentDisplay(0)
        kwargs=dict(era='CanalEra',turn=1,description='Public style',action=ui.PublicAction('develop'))
        self.assertEqual(a.update(**kwargs),b.update(**kwargs,own_intent=dict(private_cards='SECRET',future_deck='SECRET')))
        self.assertNotIn('SECRET',json.dumps(b.update(**kwargs)))
        with self.assertRaisesRegex(ValueError,'Raw observations'):a.update(**dict(kwargs,turn=2,action={'own_hand':['SECRET']}))

    def test_opt_in_hints_scope_confidence_hysteresis_hide_and_one_update(self):
        display=ui.IntentDisplay(0);base=dict(era='CanalEra',description='Public style',mode='intent-hints')
        first=display.update(**base,turn=1,own_intent=own_goal());self.assertTrue(first['uncertain'])
        same=display.update(**base,turn=1,own_intent=own_goal('network_extension'))
        self.assertEqual(same,first)
        self.assertEqual(display.update(**base,turn=2,own_intent=own_goal('network_extension')),first)
        changed=display.update(**base,turn=3,own_intent=own_goal('network_extension'))
        self.assertNotEqual(changed,first)
        self.assertNotIn('confidence',changed);self.assertNotIn('region',changed)
        public=display.update(**dict(base,mode='public'),turn=3,own_intent=own_goal())
        self.assertEqual(public['text'],'Public style');self.assertFalse(public['uncertain'])
        self.assertIsNone(display.update(**dict(base,mode='hidden'),turn=3)['text'])
        with self.assertRaisesRegex(ValueError,'another seat'):display.update(**base,turn=4,own_intent=own_goal(seat=1))
        low=ui.IntentDisplay(0).update(**base,turn=1,own_intent=own_goal(confidence=.5))
        self.assertFalse(low['uncertain'])

    def test_human_review_requires_each_style_tier_two_raters_and_same_hint(self):
        v=named();p=panel(v);review=evaluation.review_template(p,v)
        self.assertFalse(evaluation.human_review(p,v,None)['passed'])
        self.assertFalse(evaluation.human_review(p,v,review)['passed'])
        for group in v['archetypes']:
            for tier in c.TIERS:
                for rater in ('a','b'):
                    review['sessions'].append(dict(session_id=group['id']+tier+rater,rater=rater,archetype_id=group['id'],tier=tier,
                        hint_mode='public',completed=True,players=4,tactically_sound=True,
                        ratings={k:4 for k in evaluation.RATING_FIELDS},evidence='AUTHORED TEST, not a human session'))
        self.assertTrue(evaluation.human_review(p,v,review)['passed'])
        review['sessions'][0]['tactically_sound']=False
        self.assertFalse(evaluation.human_review(p,v,review)['passed'])
        review['sessions'][0]['hint_mode']='intent-hints'
        with self.assertRaisesRegex(ValueError,'hint mode'):evaluation.human_review(p,v,review)

    def test_missing_failed_stock_and_fixture_evidence_never_qualifies(self):
        v=named();p=panel(v);r=record(p['jobs'][1]);r['failure']=dict(kind='stock',type='Fixture',message='failed')
        r['result']=None;r['baseline_failures']=[r['failure']]
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=Path(tmp)
            write_json(out/'manifest.json',p)
            with patch.object(collection,'read_chunks',return_value=(p,v,iter([r]),{1},[])),patch.object(evaluation,'public_packet',return_value=dict(fixture=True)):
                report=evaluation.check(SOURCE_ROOT,out,[],out/'checked')
            self.assertFalse(report['p5_qualified']);self.assertFalse(report['release_qualified'])
            self.assertFalse(report['checks']['complete_panel']);self.assertEqual(len(report['stock_failures']),1)
            self.assertEqual(report['qualified_tiers'],[]);self.assertIn('FIXTURE',report['status'])

    def test_chunk_pins_reservations_and_duplicate_jobs_are_enforced(self):
        from brass_ai.step5.league import seeds_for
        v=named();p=panel(v)
        seeds,_=seeds_for(151000000,2,'validation')
        p['jobs']=panels.jobs_for(v,seeds,2,p['profiles'],repetitions=1)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);chunk=directory/'chunk';chunk.mkdir()
            write_json(directory/'manifest.json',p)
            r=c.sealed(dict(record(p['jobs'][0]),format=c.RECORD,contract=c.identity(),panel_id=p['id']))
            path=chunk/'game.json';write_json(path,r)
            item=dict(job=0,path=path.name,sha256=sha(path),id=r['id'])
            receipt=dict(format=c.CHUNK,contract=c.identity(),panel_id=p['id'],fixture=True,
                fixture_direct=False,tables=p['tables'],environment=r['environment'],
                panel_sha256=sha(directory/'manifest.json'),records=[item])
            write_json(chunk/'manifest.json',c.sealed(receipt))
            reservation=dict(panel_id=p['id'],fixture=True,jobs=[0])
            write_json(chunk/'reservation.json',reservation)
            def read(chunks):return list(collection.read_chunks(SOURCE_ROOT,directory,chunks)[2])
            with patch.object(panels,'load',return_value=(p,v)):
                self.assertEqual(len(read([chunk])),1)
                with self.assertRaisesRegex(ValueError,'Duplicate'):read([chunk,chunk])
                write_json(chunk/'reservation.json',dict(reservation,jobs=[0,1]))
                with self.assertRaisesRegex(ValueError,'reserved job'):read([chunk])
                write_json(chunk/'reservation.json',reservation)
                path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'evidence changed'):read([chunk])

    def test_public_packet_omits_internal_ids_and_private_fields(self):
        v=named();p=panel(v);job=p['jobs'][0];r=record(job)
        r['behavior']={str(job['seat']):dict(public_turns=[{'actions':['build']}],public_boards=[{'era':'CanalEra'}],
            internal_member='SECRET',private_goal='SECRET',own_hand=['SECRET'])}
        r['intent_display']=[dict(seat=job['seat'],era='CanalEra',turn=1,text='Investing in industry',
                                 uncertain=False,private_confidence='SECRET')]
        packet=evaluation.public_packet(p,v,[r]);text=json.dumps(packet)
        self.assertEqual(len(packet['examples']),1)
        self.assertNotIn('SECRET',text);self.assertNotIn(job['arm']['id'],text)
        self.assertNotIn('result',text);self.assertNotIn('win',text)


if __name__=='__main__':unittest.main()
