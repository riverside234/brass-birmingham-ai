"""P4 contract/evidence checks using authored fixtures, never release evidence."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from personality_fixtures import episode, parent
from brass_ai.bank import sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json
from brass_ai.learning.reward import specification
from brass_ai.personality.contracts import source_id
from brass_ai.personality.search.contracts import Profile, controller, search_config
from brass_ai.personality.repertoire import archive, discovery
from brass_ai.personality.repertoire.contracts import identity as repertoire_identity
from brass_ai.personality.deployment import roster, panels, evaluation
from brass_ai.personality.deployment.contracts import (ROSTER, ARCHETYPE, MEMBER, SAMPLING,
    Gates, identity, sealed, validate_roster, resolve)
from brass_ai.step5.league import write_json, implementation_id


def entries(groups=16, members=8, *, network='fixture', mode='searched', parent_id='parent', pool='pool'):
    cfg=search_config(specification()); result=[];elites=[]
    for group in range(groups):
        for member in range(members):
            profile=Profile(style=('neutral','industrialist','networker','supplier','investor')[group%5],
                            alpha=.15, hysteresis=(group*members+member+1)/10000)
            c=controller(network,parent_id,profile,cfg,pool,decision_mode=mode)
            point=[10.*group,float(member)*.2]
            result.append(dict(candidate_id=c['candidate_id'],controller=c,centroid=point,dispersion=.01,qualified=True,
                               quality=dict(qualified=True,setups=8,mean_win=.3,control_lower=0.)))
            if not member:elites.append(c['candidate_id'])
    return result,elites


def candidate(groups=2,members=2):
    es,reps=entries(groups,members); neighborhoods=roster.neighborhoods(es,reps,members,Gates())
    by_id={e['candidate_id']:e for e in es}; gs=[];vs=[]
    for g in neighborhoods:
        gid='archetype-'+digest(g['representative'])[:16];mids=['variant-'+digest(c)[:16] for c in g['members']]
        gs.append(dict(format=ARCHETYPE,id=gid,name=None,description=None,representative=mids[0],members=mids,supported_tiers=[],
                       neighborhood=dict(radius=g['radius'],nearest_other=g['nearest_other']),
                       sampling=dict(version=SAMPLING,weights=[1/members]*members)))
        for mid,cid in zip(mids,g['members']):
            e=by_id[cid];vs.append(dict(format=MEMBER,id=mid,archetype_id=gid,candidate_id=cid,model_id='fixture',
                                      profile=e['controller']['profile'],centroid=e['centroid'],dispersion=e['dispersion'],tier_settings={},fallback=None))
    controls=[controller('fixture','parent',Profile(style=s,alpha=0. if s=='neutral' else .15),search_config(specification()),'pool')
              for s in ('neutral','industrialist','networker','supplier','investor')]
    return sealed(dict(format=ROSTER,contract=identity(),counts=dict(archetypes=groups,per_archetype=members),
                       archetypes=gs,members=vs,shared_model=dict(ready='fixture-ready',ready_id='ready',model_id='fixture'),
                       controls=controls,search=search_config(specification()).to_dict(),decision_mode='searched',pool_id='pool',
                       full_size=groups==16 and members==8,release_qualified=False,p4_qualified=False,dependencies=[],forbidden_groups=[],gates=asdict(Gates())))


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_roster_is_exactly_16_by_8_actual_controllers_no_padding(self):
        value=candidate(16,8);validate_roster(value)
        self.assertEqual(len(value['members']),128)
        self.assertEqual(len({m['model_id'] for m in value['members']}),1)
        for group in value['archetypes']:
            self.assertIn(group['representative'],group['members'])
            self.assertEqual(group['sampling']['weights'],[.125]*8)
        bad=copy.deepcopy(value);bad['members'][1]['candidate_id']=bad['members'][0]['candidate_id']
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_roster(bad)
        bad=copy.deepcopy(value);bad['members'][0]['model_id']='new-distilled'
        with self.assertRaisesRegex(ValueError,'Mixed'):validate_roster(bad)
        bad=copy.deepcopy(value);bad['archetypes'][0]['members'][1]=bad['archetypes'][1]['members'][0]
        with self.assertRaisesRegex(ValueError,'Overlapping'):validate_roster(bad)

    def test_variants_need_repeated_separation_and_nearby_theme(self):
        es,reps=entries(2,8);gates=Gates()
        a=roster.neighborhoods(es,reps,8,gates)
        self.assertEqual(a,roster.neighborhoods(es,reps,8,gates))
        es[1]['centroid']=es[0]['centroid']
        with self.assertRaisesRegex(ValueError,'7/8'):roster.neighborhoods(es,reps,8,gates)
        es,reps=entries(2,8);es[1]['qualified']=False
        with self.assertRaisesRegex(ValueError,'7/8'):roster.neighborhoods(es,reps,8,gates)
        es,reps=entries(2,8);es[0]['dispersion']=5.
        with self.assertRaisesRegex(ValueError,'variation'):roster.neighborhoods(es,reps,8,gates)

    def test_neighborhood_noise_is_matched_to_each_pair(self):
        es=[dict(candidate_id=k,centroid=[x,0.],dispersion=noise,qualified=True,quality=dict(control_lower=0.))
            for k,x,noise in [('a',0.,.01),('a-wide',3.,.6),('a-small',.1,.01),
                              ('b',10.,.01),('b1',10.1,.01),('b2',10.2,.01)]]
        groups=roster.neighborhoods(es,['a','b'],3,Gates())
        self.assertEqual(set(groups[0]['members']),{'a','a-wide','a-small'})

    def test_online_resolution_uses_tested_profile_not_descriptor(self):
        value=candidate();member=value['members'][0];p,cfg=resolve(value,member['id'])
        self.assertEqual(p.to_dict(),member['profile']);self.assertEqual(cfg.to_dict(),value['search'])
        value['members'][0]['centroid']=[9999.,-9999.]
        self.assertEqual(resolve(value,member['id'])[0],p)
        with self.assertRaisesRegex(ValueError,'Unknown'):resolve(value,'missing')

    def test_seeded_rotated_panels_and_real_uniform_mixture(self):
        value=candidate();jobs=panels.jobs_for(value,list(range(20)),3)
        self.assertEqual(jobs,panels.jobs_for(value,list(range(20)),3))
        for member in value['members']:
            rows=[j for j in jobs if j['p4'].get('member')==member['id']]
            self.assertEqual(len(rows),80)
            for seed in range(20):self.assertEqual({j['neural_seats'][0] for j in rows if j['seed']==seed},{0,1,2,3})
        for group in value['archetypes']:
            rows=[j for j in jobs if j['p4']['role']=='mixture' and j['p4']['group']==group['id']]
            self.assertEqual({j['p4']['selected_member'] for j in rows},set(group['members']))
        self.assertEqual(len({j['index'] for j in jobs}),len(jobs))
        by_seed=[j for j in jobs if j['seed']==0 and j['neural_seats']==[0] and j['p4']['role'] in ('control','member')]
        self.assertEqual(len({j['sampling_seed'] for j in by_seed}),1)
        self.assertEqual(len({tuple(j['parent_indices']) for j in by_seed}),1)

    def test_public_variety_does_not_count_payment_or_hidden_card_changes(self):
        game=episode(4);before=evaluation.public_behavior(game,0);changed=copy.deepcopy(game)
        for row in changed['records']:
            row['view']['own_hand']=['private-hand'];row['seconds']=999
            for choice in row['choices']:choice.update(id='different-id',card='private-card')
        self.assertEqual(before,evaluation.public_behavior(changed,0))
        import json
        visible=json.dumps(evaluation.public_behavior(changed,0,include_boards=True))
        self.assertNotIn('private-hand',visible);self.assertNotIn('private-card',visible)
        self.assertNotIn('own_hand',visible);self.assertNotIn('policy_id',visible)
        self.assertIn('coal_market',visible)
        row=next(r for r in changed['records'] if r['seat']==0)
        row['choices'][row['chosen']]['kind']='pass'
        self.assertNotEqual(before['openings'],evaluation.public_behavior(changed,0)['openings'])

    def test_own_win_quality_does_not_pass_on_high_vp_or_one_lucky_game(self):
        g=Gates();control=[dict(setup=str(i),quality=dict(win_share=.25,vp=100.,placement=2.5)) for i in range(8)]
        same=copy.deepcopy(control);self.assertTrue(evaluation.quality(same,control,g)['passed'])
        for r in same:r['quality'].update(win_share=0.,vp=1000.,placement=4.)
        self.assertFalse(evaluation.quality(same,control,g)['passed'])
        self.assertFalse(evaluation.quality(control[:1],control,g)['passed'])
        for r in same:r['quality'].update(win_share=.5,vp=100.,placement=3.)
        self.assertFalse(evaluation.quality(same,control,g)['passed'])

    def test_replayability_must_beat_representative_randomness_in_both_eras(self):
        def rows(varied,rail=True):
            out=[]
            for setup in range(8):
                for i in range(8):
                    signatures={e:str(i) if varied and (e=='CanalEra' or rail) else 'same' for e in evaluation.ERAS}
                    out.append(dict(setup=str(setup),behavior=dict(openings=signatures,later=signatures)))
            return out
        g=Gates();self.assertFalse(evaluation.variety(rows(False),rows(False),g)['passed'])
        self.assertTrue(evaluation.variety(rows(True),rows(False),g)['passed'])
        self.assertFalse(evaluation.variety(rows(True,False),rows(False),g)['passed'])

    def test_blind_review_is_bound_to_panel_and_requires_two_raters_no_tactical_fail(self):
        v=candidate();p=dict(id='panel',labels=[dict(archetype_id=g['id'],name='Name '+g['id'],description='Observed behavior') for g in v['archetypes']])
        rows=[]
        for group in v['archetypes']:
            for i in range(8):
                rows.append(dict(index=len(rows),arm=dict(role='member',member=group['members'][i%2],group=group['id']),
                                 behavior=dict(public_turns={e:[] for e in evaluation.ERAS})))
        p['jobs']=[dict(index=r['index'],p4=r['arm'],seed=r['index']//4,neural_seats=[r['index']%4]) for r in rows]
        packet=evaluation.blind_packet(v,p,rows,Gates())
        truth={digest(dict(panel=p['id'],job=r['index'])):r['arm']['group'] for r in rows}
        review=dict(format='personality-blind-review/1.0.0',panel_id=p['id'],packet_id=packet['id'],labels=p['labels'],
                    ratings=[dict(example=e['id'],rater=str(rater),guess=truth[e['id']],tactically_sound=True) for e in packet['examples'] for rater in range(2)])
        self.assertTrue(evaluation.human_review(v,p,packet,rows,review,Gates())['passed'])
        review['ratings'][0]['tactically_sound']=False
        self.assertFalse(evaluation.human_review(v,p,packet,rows,review,Gates())['passed'])
        review['ratings'].append(review['ratings'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):evaluation.human_review(v,p,packet,rows,review,Gates())
        self.assertFalse(evaluation.human_review(v,p,packet,rows,None,Gates())['passed'])
        review['panel_id']='different'
        with self.assertRaisesRegex(ValueError,'another'):evaluation.human_review(v,p,packet,rows,review,Gates())

    def test_p4_sources_do_not_change_step5_or_p0_p3_contracts(self):
        self.assertEqual(implementation_id(),'f739475371ce3d0a7dfe1c505780513a130474de3c17b8bb4501da0b67b017b3')
        # P0/P1 fixes legitimately change their source contract. P4 remains a
        # separate package; it must not enter the base source-file set.
        base=SOURCE_ROOT/'src/brass_ai/personality'
        self.assertEqual(source_id(),digest({p.name:p.read_text(encoding='utf-8') for p in sorted(base.glob('*.py'))}))
        self.assertTrue(identity()['source']);self.assertTrue(repertoire_identity()['source'])

    def test_blind_cases_cover_seats_setups_members_and_do_not_change_with_chunk_order(self):
        v=candidate(16,8);jobs=panels.jobs_for(v,list(range(8)),3)
        selected=evaluation.blind_jobs(v,jobs,Gates())
        self.assertEqual(selected,evaluation.blind_jobs(v,list(reversed(jobs)),Gates()))
        for group in v['archetypes']:
            cases=[j for j in jobs if j['index'] in selected and j['p4']['group']==group['id']]
            self.assertEqual(len(cases),8)
            self.assertEqual({j['seed'] for j in cases},set(range(8)))
            self.assertEqual({j['neural_seats'][0] for j in cases},set(range(4)))
            self.assertEqual({j['p4']['member'] for j in cases},set(group['members']))
        # A partial check may show a missing case; it must never replace it with
        # a later case whose board snapshots were not prespecified for review.
        remaining=[j for j in jobs if j['index'] not in selected][:1]
        rows=[dict(index=j['index'],arm=j['p4'],behavior=dict(public_turns={},public_boards={})) for j in remaining]
        self.assertEqual(evaluation.blind_packet(v,dict(id='panel',jobs=jobs),rows,Gates())['examples'],[])

    def test_immutable_selection_pins_single_weight_and_rejects_mixed_generation(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=Path(tmp);par=parent();es,reps=entries(2,2,parent_id=model_id(par))
            ready=out/'ready';ready.mkdir();write_json(ready/'ready.json',dict(fixture=True));(ready/'model.pt').write_bytes(b'authored-weight-placeholder')
            ready_value=dict(id='ready',model=dict(model_id='fixture',path='model.pt'),reward=specification())
            proposal=out/'proposal';proposal.mkdir()
            p=dict(format=discovery.VERSION,contract=repertoire_identity(),candidates=[e['controller'] for e in es],control_id=es[0]['candidate_id'],
                   ready=ready.relative_to(SOURCE_ROOT).as_posix(),ready_id='ready',ready_sha256=sha(ready/'ready.json'),pool_id='pool')
            p['id']=digest(p);write_json(proposal/'manifest.json',p)
            enc=out/'encoder';enc.mkdir();write_json(enc/'manifest.json',dict(id='encoder'));(enc/'encoder.pt').write_bytes(b'fixture-encoder')
            a=dict(format=archive.VERSION,contract=repertoire_identity(),entries=es,elites=reps,proposal_id=p['id'],
                   proposal_path=proposal.relative_to(SOURCE_ROOT).as_posix(),pool_id='pool',sources=[],
                   encoder=dict(path=enc.relative_to(SOURCE_ROOT).as_posix(),id='encoder',sha256=sha(enc/'manifest.json')))
            a['id']=digest(a);src=out/'archive';src.mkdir();write_json(src/'manifest.json',a)
            enc_value=dict(id='encoder',report=dict(collapsed=False),training_groups=[],weights=dict(path='encoder.pt'))
            with patch.object(roster,'load_ready',return_value=(None,[par],ready_value,{},'pool')),patch.object(roster.descriptor,'load',return_value=(enc_value,None)):
                selected=roster.select(SOURCE_ROOT,src,out/'roster',groups=2,members=2)
                self.assertEqual(roster.load(out/'roster'),selected);self.assertFalse(selected['p4_qualified'])
                self.assertEqual(len([d for d in selected['dependencies'] if d['path'].endswith('/model.pt')]),1)
                self.assertFalse(list((out/'roster').glob('*.pt')))
                self.assertEqual(roster.validate_sources(SOURCE_ROOT,selected),1)
                with self.assertRaisesRegex(ValueError,'immutable'):roster.select(SOURCE_ROOT,src,out/'roster',groups=2,members=2)
                (ready/'model.pt').write_bytes(b'distilled-new-weights')
                with self.assertRaisesRegex(ValueError,'dependency changed'):roster.validate_sources(SOURCE_ROOT,selected)

    def test_panel_plan_excludes_used_setups_and_freezes_names_and_jobs(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=Path(tmp);v=candidate();source=out/'candidate';source.mkdir();write_json(source/'manifest.json',v)
            with patch.object(roster,'validate_sources',return_value=2):
                p=panels.plan(SOURCE_ROOT,source,out/'panel',setups=2)
                self.assertEqual(panels.load(SOURCE_ROOT,out/'panel',sources=True)[0],p)
                used=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=p['seeds'][0]))
                v['forbidden_groups']=[used];v=sealed({k:x for k,x in v.items() if k!='id'});write_json(source/'manifest.json',v)
                second=panels.plan(SOURCE_ROOT,source,out/'second',setups=2)
                self.assertNotIn(p['seeds'][0],second['seeds'])
                with self.assertRaisesRegex(ValueError,'roster changed'):panels.load(SOURCE_ROOT,out/'panel')
                tampered=copy.deepcopy(second);tampered['jobs'].pop();tampered=sealed({k:x for k,x in tampered.items() if k!='id'})
                write_json(out/'second/manifest.json',tampered)
                with self.assertRaisesRegex(ValueError,'assignments changed'):panels.load(SOURCE_ROOT,out/'second',sources=True)

    def test_incomplete_check_produces_report_and_never_a_qualified_roster(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=Path(tmp);v=candidate();v['encoder']=dict(path='authored-encoder',id='encoder')
            panel_path=out/'panel';panel_path.mkdir();write_json(panel_path/'manifest.json',dict(fixture=True))
            game=episode(4);job=dict(index=0,seed=3,neural_seats=[0],p4=dict(role='member',group=v['archetypes'][0]['id'],member=v['members'][0]['id']))
            p=dict(id='panel',jobs=[job]+[dict(index=i,p4=dict(role='control')) for i in range(1,100)],labels=None)
            side=dict(job=job,seconds=1.,telemetry=dict(completed=2,searches=1,seconds=.2))
            with patch.object(panels,'read_chunks',return_value=(p,v,iter([(game,side)]),{0},[])),patch.object(evaluation.descriptor,'load',return_value=(dict(id='encoder'),None)):
                result=evaluation.check(SOURCE_ROOT,panel_path,[],out/'checked')
            self.assertFalse(result['p4_qualified']);self.assertFalse(result['checks']['complete_panel'])
            self.assertEqual(len(result['missing_jobs']),99);self.assertFalse(result['release_qualified'])
            saved=roster.load(out/'checked');self.assertFalse(saved['p4_qualified'])
            self.assertTrue((out/'checked/review-template.json').exists())
            self.assertIn(game['setup_group'],saved['forbidden_groups'])

    def test_fresh_geometry_rejects_collapsed_styles_even_with_old_separated_centroids(self):
        value=candidate();jobs=panels.jobs_for(value,[1,2],2);rows=[]
        for j in jobs:
            rows.append(dict(arm=j['p4']))
        points=np.zeros((len(rows),2),dtype=np.float32)
        result=evaluation.geometry(value,rows,points,Gates())
        self.assertFalse(result['passed'])
        self.assertTrue(all(not m['distinct'] for m in result['members'].values()))
        by_id={m['id']:m for m in value['members']}
        for i,r in enumerate(rows):
            if r['arm']['role']=='member':points[i]=by_id[r['arm']['member']]['centroid']
        self.assertTrue(evaluation.geometry(value,rows,points,Gates())['passed'])

    def test_chunks_reject_duplicate_jobs_and_discovery_setup_reuse(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=Path(tmp);v=candidate();job=panels.jobs_for(v,[3],1)[0]
            p=dict(id='panel',jobs=[job]);panel_path=out/'panel';panel_path.mkdir();write_json(panel_path/'manifest.json',p)
            spec=dict(ready_id='ready',calibration_probe=False,pool_id=v['pool_id'],search=v['search'],decision_mode=v['decision_mode'],jobs=[job])
            manifest=dict(id='collection',spec=spec);directories=[]
            for i in range(2):
                directory=out/str(i);(directory/'games').mkdir(parents=True);write_json(directory/'games/manifest.json',manifest)
                receipt=sealed(dict(format=panels.CHUNK,contract=identity(),panel_id=p['id'],panel_sha256=sha(panel_path/'manifest.json'),
                                    collection_id=manifest['id'],collection_sha256=sha(directory/'games/manifest.json'),jobs=[0]))
                write_json(directory/'manifest.json',receipt);directories.append(directory)
            pair=(dict(setup_group='new-validation',split='validation'),dict(job=job))
            with patch.object(panels,'load',return_value=(p,v)),patch.object(panels,'read',side_effect=lambda *a,**kw:(manifest,iter([pair]))):
                _,_,pairs,seen,_=panels.read_chunks(SOURCE_ROOT,panel_path,directories)
                with self.assertRaisesRegex(ValueError,'Duplicate'):list(pairs)
                self.assertEqual(seen,{0})
                v['forbidden_groups']=['new-validation']
                with self.assertRaisesRegex(ValueError,'overlaps'):list(panels.read_chunks(SOURCE_ROOT,panel_path,directories[:1])[2])


if __name__=='__main__':unittest.main()
