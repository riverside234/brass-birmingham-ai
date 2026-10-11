import copy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from fixtures import rotate
from personality_fixtures import episode,parent
from brass_ai.personality.repertoire.traces import trajectory,arrays,FIELDS,MAX_TURNS
from brass_ai.personality.repertoire.descriptor import fit_arrays,transform,geometry
from brass_ai.personality.repertoire.discovery import proposals,jobs_for
from brass_ai.personality.repertoire.archive import Gates,evidence,select_elites,shortlist
from brass_ai.personality.search.contracts import search_config
from brass_ai.learning.reward import specification
from brass_ai.learning.contracts import SOURCE_ROOT,digest,read_json
from brass_ai.bank import sha
from brass_ai.step5.league import write_json
from brass_ai.personality.repertoire import descriptor,archive,discovery
from brass_ai.personality.repertoire.contracts import identity


def records():
    result=[]
    for candidate in range(3):
        for setup in range(2):
            for seat in range(4):
                game=episode(4);x,m=trajectory(game,seat);x[:,FIELDS.index('action_loan')]*=candidate+1
                result.append(dict(values=x,mask=m,metadata=dict(candidate_id=str(candidate),setup_group=str(setup),seat=seat,
                    split='train',trace_id=f'{candidate}/{setup}/{seat}',quality=dict(win_share=.25,vp=100.,placement=2.5))))
    return result


class RepertoireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_public_features_ignore_identity_outcomes_hands_timing_and_rotation(self):
        game=episode(4);before=trajectory(game,1);changed=copy.deepcopy(game)
        changed.update(game_id='other',opponents=['secret-profile']*4,result={},setup_group='other')
        for row in changed['records']:
            row['seconds']=999.;row['view']['own_hand']=['private-secret'];row['behavior']['policy_id']='changed'
            row['view']['players'][row['seat']]['vp']=999.
        for a,b in zip(before,trajectory(changed,1)):np.testing.assert_array_equal(a,b)
        changed=copy.deepcopy(game)
        for row in changed['records']:row['view']=rotate(row['view'],2);row['seat']=(row['seat']+2)%4
        for a,b in zip(before,trajectory(changed,3)):np.testing.assert_array_equal(a,b)

    def test_temporal_order_mask_and_split_restrictions(self):
        data=records();bad=copy.deepcopy(data);bad[0]['metadata']['split']='validation'
        with self.assertRaisesRegex(ValueError,'excludes'):fit_arrays(bad,epochs=1)
        bad=copy.deepcopy(data);bad[0]['mask'][0]=False
        with self.assertRaisesRegex(ValueError,'prefixes'):arrays(bad)
        payload,report=fit_arrays(data,epochs=2);a=transform(payload,data)
        self.assertEqual(a.shape,(24,2));self.assertTrue(np.isfinite(a).all());self.assertGreater(report['parameters'],0)
        changed=copy.deepcopy(data)
        for row in changed:
            n=int(row['mask'].sum());row['values'][:n]=row['values'][:n][::-1]
            row['metadata']['split']='validation'
        self.assertFalse(np.array_equal(a,transform(payload,changed)))
        self.assertEqual(transform(payload,data,method='pca').shape,(24,2));self.assertEqual(transform(payload,data,method='authored').shape,(24,6))
        self.assertIn('seat_explained_fraction',geometry(a,data))

    def test_public_merchant_access_supports_multi_region_links(self):
        game=episode(2)
        for row in game['records']:
            view=row['view'];tile=copy.deepcopy(next(m['tiles'][0] for m in view['players'][0]['mat'] if m['type']=='CottonMill'))
            view['slots']=[dict(id='Birmingham:0',types=['CottonMill'],tile=tile)]
            view['links']=[dict(id='multi-region',owner=3,regions=['Birmingham','Coventry','Oxford'])]
            view['merchants']=[dict(id='Oxford:0',types=['CottonMill'],beer=1)]
        values,mask=trajectory(game,0)
        self.assertTrue((values[mask,FIELDS.index('merchant_access_CottonMill')]==1).all())

    def test_public_effects_follow_own_choices_not_opponent_turns(self):
        game=episode(2);rows=copy.deepcopy([game['records'][i] for i in (0,1,2,3,8)])
        for i,row in enumerate(rows):
            row['step']=i
            row['view']['players'][0]['cash']=[20,18,15,65,65][i]
            for industry in ('Brewery','CottonMill'):
                tile=copy.deepcopy(next(m['tiles'][0] for m in row['view']['players'][0]['mat'] if m['type']==industry))
                tile.update(seat=0,flipped=(i>=1 if industry=='Brewery' else i>=3))
                row['view']['slots'].append(dict(id=industry+':fixture',types=[industry],tile=tile))
        # Our brewery payment flips our brewery and is followed by merchant
        # beer. An opponent later flips our cotton. Only the adjacent own
        # effect is attributed to this turn; the next turn sees both.
        merchant=rows[0]['view']['merchants'][0]['id']
        rows[0]['choices']=[dict(kind='beer',target='Brewery:fixture')];rows[0]['chosen']=0
        rows[1]['choices']=[dict(kind='beer',target='merchant:'+merchant)];rows[1]['chosen']=0
        game['records']=rows
        x,mask=trajectory(game,0)
        self.assertAlmostEqual(float(x[0,FIELDS.index('cash_change')]),-.05)
        self.assertEqual(x[0,FIELDS.index('flips_Brewery')],1)
        self.assertEqual(x[0,FIELDS.index('flips_CottonMill')],0)
        self.assertEqual(x[0,FIELDS.index('merchant_beer_used')],1)
        self.assertEqual(x[0,FIELDS.index('effects_observed')],1)

    def test_era_clear_and_missing_terminal_effects_are_censored(self):
        game=episode(2);rows=copy.deepcopy([game['records'][i] for i in (0,1,8)])
        rows[2]['view']['context']['era']='RailEra'
        rows[2]['view']['players'][0]['cash']=10000
        tile=copy.deepcopy(next(m['tiles'][0] for m in rows[0]['view']['players'][0]['mat'] if m['type']=='CottonMill'))
        for i in (0,1):
            rows[i]['view']['slots'].append(dict(id='Canal:fixture',types=['CottonMill'],tile=dict(tile,seat=0,flipped=bool(i))))
        game['records']=rows;x,_=trajectory(game,0)
        self.assertEqual(x[0,FIELDS.index('flips_CottonMill')],1)
        self.assertEqual(x[0,FIELDS.index('era_boundary')],1)
        self.assertEqual(x[0,FIELDS.index('effects_observed')],.5)
        self.assertAlmostEqual(float(x[0,FIELDS.index('cash_change')]),.30)
        self.assertEqual(x[1,FIELDS.index('effects_observed')],0)
        self.assertEqual(x[1,FIELDS.index('cash_change')],0)

    def test_seeded_proposals_share_weights_and_matched_rotations(self):
        cfg=search_config(specification());a=proposals('model','parent','pool',cfg,count=64)
        self.assertEqual(a,proposals('model','parent','pool',cfg,count=64));self.assertEqual(len({x['candidate_id'] for x in a}),64)
        self.assertEqual({x['model_id'] for x in a},{'model'});self.assertEqual(a[0]['profile']['alpha'],0)
        jobs=jobs_for(dict(candidates=a[:2]),[31,32],'train',4)
        self.assertEqual(len(jobs),16)
        for i in range(4):
            self.assertEqual(jobs[i]['parent_indices'],jobs[4+i]['parent_indices']);self.assertEqual(jobs[i]['sampling_seed'],jobs[4+i]['sampling_seed'])
        with self.assertRaisesRegex(ValueError,'Final-test'):jobs_for(dict(candidates=a[:2]),[31],'test',4)

    def test_no_single_game_promotion_or_team_quality(self):
        data=records();control=[r for r in data if r['metadata']['candidate_id']=='0']
        candidate=[r for r in data if r['metadata']['candidate_id']=='1'];g=Gates(min_setups=2)
        self.assertFalse(evidence(candidate[:1],control,g)['qualified'])
        self.assertTrue(evidence(candidate,control,g)['qualified'])
        for r in candidate:r['metadata']['quality']['win_share']=0.;r['metadata']['quality']['vp']=1000.
        result=evidence(candidate,control,g);self.assertFalse(result['qualified']);self.assertEqual(result['mean_win'],0.)

    def test_candidate_slices_preserve_matchups_and_bound_large_proposals(self):
        cfg=search_config(specification());candidates=proposals('model','parent','pool',cfg,count=2050)
        small=dict(candidates=candidates[:8]);full=jobs_for(small,[31,32],'train',4)
        sliced=[]
        for start,count in ((0,3),(3,5)):
            sliced.extend(jobs_for(small,[31,32],'train',4,candidate_start=start,candidate_count=count))
        def key(job):return (job['seed'],job['controllers'][0],job['neural_seats'][0])
        def content(job):return {k:v for k,v in job.items() if k!='index'}
        self.assertEqual({key(j):content(j) for j in full},{key(j):content(j) for j in sliced})
        with self.assertRaisesRegex(ValueError,'4096'):
            jobs_for(dict(candidates=candidates),[31],'train',4)
        seen=set();all_ids=set();control=candidates[0]['candidate_id']
        for start,count in ((0,1024),(1024,1024),(2048,2)):
            jobs=jobs_for(dict(candidates=candidates),[31],'train',4,candidate_start=start,candidate_count=count)
            self.assertEqual(len(jobs),count*4);self.assertLessEqual(len(jobs),4096)
            ids={j['controllers'][0] for j in jobs};self.assertEqual(control in ids,start==0)
            for j in jobs:self.assertNotIn(key(j),seen);seen.add(key(j))
            all_ids.update(ids)
        self.assertEqual(all_ids,{c['candidate_id'] for c in candidates})
        for start,count in ((-1,2),(8,None),(0,0),(7,2)):
            with self.assertRaisesRegex(ValueError,'candidate slice'):
                jobs_for(small,[31],'train',4,candidate_start=start,candidate_count=count)
        with self.assertRaisesRegex(ValueError,'candidate slice'):
            jobs_for(dict(candidates=[]),[31],'train',4)

    def test_sliced_collection_keeps_full_proposal_receipt(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=Path(tmp);ready=out/'ready';ready.mkdir();write_json(ready/'ready.json',dict(fixture=True))
            state=dict(id='ready',reward=specification(),model=dict(model_id='fixture'))
            def collect(root,bundle,output,jobs,config,**kwargs):
                Path(output).mkdir();write_json(Path(output)/'pins.json',dict(paths=[]))
                self.assertEqual({j['controllers'][0] for j in jobs},{c['candidate_id'] for c in p['candidates'][2:4]})
                return dict(id='slice',records=[dict(job=j['index']) for j in jobs],seconds=0.)
            with patch.object(discovery,'load_ready',return_value=(None,[parent()],state,{},'pool')):
                p=discovery.create(SOURCE_ROOT,ready,out/'proposal',count=8)
                original_hash=sha(out/'proposal/manifest.json')
                with patch.object(discovery,'collect',side_effect=collect):
                    result=discovery.execute(SOURCE_ROOT,out/'proposal',out/'slice',setups=2,candidate_start=2,candidate_count=2)
                self.assertEqual(result['games'],16)
                self.assertEqual(result['receipt']['proposal_id'],p['id'])
                self.assertEqual(result['receipt']['proposal_sha256'],original_hash)
                self.assertEqual(result['receipt']['candidate_range'],dict(start=2,count=2))
                self.assertEqual(sha(out/'proposal/manifest.json'),original_hash)

    def test_archive_filters_noise_and_shortlist_only_tested_members(self):
        entries=[dict(candidate_id=str(i),centroid=[float(i),float(i%3)],qualified=True,quality=dict(mean_win=.4,control_lower=.1)) for i in range(20)]
        entries.append(dict(candidate_id='lucky-one-game',centroid=[50.,50.],qualified=False,quality=dict(mean_win=1.,control_lower=-1.)))
        elites=select_elites(entries,Gates());self.assertNotIn('lucky-one-game',elites)
        for method in ('fps','kmeans'):
            chosen=shortlist(entries,elites,method=method)
            self.assertEqual(len(set(chosen)),16);self.assertTrue(set(chosen)<=set(elites))
            self.assertEqual(shortlist(entries,elites[:3],method=method),shortlist(entries,elites[:3],method=method))
            self.assertEqual(len(shortlist(entries,elites[:3],method=method)),3)
        self.assertEqual(shortlist(entries,[]),[])

    def test_immutable_archive_rebuild_reencodes_every_member_and_pins_evidence(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temporary:
            root=SOURCE_ROOT;out=Path(temporary);proposal=out/'proposal';proposal.mkdir()
            candidates=proposals('model','parent','pool',search_config(specification()),count=3)
            p=dict(format=discovery.VERSION,contract=identity(),candidates=candidates,control_id=candidates[0]['candidate_id'],
                ready='artifacts/fixture-ready',pool_id='pool');p['id']=digest(p);write_json(proposal/'manifest.json',p)
            data=records();validation=copy.deepcopy(data)
            for group,partition in ((data,'train'),(validation,'validation')):
                for r in group:
                    m=r['metadata'];m.update(candidate_id=candidates[int(m['candidate_id'])]['candidate_id'],pool_id='pool',model_id='model',split=partition)
                    m['setup_group']=partition+m['setup_group'];m['trace_id']=partition+m['trace_id']
            sources={}
            for name,group in (('discovery',data),('qualification',validation)):
                directory=out/name;directory.mkdir();m=dict(id=name,spec=dict(pool_id='pool'));write_json(directory/'manifest.json',m)
                receipt=dict(proposal_id=p['id'],proposal_sha256=sha(proposal/'manifest.json'),collection_id=name);receipt['id']=digest(receipt)
                write_json(directory/'discovery.json',receipt);sources[str(directory)]=(m,group)
            def read(directory):return sources[str(directory)]
            with patch.object(descriptor,'from_collection',side_effect=read),patch.object(archive,'from_collection',side_effect=read):
                encoder=descriptor.fit(root,[out/'discovery'],out/'encoder',epochs=2)
                a=archive.build(root,proposal,out/'encoder',[out/'discovery'],[out/'qualification'],out/'archive',gates=Gates(min_setups=2))
                self.assertEqual(a['reencoded_candidates'],3);self.assertFalse(a['shortlist']['release_qualified'])
                previous_hash=sha(out/'archive/manifest.json')
                descriptor.fit(root,[out/'discovery'],out/'encoder-2',epochs=2,seed=199)
                b=archive.build(root,proposal,out/'encoder-2',[],[],out/'archive-2',previous=out/'archive',gates=Gates(min_setups=2))
                self.assertEqual(b['reencoded_candidates'],3);self.assertEqual(b['previous_id'],a['id']);self.assertNotEqual(a['encoder']['id'],b['encoder']['id'])
                self.assertEqual(sha(out/'archive/manifest.json'),previous_hash)
                pins=read_json(out/'archive-2/pins.json')['paths'];self.assertIn('artifacts/fixture-ready/ready.json',pins)
                self.assertFalse(list((out/'archive-2').glob('*.pt')))
                # Qualification cannot be relabeled as train or share encoder
                # groups, even when the vector shapes are identical.
                for r in validation:r['metadata']['split']='train'
                with self.assertRaisesRegex(ValueError,'split leakage'):
                    archive.build(root,proposal,out/'encoder',[],[],out/'bad',previous=out/'archive',gates=Gates(min_setups=2))
                for r in validation:r['metadata']['split']='validation'
                # A valid replacement collection still cannot silently replace
                # the precise evidence retained in an immutable older archive.
                write_json(out/'qualification/manifest.json',dict(replaced=True))
                with self.assertRaisesRegex(ValueError,'Retained archive evidence changed'):
                    archive.build(root,proposal,out/'encoder',[],[],out/'changed',previous=out/'archive',gates=Gates(min_setups=2))

    def test_elite_mutations_retain_ancestry_and_search_confirmation_resets_evidence(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            out=Path(tmp);ready=out/'ready';ready.mkdir();write_json(ready/'ready.json',dict(fixture=True))
            state=dict(id='ready',reward=specification(),model=dict(model_id='fixture'))
            with patch.object(discovery,'load_ready',return_value=(None,[parent()],state,{},'pool')):
                p=discovery.create(SOURCE_ROOT,ready,out/'proposal',count=3)
                entries=[dict(candidate_id=c['candidate_id'],controller=c) for c in p['candidates']]
                a=dict(format=archive.VERSION,contract=identity(),proposal_path=(out/'proposal').relative_to(SOURCE_ROOT).as_posix(),
                    entries=entries,elites=[e['candidate_id'] for e in entries[:2]],shortlist=dict(members=[e['candidate_id'] for e in entries[:2]]))
                a['id']=digest(a);(out/'archive').mkdir();write_json(out/'archive/manifest.json',a)
                extended=discovery.create(SOURCE_ROOT,ready,out/'extended',count=6,from_archive=out/'archive')
                self.assertEqual(extended['candidates'][:3],p['candidates']);self.assertEqual(len(extended['ancestry']),3)
                self.assertTrue(set(extended['ancestry'].values())<=set(a['elites']));self.assertEqual(extended['ancestors'][0]['id'],p['id'])
                confirmed=discovery.create(SOURCE_ROOT,ready,out/'confirmed',decision_mode='searched',from_archive=out/'archive')
                self.assertEqual(confirmed['ancestors'],[]);self.assertTrue(confirmed['ancestry'])
                self.assertTrue(all(c['decision_mode']=='searched' for c in confirmed['candidates']))
                self.assertFalse({c['candidate_id'] for c in confirmed['candidates']} & {c['candidate_id'] for c in p['candidates']})


if __name__=='__main__':unittest.main()
