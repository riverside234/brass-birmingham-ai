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
