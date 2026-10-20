"""Validated complete-game shards and fingerprinted derived feature caches.

The authoritative raw replay wire format stays unchanged; encodings are versioned.
Imported records are validated on ingress. Independently recorded validation
receipts allow exact byte reuse; an unsigned sidecar never grants trust.
An in-memory game binds each reusable row to its complete input digest.
"""
import gzip
import json
from pathlib import Path

import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import (SOURCE_ROOT, assert_compatible, compatibility, decode_json, digest,
                                        finite_vector, keys, require)
from brass_ai.learning.features import Encoded,check_encoded
from brass_ai.learning.local_entities import check_action_selectors
from brass_ai.learning.replay import split_for, terminal_targets, validate_teacher
from .encoding import PreparedEncoder
from .inputs import BOUNDARY


def row_key(row):
    return digest(dict(input=digest(dict(view=row['view'],choices=row['choices'])),style=row['style'],intent=row['intent']))


def label_key(row):
    return digest({k:row[k] for k in ('step','seat','chosen','behavior','teacher','seconds')})


class CheckedGame(dict):
    def bind(self, arrays):
        self.features={};self.seals={}
        self.header_seal=digest({k:v for k,v in self.items() if k!='records'})
        for r,e in zip(self['records'],arrays):
            source=digest(dict(view=r['view'],choices=r['choices']))
            self.features[digest(dict(input=source,style=r['style'],intent=r['intent']))]=e
            self.seals[(r['step'],r['seat'])]=dict(input=source,label=label_key(r),style=r['style'][:],intent=r['intent'][:])
        return self

    def assert_metadata(self):
        require([(r['step'],r['seat']) for r in self['records']]==list(self.seals)
                and digest({k:v for k,v in self.items() if k!='records'})==self.header_seal,
                'Validated game metadata changed')

    def encoded(self, row, encoder,*,allow_context=False):
        source=digest(dict(view=row['view'],choices=row['choices']));seal=self.seals.get((row['step'],row['seat']))
        require(seal and source==seal['input'] and label_key(row)==seal['label'],'Validated replay record changed')
        require(allow_context or (row['style']==seal['style'] and row['intent']==seal['intent']),'Validated policy context changed')
        token=digest(dict(input=source,style=row['style'],intent=row['intent']))
        if token in self.features: return self.features[token]
        # The pinned R2 layout appends style[4] + intent[8]; neither changes
        # action features. Derive a separate neutral BC array without a second
        # walk of public history. Verify this layout/context against the source.
        require(allow_context and encoder.schema['intent_dim']==8,'Unsupported context derivation')
        finite_vector(row['style'],4,-1,1);finite_vector(row['intent'],8,-1,1)
        base=self.features[digest(dict(input=source,style=seal['style'],intent=seal['intent']))]
        require(np.array_equal(base.state[-12:],np.asarray(seal['style']+seal['intent'],dtype=np.float32)),
                'Pinned context feature layout differs')
        state=base.state.copy();state[-12:]=row['style']+row['intent'];state.setflags(write=False)
        encoded=Encoded(state,base.actions,base.stable_to_relative,base.choice_ids);self.features[token]=encoded
        return encoded


def check_game(game, arrays=None,*,validated_inputs=False):
    require(not validated_inputs or arrays is not None,'Checked ingress requires attested arrays')
    keys(game, ('game_id', 'setup_group', 'players', 'split', 'opponents', 'result', 'records'),
         ('branch','branches','branch_statistics'))
    require(not ('branch' in game and ('branches' in game or 'branch_statistics' in game)), 'Nested branch forbidden')
    require(type(game['players']) is int and game['players'] == 4, 'Replay must have four seats')
    require(all(isinstance(game[k], str) and game[k] for k in ('game_id','setup_group')), 'Missing game identity')
    require(game['split'] == split_for(game['setup_group']), 'Setup split mismatch')
    require(type(game['opponents']) is list and len(game['opponents']) == 4 and
            all(isinstance(x,str) and x for x in game['opponents']), 'Four frozen opponents required')
    terminal_targets(game['result'])
    require(type(game['records']) is list and game['records'], 'Empty game')
    encoder = PreparedEncoder(); encoded = []; previous = -1
    for step, row in enumerate(game['records']):
        from brass_ai.learning.board_topology import admit_topology
        admit_topology(row['view'])
        keys(row, ('step','seat','view','choices','chosen','behavior','style','intent','teacher','seconds'))
        require(type(row['step']) is int and row['step'] == step, 'Non-contiguous decision sequence')
        require(type(row['seat']) is int and 0 <= row['seat'] < 4 and
                row['seat'] == row['view']['seat'], 'Seat attribution mismatch')
        if not validated_inputs: BOUNDARY.check(row['view'], row['choices'])
        if arrays is None:
            e = encoder.encode(row['view'],row['choices'],row['style'],row['intent'],validated=True)
        else:
            e = arrays[step]
            check_encoded(e)
            finite_vector(row['style'],4,-1,1);finite_vector(row['intent'],8,-1,1)
            require(e.state.shape == (encoder.state_dim,) and e.actions.shape == (len(row['choices']),encoder.action_dim),
                    'Cached feature dimensions differ')
            require(e.choice_ids == tuple(c['id'] for c in row['choices']) and
                    np.array_equal(e.stable_to_relative,[(s-row['seat'])%4 for s in range(4)]), 'Cached choices/seats differ')
        check_action_selectors(e.actions)
        require(len(e.choice_ids)==len(set(e.choice_ids)),'Duplicate cached choice IDs')
        require(type(row['chosen']) is int and 0 <= row['chosen'] < len(row['choices']), 'Illegal chosen index')
        revision = row['view']['context']['revision']
        require(revision > previous, 'Repeated or reversed revision'); previous = revision
        b = row['behavior']
        if 'exploration_version' in b:
            from .exploration import check_behavior
            check_behavior(b,row['view']['context']['phase'],branch_move='branch' in game and step==0)
        else:keys(b, ('kind','policy_id','probability','temperature'))
        require(b['kind'] in ('model','random','external') and b['policy_id'] == game['opponents'][row['seat']],
                'Behavior attribution differs')
        finite_vector([b['probability']],1,0,1); require(b['probability'] > 0,'Zero behavior probability')
        finite_vector([b['temperature']],1,0);finite_vector([row['seconds']],1,0)
        validate_teacher(row['teacher'],len(row['choices']))
        if row['teacher'] is not None and row['teacher']['source'] in ('ismcts-root-visits/1.0.0','ismcts-root-visits/1.1.0','ismcts-root-visits/1.2.0'):
            require(row['teacher']['legal_ids']==[c['id'] for c in row['choices']] and
                    row['teacher']['observer']==row['seat'], 'Teacher action order/observer mismatch')
        encoded.append(e)
    if 'branch' not in game:
        require({r['seat'] for r in game['records']} == set(range(4)), 'Missing seat trajectories')
    if 'branches' in game:
        from .branching import validate_child
        require(type(game['branches']) is list and len(game['branches'])<=1,'Too many branch games')
        game=dict(game,branches=[check_game(child) for child in game['branches']])
        for child in game['branches']:validate_child(game,child)
    return CheckedGame(game).bind(encoded)


def _cache_paths(path):
    path=Path(path)
    return path.with_name(path.name+'.features-v4.npz'), path.with_name(path.name+'.features-v4.json')


def cache_version():
    paths=['recovery/'+n for n in ('encoding.py','inputs.py','replay.py')]+['runtime_contracts.py']
    return digest(dict(source={name:sha(SOURCE_ROOT/'src/brass_ai'/name) for name in paths},compatibility=compatibility()))


def cache_receipt(path):
    data,manifest=_cache_paths(path)
    return dict(data_sha256=sha(data),manifest_sha256=sha(manifest),replay_sha256=sha(path),validator=cache_version()) if data.exists() and manifest.exists() else None


def write_cache(path, games):
    rows=[r for g in games for r in g['records']]
    enc=PreparedEncoder(); arrays=[g.encoded(r,enc) for g in games for r in g['records']]
    data, manifest=_cache_paths(path); temporary=data.with_name(data.name+'.tmp')
    with temporary.open('wb') as out:
        np.savez_compressed(out, states=np.stack([e.state for e in arrays]),
            actions=np.concatenate([e.actions for e in arrays]),
            offsets=np.cumsum([0]+[len(e.choice_ids) for e in arrays]),
            seats=np.stack([e.stable_to_relative for e in arrays]))
    temporary.replace(data)
    meta=dict(format='derived-features/4.0.0',replay_sha256=sha(path),data_sha256=sha(data),
        compatibility=compatibility(),encoder=cache_version(),rows=[row_key(r) for r in rows])
    temp=manifest.with_name(manifest.name+'.tmp');temp.write_text(json.dumps(meta),encoding='utf-8');temp.replace(manifest)


def load_cache(path, games, receipt=None):
    data,manifest=_cache_paths(path)
    # A self-signed foreign cache is not trusted. Receipts come from committed
    # actor results or an audited dataset manifest, independent of this sidecar.
    if receipt is None or receipt.get('validator')!=cache_version() or not manifest.exists() or not data.exists(): return None
    require(receipt==cache_receipt(path),'Derived feature receipt changed')
    meta=decode_json(manifest.read_text(encoding='utf-8'))
    require(meta.get('format')=='derived-features/4.0.0', 'Unknown derived feature cache; explicitly re-encode raw data')
    # A changed source invalidates derived data, never the authoritative replay.
    if meta['replay_sha256']!=sha(path) or meta['compatibility']!=compatibility() or meta.get('encoder')!=cache_version(): return None
    require(meta['data_sha256']==sha(data),'Derived feature cache changed')
    rows=[r for g in games for r in g['records']]
    require(len(meta['rows'])==len(rows),'Cached row count differs')
    with np.load(data,allow_pickle=False) as source:
        states,actions,offsets,seats=(source[k] for k in ('states','actions','offsets','seats'))
        require(len(states)==len(rows) and len(offsets)==len(rows)+1 and offsets[0]==0 and offsets[-1]==len(actions),
                'Malformed derived cache')
        require(offsets.dtype.kind in 'iu' and np.all(np.diff(offsets)>0) and
                np.array_equal(np.diff(offsets),[len(r['choices']) for r in rows]),'Malformed cached offsets')
        require(all(np.isfinite(a).all() for a in (states,actions,seats)), 'Non-finite derived cache')
        out=[]
        for i,row in enumerate(rows):
            e=Encoded(states[i],actions[offsets[i]:offsets[i+1]],seats[i],tuple(c['id'] for c in row['choices']))
            check_encoded(e)
            check_action_selectors(e.actions)
            for a in (e.state,e.actions,e.stable_to_relative): a.setflags(write=False)
            out.append(e)
    return out


class SerializedGame:
    """Writer-owned validated game and its single immutable JSON encoding."""
    def __init__(self,game):
        encoder=PreparedEncoder()
        if isinstance(game,CheckedGame):game.assert_metadata()
        self.game=check_game(game,[game.encoded(r,encoder) for r in game['records']] if isinstance(game,CheckedGame) else None)
        self.line=json.dumps(self.game,separators=(',',':'),allow_nan=False)+'\n'
        self.size=len(self.line.encode('utf-8'))


def write_shard(path,games,provenance,*,cache=True):
    path=Path(path);require(not path.exists() and games,'Existing or empty shard')
    serialized=[g if isinstance(g,SerializedGame) else SerializedGame(g) for g in games]
    games=[g.game for g in serialized]
    require(len({g['game_id'] for g in games})==len(games),'Duplicate game IDs')
    for game in games:game.assert_metadata()
    checked=games
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    try:
        with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as stream:
            stream.write(json.dumps(dict(format='four-seat-replay/1.0.0',compatibility=compatibility(),provenance=provenance))+'\n')
            for game in serialized:stream.write(game.line)
        tmp.replace(path)
        if cache: write_cache(path,checked)
    finally: tmp.unlink(missing_ok=True)
    return checked


def read_shards(paths,partition=None,receipts=None,*,heuristic_only=False):
    require(partition in (None,'train','validation','test'),'Unknown partition')
    output=[];identities=set()
    for path in dict.fromkeys(str(p) for p in paths):
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            header=decode_json(next(stream));keys(header,('format','compatibility','provenance'))
            require(header['format']=='four-seat-replay/1.0.0','Unsupported replay format')
            if heuristic_only:
                from brass_ai.learning.contracts import assert_heuristic_data_compatible
                assert_heuristic_data_compatible(header['compatibility'])
            else:assert_compatible(header['compatibility'])
            require(type(header['provenance']) is dict,'Invalid provenance')
            require(partition!='train' or header['provenance'].get('purpose')!='evaluation','Evaluation replay in training')
            raw=[decode_json(line) for line in stream]
        if heuristic_only:
            require(all(r['behavior']['kind']=='external' and r['teacher'] is None for g in raw for r in g['records']),
                    'Audited heuristic ingress prohibits neural behavior and search targets')
        arrays=load_cache(path,raw,(receipts or {}).get(str(path)));start=0
        jobs=header['provenance'].get('jobs')
        if jobs is not None:require(len(jobs)==len(raw),'Replay provenance job count differs')
        for ordinal,g in enumerate(raw):
            count=len(g['records']);game=check_game(g,None if arrays is None else arrays[start:start+count],validated_inputs=arrays is not None);start+=count
            require(game['game_id'] not in identities,'Duplicate game across shards');identities.add(game['game_id'])
            # Descriptors are table metadata, separate from the immutable board
            # observation/Features-4 contract. The enclosing replay hash seals it.
            proof=jobs[ordinal]['provenance'] if jobs is not None else header['provenance']
            require('branch' not in game,'Standalone branch replay lacks parent evidence')
            if game.get('branches'):
                require(proof.get('purpose')=='training' and proof.get('kind')=='selfplay' and
                    proof.get('exploration',{}).get('branching') is not None,
                    'Branch replay outside ordinary self-play')
            if 'critic_table' in proof:
                game.critic_table=proof['critic_table'];game.table_kind=proof['kind']
            if partition is None or game['split']==partition: output.append(game)
    require(output,'No complete games in requested partition')
    return output
