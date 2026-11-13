"""Validated complete-game shards and fingerprinted derived feature caches.

Legacy raw replays and versioned compact IL replays share the same admission.
Imported records are validated on ingress. Independently recorded validation
receipts allow exact byte reuse; an unsigned sidecar never grants trust.
An in-memory game binds each reusable row to its complete input digest.
"""
import gzip
import hashlib
import io
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
from . import il_replay_storage


def row_key(row):
    return digest(dict(input=digest(dict(view=row['view'],choices=row['choices'])),style=row['style'],intent=row['intent']))


def label_key(row):
    return digest({k:row[k] for k in ('step','seat','chosen','behavior','teacher','seconds')})


class CheckedGame(dict):
    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k!='_immutable_binding'}

    def __setstate__(self,value):self.__dict__.update(value)

    def bind(self, arrays):
        from .replay_binding import count
        self.__dict__.pop('_immutable_binding',None)
        self.features={};self.seals={}
        self.header_seal=digest({k:v for k,v in self.items() if k!='records'})
        count('metadata_hash_executed')
        for r,e in zip(self['records'],arrays):
            source=digest(dict(view=r['view'],choices=r['choices']))
            count('input_hash_executed');count('label_hash_executed')
            self.features[digest(dict(input=source,style=r['style'],intent=r['intent']))]=e
            self.seals[(r['step'],r['seat'])]=dict(input=source,label=label_key(r),style=r['style'][:],intent=r['intent'][:])
        return self

    def assert_metadata(self):
        from .replay_binding import GameBinding,count
        binding=getattr(self,'_immutable_binding',None)
        if type(binding) is GameBinding:return binding.metadata(self)
        count('metadata_hash_executed')
        require([(r['step'],r['seat']) for r in self['records']]==list(self.seals)
                and digest({k:v for k,v in self.items() if k!='records'})==self.header_seal,
                'Validated game metadata changed')

    def encoded(self, row, encoder,*,allow_context=False):
        from .replay_binding import GameBinding,count,seal_encoded
        binding=getattr(self,'_immutable_binding',None)
        proof=binding.row(self,row,allow_context=allow_context) if type(binding) is GameBinding else None
        seal=self.seals.get((row['step'],row['seat']))
        if proof is None:
            source=digest(dict(view=row['view'],choices=row['choices']));count('input_hash_executed');count('label_hash_executed')
            require(seal and source==seal['input'] and label_key(row)==seal['label'],'Validated replay record changed')
            require(allow_context or (row['style']==seal['style'] and row['intent']==seal['intent']),'Validated policy context changed')
        else:source=proof.source
        token=(proof.key if proof is not None and row['style']==proof.row['style'] and row['intent']==proof.row['intent'] else
               digest(dict(input=source,style=row['style'],intent=row['intent'])))
        if token in self.features:
            if proof is not None:
                from .replay_binding import checked_encoded
                require(self.features[token] is binding.features.get(token) and checked_encoded(self.features[token]),
                        'Validated features changed')
            return self.features[token]
        # The pinned R2 layout appends style[4] + intent[8]; neither changes
        # action features. Derive a separate neutral BC array without a second
        # walk of public history. Verify this layout/context against the source.
        require(allow_context and encoder.schema['intent_dim']==8,'Unsupported context derivation')
        finite_vector(row['style'],4,-1,1);finite_vector(row['intent'],8,-1,1)
        base=self.features[digest(dict(input=source,style=seal['style'],intent=seal['intent']))]
        require(np.array_equal(base.state[-12:],np.asarray(list(seal['style'])+list(seal['intent']),dtype=np.float32)),
                'Pinned context feature layout differs')
        state=base.state.copy();state[-12:]=row['style']+row['intent'];state.setflags(write=False)
        encoded=Encoded(state,base.actions,base.stable_to_relative,base.choice_ids)
        if proof is not None:encoded=seal_encoded(encoded,_checked=True)
        self.features[token]=encoded
        if proof is not None:binding.features[token]=encoded
        return encoded


def check_game(game, arrays=None,*,validated_inputs=False,_feature_offset=0):
    require(not validated_inputs or arrays is not None,'Checked ingress requires attested arrays')
    keys(game, ('game_id', 'setup_group', 'players', 'split', 'opponents', 'result', 'records'),
         ('branch','branches','branch_statistics','local_scoring_targets'))
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
            e = arrays[_feature_offset+step]
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
    if 'local_scoring_targets' in game:
        from .local_scoring_targets import validate
        validate(game['local_scoring_targets'],game)
    if 'branches' in game:
        from .branching import validate_child
        require(type(game['branches']) is list and len(game['branches'])<=1,'Too many branch games')
        game=dict(game,branches=[check_game(child) for child in game['branches']])
        for child in game['branches']:validate_child(game,child)
    if validated_inputs:
        from brass_ai.rust_backend.lab_features import VerifiedArrays,bind
        if isinstance(arrays,VerifiedArrays):return bind(game,arrays,start=_feature_offset)
    return CheckedGame(game).bind(encoded)


def _cache_paths(path):
    path=Path(path)
    return path.with_name(path.name+'.features-v4.npz'), path.with_name(path.name+'.features-v4.json')


def cache_version():
    from .feature_receipts import review
    reviewed=review()
    if reviewed:return reviewed['original_encoder']
    paths=['recovery/'+n for n in ('encoding.py','inputs.py','replay.py','il_replay_storage.py')]+['runtime_contracts.py']
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


def load_cache(path, games, receipt=None,*,replay_sha256=None):
    from brass_ai.rust_backend.lab_features import read as read_features
    # Same derived-features/4 format for legacy/native. Read/hash each exact
    # archive once; no hash-open-hash-open loop or Python re-encoding on reuse.
    if receipt is None:return None
    return read_features(path,games,receipt,sha(path) if replay_sha256 is None else replay_sha256)


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
    require(all(m.get('local_scoring_targets',{}).get('producer',{}).get('contract') is None
                for g in games for m in [g]+g.get('branches',[])),
            'Native scoring requires the separate native replay/admission format')
    require(len({g['game_id'] for g in games})==len(games),'Duplicate game IDs')
    for game in games:game.assert_metadata()
    checked=games
    storage=il_replay_storage.storage_format(provenance)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    try:
        with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as stream:
            stream.write(json.dumps(dict(format=storage,compatibility=compatibility(),provenance=provenance))+'\n')
            for game in serialized:
                line=(json.dumps(il_replay_storage.pack(game.game),separators=(',',':'),allow_nan=False)+'\n'
                      if storage==il_replay_storage.FORMAT else game.line)
                stream.write(line)
        tmp.replace(path)
        if cache: write_cache(path,checked)
    finally: tmp.unlink(missing_ok=True)
    return checked


def read_shards(paths,partition=None,receipts=None,*,heuristic_only=False,native_admissions=None,_expected_hashes=None):
    require(partition in (None,'train','validation','test'),'Unknown partition')
    output=[];identities=set()
    for path in dict.fromkeys(str(p) for p in paths):
        verified=None
        if _expected_hashes is not None:
            from brass_ai.rust_backend.lab_features import read_bytes
            packed=read_bytes(path);verified=hashlib.sha256(packed).hexdigest()
            require(verified==_expected_hashes[path],'Self-play replay changed')
            stream=io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(packed)),encoding='utf-8')
        else:stream=gzip.open(path,'rt',encoding='utf-8')
        with stream:
            header=decode_json(next(stream))
            if header.get('format')=='four-seat-native-replay/1.0.0':
                require(not heuristic_only,'Native ordinary replay is not completed-IL heuristic data')
                from brass_ai.rust_backend.lab_admission import read as read_native
                game=read_native(path,(native_admissions or {}).get(str(path)))
                require(game['game_id'] not in identities,'Duplicate game across shards')
                identities.add(game['game_id'])
                if partition is None or game['split']==partition:output.append(game)
                continue
            keys(header,('format','compatibility','provenance'))
            require(header['format'] in (il_replay_storage.LEGACY_FORMAT,il_replay_storage.FORMAT),'Unsupported replay format')
            if header['format']==il_replay_storage.FORMAT:
                require(il_replay_storage.storage_format(header['provenance'])==il_replay_storage.FORMAT,
                        'Compact IL replay lacks pinned storage provenance')
            if heuristic_only:
                from brass_ai.learning.contracts import assert_heuristic_data_compatible
                assert_heuristic_data_compatible(header['compatibility'])
            else:assert_compatible(header['compatibility'])
            require(type(header['provenance']) is dict,'Invalid provenance')
            require(partition!='train' or header['provenance'].get('purpose')!='evaluation','Evaluation replay in training')
            raw=[decode_json(line) for line in stream]
            if header['format']==il_replay_storage.FORMAT:
                raw=[il_replay_storage.unpack(game) for game in raw]
        if heuristic_only:
            require(all(r['behavior']['kind']=='external' and r['teacher'] is None for g in raw for r in g['records']),
                    'Audited heuristic ingress prohibits neural behavior and search targets')
        require(all(m.get('local_scoring_targets',{}).get('producer',{}).get('contract') is None
                    for g in raw for m in [g]+g.get('branches',[])),
                'Native scoring cannot be relabeled as legacy replay')
        arrays=load_cache(path,raw,(receipts or {}).get(str(path)),replay_sha256=verified);start=0
        jobs=header['provenance'].get('jobs')
        if jobs is not None:require(len(jobs)==len(raw),'Replay provenance job count differs')
        for ordinal,g in enumerate(raw):
            count=len(g['records']);game=check_game(g,arrays,validated_inputs=arrays is not None,_feature_offset=start);start+=count
            require(game['game_id'] not in identities,'Duplicate game across shards');identities.add(game['game_id'])
            # Descriptors are table metadata, separate from the immutable board
            # observation/Features-4 contract. The enclosing replay hash seals it.
            proof=jobs[ordinal]['provenance'] if jobs is not None else header['provenance']
            from . import ordinary_cotton
            if ordinary_cotton.KEY in proof:
                game.cotton_seats = ordinary_cotton.validate_assignment(proof)
                for row in game['records']:
                    ordinary_cotton.check_row(row, row['seat'] in game.cotton_seats)
            else:
                require(all(ordinary_cotton.KEY not in r['behavior'] for r in game['records']),
                        'Cotton replay is missing its table assignment')
            if 'seat_contexts' in proof:
                from .neutral_training import check_rows
                for member in [game]+game.get('branches',[]):check_rows(member,proof)
            if 'local_scoring' in proof:
                from .local_scoring_targets import FIELD,identity
                require(all(proof['local_scoring'].get(k)==v for k,v in identity().items()),'Local scoring provenance differs')
                for member in [game]+game.get('branches',[]):
                    require(FIELD in member and member[FIELD]['producer']==proof['local_scoring']['producer'],
                            'Missing/foreign scoring labels in recorded collection')
            if 'ppo_opening_objective' in proof:
                from .opening_objective import validate
                game.ppo_opening_objective=validate(proof['ppo_opening_objective'])
            require('branch' not in game,'Standalone branch replay lacks parent evidence')
            if game.get('branches'):
                require(proof.get('purpose')=='training' and proof.get('kind')=='selfplay' and
                    proof.get('exploration',{}).get('branching') is not None,
                    'Branch replay outside ordinary self-play')
                require(all(child['branch']['contract']==proof['exploration']['branching']['contract']
                            for child in game['branches']), 'Branch contract differs from collection provenance')
            if 'critic_table' in proof:
                game.critic_table=proof['critic_table'];game.table_kind=proof['kind']
            if 'diversity_assignment' in proof:
                from .domino_sampling import bind
                bind(game,proof)
                for child in game.get('branches',[]):bind(child,proof)
            if partition is None or game['split']==partition: output.append(game)
    require(output,'No complete games in requested partition')
    return output
