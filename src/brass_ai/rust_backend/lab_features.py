"""Native I/O for the existing derived-features/4 cache and CheckedGame.

The scheduler holds receipts separately. Read each byte once, reuse the replay
hash already verified by admission, and retain row bindings across process IPC.
The general replay format/encoder version and historical datasets are unchanged.
"""
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from brass_ai.learning.contracts import compatibility, digest, require
from brass_ai.learning.features import Encoded, check_encoded
from brass_ai.learning.local_entities import check_action_selectors
from brass_ai.recovery.replay import CheckedGame, _cache_paths, cache_version, label_key

LIMIT = 512*1024**2


class VerifiedArrays(list):
    """Process-local result of byte/schema/row verification; never a disk flag."""
    def __init__(self,values,rows,bindings):
        super().__init__(values);self.rows=tuple(rows);self.bindings=tuple(bindings)


def bind(game,arrays,*,start=0):
    """Reuse the exact input hashes just checked by unpack(), without rehashing
    the same histories. Check object/order identity and keep detached row seals;
    subsequent input/label/context mutations still fail CheckedGame.encoded().
    """
    rows=game['records'];end=start+len(rows);encoded=arrays[start:end]
    require(len(encoded)==len(rows),'Incomplete verified features')
    if not isinstance(arrays,VerifiedArrays):return CheckedGame(game).bind(encoded)
    require(all(a is b for a,b in zip(rows,arrays.rows[start:end],strict=True)),
            'Verified feature row identity/order changed')
    checked=CheckedGame(game);checked.features={};checked.seals={}
    checked.header_seal=digest({k:v for k,v in checked.items() if k!='records'})
    for row,e,(key,source,label,style,intent) in zip(rows,encoded,arrays.bindings[start:end],strict=True):
        checked.features[key]=e
        checked.seals[(row['step'],row['seat'])]=dict(input=source,label=label,style=list(style),intent=list(intent))
    return checked

# Like admission's validator ID, this pins the loaded code for one immutable
# process. A source deployment always starts fresh workers/readers.
version=lru_cache(maxsize=1)(cache_version)


def read_bytes(path):
    import os
    with Path(path).open('rb') as stream:
        size=os.fstat(stream.fileno()).st_size
        require(0<size<=LIMIT,'Feature cache exceeds bound')
        data=stream.read(size+1)
    require(len(data)==size,'Feature cache size changed')
    return data


def pack(games, replay_sha256, *, compressed=True, _owned=False):
    from brass_ai.recovery.replay_binding import GameBinding,checked_encoded,count
    arrays=[];keys=[];sealed=True
    for game in games:
        require(isinstance(game,CheckedGame),'Feature handoff requires checked game')
        if not _owned:game.assert_metadata()
        else:require(len(game['records'])==len(game.seals),'Owned feature row coverage differs')
        order=iter(game.seals)
        for row in game['records']:
            binding=getattr(game,'_immutable_binding',None)
            if type(binding) is GameBinding:
                proof=binding.row(game,row)
                require(proof is not None,'Immutable feature row binding differs')
                e=game.encoded(row,None);arrays.append(e);keys.append(proof.key)
                continue
            if _owned:
                # Synchronous publisher-only reuse. convert() independently
                # checked these exact private rows; no borrowed mutable inputs
                # or serialized "validated" flags enter this scoped path.
                identity=(row['step'],row['seat'])
                require(identity==next(order),'Owned feature row order changed')
                seal=game.seals[identity]
                require(row['style']==seal['style'] and row['intent']==seal['intent'],'Owned feature context changed')
                key=digest(dict(input=seal['input'],style=row['style'],intent=row['intent']))
                e=game.features[key];check_encoded(e)
                require(len(e.choice_ids)==len(row['choices']),'Owned feature choices changed')
                arrays.append(e);keys.append(key);sealed=sealed and checked_encoded(e)
                count('publication_row_binding_reused');continue
            # One source hash verifies mutation and provides the cache row key.
            source=digest(dict(view=row['view'],choices=row['choices']))
            count('input_hash_executed');count('label_hash_executed')
            seal=game.seals[(row['step'],row['seat'])]
            require(seal==dict(input=source,label=label_key(row),style=row['style'],intent=row['intent']),
                    'Validated replay record changed')
            key=digest(dict(input=source,style=row['style'],intent=row['intent']))
            e=game.features[key];check_encoded(e)
            require(e.choice_ids==tuple(c['id'] for c in row['choices']),'Feature choices changed')
            arrays.append(e);keys.append(key);sealed=sealed and checked_encoded(e)
    out=io.BytesIO()
    states=np.stack([e.state for e in arrays]);actions=np.concatenate([e.actions for e in arrays])
    seats=np.stack([e.stable_to_relative for e in arrays])
    # Establish this once before the writer can attest/cache these bytes. Warm
    # reads of that exact independently bound cache need not rescan every cell.
    if not sealed:
        require(all(np.isfinite(a).all() for a in (states,actions,seats)),'Non-finite feature arrays')
        check_action_selectors(actions)
    count('pack_finite_reused' if sealed else 'pack_finite_executed')
    (np.savez_compressed if compressed else np.savez)(out,
        states=states,actions=actions,
        offsets=np.cumsum([0]+[len(e.choice_ids) for e in arrays]),
        seats=seats)
    raw=out.getvalue();require(len(raw)<=LIMIT,'Feature archive exceeds bound')
    meta=dict(format='derived-features/4.0.0',replay_sha256=replay_sha256,data_sha256=hashlib.sha256(raw).hexdigest(),
        compatibility=compatibility(),encoder=version(),rows=keys)
    return raw,meta


def write(path, games, replay_sha256, *, compressed=True, _owned=False):
    raw,meta=pack(games,replay_sha256,compressed=compressed,_owned=_owned)
    data,manifest=_cache_paths(path)
    for target,value in ((data,raw),(manifest,json.dumps(meta,separators=(',',':')).encode())):
        temporary=target.with_name(target.name+'.tmp')
        try:temporary.write_bytes(value);temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    raw=json.dumps(meta,separators=(',',':')).encode();data_hash=meta['data_sha256']
    return dict(data_sha256=data_hash,manifest_sha256=hashlib.sha256(raw).hexdigest(),
                replay_sha256=replay_sha256,validator=meta['encoder'])


def read(path, games, receipt, replay_sha256,*,deep=True):
    from brass_ai.recovery.feature_receipts import versions
    data,manifest=_cache_paths(path)
    if receipt is None or receipt.get('validator') not in versions(version()) or not data.exists() or not manifest.exists():return None
    require(receipt['replay_sha256']==replay_sha256,'Feature replay binding changed')
    meta_bytes=read_bytes(manifest);raw=read_bytes(data)
    require(hashlib.sha256(meta_bytes).hexdigest()==receipt['manifest_sha256']
            and hashlib.sha256(raw).hexdigest()==receipt['data_sha256'],'Derived feature receipt changed')
    meta=json.loads(meta_bytes)
    require(meta['format']=='derived-features/4.0.0' and meta['replay_sha256']==replay_sha256
            and meta['data_sha256']==receipt['data_sha256'] and meta['encoder']==receipt['validator']
            and meta['compatibility']==compatibility(),'Feature cache binding changed')
    return unpack(raw,games,meta,deep=deep)


def unpack(raw,games,meta,*,combined=False,deep=True):
    from brass_ai.recovery.replay_binding import _bytes_array,seal_encoded,count,enabled
    from brass_ai.recovery.feature_receipts import versions
    require(meta['encoder'] in versions(version()) and meta['compatibility']==compatibility()
            and meta['format']=='derived-features/4.0.0','Feature schema differs')
    rows=[r for g in games for r in g['records']]
    bindings=[]
    for row in rows:
        source=digest(dict(view=row['view'],choices=row['choices']))
        count('input_hash_executed');count('label_hash_executed')
        key=digest(dict(input=source,style=row['style'],intent=row['intent']))
        bindings.append((key,source,label_key(row),tuple(row['style']),tuple(row['intent'])))
    require(meta['rows']==[b[0] for b in bindings],'Feature row binding changed')
    # Bounds are checked before NumPy allocates an archive member.
    import zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        expected={'states.npy','actions.npy','offsets.npy','seats.npy'}
        if combined:expected|={'record.json.gz','features.json'}
        require(set(archive.namelist())==expected and len(archive.infolist())==len(expected)
                and sum(x.file_size for x in archive.infolist())<=LIMIT,
                'Feature archive exceeds bound')
    with np.load(io.BytesIO(raw),allow_pickle=False) as values:
        states,actions,offsets,seats=(values[k] for k in ('states','actions','offsets','seats'))
    require(states.shape==(len(rows),3128) and seats.shape==(len(rows),4)
            and actions.ndim==2 and actions.shape[1]==327 and offsets.shape==(len(rows)+1,)
            and offsets.dtype.kind in 'iu' and offsets[0]==0 and offsets[-1]==len(actions)
            and np.array_equal(np.diff(offsets),[len(r['choices']) for r in rows]),'Malformed feature arrays')
    if deep:require(all(np.isfinite(a).all() for a in (states,actions,seats)),'Non-finite feature arrays')
    # Freeze the three slabs once, rather than copy each overlapping row.
    immutable=enabled()
    if immutable:states,actions,seats=(_bytes_array(a) for a in (states,actions,seats))
    out=[]
    for i,row in enumerate(rows):
        e=Encoded(states[i],actions[offsets[i]:offsets[i+1]],seats[i],tuple(c['id'] for c in row['choices']))
        if deep:
            check_encoded(e);check_action_selectors(e.actions)
            require(np.array_equal(e.stable_to_relative,[(s-row['seat'])%4 for s in range(4)])
                    and np.array_equal(e.state[-12:],np.asarray(row['style']+row['intent'],dtype=np.float32)),
                    'Cached seat/context differs')
        if immutable:out.append(seal_encoded(e,_checked=True))
        else:
            for a in (e.state,e.actions,e.stable_to_relative):a.setflags(write=False)
            out.append(e)
    return VerifiedArrays(out,rows,bindings)
