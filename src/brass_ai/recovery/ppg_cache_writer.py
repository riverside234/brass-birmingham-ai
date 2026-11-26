"""Bounded dense critic serialization; same byte layout and row admission."""
import copy
import json
import operator
import os
import time

import numpy as np

from brass_ai.learning.contracts import require
from brass_ai.learning.features import check_encoded
from brass_ai.learning.seat_descriptors import validate_batch,unknown_numpy

SCALARS=frozenset((str,int,bool,type(None)))
VALUE=operator.itemgetter(1)
SMALL=('seat','game','context','forced','role','table','descriptor_generation')
FINITE=('state','action','targets','utility','remaining')
F4=np.dtype('<f4')
try:IOV_MAX=max(16,min(1024,os.sysconf('SC_IOV_MAX')))
except (AttributeError,ValueError,OSError):IOV_MAX=1024


def flat(context):
    """Every value an immutable JSON scalar of an exact type (no float, list or dict)."""
    return SCALARS.issuperset(map(type,context.values()))


def context_key(context):
    """The identity of a view context in the cache's context table (g6120), compared directly instead of
    through a SHA-256 of its canonical JSON. For the flat scalar contexts of a view (strings, integers,
    booleans) the sorted items and their exact types equal exactly when that canonical JSON does; anything
    else keys by the canonical JSON itself."""
    if flat(context):
        items=tuple(sorted(context.items()))
        return items,tuple(map(type,map(VALUE,items)))
    return json.dumps(context,sort_keys=True,separators=(',',':'),allow_nan=False)


def context_copy(context):
    """deepcopy(context); a flat dict of immutable scalars needs only a new dict."""
    return dict(context) if flat(context) else copy.deepcopy(context)


class _Table:
    def __init__(self):self.games={};self.contexts={};self.context_values=[];self.blocks=0


def _blocks(table,dataset,ids,source,reward_id,selection,generation,block_rows):
    """Every row's source, layout, descriptor and seat admission and its metadata, one block at a time."""
    require(type(block_rows) is int and 1<=block_rows<=1024,'Invalid critic writer block')
    games,contexts,context_values=table.games,table.contexts,table.context_values
    unknown=unknown_numpy()
    for start in range(0,len(ids),block_rows):
        records=[];metadata=[];descriptors=[]
        for i in ids[start:start+block_rows]:
            row,target,identity,encoded=dataset.data[i];role=row.get('_critic_role','learner')
            require(role in ('learner','snapshot') and row['_reward_id']==reward_id
                and selection['games'].get(identity[0])==row.get('_table_kind','selfplay')
                and row.get('behavior',{}).get('component')!='branch_move' and not row.get('_copied_prefix',False)
                and ((role=='learner' and row['behavior']['policy_id']==source['policy_id']) or
                     (role=='snapshot' and source.get('snapshot_rows') and row['behavior']['policy_id']!=source['policy_id'])),
                'Invalid cached MC row/source')
            check_encoded(encoded);context=row['view']['context'];key=context_key(context)
            if key not in contexts:contexts[key]=len(contexts);context_values.append(context_copy(context))
            require(identity[1]==row['seat'],'Nonfinite/mismatched cached critic row')
            descriptors.append(row.get('_seat_descriptors',unknown))
            metadata.append((row['seat'],games.setdefault(identity[0],len(games)),contexts[key],
                len(encoded.choice_ids)==1,('learner','snapshot').index(role),
                ('selfplay','historical','legacy').index(row.get('_table_kind','selfplay')),
                row.get('_descriptor_generation',generation-1)))
            records.append((row,target,identity,encoded))
        yield start,records,metadata,descriptors
        table.blocks+=1


def _fill_small(block,records,metadata,descriptors):
    """Every field after the state and the first action."""
    block['mapping']=[r[3].stable_to_relative for r in records]
    block['targets']=[[r[1][k] for k in ('win_share','placement','vp')] for r in records]
    block['utility']=[r[0]['_reward_return'] for r in records]
    block['remaining']=[r[0]['_reward_remaining'] for r in records]
    block['descriptors']=validate_batch(descriptors)
    for j,name in enumerate(SMALL):
        block[name]=[m[j] for m in metadata]


def _fill(block,records,metadata,descriptors):
    # Reuse the encoded arrays directly in the staged file, without a
    # second dense structured buffer. Check the destination FP32 values.
    block['state']=[r[3].state for r in records]
    block['action']=[r[3].actions[0] for r in records]
    _fill_small(block,records,metadata,descriptors)
    require(all(np.isfinite(block[k]).all() for k in FINITE),'Nonfinite/mismatched cached critic row')


def write_critic(array,dataset,ids,source,reward_id,selection,generation,*,block_rows=256):
    table=_Table();began=time.perf_counter()
    for start,records,metadata,descriptors in _blocks(table,dataset,ids,source,reward_id,selection,generation,block_rows):
        _fill(array[start:start+len(records)],records,metadata,descriptors)
    return table.games,table.context_values,dict(rows=len(ids),blocks=table.blocks,block_rows=block_rows,
        finite_array_checks=table.blocks*5,max_block_bytes=min(block_rows,len(ids))*array.dtype.itemsize,
        serialization_seconds=time.perf_counter()-began,
        scope='Same per-row source/layout/descriptor/seat checks; batched destination-dtype finite checks and contiguous writes; unchanged dense bytes')


# ---- g6120: the rows' bytes written straight to the file -----------------------------------------------------

def layout(dtype):
    """(byte offset of the small fields, their packed dtype) when a record is its state, its first action, then
    the small fields, with no padding; None otherwise (every block then takes the structured path)."""
    names=dtype.names;fields=dtype.fields
    if names[:2]!=('state','action') or fields['state'][1]!=0 or fields['action'][1]!=fields['state'][0].itemsize:return None
    start=fields['action'][1]+fields['action'][0].itemsize
    small=np.dtype([(n,fields[n][0]) for n in names[2:]])
    if start+small.itemsize!=dtype.itemsize or any(fields[n][1]-start!=small.fields[n][1] for n in names[2:]):return None
    return start,small


def _direct(encoded,dtype):
    """The row's own state and first action are already the record's bytes: C-contiguous little-endian float32."""
    s,a=encoded.state,encoded.actions
    return (isinstance(s,np.ndarray) and isinstance(a,np.ndarray) and s.dtype==F4 and a.dtype==F4
            and s.shape==dtype.fields['state'][0].shape and a.ndim==2 and len(a)>0
            and a.shape[1:]==dtype.fields['action'][0].shape and s.flags.c_contiguous and a.flags.c_contiguous)


def write_buffers(fd,buffers,*,writev=None):
    """Write the buffers, in order, at the descriptor's position. Where the system has writev, one gather write
    per IOV_MAX buffers: the kernel copies straight from the rows' own arrays into the page cache, with no
    staging copy in between. Elsewhere their bytes are joined once and written."""
    writev=getattr(os,'writev',None) if writev is None else writev
    views=[v for v in (memoryview(b).cast('B') for b in buffers) if v.nbytes]
    if writev is None:
        view=memoryview(b''.join(views))
        while view:view=view[os.write(fd,view):]
        return
    i=0
    while i<len(views):
        n=writev(fd,views[i:i+IOV_MAX])
        require(n>0,'Critic file write made no progress')
        while n:
            size=views[i].nbytes
            if n>=size:n-=size;i+=1
            else:views[i]=views[i][n:];n=0


def write_critic_file(array,dataset,ids,source,reward_id,selection,generation,*,block_rows=256,writev=None):
    """write_critic into a freshly created open_memmap file (g6120): the same bytes, admission and checks, without
    copying rows into the mapping.

    Each block goes to the file in one gather write of every row's own state and first action and a small
    buffer of the remaining fields, so there is no temporary array, no strided copy and no page fault per page
    of the mapping; the kernel's copy into the page cache is the only one. The finite check stays on the
    destination values: the remaining fields' in their buffer, and the state and action are the source bytes
    themselves, so a sealed row (checked finite when it was sealed, immutable since) needs nothing more and any
    other row is checked here. A block with any other array layout or type takes the structured path. The file
    is synced before returning, so a receipt never names bytes that are not on disk. Returns (games, contexts,
    report).
    """
    from .replay_binding import checked_encoded
    table=_Table();began=time.perf_counter();form=layout(array.dtype)
    fd=os.open(array.filename,os.O_WRONLY|getattr(os,'O_BINARY',0))
    direct=structured=source_checks=checks=0;flush_seconds=0.
    try:
        os.lseek(fd,array.offset,os.SEEK_SET)
        small=scratch=None
        for start,records,metadata,descriptors in _blocks(table,dataset,ids,source,reward_id,selection,generation,block_rows):
            n=len(records)
            if form is not None and all(_direct(r[3],array.dtype) for r in records):
                if small is None:small=np.empty(block_rows,form[1])
                part=small[:n];_fill_small(part,records,metadata,descriptors)
                require(all(np.isfinite(part[k]).all() for k in FINITE[2:]),'Nonfinite/mismatched cached critic row')
                checks+=len(FINITE)-2
                for r in records:
                    if not checked_encoded(r[3]):
                        require(np.isfinite(r[3].state).all() and np.isfinite(r[3].actions[0]).all(),
                                'Nonfinite/mismatched cached critic row')
                        source_checks+=1
                width=form[1].itemsize;tail=memoryview(part.view(np.uint8));buffers=[]
                for j,r in enumerate(records):
                    buffers+=(r[3].state,r[3].actions[0],tail[j*width:(j+1)*width])
                direct+=n
            else:
                if scratch is None:scratch=np.empty(block_rows,array.dtype)
                part=scratch[:n];_fill(part,records,metadata,descriptors);buffers=[part.view(np.uint8)]
                checks+=len(FINITE);structured+=n
            write_buffers(fd,buffers,writev=writev)
        flush_began=time.perf_counter();os.fsync(fd);flush_seconds=time.perf_counter()-flush_began
    finally:
        os.close(fd)
    return table.games,table.context_values,dict(rows=len(ids),blocks=table.blocks,block_rows=block_rows,
        finite_array_checks=checks,source_finite_rows=source_checks,direct_rows=direct,structured_rows=structured,
        writer='gather-write' if (writev or getattr(os,'writev',None)) else 'joined-write',
        max_block_bytes=min(block_rows,len(ids))*array.dtype.itemsize,
        serialization_seconds=time.perf_counter()-began-flush_seconds,critic_flush_seconds=flush_seconds,
        scope='Same per-row source/layout/descriptor/seat checks and destination finite checks (a sealed row\'s state '
              'and action from its sealing); each block written from the rows\' own arrays in one gather write; '
              'unchanged dense bytes')
