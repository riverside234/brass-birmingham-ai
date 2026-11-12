"""Bounded dense critic serialization; same byte layout and row admission."""
import copy
import time

import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.features import check_encoded
from brass_ai.learning.seat_descriptors import validate_batch,unknown_numpy


def write_critic(array,dataset,ids,source,reward_id,selection,generation,*,block_rows=256):
    require(type(block_rows) is int and 1<=block_rows<=1024,'Invalid critic writer block')
    games={};contexts={};context_values=[];blocks=0
    unknown=unknown_numpy();began=time.perf_counter()
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
            check_encoded(encoded);context=row['view']['context'];key=digest(context)
            if key not in contexts:contexts[key]=len(contexts);context_values.append(copy.deepcopy(context))
            require(identity[1]==row['seat'],'Nonfinite/mismatched cached critic row')
            descriptors.append(row.get('_seat_descriptors',unknown))
            metadata.append((row['seat'],games.setdefault(identity[0],len(games)),contexts[key],
                len(encoded.choice_ids)==1,('learner','snapshot').index(role),
                ('selfplay','historical','legacy').index(row.get('_table_kind','selfplay')),
                row.get('_descriptor_generation',generation-1)))
            records.append((row,target,identity,encoded))
        # Reuse the encoded arrays directly in the staged file, without a
        # second dense structured buffer. Check the destination FP32 values.
        block=array[start:start+len(records)]
        block['state']=[r[3].state for r in records]
        block['action']=[r[3].actions[0] for r in records]
        block['mapping']=[r[3].stable_to_relative for r in records]
        block['targets']=[[r[1][k] for k in ('win_share','placement','vp')] for r in records]
        block['utility']=[r[0]['_reward_return'] for r in records]
        block['remaining']=[r[0]['_reward_remaining'] for r in records]
        block['descriptors']=validate_batch(descriptors)
        for j,name in enumerate(('seat','game','context','forced','role','table','descriptor_generation')):
            block[name]=[m[j] for m in metadata]
        require(all(np.isfinite(block[k]).all() for k in ('state','action','targets','utility','remaining')),
            'Nonfinite/mismatched cached critic row')
        blocks+=1
    return games,context_values,dict(rows=len(ids),blocks=blocks,block_rows=block_rows,
        finite_array_checks=blocks*5,max_block_bytes=min(block_rows,len(ids))*array.dtype.itemsize,
        serialization_seconds=time.perf_counter()-began,
        scope='Same per-row source/layout/descriptor/seat checks; batched destination-dtype finite checks and contiguous writes; unchanged dense bytes')
