"""Deterministic disjoint, stratified raw-policy samples with inclusion correction."""
import hashlib
from collections import defaultdict

from brass_ai.learning.contracts import require


def stratum(row):
    from .exploration import group
    return row.get('_table_kind','selfplay')+'/'+group(row['view']['context'])


def allocation(counts,limit,minimum=1):
    """Minimum quotas, then proportional remaining capacity; no RNG use."""
    counts={k:n for k,n in sorted(counts.items()) if n}
    limit=min(limit,sum(counts.values()))
    require(limit>=min(minimum*len(counts),sum(counts.values())),
        'Policy sample budget cannot cover represented strata')
    out={k:min(minimum,n) for k,n in counts.items()}
    while sum(out.values())<limit:
        # Largest proportional deficit, with deterministic ties.
        k=max((k for k in counts if out[k]<counts[k]),
            key=lambda k:(limit*counts[k]/sum(counts.values())-out[k],k))
        out[k]+=1
    return out


def select(data,actor_ids,config):
    groups=defaultdict(list);ordinal=defaultdict(int)
    for i in actor_ids:
        row,_,identity,encoded=data[i]
        require(row.get('_critic_role','learner')=='learner' and len(encoded.choice_ids)>1
            and row.get('behavior',{}).get('component')!='branch_move'
            and not row.get('_copied_prefix',False),'Ineligible PPG policy reference')
        key=(identity,ordinal[identity]);ordinal[identity]+=1
        groups[stratum(row)].append((hashlib.sha256(repr(key).encode()).hexdigest(),i))
    counts={k:len(v) for k,v in groups.items()}
    # Every represented stratum gets disjoint clone/probe support when it has
    # two rows. Singleton strata go to the safeguard probe and are reported.
    clone_capacity={k:max(0,n-1) for k,n in counts.items()}
    clone_n=min(config['clone_rows_per_age'],sum(counts.values())//2,sum(clone_capacity.values()))
    clone_n=max(clone_n,sum(n>0 for n in clone_capacity.values()))
    clone=allocation(clone_capacity,clone_n) if clone_n else {}
    remaining={k:counts[k]-clone.get(k,0) for k in counts}
    probe_n=min(config['current_probe_rows'],sum(remaining.values()))
    probe=allocation(remaining,probe_n) if probe_n else {}
    selected=[]
    for k,rows in sorted(groups.items()):
        rows.sort();n=clone.get(k,0);m=probe.get(k,0)
        selected.extend(dict(index=i,stratum=k,rank=h,split='clone') for h,i in rows[:n])
        selected.extend(dict(index=i,stratum=k,rank=h,split='probe') for h,i in rows[n:n+m])
    return selected,counts


def phase_indices(entries,counts,limit,split):
    groups=defaultdict(list)
    for i,e in enumerate(entries):
        if e['split']==split:groups[e['stratum']].append((e['rank'],i))
    quotas=allocation({k:len(v) for k,v in groups.items()},limit) if groups else {}
    selected=[];weights=[];support={}
    population=sum(counts.values())
    for k,rows in sorted(groups.items()):
        rows.sort();n=quotas[k]
        selected.extend(i for _,i in rows[:n])
        # Correct oversampled rare cells back to the eligible row population.
        weights.extend([counts[k]/population/n]*n)
        support[k]=dict(eligible=counts[k],selected=n,weight_mass=counts[k]/population)
    missing=sorted(set(counts)-set(groups))
    return selected,weights,dict(strata=support,missing_strata=missing,
        eligible=population,selected=len(selected),weight_mass=sum(weights))
