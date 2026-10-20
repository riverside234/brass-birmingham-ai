"""Bounded native fork/hidden-information checks on 100 independent games."""
import argparse
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import random
import time

import numpy as np

from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery.host_runtime import training_host
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.step5.league import write_json


def run_group(indices):
    rows=[];encoder=PreparedEncoder()
    with RulesHost(training_host(SOURCE_ROOT)) as host:
        for game in indices:
            rng=random.Random(8000+game);context=host.reset(8000+game)
            action_count=0
            while action_count<12:
                choices=host.choices();choice=rng.choice(choices)
                action_count+=int(context['phase']=='action')
                context=host.apply(context,choice)
            while context['phase']!='action':
                context=host.apply(context,rng.choice(host.choices()))
            cp=host.call('checkpoint');views=[host.observe(s) for s in range(4)];choices=host.choices()
            actor=context['actor']
            encoded=encoder.encode(views[actor],choices,[0]*4,[0]*8).state.copy()
            host.call('fork_save');host.call('fork_resume',shuffle_seed=game+12345)
            new_views=[host.observe(s) for s in range(4)]
            require(new_views==views,'Deck shuffle changed a permitted observation')
            require(host.choices()==choices,'Deck shuffle changed legal actions')
            require(np.array_equal(encoded,encoder.encode(new_views[actor],choices,[0]*4,[0]*8).state),'Deck shuffle changed encoding')
            changed=host.call('checkpoint')['commands'][-1]['training_deck']
            original=cp['commands'][-1]['cards_after']
            require({k:v for k,v in changed.items() if k!='deck'}=={k:v for k,v in original.items() if k!='deck'},
                'Fork shuffle touched hands, discards or wild decks')
            require(Counter(changed['deck'])==Counter(original['deck']),'Fork shuffle changed the deck multiset')
            # Restore the original snapshot, replay one full continuation twice.
            context=host.call('restore',checkpoint=cp);host.call('fork_save')
            decisions=[]
            while not context['terminal']:
                require(len(decisions)<2000,'Native smoke decision cap')
                choice=rng.choice(host.choices());context=host.apply(context,choice)
                decisions.append((choice['id'],context))
            expected_result=host.call('result');expected_hash=host.call('canonical_hash')
            context=host.call('fork_resume',shuffle_seed=None)
            for choice,expected in decisions:
                context=host.apply(context,dict(id=choice))
                require(context==expected,'Fork replay context differs')
            require(host.call('result')==expected_result and host.call('canonical_hash')==expected_hash,
                'Fork replay terminal state differs')
            rows.append(dict(seed=8000+game,continuation_decisions=len(decisions),
                shuffle_changed_deck=changed['deck']!=original['deck']))
            print(f'Fork seed {8000+game}: exact full continuation ({len(decisions)} decisions)',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--games',type=int,default=100)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new artifact output')
    require(1<=a.workers<=8 and 1<=a.games<=100,'Bounded fork smoke only')
    out.mkdir(parents=True);began=time.monotonic();rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        pending=[executor.submit(run_group,list(range(i,a.games,a.workers))) for i in range(a.workers)]
        for future in as_completed(pending):rows.extend(future.result())
    rows.sort(key=lambda r:r['seed'])
    write_json(out/'report.json',dict(passed=True,games=a.games,elapsed_seconds=time.monotonic()-began,
        workers=a.workers,
        terminal_replays_exact=True,all_seat_observations_unchanged=True,acting_seat_features_unchanged=True,
        deck_only_shuffle=True,rows=rows,scope='Seeded random legal policies; native full-continuation determinism, not playing strength'))


if __name__=='__main__':main()
