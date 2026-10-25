"""L05 ordered staged choices and transitions against frozen C# execution."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import unittest

from brass_ai.native import NativeWorker, NativeError
from brass_ai.rust_backend.contracts import first_difference

ROOT=Path(__file__).resolve().parents[2]
CAPTURE=Path(os.environ.get('BRASS_RULES_CAPTURE',ROOT/'artifacts/rust-backend/20261025-l05-capture-v2'))
WORKER=Path(os.environ.get('BRASS_NATIVE_WORKER',ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-worker.exe'))
BRANCHES=Path(os.environ.get('BRASS_RULES_BRANCHES',ROOT/'artifacts/rust-backend/20261025-l05-branches'))


def rows(name):
    receipt=json.loads((CAPTURE/'capture.json').read_text())
    path=CAPTURE/name
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['files'][name]
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        yield from map(json.loads,stream)


class RulesParityTests(unittest.TestCase):
    def worker(self):
        worker=NativeWorker(WORKER,timeout=30);self.addCleanup(worker.close);return worker

    def same(self,expected,actual,label):
        for field in ('state','choices','chance'):
            diff=first_difference(expected[field],actual[field])
            if diff is not None:
                dest=os.environ.get('BRASS_RULES_FAILURE')
                if dest:Path(dest).write_text(json.dumps(dict(label=label,expected=expected,actual=actual),indent=2)+'\n')
                self.fail(f'{label}: {field}: {diff}')

    def test_all_root_legal_sets_and_chance(self):
        worker=self.worker();count=0
        for i,row in enumerate(rows('roots.jsonl.gz')):
            worker.import_state(i%2,row['expected']['state'])
            self.same(row['expected'],worker.rules_snapshot(i%2),row['id']);count+=1
        self.assertEqual(count,546)

    def test_complete_game_transitions(self):
        from brass_ai.learning.reward import CostLedger, terminal_utility
        frozen=ROOT/'artifacts/rust-backend/20261024-g890-l01-l02'
        reward=json.loads((frozen/'baseline.json').read_text())['model']['reward']
        results={g['id']:g for g in json.loads((frozen/'local/fixtures/corpus.json').read_text())['games']}
        worker=self.worker();counts=Counter();stages=Counter();partitions=Counter();regressions={};games=[];session=0
        for row in rows('games.jsonl.gz'):
            if 'initial' in row:
                session=len(games)%2;games.append(row['game']);partitions[row['partition']]+=1;expected=row['initial'];worker.import_state(session,expected['state'])
                self.same(expected,worker.rules_snapshot(session),row['game']+' initial');revision=expected['state']['context']['revision']
                previous=worker.rules_snapshot(session);ledger=CostLedger(reward)
            else:
                command=row['command'];expected=row['expected'];label=f'{row["game"]} step {row["step"]} {command}'
                context=previous['state']['context'];stages[context['era']+':'+('chance' if context['chance'] else context['phase'])]+=1
                # The existing reward implementation consumes public fields from Rust.
                # This checks consequences without introducing a new training reward port.
                view={k:previous['state'][k] for k in ('context','players','slots')}
                if not previous['state']['chance']:ledger.observe(view)
                if 'choice' in command:
                    choice=next(c for c in previous['choices'] if c['id']==command['choice']);ledger.chosen(view,choice)
                    actual=worker.apply_choice(session,revision,command['choice']);counts[command['choice'].split('|')[0]]+=1
                else:actual=worker.resolve_chance(session,revision,command['card']);counts['chance']+=1
                self.same(expected,actual,label);revision=expected['state']['context']['revision']
                before,after=previous['state'],actual['state']
                if 'choice' in command and choice['kind']=='scout':
                    # Regression: trade in place, including duplicate card types;
                    # appending the wildcards changes ordered hidden state.
                    s=before['context']['actor'];hand=list(before['cards']['hands'][s]);hand.remove(choice['card'])
                    first=hand.index(choice['cards'][0]);second=next(i for i,c in enumerate(hand) if i!=first and c==choice['cards'][1])
                    hand[first]='wild:industry';hand[second]='wild:location'
                    self.assertEqual(after['cards']['hands'][s],hand,label)
                    regressions.setdefault('scout-preserves-card-positions',label)
                if 'choice' in command and choice['kind']=='beer':
                    for region,points in [('Nottingham:',3),('Shrewsbury:',4)]:
                        if choice.get('target','').startswith('merchant:'+region):
                            s=before['context']['actor']
                            self.assertEqual(after['players'][s]['vp']-before['players'][s]['vp'],points,label)
                            regressions.setdefault(region[:-1]+'-merchant-vp',label)
                if before['context']['era']!=after['context']['era']:
                    for s in range(4):
                        self.assertEqual(before['players'][s]['mat'],after['players'][s]['mat'],label)
                        self.assertEqual(after['players'][s]['bonus_vp'],0,label)
                    regressions.setdefault('canal-mat-tiles-remain-in-rail',label)
                    regressions.setdefault('canal-bonus-vp-reset',label)
                previous=actual
                if actual['state']['context']['terminal']:
                    self.assertEqual([p['cash'] for p in before['players']],[p['cash'] for p in after['players']],label)
                    self.assertEqual([p['bonus_vp'] for p in after['players']],[0]*4,label)
                    regressions.setdefault('final-rail-no-income-and-bonus-reset',label)
                    ledger.observe({k:actual['state'][k] for k in ('context','players','slots')})
                    result=results[row['game']];utility=terminal_utility(actual['state']['result'],reward).tolist()
                    self.assertEqual(utility,result['terminal_utility']);self.assertEqual(ledger.spent.tolist(),result['insolvency_cost'])
                    self.assertEqual([v-c for v,c in zip(utility,ledger.spent)],result['game_utility'])
                if row['step']%31==0:
                    worker.save_state(session);worker.import_state(session,expected['state']);worker.restore_state(session)
                    self.same(expected,worker.rules_snapshot(session),label+' restore')
        self.assertEqual(len(games),20);self.assertEqual(sum(counts.values()),8344)
        self.assertTrue({'scout-preserves-card-positions','canal-mat-tiles-remain-in-rail','Nottingham-merchant-vp','Shrewsbury-merchant-vp',
                         'canal-bonus-vp-reset','final-rail-no-income-and-bonus-reset'}<=regressions.keys())
        report=dict(format='brass-native-rules-parity/1.0.0',build=worker.hello['build'],roots=546,games=games,
                    transitions=sum(counts.values()),coverage=dict(counts),stages=dict(stages),game_partitions=dict(partitions),
                    regressions=regressions,reward_games=20,unexplained_differences=0,sessions=2,threads_per_session=1)
        if dest:=os.environ.get('BRASS_RULES_PARITY_REPORT'):Path(dest).write_text(json.dumps(report,indent=2)+'\n')

    def test_rejected_choices_chance_and_stale_revision_are_atomic(self):
        worker=self.worker();selected={}
        for row in rows('roots.jsonl.gz'):
            state=row['expected']['state'];key='chance' if state['chance'] else state['context']['phase']
            selected.setdefault(key,row)
        self.assertEqual(set(selected),{'action','payment','continuation','bonus','debt','chance','terminal'})
        other=deepcopy(selected['action']['expected']);other['state']['context']['revision']+=7000
        worker.import_state(1,other['state']);worker.save_state(1)
        for key,row in selected.items():
            expected=row['expected'];worker.import_state(0,expected['state']);worker.save_state(0);revision=expected['state']['context']['revision']
            commands=[lambda:worker.apply_choice(0,revision,'invalid|||||'),lambda:worker.resolve_chance(0,revision,'wild:industry')]
            if expected['choices']:commands.append(lambda:worker.apply_choice(0,revision+1,expected['choices'][0]['id']))
            if expected['chance']:commands.append(lambda:worker.resolve_chance(0,revision+1,expected['chance']['outcomes'][0]['card']))
            for command in commands:
                with self.assertRaises(NativeError):command()
                self.same(expected,worker.rules_snapshot(0),key+' rejected current')
                worker.restore_state(0);self.same(expected,worker.rules_snapshot(0),key+' rejected save')
                worker.restore_state(1);self.same(other,worker.rules_snapshot(1),key+' other owner')

    def test_authored_and_complex_branch_alternatives(self):
        receipt=json.loads((BRANCHES/'capture.json').read_text());path=BRANCHES/'branches.jsonl.gz'
        with path.open('rb') as f:self.assertEqual(hashlib.file_digest(f,'sha256').hexdigest(),receipt['files'][path.name])
        worker=self.worker();count=0
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                session=count%2;state=row['before']['state'];worker.import_state(session,state);worker.save_state(session)
                revision=state['context']['revision'];command=row['command']
                actual=worker.apply_choice(session,revision,command['choice']) if 'choice' in command else worker.resolve_chance(session,revision,command['card'])
                self.same(row['expected'],actual,row['root']+' '+str(command))
                worker.import_state(session,actual['state']);self.same(row['expected'],worker.rules_snapshot(session),row['root']+' resulting state import')
                worker.restore_state(session);self.same(row['before'],worker.rules_snapshot(session),row['root']+' branch restore');count+=1
        self.assertEqual(count,receipt['transitions'])


if __name__=='__main__':unittest.main()
