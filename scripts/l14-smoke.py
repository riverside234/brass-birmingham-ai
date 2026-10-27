"""Bounded L14 follow-ups; never runs a tournament or changes installed files.

Every invocation is a separate, recorded group (90 s process deadline). All
matched searches retain 4 s and two single-thread owners. Count-mode cases
isolate changed code and cannot be interpreted as a timed search speed claim.
"""
import argparse
import copy
import json
import runpy
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
read, write, receipt = (SUPPORT[n] for n in ('read','write','receipt'))
STUDY = ROOT/'artifacts/rust-backend/20261026-l14-study-v2'
SMOKE = ROOT/'artifacts/rust-backend/20261026-l14-smoke'
BIN = SMOKE/'candidate-native'
L10 = ROOT/'artifacts/rust-backend/20261025-l10-capture'


def run(binary, request, folder, label):
    inp, out = folder/(label+'-input.json'), folder/(label+'-output.json')
    assert not out.exists()
    inp.write_text(json.dumps(request,separators=(',',':'),allow_nan=False),encoding='utf-8')
    started=time.monotonic()
    with (folder/(label+'.log')).open('w',encoding='utf-8') as stream:
        with subprocess.Popen([str(binary),str(inp),str(out)],stdout=stream,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW) as process:
            try:
                code=process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                # Kill only this owned diagnostic process tree, including its
                # external evaluator children; never leave a hung test running.
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=stream,stderr=subprocess.STDOUT,timeout=10,check=False)
                if process.poll() is None:process.kill()
                process.wait(timeout=10)
                write(folder/(label+'-timeout.json'),dict(pid=process.pid,timeout_seconds=90,owned_process_tree_retired=True))
                raise
            if code:raise subprocess.CalledProcessError(code,[str(binary),str(inp),str(out)])
    write(folder/(label+'-receipt.json'),dict(binary=receipt(binary),input=receipt(inp),output=receipt(out),
          wall_seconds=time.monotonic()-started,timeout_seconds=90))
    print(label,'complete',flush=True)
    return read(out)


def semantics(value):
    # Instrumentation and monotonic scheduling times are not semantics.
    excluded={'metrics','neural','elapsed_ns','started_ns','finished_ns','response_ready_ns',
              'retired_ns','ready_ns','published_ns','elapsed','profile','last_completion_ns'}
    if isinstance(value,dict):return {k:semantics(v) for k,v in value.items() if k not in excluded and not k.endswith('_ns')}
    if isinstance(value,list):return [semantics(v) for v in value]
    return value


def traces(folder):
    # Recorded input/reply tapes independently detect views, ledgers, choices,
    # first-new leaves and backup changes. Include all captured phase families.
    files=[L10/f'brass-{i:03d}-request.json' for i in (0,3,8,26,30,41)]
    # Add the first debt and continuation examples without an exhaustive run.
    for p in sorted(L10.glob('brass-*-request.json')):
        phase=read(p)['cases'][0]['input']['view']['context']['phase']
        if phase in ('debt','continuation') and not any(read(f)['cases'][0]['input']['view']['context']['phase']==phase for f in files):files.append(p)
    assert files
    request=dict(cases=[read(p)['cases'][0] for p in files])
    old=run(SMOKE/'baseline-native/gumbel-trace.exe',request,folder,'before')
    new=run(BIN/'gumbel-trace.exe',request,folder,'after')
    assert semantics(old)==semantics(new),'search trace changed'
    write(folder/'results.json',dict(passed=True,cases=len(files),full_trace_equal=True,
          phases=sorted({c['input']['view']['context']['phase'] for c in request['cases']}),
          source_files=[receipt(p) for p in files]))


def pairs(folder, kind, arm, repetition):
    req=read(STUDY/'native-components'/('timed-input.json' if kind=='timed' else 'components-nn-input.json'))
    req['cases']=req['cases'][:6]
    for c in req['cases']:c['config']['seed']+=repetition
    if kind=='allocation':
        for c in req['cases']:c['allowance_ms']=60000
    binary=(SMOKE/'baseline-native' if arm=='before' else BIN)/('gumbel-pair-alloc.exe' if kind=='allocation' else 'gumbel-pair.exe')
    result=run(binary,req,folder,f'{kind}-{arm}-{repetition}')
    for r in result['results']:
        assert r['result']['status']=='complete' and not r['busy_after_retirement']
    write(folder/'results.json',dict(passed=True,kind=kind,arm=arm,repetition=repetition,
          protocol=receipt(SMOKE/'protocol.json'),rows=[dict(name=r['name'],elapsed_ns=r['elapsed_ns'],
          allocations=r['allocations'],selection=r['result']['decision']['selection'],
          ready_ns=r['result']['decision']['response_ready_ns']) for r in result['results']]))


def external(folder,mode,timed):
    req=read(STUDY/'native-components'/('timed-input.json' if timed else 'components-nn-input.json'))
    req['cases']=req['cases'][:2]
    if not timed:
        for c in req['cases']:c['config']['simulations']=16
    checkpoint=read(ROOT/'artifacts/rust-backend/20261024-g890-l01-l02/baseline.json')['model']['path']
    req['external']=dict(python=sys.executable,script=str(ROOT/'tests/RustBackend/external_evaluator.py'),mode=mode,
        model=str(Path(req['manifest']).parent),checkpoint=str(ROOT/checkpoint),
        reference_source=str(ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/reference-source'))
    shutil.copyfile(ROOT/'tests/RustBackend/external_evaluator.py',folder/'external_evaluator.executed.py')
    result=run(BIN/'gumbel-pair.exe',req,folder,mode)
    assert all(r['result']['status']=='complete' and not r['busy_after_retirement'] for r in result['results'])
    write(folder/'results.json',dict(passed=True,mode=mode,timed=timed,algorithm='Same Rust Pair and session implementation',
        python_reference_scope='External evaluator control only; not an independent Python timed search',
        external_source=receipt(ROOT/'tests/RustBackend/external_evaluator.py'),rows=result['results']))


def python_reference(folder,timed,whole_turn=False,evaluator_mode='ort'):
    controls=runpy.run_path(str(ROOT/'scripts/l14-search-controls.py'));controls['imports']()
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.native.simulation import NativeSpielGame
    from brass_ai.searchlab.spiel import action_id
    from brass_ai.learning.reward import CostLedger
    module=runpy.run_path(str(ROOT/'tests/RustBackend/gumbel_session_reference.py'))
    shutil.copyfile(ROOT/'tests/RustBackend/gumbel_session_reference.py',folder/'reference.py')
    model=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/model/model.onnx'
    def evaluator():
        value=module['gp']['Neural'](model)
        if evaluator_mode=='torch':
            external=runpy.run_path(str(ROOT/'tests/RustBackend/external_evaluator.py'))
            checkpoint=ROOT/read(ROOT/'artifacts/rust-backend/20261024-g890-l01-l02/baseline.json')['model']['path']
            value.evaluator=external['Model']('torch',model.parent,checkpoint)
        else:assert evaluator_mode=='ort'
        return value
    req=read(STUDY/'native-components'/('timed-input.json' if timed else 'components-nn-input.json'))
    rows=[];started=time.monotonic()
    for index,case in enumerate(req['cases'][:2]):
        worker=NativeWorker(L10/'reference-worker.exe',timeout=10)
        pair=module['Pair'](worker,evaluator)
        environment=None;state=None;env_worker=None
        try:
            config=copy.deepcopy(case['config']);config['simulations']=16
            # Warm the same evaluator/feature path before starting the clock.
            warm=read(L10/'brass-003-request.json')['cases'][0]['replies'][0]['info']
            for pool,owner in zip(pair.pools,pair.owners):pool.submit(owner.evaluator,warm).result()
            mode=copy.deepcopy(case['mode'])
            if timed:
                calibration=read(SMOKE/'gpy-count-02/results.json')['rows'][index]['result']['cost_samples']
                values=[sorted(bucket)[(len(bucket)-1)//4] for owner in calibration for bucket in owner if len(bucket)>=4]
                assert len(values)==2
                mode['startup']['estimate']=dict(nanoseconds=max(values),source='G_py singleton last-eight lower-quartile calibration, maximum of the two owners; gpy-count-02')
            if whole_turn:
                env_worker=NativeWorker(L10/'reference-worker.exe',timeout=10)
                environment=NativeSpielGame(worker=env_worker,session=0,root=ROOT)
                state=environment.from_observation(case['input']['view'],91234,case['input']['recall'])
            result=pair.decide(case['input'],config,mode,4000 if timed else 60000)
            record=dict(name=case['name'],result=result)
            if whole_turn:
                ledger=CostLedger(case['input']['reward'],**case['input']['ledger']);ledger.observe(case['input']['view'])
                ordinary=tuple(state.context[k] for k in ('era','round','turn','turn_owner','actor'))
                original_deadline=pair.deadline;decisions=[result];closed=0
                while True:
                    for a in decisions[-1]['selection']['actions']:
                        for st in a['action']['steps']:
                            choice=next(c for c in state.legal if c['id']==st['choice'])
                            ledger.chosen(state.views[state.context['actor']],choice);state.apply_action(action_id(st['choice']))
                            if not state.is_chance_node():ledger.observe(state.views[case['input']['view']['seat']])
                        closed+=1
                    if state.is_terminal() or state.is_chance_node() or tuple(state.context[k] for k in ('era','round','turn','turn_owner','actor'))!=ordinary:break
                    assert closed<2
                    next_input=copy.deepcopy(case['input']);seat=state.context['actor']
                    next_input['view']=copy.deepcopy(state.views[seat]);next_input['recall']=copy.deepcopy(state.recall[seat])
                    next_input['context']['remaining']=list(ledger.remaining)
                    next_input['ledger']=dict(spent=list(ledger.spent),settled=sorted(ledger.settled))
                    decisions.append(pair.decide(next_input,config,mode,4000 if timed else 60000))
                    assert pair.deadline==original_deadline,'ordinary clock reset'
                record.update(whole_turn_complete=True,complete_actions=closed,decisions=decisions,
                    original_deadline_preserved=pair.deadline==original_deadline,
                    elapsed_ns=(4000 if timed else 60000)*1_000_000-(original_deadline-time.perf_counter_ns()))
            rows.append(record)
        finally:
            if state is not None:state.close()
            if environment:environment.close()
            if env_worker:env_worker.close()
            pair.close();worker.close()
        assert time.monotonic()-started<90,'smoke group exceeded its bound'
    write(folder/'results.json',dict(passed=True,timed=timed,evaluator=evaluator_mode,rows=rows,wall_seconds=time.monotonic()-started,
          source=receipt(ROOT/'tests/RustBackend/gumbel_session_reference.py')))
    print('G_py',len(rows),'cases completed',flush=True)


def rules(folder):
    controls=runpy.run_path(str(ROOT/'scripts/l14-search-controls.py'));controls['imports']()
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.native.simulation import NativeSpielGame
    from brass_ai.searchlab.spiel import BrassSpielGame,action_id
    # Reuse the already admitted immutable host inputs in a fresh writable copy.
    prior=STUDY/'legacy-controls-v4/reference-host';private=folder/'reference-host'
    shutil.copytree(prior,private,ignore=shutil.ignore_patterns('__pycache__'))
    pins=[receipt(p) for p in private.rglob('*') if p.is_file()]
    write(folder/'host-pins.json',pins)
    native=NativeWorker(L10/'reference-worker.exe',timeout=10)
    games={'C#':BrassSpielGame(root=private,fast=True),'Rust':NativeSpielGame(worker=native,session=0,root=private)}
    def projection(s):
        context={k:v for k,v in s.context.items() if k!='game_id'}
        return dict(context=context,views=s.views,legal=sorted(s.legal,key=lambda c:c['id']),chance=s.pending,result=s.final_result)
    rows=[];started=time.monotonic()
    try:
        for index in (0,3,26,30,41,13):
            case=read(L10/f'brass-{index:03d}-request.json')['cases'][0]['input'];expected=None;trace=None
            for rep in range(2):
                for arm in (['C#','Rust'] if rep==0 else ['Rust','C#']):
                    game=games[arm];times={'sample_ns':0,'clone_ns':0,'legal_ns':0,'step_ns':0};states=[];ids=[]
                    begin=time.perf_counter_ns();state=game.from_observation(case['view'],91234,case['recall']);times['sample_ns']=time.perf_counter_ns()-begin
                    try:
                        begin=time.perf_counter_ns();clone=state.clone();times['clone_ns']=time.perf_counter_ns()-begin
                        try:assert projection(clone)==projection(state)
                        finally:
                            if hasattr(clone,'close'):clone.close()
                        for step in range(8):
                            states.append(projection(state))
                            if state.is_terminal() or state.is_chance_node():break
                            begin=time.perf_counter_ns();legal=state.legal_actions();times['legal_ns']+=time.perf_counter_ns()-begin
                            assert legal
                            if trace is None:
                                choice=min(state.legal,key=lambda c:(c['kind']!='finish',c['id']))['id']
                            else:
                                if step==len(trace):break
                                choice=trace[step]
                            ids.append(choice);begin=time.perf_counter_ns();state.apply_action(action_id(choice));times['step_ns']+=time.perf_counter_ns()-begin
                            if choice.startswith('finish|') or choice.startswith('liquidate|'):
                                states.append(projection(state));break
                        if expected is None:expected=states;trace=ids
                        else:assert states==expected and ids==trace,(index,arm,'rules mismatch')
                        rows.append(dict(index=index,era=case['view']['context']['era'],phase=case['view']['context']['phase'],arm=arm,repeat=rep,steps=len(ids),trace=ids,**times))
                    finally:
                        if hasattr(state,'close'):state.close()
            assert time.monotonic()-started<90,'rules smoke group bound'
    finally:
        for game in games.values():game.close()
        native.close()
    write(folder/'results.json',dict(passed=True,rows=rows,wall_seconds=time.monotonic()-started,
        boundary='Same Python simulator API: sampled world, clone, legal_actions and identical staged steps. Includes each adapter IPC/projection/reconstruction; not a bare Rust-vs-C# language claim.',
        parity='Exact context except transport UUID, all four sampled views, legal choices, chance/result and ordered steps. No rules differences normalized.',host_pins=receipt(folder/'host-pins.json')))
    print('Rules API smoke passed',flush=True)


def quality(folder):
    """Two bounded decisions, not a complete-game or strength experiment."""
    controls=runpy.run_path(str(ROOT/'scripts/l14-search-controls.py'));controls['imports']()
    from brass_ai.bank import FrozenOpponent
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.native.simulation import NativeSpielGame
    from brass_ai.searchlab.spiel import action_id
    bank='bank-3dc65253a610191d66a5/pottery-flex-v3-masked-economy'
    opponent=FrozenOpponent(ROOT,bank,purpose='evaluation');opponent.worker.timeout=15
    native=NativeWorker(L10/'reference-worker.exe',timeout=10)
    game=NativeSpielGame(worker=native,session=0,root=ROOT);rows=[]
    template=read(STUDY/'native-components/timed-input.json')
    try:
        for index in (3,33):
            source=read(L10/f'brass-{index:03d}-request.json')['cases'][0];inp=copy.deepcopy(source['input']);inp.pop('shuffle_tape',None)
            case=copy.deepcopy(template['cases'][0]);case['input']=inp;case['name']=f'quality-{index:03d}'
            case['mode']['startup']['phase']=inp['view']['context']['phase']
            request={k:v for k,v in template.items() if k!='cases'};request['cases']=[case]
            decision=run(BIN/'gumbel-pair.exe',request,folder,f'nn-{index:03d}')['results'][0]['result']['decision']
            nn_actions=decision['result']['action_ids']
            state=game.from_observation(inp['view'],91234,inp['recall'])
            try:choices=copy.deepcopy(state.legal)
            finally:state.close()
            profile=dict(opponent.profile,milliseconds=4000,simulations=100000,deadline_admission=True)
            start=time.perf_counter_ns()
            v4=opponent.worker.call('plan_forest',observation=inp['view'],choices=choices,profile=profile,
                fixed_work=False,sampling_seed=91234,intent=None,search_workers=2)
            v4_ns=time.perf_counter_ns()-start
            assert v4['failures']==0
            write(folder/f'v4-{index:03d}.json',v4)
            for arm,actions,elapsed in [('Rust NN',nn_actions,decision['response_ready_ns']),('qualified V4',v4['selected']['actions'],v4_ns)]:
                state=game.from_observation(inp['view'],91234,inp['recall']);closed=0;trace=[]
                try:
                    for action in actions:
                        assert action
                        for semantic in action:
                            assert semantic in state.semantic_map().values(),(arm,semantic)
                            state.apply_action(action_id(semantic));trace.append(semantic)
                        assert action[-1].startswith(('finish|','liquidate|'));closed+=1
                    rows.append(dict(index=index,era=inp['view']['context']['era'],phase=inp['view']['context']['phase'],arm=arm,
                        elapsed_ns=elapsed,actions=actions,complete_actions=closed,terminal=state.is_terminal(),official_result=state.final_result,
                        final_context=state.context,legal=True))
                finally:state.close()
    finally:game.close();native.close();opponent.close()
    write(folder/'results.json',dict(passed=True,rows=rows,control=bank,whole_turn_ms=4000,sessions=2,threads_per_session=1,
        scope='Decision and near-terminal complete-action legality/settlement smoke only. No win-rate, noninferiority or statistical strength claim.',
        control_manifest=receipt(ROOT/'training/opponents/bank'/bank.split('/')[0]/'manifest.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['traces','count','allocation','timed','external','gpy','rules','quality'])
    p.add_argument('--label',required=True);p.add_argument('--arm',choices=['before','after'],default='after')
    p.add_argument('--repetition',type=int,choices=[0,1],default=0);p.add_argument('--mode',choices=['torch','ort','scripted'],default='ort')
    p.add_argument('--timed',action='store_true');p.add_argument('--whole-turn',action='store_true');a=p.parse_args()
    folder=SMOKE/a.label;folder.mkdir(exist_ok=False)
    shutil.copyfile(__file__,folder/'runner.py')
    if a.kind=='traces':traces(folder)
    elif a.kind=='external':external(folder,a.mode,a.timed)
    elif a.kind=='gpy':python_reference(folder,a.timed,a.whole_turn,a.mode)
    elif a.kind=='rules':rules(folder)
    elif a.kind=='quality':quality(folder)
    else:pairs(folder,a.kind,a.arm,a.repetition)
