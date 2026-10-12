"""Bounded whole-turn CPU/Intel comparisons; strength is a separate P5 gate."""
from pathlib import Path
import platform
import statistics
from threading import Event,Thread,Timer
import time

from brass_ai.learning.contracts import require,digest
from .contracts import BENCHMARK,BACKENDS,read,write,seal
from .runtime import Runtime
from .sampling import new_save


def processor_name():
    if platform.system()=='Windows':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key,'ProcessorNameString')[0]
        except OSError:pass
    return platform.processor()


def measure(directory,cases,output,*,backends=('ort-cpu',),tiers=('easy','medium','hard'),fixture=False,mono=None,unity_running=False):
    require(1<=len(cases)<=24 and set(backends)<=set(BACKENDS) and len(set(backends))==len(backends),'Use 1–24 permitted positions and distinct supported backends')
    require(set(tiers)<={'easy','medium','hard'} and tiers,'Unknown tiers')
    out=Path(output);require(not out.exists(),'Do not overwrite a benchmark')
    results=[];package_id=None
    for backend in backends:
        runtime=None;records=[];stop=Event();memory=[];cpu=[];timers=[]
        try:
            import psutil
            proc=psutil.Process();cpu_before=proc.cpu_times();wall=time.monotonic()
            def monitor():
                while not stop.wait(.05):
                    try:
                        processes=[proc]+proc.children(recursive=True)
                        memory.append(sum(p.memory_info().rss for p in processes if p.is_running()))
                    except (psutil.Error,OSError):pass
            thread=Thread(target=monitor,daemon=True);thread.start()
            start=time.monotonic();runtime=Runtime(directory,backend,fixture=fixture,mono=mono);cold=time.monotonic()-start
            package_id=runtime.package['id']
            for tier in tiers:
                for index,case in enumerate(cases):
                    view=case['view'];seat=view['seat'];save=new_save(runtime.package,f'benchmark-{index}',str(case.get('seed',index)),
                        [dict(seat=seat,archetype=runtime.package['roster']['archetypes'][index%16]['id'],tier=tier)])
                    if case.get('interrupt'):
                        timer=Timer(case.get('interrupt_after',.1),lambda:runtime.interrupt(cancel=case['interrupt']=='cancel'))
                        timer.start();timers.append(timer)
                    request=dict(protocol='personality-game-rpc/1.0.0',request_id=f'{tier}-{index}',view=view,choices=case['choices'],save=save,
                                 insolvencies=case.get('insolvencies',[0]*4))
                    start=time.monotonic();reply=runtime.plan(request);elapsed=time.monotonic()-start
                    records.append(dict(case=index,tier=tier,era=view['context']['era'],seed=case.get('seed',index),
                        status=reply['status'],interrupt=case.get('interrupt'),paths=reply['paths'],metrics=reply['metrics'],seconds=elapsed,
                        available_kinds=sorted({c['kind'] for c in case['choices']})))
                    for timer in timers:timer.cancel();timer.join()
                    timers.clear()
            runtime.close();inference=runtime.service.report();runtime=None
            c=proc.cpu_times();cpu_seconds=c.user+c.system-cpu_before.user-cpu_before.system
            elapsed=time.monotonic()-wall
            results.append(dict(backend=backend,available=True,records=records,cold_seconds=cold,
                wall_seconds=elapsed,process_cpu_percent=100*cpu_seconds/max(elapsed,1e-6),
                # Native child CPU totals aren't reliable after exit; don't invent them.
                cpu_scope='Python parent only; native simulator CPU excluded',
                peak_tree_rss=max(memory,default=None),inference=inference,
                completed_simulations=sum(r['metrics']['completed'] for r in records),
                qualified_samples=sum(r['metrics']['qualified_samples'] for r in records),
                device_utilization=None,device_memory=None))
        except Exception as error:
            results.append(dict(backend=backend,available=False,error=type(error).__name__+': '+str(error),records=records))
        finally:
            for timer in timers:timer.cancel();timer.join()
            if runtime is not None:runtime.close()
            stop.set()
            if 'thread' in locals():thread.join()
    report=seal(dict(format=BENCHMARK,package_id=package_id,fixture=bool(fixture),unity_running=unity_running,
        cases_hash=digest(cases),environment=dict(system=platform.system(),machine=platform.machine(),processor=processor_name()),
        profiles=list(tiers),results=results,qualified=False,
        note='Fixture/direct execution timings are not difficulty evidence. Device utilization unavailable unless separately instrumented.'))
    write(out,report);return report


def recommendation(report):
    """Choose only among equal-case, full-profile measurements with valid turns."""
    candidates=[]
    for result in report['results']:
        rows=[r for r in result.get('records',[]) if not r.get('interrupt')]
        if not result['available'] or not rows or any(r['status']!='ok' for r in rows):continue
        total=sum(r['seconds'] for r in rows);completed=sum(r['metrics']['completed'] for r in rows)
        candidates.append(dict(backend=result['backend'],completed_per_second=completed/max(total,1e-9),
                               median_seconds=statistics.median(r['seconds'] for r in rows)))
    if not candidates:return dict(backend=None,status='No valid measured backend')
    best=max(candidates,key=lambda x:(x['completed_per_second'],-x['median_seconds']))
    return dict(**best,status='Fixture timing only' if report['fixture'] else 'Timing candidate; verify release coverage and playing strength',comparisons=candidates)
